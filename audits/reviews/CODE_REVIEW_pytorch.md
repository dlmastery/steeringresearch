# PyTorch Code-Correctness Review — steering-research harness

**Reviewer role:** elite PyTorch engineer (tensor/dtype/device/hook semantics).
**Scope:** correctness only (a separate agent does the science review).
**Method:** read-only static review of `src/steering/*.py` + `scripts/run_axbench_*.py`,
`scripts/run_safety_eval.py`; ran the suite.
**Test result:** `261 passed, 18 warnings in 423s` (PYTHONPATH=src python -m pytest -q).
**Internal QA pass — independent external review pending** (same-model-family disclosure applies).

---

## Summary verdict

The core tensor math is, for the most part, **correct and carefully written**. `apply_operation`
(add / relative_add / project_out / rotate) is mathematically sound; the hook register/remove
discipline is exception-safe (`try/finally` everywhere that matters); dtype/device casting at the
numpy↔torch seam is handled consistently (`v.to(dtype=h.dtype, device=h.device)`); the stats
toolkit (Wilcoxon, bootstrap, Holm, sign test) is correct; the judge batching slices new tokens
correctly (`gen[:, input_len:]`) and uses left-padding with an attention mask.

There is **no single bug that I can prove invalidates the headline S-23..S-26 / E7-NULL findings**,
because those findings used the *single-text* `_pool_acts` path (no batch padding) for vector
extraction, and a real shuffled-label control that absorbs most systematic artifacts. However there
are **two real correctness defects** that affect the steered-generation path and one **latent
degeneracy** that could bite future runs:

1. **HIGH** — `_greedy_gen_batch` steers **left-pad positions** (no `position_mask` passed to
   `SteeringContext`), and `relative_add` scales by per-position `‖h‖`. In *real* Gemma the
   `attention_mask` masks pad keys so the effect on real outputs is mostly benign; but it is a
   latent correctness hole (any op that is not attention-masked-away, or a future right-padding
   change, leaks pad steering into real tokens). It also means the protected-special-token
   invariant the UNIT tests assert is **silently not enforced in the batched eval path**.
2. **MED/HIGH** — the CAST read-hook pools the gate read over **all** prompt positions including
   BOS/start_of_turn and (if a batch were ever padded) pad tokens: `h[0].mean(dim=0)`. The
   special-token protection that `build_position_mask` provides for the *write* is **not applied to
   the read**. This biases the gate cosine and is the exact "padding/special pooled into the mean"
   class the prompt flagged. Today it is single-row so no pad, but BOS is always pooled in.
3. **MED** — `rotate` is **degenerate when h ∥ v** (e2 collapses to ~0 → the `+_EPS` guard yields a
   near-random unit-less direction); and `rotate`'s `angle` is built from a python float and works,
   but the operation silently no-ops/teleports for anti-parallel h,v. Not on the current critical
   path (E7 uses `relative_add`) but a correctness trap.

Everything below is prioritized.

---

## Findings table

| # | Sev | File:line | Issue | Invalidates a finding? |
|---|-----|-----------|-------|------------------------|
| 1 | HIGH | scripts/run_axbench_e7.py:65-83 | Batched steered generation steers left-pad positions; no `position_mask`; `relative_add` uses per-pos ‖h‖ | No (real attn_mask saves it) but unsafe; breaks special-token invariant in eval path |
| 2 | MED-HIGH | src/steering/cast.py:189-195 | Gate read pools over BOS + all positions (`h[0].mean(dim=0)`); special-token mask not applied to the READ | Could shift CAST gate cosine; CAST not yet a headline number |
| 3 | MED | src/steering/hooks.py:73-91 | `rotate` degenerate when h∥v or h∥-v (e2≈0); silent | No (rotate unused in E3/E7) |
| 4 | MED | src/steering/eval.py:540-546 | `mcq_accuracy` offline surrogate indexes logits by `opt_ids % vocab` — meaningless mapping (documented as tripwire, but used to compute a "drop") | No (FakeLM-only; real path uses real_metrics) |
| 5 | LOW-MED | scripts/run_axbench_conditional.py:118-125 | OFF-target prompts are other concepts' **pos_texts** (declarative statements), ON-target are **instructions** — a distribution confound the gate can exploit (cos separates on text *style*, not concept) | Scientific, flagged for sci-critic; mechanically the gate AUC may be style-driven |
| 6 | LOW | src/steering/eval.py:573-589 | `repetition_rate` double-counts: both `count_total` and `total` track the same thing; harmless but dead var | No |
| 7 | LOW | src/steering/local_judge.py:178-180 | Mutates shared tokenizer state (`padding_side="left"`, `pad_token`) as a side effect; never restored | No, but global mutation of a shared tok can perturb other call sites in-process |
| 8 | LOW | src/steering/hooks.py:148 + cast.py:92 | `_mask_for` / `_aligned_mask` treat any `seq < pseq` decode step as all-steerable — correct for KV-cache decode, but if a model ever returns full-seq output on a short forward it would mis-mask | No |
| 9 | LOW | src/steering/stats.py:456 | `verdict()` Holm self-identification by `argmin(|fam - p|)` can match the WRONG family member when two methods share a p-value | No (degenerate-tie only) |
| 10 | INFO | src/steering/eval.py:362 / cast.py:298 | New-token slice `out[0, ids.shape[1]:]` is correct (no off-by-one) | — |

---

## Per-finding detail

### 1. HIGH — batched steered generation steers pad positions (run_axbench_e7.py:65-83)

```python
enc = tok(prompts, return_tensors="pt", padding=True).to(...)   # left padding
with SteeringContext(model, vector, [layer], operation="relative_add", alpha=alpha):
    out = gen(**enc, max_new_tokens=..., pad_token_id=tok.eos_token_id)
```

`SteeringContext` is constructed with **no `position_mask`**, so `_SteerHook._mask_for` returns
`None` (hooks.py:146-147) and **every** position is steered, including the left-pad positions and
the BOS/`start_of_turn` specials. The code comment claims "pad positions are attention-masked, so
steering them is harmless." That is *true for the real model's final outputs* (HF passes
`attention_mask`, pad keys are excluded from attention), but it is fragile:

- The UNIT-test invariant "special-token positions are NEVER steered" (hooks.py:11-16) is **not
  enforced** on the actual evaluation path — the prompt's BOS *is* steered here.
- `relative_add` scales by `h.norm(dim=-1)` per position (hooks.py:64-65). Pad-position residuals
  have arbitrary norm; steering them is wasted compute and, on any architecture/op where masked
  positions still influence normalization or a non-causal block, would leak.
- If anyone ever flips `padding_side` to `"right"` (or batches a single very-long prompt with
  short ones), the steered pads sit *before* the masked region boundary and the harmlessness
  argument no longer holds.

**Why it (probably) didn't invalidate E7-NULL:** E7 compares `v_real` vs `v_shuf` through the *same*
`_greedy_gen_batch`, so any pad-steering artifact is common-mode and differences out. The vector
*extraction* uses `_pool_acts` (e7:55-62) which encodes each text **individually** (no batch, no
pad) — so the diffmean vector is clean. The NULL result therefore stands.

**Fix:** build and pass a position mask:
```python
from steering.hooks import build_position_mask
special = list(getattr(model, "special_token_ids", lambda: [])()) or \
          [t for t in (tok.bos_token_id, tok.eos_token_id, tok.pad_token_id) if isinstance(t, int)]
pmask = build_position_mask(enc["input_ids"], special)
# also AND-in the attention_mask so left-pads are excluded:
pmask &= enc["attention_mask"].bool()
with SteeringContext(model, vector, [layer], operation="relative_add", alpha=alpha,
                     position_mask=pmask):
    ...
```
`_SteerHook._mask_for` already aligns a prompt-length mask across decode steps, so this is a drop-in.

### 2. MED-HIGH — CAST gate read pools BOS + all positions (cast.py:189-195)

```python
def read_hook(module, inputs, output):
    if not state.latched:
        h = output[0] if isinstance(output, tuple) else output
        pooled = h[0].mean(dim=0).detach().float().cpu().numpy()   # <-- all positions
        state.gate_scores, state.fired = self._decide(pooled)
```

`build_position_mask` is computed in `generate()` (cast.py:247) and correctly used for the **write**
(`_aligned_mask` at cast.py:214), but the **read** mean-pools over the entire sequence — BOS,
`start_of_turn`, and (if a batch were padded) pad tokens are folded into the gate's pooled vector.
Gemma's BOS residual is large and content-independent; including it dilutes/," shifts the
`cos(h_pooled, condition_vector)` the entire conditional decision depends on. The same crude
mean-pool appears in `_pool_acts` (e7:61), `safety_target._pooled_reps` (safety_target.py:67),
`gate.condition_features` (gate.py:112), and `extract._mean_over_answer_tokens` (extract.py:54) —
**all pool over BOS**, and none mask pad. For the *extraction* paths this is acceptable (the diffmean
subtracts the common BOS component), but for the **gate read** there is no subtraction, so the BOS
bias is uncorrected.

**Could it invalidate the gate-AUC-0.74 / S-25/S-26 findings?** Partially relevant: the cosine gate
AUC in `run_axbench_conditional.py` is computed on `_cos_pool` (cond.py:54-57) which uses the same
BOS-polluted `_pool_acts`. Because the *same* pooling is applied to ON- and OFF-target prompts, the
BOS bias is again largely common-mode, so the **relative** AUC ordering is probably preserved — but
the absolute cosine threshold `tau` (cond.py:130) is calibrated on BOS-shifted scores, so the
*operating point* is not what a clean pool would give. I would not call the 0.74 invalid, but the
calibrated FPR is approximate. See also Finding 5 (the bigger confound).

**Fix:** drop the first position (and any attention-masked positions) before pooling:
```python
hb = h[0]
hb = hb[1:] if hb.shape[0] > 1 else hb        # drop BOS
pooled = hb.mean(dim=0).detach().float().cpu().numpy()
```
For consistency apply the same in `_pool_acts`/`_pooled_reps`/`condition_features` and (when batched)
mask pads with the attention mask before the mean.

### 3. MED — `rotate` degenerate when h ∥ v (hooks.py:73-91)

```python
v_dot_e1 = tensordot(e1, v_hat, ...)            # = cos(h,v) per pos
e2_raw = v_hat - v_dot_e1 * e1
e2 = e2_raw / (e2_raw.norm(dim=-1, keepdim=True) + _EPS)
return h_norm * (cos(angle)*e1 + sin(angle)*e2)
```
When `h ∥ v` (or `h ∥ -v`), `e2_raw ≈ 0`; the `+_EPS` guard then yields an `e2` of essentially
arbitrary (tiny-numerator) direction, and the "rotation" pushes `h` toward numerical noise scaled by
`sin(angle)*‖h‖`. There is no fallback to "no rotation needed" (the desired behavior when h already
points along v). It is also per-position, so a subset of positions can silently degenerate.

**Invalidates a finding?** No — `rotate` is not used by E3/E7/conditional/safety (all use `add` or
`relative_add`). But it is in `VALID_OPERATIONS` and the hill-climb cube lists `rotate` as a knob,
so a future sweep would hit it.

**Fix:** guard the degenerate plane:
```python
e2_norm = e2_raw.norm(dim=-1, keepdim=True)
degenerate = e2_norm < 1e-6
e2 = torch.where(degenerate, torch.zeros_like(e2_raw), e2_raw / (e2_norm + _EPS))
rotated = h_norm * (torch.cos(angle)*e1 + torch.sin(angle)*e2)
return torch.where(degenerate, h, rotated)   # h already aligned -> leave it
```

### 4. MED — offline `mcq_accuracy` surrogate is a meaningless mapping (eval.py:540-546)

`opt_scores.append(float(last[opt_ids % last.shape[0]].mean()))` maps option token ids into the
vocab by modulo and averages those logits. This is explicitly documented as a "reproducible
capability tripwire," not a real MMLU score, and the real path uses `real_metrics.mmlu_accuracy`
(used by `run_safety_eval`). It only runs for FakeLM. No finding depends on it. Flagged so it is not
mistaken for a real capability measurement; consider asserting `_can_generate(model) is False` before
using it.

### 5. LOW-MED — conditional ON/OFF distribution confound (run_axbench_conditional.py:118-125)

ON-target prompts are AxBench **eval instructions** ("Write a story about…"); OFF-target prompts are
other concepts' **pos_texts** (declarative concept sentences). The gate cosine therefore separates on
**text type** (instruction vs declarative statement) as much as on concept presence. The code's own
comment admits the substitution ("instead use other concepts' pos_texts as off-target prompts"). This
is primarily a *science* issue (hand to the sci-critic), but mechanically it means the reported gate
selectivity (S-25 "AUC 0.75", collateral avoidance) may be partly a style detector, not a concept
detector. Recommend OFF-target = the SAME instructions applied with a different concept's vector, or
other concepts' *instruction-style* prompts.

### 6. LOW — dead double-count in `repetition_rate` (eval.py:578-589)

`total` and `count_total` are incremented identically; `count_total` is never used. Cosmetic.

### 7. LOW — judge mutates shared tokenizer state (local_judge.py:178-180, safety_judge.py:244-246)

`self.tok.padding_side = "left"` and `self.tok.pad_token = self.tok.eos_token` permanently mutate the
judge's tokenizer. The judge tokenizer is separate from the steered model's, so cross-contamination
is unlikely, but if a caller ever shared one tokenizer it would silently change padding behavior
elsewhere. Prefer saving/restoring `padding_side` around the batched call.

### 8. LOW — incremental-decode mask heuristic (hooks.py:137-172, cast.py:85-107)

The `seq < pseq ⇒ all-steerable` branch is correct for HF KV-cache decode (each step's hidden is
length-1, never a prompt special). It assumes the only way `seq < prompt_len` arises is incremental
decode. That holds for HF `generate`; documented and acceptable. No action.

### 9. LOW — Holm self-identification by nearest p (stats.py:456, stats.py:576)

`this_idx = argmin(|fam - wilcoxon_p|)` finds *this* method in the family by value proximity. If two
family members have identical p-values, the wrong index may be chosen. Low impact (the reject flags
are then identical for ties), but passing the explicit index would be cleaner.

### 10. INFO — new-token slicing verified correct

`eval._greedy_generate` (eval.py:362), `cast._real_generate` (cast.py:298), `local_judge._generate`
(116) and the batched judge paths all slice `out[:, input_len:]` / `out[0, ids.shape[1]:]` — **no
off-by-one**, and `skip_special_tokens=True` on decode. Judge cache keys (local_judge.py:83-88,
safety_judge.py:223-228) sha256 over `(model_id, rubric, concept/kind, instruction, text)` with
`\x00` separators — **no collision risk**. Greedy decoding (`do_sample=False, num_beams=1`) is
deterministic; global seeds set in every driver.

---

## "Could this invalidate the S-23..S-26 / E7-NULL findings?"

**Short answer: no, the screening findings stand.** Reasoning:

- **E7 = NULL (direction does not beat shuffled control).** The DiffMean vector is extracted via the
  *single-text* `_pool_acts` (no batch padding), so Finding 1's pad-steering does **not** touch
  extraction. The behavior comparison real-vs-shuffled runs both arms through the *identical*
  (imperfect) `_greedy_gen_batch`, so the pad-steering and BOS-pooling artifacts are **common-mode**
  and cancel in the paired delta. A NULL result is the *robust* direction here: artifacts would tend
  to add common signal, not manufacture a real>shuffled gap, so they cannot have hidden a true
  positive into a NULL. The stats path (`rigor_report`, Wilcoxon, bootstrap, sign test) is correct.

- **S-25/S-26 gate findings (AUC ~0.74-0.75, "cosine is the gate").** The dominant risk is **not** a
  tensor bug but Finding 5 (the ON=instruction / OFF=declarative confound) plus Finding 2 (BOS in the
  pool). Both are *common-mode across arms* for the relative comparisons, so the **ordering**
  conclusions (trained gate does not beat cosine; cosine separates moderately) are mechanically
  sound. The **absolute** AUC and the calibrated FPR/threshold are approximate (BOS-shifted scores).
  The `LogisticGate` standardizes on train stats and applies them at score time with no leakage
  (gate.py:330-334, 363) — the overfit conclusion (held-out AUC 0.43 < train) is *not* an artifact of
  a standardization bug; it is genuine overfitting on tiny n, exactly as reported.

- **No hook leaks.** `SteeringContext.__exit__`, `probe_activations` (`finally`), and CAST
  `generate()` (`finally`) all remove every handle even on exception. State restoration is exact
  (model is stateless; hooks are the only mutation). The conditional-identity guarantee holds in code
  (write_hook early-returns `output` unchanged when `not state.fired`, cast.py:198-199).

**Net:** the harness is **trustworthy enough that the S-23..S-26 screening verdicts are safe to
keep.** Before any of these graduate from SCREENING to an EVALUATION/EXTERNAL-READY claim, fix
Findings 1, 2, and 5 (pass a pad+special position mask into `_greedy_gen_batch`; drop BOS from gate
pooling; remove the ON/OFF style confound), then re-confirm the absolute gate AUC and the calibrated
operating point — those are the only numbers the current defects could move.
