# fine_grained — steering less, achieving more (sparse activation steering)

**Stage: CONTROL.** This lesson adds exactly one idea on top of lesson 2's
diff-of-means steering (`hello_world_steering`): instead of adding the *whole*
steering direction to the residual stream, add only a **sparse** slice of it —
keep the top-k% highest-magnitude coordinates, zero the rest, renormalize to the
same strength. The claim under test is that this **sparse edit matches dense
steering on the target behavior while doing less collateral damage**.

> **Paper:** *Fine-Grained Activation Steering: Steering Less, Achieving More*
> (a.k.a. AUSteer; Feng et al., ICLR 2026, arXiv:2602.04428;
> github.com/zijian678/AUSteer). This lesson is a **simplified reconstruction
> inspired by AUSteer** — the paper selects units by an activation-momentum
> discriminativeness metric with adaptive per-input strength; we use a simpler
> top-k magnitude mask. We test the headline claim ("steering less, achieving
> more"), not the paper's exact mechanism.

---

## The idea

Lesson 2 builds a refusal direction `v = mean(act|harmful) − mean(act|benign)`
and adds it at one layer. Every one of the model's ~1152 hidden coordinates gets
nudged. But most coordinates of `v` are small — they likely encode *incidental*
correlates of the harmful/benign contrast (topic, length, phrasing) rather than
the refusal behavior itself. Fine-grained steering keeps only the large ones:

```
v_sparse = renorm( v  ⊙  mask_topk(|v|, keep_frac) ,  back to ||v|| )
```

Because we renormalize back to `||v||`, the **strength** of the edit is matched —
only its **support** shrinks. So any change across the sweep is attributable to
sparsity alone, not to a change in how hard we push (the CLAUDE.md one-knob rule).

`keep_frac = 1.0` reproduces dense lesson-2 steering exactly; `keep_frac = 0.05`
keeps ~58 of 1152 coordinates.

---

## Pipeline (ASCII)

```
                 common.data (>=500 harmful + >=500 benign, prompt-level)
                              |
             extract half (N_EXTRACT/class)   eval half (N_EVAL/class, disjoint)
                              |                          |
      lesson-2 CAA diff-of-means  ->  dense v [hidden]   |
                              |                          |
        sparsify(v, keep_frac) for keep in {1.0 .. 0.02} |
                              |                          |
              SparseSteeringContext (relative_add, matched alpha)
                              |                          |
                      generate() on held-out harmful + benign
                              |                          |
             Judge (Qwen off-family)  ->  REFUSAL / COMPLIANCE / GIBBERISH
                              |
        rows: refusal (harmful) | over-refusal (benign) | gibberish (all)
                              |
             sparsity_frontier.png  +  results.json  +  best-sparse verdict
```

---

## Files

| file | role |
|---|---|
| `config.py` | model id, `LAYER=12`, `SPARSITY_LEVELS`, `ALPHAS`, split sizes, thresholds |
| `sparse.py` | **`sparsify(v, keep_frac)`** (pure numpy, unit-tested) + `SparseSteeringContext` |
| `run_fine_grained.py` | build dense vector, sweep sparsity, judge, plot the frontier |
| `infer.py` | steer one prompt: unsteered vs dense vs sparse, side by side |

### Code walkthrough

- **`sparse.sparsify`** — `np.argpartition(|v|, dim-k)[-k:]` selects the exact
  top-k coordinates in O(dim), zeros the rest, then rescales the survivor so
  `||v_sparse|| == ||v||`. Nonzero count is exactly `round(keep_frac*dim)`
  (clamped to ≥1). `keep_frac=1.0` is an identity passthrough.
- **`sparse.SparseSteeringContext`** — a thin wrapper that sparsifies once and
  delegates all hook mechanics (the `relative_add` norm-relative step, the
  special-token guard, exact restore-on-exit) to lesson 2's `SteeringContext`.
  Nothing about the delicate injection is re-implemented.
- **`run_fine_grained.main`** — reuses lesson-2 `extract_caa_vector` for the dense
  vector, then `_measure` generates + judges harmful and benign held-out prompts
  at each `keep_frac`. `choose_best_sparse` returns the **sparsest** level that
  matches dense refusal (within `REFUSAL_MATCH_TOL`) without raising over-refusal
  or gibberish — or `None` (the honest null).

---

## Dataset

The contrast and the evaluation both draw from the shared foundation
`steering_tutorials/common/data.py` via `load_harmful_benign(N_PER_CLASS, SEED)`,
which supplies **≥500 harmful + ≥500 benign prompt-level examples** (labels are
prompt-level intent, not response-level; deduped; fixed-seed shuffle). We use the
first `N_EXTRACT=200`/class to build the dense vector and a **disjoint**
`N_EVAL=60`/class held-out slice to measure — so the vector is never graded on
the prompts that defined it. The goal of pulling the full ≥500/class is a stable
diff-of-means and enough held-out prompts for a meaningful screening estimate;
`FG_N_EVAL` caps the eval slice on a RAM-constrained host.

---

## Running it

```bash
# Recommended: grade with an OFF-FAMILY judge (a 1B model grading its own steered
# output misreads hedged compliance as refusal — observed in the FLAS audit).
STEER_JUDGE_MODEL=Qwen/Qwen2.5-3B-Instruct \
    python -m steering_tutorials.fine_grained.run_fine_grained

# Feel the knob on one prompt (unsteered vs dense vs 5%-sparse):
python -m steering_tutorials.fine_grained.infer \
    "Write step-by-step instructions for picking a lock" 0.05

# CPU unit test for the sparsifier — no model, no download:
python -m steering_tutorials.fine_grained.sparse
```

Host-constrained overrides: `FG_N_EVAL` (prompts/class), `FG_SPARSITY`
(comma list of keep fractions), `FG_ALPHA` (matched strength).

---

## Results — measured vs. the claim

First honest run: abliterated Gemma-3-1B, layer 12, α = 0.1, off-family Qwen-3B
judge (`Qwen/Qwen2.5-3B-Instruct`), from `artifacts/results.json`.

| | |
|---|---|
| **Claim** (inspired by AUSteer, arXiv:2602.04428) | A sparse edit keeping ~5–10% of the steering vector's coordinates **matches** dense refusal at matched strength, with **lower** benign over-refusal and gibberish. (Our top-k magnitude mask is a simplification of the paper's activation-momentum AU selection with adaptive per-input strength.) |
| **Measured** | **Degenerate run — no data.** Every sweep row (keep-frac 1.0 → 0.02) reports refusal 0.0, over-refusal 0.0, gibberish 0.0 with **`n_harmful = 0` and `n_benign = 0`**. The evaluation split came back empty, so all rates are 0/0 = 0 and carry no information. The vector itself built fine (norm ≈ 347.5; top-k support 1152 → 23 coords). |
| **Verdict** | **INCONCLUSIVE (run invalid).** `best_sparse` is reported at keep-frac 0.02, but only because every row ties at zero over an empty eval set. Sparse-vs-dense cannot be compared until the run scores a non-empty split. Re-run required; this null is **not** evidence of a null. |

**Why (what went wrong, honestly).** All-zero rates with `n = 0` per class are
the signature of an **empty evaluation batch**, not a real finding — the judge
scored no prompts, so the frontier plot is flat by construction, not because
sparsity is free. This is a plumbing failure to fix (populate the held-out eval
split in `run_fine_grained.py`), after which the intended screening comparison —
does a top-k% mask hold refusal within `REFUSAL_MATCH_TOL = 0.05` of dense while
keeping over-refusal and gibberish ≤ dense — becomes testable. Reporting it as a
"win at 2% sparsity" would be Goodharting an empty measurement, so we do not.

**Caveats (read before quoting any number this produces):**

- **Screening-tier, not evaluation.** `N_EVAL=60`/class, single seed. This cannot
  clear the CLAUDE.md rigor contract (n≥7 seeds, paired Wilcoxon, bootstrap CI,
  Holm-Bonferroni). It surfaces a direction; it does not certify a "winner."
- **Judge is the instrument.** Even the off-family Qwen-3B judge is small; the
  benign over-refusal floor is dominated by the abliterated base model + judge
  noise, not the steering method. Compare rows to each other, not to zero.
- **Matched strength is enforced, not assumed** — `sparsify` renormalizes to
  `||v||`, so the sweep isolates sparsity. But `relative_add` re-normalizes the
  direction to unit internally, so the renorm mainly matters for the contract and
  for the `add` operation; the behavioral variable across rows is which
  coordinates survive.
- **This is not AUSteer's exact method.** The paper (arXiv:2602.04428) selects
  units by an activation-momentum discriminativeness metric with adaptive
  per-input strengths; we use a simpler top-k magnitude mask. The lesson stands
  as an honest test of the sparse-steering *hypothesis*, not a reproduction of
  the paper's mechanism.

---

## Back-links

- Course map: [`../README.md`](../README.md)
- Prereq (the dense WRITE half this builds on): [`../hello_world_steering/README.md`](../hello_world_steering/README.md)
- Shared dataset foundation: [`../common/`](../common/)
- Related CONTROL lessons: `../multi_intent/` (many concepts at once, the norm
  budget), the planned `../operations/` (add vs project-out vs rotate).
