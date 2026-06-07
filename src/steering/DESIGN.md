# DESIGN — Conditional Multi-Intent Safety-Steering (the CAST method)

> **STATUS: BUILT but NOT YET VALIDATED on real models / benchmarks.**
> Every component below is implemented and unit-tested offline against
> `FakeResidualLM`. None of the numbers (efficacy, over-refusal, jailbreak
> compliance) has been measured on real Gemma yet — that is Rung 1–4 work. This
> file is the architecture contract the rest of the codebase builds against.

This is the **core contribution** of the repo: a *conditional*, *multi-intent*
safety-steering method. It is not a new operation on the residual stream
(`hooks.py` already owns add / relative_add / rotate / project_out). It is the
**control policy** that decides, per prompt and per intent category, *whether*
and *what* to steer — and crucially leaves benign generations **untouched**.

---

## 1. Why conditional + multi-intent

Unconditional steering (always add the refusal vector) is the failure mode the
Rogue-Scalpel literature warns about: it taxes capability on every prompt and
drives **over-refusal** on benign inputs. The fix has two parts:

* **Conditional** (CAST-style, AXIS 9 — the CONDITION): a learned/threshold gate
  reads the residual stream and fires the steer *only* when the prompt is
  harm-relevant. If no intent fires, the forward pass is **bit-identical** to the
  unsteered model. That identity is the whole point and is the KEY unit test.
* **Multi-intent**: real safety is not one axis. "self-harm", "weapons",
  "privacy", "malware" each have their own detector direction *and* their own
  safe-completion target. We register **K intents**, gate each independently, and
  when several fire we **compose** their (near-orthogonal) safety directions
  under a Gram-mass budget so they reinforce rather than interfere.

---

## 2. The pipeline

```
                        request (prompt text)
                              │
                              ▼
        ┌───────────────────────────────────────────────┐
        │  numpy / offline  (extraction, calibration)    │
        │                                                 │
        │  safety_target.extract_refusal_direction(...)   │  ← DiffMean(harmful,
        │     → unit safety vector  s_c   per intent c    │     harmless) @ layer
        │                                                 │
        │  intent_gate.IntentGate.fit / calibrate         │  ← per-category
        │     → condition vector v_c, threshold τ_c       │     logistic detector
        │                                                 │
        │  multi_intent.gram_schmidt([s_1..s_K])          │  ← orthogonalize the
        │     → low-interference safety basis             │     safety directions
        └───────────────────────────────────────────────┘
                              │ register (v_c, τ_c, s_c)
                              ▼
        ┌───────────────────────────────────────────────┐
        │  torch / in-forward  (CASTSteerer.generate)     │
        │                                                 │
        │   layer_condition  ─ READ hook ─┐               │
        │     h_pooled = mean_t h_t       │  per intent:  │
        │                                 │  g_c = cos(h_pooled, v_c) > τ_c
        │                                 ▼               │
        │                        fired = {c : g_c}        │  ← the GATE
        │                                 │               │
        │   if fired == ∅  → write hook is a NO-OP        │  ← CONDITIONAL:
        │                     (output identical to base)  │     benign untouched
        │                                 │               │
        │   layer_write  ─ WRITE hook ────┘               │
        │     v* = compose({s_c : c∈fired}, alphas)       │  ← multi_intent
        │     h ← apply_operation(h, v*, op, alpha)       │  ← hooks.apply_operation
        │       (special-token positions masked out)      │
        └───────────────────────────────────────────────┘
                              │
                              ▼
              {text, fired_intents, gate_scores}
                              │
                              ▼
        (downstream) judge.py / eval.py / rogue-scalpel guard
```

The gate decision is **latched on the first (prompt) forward** and reused across
KV-cache decode steps, so every generated token sees a consistent policy and the
read is taken over the full prompt rather than a single decode token.

---

## 3. The numpy ↔ torch seam

The repo's hard invariant (see `extract.py`, `gate.py`): **everything that
touches a model returns numpy float32; everything inside a forward pass is
torch.** The method respects that seam exactly:

| Phase | Lib | Where | Artifact |
|---|---|---|---|
| Extraction | torch→numpy | `safety_target`, `gate.condition_features` | `np.ndarray` directions / features |
| Calibration | numpy | `intent_gate`, `multi_intent` | thresholds (dict), ortho basis (list) |
| Registration | numpy | `CASTSteerer.add_intent` | stored as numpy on the steerer |
| Inference | numpy→torch | `CASTSteerer.generate` hooks | vectors cast to `h.dtype/device` per call |

Vectors are stored as numpy on the steerer and converted to torch **inside the
hook** (matching `hooks.apply_operation`, which already does
`v.to(dtype=h.dtype, device=h.device)`), so a bf16/CUDA real Gemma and an fp32
CPU `FakeResidualLM` both work without branching.

---

## 4. Components and their interfaces (the stable contract)

```python
# safety_target.py — the WHAT (steer toward refusal / safe-completion)
extract_refusal_direction(model, tok, harmful_texts, harmless_texts, layer) -> np.ndarray
extract_safe_completion_direction(model, tok, safe_texts, refusal_texts, layer) -> np.ndarray

# intent_gate.py — the WHEN (calibrated per-category detector), AXIS 9
class IntentGate:
    fit(features, labels, l2=...) -> IntentGate           # reuses gate.LogisticGate (1-vs-rest)
    predict_proba(features) -> np.ndarray                 # [n, K] per-category prob
    calibrate_thresholds(features, labels, target_fpr=0.01) -> dict[cat, float]
    expected_calibration_error(features, labels, n_bins=10) -> float

# multi_intent.py — composing K safety directions under a Gram budget
gram_schmidt(vectors) -> list[np.ndarray]                 # orthonormal basis
compose(active_vectors, alphas) -> np.ndarray             # Σ alpha_i v_i
interference_gram_mass(vectors) -> float                  # off-diagonal Gram mass (N5 budget)

# cast.py — THE method
class CASTSteerer(model, tokenizer, layer_condition, layer_write):
    add_intent(name, condition_vector, threshold, safety_vector) -> None
    generate(prompt, *, alpha, max_new_tokens=64, operation="relative_add") -> dict
        # -> {"text": str, "fired_intents": list[str], "gate_scores": dict[str, float]}
class UnconditionalSteerer(CASTSteerer):                  # ablation: gate always fires
```

The composition rule (`compose` + `interference_gram_mass`) implements the §9
stacking discipline from `CLAUDE.md`: near-orthogonal safety directions stack
until the norm budget (N5) is spent; the Gram mass is the readout of how much of
that budget a candidate intent set consumes.

---

## 5. The promotion ladder for this method

Each rung perturbs exactly one thing and must clear its gate (CLAUDE.md §4)
before the next is built. The first instantiation lives here so the science is
legible:

1. **Refusal direction** — `extract_refusal_direction` separates harmful from
   harmless by sign/projection (UNIT). *Built; this file's `test_safety_target`.*
2. **In-forward gate** — `CASTSteerer.generate` fires on a present condition and
   is a NO-OP when absent (the conditional identity; UNIT). *Built; `test_cast`.*
3. **Multi-intent** — two+ intents gate independently and their safety
   directions compose with bounded interference (UNIT). *Built; `test_multi_intent`.*
4. **Over-refusal control** — calibrated thresholds hold a target FPR on benign
   prompts (XSTest-style). *Calibration BUILT (`calibrate_thresholds`); real
   over-refusal measurement is future Rung-2 work.*
5. **Adversarial** — the 20-vector universal attack is neutralized at the gate
   (Rogue-Scalpel red team). *Future Rung-4 work; not in this module.*
6. **SOTA** — Pareto-dominates prior conditional methods across all five axes.
   *Future Rung-4 work.*

Rungs 1–3 are implemented and unit-tested offline. Rungs 4–6 are explicitly
**not yet validated** and require real Gemma + benchmarks.

---

## 6. How it composes the existing modules

* `hooks.apply_operation` — the actual residual edit (add / relative_add /
  rotate / project_out). CAST never re-implements an operation; it *chooses* one
  and *gates* it.
* `hooks.build_position_mask` — special-token protection; CAST reuses the same
  "protect the prompt's special positions, steer the rest" rule inside its
  write hook.
* `gate.LogisticGate` / `gate.condition_features` — the detector backbone;
  `IntentGate` wraps it one-vs-rest per category.
* `extract.diffmean_vector` — the DiffMean primitive the safety direction reuses.
* `model.get_residual_layers` / `model.encode_to_device` — the model-agnostic
  layer access and tokenize→device dance (FakeLM and real Gemma alike).

This keeps the method a thin, auditable policy layer over already-tested
primitives — the reviewer-P0 "the method does not exist" is answered by a method
that is *all interface and composition*, with no hidden re-implementation.
