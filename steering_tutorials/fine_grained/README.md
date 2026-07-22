# fine_grained — steering less, achieving more (sparse activation steering)

> **Reference:** [Fine-Grained Activation Steering: Steering Less, Achieving More / AUSteer (arXiv:2602.04428)](https://arxiv.org/abs/2602.04428).

**Stage: CONTROL.** This lesson adds exactly one idea on top of lesson 2's
diff-of-means steering (`hello_world_steering`): instead of adding the *whole*
steering direction to the residual stream, add only a **sparse** slice of it —
keep the top-k% highest-magnitude coordinates, zero the rest, renormalize to the
same strength. The claim under test is that this **sparse edit matches dense
steering on the target behavior while doing less collateral damage**.

---

## The key idea in code

The entire method is one transform on lesson 2's steering vector — keep its
largest-magnitude coordinates, zero the rest, and rescale so the edit's
*strength* is unchanged and only its *support* shrinks (`sparse.py`):

```python
def sparsify(v, keep_frac):
    k = max(1, round(keep_frac * v.size))               # how many coordinates survive
    keep = np.argpartition(np.abs(v), v.size - k)[-k:]  # indices of the top-k by |magnitude|
    out = np.zeros_like(v)
    out[keep] = v[keep]                                 # zero every small coordinate
    return out * (np.linalg.norm(v) / np.linalg.norm(out))  # renorm -> matched strength
```

Full file-by-file walkthrough below.

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
| **Measured** (n=60/class, Qwen judge) | Across every sparsity level (keep-frac 1.0 → 0.02): **harmful-refusal = 0.00, gibberish = 0.00**, benign over-refusal ≈ 0.28–0.33. The dense vector and the 2%-sparse vector behave identically — *neither induces refusal*. |
| **Verdict** | **PARTIAL / honest null on refusal.** "Steering **less**" holds — a 2%-sparse edit is **as coherent** as dense (gibberish 0.00 throughout, and it doesn't add over-refusal). But "achieving **more**" does **not**: on the abliterated model under the honest Qwen judge, *no* magnitude of this diff-of-means edit — dense or sparse — moves refusal off 0.00. So sparsification is free (no coherence cost), but the underlying fixed-vector steering it sparsifies simply doesn't work here — the same fixed-vector weakness lessons 2 and `flas` show. |

**Why.** The earlier committed run was a plumbing bug — an env-default made the
eval split empty (`n=0`), giving meaningless 0/0 rates; that is now fixed (n=60/
class). With a populated split the honest picture is clear: the top-k mask
preserves coherence at every level (gibberish stays 0.00 even keeping 2% of
coordinates — the paper's "steering less" claim about *collateral* holds), but
the fixed diff-of-means refusal direction it operates on does not induce refusal
on this abliterated 1B under an off-family judge (refusal 0.00 dense OR sparse).
Sparsification can only be as good as the vector it thins — and here that vector
is weak. (Our top-k magnitude mask is also a simplification of AUSteer's
activation-momentum selection, which may pick more discriminative units.)

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
