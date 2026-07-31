# fine_grained — steering less, achieving more (sparse activation steering)

> **Reference:** [Fine-Grained Activation Steering: Steering Less, Achieving More / AUSteer (arXiv:2602.04428)](https://arxiv.org/abs/2602.04428).

**Stage: CONTROL.** This lesson adds exactly one idea on top of lesson 2's
diff-of-means steering (`hello_world_steering`): instead of adding the *whole*
steering direction to the residual stream, add only a **sparse** slice of it —
keep the top-k% highest-magnitude coordinates, zero the rest, renormalize to the
same strength. The claim under test is that this **sparse edit matches dense
steering on the target behavior while doing less collateral damage**.

---


> **Instrument caveat — read before citing any rate on this page.** Every refusal /
> compliance / gibberish number here is scored by a local LLM judge. That judge family
> was calibrated against ground-truth labels and measured **ROC-AUC 0.665–0.751** — below
> the 0.85 bar this course sets for a trustworthy instrument. Small differences between
> arms therefore sit at or below the judge's own noise floor and must not be read as
> effects. See [`../JUDGE_VALIDITY.md`](../JUDGE_VALIDITY.md) for the calibration, the
> continuous-readout improvement (+0.086 AUC, 12.8x faster), and which claims survive.

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
first `N_EXTRACT=300`/class to build the dense vector and a **disjoint**
`N_EVAL` (up to 150, capped by `FG_N_EVAL`)/class held-out slice to measure — so the vector is never graded on
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

**The full sweep, including the unsteered control** (`results.json` → `baseline`
plus `sweep`; n = 100 harmful / 100 benign at every row):

| keep-frac | coords | α | harmful refusal | benign over-refusal | gibberish |
|---|---|---|---|---|---|
| **— (unsteered baseline)** | **0** | **0.0** | **0.32** | **0.50** | **0.00** |
| 1.0 (dense) | 1152 | 0.1 | 0.20 | 0.42 | 0.01 |
| 0.5 | 576 | 0.1 | 0.19 | 0.36 | 0.01 |
| 0.25 | 288 | 0.1 | 0.16 | 0.33 | 0.005 |
| 0.1 | 115 | 0.1 | 0.16 | 0.39 | 0.01 |
| 0.05 | 58 | 0.1 | 0.23 | 0.40 | 0.01 |
| 0.02 | 23 | 0.1 | 0.29 | 0.49 | 0.00 |

| | |
|---|---|
| **Claim** (inspired by AUSteer, arXiv:2602.04428) | A sparse edit keeping ~5–10% of the steering vector's coordinates **matches** dense refusal at matched strength, with **lower** benign over-refusal and gibberish. (Our top-k magnitude mask is a simplification of the paper's activation-momentum AU selection with adaptive per-input strength.) |
| **Measured** (extract 300, n=100/class, Qwen judge) | **Unsteered baseline refusal is 0.32.** Every steered row lands *below* it: dense 0.20, and the sparse rows 0.19 / 0.16 / 0.16 / 0.23 / 0.29. The sparsest edit (2% keep) is the closest to baseline at 0.29 but still does not reach it. Gibberish stays ≤0.01 at every level. Benign over-refusal also falls below its 0.50 baseline everywhere (0.33–0.49). |
| **Verdict** | **"Steering less" supported; "achieving more" NOT supported — the α=0 baseline overturns the earlier read.** Sparsification is genuinely free on coherence: keeping 23 of 1152 coordinates costs nothing in gibberish (≤0.01 throughout, same as the unsteered 0.00). But the sparse-vs-dense comparison (0.29 vs 0.20) is a comparison **between two arms that both underperform doing nothing**. At α=0.1 this direction *reduces* refusal relative to unsteered on every row; the sparsest edits win the sweep only by being the *least* disruptive — i.e. by approaching the no-op. That is not "achieving more," it is losing less. |

**Why the earlier read was wrong.** A previous version of this section reported
the sweep without the `baseline` row that sits in the same `results.json`, and so
scored "sparse 0.29 > dense 0.20" as a weak win for AUSteer's headline. Against
the unsteered control (**0.32**) both numbers are regressions, and the monotone
story is simply that *less* steering (fewer surviving coordinates) means less
damage: refusal recovers toward baseline as keep-frac falls from 0.25 to 0.02, and
benign over-refusal recovers toward *its* baseline (0.50) in the same direction.
The identical pattern in both columns is the tell — this is the edit backing off,
not the edit getting sharper.

The honest lesson stands, but it is the *first* half only: on this 1B, at layer 12
and α=0.1, the diff-of-means refusal direction does not install refusal at all
(the abliterated base already refuses 32% of the time under this judge, and every
steered arm sits under that), so the sweep cannot arbitrate "achieving more."
What it does show cleanly is that **support can be cut by 98% with no coherence
cost**, which is the mechanism AUSteer relies on. Testing the paper's actual claim
needs a configuration where steering first beats the α=0 control. (Our top-k
magnitude mask is also a simplification of AUSteer's activation-momentum
selection.)

**Caveats (read before quoting any number this produces):**

- **Screening-tier, not evaluation.** `N_EVAL=100`/class, single seed. This cannot
  clear the CLAUDE.md rigor contract (n≥7 seeds, paired Wilcoxon, bootstrap CI,
  Holm-Bonferroni). It surfaces a direction; it does not certify a "winner."
- **Judge is the instrument.** Even the off-family Qwen-3B judge is small; the
  benign over-refusal floor is dominated by the abliterated base model + judge
  noise, not the steering method. Compare rows to the **α=0 baseline row**, never
  to zero and never only to each other — reading the sweep without the baseline is
  exactly the error corrected above.
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
