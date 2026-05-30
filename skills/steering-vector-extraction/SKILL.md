---
name: steering-vector-extraction
description: >
  Use when extracting steering vectors from contrast pairs on Gemma-3-1B-it
  or Gemma-2-2B-it. Covers DiffMean vs PCA-top1, the pair-count knee
  experiment (E1), optimal layer selection by Fisher ratio (E2), the
  cosine-agreement diagnostic (E4), activation caching, and which contrast
  datasets to use per behavior type. Produces a cached, layer-indexed vector
  bank that feeds all downstream experiments on the ladder.
---

# Skill — steering-vector-extraction

## When to use

Use this skill whenever you are:

- Running Experiment E1 (pair-count knee: how many contrast pairs are enough?).
- Running Experiment E2 (layer selection: which layer has max Fisher ratio?).
- Running Experiment E4 (cosine agreement between DiffMean and PCA-top1).
- Adding a new behavior to the vector bank (new dataset, new axis).
- Diagnosing a weak or incoherent intervention (the cause is often a poorly
  extracted vector or a wrong injection layer).
- Caching activations for the first time on a new model or dataset.

This skill is NOT for applying the extracted vector to a live model
(see `../../skills/steering-intervention-lib/SKILL.md`) or for evaluating
behavior change (see `../../skills/steering-eval-bundle/SKILL.md`).

---

## 0. The Linear Representation Hypothesis in one line

Concepts encode as approximately linear directions in the residual stream.
Extracting a steering vector = estimating that direction from contrast pairs.
Two estimators are always computed and compared: DiffMean and PCA-top1.

---

## 1. The contrast pair format

A contrast pair is two prompts (or activations) that differ ONLY along the
target behavior axis. The ideal pair is matched on topic, length, and surface
form so the extracted direction is attributable to the behavior, not to
confounds.

```
positive example:  "Tell me how to build a bomb."       (behavior present)
negative example:  "Tell me how to bake a cake."        (behavior absent)
```

For sentiment, toxicity, and style behaviors, the ActAdd / CAA datasets
(Anthropic model-written evals, Rimsky/Panickssery CAA behavior sets) are
the canonical source. For refusal/safety, use Sorry-Bench / Alpaca harmful-
harmless pairs. For fine-grained concept control, use the AxBench concept set
(Stanford, arXiv:2501.17148), which is the primary apples-to-apples benchmark.

For each pair, record activations at the LAST non-padding token of the shared
prefix (the "meaning-bearing" position). This is the standard aggregation used
by CAA; alternatives (mean-pool over full sequence, last token of instruction)
are swept as part of E2.

---

## 2. DiffMean extraction

DiffMean is the difference of class means across the contrast pairs:

```python
# h_pos[i]: activation at layer l for positive example i, shape (d,)
# h_neg[i]: activation at layer l for negative example i, shape (d,)
v_diffmean = mean(h_pos, axis=0) - mean(h_neg, axis=0)
v_diffmean = v_diffmean / np.linalg.norm(v_diffmean)   # unit-norm
```

DiffMean is the maximum-likelihood estimator of the mean difference under
isotropic Gaussian assumptions. Its variance decreases as 1/N (N = number of
pairs), so the pair-count knee experiment (E1) directly measures when this
variance is low enough to be useful.

Why noise cancels: individual activation fluctuations are (approximately)
mean-zero across many prompts, so the shared behavior direction accumulates
while noise averages out. This is the fundamental reason steering needs no
training.

Implementation: `src/steering/extract.py:extract_diffmean(layer_idx, dataset_id)`.

---

## 3. PCA-top1 extraction

PCA-top1 takes the top principal component of the pooled contrast activations:

```python
# Pool all activations: both positive and negative
all_h = np.vstack([h_pos, h_neg])          # shape (2N, d)
# Center and PCA
all_h_centered = all_h - all_h.mean(axis=0)
U, S, Vt = np.linalg.svd(all_h_centered, full_matrices=False)
v_pca = Vt[0]                              # top principal component
# Sign convention: align with diffmean direction
if np.dot(v_pca, v_diffmean) < 0:
    v_pca = -v_pca
```

PCA-top1 captures the direction of maximum variance in the combined
activation cloud. When the behavior is the dominant source of variance
(i.e., the concept is cleanly represented), PCA-top1 and DiffMean converge
to the same direction. High cosine similarity between them is therefore a
quality indicator (Experiment E4).

When PCA-top1 diverges from DiffMean (cosine < 0.8), it often means there
is a confounding axis (e.g., prompt length) that dominates variance.
Diagnose and fix the contrast pairs before proceeding.

Implementation: `src/steering/extract.py:extract_pca_top1(layer_idx, dataset_id)`.

---

## 4. Experiment E1 — the pair-count knee

**Hypothesis:** DiffMean estimation converges to its asymptote at some
finite N; adding more pairs beyond that knee produces no measurable gain in
steering efficacy (behavior score on a fixed held-out set).

**Protocol:**
1. Fix the model (Gemma-3-1B), the layer (use layer 10 as a proxy; will be
   refined by E2), and the eval prompt set (AxBench-mini, 50 prompts).
2. Sweep N_pairs ∈ {4, 8, 16, 32, 64, 128, 256}. For each N, extract
   v_diffmean, apply at alpha=0.8, measure behavior score on the held-out set.
3. Plot behavior score vs N_pairs. Identify the knee: the smallest N where
   the score is within 2 percentage points of the N=256 asymptote.
4. Pre-register: predict knee at N ∈ [16, 64] based on signal-to-noise
   intuition from the CAA paper (Rimsky et al., 2310.01405).

**Gate:** If the knee is at N ≤ 32, use N = 32 as the default extraction
budget for all subsequent experiments (enables fast iteration at Rung-1
SMOKE). If the knee is higher, adjust and log the reason.

The pair-count knee is fixed per behavior type (safety vs sentiment vs
style may differ); always measure it for a new behavior before building a
full extraction pipeline.

---

## 5. Experiment E2 — optimal layer by Fisher ratio

**Hypothesis:** The layer with the highest class-separability of the contrast
activations (measured by Fisher's linear discriminant ratio) is the best
injection layer for that steering vector.

**Fisher ratio at layer l:**

```python
# h_pos_l, h_neg_l: (N, d) matrices at layer l
mu_pos = h_pos_l.mean(axis=0)
mu_neg = h_neg_l.mean(axis=0)
# Between-class variance (scalar, projected onto the mean-difference direction)
S_B = np.dot(mu_pos - mu_neg, mu_pos - mu_neg)
# Within-class variance (pooled)
S_W = np.var(h_pos_l @ (mu_pos - mu_neg) / np.linalg.norm(mu_pos - mu_neg)) \
    + np.var(h_neg_l @ (mu_pos - mu_neg) / np.linalg.norm(mu_pos - mu_neg))
fisher_ratio[l] = S_B / (S_W + 1e-8)
```

**Protocol:**
1. Cache activations at ALL layers for the contrast pairs (done once).
2. Compute Fisher ratio at every layer.
3. Plot Fisher ratio vs layer index.
4. Select `optimal_layer = argmax(fisher_ratio)`, subject to guard-layer C
   constraints (avoid the empirically-fragile mid-layer band if applicable).
5. Cross-validate: run SMOKE at `optimal_layer` and at `optimal_layer ± 2`;
   confirm the selected layer has highest behavior score.

**Pre-registration (before running):** predict that the optimal layer for
Gemma-2-2B refusal is in the range [14, 20] (upper-middle layers), based on
the Rogue Scalpel paper (arXiv:2509.22067 F3) which finds fragility peaks in
early-middle layers, implying that the cleaner representation of the concept
is in later layers.

Implementation: `src/steering/extract.py:compute_fisher_ratio_all_layers()`.

---

## 6. Experiment E4 — cosine agreement diagnostic

**Purpose:** Validate extraction quality. DiffMean and PCA-top1 should agree
on the direction of the behavior axis if the contrast pairs are clean and the
behavior is the dominant variance source.

```python
cosine_agreement = np.dot(v_diffmean, v_pca)   # both unit-normed
```

**Decision thresholds:**

| cosine_agreement | Action |
|---|---|
| ≥ 0.95 | Extraction is clean. Use v_diffmean for all subsequent experiments (lower variance at matched N). |
| 0.80 – 0.95 | Acceptable. Use v_diffmean; log the discrepancy; inspect PCA eigenvalue spectrum for confounds. |
| < 0.80 | STOP. The contrast pairs have a dominant confound (topic, length, style). Fix the pairs before proceeding. |

**Additional diagnostics at E4:**
- PCA eigenvalue spectrum: if eigenvalue[0] is not >> eigenvalue[1], the
  behavior direction is not dominant; the vector is noisy.
- Cosine of v_diffmean with the first 5 PCA components at the injection layer:
  if the first PCA component is not the behavior direction, the within-class
  variance is dominated by something else.

**Default choice:** DiffMean is preferred when cosine_agreement ≥ 0.80
because it is unbiased and has known variance-reduction scaling (1/N).
PCA-top1 is used as a robustness check and becomes the primary estimator
only when it outperforms DiffMean on held-out behavior score at matched N,
which is pre-registered before the comparison.

---

## 7. Activation caching

Caching is non-negotiable on a 16 GB budget. Running a forward pass on the
full dataset at all layers costs ~5–10 minutes per model/dataset pair; it
must happen only once.

Cache layout (written by `src/steering/extract.py`):

```
cache/activations/
  <model_id>/
    <dataset_id>/
      layer_<ll>_positive.pt    # shape (N_pos, d), torch float16
      layer_<ll>_negative.pt    # shape (N_neg, d), torch float16
      stats.json                # {mean_norm: float, eigenvalues: [k], ...}
    metadata.json               # {model_id, dataset_id, n_pairs, seed, sha256_prompt_list}
```

Cache invalidation: if `metadata.json` SHA-256 does not match the current
prompt list, delete and recompute. Never silently use a stale cache.

Memory: Gemma-2-2B has d = 2304, 26 layers, N = 256 pairs.
Full cache size: 26 × 2 × 256 × 2304 × 2 bytes ≈ 600 MB. Fits comfortably
in available RAM (not VRAM; activations are offloaded immediately).

---

## 8. Contrast datasets by behavior type

| Behavior | Primary dataset | Fallback |
|---|---|---|
| Fine-grained concept control | AxBench concept set (arXiv:2501.17148) | Custom pairs from the AxBench template |
| Refusal / harmful compliance | Anthropic CAA behavior sets (Rimsky et al., 2310.01405) | Sorry-Bench harmful-harmless pairs |
| Truthfulness | TruthfulQA contrastive (true vs false statements) | CAA hallucination set |
| Sentiment / toxicity | IMDB / Yelp polarity pairs; RealToxicityPrompts | ActAdd "Love"-"Hate" pairs |
| Safety condition vector | Sorry-Bench harmful vs Alpaca harmless | CAST condition set |
| SAE-feature attack set | GemmaScope SAE feature dictionaries (google/gemma-scope) | Neuronpedia top-1000 features |
| Persona traits | Anthropic persona trait descriptions (arXiv:2507.21509) | Custom trait prompts |

Dataset splits are PINNED (fixed seed, fixed indices). The same held-out set
is used from Rung-1 through Rung-4. Never use the extraction pairs as the
eval set (data leakage). The data-split audit from the meta-process
(`../../meta-skills/autoresearch-meta/SKILL.md` §10) must pass before any
model build on a new dataset.

---

## 9. The identifiability caveat (axis 10)

Per arXiv:2602.06801 (Venkatesh & Kurapath, ICLR 2026 ws), steering vectors
are fundamentally non-identifiable: there exists a large equivalence class of
behaviorally-indistinguishable interventions at any layer. DiffMean and PCA-
top1 are two different gauge choices within this equivalence class.

**Practical implication:** do NOT over-interpret the recovered vector as "the
refusal direction" or "the truth direction". It is A direction that achieves
the behavior at the measured layer. The cosine-agreement diagnostic (E4) is
the cheapest check on whether DiffMean and PCA-top1 pick the same gauge rep,
but neither is canonical. Choose the representative that is most on-manifold
(lowest off-shell displacement after the ADD step), which is the coset-min-
collateral post-processing step from hypothesis N15.

---

## Hard rules

1. NEVER use the extraction pairs as the eval set. Disjoint splits are
   enforced by `audit_or_die()` in `src/steering/extract.py`.
2. ALWAYS run E4 (cosine agreement) before using any vector in a sweep.
   If cosine < 0.80, STOP and fix the pairs.
3. ALWAYS cache activations once; do not recompute inside a sweep loop.
4. Pin contrast-pair indices with a fixed seed before any experiment. Record
   the seed in `metadata.json`.
5. ALWAYS pre-register the predicted knee (E1) and the predicted optimal
   layer range (E2) BEFORE running the experiments.
6. Use unit-normed vectors throughout; the alpha parameter controls magnitude.
7. The default estimator is DiffMean (lower variance); PCA-top1 is a
   diagnostic and an alternative only when pre-registered.

---

## Anti-patterns

| Anti-pattern | Consequence | Do instead |
|---|---|---|
| Same pairs for extraction and eval | Behavior score is inflated; the vector memorized the eval prompts | Strict train/eval split enforced at extraction time |
| Skipping E4 | Using a noisy vector; poor SMOKE results that look like a method failure | Always run cosine agreement diagnostic first |
| Recomputing activations in each sweep trial | 10× slowdown; risk of non-reproducibility | Cache once with pinned prompts |
| Sweeping alpha without a fixed injection layer | Two-axis confound; can't attribute gain | Fix layer (from E2) before sweeping alpha |
| Treating PCA-top1 as "more accurate" than DiffMean | PCA-top1 can pick up confounding axes | Use cosine agreement to choose; DiffMean is default |
| Forgetting sign convention on PCA-top1 | Steers in the wrong direction | Always align PCA sign to DiffMean sign |

---

## Cross-references

- Applying the extracted vector to a live model:
  `../../skills/steering-intervention-lib/SKILL.md`
- Evaluating the extracted vector's behavioral effect:
  `../../skills/steering-eval-bundle/SKILL.md`
- Guard-layer C (avoid fragile layers, relevant to E2 layer selection):
  `../../skills/steering-rogue-scalpel-guard/SKILL.md`
- 12-axis framework (axis 7 = HOW DERIVED, axis 10 = identifiability):
  `corpus/steering-missed-dimensions-and-highdim-algebra.md`
- Meta-process data-split audit discipline:
  `../../meta-skills/autoresearch-meta/SKILL.md`
- Source modules: `src/steering/extract.py`, `src/steering/hooks.py`
