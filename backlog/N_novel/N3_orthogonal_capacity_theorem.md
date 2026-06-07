# N3 — The Orthogonal Capacity Theorem (Empirical)

> **One-line claim:** The number of behaviors stackable without interference equals
> the effective local dimensionality (participation ratio) of the activation manifold
> at the injection layer, not the hidden size d_model; stacking degrades sharply once
> N exceeds the participation ratio.
>
> **Primary axes:** A12 (basis/superposition), A3 (how-much/coefficient)
> **Status:** UNTESTED

---

## In Plain English

**What we're testing, simply:** When you ask the model to do several steered things
at once (be safe *and* polite *and* factual), the behaviors start stepping on each
other. This doc asks: *how many* can you stack before they collide? The answer it
proposes isn't the model's raw size — it's how many genuinely independent "things"
the model's internal state is juggling at that processing step.

**Key terms (defined here):**
- **Steering / steering vector** — changing behavior by adding a chosen direction to
  the model's internal "thought" mid-sentence; each behavior is one arrow.
- **Residual stream** — the model's running internal thought; what we edit.
- **Layer** — one of the model's processing steps; a knob.
- **Stacking** — applying several steering arrows at the same time.
- **Surface (manifold)** — the natural region where healthy thoughts live.
- **Participation ratio / effective rank** — how many independent directions the
  thought is really using at that step (its "room"). A small number means little room;
  a big number means lots of room.
- **Near-orthogonal** — arrows pointing in genuinely different directions, so they
  don't overlap and interfere.

**Why we're doing this (the point):** If we can read off "how much room" a layer has,
we can predict in advance how many behaviors it will hold before they start clashing —
no trial-and-error.

**What the result would mean:** A win means the "room" number reliably predicts the
breaking point of stacking. A loss means stacking capacity is governed by something
else (like layer depth) instead.

See [`../GLOSSARY.md`](../GLOSSARY.md) for any other term.

---

## 1. Motivation (>= 100 words)

When practitioners stack multiple steering vectors simultaneously (e.g., a safety
vector, a politeness vector, and a factuality vector), they observe interference:
behaviors that worked independently degrade when combined. The intuitive explanation
is "the vectors are not orthogonal." But orthogonality in R^d_model is not the
right metric, because not all d_model directions are behaviorally active. The
activation manifold at any given layer has an effective intrinsic dimensionality
much lower than d_model — this is the participation ratio (PR), defined as the
squared sum of eigenvalues divided by the sum of squared eigenvalues of the
activation covariance matrix: PR = (sum lambda_i)^2 / (sum lambda_i^2). PR measures
how many "effective directions" the distribution uses. The Johnson-Lindenstrauss
principle (F-B from the missed dimensions document) guarantees that in R^k one can
pack O(k) near-orthogonal unit vectors; in a k-dimensional subspace, packing more
than ~k vectors means at least one pair has non-negligible cosine overlap. Since
behavior vectors extracted by DiffMean live primarily in the high-variance subspace
of activations (they are means of natural activations, hence correlated with the
principal components of the covariance), the effective capacity for non-interfering
vectors is not d_model but PR. This is the orthogonal capacity theorem: N_safe ~ PR
at the injection layer. Layers with high PR (rich, high-dimensional distributions)
accommodate more simultaneous behaviors; layers with low PR (bottleneck layers,
embedding layers) are quickly saturated.

## 2. Formal Hypothesis (>= 50 words)

Let PR(L) be the participation ratio of the activation covariance at layer L,
computed on 1000 natural (unsteered) forward passes. Let N_knee(L) be the number
of simultaneously stacked vectors at which the joint behavior success rate drops
below 85% of the average solo success rate. The claim is:

  Spearman(PR(L), N_knee(L)) >= 0.70

measured across layers L in {8, 12, 16, 20, 24} on Gemma-3-1B, using 5 near-
orthogonal behavior vectors drawn from the synthetic suite. Additionally, N_knee(L)
<= 1.5 * PR(L) / d_model_scale for all L, where d_model_scale is a normalizing
constant.

## 3. Falsifier (>= 30 words)

If Spearman(PR(L), N_knee(L)) < 0.50 across the five tested layers, or if N_knee
is independent of PR and instead tracks Fisher ratio or simple layer depth, the
orthogonal capacity theorem is FALSIFIED. Status moves to `FALSIFIED`. If the
correlation is 0.50-0.69, status is `INCONCLUSIVE`.

## 4. Citations (Citation Rigor >= 80 words)

```
Johnson, William B. & Lindenstrauss, Joram. 1984. 'Extensions of Lipschitz
mappings into a Hilbert space.' Contemporary Mathematics 26. The JL lemma
establishes that O(1/epsilon^2 * log n) dimensions suffice to embed n points
with epsilon distortion; equivalently, packing near-orthogonal vectors in k
dimensions is limited to O(exp(k)) unit vectors at epsilon-orthogonality.
This is the mathematical foundation for N3's capacity argument: when effective
dimension is PR not d_model, the cap on stackable near-orthogonal vectors scales
with PR.

Turner et al. 2023. 'Activation Addition: Steering Language Models Without
Optimization' arXiv:2312.06681 (CAA). The practical stacking setup N3 tests
against; CAA observes interference between stacked vectors but does not explain
it geometrically.

Venkatesh & Kurapath 2026. 'On the Non-Identifiability of Steering Vectors'
arXiv:2602.06801. Shows that the null space of downstream readouts is large;
equivalently, many directions in R^d_model are inert. PR measures the complement
of the null space — the active subspace — which is exactly the capacity we claim
limits stacking.

Gao et al. 2026. 'CRH' arXiv:2605.01844 (ICML 2026). CRH's effective-rank
analysis of angular vs radial steering is a direct empirical probe of the local
dimensionality at steering layers; their rank statistics map directly onto PR.
```

## 5. Mechanism

Participation ratio PR = (sum_i lambda_i)^2 / (sum_i lambda_i^2) for eigenvalues
{lambda_i} of Cov(h). For a uniform distribution over k orthogonal directions,
PR = k exactly. For a distribution concentrated in one direction, PR = 1. PR
thus measures the effective number of directions the distribution "uses."

Why behavior vectors cluster in the high-variance subspace: DiffMean vectors are
differences of conditional means, and the Fisher discriminant theorem shows that
the discriminative direction lies in the span of the between-class scatter matrix,
which is a subset of the within-class covariance eigenvectors (the high-lambda
directions). Therefore, N behavior vectors drawn by DiffMean all live in
approximately the same PR-dimensional subspace. Stacking N > PR vectors in a
PR-dimensional subspace necessarily produces pairwise overlaps >= N^(-1/2)
(by a pigeonhole-style argument on unit vectors in R^PR), which translates to
interference via the F-F superposition budget: each vector "bleeds" into others
at rate cos(v_i, v_j) * sensitivity_j, and when cos >= N^(-1/2) this becomes
non-negligible.

Predicted N_knee: for PR ~ 50-200 (typical for mid-layers of a 1B model) and
d_model = 2048, N_knee ~ sqrt(PR) ~ 7-14 simultaneous vectors. This matches the
empirical observation from E11 that coverage leaks after N~5 and from E47's
"3 orthogonal vectors" recipe.

## 6. Predicted Delta

| Metric | Predicted Value | Rationale |
|---|---|---|
| Spearman(PR, N_knee) across layers | >= 0.70 | PR is the geometric capacity |
| N_knee at high-PR layers (L16, L20) | 8-15 vectors | PR ~ 100-200 at these layers |
| N_knee at low-PR layers (L8) | 2-5 vectors | PR ~ 20-40 at early layers |
| PR correlation with Fisher ratio | 0.3 - 0.6 | Partially overlapping, not identical |
| Improvement in stacking from PR-guided layer selection | +10% to +20% joint success | Choosing high-PR layers for injection |

## 7. Protocol

### 7.1 Primary experiment

- Model: Gemma-3-1B-it
- Layers: {8, 12, 16, 20, 24}
- Natural activations: 1000 forward passes on WikiText-103 validation, extract h_L
- PR computation: eigendecompose sample covariance Cov(h_L); compute PR formula
- Behavior vectors: 5 DiffMean vectors from 5 distinct synthetic behaviors
  (polite/rude, factual/evasive, concise/verbose, formal/casual, direct/hedged)
- Stacking: add k=1,2,3,4,5 vectors simultaneously at each layer, measure joint
  behavior success rate as fraction of (behavior_i goal met) averaged over k behaviors
- N_knee: find smallest k where joint success drops below 85% of average solo success
- Metrics: PR(L), N_knee(L), Spearman correlation
- Seeds: 3 behavior-extraction seeds x 3 evaluation prompt seeds
- Wall-clock: ~4 hours on RTX 4090

### 7.2 Where it shines

Layer-selection experiments (E14, N20): PR is a behavior-free predictor of stacking
capacity, so it can guide layer selection before any behavior extraction is done.
If N3 holds, PR is a cheap pre-screening tool for safe injection site selection.

## 8. Cross-References

- N5 (norm-budget): N3 explains WHICH vectors saturate the budget (those in the low-PR
  subspace interfere most); N5 measures WHEN the budget is saturated
- N18 (interference-budget additivity): N18 predicts degradation from sum-of-cos;
  N3 predicts the threshold from geometry; they are complementary predictions
- N17 (concentration penalty): off-shell displacement is the outcome variable;
  N3 predicts when stacking pushes the aggregate delta-h large
- F-B (JL lemma from missed dimensions doc): the mathematical foundation
- E18 (interference vs Gram mass): the experimental precedent N3 must subsume
- IDEA_TABLE.md: N3 row, axes A12

## 9. Committee Q&A

**Q: Participation ratio measures the global covariance — why would it predict
local stacking capacity for a specific input distribution?**

> Fair point. PR is a global statistic; local PR near a specific activation h
> (computed on the k nearest neighbors) would be more precise. The experiment
> should additionally test local PR vs N_knee to check whether global PR is
> sufficient or local PR is needed. Prediction: global PR is a lower bound;
> local PR at the typical prompt activation is a tighter predictor.

**Q: DiffMean vectors are not guaranteed to lie in the high-variance subspace.
A behavior like "refuse harmful requests" might be in a low-variance direction.**

> True, and this is a genuine uncertainty. The falsifier allows for this: if
> PR predicts nothing (rho < 0.50), one candidate explanation is that behavior
> vectors span directions outside the natural covariance eigenvectors. In that
> case N3 would be FALSIFIED and the correct capacity would be a different
> geometric quantity (e.g., the rank of the behavior-vector Gram matrix itself).

**Q: Is the 85% joint-success threshold for N_knee arbitrary?**

> It is pre-registered (see Section 2) to avoid post-hoc threshold selection.
> The 85% threshold was chosen to match the "no cross-degradation" criterion
> from E17 (90% at k=2) adjusted down slightly for k>2 stacking.

## 10. Verification Checklist

- [ ] PR computation implemented and validated on synthetic data (uniform k-dim
      distribution should give PR=k within 5%)
- [ ] PR measured at all 5 layers and recorded in EXPERIMENT_LEDGER.md
- [ ] N_knee measurement protocol implemented with 85% threshold
- [ ] 5 behaviors x 5 layer x 5 stacking depths = 125 cells recorded
- [ ] Spearman correlation computed and reported with p-value
- [ ] Local PR vs global PR ablation added to run
- [ ] Result reflected in IDEA_TABLE.md N3 row; cross-ref N18, N5 docs updated

## 11. Status Journal

- 2026-05-31 — Design doc created. Status: UNTESTED. The JL lemma provides the
  mathematical backbone; no empirical data for LLM steering exists. E18 (interference
  vs Gram mass) is the closest existing test but has not been run. N3 predicts that
  E18's "Gram mass" predictor is a proxy for PR-based interference.

---

## Addendum: Research-Scientist Critique

### Prior plausibility

MEDIUM-HIGH. The JL-based argument is mathematically sound. The participation ratio
is a well-established concept in random matrix theory and neuroscience (Litwin-Kumar
et al. 2017 use it to study neural dimensionality). The question is whether the
behavioral-vector subspace overlaps with the high-PR subspace of natural activations.

### Mechanism scrutiny

The Fisher discriminant argument for why DiffMean vectors lie in the high-variance
subspace is an approximation: it assumes the within-class covariance of the contrast
set matches the natural activation covariance. If the contrast-set covariance is
concentrated in different directions, DiffMean vectors may span a subspace orthogonal
to the high-PR natural directions — in which case stacking capacity is governed by
a different geometric object. This is testable: compute cos(v_i, PC_j) for each
behavior vector and each principal component.

### Confounds

1. Interference might be driven by magnitude, not angle: even orthogonal vectors
   can interfere via nonlinear LayerNorm effects if their combined magnitude is large.
   N3 must control for total ||sum alpha_i v_i|| (the N5 norm-budget confound).
2. PR varies with the corpus: PR(L) computed on WikiText may differ from PR(L)
   computed on the steering-relevant prompt distribution (harmful/harmless prompts).
   If prompt-conditioned PR is much lower than corpus PR, N_knee will be
   overestimated by the corpus-based PR.

### Does PR specifically matter?

UNCERTAIN. The correlation with N_knee might be driven by layer depth (later layers
tend to have both higher PR and better stacking capacity) rather than by the geometric
claim. Controlling for layer depth in the Spearman analysis is essential.

### Literature precedent

Mante et al. 2013 (Science, "Context-dependent computation by recurrent dynamics")
show that representational subspace dimensionality gates the number of tasks a neural
population can simultaneously encode. The LLM steering analogy is direct but untested.
Geiger et al. 2024 (arXiv:2301.04709, Interchange Intervention Training) shows that
causal intervention efficacy correlates with the rank of the causal subspace, which
is related to PR.

### Skeptical effect-size estimate

Spearman(PR, N_knee) = 0.40-0.60 (vs claimed >= 0.70). Rationale: layer depth
confound will eat some of the correlation; PR captures the right concept but the
exact formula may not be the most predictive geometric quantity. Still, 0.40-0.60
would support the qualitative claim that high-PR layers accommodate more vectors.

### Minimum distinguishing experiment

Two layers: L8 (expected low PR) and L16 (expected high PR). Stack k=1..5 vectors
and measure joint success. If N_knee(L16) > N_knee(L8) and the ratio matches PR(L16)/PR(L8)
within 50%, the claim is supported. Cost ~1 hour. Run before the full 5-layer protocol.

### Verdict

TESTABLE-MEDIUM-CONFIDENCE. The mathematical argument is clean; the empirical test is
direct. The main risk is the layer-depth confound and the corpus-vs-prompt PR mismatch.
The minimum experiment (2 layers, ~1 hour) should be run first to decide whether the
full protocol is worth the 4 hours.

---

## Pseudocode & Methodology

This section specializes [`../METHODOLOGY.md`](../METHODOLOGY.md). N3 predicts the number of stackable behaviors equals the **participation ratio** of the layer, not d_model. **UNTESTED** — the probe exists, the stacking-knee sweep does not.

### 1. Steering-vector recipe (capacity = participation ratio)

The per-layer capacity quantity is computed directly from `geometry.participation_ratio`:

```python
# 1000 natural forward passes -> activation batch H_L  [n, dim]
PR_L = participation_ratio(H_L)          # geometry.participation_ratio: (Σλ)²/Σλ²  (N3 at injection layer)

# behavior vectors are ordinary DiffMean, stacked additively (METHODOLOGY §2 add):
#   h' = h + Σ_{i=1..k} alpha_i v_i           v_i = bank[L][i]["diffmean"]
```

### 2. Experiment procedure

```text
1. For L in {8,12,16,20,24}: PR_L = participation_ratio(1000 natural activations at L).
2. Extract 5 near-orthogonal DiffMean behavior vectors.
3. For each L, stack k=1..5 vectors; measure joint behavior success (fraction of behaviors meeting goal).
4. N_knee(L) = smallest k where joint success < 0.85 * mean solo success.
5. rho = Spearman(PR_L, N_knee_L) across the 5 layers; ablate local-PR (kNN) vs global-PR.
   CONTROL the §confound: hold total ‖Σ alpha_i v_i‖ fixed (N5 norm-budget) so interference ≠ magnitude.
```

### 3. Measurement & decision rule

- **Primary metric:** Spearman(PR(L), N_knee(L)) across layers.
- **Pre-registered falsifier (§3):** Spearman < 0.50, OR N_knee tracks Fisher/depth instead of PR ⇒ FALSIFIED; 0.50–0.69 ⇒ INCONCLUSIVE.
- **Verdict logic:** must survive the layer-depth partial correlation (PR, not depth, drives the result).

### 4. Where the code is / status

UNTESTED. `geometry.participation_ratio` is implemented and tested; the **multi-vector stacking-knee sweep** (joint-success evaluation over k stacked vectors with matched norm budget) is the missing driver — that is why N3 is UNTESTED.

See [`../METHODOLOGY.md`](../METHODOLOGY.md) for the shared recipe.

---

## Provenance & Tracing

No experiments run yet — see this design doc's protocol (§7) for what would be run. Once a campaign logs rows for this hypothesis, re-run `scripts/build_provenance.py` to generate `hypotheses/PROVENANCE/N3.md`.
