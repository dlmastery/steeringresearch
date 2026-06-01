# N18 — Interference-Budget Additivity

> **One-line claim:** Stacking k near-orthogonal safety vectors is safe iff the
> summed interference budget (F-F superposition formula: sum of |cos(v_i, w_j)|
> overlaps) stays below a threshold; degradation is predicted by the sum of cross-
> vector cosine overlaps, not by k alone.
>
> **Primary axes:** A12 (basis/superposition), A3 (how-much/coefficient)
> **Status:** UNTESTED

---

## 1. Motivation (>= 100 words)

Multi-vector stacking is a central use case for activation steering: simultaneously
enforcing refusal of harmful requests, maintaining factual accuracy, and preserving
a specific communication style requires stacking three or more behavior vectors. The
naive intuition is "add more vectors, get more interference." But this is imprecise:
TWO highly-overlapping vectors interfere severely, while FIVE perfectly-orthogonal
vectors do not interfere at all. The interference is not in the NUMBER of vectors but
in their PAIRWISE GEOMETRIC RELATIONSHIPS. The F-F finding from the missed-dimensions
document formalizes this: in a superposition model, the damage from stacking is the
sum over all (i, j) pairs of |alpha_i| * |cos(v_i, w_j)| * sensitivity_j, where
w_j are the directions of other stored features and sensitivity_j is how much the
model output changes when feature j is perturbed. This is the INTERFERENCE BUDGET:
the scalar quantity that must stay below a threshold for safe stacking. The specific
falsifiable claim of N18 is that this sum — not k alone — predicts the degradation
observed in stacking experiments. If true, it provides a QUANTITATIVE stacking law:
a practitioner can compute the interference budget BEFORE stacking, using only the
vector set and a model sensitivity probe, and predict whether the stack will cause
degradation. This is a practical screening tool that eliminates the trial-and-error
of stacking experiments.

## 2. Formal Hypothesis (>= 50 words)

Let V = {v_1, ..., v_k} be a set of k behavior vectors with alpha_i. Define the
interference budget IB(V) = sum_{i != j} |alpha_i| * |cos(v_i, v_j)|. The claim is:

  Spearman(IB(V), joint-behavior-degradation) >= 0.70

across configurations with k in {2, 3, 4, 5}, alpha_i in {0.5, 1.0, 2.0}, and
random vector sets from the synthetic behavior suite, on Gemma-3-1B-it.

Furthermore, IB(V) predicts degradation better than k alone:
  Spearman(IB, degradation) > Spearman(k, degradation) by >= 0.10.

## 3. Falsifier (>= 30 words)

If Spearman(IB, degradation) < 0.50, or if k alone predicts degradation at least as
well as IB (Spearman difference < 0.05), the interference-budget claim is FALSIFIED.
If IB predicts direction but not magnitude of degradation, status is INCONCLUSIVE.

## 4. Citations (Citation Rigor >= 80 words)

```
Schwinn et al. 2025. 'Rogue Scalpel' arXiv:2509.22067. The rogue damage from
multiple feature-steering vectors is the sum of interference terms; Rogue Scalpel's
817/1000 result shows that the interference is widespread when cross-cosines are
non-negligible. N18 quantifies this as a budget formula and tests its predictive power.

Turner et al. 2023. 'Activation Addition' arXiv:2312.06681. Multi-vector addition
is the baseline method; N18 predicts the failure condition from the CAA stacking regime.

Venkatesh & Kurapath 2026. 'Non-Identifiability' arXiv:2602.06801. The effective
subspace structure means interference only occurs in the d_active dimensions; in the
null space, overlaps are behaviorally inert. N18's interference budget should be
computed in the effective subspace (not full d_model), using only the top-PR eigenvectors.
This is the connection to N3 (participation ratio governs capacity).

Wurgaft et al. 2026. 'Manifold Steering' arXiv:2605.05115. Off-manifold excursions
from stacked vectors compound: IB measures the expected total off-manifold displacement
from all cross-interference terms. N18's threshold is the N5 norm-budget threshold
applied to the cross-interference component.
```

## 5. Mechanism

The superposition model (F-F): in high dimensions, features {w_j} are stored as
approximately orthogonal directions. The activation h = sum_j a_j w_j (sparse).
Adding a steering vector alpha_i v_i perturbs the read-out of other features:
  delta(a_j) = alpha_i * cos(v_i, w_j)

If alpha_i * cos(v_i, w_j) is non-negligible, feature j is perturbed even though
the steering was aimed at i. For stacking k vectors:
  total_delta(a_j) = sum_i alpha_i * cos(v_i, w_j)

The total interference on j from the full stack is the sum over i of the cross-cosines.
The aggregate interference budget IB = sum_{i != j} |alpha_i| * |cos(v_i, v_j)| is
the first-order approximation of total cross-feature perturbation (using the behavior
vectors v as proxies for the feature directions w, which holds when behavior vectors
are aligned with feature directions — the N4/N8 assumption).

Threshold: the threshold below which stacking is "safe" corresponds to IB < IB_safe
where IB_safe is the steering budget B (N5) minus the individual steering contributions.
For near-orthogonal vectors (|cos| < 0.1), IB ≈ 0 for any k — this is why near-
orthogonal stacking works in E17. For partly-overlapping vectors (|cos| ~ 0.3), IB
grows linearly with k — explaining why coverage "leaks" after N~5 in E11.

## 6. Predicted Delta

| Metric | Predicted Value | Rationale |
|---|---|---|
| Spearman(IB, degradation) | 0.70 - 0.85 | F-F superposition formula |
| Spearman(k, degradation) | 0.30 - 0.50 | k is a crude proxy for IB |
| IB advantage over k alone (Spearman diff) | >= 0.10 | The claim |
| IB threshold for safe stacking | ~0.30 - 0.50 total | Empirically determined |
| N_safe vectors before IB exceeds threshold | 3-10, depending on cos(v_i,v_j) | Matches N3's PR prediction |

## 7. Protocol

### 7.1 Primary experiment

- Model: Gemma-3-1B-it @L16
- Vector sets: sample 50 random subsets of k behaviors (k in {2,3,4,5}) from the
  10-behavior primitive pool, with varying pairwise cosines
- alpha: all vectors at alpha = 1.0 * ||h|| / k (equal budget allocation)
- IB computation: IB(V) = sum_{i != j} |alpha_i| * |cos(v_i, v_j)|
- Degradation metric: joint behavior success rate (fraction of steered behaviors
  achieving > 50% behavior-cosine shift), relative to solo performance
- k-alone model: Spearman(k, degradation) as comparison
- Seeds: 3 vector-set sampling seeds x 3 evaluation seeds
- Total configurations: 50 subsets x 4 sizes x 3 x 3 = 1800 cells (use 200 random)
- Wall-clock: ~4 hours on RTX 4090

### 7.2 Where it shines

Deployment planning: before deploying a k-vector safety stack, compute IB to predict
whether the stack will cause degradation. If IB < IB_safe, proceed; if IB >= IB_safe,
apply Gram-Schmidt (E19) to reduce the cross-cosines and lower IB.

## 8. Cross-References

- N3 (orthogonal capacity): N3 predicts N_safe from PR; N18 predicts N_safe from IB;
  they should agree when IB is dominated by the PR-related capacity constraint
- N5 (norm-budget): IB contributes to the total offshell displacement; IB < IB_safe
  should correspond to total offshell < B (the N5 budget)
- E18 (interference vs Gram mass): E18 tests monotonicity of interference vs Gram
  off-diagonal — N18 makes this quantitative with the IB formula and a predictive target
- E19 (Gram-Schmidt orthogonalization): the remedy when IB exceeds threshold; N18
  motivates when to apply E19
- E17 (near-orthogonal stacking): if |cos(v_i,v_j)| < 0.1, IB ≈ 0 and N18 predicts
  no degradation — consistent with E17's prediction
- N12 (capstone): the basis/superposition axis (A12) in the unified operator uses
  the IB formula to determine when Gram-Schmidt pre-processing is required
- IDEA_TABLE.md: N18 row, axes A12+A3

## 9. Committee Q&A

**Q: The IB formula uses behavior vectors v_i as proxies for the underlying feature
directions w_j. But behavior vectors are aggregate (DiffMean) and the features w_j
are individual neurons/circuits. Isn't this a poor approximation?**

> Yes, this is an approximation. The quality of the proxy improves when: (a) behavior
> vectors are aligned with feature directions (the N8 causal-feature claim), and (b)
> there are few active features (sparse activations, typical for MLP layers). An
> explicit test of the proxy quality: compute IB with DiffMean vectors, IB with
> GemmaScope SAE feature vectors, and compare their predictive power. If SAE features
> give higher Spearman, the proxy approximation is the bottleneck.

**Q: The budget allocation alpha = 1.0 * ||h|| / k gives equal budget to all vectors.
In practice, behaviors have different optimal alpha. How does heterogeneous alpha
affect the IB formula?**

> IB is a WEIGHTED sum: IB = sum_{i!=j} |alpha_i| * |cos(v_i, v_j)|. Heterogeneous
> alpha shifts the weight of each term. The protocol includes an ablation with
> alpha_i drawn from the per-behavior optimal values (identified in the solo steering
> sweep); the IB formula is the same, just with different weights. The test should
> confirm that IB (not k, not total alpha) is the predictor regardless of the
> alpha allocation strategy.

## 10. Verification Checklist

- [ ] IB formula implemented and validated on synthetic orthogonal/overlapping vector pairs
- [ ] 200 random vector subsets (k in {2,3,4,5}) sampled, pairwise cos documented
- [ ] Solo behavior success rates logged as reference for degradation computation
- [ ] IB computed for all 200 subsets; degradation measured for all subsets
- [ ] Spearman(IB, degradation) and Spearman(k, degradation) computed with p-values
- [ ] SAE feature proxy comparison (if SAE vectors available): IB_SAE vs IB_DiffMean predictive power
- [ ] IB threshold identified from empirical distribution (50th percentile of safe vs degraded)
- [ ] IDEA_TABLE.md N18 row updated

## 11. Status Journal

- 2026-05-31 — Design doc created. Status: UNTESTED. The F-F superposition formula
  is the mathematical grounding; the E18 experiment (interference vs Gram mass) is
  the experimental predecessor and is also UNTESTED. N18 is the quantitative version
  of E18, adding the specific IB formula and the predictive Spearman claim. No data
  exists for either.

---

## Addendum: Research-Scientist Critique

### Prior plausibility

MEDIUM. The superposition model (F-B, F-F) is well-motivated by the Anthropic toy-model
work on feature superposition. The IB formula is a natural first-order linearization.
The main empirical question is whether DiffMean vectors are good proxies for the true
feature directions w_j — which is testable with the SAE comparison.

### Mechanism scrutiny

The IB formula assumes linear interference: total damage = sum of pairwise overlaps.
This ignores higher-order interactions: three nearly-aligned vectors may interfere
constructively (amplifying the damage beyond the pairwise sum) or destructively
(canceling). For k=2 and k=3, the linear approximation is likely adequate; for k=5+,
higher-order corrections may matter.

### Confounds

1. The degradation metric (joint behavior success rate) is measured at the same
   total alpha, but higher k means weaker individual alpha per vector. Some of the
   "degradation" from higher k may simply be insufficient individual alpha.
   Control: measure degradation at iso-individual-alpha (not iso-total-alpha).
2. The pairwise cosines are measured between extracted behavior vectors; the true
   interference depends on the model's sensitivity, which varies by layer and
   direction. A sensitivity-weighted IB (IB_sens = sum |alpha_i| * |cos| * sensitivity_j)
   may be more predictive.

### Does the IB formula specifically matter?

MODERATELY. The claim that SUM of pairwise overlaps (not COUNT k) predicts
degradation is practically important: it tells practitioners to worry about the
GEOMETRY of the vector set, not just its size. Even if Spearman is 0.60 (below
0.70 threshold), the directional finding is useful for stacking design.

### Literature precedent

Rogue Scalpel (arXiv:2509.22067) shows that the interference is widespread and
depends on feature overlaps; N18 makes this predictive with a closed-form formula.
The formula itself is not new (it is the standard superposition model's cross-term),
but its application as a predictive tool for safe stacking is novel.

### Skeptical effect-size estimate

Spearman(IB, degradation): 0.55-0.70 (vs claimed 0.70-0.85). Main risk: the proxy
approximation (DiffMean as feature direction proxy) reduces correlation. Spearman(k,
degradation): 0.25-0.45 (vs claimed 0.30-0.50). The IB advantage over k: 0.05-0.15
— may be below the claimed 0.10 threshold if the proxy is poor.

### Minimum distinguishing experiment

k=2 vs k=3 at controlled pairwise cosines: 5 pairs with |cos|=0.05 (near-orthogonal)
and 5 pairs with |cos|=0.30 (moderate overlap). Predict IB_low < IB_high; measure
degradation. If degradation(high-cos) > degradation(low-cos) consistently, the
directional claim holds. Cost ~1 hour.

### Verdict

TESTABLE-MEDIUM. The minimum experiment (1 hour) tests the directional claim cleanly.
The quantitative Spearman >= 0.70 threshold may be too demanding given the proxy
approximation; consider lowering to 0.55 as the primary threshold with 0.70 as the
aspirational target. Recommend the sensitivity-weighted IB variant as a secondary test.

---

## Provenance & Tracing

No experiments run yet — see this design doc's protocol (§7) for what would be run. Once a campaign logs rows for this hypothesis, re-run `scripts/build_provenance.py` to generate `hypotheses/PROVENANCE/N18.md`.
