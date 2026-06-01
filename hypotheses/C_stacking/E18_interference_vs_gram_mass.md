# E18 — Interference Grows Monotonically with Off-Diagonal Gram Mass

> **One-line claim:** Interference between stacked behavior vectors grows
> monotonically with the summed off-diagonal Gram mass of the active vector set.
>
> **Source design space:** Block C — Stacking and Multi-Vector Composition (E17–E26).
>
> **Implementation status:** `PENDING — UNTESTED`. Requires multi-vector injection
> across 2–5 vectors and interference measurement. Builds on E17 data.

---

## 1. Motivation (>= 100 words)

E17 tests a binary claim: do two near-orthogonal vectors stack cleanly? E18 asks
a richer quantitative question: does the DEGREE of interference (across all
vector pairs, not just one) scale predictably with the Gram matrix of the stacked
set? The Gram matrix G_ij = cos(v_i, v_j) encodes all pairwise relationships;
its off-diagonal mass sum_ij |G_ij| (i ≠ j) is the total "non-orthogonality"
of the set. If interference scales monotonically with this quantity, we have
a single scalar predictor of stacking quality — a norm-budget partner to N5.
This is the stacking version of N5's collapse-curve prediction: just as N5
predicts a master collapse curve for total offshell displacement, E18 predicts
a monotone interference curve for total Gram off-diagonal mass. The geometry
result N16 (SUPPORTED: angular predicts rotation PPL with R²=0.997, radial
predicts additive PPL with R²=0.81) is relevant because the interference is
mediated by the radial (additive) component of each vector's overlap — Gram
off-diagonal mass is precisely the sum of radial interference terms. Establishing
this monotone relationship enables automatic stack auditing: compute the Gram
matrix, check off-diagonal mass, predict interference without running the full
joint evaluation.

---

## 2. Formal Hypothesis (>= 50 words)

For a set of N stacked behavior vectors {v_1,...,v_N} with alphas {alpha_1,...,alpha_N},
the measured behavioral interference (1 - joint_efficacy / solo_efficacy, averaged
over all behaviors) is a monotone increasing function of the summed off-diagonal
Gram mass:

    M = sum_{i != j} |cos(v_i, v_j)|   (Gram off-diagonal sum)

Formal claim: over the range N in {2, 3, 4, 5} and vector sets covering
|cos(v_i,v_j)| from 0.0 to 0.8 (by design), a monotone regression (Spearman
rho >= 0.7) of interference vs M is statistically supported at 3-seed median
on Gemma-2-2B-it. The relationship need not be linear — monotone is sufficient
for predicting stack quality. The N=2 data point from E17 provides the anchor.

---

## 3. Falsifier (>= 30 words)

If Spearman rho between interference and Gram off-diagonal mass M is below 0.7
across the N={2,3,4,5} sweep, the hypothesis is FALSIFIED — interference is
not predictable from pairwise cosines alone, and more complex interactions
(e.g., three-way Gram terms) dominate. If Spearman rho >= 0.7 but the relationship
is not monotone (i.e., there exists a pair where higher M correlates with lower
interference), the hypothesis is also FALSIFIED.

---

## 4. Citations (Citation Rigor >= 80 words)

```
[N5 geometry result — this project, C2]: logPPL = 5.40 + 2.87 * offshell_R,
R² = 0.81 — the master collapse curve for total offshell displacement; E18
seeks an analogous master curve for interference vs Gram mass; both are
manifestations of the norm-budget conservation law.

[N16 geometry result — this project, C3b]: angular predicts rotation logPPL
R²=0.997; radial predicts additive logPPL R²=0.81 — the radial (additive)
component is the interference channel; Gram off-diagonal mass captures the
sum of radial interference terms across all pairs.

Arditi, Andy, et al. 2024 arXiv 'Refusal in Language Models Is Mediated by a
Single Direction' (arXiv:2406.11717) — refusal direction as v1; the Gram mass
between the refusal direction and any additional behavior vector determines
the interference; this paper motivates measuring Gram mass across safety behaviors.

[Steering-stackable-vs-competing-analysis.md, §3.3, this project] — "Even
independent safety vectors compete once their sum moves h outside the natural
activation manifold" and "Off-diagonal Gram mass = interference" — the decision
matrix that motivates this hypothesis; see KEY INSIGHT 4 in first-principles doc.
```

---

## 5. Mechanism

### 5.1 Interference formula

The joint activation edit for N vectors:

    delta_h = sum_i alpha_i * v_i

The projection of delta_h onto v_j (the interference felt by behavior j):

    interference_j = sum_{i != j} alpha_i * cos(v_i, v_j)

This is exactly the off-diagonal mass (weighted by alpha_i). The behavioral
interference for behavior j is approximately proportional to this value
(assuming behaviors respond linearly to their direction being perturbed).

Total interference index:

    I = (1/N) * sum_j [1 - efficacy_j / efficacy_j_solo]

Expected to be monotone increasing in M = sum_{i!=j} |cos(v_i,v_j)|.

### 5.2 N5 joint norm growth

For N vectors with Gram matrix G:

    ||delta_h||^2 = alpha^2 * [N + 2 * sum_{i<j} cos(v_i,v_j)]
                 = alpha^2 * [N + off-diagonal-Gram-sum]

The off-diagonal Gram sum directly determines the joint norm growth beyond
the pure-orthogonal (Pythagorean) baseline of sqrt(N) * alpha. This links
the Gram mass to the N5 norm budget: high Gram mass → high norm → more logPPL
degradation AND more behavioral interference.

### 5.3 Protocol sketch

Construct vector sets {V_k} (k = 1 to 10) with controlled Gram off-diagonal
masses M_k by mixing vectors from different behaviors and scaling their cosines
via linear combinations:

```python
# Construct a 3-vector set with target Gram mass M
v1 = DiffMean(behavior_A)  # near-orthogonal basis
v2 = DiffMean(behavior_B)
v3 = cos_target * v1 + sqrt(1-cos_target^2) * DiffMean(behavior_C)
# now cos(v1, v3) = cos_target; vary cos_target to sweep M
```

---

## 6. Predicted Delta

| Metric | Predicted value | Rationale |
|---|---|---|
| Spearman rho (interference vs M) | >= 0.70 | Core claim |
| Interference at M ~ 0 (all orthogonal) | < 5% | Orthogonal baseline from E17 |
| Interference at M ~ 2 (2 pairs, cos=0.5) | 15-30% | Moderate off-diagonal mass |
| Interference at M ~ 6 (5-vector, cos=0.5) | 40-60% | High off-diagonal mass |
| N5 norm growth vs M | monotone, R² > 0.80 | N5 law predicts this |

Pre-registered; [NEEDS VERIFICATION] on Gemma-2-2B.
N5 (SUPPORTED) and N16 (SUPPORTED) provide mechanistic grounding.

---

## 7. Experimental Protocol

### 7.1 Primary experiment

- **Model:** Gemma-2-2B-it, 4-bit; smoke on Gemma-3-1B-it
- **Vector sets:** N in {2, 3, 4, 5}; for each N, construct sets at 3 Gram-mass
  levels (low: M < 0.5, medium: M in [0.5,2.0], high: M > 2.0)
- **Interference measurement:** joint efficacy vs solo efficacy for each behavior
  in each set; interference index I = 1 - mean(joint/solo)
- **Gram mass M:** computed from cosine matrix before each run; logged
- **Metric:** Spearman rho (I, M) across all (N, mass-level, seed) data points
- **Seeds:** 3 per (N, mass-level) combination
- **Wall-clock:** ~3 h on 4090 (12 conditions × 3 seeds)

### 7.2 Where it shines

The monotone relationship is most clearly visible when: (a) M range is wide
(from near-0 to > 2), (b) behaviors have similar solo efficacy (so interference
is not confounded by differential solo performance), (c) alpha is below the
N5 cliff for all individual vectors.

---

## 8. Cross-References

- **E17** (near-orthogonal stacking): provides N=2, M ~ 0 data point (anchor)
- **E19** (Gram-Schmidt): reduces M to near-zero; E18 predicts zero interference
  at M = 0 post-orthogonalization
- **E22** (norm budget cap): joint norm = N5 budget input; M governs norm growth
- **N5** (norm budget, SUPPORTED): M determines norm-budget overhead per vector
- **N16** (radial/angular, SUPPORTED): M = sum of radial interference terms
- **N3** (orthogonal capacity theorem): capacity = participation ratio; E18
  gives the empirical interference-vs-M curve that N3's theory should predict
- **IDEA_TABLE.md** Block C row E18

---

## 9. Committee Q&A

**Q: Why Spearman rho (rank correlation) rather than Pearson?**

> The relationship between interference and M need not be linear — it could be
> convex, accelerating as M increases. Spearman captures monotonicity regardless
> of functional form. Pearson would require committing to linearity, which is
> too strong a claim at this stage.

**Q: Is the artificial Gram-mass construction (cos-target blending) ecologically
valid?**

> Partially. Artificially constructed vector sets let us control M precisely,
> enabling the regression. The experiment should also be run with NATURAL vector
> sets (drawn from different safety behaviors) to validate ecological relevance.
> The two analyses (artificial controlled + natural observed) provide complementary
> evidence.

---

## 10. Verification Checklist

- [ ] E17 N=2 data points incorporated (anchor for the regression)
- [ ] 12 (N, mass-level) conditions constructed and logged
- [ ] Gram matrix G_ij computed and logged for each condition
- [ ] Interference index I computed for each condition at 3 seeds
- [ ] Spearman rho (I, M) computed across all conditions; compared to 0.7
- [ ] N5 norm growth vs M also computed and plotted (cross-validation)
- [ ] Natural vector sets (real safety behaviors) also plotted on same curve
- [ ] IDEA_TABLE.md row E18 updated

---

## 11. Status Journal

- 2026-05-31 — Design doc created. Status: UNTESTED / PENDING. Geometry
  grounding: N5 (logPPL = 5.40 + 2.87 * offshell, R² = 0.81, SUPPORTED)
  and N16 (angular vs radial R²=0.997/0.81, SUPPORTED) directly predict the
  Gram-mass / norm-budget / interference relationship. The formula
  ||delta_h||^2 = alpha^2 * [N + off-diagonal-Gram-sum] is an exact linear
  algebra identity, making the norm-growth prediction exact (no approximation).
  The behavioral interference prediction is the empirical test. Code needed:
  multi-vector injection hook; Gram matrix computation; interference index
  computation. Blocked on multi-vector injection.

---

## Addendum: Research-Scientist Critique

*Reviewer: SciCritic-C.*

### Prior plausibility

HIGH for the norm-growth prediction (exact algebra). MEDIUM for the behavioral
interference prediction (depends on behaviors responding linearly to their
direction being perturbed — a reasonable but not guaranteed assumption).

### Mechanism scrutiny

The derivation of the interference formula is exact linear algebra. The behavioral
approximation is the empirical claim. The Spearman rho threshold (0.7) is a
reasonable minimum for a "monotone" relationship.

### Confounds

1. **Alpha variation:** the interference index depends on the alphas of all
   stacked vectors; different alphas will produce different M-weights. The
   protocol should fix all alphas at the same value for the Gram-mass sweep.
2. **Behavior vs coherence interference:** "interference" is defined as efficacy
   degradation. PPL degradation (coherence interference) may follow a different
   Gram-mass curve.

### Verdict

**NOVEL+TESTABLE** with strong N5+N16 geometry grounding. The formula linking
Gram off-diagonal mass to norm growth is an exact prediction; the behavioral
monotonicity is the empirical novelty. High scientific value as the universal
stacking predictor.

---

## Provenance & Tracing

Full per-hypothesis provenance (exact experiments, reproduce commands, artifact links, reasoning trace): [`PROVENANCE/E18.md`](../PROVENANCE/E18.md).

- **Experiments:** analysis campaign (computed quantities in the campaign JSON; see the provenance file).
- **Reproduce:**

```bash
PYTHONPATH=src python scripts/campaign_sweep.py --model models/google/gemma-3-270m-it --quant none --hyp E18 --tag-prefix E18-interf --layers 16 --alphas 0.1 --ops relative_add --behaviors anger formality happiness ocean  # 2-5 vector stacks; interference vs summed off-diagonal Gram mass
```
