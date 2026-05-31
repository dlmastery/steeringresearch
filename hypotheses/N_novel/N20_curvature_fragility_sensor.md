# N20 — Curvature as a Fragility Sensor

> **One-line claim:** Local manifold curvature (or local effective-rank collapse)
> at a layer predicts that layer's rogue-fragility, giving a cheap, behavior-free
> way to pick safe injection layers. Screening: eff-rank Spearman -0.21 with
> fragility (underpowered, n=1); but max-Fisher L12 = most fragile (C4, consistent).
>
> **Primary axes:** A8 (geometry/curvature), A1 (where/site)
> **Status:** INCONCLUSIVE (screening, n=1 — underpowered; not an external claim)

---

## 1. Motivation (>= 100 words)

Choosing the right injection layer is one of the most important decisions in activation
steering. The literature uses Fisher ratio (linear separability) or empirical search;
screening result S-5 FALSIFIED the Fisher-ratio predictor (rho=0.14 on Gemma-270m).
A better predictor is needed. The geometric hypothesis is: layers with high local
curvature of the activation manifold are fragile — they amplify small perturbations
into large behavioral changes (including unintended rogue behaviors). High curvature
means the manifold "bends sharply" at that layer; a perturbation of fixed magnitude
causes a larger effective behavioral change than at a low-curvature layer. This is
why max-Fisher L12 was the most fragile layer in the C4 screening: Fisher ratio
measures discriminative sharpness, which is related to local curvature (a high-Fisher
layer has activations concentrated along a few directions — a signature of high
effective curvature). But Fisher ratio is not the right proxy because it mixes
discrimination ability (useful) with curvature (fragility-inducing). A direct
curvature measurement — effective rank of the local activation covariance — separates
the two. Low effective rank (concentrated covariance) = high curvature = high fragility.
High effective rank (spread covariance) = low curvature = low fragility, safe to steer.
The prediction is: effective rank (or participation ratio) NEGATIVELY predicts fragility,
i.e., layers with high PR are safe injection targets.

## 2. Formal Hypothesis (>= 50 words)

Let eff_rank(L) be the effective rank of the activation covariance at layer L (computed
as exp(entropy of the normalized eigenvalue distribution)). Let fragility(L) be the
rogue-compliance rate CR increase per unit alpha at layer L (measured by steering with
the max-Fisher direction and recording CR). The claim is:

  Spearman(eff_rank(L), fragility(L)) <= -0.60

(negative: high eff_rank = low fragility). Measured across layers L in {8, 10, 12,
14, 16, 18, 20, 22} on Gemma-3-1B-it. Additionally, selecting the layer with max
eff_rank as the injection layer should give the lowest CR at matched behavior.

## 3. Falsifier (>= 30 words)

If Spearman(eff_rank, fragility) is in (-0.40, +0.20) across 8 layers on Gemma-3-1B-it
(i.e., the correlation is weak or wrong sign), the curvature-fragility claim is
FALSIFIED. The C4 screening had rho=-0.21 (negative but weak); the held-out test
targets rho <= -0.60 for the full claim to hold.

## 4. Citations (Citation Rigor >= 80 words)

```
Wurgaft et al. 2026. 'Manifold Steering' arXiv:2605.05115. The M_h manifold curvature
governs the divergence between geodesic and chord paths; high curvature = large chord-
geodesic gap = large potential for off-manifold damage. Effective rank captures the
manifold's local intrinsic dimensionality, which inversely relates to curvature:
low-dimensional (low eff_rank) = high curvature.

Schwinn et al. 2025. 'Rogue Scalpel' arXiv:2509.22067. The rogue-compliance rate
is the primary fragility metric; 817/1000 random SAE features cause rogue compliance
at specific layers. N20 predicts that the LAYER of maximum rogue compliance is
predictable from eff_rank without any behavioral probing.

Venkatesh & Kurapath 2026. 'Non-Identifiability' arXiv:2602.06801. The null space
of downstream readouts is large at all layers; however, the BEHAVIORALLY ACTIVE
subspace has layer-specific dimensionality. Effective rank at a layer measures the
size of this active subspace; low eff_rank = small active subspace = steering
has high leverage = high fragility.

SCREENING RESULT: C4 campaign (from IDEA_TABLE.md N20 row): eff_rank Spearman -0.21
with fragility, underpowered; but max-Fisher L12 = most fragile layer (consistent
with the high-curvature = high-fragility direction). n=1, INCONCLUSIVE.
```

## 5. Mechanism

Effective rank: eff_rank(L) = exp(H(lambda)), where H(lambda) = -sum_i (lambda_i /
sum lambda_j) * log(lambda_i / sum lambda_j) is the Shannon entropy of the normalized
eigenvalue distribution. eff_rank = 1 when all variance is in one direction (most
concentrated, highest curvature); eff_rank = d_model when all directions have equal
variance (isotropic, lowest curvature).

Curvature-fragility link: at a high-curvature (low eff_rank) layer, the activation
manifold is concentrated along a few principal directions. A steering vector that is
not perfectly aligned with these directions has a large component in the "off-
manifold" space relative to the layer's intrinsic dimensionality. This off-manifold
component causes a disproportionately large behavioral change (high leverage) per unit
of steering magnitude — which is precisely what "fragility" means (high rogue-compliance
rate per unit alpha).

Prediction: eff_rank(L) negatively predicts fragility(L) because:
  fragility(L) ≈ C / eff_rank(L) * sensitivity(L)

where sensitivity(L) is the behavioral sensitivity (delta_behavior / delta_h) at
layer L. Layers with both low eff_rank AND high sensitivity are the most dangerous.

Fisher ratio vs eff_rank: E2 showed that Fisher ratio (linear separability) fails
to predict steering efficacy (rho=0.14). Fisher ratio measures the between-class
variance / within-class variance — a ratio that is high when the target behavior
is well-separated at that layer. This is useful for DETECTION (N2, gating) but
is not the right proxy for FRAGILITY. Effective rank measures the spread of the
activation distribution, not its discriminativeness — a different geometric quantity.

## 6. Predicted Delta

| Metric | Predicted Value | Rationale |
|---|---|---|
| Spearman(eff_rank, fragility) across 8 layers | <= -0.60 | Curvature-fragility link |
| Layer with max eff_rank | Lowest fragility among tested layers | Main practical claim |
| Layer with min eff_rank | Highest fragility (probably L12 from C4) | Consistent with C4 screening |
| Behavior efficacy at max-eff_rank layer | Not necessarily highest | Efficacy and safety are different |
| Spearman(Fisher, fragility) | 0.10 - 0.30 (weak, wrong direction) | E2 result applied to fragility |

## 7. Protocol

### 7.1 Primary experiment

- Model: Gemma-3-1B-it (held-out from C4 which used Gemma-270m)
- Layers: {8, 10, 12, 14, 16, 18, 20, 22} (8 layers)
- eff_rank computation: 2000 forward passes on WikiText prompts; compute activation
  covariance eigenvalues; compute exp(Shannon entropy of normalized spectrum)
- Fisher ratio: same covariance data, different formula (between/within class scatter)
- Fragility measurement: steer with max-Fisher direction at each layer, record
  CR (rogue-compliance rate using adversarial probe) per unit alpha
- Spearman correlations: Spearman(eff_rank, fragility), Spearman(Fisher, fragility)
- Layer selection test: use max-eff_rank layer as injection; compare CR to L16 and L12
- Seeds: 3 covariance estimation seeds x 3 steering evaluation seeds
- Wall-clock: ~4 hours on RTX 4090

### 7.2 Where it shines

Zero-shot layer selection: eff_rank is computed entirely from natural (unsteered)
activations with no behavioral probing required. If N20 holds, a single corpus pass
at each layer determines the safe injection layers for ANY behavior, eliminating
the layer-sweep calibration step.

## 8. Cross-References

- N3 (orthogonal capacity): eff_rank ≈ PR at each layer; high eff_rank = high capacity
  for safe stacking; N20 adds the fragility prediction on top of N3's capacity prediction
- N11 (curvature-aware alpha): N11 is the per-prompt curvature; N20 is the per-layer
  curvature; they are complementary spatial scales
- E2 (Fisher layer selection): FALSIFIED (rho=0.14); N20 proposes eff_rank as the
  replacement predictor with mechanistic motivation
- N15 (coset min-collateral): the fragile subspace identified by N20 is the target
  of N15's coset projection — they are designed to work together
- S-5: E2 FALSIFIED; the screening result motivating eff_rank as the replacement
- C4 screening result: INCONCLUSIVE (rho=-0.21), providing the prior for the
  held-out prediction (rho <= -0.60 with more power)
- IDEA_TABLE.md: N20 row, axes A8+A1

## 9. Committee Q&A

**Q: The screening result rho=-0.21 is very close to zero. Why should we expect
the held-out result to be rho <= -0.60? This seems like a substantial extrapolation
from an inconclusive prior.**

> The C4 screening was underpowered (n=1 seed, Gemma-270m, few layers tested).
> The rho=-0.21 is a NEGATIVE direction (correct sign) but statistically unreliable.
> The held-out protocol on Gemma-3-1B-it with 8 layers and 3 seeds is powered to
> detect rho=-0.40 (p < 0.10) and rho=-0.60 (p < 0.05). The -0.60 target is
> calibrated from the theoretical prediction (curvature-fragility link), not from
> extrapolating the -0.21 directly. If the true rho is -0.30, the experiment will
> correctly report rho in [-0.30, -0.50] and the status will be INCONCLUSIVE or WEAK.

**Q: Effective rank is computed from a corpus of WikiText prompts. If the
fragility probe uses adversarial prompts, the eff_rank and fragility are measured
on different distributions. How do you ensure comparability?**

> Eff_rank from WikiText captures the NATURAL activation geometry; fragility from
> adversarial prompts captures the WORST-CASE geometry. If the natural geometry
> predicts worst-case fragility, that is the desirable property: a behavior-free
> probe (from any natural corpus) predicts the adversarial failure mode. If the
> natural and adversarial geometries are independent, eff_rank will not predict
> fragility and the hypothesis fails — which is the clean falsification.

**Q: Fisher ratio (E2) failed to predict STEERING EFFICACY. Does this mean it
would also fail to predict FRAGILITY, which is a different target?**

> Not necessarily. Fisher ratio is a discrimination measure; it predicts how easily
> you can detect the behavior AT a layer. Fragility is a sensitivity measure; it predicts
> how much rogue behavior a perturbation at that layer causes. These are different
> enough that Fisher's failure at efficacy prediction doesn't directly inform its
> performance at fragility prediction. N20 predicts Fisher fails at BOTH (rho 0.10-0.30
> for Fisher vs fragility), and eff_rank succeeds where Fisher fails.

## 10. Verification Checklist

- [ ] Eff_rank formula implemented and validated: uniform distribution gives eff_rank = d_model
- [ ] 2000-pass covariance estimation at each of 8 layers; eigenvalue spectrum saved
- [ ] Fisher ratio computed at same layers (for comparison)
- [ ] Fragility measurement: CR per unit alpha at each layer using adversarial probe
- [ ] Spearman(eff_rank, fragility) computed with p-value and bootstrap CI
- [ ] Spearman(Fisher, fragility) computed for comparison
- [ ] Layer selection test: max-eff_rank layer vs L16 vs L12 CR comparison
- [ ] 3 covariance x 3 steering seeds = 9 cells per (layer, metric)
- [ ] C4 comparison: same layer identified as most fragile? Consistency check
- [ ] Status promoted from INCONCLUSIVE to SUPPORTED or FALSIFIED
- [ ] IDEA_TABLE.md N20 row updated

## 11. Status Journal

- 2026-05-31 — Design doc created. Status: INCONCLUSIVE (screening, n=1).
  C4 screening: eff_rank Spearman -0.21 with fragility, underpowered. Max-Fisher
  L12 = most fragile (C4), consistent with the high-curvature = high-fragility
  direction but based on only one model (Gemma-270m) and one seed. The held-out
  test on Gemma-3-1B-it with 8 layers and 3 seeds is the next required step.
  E2 FALSIFIED (Fisher ratio fails at efficacy); N20 proposes the replacement
  (eff_rank for fragility). These are different targets and different models.

---

## Addendum: Research-Scientist Critique

### Prior plausibility

MEDIUM. The curvature-fragility link is theoretically motivated but the specific
operationalization (eff_rank = fragility proxy) is a simplification. Effective rank
measures a global property of the layer's activation distribution; fragility is a
local property (how the manifold responds to perturbation at a specific point h).
The global measure may not accurately represent the local curvature relevant to
fragility.

### Mechanism scrutiny

The argument "low eff_rank = concentrated activation = high curvature = high fragility"
has two gaps: (1) concentration of the covariance does not directly imply high curvature
of the manifold (a flat low-dimensional manifold has low eff_rank but low curvature);
(2) high curvature does not necessarily mean high rogue compliance — it means high
sensitivity, which can manifest as high efficacy OR high fragility depending on the
direction of curvature.

### Confounds

1. The C4 screening found max-Fisher L12 is most fragile; this is CONSISTENT with
   N20 if L12 also has low eff_rank. But if L12 is most fragile AND has high eff_rank,
   the mechanism story is wrong. Check the eff_rank of L12 in the Gemma-270m data.
2. Fragility (CR per alpha) may be dominated by the specific adversarial probe used,
   not by the layer's geometry. Probe-specific results would not generalize.

### Does eff_rank specifically matter?

MODERATELY. If eff_rank correctly identifies safe injection layers (rho <= -0.60),
it provides a cheap, behavior-free layer-selection tool — practically valuable.
Even a weak correlation (rho = -0.40) would support preferring high-eff_rank layers
over the Fisher-ratio selection (which has rho = 0.14 for a different target).

### Literature precedent

Effective rank as a measure of representational complexity: Roy & Vetterli 2007 (EURASIP)
define effective rank for signal subspaces. In neural networks: Gauthier et al. 2019
use effective rank to measure representation quality. None apply it specifically to
steering fragility.

### Skeptical effect-size estimate

Spearman(eff_rank, fragility): -0.30 to -0.55 (vs claimed <= -0.60). Main risks:
global vs local mismatch; probe-specific fragility; layer confound (late layers tend
to have both higher eff_rank and different behavioral roles). The screening rho=-0.21
suggests the signal is real but small; -0.50 is a realistic expectation with more power.

### Minimum distinguishing experiment

Two extreme layers: min-eff_rank (expected most fragile) vs max-eff_rank (expected
least fragile). Steer at each; measure CR. If CR(min-eff_rank) > 2 * CR(max-eff_rank),
the directional claim is supported. Cost ~45 min. Check: is L12 (C4's most fragile
layer) the min-eff_rank layer? This consistency check is the free first test.

### Verdict

INCONCLUSIVE — REPLICATION REQUIRED. The C4 screening is underpowered (rho=-0.21,
n=1). The minimum experiment (2 layers, 45 min) should be the first step; it directly
tests whether eff_rank and fragility rank in the right direction. If the min-eff_rank
layer is consistently more fragile than the max-eff_rank layer (across 3 seeds), the
hypothesis is WEAKLY SUPPORTED and the full 8-layer protocol is warranted. If not,
the hypothesis may be FALSIFIED and the curvature-fragility mechanism requires revision.
