# N5 — The Norm-Budget Conservation Law

> **One-line claim:** There is a conserved edit budget B ~ q-quantile(||h||); coherence
> collapses when ||sum alpha_i v_i|| > B regardless of how it is spent (one strong
> vector or many weak), and this collapse is captured by the master curve
> log PPL = 5.40 + 2.87 * offshell with R2=0.81 pooled over 2 models/8 layers.
>
> **Primary axes:** A3 (how-much/coefficient), A12 (basis/superposition)
> **Status:** SUPPORTED (screening, n=1 — not an external claim; gate to rung 3)

---

## 1. Motivation (>= 100 words)

A fundamental question in activation steering is: why does coherence collapse?
The empirical observation (S-2 through S-8) is that log-perplexity rises steeply
beyond a threshold of the steering coefficient alpha. Three competing explanations
exist: (1) the direction v is wrong, causing semantic mismatch; (2) the magnitude
alpha*||v|| is too large, pushing the activation out of the data distribution;
(3) there is a layer-specific sensitivity threshold. The norm-budget hypothesis
unifies explanation (2) across all multi-vector, multi-layer, and multi-model
settings: the collapse is governed by the displacement of h from its natural shell,
measured as offshell = delta(||h||) / ||h||. If this is correct, it implies a
universal law — one master curve relating log-PPL to offshell displacement,
collapsing across different vectors, layers, and models. This "conservation" framing
is not exact conservation (like energy) but rather a universal scaling law: the
system tolerates off-shell displacement up to a threshold, then fails coherently.
The practical implication is immediate: cap ||delta h|| at the empirical quantile,
and coherence is preserved regardless of HOW the budget is spent. This transforms
the stacking problem from a combinatorial one (which vectors interfere?) into a
scalar one (is the aggregate displacement within budget?). Screening result S-6
provides initial support: log PPL = 5.40 + 2.87*offshell, R2=0.81 across 23 rows
from 2 models and 8 layers. This is the strongest screening result in the program.

## 2. Formal Hypothesis (>= 50 words)

Let offshell(h, delta_h) = ||h + delta_h|| - ||h|| normalized by ||h||, where
delta_h = sum_i alpha_i v_i is the aggregate steering perturbation. The claim is:

(A) log PPL = a + b * offshell holds with R2 >= 0.75 when pooled across vectors,
    layers, and models after controlling for base PPL.

(B) Capping ||delta_h|| at the empirical 80th percentile of natural ||h|| - ||h||'
    (natural activation norm variation) prevents log-PPL from exceeding 1.5 * baseline
    log-PPL for any stacking configuration tested in E17-E22.

(C) The specific form of delta_h (one strong vector vs many weak vectors) has no
    additional predictive power for log-PPL beyond the offshell scalar itself, i.e.,
    the residuals of the offshell model are not reduced by adding vector-count or
    directional features.

## 3. Falsifier (>= 30 words)

If R2 of the log-PPL ~ offshell linear model drops below 0.60 on a held-out dataset
(new models, new behaviors, new alpha ranges), or if adding vector-count as a
predictor improves R2 by more than 0.05, claim (C) is FALSIFIED. If the norm-cap at
80th percentile fails to prevent PPL > 1.5x baseline in any E17-E22 condition,
claim (B) is FALSIFIED. Full falsification requires ALL three claims to fail.

## 4. Citations (Citation Rigor >= 80 words)

```
Gao et al. 2026. 'Cylindrical Representation Hypothesis' arXiv:2605.01844 (ICML 2026).
CRH decomposes delta_h into radial (offshell) and angular (directional) components.
The R2=0.81 screening result (S-6) measures the radial component's predictive power.
CRH predicts that radial governs ADDITION coherence (confirmed: R2=0.81 for add
in C2) while angular governs ROTATION coherence (R2=0.997 for rotation in C3b).
N5 generalizes the radial finding to the multi-vector stacking regime.

Wurgaft et al. 2026. 'Manifold Steering' arXiv:2605.05115. The off-manifold
displacement is the key failure mode in manifold steering; offshell = delta(||h||)/||h||
is the spherical approximation to off-manifold displacement (exact when the manifold
is a sphere, approximate for more complex geometries). N5 is the empirical
operationalization of this theoretical failure mode.

Venkatesh & Kurapath 2026. 'On the Non-Identifiability of Steering Vectors'
arXiv:2602.06801. Demonstrates that the behavioral effect is governed by
displacement in the effective subspace; the null-space component has no behavioral
effect but DOES contribute to offshell via norm increase. N5 predicts this: null-space
components push h off-shell without improving behavior, wasting budget.

Turner et al. 2023. 'Activation Addition' arXiv:2312.06681. Observes that very large
alpha degrades generation quality; N5 gives the geometric explanation (offshell crossing
the threshold) and provides a quantitative prediction (the master curve).
```

## 5. Mechanism

The mechanism is grounded in the concentration of measure result (F-A from the
missed dimensions document): in R^d with large d, natural activations h concentrate
on a thin spherical shell of radius ~sqrt(d). The LayerNorm layers following each
residual block normalize the activation to a fixed scale, but the PRE-LayerNorm
activation is the one modified by steering. When ||h + delta_h|| deviates from ||h||
significantly, the downstream LayerNorm sees an out-of-distribution scale, and the
learned scale/bias parameters (gamma, beta) are no longer calibrated for this
magnitude — producing soft-max/softmin saturation in subsequent attention and MLP
layers, which manifests as incoherent token distributions (high perplexity).

The linear model log PPL = a + b * offshell is the first-order Taylor expansion of
this saturation effect: for small offshell, saturation is approximately linear in
the magnitude excess. The R2=0.81 from 23 screening rows suggests that this linear
approximation is adequate over the tested offshell range (0 to 0.3).

Norm cap implementation: cap ||delta_h||_2 at the 80th percentile of empirical
||h_{natural} - h_{natural}'|| for random pairs of natural activations at the same
layer. This threshold represents the typical scale of natural activation variation,
providing a principled upper bound on "safe" displacement.

## 6. Predicted Delta

| Metric | Predicted Value | Rationale |
|---|---|---|
| R2 of log-PPL ~ offshell (held-out set) | >= 0.75 | S-6 shows R2=0.81 on training set |
| R2 improvement from adding vector-count | < 0.05 | Conservation claim |
| R2 improvement from adding direction features | < 0.05 | Scalar sufficiency |
| PPL at offshell = 0 (pure rotation) | baseline PPL | Zero radial displacement |
| Max PPL reduction from norm-cap | >= 50% of uncapped exceedance | Budget enforcement |
| Cross-model law coefficient b | 2.5 - 3.5 | S-6 found b=2.87; architecture-robust |

## 7. Protocol

### 7.1 Primary experiment (held-out generalization)

- Model: Gemma-3-1B-it (held out from S-6 which used Gemma-270m and Qwen-0.5B)
- Layers: {10, 14, 18, 22} (different from S-6's layer set)
- Stacking conditions: k=1,2,3,4 vectors, each at alpha in {0.5, 1.0, 2.0, 4.0}
- Offshell measurement: ||h_steered|| - ||h_natural|| normalized by ||h_natural||
- Log-PPL: WikiText-103 next-token log-perplexity on 100 random continuations
- Regression: fit log PPL = a + b * offshell on all (k, alpha, layer) rows;
  compare R2 to S-6's 0.81; test if adding k or directional features improves R2
- Norm cap: implement at 80th percentile of natural activation norm variation;
  verify PPL stays <= 1.5 * baseline
- Seeds: 3 extraction seeds, 3 stacking orderings
- Wall-clock: ~3 hours on RTX 4090

### 7.2 Where it shines

Multi-vector stacking (E17-E22): if the scalar offshell law holds for k=1,2,3,4
simultaneously, practitioners can predict coherence collapse from a single scalar
measurement before generating text, enabling prospective norm-budget management.

## 8. Cross-References

- N17 (concentration penalty): N17 is the strong form of N5 (Spearman +0.71 from S-6);
  N5 is the linear regression version; they are complementary support for the same law
- N3 (orthogonal capacity): N3 predicts WHEN stacking fails (N > PR); N5 predicts
  HOW (via offshell); both are needed for a complete stacking theory
- N16 (CRH radius/angle): radial component = offshell; angular component = direction.
  S-7 confirmed the decomposition (R2=0.81 radial, R2=0.997 angular separately)
- E7 (relative alpha): relative_add (alpha * ||h||) directly controls offshell;
  S-9 confirmed relative_add gives cleaner cliff, consistent with N5
- E22 (norm budget and collapse): E22 is the direct experimental test of N5's
  norm-cap protocol
- N12 (capstone): the "capped at budget B" term in the unified operator is N5

## 9. Committee Q&A

**Q: The S-6 law log PPL = 5.40 + 2.87*offshell was fit on 23 rows from 2 small
models. Isn't the R2=0.81 an overfit to a small, non-random sample?**

> Yes, which is exactly why the falsifier requires held-out generalization on a
> DIFFERENT model (1B) at DIFFERENT layers. R2 dropping from 0.81 to 0.75 is
> acceptable; dropping below 0.60 would falsify the generalization claim.
> The intercept (5.40) and slope (2.87) are expected to shift across models;
> the claim is that the linear form holds, not that the specific coefficients are universal.

**Q: log PPL could correlate with offshell simply because both correlate with alpha.
This is a confound, not evidence of a causal norm-budget mechanism.**

> This is the key confound. The test is: does offshell predict log-PPL BEYOND alpha?
> Include alpha as a control variable in the regression and test whether offshell
> adds R2. If offshell R2 improvement over alpha-alone is > 0.15, the specific
> norm metric matters beyond raw steering magnitude.

**Q: The 80th percentile norm cap threshold is arbitrary. How is it chosen?**

> It is the 80th percentile of natural activation pair-wise norm differences,
> representing the upper range of NATURAL activation variation at that layer.
> The threshold is behavior-free and pre-registered. Alternative thresholds
> (70th, 90th) should be tested as ablations.

## 10. Verification Checklist

- [ ] Held-out model (Gemma-3-1B-it) not used in S-6
- [ ] Held-out layers (10, 14, 18, 22) not used in S-6
- [ ] Regression log PPL ~ offshell fit and R2 >= 0.75 verified
- [ ] Regression with vector-count added: R2 improvement < 0.05 verified
- [ ] Alpha-alone regression as control: offshell improvement > 0.15 over alpha
- [ ] Norm-cap at 80th percentile implemented; PPL cap effectiveness logged
- [ ] Cross-model coefficient b in [2.5, 3.5] reported
- [ ] Result promoted from SUPPORTED(screening) to SUPPORTED(rung-2) in IDEA_TABLE.md
      after n>=7 + Wilcoxon + bootstrap CI on held-out set

## 11. Status Journal

- 2026-05-31 — Design doc created. Status: SUPPORTED (screening, n=1). Screening
  result S-6: log PPL = 5.40 + 2.87*offshell, R2=0.81, Spearman=+0.705, Pearson=0.899
  over 23 rows (Gemma-270m + Qwen-0.5B, 8 layers, alpha 1-24). S-9 corroborates via
  relative_add (E7) showing clean cliff at alpha=0.1 * ||h||. Requires held-out
  generalization on Gemma-3-1B to advance beyond screening. The strongest single
  screening result in the program.

---

## Addendum: Research-Scientist Critique

### Prior plausibility

HIGH. The concentration-of-measure argument (F-A) is a mathematical theorem, not
an empirical claim. The LayerNorm saturation mechanism is a known failure mode.
The linear approximation (R2=0.81) is plausible as a first-order effect.
The main uncertainty is whether the slope b is universal or model-specific.

### Mechanism scrutiny

The mechanism requires: (a) LayerNorm saturation is the proximate cause of PPL
increase, and (b) saturation is approximately linear in offshell magnitude. Neither
is fully verified. Alternative: the PPL increase is due to the DIRECTION change
(moving toward off-topic probability mass), and offshell just correlates with the
magnitude of directional change. This alternative is tested by the angular-rotation
control: pure rotation (zero offshell) should have PPL ~ baseline, which is the
R2=0.997 for angular (S-7) supporting the mechanism direction.

### Confounds

1. Alpha confound: offshell and alpha are correlated by construction. The key test
   is whether offshell predicts PPL BEYOND alpha; this analysis was not done in
   the S-6 screening pass.
2. Base PPL variation: different prompt types have different base PPLs; the intercept
   a=5.40 absorbs this if it's constant, but it varies. The regression should include
   base PPL as a control.
3. Direction confound: both offshell and log-PPL might be caused by a third variable
   (distance from the natural distribution in a higher-dimensional sense). An L2
   distance-to-manifold metric might outperform the scalar offshell measure.

### Does the specific norm claim matter?

YES, more than most claims in this program. If the scalar offshell law holds, it
reduces the stacking optimization problem to a scalar budget problem, which has O(N)
instead of O(N^2) complexity. The practical value is high. Even a partially valid
law (R2=0.65) provides useful guidance.

### Literature precedent

S-6's law logPPL ~ offshell is original to this program's screening. Adjacent:
Zou et al. 2023 (representation engineering) observe that very large magnitudes
degrade coherence; they use this empirically without the geometric framing. The
cylindrical decomposition (CRH, arXiv:2605.01844) provides the formal separation
of radial and angular effects that makes N5 a principled claim rather than an
empirical observation.

### Skeptical effect-size estimate

Held-out R2: 0.65-0.80 (vs claimed >= 0.75). Main risk: the slope b=2.87 is
model-specific; it may be 1.5-4.5 across architectures, reducing pooled R2.
The scalar sufficiency claim (R2 improvement from vector-count < 0.05) is
the riskier part: early layer stacking may have directional effects not captured
by offshell.

### Minimum distinguishing experiment

Alpha confound control: fit (log PPL ~ offshell) and (log PPL ~ alpha) and
(log PPL ~ offshell + alpha) on the existing S-6 data (23 rows). Report R2 for
each. If offshell adds > 0.10 R2 over alpha alone, the specific norm metric is
vindicated beyond the trivial magnitude confound. Cost: 5 minutes of analysis
on existing data.

### Verdict

SUPPORTED-SCREENING: the strongest signal in the program. The held-out generalization
test on Gemma-3-1B is the next required step before claiming the law is universal.
The alpha confound must be tested on existing data before the 4-hour held-out run.
Recommend: (1) 5-min alpha-confound analysis on S-6 data, (2) if offshell adds
R2 > 0.10, run full held-out protocol, (3) if < 0.10, revisit the offshell
formulation.

---

## Pseudocode & Methodology

This section specializes [`../METHODOLOGY.md`](../METHODOLOGY.md). N5 is the **multi-vector norm-budget** law: it claims `log PPL = a + b·offshell` is a *universal master curve* collapsing across vectors, layers, and models. It is **TESTED at rung-3 and FALSIFIED across scale** (the within-pool fit was an artifact).

### 1. Steering-vector recipe (the aggregate displacement)

N5 is a geometry-of-the-injection hypothesis. The vector is DiffMean (METHODOLOGY §1.3); the quantity of interest is the **aggregate** off-shell displacement of the (possibly stacked) edit `δh = Σ_i alpha_i v_i`:

```python
# single- or multi-vector edit, injected via relative_add (METHODOLOGY §2):
#   h' = h + Σ_i alpha_i * ||h|| * unit(v_i)

# RADIAL off-shell of the AGGREGATE edit (geometry.offshell_displacement):
offshell = offshell_displacement(h_base, h_steer)          # |‖h'‖-‖h‖| / ‖h‖

# cumulative norm budget over stacked steps (geometry.norm_budget, the N5 budget):
budget = norm_budget(deltas, h_base)                       # Σ_steps ‖Δh_step‖ / ‖h‖
```

`geometry.norm_budget` accumulates `Σ ‖Δh_step‖ / ‖h_base‖` over a leading step axis (reduces to `‖Δh‖/‖h‖` for one vector) — this is the conserved "budget B" N5 hypothesizes.

### 2. Experiment procedure (rung-3 held-out master-curve test — `scripts/rung3_n17.py`)

N5 shares the driver with N17 (same pooled points). The N5-specific step is the **held-out law fit**:

```text
1..4.  Identical to N17: collect (model, layer, alpha, offshell, real_ppl) points on REAL WikiText-2,
       two model scales (gemma-3-270m-it, gemma-3-1b-it).
5. FIT the N5 law on the SMALL model only:   b, a = polyfit(offshell_270m, log ppl_270m, deg=1)
6. PREDICT the large model:                  log_ppl_pred_1b = a + b * offshell_1b
7. HELD-OUT R^2 = 1 - SS_res/SS_tot on the 1b points (does the 270m-fit law transfer?).
```

The decision is on **held-out** R² (predict-across-scale), not the in-pool R² — exactly to avoid the HARKing trap the §9 Q&A warns about.

### 3. Measurement & decision rule

- **Primary metric:** held-out R² of `log PPL = a + b·offshell` (fit on 270m, predict 1b).
- **Pre-registered falsifier (§3):** R² < 0.60 on a held-out dataset ⇒ claim (A)/(C) FALSIFIED.
- **ACTUAL OBSERVED RESULT (rung-3):** held-out **R² = −1.60** (worse than predicting the mean) ⇒ **VERDICT: FALSIFIED across scale.** The intercept and slope do NOT transfer between model scales; there is no single universal collapse curve. The previously reported within-pool **R² = 0.81 was an artifact** of fitting and evaluating on the same pool. Note the *rank* relationship still holds (N17 Spearman +0.585 SUPPORTED) — off-shell predicts the ORDER of incoherence, but not a scale-invariant linear LAW.

### 4. Where the code is / status

- **Driver:** `scripts/rung3_n17.py` (the `n5_law_fit_270m` + `heldout_R2_on_1b` keys in `ideas/_campaigns/RUNG3_N17.json`).
- **Probes:** `geometry.offshell_displacement`, `geometry.norm_budget` in `src/steering/geometry.py`.
- **Status:** TESTED, **FALSIFIED** (the universal-law claim). The norm-budget *cap* as a practical safety heuristic (claim B) and the scalar-sufficiency claim (C) were not separately confirmed; they remain to revisit with per-model (not pooled) coefficients.

See [`../METHODOLOGY.md`](../METHODOLOGY.md) for the shared recipe.

---

## Provenance & Tracing

Full per-hypothesis provenance (exact experiments, reproduce commands, artifact links, reasoning trace): [`PROVENANCE/N5.md`](../PROVENANCE/N5.md).

- **Experiments:** analysis campaign (computed quantities in the campaign JSON; see the provenance file).
- **Reproduce:**

```bash
PYTHONPATH=src python scripts/rung3_n17.py  # same run as N17: fit log PPL = a + b*offshell on gemma-3-270m, predict gemma-3-1b, report held-out R2
```
