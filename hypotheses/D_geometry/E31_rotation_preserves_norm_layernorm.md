# E31 — Rotation Preserves Activation Norm and LayerNorm Statistics

> **One-line claim:** Rotational steering leaves the hidden-state L2 norm — and
> therefore the inputs to downstream LayerNorm — statistically unchanged, explaining
> the capability preservation advantage attributed to norm-preserving operations.
>
> **Source design space:** Block D — Geometry and Rotational Methods (E27–E33).
>
> **Implementation status:** PARTIALLY OBSERVED. Our rotate operator is norm-
> preserving by construction (Givens rotation matrix is orthogonal). The angular-
> displacement metric (1-cos theta) was added in C3 and confirmed predictive at
> R2=0.997. The downstream LayerNorm-variance claim is UNTESTED empirically.

---

## 1. Motivation (>= 100 words)

Rotational operations are attractive for steering because they are by construction
norm-preserving: a Givens rotation matrix R is orthogonal (R^T R = I), so ||R h||
= ||h|| for any h. This is not a hypothesis — it is a mathematical identity. The
hypothesis in E31 concerns the DOWNSTREAM consequences of this norm preservation.
Specifically: after the rotation h' = R h is computed at layer L, the next
operation in a standard transformer is usually a LayerNorm, which normalizes each
hidden state to unit variance. If ||h'|| = ||h||, the LayerNorm statistics (mean
and variance of the pre-norm inputs across the batch and sequence) are unchanged
from the unsteered case. This means the post-LN outputs are also unchanged in
their scale statistics. In contrast, additive steering h' = h + alpha*v changes
||h'||, which changes the pre-LN mean and variance, which changes the scale of
the post-LN outputs, which propagates through the MLP and attention computations
downstream — a cascade of distributional shifts that degrade capability. This is
the proposed mechanistic explanation for why norm-preserving operations are claimed
to be more capability-preserving than additive operations. Our C3 campaign found
that full-vector rotation catastrophically degrades PPL (PPL ∈ {1e10..1e18} at
theta=1-4 rad), which means ANGULAR displacement (direction change) is the actual
damage mechanism, not norm change. The question for E31 is whether, for small
theta where rotation is practical, the norm preservation advantage manifests
in measurable LayerNorm statistics. The Spherical Steering paper (mentioned in
corpus 1.3) cites effective-rank analysis to argue that spherical rotation is more
"collapse-efficient" than additive steering — E31 tests this claim directly with
LayerNorm variance measurements. The practical importance is significant: if
LayerNorm variance is preserved by rotation, this predicts MMLU capability
retention should be higher for matched-behavior rotation vs matched-behavior
addition.

---

## 2. Formal hypothesis (>= 50 words)

At matched behavior-success rate and matched PPL (both operations producing
approximately equal behavioral outputs with equal generation quality), rotational
steering at angle theta produces post-LayerNorm variance (Var[LN(h'_L+1)]) that
deviates from the unsteered baseline Var[LN(h_L+1)] by < 5% (relative), while
additive steering at the alpha that produces the same behavior produces a deviation
> 10% (relative). This LayerNorm variance preservation advantage of rotation over
addition should be observable at layer L+1 (immediately downstream of the
intervention layer) and should attenuate at deeper layers as the model's self-
normalization partially corrects the perturbation.

---

## 3. Falsifier (>= 30 words)

If the post-LayerNorm variance deviation for rotation is > 8% relative (no
statistically significant advantage over addition) at matched behavior and PPL,
the LayerNorm-preservation mechanism is rejected: rotation's capability advantage
does not operate through the proposed LN-variance pathway. If both operations show
< 5% LN-variance deviation (both are well-behaved), the claim is trivially
uninteresting and the mechanistic story is incomplete.

---

## 4. Citations (Citation Rigor format, >= 80 words)

```
Angular Steering: Liu et al. 2025 'Angular Steering: Norm-Preserving Direction
Interventions in Language Models' (arXiv:2510.26243) — proposes norm preservation
as the key advantage of rotation over addition; E31 tests the mechanistic pathway
(LN statistics) rather than accepting the claim at face value.

Selective Steering: Bai et al. 2026 'Selective Steering: Norm-Preserving 2D-Plane
Rotation for Large Language Models' (arXiv:2601.19375) — explicitly cites LayerNorm
disruption as the mechanism behind additive-steering capability degradation on small
models; E31 measures this directly.

Spherical Steering / Minimizing Collateral Damage: ICML 2026 (mentioned in corpus
section 1.3) — claims spherical rotation is more "collapse-efficient" per unit of
behavior change; uses effective-rank analysis. E31 tests the LayerNorm-variance
prediction of this claim.

Cylindrical Representation Hypothesis: Gao et al. 2026 (arXiv:2605.01844) —
CRH establishes that radial (norm) displacement governs additive coherence cost
(R2=0.81 in our C2 data) while angular displacement governs rotational coherence
cost (R2=0.997 in our C3b data). E31 completes the mechanistic picture by
measuring downstream LN consequences of these two displacement types.

Concentration of Measure (corpus Part 2, F-A): high-d Gaussians concentrate on
a thin shell at radius sqrt(d); adding alpha*v moves off the shell (radially);
rotation stays on the shell. LayerNorm normalization corrects for shell deviations
in part, but not perfectly — E31 measures the residual error.

Ba et al. 2016 'Layer Normalization' (arXiv:1607.06450) — defines LayerNorm
and its sensitivity to input scale; E31 uses this to motivate why norm changes
propagate to downstream statistics.
```

---

## 5. Mechanism

### 5.1 The LayerNorm cascade

Let the intervention be at layer L, producing h'_L (steered) vs h_L (unsteered).
The next computation is typically:
  ln_input_L+1 = h'_L + MLP_L(LN(h'_L))    (or attention + MLP in the residual)
  LN computes: LN(x) = (x - E[x]) / sqrt(Var[x] + eps) * gamma + beta

For additive steering: h'_L = h_L + delta, where delta = alpha * v.
The norm ||h'_L|| = sqrt(||h_L||^2 + 2*alpha*dot(h_L, v) + alpha^2*||v||^2).
If dot(h_L, v) != 0 (typically true since v is derived from h_L statistics),
||h'_L|| != ||h_L||. The LN sees a different input norm → different Var[h'_L] →
different LN output scale → downstream MLP and attention compute differently.

For rotation: h'_L = R h_L where R is orthogonal. ||h'_L|| = ||h_L|| exactly.
The LN input has the same norm. However, the DIRECTION of h'_L is different from
h_L, so Var[LN(h'_L)] may still differ from Var[LN(h_L)] if LN variance is
computed over the token dimension (batch + sequence) and the direction change
shifts the cross-token distribution. This is the subtle point: LN computes
statistics over (B, T) but the direction of each h_i is changed by rotation —
so even at fixed norm, the Var[LN] can change if the rotation redistributes
energy across the d dimensions.

### 5.2 The partial-correction argument

LayerNorm normalizes by computing Var[h_i, i over feature dims d]:
Var = mean_d((h_id - mean_d(h_id))^2)

A rotation R(theta) in the (h, v) plane changes two coordinates of h by ~theta.
For d=4096, this redistributes ~2/4096 of the variance budget — approximately
theta^2 / (4096) change in normalized variance per theta. At theta=0.1 rad, the
expected LN-variance change is ~ 0.01/4096 ~ 2.4e-6 (negligible). This suggests
rotation produces near-zero LN-variance deviation for theta < 0.3 rad, while
additive steering at alpha=0.1 * ||h|| produces a much larger deviation (alpha^2
added to the norm-squared).

---

## 6. Predicted Delta (pre-registered)

| Metric | Rotation (theta=0.1 rad) | Addition (matched behavior) |
|---|---|---|
| ||h'|| deviation from baseline | 0.00 (exact) | ~ 3-8% |
| Post-LN variance deviation at L+1 | < 2% (near zero) | 5-15% |
| Post-LN variance deviation at L+5 | < 1% (corrected) | 2-8% (attenuated) |
| MMLU delta (capability) | minimal (< 1pp) | moderate (2-5pp at cliff) |
| Correlation: LN-var deviation vs PPL | should confirm causality | same |

---

## 7. Protocol

### 7.1 Primary experiment

Models: Gemma-3-270M @L16, Gemma-3-1B @L18.
Operations: full-vector rotation at theta ∈ {0.05, 0.10, 0.20} rad; additive
at matched displacement (relative_add at matching fractional alpha from C3b/C9b).
Measurement: after each forward pass, extract h'_L, LN(h'_L), h_{L+1}, h_{L+5}
(five layers downstream). Compute:
  norm_dev = (||h'_L|| - ||h_L||) / ||h_L||    (should be 0 for rotation)
  ln_var_dev = |Var[LN(h'_L)] - Var[LN(h_L)]| / Var[LN(h_L)]    (at L+1, L+5)
Record behavior, PPL, norm_dev, ln_var_dev_L+1, ln_var_dev_L+5.
n=3 seeds, 50 prompts each. Wall-clock: ~2 hours per model.

### 7.2 Where it shines

E31 is the mechanistic-causality experiment for the rotation family: it tests WHY
rotation preserves capability (if it does). It is most informative as a post-hoc
check after E27-A (selective rotation) demonstrates the behavioral advantage —
E31 then explains the mechanism.

---

## 8. Cross-references

- E27 (rotation vs addition): E31 provides the mechanistic explanation for E27-A
  if confirmed
- E3 (alpha coherence cliff): the LN-variance cascade is the proposed mechanism
  behind the super-linear PPL cliff observed in S-2/S-4
- N17 (norm budget): LN-variance preservation is the downstream consequence of
  staying on the norm shell (N17/N5)
- Screening: C3/C3b — the rotate op is norm-preserving by construction (confirmed
  as off-shell = 0 in the rotate rows); the LN-variance claim is untested
- Papers: Angular Steering 2510.26243, Selective Steering 2601.19375, CRH
  2605.01844

---

## 9. Committee Q&A

**Q: The rotate op is norm-preserving by construction — isn't E31 trivially true?**

> The norm preservation (||Rh|| = ||h||) is trivially true by mathematics. The
> hypothesis is about the DOWNSTREAM CONSEQUENCES: does norm preservation imply
> LayerNorm-variance preservation? This is not trivial. LN computes Var over the
> feature dimension d, not over the norm alone. A rotation that changes the
> direction of h can change Var[LN(h)] even at fixed norm. E31 tests whether this
> secondary effect is negligible at practical theta values.

**Q: The C3b data showed rotation degrades PPL catastrophically at theta >= 0.2 rad.
Isn't E31 already falsified?**

> No. C3b shows that large angular displacement (theta = 0.5 rad, 1-cos = 0.12)
> destroys PPL — but this is the angular mechanism, not the LN-variance mechanism.
> E31 tests the hypothesis that for SMALL theta (0.05-0.1 rad) where rotation is
> practical, the LN-variance deviation is near zero, while additive steering at
> matched behavior already shows significant LN-variance deviation. The two damage
> modes are separable at small angles.

**Q: How does this relate to the MMLU capability retention claim?**

> E31 provides a chain: rotation → near-zero LN-variance deviation → near-zero
> downstream distribution shift → near-zero MMLU capability loss. This causal chain
> can be tested by correlating LN-variance deviation with MMLU delta across the
> rotation and addition conditions.

---

## 10. Verification checklist

- [ ] LN-variance measurement added to geometry.py (measure Var[LN(h_L)] before
      and after intervention, at L, L+1, L+5)
- [ ] Norm deviation measurement confirmed: ||Rh|| - ||h|| == 0 to machine precision
- [ ] Additive steering at matched behavior: LN-var deviation > 5% (relative)
- [ ] Rotational steering at matched behavior: LN-var deviation < 2% (relative)
- [ ] MMLU delta correlation with LN-var deviation measured across conditions
- [ ] Results consistent across >= 2 models
- [ ] Rows added to EXPERIMENT_LEDGER.md

---

## 11. Status journal

- 2026-05-31 — Created. PARTIALLY OBSERVED: the rotate op is mathematically norm-
  preserving (off-shell Δ||h|| = 0 in C3b, as expected). The downstream LayerNorm-
  variance claim is UNTESTED empirically. The angular-displacement law (R2=0.997)
  shows angular damage is the real mechanism for rotation; LN-variance is the
  proposed pathway from norm-preservation to capability. Priority: medium-high —
  this is the mechanistic causality test for the rotation family.

---

## Addendum: Research-Scientist Critique

*Reviewer: SciCritic-D. Critiquing the idea, not the implementation.*

### Prior plausibility

**HIGH for the norm-preservation fact; MEDIUM for the LN-cascade claim.** The
mathematical identity ||Rh|| = ||h|| is uncontroversial. The claim that this norm
preservation translates into LN-variance preservation is plausible but depends on
the specifics of how Transformer LayerNorm is computed. For d=4096, a rotation in
a 2D plane changes 2 of 4096 components by ~theta, producing a LN-variance
deviation of order theta^2/d^2 — which is approximately 0 for theta < 0.3 rad.
This asymptotic argument strongly supports the hypothesis for small-angle rotation.

### Mechanism scrutiny

The calculation in section 5.2 (LN-variance change ~ theta^2 / d) is the key
quantitative prediction. At theta=0.1 rad and d=4096, the change is ~2.4e-6
(relative), essentially zero. This means for FULL-VECTOR rotation at theta=0.1
rad, LN-variance deviation is negligible. For ADDITIVE steering at the same
matched displacement (fractional alpha ~0.1, delta ~ 0.1 * ||h||), the norm
changes by approximately 5-10% relative. The LN-variance change for addition is
proportional to the norm change squared — approximately 25-100% of the relative
norm change. So the comparison is:
  rotation: LN-var deviation ~ 0% at theta=0.1 rad
  addition: LN-var deviation ~ 1-10% at matched behavior

This is a large predicted difference and should be easy to measure. However: C3b
shows rotation at theta=0.1 rad DEGRADES PPL by +42% vs addition, even though the
LN-variance would be near zero by this argument. This means the angular direction
change IS the damage mechanism at theta=0.1 rad — not LN-variance. E31's LN-
variance story explains why full-vector rotation at large theta (1-4 rad) is worse
than addition — but at theta=0.1 rad (the practical regime), the PPL degradation
is already large from angular damage alone, before LN-variance matters.

### Confounds

1. **Angular vs. norm confound:** C3b data make it clear that angular displacement
   is the damage mechanism for rotation, not norm change. E31's hypothesis that
   norm-preservation → capability-preservation may be a moot point if angular damage
   dominates before norm damage becomes relevant.
2. **LN residual correction:** modern transformers have pre-LN architectures where
   LN is applied BEFORE the attention/MLP, followed by a residual add. The
   post-intervention h = h' + residual may recover much of the distributional shift
   at subsequent layers.
3. **Small-angle regime:** at theta=0.1 rad (where rotation is practical), the
   angular damage is already 42% PPL degradation. LN-variance preservation at this
   theta would be preserved (as computed), but the capability would already be
   degraded. The causal chain (norm preservation → LN preservation → capability
   preservation) breaks down at this scale.

### Does it specifically matter?

**YES, mechanistically.** If E31 is confirmed, it provides a clean theoretical
basis for preferring norm-preserving operations. If E31 is confirmed but rotation
still degrades PPL (C3b), the implication is that angular displacement is the
dominant mechanism and norm preservation is a secondary (unmeasured) benefit that
matters only when angular displacement is controlled.

### Skeptical effect-size re-prediction

LN-variance deviation: rotation < 1% (essentially zero at theta <= 0.1 rad);
addition ~5-10% at matched behavior. This is a large and easily measurable
difference. The claim is likely CONFIRMED but the PPL advantage does not follow
from LN preservation alone — angular damage dominates. A confirmed E31 would
refine the mechanism story without changing the practical conclusion from C3b.

### Minimum-distinguishing experiment

Add LN-variance logging to the existing C3b runs (10 lines of code). Re-run C3b
with the new measurement. If rotation LN-var < 1% and addition LN-var > 5% at
matched behavior, E31 is confirmed in 30 minutes of additional analysis.

### Verdict

**HIGH CONFIDENCE OF CONFIRMATION for the LN-variance claim (the math predicts it
clearly). The important implication is that norm preservation does NOT protect
capability at practical rotation angles — angular damage dominates. Confirmed E31
would refine the mechanistic picture of E27-A rather than validate its practical
advantage.**
