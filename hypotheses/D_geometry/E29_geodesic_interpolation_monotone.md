# E29 — Geodesic / Spherical Interpolation Gives More Monotone Behavior Control

> **One-line claim:** Spherical (SLERP) interpolation between the unsteered hidden
> state h and the steered target gives more monotone and predictable behavior-vs-
> control-parameter curves than linear (LERP) alpha-scaling of the behavior vector.
>
> **Source design space:** Block D — Geometry and Rotational Methods (E27–E33).
>
> **Implementation status:** UNTESTED. No screening data as of 2026-05-31.

---

## 1. Motivation (>= 100 words)

The standard additive steering operation h' = h + alpha * v is a linear
interpolation in the Euclidean sense: as alpha increases from 0, h' moves along
a straight chord from the origin through the end-point h + v. On a curved manifold,
this chord quickly departs from the manifold's surface, because a manifold is not
flat except locally. The Manifold Steering paper (arXiv:2605.05115) demonstrates
empirically that staying on the activation manifold M_h — rather than cutting
through the off-manifold region — produces behavioral trajectories that more
faithfully track the output manifold M_y. The Curveball paper (arXiv:2603.09313)
makes the complementary point: the optimal steering path is curved. A natural
approximation to manifold-following, available without fitting the manifold
explicitly, is spherical linear interpolation (SLERP): instead of adding alpha * v
to h, we interpolate from h to a target direction on the unit hypersphere,
sweeping a geodesic arc rather than a chord. SLERP is defined as
SLERP(h, t, alpha) = sin((1-alpha)*theta)/sin(theta) * h + sin(alpha*theta)/
sin(theta) * t where theta = arccos(dot(h/||h||, t/||t||)) and t is the target
direction. SLERP preserves norm throughout the interpolation by construction.
The core claim is that this norm-preserving, arc-following interpolation should
produce a more monotone behavior-vs-alpha curve than LERP, because SLERP stays
closer to the in-distribution shell and avoids the nonlinear perplexity spikes
that occur when LERP exits the manifold. Our screening data already show that the
LERP-based coherence cliff is super-linear in alpha (S-2, S-4) — a non-monotone
collapse mode. SLERP-based control should avoid this non-linearity by keeping h
on the shell throughout the sweep.

---

## 2. Formal hypothesis (>= 50 words)

For a fixed target direction t = h + v (the additively-steered endpoint at alpha=1),
the SLERP-based behavior control curve — sweeping the interpolation parameter s
from 0 to 1 — will be more monotone than the LERP-based curve sweeping alpha from
0 to 1. Monotonicity is quantified as the fraction of adjacent pairs (s_i, s_{i+1})
for which behavior success is non-decreasing: monotone fraction >= 0.85 for SLERP
vs <= 0.70 for LERP on the same sweep. Additionally, the perplexity curve for SLERP
should remain below 2x the baseline perplexity throughout the entire control range
s ∈ [0, 1], whereas LERP is expected to exceed 2x baseline at alpha > cliff.

---

## 3. Falsifier (>= 30 words)

If the behavior-vs-control-parameter curve for SLERP shows monotone fraction
< 0.80 (non-monotone collapses), or if SLERP perplexity exceeds 2x baseline at
any point in s ∈ [0, 0.8], or if the LERP curve is equally monotone (monotone
fraction >= 0.85), this hypothesis is DISCARDED: SLERP provides no monotonicity
advantage over LERP and the geodesic intuition does not translate to steering
control. Status moves to x disproved.

---

## 4. Citations (Citation Rigor format, >= 80 words)

```
Manifold Steering: Wurgaft, Rager, Kowal et al. 2026 'Manifold Steering: Geometry-
Aware Activation Interventions' (arXiv:2605.05115) — the primary motivation for E29:
demonstrates that steering along the activation manifold M_h produces behavioral
trajectories that follow the output manifold M_y, while linear (chord) steering
cuts through off-manifold regions and produces unnatural outputs.

Curveball Steering: Raval, Song, Wu et al. 2026 'Curveball Steering: The Right
Direction To Steer Isn't Always Linear' (arXiv:2603.09313) — shows that the
optimal steering path is curved rather than linear; motivates the SLERP
approximation as a practical curved-path alternative.

Spherical Steering / Angular Steering: Liu et al. 2025 'Angular Steering: Norm-
Preserving Direction Interventions in Language Models' (arXiv:2510.26243) — the
norm-preserving rotation family that E29 extends to interpolation along a geodesic
arc rather than a fixed-angle rotation.

CAA / DiffMean baseline: Turner et al. 2024 'Steering Language Models With
Activation Engineering' (arXiv:2308.10248) — the LERP baseline; standard additive
steering with alpha sweep is the comparison condition for E29.

Selective Steering: Bai et al. 2026 (arXiv:2601.19375) — 2D-plane selective
rotation; the method closest to SLERP in spirit, restricted to one plane; E29
generalizes the norm-preserving geometric interpolation idea.

Cylindrical Representation Hypothesis: Gao et al. 2026 (arXiv:2605.01844) —
CRH establishes that angular displacement (1-cos theta) is the correct predictor
of rotational-family coherence cost; SLERP sweeps theta continuously and should
stay in the low-angular-displacement regime for moderate s.
```

---

## 5. Mechanism

### 5.1 SLERP geometry

Standard LERP: h'(alpha) = h + alpha * v. This moves h along a straight line
in R^d. The distance from the origin grows as ||h + alpha*v|| which increases
when v is not exactly orthogonal to h. Off-shell displacement: Δ||h|| = ||h +
alpha*v|| - ||h|| ≈ alpha * dot(h, v) / ||h|| for small alpha. This is the
radial component that our C2 data show predicts log-PPL at R2=0.81.

SLERP: h'(s) = SLERP(h, t, s) where t = h + v (or any target direction).
Normalizing both to the sphere: SLERP(h_hat, t_hat, s) stays on the unit sphere
for all s ∈ [0, 1]. The angular displacement at parameter s is s * theta where
theta = arccos(dot(h_hat, t_hat)) — it grows linearly with s, and SLERP is
the unique constant-speed path on the sphere from h_hat to t_hat. After the
SLERP, rescale back to the original ||h|| to restore the norm.

Predicted coherence: angular displacement = s * theta. From the C3b angular
law log PPL = 4.57 + 43.1 * (1-cos(s*theta)), SLERP should produce a smooth,
monotone PPL curve for small theta (< 0.3 rad). The non-monotone cliff of LERP
occurs because LERP simultaneously changes both norm (radial) and direction
(angular) — two damage sources. SLERP separates these: the norm is held fixed
(no radial damage), and angular damage grows monotonically with s. This is why
SLERP should produce monotone behavior and PPL curves.

### 5.2 Target selection

The target direction t can be chosen as:
(a) h + v (the LERP endpoint at alpha=1) — equivalent to finding the direction
    from h toward the behavior-shifted point and interpolating along the arc
(b) The DiffMean vector v itself, used as a target direction on the sphere

Option (a) makes SLERP a direct path from the original state to the state that
additive steering produces at full strength, but traveling along the arc of the
sphere rather than the chord. This is the correct comparison for E29.

---

## 6. Predicted Delta (pre-registered)

| Metric | SLERP prediction | LERP baseline (observed) |
|---|---|---|
| Monotone fraction (behavior) | >= 0.85 | ~ 0.60 (cliff after alpha~0.2) |
| PPL at s=0.5 / alpha=0.5 | < 1.5x baseline | 2-5x baseline (C3b/C9b data) |
| Max behavior success in sweep | ~ same as LERP peak | peaks at alpha~0.1-0.2 |
| Behavior peak s value | ~ 0.5-0.7 | alpha ~ 0.1-0.2 (fractional) |
| Angular displacement at peak | s * arccos(cos_target) | lower due to radial coupling |

---

## 7. Protocol

### 7.1 Primary experiment

Models: Gemma-3-270M and Gemma-3-1B. Layer: L16 (best from C1) and L18 (best
for 1B, C6). Direction: Rogue-Scalpel DiffMean direction (same as C3b/C9b).
LERP baseline: relative_add at fractional alpha ∈ {0.02, 0.05, 0.10, 0.20, 0.40}
(already have this from C9b for 270M; run fresh for 1B).
SLERP: implement SLERP(h, h + v_normalized * ||h||, s) at s ∈ {0.1, 0.2, 0.3,
0.4, 0.5, 0.6, 0.7, 0.8}. Record behavior, PPL, angular displacement, radial
displacement, composite. n=3 seeds; n=7 for gate.
Wall-clock: ~3 hours per model, 2 models = ~6 hours for n=3.

### 7.2 Where it shines

E29 is expected to be most informative for large-alpha regimes: for very strong
steering (target behavior = max control) where LERP collapses, SLERP should
maintain coherence by construction. The ideal test is: push behavior as high as
possible, then compare PPL. If SLERP allows PPL < 200 where LERP > 1000 at the
same behavior success, the claim is confirmed dramatically.

---

## 8. Cross-references

- E27 (rotation): SLERP is geometrically between rotation and linear interp;
  E27's selective rotation is a special case of SLERP at a fixed angle
- E31 (norm preservation): SLERP preserves norm by construction, same as rotation
- E33 (curved flow): FLAS is a multi-step curved transport; SLERP is a two-point
  single-step approximation
- Corpus: N13 (geodesic > chord), N16 (CRH cylindrical decomposition)
- Papers: Manifold Steering 2605.05115, Curveball 2603.09313, CRH 2605.01844

---

## 9. Committee Q&A

**Q: Is SLERP actually "geodesic" on the activation manifold?**

> No. SLERP is geodesic on the hypersphere S^{d-1}, not on the general activation
> manifold M_h. The two coincide only if the activation manifold IS the hypersphere,
> which is approximately true in the high-d concentration-of-measure sense (most
> activations lie near a sphere of radius sqrt(d)). SLERP is therefore an
> approximation to geodesic steering, not the exact geodesic. The honest statement
> of E29 is "SLERP approximates manifold-following better than LERP."

**Q: Why not use actual manifold-following (Manifold Steering 2605.05115)?**

> Manifold steering requires fitting the manifold M_h explicitly, which is expensive
> (requires a separate manifold model). SLERP is a zero-cost approximation using
> only the geometry of the hypersphere. E29 tests whether this zero-cost
> approximation recovers the benefit. If SLERP works, there is no need for the
> expensive manifold fit for this purpose.

**Q: How does this differ from just norm-normalizing and using relative_add?**

> Relative_add (C9b) normalizes the direction v but still moves h along a chord
> (radially and angularly). SLERP holds the norm constant AND moves along the arc.
> For small angles, the two are nearly identical; for large s (strong steering),
> SLERP's norm preservation becomes the key advantage.

---

## 10. Verification checklist

- [ ] SLERP operator implemented in geometry.py and unit-tested (SLERP(h,t,0) = h,
      SLERP(h,t,1) = t_hat * ||h||, ||SLERP(h,t,s)|| = ||h|| for all s)
- [ ] LERP baseline reproduced from C9b on 270M @L16
- [ ] SLERP sweep at s ∈ {0.1..0.8} on 270M @L16, n>=3
- [ ] SLERP sweep on 1B @L18, n>=3
- [ ] Monotone fraction computed for both curves on each model
- [ ] PPL comparison at matched behavior success point
- [ ] Angular displacement metric logged throughout (should grow monotone for SLERP)
- [ ] Rows added to EXPERIMENT_LEDGER.md

---

## 11. Status journal

- 2026-05-31 — Created. UNTESTED. The motivation is grounded in C3b/C9b data
  showing the LERP non-monotone cliff, and in the Manifold Steering / Curveball
  literature. SLERP implementation is a 10-line addition to geometry.py. Priority:
  medium, after E27-A (selective rotation) which is more directly falsifiable.

---

## Addendum: Research-Scientist Critique

*Reviewer: SciCritic-D. Critiquing the idea, not the implementation.*

### Prior plausibility

**MEDIUM.** The geometric intuition is correct: SLERP is norm-preserving and
arc-following, and the angular-displacement law from C3b (log PPL = 4.57 + 43.1 *
(1-cos theta), R2=0.997) predicts that monotone angular growth → monotone PPL
growth for small theta. The question is whether the behavior-success curve is also
monotone when the perplexity curve is. These are different: behavior tracks the
direction, perplexity tracks the deviation from the data manifold. SLERP guarantees
monotone angular displacement but not monotone behavior success.

### Mechanism scrutiny

The angular-displacement law is the strongest mechanistic anchor for E29. If
log PPL is a monotone function of angular displacement, and SLERP produces monotone
angular displacement (s * theta grows monotonically in s), then PPL is monotone in
s under SLERP. The concern is the behavior success curve: it is not obviously
monotone in angular displacement. Our data show behavior peaks at alpha~0.2
(fractional) for LERP and then declines — this decline is not only due to PPL
collapse but also to over-steering (the behavior direction is over-shot). SLERP
will exhibit the same overshoot at s near 1 (when h is fully rotated to t). The
monotonicity claim must be bounded to s < 0.8 to be defensible.

### Confounds

1. **Target direction confound:** the target t = h + v is the LERP endpoint at
   alpha=1, which is already in a region where LERP has degraded. SLERP to this
   target follows the arc to the same endpoint — it will eventually arrive at the
   same degraded state. SLERP delays the damage but does not eliminate it.
2. **Behavior-probe confound:** the projection-based behavior proxy may be insensitive
   to small angular changes; real generation-based evaluation may show non-monotone
   behavior earlier than the proxy suggests.
3. **Baseline-curve confound:** the LERP curve in C9b uses fractional alpha (= fraction
   of ||h||), which is not directly comparable to the SLERP s parameter. The
   comparison requires careful matching of the effective angular displacement.

### Does it specifically matter?

**MEDIUM.** If SLERP produces more monotone control curves, it simplifies
hyperparameter selection: practitioners can sweep s in [0,1] and get predictable
behavior. With LERP, the cliff makes alpha selection brittle. This is a usability
improvement more than a scientific breakthrough.

### Literature precedent

SLERP for latent space interpolation is well-established in generative models
(VAEs, diffusion models) where interpolating on the sphere produces smoother
latent-space traversals than LERP. The direct application to steering vectors is
new but the geometric argument is identical. No direct steering-SLERP paper exists
as of May 2026, making this a genuine empirical gap.

### Skeptical effect-size re-prediction

Monotone fraction improvement: expect SLERP monotone fraction ~ 0.80-0.90 vs LERP
~ 0.65-0.75 (moderate improvement). The strongest effect will be at large s
(strong steering) where LERP has already collapsed. At moderate s (< 0.5), the
two methods are nearly identical.

### Minimum-distinguishing experiment

Compare LERP and SLERP at matched angular displacement on Gemma-3-270M @L16,
n=3, using real PPL (WikiText-103 sample). If SLERP PPL < LERP PPL by > 20% at
the same angular displacement, the claim is confirmed. Cost: ~2 hours on a 4090.

### Verdict

**PLAUSIBLE AND TESTABLE. Grounded in the C3b angular-displacement law. Expected
to show moderate monotonicity improvement at large s. The zero-cost SLERP
implementation makes this a high-value / low-cost experiment.**

---

## Pseudocode & Methodology

This section specializes [`../METHODOLOGY.md`](../METHODOLOGY.md) to SLERP (geodesic
arc) vs LERP (linear add) control curves. Status: `UNTESTED` — SLERP is a ~10-line
addition to `geometry.py`. This is a GEOMETRY/OPERATION hypothesis: the same DiffMean
direction, swept along an arc instead of a chord.

### 1. Steering-vector recipe (DiffMean target; SLERP vs the add chord)

```python
v = extract.build_vector_bank(model, tok, load_concept(trait), L)[L]["diffmean"]  # METHODOLOGY §1.3
t = h + v                                          # the LERP endpoint at alpha=1 (target direction)
# LERP baseline = METHODOLOGY §2 relative_add: h' = h + alpha*||h||*unit(v)  (chord; changes norm+dir)
# SLERP (§5.1) — norm-preserving geodesic on the hypersphere, parameter s in [0,1]:
def slerp(h, t, s):
    h_hat, t_hat = unit(h), unit(t)
    theta = arccos(dot(h_hat, t_hat))
    arc = (sin((1-s)*theta)/sin(theta))*h_hat + (sin(s*theta)/sin(theta))*t_hat
    return ||h|| * unit(arc)                        # rescale to original norm -> ||h'||=||h|| for all s
```

### 2. Experiment procedure

```text
for model in {270M @L16, 1B @L18}:
  LERP: relative_add at fractional alpha in {0.02,0.05,0.10,0.20,0.40}     # hooks.apply_operation
  SLERP: s in {0.1,...,0.8}; h' = slerp(h, h + v*||h||, s)
  for each point, seed in 1..3:
    measure behavior, PPL,
            geometry.offshell_displacement   # ~0 for SLERP (norm held), radial>0 for LERP
            geometry.angular_displacement    # 1-cos; grows monotone s*theta under SLERP
monotone_fraction = frac of adjacent (s_i,s_{i+1}) with non-decreasing behavior
```

### 3. Measurement & decision rule

- PRIMARY metric: monotone fraction of the behavior-vs-control curve (+ PPL ceiling).
- Pre-registered FALSIFIER (§3): SLERP monotone fraction `< 0.80`, OR SLERP PPL `> 2x`
  baseline anywhere in `s∈[0,0.8]`, OR LERP equally monotone (`>= 0.85`) ⇒ DISCARDED.
- Predicted (§6): SLERP monotone fraction `>= 0.85` vs LERP `~0.60`; PPL at s=0.5 `< 1.5x` baseline.

### 4. Where the code is / status

LERP (`relative_add`) and the geometry probes exist; the C9b LERP baseline is already
logged for 270M. MISSING: the **SLERP operator** in `geometry.py` (unit-test:
slerp(h,t,0)=h, slerp(h,t,1)=t̂·‖h‖, ‖slerp‖=‖h‖ ∀s) plus its injection wiring.
`UNTESTED`.

See [`../METHODOLOGY.md`](../METHODOLOGY.md) for the shared recipe.

---

## Provenance & Tracing

No experiments run yet — see this design doc's protocol (§7) for what would be run. Once a campaign logs rows for this hypothesis, re-run `scripts/build_provenance.py` to generate `hypotheses/PROVENANCE/E29.md`.
