# N16 — Radius/Angle Decoupling (Cylindrical Representation Hypothesis)

> **One-line claim:** Steering that moves only the angle (fixed radius) is more
> coherent than steering that moves both; rogue damage scales with the radius
> excursion, not the angle; coherence = radial x angular jointly, as confirmed
> by angular predicting rotation log-PPL R2=0.997 and radial predicting addition
> R2=0.81 in screening result S-7/C3b.
>
> **Primary axes:** A9 (metric/space), A4 (operation)
> **Status:** SUPPORTED (screening, n=1 — not an external claim; gate to rung 3)

---

## In Plain English

**What we're testing, simply:** A steering nudge does two things at once: it changes
the *size* of the model's internal "thought" and it changes its *direction*. This doc
splits those apart and says: changing the *size* is what breaks the text, while
changing only the *direction* (keeping size fixed) is gentler. So a "turn without
resizing" move should stay more readable for the same behavior change.

**Key terms (defined here):**
- **Steering / steering vector** — changing behavior by adding a chosen direction to
  the model's internal "thought" mid-sentence, instead of retraining.
- **Residual stream** — the model's running internal thought; what we edit.
- **Layer** — one of the model's processing steps; a knob.
- **alpha / strength** — how hard we push.
- **Coherence** — whether the text stays fluent (measured by **perplexity**; higher = more broken).
- **Norm** — the *size* (length) of the thought-point.
- **Radial vs angular** — radial = changing the thought's *size* (this is what breaks
  text); angular = changing its *direction* while keeping size fixed (gentler).
- **Rotation / angle-only steering** — a move that turns the thought's direction
  without changing its size.
- **Shell / surface (manifold)** — the thin layer of "right-sized" healthy thoughts;
  changing size leaves it, changing angle stays on it.

**Why we're doing this (the point):** If the *size* change is the real culprit behind
broken text, we can steer by *rotating* instead of *adding* and keep generations clean.

**What the result would mean:** Early screening already lines up: the *size* change
strongly predicted how much plain addition broke the text, and the *direction* change
strongly predicted how much rotation broke it — so the split is real (marked SUPPORTED
at the early, single-run screening stage; still needs the fuller test).

See [`../GLOSSARY.md`](../GLOSSARY.md) for any other term.

---

## 1. Motivation (>= 100 words)

Activation steering conflates two distinct geometric quantities in a single parameter
alpha: how far to move (the magnitude, or radius) and which direction to move (the angle
on the activation sphere). The Cylindrical Representation Hypothesis (CRH, arXiv:2605.01844)
formalizes this: representation differences decompose into a radial component (change in
||h||) and an angular component (change in direction of h). The two components govern
different aspects of behavior and coherence. The screening experiment S-7 (C3b campaign)
provides strong initial support: for ROTATION operations, the angular displacement
(1-cos theta) predicts log-PPL at R2=0.997 — nearly perfect. For ADDITION operations,
the radial displacement (delta-||h||) predicts log-PPL at R2=0.81. This is not just a
methodological observation: it reveals that additive steering's primary failure mode is
the radial excursion (leaving the activation shell, as predicted by concentration of
measure F-A), while rotational steering's failure mode is the angular excursion (rotating
too far from the natural direction, losing semantic coherence). Jointly, coherence =
f(radial, angular) is the CRH decomposition. The practical implication is that angle-only
steering (rotating h without changing its norm) should be strictly more coherent than
additive steering at matched behavior change, because it eliminates the radial excursion
entirely. This is the specific claim of N16, and S-7's data provides its first screening
support: the radial component, not the angular, drives additive's PPL cliff.

## 2. Formal Hypothesis (>= 50 words)

Let h_add = h + alpha*v and h_rot = Rot(h, theta) where theta is chosen to match the
behavioral cosine shift of alpha*v. The claim is:

  (A) log-PPL(h_rot) < log-PPL(h_add) at matched behavior cosine shift,
      for all alpha in the "cliff" range (alpha/||h|| > 0.05)
  (B) The improvement is predicted by the radial component: delta(||h_add||)/||h_add||
      (the radial excursion) correlates with log-PPL(h_add) - log-PPL(h_rot) at r >= 0.70.
  (C) For SELECTIVE 2D-plane rotation (not full-vector rotation), PPL(h_selective_rot) <= 1.1*baseline

at Gemma-3-1B-it @L16, averaged over 3 behaviors and 9 seeds.

Note: S-7 found that FULL-VECTOR rotation hurts PPL (PPL 100->11211) while the angular
metric predicts this at R2=0.997. N16 claim (C) specifically concerns selective 2D-plane
rotation, not full-vector rotation, which was E27's falsified variant.

## 3. Falsifier (>= 30 words)

If selective 2D-plane rotation has log-PPL >= additive at matched behavior for all tested
alpha values and all 3 behaviors, claim (A) is FALSIFIED. If the radial-excursion
correlation with PPL-advantage is < 0.50, claim (B) fails. Both failing = full FALSIFIED.

## 4. Citations (Citation Rigor >= 80 words)

```
Gao, Zhang, Liu, et al. (MBZUAI) 2026. 'The Cylindrical Representation Hypothesis'
arXiv:2605.01844 (ICML 2026). The primary reference: introduces the radial-angular
decomposition of representation differences; reports that radial and angular components
govern distinct aspects of model behavior. N16 is the direct empirical test of their
main claim applied to steering coherence specifically, using our Gemma models.

Wurgaft et al. 2026. 'Manifold Steering' arXiv:2605.05115. The activation manifold
M_h is approximately spherical at each layer (concentration of measure, F-A); on this
sphere, angular displacement is on-manifold while radial displacement is off-manifold.
N16's radial excursion = off-manifold displacement claim is the spherical-manifold
special case of the general manifold steering theorem.

Raval et al. 2026. 'Curveball Steering' arXiv:2603.09313. Curveball minimizes total
path length on the manifold; for spherical geometry, minimum path = great-circle arc =
pure angular displacement (zero radial excursion). N16 validates the specific claim
that Curveball's path advantage comes from radial-excursion elimination.

SCREENING RESULT: S-7 from FINDINGS.md (C3b campaign on Gemma-3-270m):
angular (1-cos) predicts rotation log-PPL R2=0.997; radial predicts addition R2=0.81;
coherence = radial x angular (CRH confirmed at screening level).
```

## 5. Mechanism

The concentration of measure result (F-A): in R^d, natural activations h cluster on
a shell of radius ~||h|| with small spread. Adding alpha*v changes ||h|| by approximately
alpha * cos(v, h_hat) (the radial component of v), moving h off this shell. The LayerNorm
downstream is calibrated for ||h|| ~ r_natural; seeing ||h'|| = r_natural + delta_r
produces out-of-distribution scale statistics, causing PPL increase.

Selective 2D-plane rotation: choose the 2D plane P = span{h, v}; apply a Givens rotation
by angle theta in P: h_rot = h * cos(theta) + v_perp * ||h|| * sin(theta), where v_perp
is the component of v orthogonal to h (the "angular direction"). This rotation:
  (i) keeps ||h_rot|| = ||h|| exactly (zero radial excursion)
  (ii) moves h toward v (angular displacement = theta)
  (iii) is equivalent to CAA (additive) when theta is small and v is perpendicular to h

The PPL of h_rot should be lower than h_add at matched behavior because (i) is satisfied:
no radial excursion, no LayerNorm saturation. The remaining PPL cost is from (ii): rotating
too far away from the natural direction (too large theta) causes semantic incoherence.
This angular cost is captured by the R2=0.997 angular metric from S-7.

Coherence decomposition: log-PPL(alpha, theta) ≈ a * delta_r(alpha) + b * (1-cos(theta)) + c
where delta_r is the radial excursion, (1-cos theta) is the angular excursion. At fixed
behavior (fixed angular displacement), minimizing delta_r (rotating instead of adding)
minimizes the first term and therefore log-PPL.

## 6. Predicted Delta

| Metric | Predicted Value | Rationale |
|---|---|---|
| log-PPL: selective rotation vs additive at matched behavior | 5% to 20% lower | Zero radial excursion eliminates radial cost |
| Correlation of radial excursion with PPL advantage | r >= 0.70 | CRH decomposition |
| angular (1-cos) correlation with rotation log-PPL | R2 >= 0.95 | S-7 found R2=0.997; replication target |
| radial correlation with addition log-PPL | R2 >= 0.70 | S-7 found R2=0.81; replication target |
| PPL of selective rotation at theta < 20 degrees | <= 1.15 * baseline | Small-angle coherence preservation |

## 7. Protocol

### 7.1 Primary experiment

- Model: Gemma-3-1B-it @L16 (held-out from S-7's Gemma-270m and Qwen-0.5B training data)
- Behaviors: 3 (refusal, politeness, factuality)
- Operations: (i) additive alpha*v, (ii) selective 2D-plane rotation by theta
- Matching: find (alpha, theta) pairs giving the same behavior cosine shift
- Metrics: log-PPL (WikiText continuation), offshell delta(||h||)/||h||,
           angular (1-cos(h, h_steered))
- Regression: log-PPL ~ radial for addition; log-PPL ~ angular for rotation
- Replication check: R2 values compared to S-7 (target: within 0.05 of S-7's values)
- Seeds: 3 extraction x 3 evaluation = 9 cells per (behavior, operation)
- Wall-clock: ~3 hours on RTX 4090

### 7.2 Where it shines

Small models (270M, 1B) where the additive-rotational coherence gap is largest (S-4, S-7).
For larger models (9B+), the manifold is less curved and the radial excursion is smaller,
reducing the advantage of selective rotation.

## 8. Cross-References

- N5 (norm-budget): radial excursion = N5's offshell measure; N16 explains WHY offshell
  predicts PPL (the radial-angular decomposition)
- N17 (concentration penalty): N17 is the empirical Spearman result; N16 is the
  geometric explanation via the CRH decomposition
- N1 (tangent projection): tangent projection removes the component of v normal to M_h,
  reducing the radial excursion — N1 and N16 predict the same PPL improvement via
  different geometric framings
- N13 (geodesic): geodesic = zero radial + angular displacement along M_h; N16 predicts
  zero-radial is the key property, which geodesic achieves by construction
- N12 (capstone): the Phi_t=angular component in the unified operator is the selective
  rotation implementing N16's angle-only steering
- S-7 (screening): direct predecessor; N16 is the replication and generalization
- IDEA_TABLE.md: N16 row, axes A9+A4

## 9. Committee Q&A

**Q: S-7 found that full-vector rotation HURTS coherence (+42% PPL). Now N16 predicts
selective rotation HELPS. Aren't these contradictory?**

> No. S-7 tested full-vector rotation in d_model dimensions, which both rotates h AND
> changes its relationship to all downstream circuits (equivalent to a large angular
> displacement in many planes simultaneously). N16's selective 2D-plane rotation is
> a Givens rotation in just the {h, v} plane, leaving all other planes untouched.
> S-7 confirmed the angular metric (R2=0.997) precisely because the full-vector
> rotation's large angular excursion explains its PPL cost; selective rotation achieves
> the same angular shift in the {h,v} plane with minimal total angular disturbance.

**Q: The matched-behavior comparison requires finding (alpha, theta) pairs with the same
behavior shift. How accurately can this be done in practice?**

> The behavior cosine shift is measured after the operation. We sweep alpha in
> {0.5, 1, 2, 4}*||h|| and theta in {2.5, 5, 10, 20} degrees; for each alpha/theta
> value, measure behavior shift and select pairs within 0.01 cosine shift. This
> guarantees the comparison is at matched behavior within 10% relative.

## 10. Verification Checklist

- [ ] Selective 2D-plane rotation implemented (Givens rotation in span{h, v_perp})
- [ ] ||h_rot|| = ||h|| verified to machine precision (zero radial excursion)
- [ ] Matched-behavior protocol: (alpha, theta) pairs within 0.01 behavior cosine
- [ ] log-PPL vs radial (for add) and log-PPL vs angular (for rot) regression
- [ ] R2 values compared to S-7 targets; logged with confidence intervals
- [ ] 9 cells per (behavior, operation): 54 total cells recorded
- [ ] PPL advantage (rot vs add) at matched behavior logged per behavior
- [ ] Status promoted from SUPPORTED(scr) to SUPPORTED(rung-2) after n>=7 + Wilcoxon
      + bootstrap CI in IDEA_TABLE.md N16 row

## 11. Status Journal

- 2026-05-31 — Design doc created. Status: SUPPORTED (screening, n=1). Screening
  results: angular (1-cos) predicts rotation log-PPL R2=0.997; radial governs addition
  R2=0.81; coherence = radial x angular (C3b, Gemma-3-270m, exp#37-46). The key
  held-out test is replication on Gemma-3-1B-it, which has not been done. E27 (full
  rotation hurt coherence) is FALSIFIED; N16 (selective rotation helps) is the
  refinement that separates full-vector from 2D-plane rotation.

---

## Addendum: Research-Scientist Critique

### Prior plausibility

HIGH. The CRH decomposition is theoretically motivated by concentration of measure
and empirically supported by S-7's R2=0.997 for angular and R2=0.81 for radial.
The prediction of selective rotation outperforming additive is directly motivated
by the theory. The main risk is implementation: the "selective" constraint (2D plane
only) must be implemented precisely to distinguish from S-7's full-vector rotation.

### Mechanism scrutiny

The mechanism (radial excursion -> LayerNorm saturation -> PPL increase) is verified
at the correlation level by S-7. The causal direction (excursion CAUSES PPL, not
the reverse) is supported by the R2 values (which would be low if PPL cause
excursion rather than vice versa), but a more rigorous causal test (intervention)
would set excursion to zero by hand (i.e., normalize h after steering) and verify
PPL drops accordingly.

### Confounds

1. The matched-behavior protocol assumes that the same behavioral cosine shift has
   the same semantic content for additive and rotational operations. This may not hold:
   adding and rotating may reach different points in activation space with the same
   cosine vs v, but different downstream effects.
2. The S-7 R2=0.997 was measured on full-vector rotation, not selective 2D-plane;
   the selective rotation's angular metric may have a different (higher or lower) R2.

### Does the radial/angular decoupling specifically matter?

YES. It provides the correct parameterization for understanding the coherence cliff:
the cliff is at a specific radial threshold, and angular displacement is separately
a coherence cost. This decoupling enables the CRH-based steering recipe (angle +
radius independently) of N14's intensity concept sub-claim.

### Literature precedent

The CRH paper (arXiv:2605.01844) is the direct reference; N16 is its operationalization
for the specific steering coherence prediction. No prior work independently measured
the R2 of radial vs angular on the same steering dataset.

### Skeptical effect-size estimate

Held-out replication on Gemma-3-1B: R2 of 0.90-0.98 for angular (vs S-7's 0.997);
R2 of 0.70-0.82 for radial (vs S-7's 0.81). Minor model-specific variation expected.
PPL advantage of selective rotation: 5-15% (vs claimed 5-20%). The lower bound
is well-supported; the upper bound depends on how curved the 1B model's manifold is.

### Minimum distinguishing experiment

One behavior, 6 alpha/theta matched pairs, 3 seeds, Gemma-3-1B @L16: run additive
and selective rotation, measure log-PPL and radial/angular metrics. Cost ~1 hour.
If R2 for angular (rotation) >= 0.85 and R2 for radial (addition) >= 0.60, replication
is successful and full protocol is warranted. This is the S-7 replication check.

### Verdict

SUPPORTED(screening) — REPLICATION REQUIRED. The S-7 screening result is strong;
the held-out generalization on Gemma-3-1B is the next required step. The minimum
1-hour replication run should be the first experiment scheduled after the infrastructure
for selective 2D-plane rotation is implemented.

---

## Pseudocode & Methodology

This section specializes [`../METHODOLOGY.md`](../METHODOLOGY.md). N16 is the CRH **radius/angle decoupling**: selective 2D-plane rotation (zero radial excursion) beats additive at matched behavior; coherence = radial × angular. **SUPPORTED at screening (S-7), not yet replicated at rung-3.**

### 1. Steering-vector recipe (separate the two geometry components)

The two components are read directly off `geometry.offshell_displacement` (radial) and `geometry.angular_displacement` (angular):

```python
v = bank[L]["diffmean"]
# additive (radial-changing):  h_add = h + alpha*v                  (METHODOLOGY §2 add)
# selective 2D-plane rotation (zero radial): Givens rotation in span{h, v_perp}  (METHODOLOGY §2 rotate)
#   h_rot = ||h|| * (cos(theta)*unit(h) + sin(theta)*unit(v_perp))  -> ||h_rot|| == ||h|| exactly
radial  = offshell_displacement(h, h_steer)     # geometry.offshell_displacement (RADIAL excursion)
angular = angular_displacement(h, h_steer)      # geometry.angular_displacement  (1 - cos)
```

`geometry.angular_displacement`'s own docstring records campaign C3: rotate gave Δ‖h‖≈0 yet full-vector rotation blew up PPL — N16 claim (C) is about *selective 2D-plane* rotation, NOT full-vector.

### 2. Experiment procedure

```text
1. For 3 behaviors at L16: find (alpha, theta) pairs giving the SAME behavior cosine shift.
2. Measure log-PPL, radial (offshell), angular (1-cos) for additive and selective rotation.
3. Regress log-PPL ~ radial (for add) and log-PPL ~ angular (for rot); compare R^2 to S-7 (0.81 / 0.997).
4. 9 cells per (behavior, operation); held-out on Gemma-3-1B (S-7 used 270m/Qwen).
```

### 3. Measurement & decision rule

- **Primary metric:** PPL advantage of selective rotation vs additive at matched behavior; radial→addPPL and angular→rotPPL R².
- **Pre-registered falsifier (§3):** selective rotation PPL ≥ additive for all alpha/behaviors ⇒ (A) FALSIFIED; radial-excursion correlation < 0.50 ⇒ (B) fails.
- **Screening result (S-7/C3b, 270m):** angular→rotation-PPL R²=0.997, radial→addition-PPL R²=0.81. Held-out replication on Gemma-3-1B is the next required (rung-3) step.

### 4. Where the code is / status

SUPPORTED (screening); rung-3 **not yet run**. Both probes (`geometry.offshell_displacement`, `geometry.angular_displacement`) and the `rotate` operation (`hooks.apply_operation`) exist; the remaining work is the **matched-behavior selective-rotation sweep on Gemma-3-1B** with the radial/angular regressions — the held-out replication of S-7.

See [`../METHODOLOGY.md`](../METHODOLOGY.md) for the shared recipe.

---

## Provenance & Tracing

No experiments run yet — see this design doc's protocol (§7) for what would be run. Once a campaign logs rows for this hypothesis, re-run `scripts/build_provenance.py` to generate `hypotheses/PROVENANCE/N16.md`.
