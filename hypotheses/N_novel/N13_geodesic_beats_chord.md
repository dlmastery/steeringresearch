# N13 — Geodesic > Chord: Manifold Steering Reduces Off-Manifold Damage

> **One-line claim:** For matched behavior change, manifold/geodesic steering yields
> strictly lower off-manifold displacement and lower rogue-compliance (harmful content
> generation under adversarial probe) than linear additive steering.
>
> **Primary axes:** A8 (geometry/path), A4 (operation)
> **Status:** UNTESTED

---

## In Plain English

**What we're testing, simply:** To move the model's internal "thought" from here to
there, you can cut a straight line *through* empty space (a shortcut), or follow the
curved healthy-thought surface the long way around. The straight shortcut passes
through "off-world" regions where the model behaves strangely — including unsafe ways.
This doc says following the curve (even though it's longer) stays safer and breaks the
text less, for the same change in behavior.

**Key terms (defined here):**
- **Steering / steering vector** — changing behavior by adding a chosen direction to
  the model's internal "thought" mid-sentence, instead of retraining.
- **Residual stream** — the model's running internal thought; what we edit.
- **Surface / manifold** — the natural region where healthy thoughts live.
- **Chord** — the straight-line shortcut that cuts *through* off-surface space (plain
  additive steering).
- **Geodesic** — the shortest path that stays *on* the curved surface (manifold-following
  steering).
- **Off-manifold / off-shell displacement** — how far the nudge knocks the thought off
  the healthy surface.
- **Rogue compliance** — the model producing harmful content when probed; a safety leak
  we want to avoid.

**Why we're doing this (the point):** If the shortcut is what drags the model into
unsafe, off-surface regions, then following the curve should cut the safety risk and
keep text cleaner — at no cost in behavior.

**What the result would mean:** A win means the on-surface path leaks less harmful
content and stays off-surface less. A loss means the curved path is no safer than the
shortcut (or our curve approximation was too crude to tell).

See [`../GLOSSARY.md`](../GLOSSARY.md) for any other term.

---

## 1. Motivation (>= 100 words)

The distinction between a geodesic (the shortest path on a curved surface) and a chord
(the straight line connecting two points, passing through the interior) is the central
geometric insight of the 2026 manifold steering wave. The Rogue Scalpel paper showed
that 817 of 1000 random SAE feature directions can produce harmful outputs — not
because any specific harmful concept is encoded there, but because off-manifold
activation regions have undifferentiated, potentially dangerous properties. The chord
(linear additive steering) cuts through these off-manifold regions; the geodesic
(manifold-following steering) stays within the data distribution. This is not merely
aesthetic: the Manifold Steering paper (2605.05115) demonstrates that geodesic steering
produces behavioral trajectories that follow the behavior manifold M_y, while chord
steering produces behavioral jumps that may land in unintended regions of M_y. The
key falsifiable prediction is that geodesic steering reduces the ROGUE COMPLIANCE
RATE — the fraction of steered generations that satisfy a harmful adversarial probe
— because it avoids the off-manifold regions where rogue behavior lives. Simultaneously,
geodesic steering should have lower off-manifold displacement (delta-||h||) at matched
behavior change, since by construction it moves along M_h rather than cutting through
its exterior.

## 2. Formal Hypothesis (>= 50 words)

Let CR(method, alpha) be the rogue-compliance rate (fraction of steered outputs
satisfying an adversarial probe) at behavior coefficient alpha. Let offshell(method,
alpha) = delta(||h||)/||h|| be the off-manifold displacement. The claim is:

  At matched behavior-cosine shift (delta_cos = 0.10):
  (A) offshell(geodesic) < 0.5 * offshell(additive)
  (B) CR(geodesic) < 0.7 * CR(additive)

measured on Gemma-3-1B-it, using a behavior direction extracted by DiffMean and
a geodesic path approximated by the Curveball / Manifold Steering discretization.

## 3. Falsifier (>= 30 words)

If offshell(geodesic) >= 0.8 * offshell(additive) at matched behavior, claim (A) is
FALSIFIED. If CR(geodesic) >= 0.9 * CR(additive), claim (B) fails. If the geodesic
approximation (Curveball discretization) does not measurably reduce off-manifold
displacement vs additive, the approximation is too coarse to test the true geodesic claim.

## 4. Citations (Citation Rigor >= 80 words)

```
Wurgaft et al. 2026. 'Manifold Steering' arXiv:2605.05115 (ICML 2026). The primary
reference: establishes the M_h activation manifold and M_y behavior manifold;
demonstrates that geodesic steering on M_h tracks M_y trajectories; shows the chord
produces off-manifold excursions. N13 is the empirical test of their main claim
applied specifically to rogue-compliance as the outcome metric.

Raval et al. 2026. 'Curveball Steering' arXiv:2603.09313. Implements the curved
path; the Curveball discretization is used as the geodesic approximation in N13's
protocol. Their paper reports coherence improvements but does not measure rogue
compliance directly.

Schwinn et al. 2025. 'Rogue Scalpel: The Risks of Broad Feature-Steering Attack
Surfaces' arXiv:2509.22067. Establishes the rogue-compliance rate as the primary
off-manifold damage metric: 817/1000 random SAE feature directions produce harmful
output. N13 uses CR as the primary safety-relevant outcome for geodesic vs chord.

Gao et al. 2026. 'CRH' arXiv:2605.01844. CRH's finding that radial displacement
governs additive coherence (R2=0.81 in S-6) means offshell is the right metric;
geodesic steering's lower offshell should correspond to lower PPL and lower CR.
```

## 5. Mechanism

The chord (additive vector) from h to h + alpha*v passes through the region
R^d \ M_h (exterior of the activation manifold). In this exterior, LayerNorm
statistics are out-of-distribution and the model's learned representations break
down — producing unpredictable, potentially harmful activations (the Rogue Scalpel
mechanism). The geodesic stays within M_h throughout the path, so the LayerNorm
statistics remain calibrated.

Geodesic approximation: Curveball steering (arXiv:2603.09313) implements a discrete
approximation to the geodesic by computing the local curvature correction at the
starting point and adding a second-order correction term: h_geo = h + alpha*v +
(alpha^2/2) * curv_correction, where curv_correction is estimated from the local
Hessian or the kNN-PCA curvature. This is the Riemannian exponential map approximation.

CR reduction mechanism: rogue compliance CR requires the activation to visit regions
of activation space where the "harmful" detector direction has large projection.
Off-manifold regions are generic (F-B: in high d, any direction has many near-
orthogonal copies, including harmful-detector-adjacent directions). Staying on the
manifold reduces the probability of inadvertently visiting these generic harmful regions.

## 6. Predicted Delta

| Metric | Predicted Delta (geodesic vs additive) | Rationale |
|---|---|---|
| offshell at matched behavior | -50% to -70% relative | Geodesic construction |
| Rogue compliance rate | -30% to -50% relative | Fewer off-manifold visits |
| Behavior cosine shift | 0% difference at matched control | By design (matched behavior) |
| PPL | -10% to -25% relative | Less off-manifold = less saturation |
| Computation overhead | +20% to +50% | One Hessian-vector product needed |

## 7. Protocol

### 7.1 Primary experiment

- Model: Gemma-3-1B-it @L16
- Geodesic method: Curveball approximation (second-order Riemannian exponential map)
- Additive baseline: standard CAA (alpha*v)
- Matched behavior: sweep alpha (additive) and t (geodesic) to achieve same delta_cos = 0.10
- Metrics: offshell(delta-||h||/||h||), rogue-compliance rate (fraction of outputs
  triggering the adversarial probe — a simple harmful-content classifier), log-PPL
- Rogue probe: same probe as used in S-4 (CR=0.80 baseline rising to 1.00 under steering)
- Behaviors: 3 directions (refusal, politeness, factuality)
- Seeds: 3 x 3 = 9 cells per method
- Wall-clock: ~4 hours on RTX 4090

### 7.2 Where it shines

Safety-critical applications where rogue compliance is the primary failure mode:
jailbreak-resistance testing, red-teaming, safety steering. The geodesic advantage
should be largest when the steering alpha is large (stronger behavior change requires
longer manifold path vs longer chord, increasing the off-manifold gap).

## 8. Cross-References

- N5 (norm-budget): offshell is the N5 predictor; geodesic reduces offshell
- N16 (CRH radius/angle): geodesic moves along the angular direction with minimal
  radial change; this is the CRH-motivated "angle-only" steering
- N17 (concentration penalty): geodesic is predicted to reduce the N17 off-shell
  penalty; N13 and N17 are complementary measures of the same phenomenon
- N19 (trajectory beats endpoint): geodesic steering IS a trajectory-based method;
  N19 distributes budget across layers; N13 distributes the step along the manifold
- N12 (capstone): Phi_t in the unified operator is the geodesic flow component;
  N13 validates this component's value
- Rogue Scalpel paper: the CR metric is from arXiv:2509.22067
- IDEA_TABLE.md: N13 row, axes A8+A4

## 9. Committee Q&A

**Q: The Curveball approximation is a second-order correction. How accurately does
it approximate the true geodesic, and how do you validate the approximation?**

> The approximation error is O(alpha^3 * kappa^2), which is small for alpha < 0.1
> (the typical steering range). Validation: on a synthetic manifold (sphere in R^d)
> where the true geodesic is known analytically, verify that the Curveball approximation
> achieves < 5% path-length deviation at alpha = 0.15. If the approximation is poor,
> the experiment tests "Curveball vs additive" not "geodesic vs chord" — the distinction
> matters for interpretation.

**Q: The rogue compliance rate depends on the specific adversarial probe used.
How is the probe chosen to avoid confirmation bias?**

> The probe is the same harmful-content classifier used in the Rogue Scalpel paper's
> evaluation, reproduced independently on our model. It is fixed before any steering
> experiment is run. Alternatively, use the SAME probe from S-4 (CR=0.80 baseline)
> which is already calibrated. The probe is not changed post-hoc.

## 10. Verification Checklist

- [ ] Curveball approximation implemented with Hessian-vector product or kNN curvature
- [ ] Synthetic manifold validation: < 5% path-length error vs true geodesic at alpha=0.15
- [ ] Rogue probe fixed and documented before experiment
- [ ] Matched behavior protocol: delta_cos = 0.10 achieved for both methods
- [ ] offshell, CR, PPL measured for all 9 cells per method
- [ ] Ratio offshell(geo)/offshell(add) and CR(geo)/CR(add) reported with CIs
- [ ] IDEA_TABLE.md N13 row updated

## 11. Status Journal

- 2026-05-31 — Design doc created. Status: UNTESTED. The Manifold Steering paper
  (arXiv:2605.05115) and Curveball paper (arXiv:2603.09313) provide the theoretical
  and algorithmic foundation; neither has been reproduced on our Gemma models.
  S-7 (angular rotation at 2D-plane beats full-vector rotation) is consistent with
  the idea that the "chord" through the full d_model space is worse than a curved path,
  but N13 requires explicitly fitting the geodesic approximation.

---

## Addendum: Research-Scientist Critique

### Prior plausibility

MEDIUM-HIGH. The geometric argument is mathematically sound. The Manifold Steering
paper (arXiv:2605.05115) demonstrates the M_h <-> M_y bidirectional link empirically.
The main practical uncertainty is whether the Curveball second-order approximation
is accurate enough at realistic steering scales (alpha/||h|| ~ 0.1-0.2) to observe
the predicted off-manifold reduction.

### Mechanism scrutiny

The CR reduction mechanism (staying on M_h avoids generic harmful regions) is
plausible but not proven for the specific Gemma model family. The Rogue Scalpel
(arXiv:2509.22067) showed high CR for random feature directions; N13 claims geodesic
steering avoids these regions by design. But if M_h itself passes near harmful-adjacent
regions (which is possible if the training data contains harmful content), the geodesic
may not reduce CR.

### Confounds

Curvature estimation accuracy is the primary confounder: if the Curveball approximation
underestimates local curvature, the "geodesic" is actually another chord, and the
off-manifold reduction will be smaller than predicted. The synthetic validation step
is essential to rule out this confound before interpreting the main result.

### Skeptical effect-size estimate

offshell reduction: 20-40% (vs claimed 50-70%). CR reduction: 10-25% (vs claimed 30-50%).
The Curveball approximation at realistic scales likely achieves O(alpha^2) improvement,
which for alpha~0.1 corresponds to ~1% correction — possibly within noise.

### Minimum distinguishing experiment

Single behavior, matched delta_cos = 0.05 (small, to minimize approximation error):
measure offshell for Curveball vs additive at 3 seeds. Cost ~1 hour. If offshell ratio
< 0.80 (at least 20% reduction), proceed to full protocol.

### Verdict

TESTABLE-MEDIUM. The synthetic manifold validation is essential. If the approximation
is inaccurate at realistic steering scales, N13 cannot test the true geodesic claim.
The minimum experiment should include both the synthetic validation and one real-model
behavior measurement before committing to the full 4-hour protocol.

---

## Pseudocode & Methodology

This section specializes [`../METHODOLOGY.md`](../METHODOLOGY.md). N13 claims a geodesic (manifold-following) step has lower off-shell displacement and lower rogue-compliance than the additive chord at matched behavior. **UNTESTED** — needs a geodesic integrator.

### 1. Steering-vector recipe (geodesic vs chord)

```python
v = bank[L]["diffmean"]
# chord (baseline): additive  h_add = h + alpha*v        (METHODOLOGY §2 add)
# geodesic (NEW): Curveball 2nd-order Riemannian exp-map
#   h_geo = h + alpha*v + (alpha**2 / 2) * curv_correction   # curv_correction from Hessian-vec or kNN-PCA
# match behavior so both reach delta_cos = 0.10, then compare:
off_add = offshell_displacement(h, h_add)               # geometry.offshell_displacement
off_geo = offshell_displacement(h, h_geo)               # expected strictly lower
```

### 2. Experiment procedure

```text
1. Validate the geodesic approximation on a synthetic sphere: <5% path-length error vs analytic geodesic at alpha=0.15.
2. Fix the rogue probe (reuse the S-4 harmful-content classifier) BEFORE steering.
3. For 3 behaviors: sweep alpha (add) and t (geo) to matched delta_cos=0.10.
4. Measure offshell, rogue-compliance rate CR, log-PPL for both ops (9 cells each).
5. Report ratios offshell(geo)/offshell(add) and CR(geo)/CR(add) with CIs.
```

### 3. Measurement & decision rule

- **Primary metrics:** off-shell ratio and rogue-compliance ratio (geodesic / additive).
- **Pre-registered falsifier (§3):** offshell(geo) ≥ 0.8·offshell(add) ⇒ (A) FALSIFIED; CR(geo) ≥ 0.9·CR(add) ⇒ (B) fails; geodesic indistinguishable from chord ⇒ approximation too coarse.
- **Verdict logic:** the synthetic validation must pass first, else the test is "Curveball vs additive", not "geodesic vs chord".

### 4. Where the code is / status

UNTESTED. `geometry.offshell_displacement` exists and the safety/PPL probes exist, but the **geodesic integrator** (Curveball second-order exp-map via Hessian-vector product or kNN-PCA curvature) and the rogue-compliance probe are not implemented — that missing machinery is why N13 is UNTESTED.

See [`../METHODOLOGY.md`](../METHODOLOGY.md) for the shared recipe.

---

## Provenance & Tracing

No experiments run yet — see this design doc's protocol (§7) for what would be run. Once a campaign logs rows for this hypothesis, re-run `scripts/build_provenance.py` to generate `hypotheses/PROVENANCE/N13.md`.
