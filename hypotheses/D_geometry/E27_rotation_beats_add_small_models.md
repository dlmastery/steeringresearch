# E27 — Rotation Beats Addition on Small Models (< 3 B)

> **One-line claim:** Norm-preserving rotation beats additive steering on small
> (< 3 B) models where additive edits more easily exit the activation manifold.
>
> **Source design space:** Block D — Geometry and Rotational Methods (E27–E33).
>
> **Implementation status:** SCREENING (n=1, FALSIFIED for full-vector rotation,
> SUPPORTED for scale-fragility sub-claim). See S-7 and S-8 in FINDINGS.md.

---

## 1. Motivation (>= 100 words)

Small language models occupy a geometric regime quite different from their
multi-billion-parameter siblings. Their representational manifolds are lower-
dimensional, their activation norms at comparable layers are smaller, and the
in-distribution shell — the thin spherical shell where legitimate activations
cluster by concentration-of-measure — is proportionately thinner. Additive
steering, which works by adding a fixed vector alpha*v to a hidden state h,
necessarily changes the L2 norm of h and therefore moves it radially away from
that shell. For a 1B model with an activation norm around 50, a displacement of
alpha=5 moves h approximately 10% off-shell; for a 270M model the same alpha can
represent a 30-40% displacement. This makes additive steering on small models
intrinsically fragile — perplexity spikes at lower alpha values, and the coherence
window (the range of alpha that improves behavior without destroying fluency)
either narrows or disappears entirely. Rotational steering, which rotates h within
a plane toward the target direction while preserving its norm, should in principle
keep the activation on the in-distribution shell entirely: a norm-preserving
operation by construction. Angular Steering (arXiv:2510.26243) proposed exactly
this for transformer hidden states; Selective Steering (arXiv:2601.19375) added
the insight that selecting only the 2D plane of the behavior direction (rather
than rotating the full d-dimensional hidden state) is necessary to avoid the
large angular displacement that full-vector rotation incurs on small models. The
CRH (arXiv:2605.01844) provides the formal decomposition: activation differences
live on a cylinder, with a radial component (norm change, governing additive
damage) and an angular component (direction change, governing rotational damage).
The motivating prediction is that, for small models where additive alpha-scaling
hits the coherence cliff early, rotational steering in the selective 2D-plane
sense should allow the same behavior change at a lower total coherence cost. Our
campaign C3/C3b has already tested part of this claim.

---

## 2. Formal hypothesis (>= 50 words)

**E27 contains two separable sub-claims that the screening data resolve differently
and that must be tracked independently:**

**Sub-claim E27-A (operation):** Among operation types applied at matched behavior-
shift magnitude on a < 3B model, selective-plane norm-preserving rotation (as in
Selective Steering 2601.19375) will produce strictly lower perplexity than
full-vector additive steering. The predicted gap is >= 10% lower PPL at matched
behavior-success rate, with angular displacement predicting rotation's PPL at
R2 >= 0.90 and radial displacement predicting addition's PPL at R2 >= 0.80.

**Sub-claim E27-B (scale):** The coherence window (range of control parameter
yielding behavior success > 0.5 without PPL > 150) is strictly absent on 270M
models, narrow on 1B models, and wider on > 1B models, i.e., smaller models are
more fragile regardless of operation type.

---

## 3. Falsifier (>= 30 words)

**E27-A is FALSIFIED (screening) for full-vector rotation:** at matched behavior
~= 0.57, additive steering (PPL 92.8) beats full-vector rotation (PPL 131.4,
+42%) on Gemma-3-270M @L16 (C3b, S-7, n=1). The caveat is critical: this
falsifies full-vector rotation, not selective-plane rotation. E27-A remains
UNTESTED for selective-plane rotation.

**E27-A formal re-falsifier:** if selective-plane rotation on Gemma-3-270M (the
smallest tested model) produces PPL > 110 at behavior success >= 0.55 when
matched-displacement additive steering produces PPL < 100, the rotational
advantage claim is definitively rejected for both full and selective variants.

**E27-B formal falsifier:** if the coherence window (behavior > 0.5, PPL < 150)
is as wide on 270M as on 1B at matched fractional alpha, the scale-fragility claim
is rejected. Current data SUPPORT E27-B (270M no window; 1B narrow window; S-4,
S-8), but n=1 and this requires reproduction at n >= 7.

---

## 4. Citations (Citation Rigor format, >= 80 words)

```
Turner, Alex et al. 2024 'Steering Language Models With Activation Engineering'
(arXiv:2308.10248) — founding CAA additive baseline against which rotation claims
are measured; establishes the DiffMean paradigm that E27 contests.

Arditi, Andy et al. 2024 'Refusal in Language Models is Mediated by a Single
Direction' (arXiv:2406.11717) — demonstrates that additive removal of the refusal
direction transfers across models, motivating the cross-scale comparison in E27.

Angular Steering: Liu et al. 2025 'Angular Steering: Norm-Preserving Direction
Interventions in Language Models' (arXiv:2510.26243) — the primary claim we test:
rotation preserves norm and avoids the manifold-exit problem of addition.

Selective Steering: Bai et al. 2026 'Selective Steering: Norm-Preserving 2D-Plane
Rotation for Large Language Models' (arXiv:2601.19375) — the refinement that
restricts rotation to a single 2D plane, avoiding the large angular displacement
that full-vector rotation incurs; this is the variant E27-A MUST test.

Cylindrical Representation Hypothesis: Gao et al. 2026 'The Cylindrical
Representation Hypothesis: Radial and Angular Decomposition of Activation Steering'
(arXiv:2605.01844) — the formal framework that explains why radial (additive) and
angular (rotational) displacements are the correct predictors of each method's
coherence cost; confirmed in our C3b data at R2=0.997 (angular) and R2=0.81
(radial).

Manifold Steering: Wurgaft et al. 2026 'Manifold Steering: Geometry-Aware
Activation Interventions' (arXiv:2605.05115) — establishes that manifold-aware
operations generalise the rotation-vs-addition tradeoff across non-linear
manifolds.
```

---

## 5. Mechanism

### 5.1 Full-vector rotation vs selective-plane rotation

When a full d-dimensional hidden state h is rotated toward a target direction v,
the rotation matrix R(theta) in the span{h, v} plane displaces h by an angular
distance of theta. For theta = 0.1 rad the angular displacement metric
(1 - cos theta) = 0.005, and our data confirm log PPL = 4.57 + 43.1 * (1-cos)
at R2 = 0.997 (C3b). At theta = 0.5 rad (1-cos = 0.122) the model is
non-functional (PPL > 11000). This is NOT norm change — off-shell Δ||h|| ≈ 0
throughout — but the direction change in the full d-space causes the LayerNorm
inputs downstream to see a different direction, which cascades.

Selective-plane rotation (Selective Steering 2601.19375) restricts the rotation
to the 2D plane spanned by h and v, leaving all components orthogonal to this
plane unchanged. The angular displacement is now theta but only within the plane;
all other dimensions of h are untouched. This means the magnitude of the angular
shift seen by downstream LayerNorm is theta / sqrt(d), reduced by approximately
64x for d = 4096. This is the claimed advantage: selective rotation should
allow theta as large as 0.5 rad without the PPL explosion that full-vector
rotation produces at the same angle.

### 5.2 Scale-fragility mechanism

The coherence window width scales inversely with the sharpness of the activation
norm distribution. Smaller models have lower effective rank in their residual
stream, meaning activations concentrate in a lower-dimensional subspace with a
relatively thinner shell. A displacement of magnitude delta moves off-shell by
delta / ||h||; for 270M models ||h|| is approximately 30-40 at L12-16, while
for 1B models ||h|| is approximately 60-80. Therefore the same absolute alpha
displaces the 270M model proportionally further, explaining why the coherence
window emerges with scale (S-4, S-8).

---

## 6. Predicted Delta (pre-registered)

| Metric | E27-A (selective rotation vs add) | E27-B (scale comparison) |
|---|---|---|
| PPL at matched behavior | >= -10% (rotation better) | 270M: no window; 1B: narrow window |
| Behavior success at cliff | >= 0.55 for selective rotation | Monotone increase in window width with scale |
| Angular R2 (rotation coherence) | >= 0.90 | Not applicable |
| Radial R2 (add coherence) | >= 0.80 (confirmed C2 R2=0.81) | Not applicable |
| Composite delta | [+0.05, +0.20] selective rotation vs add | Cross-scale ordering SUPPORTED (n=1) |

Note: E27-A for FULL-vector rotation is FALSIFIED (-42% PPL degradation relative
to addition at matched behavior). Only the selective-plane variant remains open.

---

## 7. Protocol

### 7.1 Primary experiment (E27-A: selective rotation)

Implement selective-plane rotation on Gemma-3-270M and Gemma-3-1B:
compute the 2D rotation matrix in span{h_normalized, v_normalized}, apply it
only to the component of h lying in that plane, leave orthogonal components
unchanged. Sweep theta in {0.02, 0.05, 0.10, 0.20, 0.30} rad. Record behavior,
PPL, angular displacement, radial displacement, composite. Compare to matched-
displacement additive runs from C3b/C9b. n=3 seeds minimum; n=7 for gate.
Success criterion: selective rotation PPL < additive PPL by >= 10% at behavior
>= 0.55 on at least one model.

### 7.2 Where it shines

E27-B (scale fragility) is the strongest current signal. The harness should
systematically add Gemma-3-4B-it or Qwen-2.5-3B as the upper-end model to
confirm the 270M < 1B < 3B fragility ordering with matched infrastructure.

---

## 8. Cross-references

- Related hypotheses: E31 (rotation preserves norm/LN stats), E29 (geodesic
  monotonicity), E23 (same-plane vs orthogonal-plane composition)
- Corpus: N16 (CRH / cylindrical decomposition), N17 (concentration/norm budget)
- Screening results: S-7 (E27-A falsified full-vector), S-4, S-8 (E27-B supported)
- Corpus papers: Angular Steering 2510.26243, Selective Steering 2601.19375,
  CRH 2605.01844, Manifold Steering 2605.05115
- Campaign results: C3_results.md (C3/C3b), C6_results.md (scale comparison)

---

## 9. Committee Q&A

**Q: The screening already falsified E27 — why keep it open?**

> The screening falsified full-vector rotation (the simple implementation). The
> corpus hypothesis and the Selective Steering paper both specify 2D-plane rotation,
> which is a distinct operation. A correct test of E27-A requires implementing the
> selective-plane variant. Closing the hypothesis before testing the correct
> implementation would be a design error.

**Q: Does the scale-fragility claim (E27-B) require rotation at all?**

> No. E27-B is an independent sub-claim about additive steering fragility as a
> function of model scale. It is grouped under E27 because the original motivation
> was that rotation would compensate for small-model fragility. E27-B stands on its
> own regardless of the E27-A result.

**Q: Isn't the 42% PPL difference at n=1 noise?**

> Yes, it is a screening observation. The pattern across theta={0.05..0.5} is
> monotone with four data points and R2=0.997 for the angular predictor, which
> strongly suggests a real effect — but the claim requires n>=7 and real behavior
> evaluation, not the projection proxy, before promotion.

**Q: If selective rotation is norm-preserving, does E31 automatically hold?**

> Yes, by construction — a selective 2D rotation preserves the overall L2 norm of
> h exactly. E31 tests whether this norm preservation translates into downstream
> LayerNorm variance preservation, which is a separate empirical question.

---

## 10. Verification checklist

- [ ] Selective-plane rotation operator implemented and unit-tested (rotation by
      theta=pi/2 on a 2D subspace of a 4096-d vector recovers expected geometry)
- [ ] C3b additive results reproduced with `relative_add` at n>=3 seeds
- [ ] E27-A selective-rotation sweep completed on Gemma-3-270M, n>=3
- [ ] E27-A selective-rotation sweep completed on Gemma-3-1B, n>=3
- [ ] Angular displacement metric confirmed at R2>=0.90 for selective rotation
- [ ] E27-B scale comparison extended to >= 3 model sizes at n>=7
- [ ] Composite delta computed with fingerprinted formula a9001e87087e
- [ ] Result rows added to EXPERIMENT_LEDGER.md

---

## 11. Status journal

- 2026-05-31 — Created. E27-A FALSIFIED (screening, n=1) for full-vector rotation
  (C3b/S-7: +42% PPL vs addition at matched behavior). E27-B SUPPORTED (screening,
  n=1) for scale-fragility ordering (S-4/S-8: 270M no window < 1B narrow window).
  E27-A remains UNTESTED for selective-plane rotation. Next action: implement
  selective-plane rotation operator and run C3c sweep.

---

## Addendum: Research-Scientist Critique

*Reviewer: SciCritic-D (elite-research-scientist critic). Critiquing the scientific
idea and experimental evidence, not the implementation.*

### Prior plausibility

**LOW-to-MEDIUM for E27-A as originally stated; MEDIUM-HIGH for E27-B.**
The intuition that norm-preserving operations are gentler is sound and grounded
in the concentration-of-measure argument. However, the falsification of full-vector
rotation (the natural interpretation of "norm-preserving rotation") at n=1 is a
serious early signal. The Selective Steering rescue argument is plausible but
relies on a specific implementation (2D-plane restriction) that the original E27
claim did not specify. It looks like hypothesis-rescue via narrowing, which should
be flagged: the claim is now unfalsified only because the tested version differs
from the original.

### Mechanism scrutiny

The mechanism for E27-A (selective rotation reduces angular displacement by
1/sqrt(d)) is geometrically sound. The 2D-plane rotation applies theta to a
subspace of dimension 2 within d=4096, so the effective angular shift seen by
downstream operations scales as theta * sqrt(2/d) ≈ theta / 45 for d=4096.
This is the correct computation. The CRH angular predictor (R2=0.997) further
supports the mechanism. The concern is that even a small selective-rotation
angle changes the h direction within the important subspace — and if the
behavior is encoded primarily in that subspace (which is why we rotate it), then
the angular change in the relevant subspace is not small at all, it is theta.
The "dilution by 1/sqrt(d)" argument assumes the downstream damage is spread
over all d dimensions, but LayerNorm reads the full vector including the rotated
subspace at its full theta angular change.

### Confounds

1. **Layer choice confound:** C3b ran at L16, identified as best in C1. A selective
   rotation at L12 (max-Fisher layer) or L18 might behave differently.
2. **Model confound:** Gemma-3-270M is the smallest and most fragile tested model;
   the E27-A advantage may be most visible at this scale but the test is hardest
   here because any operation is fragile.
3. **Behavior proxy confound:** behavior is measured as projection cosine, not real
   generation quality. Selective rotation may improve projection while worsening
   actual outputs.

### Does it specifically matter?

**For practice, MEDIUM.** If selective rotation matches or beats additive steering
on small models, it is a meaningful practical contribution because small models
(< 3B) are the primary deployment target for resource-constrained applications.
The 1/sqrt(d) attenuation argument also generalizes to a design principle for
other operations (apply edits in low-rank subspaces to limit angular spread).

### Literature precedent

Selective Steering (2601.19375) explicitly claims to fix Angular Steering's norm-
violation problem on small models — so the positive E27-A claim has direct
published support for the selective variant. Our falsification of the full-vector
variant is consistent with the published literature; we are on the right track
testing the selective variant.

### Skeptical effect-size re-prediction

For selective-plane rotation vs. additive at matched fractional displacement, my
prior is: Δ(PPL) ∈ [-15%, +5%] (90% CI), i.e., selective rotation is likely
slightly better but not guaranteed. The claimed >= 10% advantage is within a
plausible range given the mechanism. Effect size on behavior success: Δ < 0.05
absolute (the operations target the same plane and the behavioral projection
should be similar).

### Minimum-distinguishing experiment

Run selective-plane rotation (implementing the exact 2D Givens rotation in
span{h, v}) vs. matched-displacement additive at theta = 0.1 rad on Gemma-3-270M
@L16, n=7, reporting PPL and behavior via real LLM-judge. Wall-clock: < 2 hours
on a 4090. If PPL selective < PPL additive by >= 10% at behavior parity, E27-A
is confirmed. If not, the hypothesis is narrowed to "selective rotation is
equivalent to addition but not better."

### Verdict

**SPLIT: E27-A PARTIALLY OPEN (screening falsification of wrong variant; correct
variant untested); E27-B SCREENING-SUPPORTED (scale-fragility ordering at n=1).
Both sub-claims are testable within current infrastructure. Priority: implement
selective-plane rotation for E27-A; extend scale comparison to n>=7 for E27-B.**

---

## Pseudocode & Methodology

This section specializes [`../METHODOLOGY.md`](../METHODOLOGY.md) to the rotation-vs-
addition operation comparison on small models. **Status: SCREENED (n=1) — E27-A
FALSIFIED for full-vector rotation (S-7); E27-B scale-fragility SUPPORTED (S-4/S-8);
selective-plane rotation UNTESTED.** This is a GEOMETRY/OPERATION hypothesis: the
vector is the usual DiffMean — what changes is the injection operation.

### 1. Steering-vector recipe (one DiffMean direction, three operations)

```python
# METHODOLOGY §1.3 — closed-form DiffMean, the SAME vector for add and rotate:
v = extract.build_vector_bank(model, tok, load_concept("ocean"), L)[L]["diffmean"]
# Operations (METHODOLOGY §2):
#   add     : h' = h + alpha * v
#   rotate  : e1 = unit(h); e2 = unit(v - (v·e1)e1)                 # full-vector (TESTED, FALSIFIED)
#             h' = ||h|| * (cos(alpha)*e1 + sin(alpha)*e2)          # norm-preserving
#   selective-plane rotate: apply the 2D Givens rotation only to the span{h,v}
#             component, leave the d-2 orthogonal dims untouched     # UNTESTED variant
```

### 2. Experiment procedure

```text
# scripts/campaign_sweep.py --ops add rotate project_out --sources diffmean (C3/C3b)
for op in {add, rotate, project_out}:
  for alpha/theta in sweep_grid:
    for seed in seeds:
      h' = hooks.apply_operation(h, v, op, alpha)
      measure: behavior, PPL,
               geometry.offshell_displacement   # ~0 for rotate (norm-preserving), >0 for add
               geometry.angular_displacement    # 1-cos theta; the rotate damage predictor
# E27-B: repeat add sweeps across model sizes {270M, 1B, >1B}; map coherence-window width
```

### 3. Measurement & decision rule

- PRIMARY metric: PPL at matched behavior (E27-A); coherence-window width vs scale (E27-B).
- Pre-registered FALSIFIER (§3): selective-plane rotation PPL `>110` at behavior `>=0.55`
  while matched-displacement add PPL `<100` ⇒ rotation advantage rejected. E27-B: window
  as wide on 270M as 1B at matched fractional α ⇒ scale-fragility rejected.
- **Actual (n=1):** at matched behavior ≈0.57, add PPL 92.8 beats **full-vector rotate
  PPL 131.4 (+42%)** on Gemma-3-270M @L16 ⇒ **E27-A FALSIFIED for full-vector rotation.**
  Angular predictor R²=0.997, radial R²=0.81 (C3b) confirmed. E27-B SUPPORTED (270M no
  window < 1B narrow window). Selective-plane variant remains the open re-test.

### 4. Where the code is / status

`add`, `rotate`, `project_out` all exist in `hooks.apply_operation`; driven by
`scripts/campaign_sweep.py` (exp#28–46). `geometry.angular_displacement` /
`offshell_displacement` are the operation-specific damage predictors. MISSING for the
open re-test: a **selective-plane (2D Givens) rotation operator** and an `n>=7` cross-
scale sweep. E27-A (full-vector) is TESTED+FALSIFIED; E27-A (selective) and the n>=7
E27-B confirmation are UNTESTED.

See [`../METHODOLOGY.md`](../METHODOLOGY.md) for the shared recipe.

---

## Provenance & Tracing

Full per-hypothesis provenance (exact experiments, reproduce commands, artifact links, reasoning trace): [`PROVENANCE/E27.md`](../PROVENANCE/E27.md).

- **Experiments:** exp# 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 123 (`autoresearch_results/experiment_log.jsonl`).
- **Reproduce:**

```bash
PYTHONPATH=src python scripts/campaign_sweep.py --model models/google/gemma-3-270m-it --quant none --hyp E27 --tag-prefix C3-E27-op --behavior ocean --layers 16 --alphas 1.0 2.0 4.0 --ops add rotate project_out --sources diffmean
```
