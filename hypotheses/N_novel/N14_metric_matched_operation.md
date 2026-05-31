# N14 — Metric-Matched Operation

> **One-line claim:** The optimal steering operation is determined by the concept's
> geometric structure: hierarchical concepts -> hyperbolic operations; polar/intensity
> concepts -> cylindrical; directional traits -> spherical; metric-mismatched operations
> cost coherence.
>
> **Primary axes:** A9 (metric/space), A4 (operation)
> **Status:** UNTESTED

---

## 1. Motivation (>= 100 words)

All current steering methods apply one of three operations: additive (Euclidean),
rotational (spherical), or conceptor-based. The choice of operation is typically
motivated by engineering considerations — which is simplest, which has the best
software support — not by the geometric structure of the concept being steered.
But the F-D finding from the missed dimensions document establishes that the right
operation is DERIVED from the metric of the representation space: in Euclidean space,
add; on a sphere, rotate; in hyperbolic space, apply Mobius addition; on a cylinder,
decompose into angle and radius operations. The HyCon paper (arXiv:2603.14093)
demonstrates that hierarchical concepts (which have tree-like structure) are better
modeled in hyperbolic space, where tree-like structures embed with low distortion.
When a hierarchical concept (e.g., "toxicity" which has sub-categories: hate speech,
threats, harassment) is steered using Euclidean addition, the operation treats the
hierarchy as flat — ignoring the tree structure that gives the concept its coherent
definition. Using the hyperbolic Mobius addition respects the hierarchy's geometry:
moving toward "toxicity" in hyperbolic space moves toward the parent, and the
sub-categories are automatically included without separately specifying each. The
cylindrical metric applies to intensity concepts: "politeness" has both a direction
(polite vs rude) and an intensity (very polite vs slightly polite). CRH's radial-
angular decomposition is exactly the cylindrical metric. Steering "politeness" should
control angle and radius independently for stable results.

## 2. Formal Hypothesis (>= 50 words)

For each of three concept categories (hierarchical, intensity/polar, purely directional),
the metric-matched operation should outperform mismatched operations on coherence
(log-PPL) at matched behavior-cosine shift. Specifically:

  (A) Hierarchical concepts: Mobius addition (hyperbolic) > additive (Euclidean) by
      at least 10% PPL reduction at matched behavior.
  (B) Intensity concepts: cylindrical (angle + radius control) > additive by at least
      8% PPL reduction at matched behavior.
  (C) Directional concepts: angular rotation (spherical) > additive by at least 5%
      PPL reduction — already tested for the case in S-7 (supported directionally).

## 3. Falsifier (>= 30 words)

If the metric-matched operation fails to outperform additive (Euclidean) on PPL by
at least 5% for any of the three concept categories (A), (B), or (C), that category's
claim is FALSIFIED. If ALL three categories show < 5% PPL advantage, the general
metric-matching principle is FALSIFIED for this model family.

## 4. Citations (Citation Rigor >= 80 words)

```
Briglia, Facchiano, Cursi, et al. 2026. 'Not All Latent Spaces Are Flat: Hyperbolic
Concept Control (HyCon)' arXiv:2603.14093. The direct reference for claim (A):
HyCon implements Mobius addition in Poincare ball hyperbolic space and reports
increased stability for hierarchical concept control. N14 tests this on our Gemma
models with the specific metric-mismatch ablation (Euclidean vs Mobius).

Gao et al. 2026. 'CRH' arXiv:2605.01844 (ICML 2026). The direct reference for
claim (B): CRH decomposes activations into radial (intensity) and angular (direction)
components; treating "intensity concepts" with cylindrical operations (control radius
and angle separately) is the CRH prescription. Screening result S-7 supports the
angular prediction (R2=0.997); N14 tests the full cylindrical vs Euclidean claim.

Raval et al. 2026. 'Curveball Steering' arXiv:2603.09313. Curveball optimizes the
steering path; for directional traits (claim C), the optimal path is the great-circle
arc on the sphere, which Curveball approximates. N14's directional concept test is
the overlap case between N13 and N14.
```

## 5. Mechanism

The metric-matching principle follows from F-D in the missed dimensions document:

Hierarchical concepts (toxicity, safety-categories, logical hierarchy): in hyperbolic
space H^d, tree structures embed with low distortion (Poincare embedding, Nickel &
Kiela 2017). Moving along a geodesic in H^d toward the "toxicity" concept moves toward
the subtree root, automatically including all sub-types. The Mobius addition is the
group operation in the Poincare ball: x +_M y = ((1+2c<x,y>+c||y||^2)x +
(1-c||x||^2)y) / (1+2c<x,y>+c^2||x||^2||y||^2) with curvature parameter c=1.
In flat space, subtree members project to different Euclidean directions; an additive
step toward the parent inadvertently moves some sub-types closer and others farther
(non-uniform effect).

Intensity/polar concepts (politeness-level, confidence, assertiveness): these have
two independent degrees of freedom — direction (polite vs rude) and intensity (how
much). CRH's cylindrical decomposition (arXiv:2605.01844) separates these exactly:
the angular component controls the direction, the radial component controls the
intensity. Euclidean addition couples both in one alpha, making it impossible to
increase intensity without also changing direction (unless v happens to be exactly
radial). Cylindrical steering: alpha_angle = target_direction; alpha_radius = target_intensity;
update = rotate by alpha_angle then scale by alpha_radius.

Directional traits (factual, creative, technical): these have one primary degree of
freedom (the direction of the trait), no meaningful intensity variation, and live on
a sphere. Spherical rotation (angular steering) is the natural operation — supported
by S-7 (R2=0.997 for angular predicting rotation log-PPL).

## 6. Predicted Delta

| Metric | Category | Predicted Advantage (metric-matched vs additive) | Rationale |
|---|---|---|---|
| log-PPL | Hierarchical | >= 10% reduction | Hyperbolic preserves subtree structure |
| log-PPL | Intensity/polar | >= 8% reduction | CRH decomposition, supported by S-7 |
| log-PPL | Directional | >= 5% reduction | Angular steering, S-7 directional support |
| Behavior-cosine shift | All | 0% difference (matched by design) | Same behavior, different path |
| Rogue compliance rate | Hierarchical | >= 20% reduction | Mobius stays within subtree |

## 7. Protocol

### 7.1 Primary experiment

- Model: Gemma-3-1B-it @L16
- Behaviors:
  (A) Hierarchical: toxicity steering (general -> specific sub-types hierarchy)
  (B) Intensity: politeness degree steering (slightly vs very polite)
  (C) Directional: factual vs creative (a pure directional contrast)
- Operations: for each concept: (i) additive (Euclidean), (ii) metric-matched
  (Mobius for A, cylindrical for B, angular for C)
- Implementation: Mobius addition in Poincare ball (c=1); cylindrical = angle
  rotation + radius scaling; angular = 2D-plane Givens rotation
- Matching: for each behavior, matched delta_cos = 0.08 across both operations
- Metrics: log-PPL, behavior cosine, offshell, rogue compliance (category A only)
- Seeds: 3 behavior-extraction x 3 evaluation = 9 cells per (concept, operation) pair
- Wall-clock: ~5 hours on RTX 4090

### 7.2 Where it shines

Deployments with mixed concept types: a system that steers both hierarchical safety
concepts and directional stylistic concepts should use metric-matched operations for
each type, rather than a single universal operation. N14 provides the empirical basis
for this routing decision.

## 8. Cross-References

- N16 (CRH radius/angle): N14 claim (B) is exactly the N16 CRH claim applied to
  intensity concepts specifically; N16 is the general version, N14 (B) is the
  category-specific test
- N13 (geodesic vs chord): N13 tests geodesic for directional traits; N14 extends
  to hierarchical and intensity concepts
- HyCon paper (arXiv:2603.14093): the primary reference for Mobius addition implementation
- N12 (capstone): the "metric" axis (A9) in the unified operator is set by concept
  category; N14's categorization table is the routing function for A9
- IDEA_TABLE.md: N14 row, axes A9+A4

## 9. Committee Q&A

**Q: How do you classify a concept as "hierarchical" vs "intensity" vs "directional"?
This classification is itself subjective.**

> The classification uses a pre-registered, behavior-free criterion: hierarchical
> = the concept has a taxonomic sub-type structure (verifiable by ontology lookup,
> e.g., toxicity subtypes are documented in AxBench); intensity = the concept has
> a natural magnitude scale (e.g., politeness is rated 1-5 in human annotation data);
> directional = binary contrast with no natural magnitude scale. The classification
> is determined BEFORE any steering experiment and does not use PPL to decide.

**Q: HyCon uses a Poincare ball embedding of the concept; how do you know the
steering direction in Euclidean activation space corresponds to a direction in the
Poincare ball?**

> We use the exponential map: given a Euclidean steering direction v at origin 0 in
> the Poincare ball, the geodesic path is exp_0(v) = tanh(||v||/2) * v/||v||. This
> maps the Euclidean direction to the corresponding Mobius-space direction. We then
> apply Mobius addition along this direction. The curvature parameter c=1 is the
> standard Poincare ball setting; c could be treated as a hyperparameter.

## 10. Verification Checklist

- [ ] Concept classification (hierarchical/intensity/directional) documented pre-experiment
- [ ] Mobius addition implementation verified on synthetic Poincare ball data
- [ ] Cylindrical decomposition: angle and radius control independently implemented
- [ ] Matched delta_cos = 0.08 achieved for all (concept, operation) pairs
- [ ] log-PPL measured for all 9 cells per (concept, operation) pair
- [ ] Rogue compliance measured for hierarchical category
- [ ] Metric advantage table reported with p-values and CIs
- [ ] IDEA_TABLE.md N14 row updated

## 11. Status Journal

- 2026-05-31 — Design doc created. Status: UNTESTED. HyCon (arXiv:2603.14093) and
  CRH (arXiv:2605.01844) provide the algorithmic foundation. S-7 provides directional
  support for claim (C) (angular beats additive for rotation). Claims (A) and (B)
  require implementing Mobius addition and cylindrical decomposition, neither of
  which is currently in the project codebase.

---

## Addendum: Research-Scientist Critique

### Prior plausibility

MEDIUM. The metric-matching principle is mathematically sound; the question is whether
Gemma's activation space has enough geometric structure for the concept-specific
metrics to be measurably superior. If the activation space is approximately Euclidean
at all scales relevant to steering (alpha/||h|| < 0.2), metric-matching provides no
measurable gain.

### Mechanism scrutiny

Claim (A) requires that the Poincare ball metric (defined on a hyperbolic embedding)
is the correct metric for the activation space representation of hierarchical concepts.
But GemmaScope features are in a Euclidean activation space; the Poincare ball metric
is imposed on top of this via the exp-map. If the concept's hierarchical structure
is not reflected in the GEOMETRIC structure of the activation space (i.e., if
"toxicity" and "hate-speech" are not geometrically parent-child in activation space),
the hyperbolic operation provides no advantage.

### Confounds

1. The concept classification (hierarchical/intensity/directional) drives the
   predicted effect; if the classification is wrong, the experiment tests the
   wrong operation for the wrong concept.
2. The implementation complexity (Mobius addition, Poincare ball) introduces more
   potential implementation bugs than simple additive/rotational methods.

### Skeptical effect-size estimate

PPL advantage: 3-7% for hierarchical (vs claimed 10%); 4-8% for intensity (vs claimed 8%);
5-10% for directional (vs claimed 5% — but S-7 already supports this direction).
The hierarchical case is the most uncertain; the directional case has the strongest prior.

### Minimum distinguishing experiment

Directional concept (C) only: additive vs angular at matched delta_cos = 0.08, one
behavior, 3 seeds (~1 hour). S-7 screening result makes this likely to show > 5%
PPL advantage. If confirmed, proceed to intensity (B) and hierarchical (A) in that
priority order.

### Verdict

TESTABLE-MEDIUM. The directional concept case (C) has strong prior support from S-7
and should be run first as the most likely-to-succeed sub-claim. The hyperbolic case
(A) is the most novel and uncertain. Implement in order: (C) first, then (B), then (A).
