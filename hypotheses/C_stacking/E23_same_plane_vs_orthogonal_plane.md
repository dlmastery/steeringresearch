# E23 — Same-Plane vs Orthogonal-Plane Additive + Rotational Composition

> **One-line claim:** Additive and rotational edits on the same plane interfere
> destructively, but on orthogonal planes they stack; site/operation governs
> composability, not method name.
>
> **Source design space:** Block C — Stacking and Multi-Vector Composition (E17–E26).
>
> **Implementation status:** `PENDING — UNTESTED`. Requires both additive and
> rotational injection code plus plane-identification tooling.

---

## 1. Motivation (>= 100 words)

The decision matrix in the corpus (steering-stackable-vs-competing-analysis.md)
states: "same axis + different op (add vs rotate) -> COMPETE (double-count,
off-manifold)". This is the core prediction that additive and rotational methods
cannot be safely combined on the same 2D behavior plane. However, the competition
is plane-specific: if an additive edit moves h along the refusal direction and
a rotational edit rotates h in an orthogonal plane (say, the formality-sentiment
plane), they operate on geometrically independent subspaces and should stack
without interference. The existing literature (Angular Steering arXiv:2510.26243,
Selective Steering arXiv:2601.19375) focuses on the competition within a plane;
no paper tests the positive case (orthogonal planes compose). This experiment
provides the first systematic test of the plane-specificity claim:
COMPETITION is not between method names (add vs rotate), it is between
methods that share a plane AND use incompatible operations. Methods on
orthogonal planes stack regardless of their operation type. This is the
cleanest algebraic claim in Block C and will either confirm or challenge the
7-axis decision rule.

---

## 2. Formal Hypothesis (>= 50 words)

The 7-axis decision rule (corpus: steering-first-principles-v2) states:
composability is determined by (AXIS 1: site) and (AXIS 2: direction/plane)
— not by AXIS 4 (operation) in isolation. Two edits compete if and only if
they target the same plane with incompatible operations. Two edits on orthogonal
planes compose regardless of their operation type. Formal claim: (a) additive
edit on plane P1 + rotational edit on SAME plane P1 produces joint success rate
< 70% of the better solo method (interference); (b) additive edit on P1 +
rotational edit on ORTHOGONAL plane P2 produces joint success rate >= 90%
of each solo method (stacking), at 3-seed median on Gemma-2-2B-it.

---

## 3. Falsifier (>= 30 words)

If (a) same-plane composition achieves >= 90% of solo efficacy (no interference
between add and rotate on the same plane), the decision rule is FALSIFIED for
the competition prediction. If (b) orthogonal-plane composition achieves < 90%
of solo efficacy, the stacking prediction is FALSIFIED. Either failure invalidates
the plane-based decision rule in the 7-axis framework.

---

## 4. Citations (Citation Rigor >= 80 words)

```
(anon) 2025 arXiv 'Selective Steering: Discriminative Layer Identification for
Targeted Activation Steering' (arXiv:2601.19375) — introduces norm-preserving
rotation as an alternative to additive steering; explicitly claims that adding
and rotating in the same plane is redundant and potentially harmful; provides
the competition hypothesis this experiment tests.

(anon) 2025 arXiv 'Angular Steering for Language Models' (arXiv:2510.26243) —
angular steering as a rotational method; the paper motivates rotation over
addition for small models where additive edits are more likely to leave the
manifold; the same-plane competition is implicit in this motivation.

[N16 radial/angular result, C3b, this project, SUPPORTED]: angular predicts
rotation logPPL R² = 0.997; radial predicts additive logPPL R² = 0.81 —
the decoupling of radial (additive) and angular (rotational) components is the
first-principles justification for why mixing the two on the SAME plane is
problematic: each operation moves h differently in the same 2D subspace.

[Steering-stackable-vs-competing-analysis.md §3.1, this project]: "Two
philosophies target the same refusal/behavior plane with incompatible operations.
You pick one: either you add along the direction or you rotate toward it.
Applying both to the same plane double-counts and tends to push h off-manifold."
```

---

## 5. Mechanism

### 5.1 Same-plane interference

Let P1 = span{v_behavior, v_orthogonal} be the 2D behavior plane.
Additive edit: h <- h + alpha * v_behavior  (moves h along v_behavior)
Rotational edit on P1: h <- Rot(theta) * h  (rotates h within P1)

The combined edit: h <- Rot(theta) * (h + alpha * v_behavior)
This is NOT equivalent to either solo edit; it applies the rotation to the
already-displaced h, producing a different trajectory through P1. The rotation
"doubly-perturbs" the behavior direction: first by adding alpha * v_behavior,
then by rotating the result. This can either overshoot or undershoot the target
behavior, depending on the starting direction of h — hence "interference."

### 5.2 Orthogonal-plane composition

Let P2 = span{v_2, v_orthogonal2} where v_2 perp to v_behavior.
Rotational edit on P2: h <- Rot(theta) * h restricted to P2 direction

The P1 additive edit moves h along v_behavior (within P1).
The P2 rotational edit rotates h within P2 (zero component along v_behavior).

The two edits commute (because P1 and P2 are orthogonal): their combination
is exactly the sum of the two individual edits with no interaction. This is
the stacking prediction.

---

## 6. Predicted Delta

| Metric | Predicted value | Rationale |
|---|---|---|
| Same-plane joint success (add+rotate on P1) | < 70% of solo | Core claim: competition |
| Orthogonal-plane joint success (add on P1, rotate on P2) | >= 90% of solo | Core claim: stacking |
| Same-plane PPL overhead (vs best solo) | [+0.5, +2.0] | N5 off-manifold from double-perturbation |
| Orthogonal-plane PPL overhead | [0, +0.5] | Orthogonal composition: additive but small |

N16 (SUPPORTED: radial vs angular decoupled R²=0.997/0.81) provides
mechanistic grounding. Pre-registered; [NEEDS VERIFICATION] on Gemma-2-2B.

---

## 7. Experimental Protocol

### 7.1 Primary experiment

- **Model:** Gemma-2-2B-it, 4-bit; smoke on Gemma-3-1B-it
- **Plane identification:** P1 = refusal plane (v1 = refusal DiffMean, v_orth
  = perpendicular in refusal subspace); P2 = sentiment plane (v2 = sentiment
  DiffMean, confirmed |cos(v1,v2)| < 0.1)
- **Conditions:** (1) additive solo P1, (2) rotational solo P1, (3) add P1 +
  rotate P1 (same-plane), (4) add P1 + rotate P2 (orthogonal-plane)
- **Metrics:** behavior efficacy for refusal (P1) and sentiment (P2); PPL;
  offshell ||delta h||
- **Seeds:** 3 (screening)
- **Wall-clock:** ~2 h on 4090

### 7.2 Where it shines

The competition prediction (same-plane) is most visible when both operations
target the exact same 2D subspace. If the planes are only approximately
orthogonal (|cos| = 0.15 rather than 0), the orthogonal-plane result will
show mild interference — a partial NEAR-MISS.

---

## 8. Cross-References

- **E17** (near-orthogonal stacking): additive + additive on orthogonal planes
  — the all-additive version of E23's orthogonal condition
- **N16** (radial/angular, SUPPORTED): the fundamental decoupling that predicts
  why orthogonal planes compose regardless of operation type
- **E31** (rotation preserves norm): rotational solo on P1 is the reference
  for the same-plane interference measurement
- **IDEA_TABLE.md** Block C row E23

---

## 9. Committee Q&A

**Q: How do you ensure the rotational edit stays strictly within P1?**

> Angular Steering rotates h within the 2D plane defined by h and v_behavior.
> This is exactly P1 by construction. Selective Steering is norm-preserving and
> targets only the behavior subspace. We use one of these implementations to
> ensure the rotation is confined to P1.

**Q: Why is the 70% threshold for same-plane competition vs 90% for orthogonal
stacking?**

> The asymmetric thresholds reflect the direction of the prediction: competition
> implies meaningful degradation (< 70% of solo), stacking implies near-full
> preservation (>= 90%). The gap between 70% and 90% is the discriminative range
> where the decision rule makes a falsifiable distinction.

---

## 10. Verification Checklist

- [ ] Plane P1 and P2 identified; |cos(v1,v2)| < 0.1 confirmed
- [ ] Four conditions implemented and evaluated
- [ ] Same-plane joint success < 70% (competition prediction)
- [ ] Orthogonal-plane joint success >= 90% (stacking prediction)
- [ ] PPL overhead compared between same-plane and orthogonal-plane conditions
- [ ] IDEA_TABLE.md row E23 updated

---

## 11. Status Journal

- 2026-05-31 — Design doc created. Status: UNTESTED / PENDING. N16 (SUPPORTED:
  R²=0.997/0.81 for angular/radial) is the primary geometry grounding.
  Code needed: rotational injection (Angular Steering or Selective Steering
  implementation); plane identification from DiffMean. Blocked on rotational
  injection code.

---

## Addendum: Research-Scientist Critique

*Reviewer: SciCritic-C.*

### Prior plausibility

HIGH for the orthogonal-plane stacking prediction (follows from linear algebra).
MEDIUM for the same-plane competition prediction (the "double-count" argument
is qualitative; the magnitude of interference depends on the relative phase
of the additive and rotational edits).

### Confounds

1. **Alpha calibration:** if the additive and rotational alphas are not calibrated
   to produce equal solo efficacy, the same-plane competition may appear as
   one method dominating the other rather than mutual interference.
2. **Plane purity:** real behavior planes in LLMs are not exactly 2D; they are
   low-rank but not rank-1. The "same plane" condition is an approximation.

### Verdict

**NOVEL+TESTABLE** as a direct test of the 7-axis decision rule's plane-specificity
claim. If confirmed, this validates the entire competition/stacking decision matrix.

---

## Pseudocode & Methodology

This section specializes [`../METHODOLOGY.md`](../METHODOLOGY.md) to the plane-specific
composition test: add+rotate on the SAME plane (compete) vs on ORTHOGONAL planes
(stack). Status: `UNTESTED` — needs the `rotate` op wired into a multi-site injection
plus plane identification.

### 1. Steering-vector recipe (two planes from DiffMean)

```python
# Behavior directions are closed-form DiffMean (METHODOLOGY §1.3).
v1 = normalize(extract.build_vector_bank(model,tok,load_concept("refusal"),L)[L]["diffmean"])
v2 = normalize(extract.build_vector_bank(model,tok,load_concept("sentiment"),L)[L]["diffmean"])
assert abs(dot(v1, v2)) < 0.1                       # P1 ⊥ P2 confirmed before running
# P1 = span{v1, ...}  (refusal plane);  P2 = span{v2, ...} (sentiment plane)
```

### 2. Experiment procedure (the four conditions)

```text
# add op (METHODOLOGY §2): h' = h + alpha*v1
# rotate op (METHODOLOGY §2, norm-preserving Gram-Schmidt in the (h,v) plane):
#   e1 = unit(h); e2 = unit(v - (v·e1)e1);  h' = ||h||*(cos(alpha)*e1 + sin(alpha)*e2)
conditions = {
  add_solo_P1   : apply_operation(h, v1, "add",    alpha),
  rot_solo_P1   : apply_operation(h, v1, "rotate", theta),
  same_plane    : apply_operation( apply_operation(h, v1,"add",alpha), v1, "rotate", theta),  # both on P1
  ortho_plane   : apply_operation( apply_operation(h, v1,"add",alpha), v2, "rotate", theta),  # add P1 + rot P2
}
for cond in conditions:
  for seed in 1..3:
    measure: efficacy_refusal(P1), efficacy_sentiment(P2), PPL,
             geometry.offshell_displacement, geometry.angular_displacement   # METHODOLOGY §3
```

### 3. Measurement & decision rule

- PRIMARY metric: joint success as a fraction of the better solo method, per plane.
- Pre-registered FALSIFIER (§3): (a) same-plane achieves `>= 90%` of solo (no
  competition) ⇒ decision rule FALSIFIED; (b) orthogonal-plane achieves `< 90%`
  (no stacking) ⇒ FALSIFIED. Either invalidates the 7-axis plane rule.
- Secondary (§6): same-plane PPL overhead `[+0.5,+2.0]` (off-manifold double-count)
  vs orthogonal-plane `[0,+0.5]` (N16 radial/angular decoupling).

### 4. Where the code is / status

`add` and `rotate` both exist in `hooks.apply_operation`. MISSING: composing two
operations in one forward pass on configurable planes, plane identification from
DiffMean, and confirming rotation stays within P1. `UNTESTED`. (Note: full-vector
rotation was screened FALSIFIED for solo efficacy/PPL in E27/S-7 — here rotation is
used only as a same-plane competitor, not as the primary method.)

See [`../METHODOLOGY.md`](../METHODOLOGY.md) for the shared recipe.

---

## Provenance & Tracing

No experiments run yet — see this design doc's protocol (§7) for what would be run. Once a campaign logs rows for this hypothesis, re-run `scripts/build_provenance.py` to generate `hypotheses/PROVENANCE/E23.md`.
