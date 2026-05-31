# E21 — Conceptor AND-Composition vs Summed Vectors for 3+ Goals

> **One-line claim:** Conceptor AND-composition beats summed behavior vectors
> for >= 3 simultaneous goals at fixed coherence (matched PPL).
>
> **Source design space:** Block C — Stacking and Multi-Vector Composition (E17–E26).
>
> **Implementation status:** `PENDING — UNTESTED`. Requires Conceptor
> implementation (ellipsoidal activation-set representation) and
> multi-goal evaluation framework.

---

## 1. Motivation (>= 100 words)

When more than two behaviors must be jointly enforced, simple vector summation
runs into the interference and norm-budget problems documented in E17–E19.
The summed vector delta_h = sum_i alpha_i * v_i grows in norm and in off-
diagonal Gram mass as N increases, approaching the N5 coherence cliff. Conceptors
(arXiv:2410.16314) provide a mathematically principled alternative: represent
each behavioral goal as an ellipsoidal region in activation space (the set
of activations consistent with that behavior) and combine goals via logical
AND — the intersection of ellipsoids. The AND-combined conceptor encodes
"all behaviors simultaneously" without additive norm growth, because intersection
is a region-narrowing operation (it restricts the acceptable activation space,
not adds to the displacement). For >= 3 goals, the summed-vector approach
accumulates O(N) interference while Conceptor AND-composition narrows the
acceptable activation region without compounding norm. The corpus (steering-
stackable-vs-competing-analysis.md §2.2) reports that the AND-combined conceptor
outperforms mean-combined steering vectors for multi-goal composition [NEEDS
VERIFICATION on Gemma-2-2B]. This experiment provides the first falsifiable
test of this claim at 3+ goals on a specific model.

---

## 2. Formal Hypothesis (>= 50 words)

Because Conceptor AND-composition operates by restricting the acceptable
activation set to the intersection of per-behavior ellipsoids rather than
accumulating additive displacement, it avoids the N5 norm-budget violation
that degrades summed vectors at N >= 3. The AND-combined conceptor projects
the activation h onto the intersection region using a smooth operator:

    C_AND = (C_1 * C_2 * ... * C_N)^{1/N}  (approximate, see Jaeger 2014)

This projection does not increase ||h|| (it moves h within its current
norm shell) while restricting it to the region where all N behaviors are
simultaneously active. Formal claim: for N = 3 simultaneously enforced
behaviors, Conceptor AND achieves multi-goal success rate >= 10 percentage
points above summed vectors at matched PPL (within 0.5 logPPL of solo baseline),
at 3-seed median on Gemma-2-2B-it.

---

## 3. Falsifier (>= 30 words)

If Conceptor AND multi-goal success rate is not >= 10 pp above summed vectors
at matched PPL (within 0.5 logPPL of solo baseline) for N = 3 behaviors, the
hypothesis is FALSIFIED — the AND-composition algebra provides no practical
advantage over simple summation for Gemma-2-2B. If AND matches summed vectors
exactly (within 2 pp) at matched PPL, status is NEAR-MISS.

---

## 4. Citations (Citation Rigor >= 80 words)

```
Jaeger, Herbert, 2014 'Controlling recurrent neural networks by conceptors'
(arXiv:1403.3369); and Salin, Ethan, et al. 2024 arXiv 'Conceptors for
Activation Steering' (arXiv:2410.16314) — introduces Conceptors as ellipsoidal
representations of activation sets for LLMs; the AND-composition operator is
the primary mechanism under test; reports that "AND-combined conceptor
outperformed mean-combined steering vectors" [NEEDS VERIFICATION on Gemma-2-2B];
the algebraic formulation of AND-composition is inherited from Jaeger's matrix
framework.

[N5 geometry result, C2, this project]: logPPL = 5.40 + 2.87 * offshell,
R² = 0.81 — the norm-budget law predicts why summed vectors fail at N >= 3:
joint norm grows as sqrt(N)*alpha (orthogonal case) or N*alpha (parallel case),
both exceeding the budget B at large N. Conceptor AND avoids additive norm growth.

[N16 radial/angular result, C3b, this project]: angular predicts rotation logPPL
R² = 0.997; radial predicts additive logPPL R² = 0.81 — Conceptor AND is a
projection operation (angular, not radial); predicts lower PPL degradation than
additive summation at matched behavior change.

Arditi, Andy, et al. 2024 arXiv 'Refusal in Language Models Is Mediated by a
Single Direction' (arXiv:2406.11717) — refusal direction as one of the 3
behaviors; its Conceptor representation (the ellipsoid of "refusing activations")
should be included in the AND composition for multi-safety Conceptors.
```

---

## 5. Mechanism

### 5.1 Conceptor construction

For behavior B with activation set A_B = {h | h is an activation consistent
with B}:

    C_B = A_B * A_B^T * (A_B * A_B^T + alpha^{-2} * I)^{-1}
        where A_B is the matrix of activations on behavior-B prompts

C_B is a positive semi-definite matrix (conceptor); its eigenvalues are in [0,1].
The AND operator:

    C_AND(C_1, C_2) ~ (C_1^{-1} + C_2^{-1} - I)^{-1}   (soft AND approximation)

Applying C_AND to the current activation h:

    h_steered = C_AND * h   (matrix-vector product)

This constrains h to the intersection of the two behavior regions — no additive
displacement, no norm increase.

### 5.2 N5 comparison

Summed: delta_h = sum_i alpha_i * v_i; ||delta_h|| grows as O(sqrt(N))
Conceptor AND: delta_h = (C_AND - I) * h; ||delta_h|| depends on how far
h is from the intersection region — typically small for well-trained conceptors.

For N = 3 behaviors, the Conceptor AND approach is predicted to have lower
||delta_h|| (and therefore lower logPPL degradation) than summed vectors at
matched behavior efficacy.

---

## 6. Predicted Delta

| Metric | Predicted value | Rationale |
|---|---|---|
| Multi-goal success (Conceptor AND, N=3) | [60%, 80%] | AND-composition expected 10+ pp above summed |
| Multi-goal success (summed, N=3) | [45%, 70%] | N5 norm growth expected to degrade at N=3 |
| PPL overhead (Conceptor AND vs solo) | [0, +0.5] | Angular operation; lower than additive |
| PPL overhead (summed, N=3 vs solo) | [+0.5, +2.0] | N5 radial excursion at N=3 |
| Conceptor AND vs summed gap (success) | >= 10 pp | Core claim |

Pre-registered; [NEEDS VERIFICATION] on Gemma-2-2B. N5 and N16 (both SUPPORTED)
provide mechanistic grounding for the differential PPL prediction.

---

## 7. Experimental Protocol

### 7.1 Primary experiment

- **Model:** Gemma-2-2B-it, 4-bit; smoke on Gemma-3-1B-it
- **Behaviors:** 3 behaviors (refusal, sentiment-positive, formality-formal)
- **Conditions:** (1) solo each, (2) summed vectors, (3) Conceptor AND-composition
- **Metrics:** multi-goal success rate (all 3 behaviors achieved simultaneously);
  PPL; offshell ||delta h||; MMLU delta
- **Conceptor construction:** 50 activations per behavior at the injection layer
- **Seeds:** 3 (screening); 7 if gap is between 7-13 pp
- **Wall-clock:** ~2 h on 4090 (Conceptor construction: one forward pass per
  50 activations per behavior)

### 7.2 Where it shines

Conceptors work best when behaviors have distinct activation-set geometries
(different covariance structures) — the AND-composition of different ellipsoids
produces a non-trivial intersection. For behaviors with near-identical covariance
(e.g., two safety behaviors), the AND operator approximates identity.

---

## 8. Cross-References

- **E17** (near-orthogonal stacking): summed-vector baseline for 2 behaviors
- **E18** (Gram mass): summed-vector N=3 falls on the E18 interference curve;
  Conceptor AND should be below the curve (off-manifold less)
- **E22** (norm budget cap): capped summed vectors may approach Conceptor AND
  in coherence — comparison needed
- **N5** (norm budget, SUPPORTED): Conceptor AND is the "angular" approach;
  avoids the radial excursion that N5 penalizes
- **N16** (radial/angular, SUPPORTED): C_AND is a projection (angular, norm-
  preserving); predicts lower PPL than additive at matched behavior
- **IDEA_TABLE.md** Block C row E21

---

## 9. Committee Q&A

**Q: Aren't Conceptors computationally expensive for inference?**

> Applying C_AND to h is a matrix-vector product: O(D^2) per token.
> For D = 2304 (Gemma-2-2B hidden size), this is 5.3 M floating point ops
> per token — comparable to one transformer attention head. The Conceptor
> matrix is precomputed and cached. Compute cost is manageable on 4090.

**Q: Does Conceptor AND require special training or is it computed from activations?**

> Conceptors are computed directly from activation samples (no gradient descent).
> The construction requires only a forward pass to collect activations for
> each behavior — the same data needed for DiffMean. No training overhead.

**Q: What if the AND intersection is empty (no activations satisfy all 3 behaviors)?**

> If the 3 behavior activation sets are disjoint, C_AND will be near-zero
> (the null matrix), and no steering is possible — the hypothesis is
> INFEASIBLE for those behaviors. The experiment pre-checks that the
> pairwise AND intersections are non-empty before the 3-way AND.

---

## 10. Verification Checklist

- [ ] 50 activations per behavior collected at injection layer
- [ ] Conceptor matrices C_B1, C_B2, C_B3 constructed; eigenvalue spectra logged
- [ ] C_AND constructed; applied to test activations; verify h stays in-distribution
- [ ] Multi-goal success rate: Conceptor AND vs summed at matched PPL
- [ ] Gap compared to 10 pp threshold
- [ ] Offshell ||delta h|| for both conditions (N5 comparison)
- [ ] IDEA_TABLE.md row E21 updated

---

## 11. Status Journal

- 2026-05-31 — Design doc created. Status: UNTESTED / PENDING. Geometry
  grounding: N5 (SUPPORTED) and N16 (SUPPORTED) both predict Conceptor AND
  (angular/projection) will show lower PPL cost than summed vectors (radial/
  additive) at matched behavior efficacy. Code needed: Conceptor construction
  (matrix algebra, no backprop); AND operator implementation. Blocked on
  multi-vector evaluation framework from E17.

---

## Addendum: Research-Scientist Critique

*Reviewer: SciCritic-C.*

### Prior plausibility

MEDIUM. Conceptors are well-motivated theoretically and the corpus reports a
positive result [NEEDS VERIFICATION]. The main risk: the soft AND approximation
may not be accurate for LLM activation spaces, where activation distributions
are not Gaussian ellipsoids. The matrix inversion required for the AND operator
may be numerically unstable for near-degenerate conceptors.

### Mechanism scrutiny

The N5+N16 predictions are mechanistically sound: Conceptor AND is a projection
(angular), not an addition (radial), so it avoids the radial excursion that N5
penalizes. The prediction of lower PPL cost is well-grounded.

### Confounds

1. **Aperture parameter:** the Conceptor construction requires an aperture alpha
   hyperparameter; the result may be sensitive to this choice. The protocol
   should sweep alpha.
2. **Matched PPL:** "matched PPL" requires finding the Conceptor aperture that
   produces the same PPL as the summed-vector condition — this may be non-trivial.

### Verdict

**NOVEL+TESTABLE** with N5+N16 geometry grounding. The aperture sensitivity
is the main hyperparameter risk. The [NEEDS VERIFICATION] status of the corpus
claim makes this a genuine reproduction test.
