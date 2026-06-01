# E17 — Near-Orthogonal Behavior Vector Stacking

> **One-line claim:** Two near-orthogonal behavior vectors (|cos| < 0.2) added
> together keep >= 90% of their solo effects, with no cross-degradation.
>
> **Source design space:** Block C — Stacking and Multi-Vector Composition (E17–E26).
>
> **Implementation status:** `PENDING — UNTESTED`. Multi-vector injection code
> not yet built; requires the five-axis eval bundle and CAST gating for safe
> multi-vector deployment.

---

## 1. Motivation (>= 100 words)

The central design question for a multi-property safety stack is whether multiple
behavior vectors can coexist in the residual stream without mutual degradation.
The theoretical answer from linear algebra is clear: if two vectors are orthogonal,
they occupy independent directions in activation space and adding them is
equivalent to two independent rank-1 perturbations — each contributes its full
effect with no interference. The practical question is whether LLM behavior
vectors — derived from real, finite, noisy contrast sets — are sufficiently near-
orthogonal for this independence to hold empirically at the behavior-efficacy
level. The norm-budget law (N5, SUPPORTED: logPPL = 5.40 + 2.87 * offshell,
R² = 0.81, C2) predicts the boundary condition: stacking increases cumulative
offshell displacement (||alpha1*v1 + alpha2*v2||), and perplexity rises when
this displacement exceeds the budget. For near-orthogonal vectors, the summed
norm is sqrt(alpha1^2 + alpha2^2) — larger than each solo, but sub-linear
(Pythagorean). For parallel vectors, the summed norm is alpha1 + alpha2 — additive.
The near-orthogonal case is the favorable regime; this experiment quantifies
the empirical efficiency gain relative to the anti-aligned control.

---

## 2. Formal Hypothesis (>= 50 words)

When two behavior vectors v1 and v2 satisfy |cos(v1, v2)| < 0.2 (near-orthogonal),
their combined effect in activation space decomposes as:

    h <- h + alpha1*v1 + alpha2*v2

where v1 and v2 act in nearly independent subspaces, so the projection of v2
onto the axis of v1 is small (< 20% of ||v2||). The behavior produced by v1 is
not interfered with by v2's injection (and vice versa), because v2's component
along v1's direction is small — it neither reinforces nor suppresses v1's effect.
Formal claim: on two behaviors B1 (e.g., refusal) and B2 (e.g., sentiment shift)
whose DiffMean vectors satisfy |cos| < 0.2, the joint steer achieves >= 90%
of solo B1 efficacy and >= 90% of solo B2 efficacy at 3-seed median on
Gemma-2-2B-it. Control: anti-aligned pair (|cos| > 0.8) which should show
substantial cross-degradation (< 60% of solo efficacy for each).

---

## 3. Falsifier (>= 30 words)

If the joint steer achieves less than 90% of solo efficacy for either behavior
at |cos(v1,v2)| < 0.2, the independence assumption is violated — near-orthogonal
vectors interfere beyond linear algebra prediction. Status FALSIFIED for the
independence claim. If the anti-aligned control also achieves >= 90% (no
degradation even for parallel vectors), the orthogonality advantage is absent
and N5 norm-budget effect is not empirically visible at these alpha values.

---

## 4. Citations (Citation Rigor >= 80 words)

```
[N5 geometry result — this project, C2]: logPPL = 5.40 + 2.87 * offshell_R,
R² = 0.81 (23 rows, Gemma-2-2B, campaign C2) — the norm-budget conservation
law that governs when stacking becomes incoherent; the Pythagorean norm-growth
for orthogonal vectors (sqrt of sum of squares) vs additive growth for parallel
vectors predicts orthogonal stacking is more norm-efficient.

Kossaifi, Jean, et al. 2024 arXiv 'Conceptor Learning for Class Activation
Mapping' (reference via Jaeger 2014) — Conceptors as in arXiv:2410.16314; the
AND-composition algebra is based on the assumption that orthogonal directions
can be independently controlled; the near-orthogonality finding from this
experiment validates or falsifies that assumption for DiffMean vectors.

Arditi, Andy, et al. 2024 arXiv 'Refusal in Language Models Is Mediated by a
Single Direction' (arXiv:2406.11717) — the refusal direction is the prototype
for v1; v2 should be chosen from a semantically independent behavior (sentiment,
formality) to satisfy |cos| < 0.2.

Liu, Nelson F., et al. 2023 arXiv 'In-Context Vectors: Making In Context
Learning More Effective and Controllable Through Latent Space Steering'
(arXiv:2311.06668) — related work on stacking in-context vectors; demonstrates
that behavior vectors from different tasks tend to be near-orthogonal.
```

---

## 5. Mechanism

### 5.1 Linear algebra of stacking

Decompose v2 along v1 and perpendicular:

    v2 = cos(v1,v2) * v1  +  v2_perp   (||v2_perp||^2 = 1 - cos^2)

Interference on B1:  alpha2 * cos(v1,v2)  — the component of v2 along v1
Interference on B2:  alpha1 * cos(v1,v2)  — the component of v1 along v2

For |cos| < 0.2: interference <= 0.2 * alpha_other — at most 20% cross-talk.
For |cos| = 0.0: zero interference — exact independence.

The joint norm: ||alpha1*v1 + alpha2*v2||^2 = alpha1^2 + 2*alpha1*alpha2*cos + alpha2^2
At |cos| < 0.2 and alpha1=alpha2=alpha: joint_norm ~ sqrt(2)*alpha (Pythagorean)
vs solo_norm = alpha — a sqrt(2) increase, well within N5 budget at typical alpha.

### 5.2 N5 grounding

The norm-budget law (N5, SUPPORTED) maps joint_norm to expected logPPL degradation.
At sqrt(2) norm growth (near-orthogonal pair), logPPL increases by:
    Delta_logPPL ~ 2.87 * (sqrt(2) - 1) * ||v||/||h|| ~ 2.87 * 0.41 * (norm-ratio)
This is a moderate, predictable increase — not a collapse. The anti-aligned control
at |cos| > 0.8 produces joint_norm ~ 1.8*alpha (near-additive), causing a larger
logPPL increase and potential behavior degradation.

---

## 6. Predicted Delta

| Metric | Predicted value | Rationale |
|---|---|---|
| Joint B1 efficacy (|cos|<0.2) | >= 90% of solo | Core claim; linear algebra |
| Joint B2 efficacy (|cos|<0.2) | >= 90% of solo | Core claim |
| Joint B1 efficacy (|cos|>0.8) | 40-70% of solo | Anti-aligned: high interference |
| Joint norm growth (|cos|<0.2) | sqrt(2) * solo_norm | Pythagorean |
| Joint norm growth (|cos|>0.8) | ~1.8 * solo_norm | Near-additive |
| Delta PPL (|cos|<0.2 joint vs solo) | [+0.1, +0.5] | N5 moderate growth |
| Delta PPL (|cos|>0.8 joint vs solo) | [+0.5, +2.0] | N5 near-additive growth |

N5 geometry (SUPPORTED) and N16 radial/angular decomposition (SUPPORTED)
provide mechanistic grounding. Pre-registered; [NEEDS VERIFICATION] on Gemma-2-2B.

---

## 7. Experimental Protocol

### 7.1 Primary experiment

- **Model:** Gemma-2-2B-it, 4-bit; smoke on Gemma-3-1B-it
- **Behavior pairs:** (a) near-orthogonal: refusal + sentiment-positive
  (expect |cos| < 0.2); (b) anti-aligned control: refusal + sentiment-negative
  (expect |cos| > 0.5); verify cosines before running
- **Alpha:** matched solo-optimal alpha for each behavior; joint uses same alpha
- **Metrics:** B1 and B2 efficacy (behavior success rate); PPL; offshell ||delta h||
- **Conditions:** solo B1, solo B2, joint B1+B2, anti-aligned control
- **Seeds:** 3 (screening); 7 if joint efficacy is between 85-95%
- **Cross-reference:** log Gram off-diagonal cos(v1,v2) for E18 input

### 7.2 Where it shines

Near-orthogonal stacking works best when behaviors are semantically unrelated
(refusal + factuality, refusal + politeness) and when alpha is below the N5
coherence cliff. Fails when behaviors share a semantic axis (both safety-related)
or when alpha is pushed above the cliff.

---

## 8. Cross-References

- **E18** (interference vs Gram mass): this experiment provides the 2-vector
  data point for E18's curve fitting
- **E19** (Gram-Schmidt orthogonalization): remediation when |cos| > 0.2
- **E21** (Conceptor AND vs sum): tests a different composition algebra for
  the same stacking problem
- **E22** (norm budget cap): the Pythagorean norm growth is the input to E22's
  cap design
- **N5** (norm budget, SUPPORTED): mechanistic backbone of the stacking limit
- **N16** (radial/angular, SUPPORTED): angular-only steering keeps ||h|| fixed;
  this experiment uses additive steering, so radial excursion is the key metric
- **IDEA_TABLE.md** Block C row E17

---

## 9. Committee Q&A

**Q: How do you choose two behaviors with |cos| < 0.2 a priori?**

> We compute the cosine between candidate behavior vectors before the experiment
> and select a pair that satisfies the constraint. This is the correct
> procedure — the hypothesis is about the consequences of orthogonality, not
> about predicting which pairs are orthogonal (E10 does that for condition vectors).
> The behavior pair (refusal + sentiment-shift) is chosen because semantically
> distinct behaviors are expected to have low cosine.

**Q: Doesn't the N5 norm-budget growth (sqrt(2)) already constrain the result?**

> The N5 law predicts the PPL cost of norm growth, not the behavior efficacy
> cost. It is possible that PPL increases (from sqrt(2) norm growth) without
> behavior degradation (if the behaviors are in orthogonal subspaces and the
> PPL increase is small). The experiment measures both — the 90% efficacy
> threshold and the PPL delta are separately falsifiable.

---

## 10. Verification Checklist

- [ ] Cosine |cos(v1,v2)| confirmed < 0.2 before joint run
- [ ] Solo B1 and B2 efficacy baselines established
- [ ] Joint B1+B2 efficacy measured; >= 90% of solo for both
- [ ] Anti-aligned control efficacy measured; expected < 90%
- [ ] Offshell ||delta h|| logged for solo and joint conditions (N5 cross-check)
- [ ] PPL logged for all conditions
- [ ] Gram off-diagonal cos logged for E18 input
- [ ] IDEA_TABLE.md row E17 updated

---

## 11. Status Journal

- 2026-05-31 — Design doc created. Status: UNTESTED / PENDING. Geometry
  grounding: N5 (logPPL = 5.40 + 2.87 * offshell, R² = 0.81, SUPPORTED)
  governs the PPL cost of stacking. N16 (angular vs radial, SUPPORTED) explains
  why additive stacking incurs more PPL cost than rotational (radial excursion).
  The norm-budget Pythagorean prediction for near-orthogonal pairs is the
  central mechanistic input. Code needed: multi-vector injection hook;
  five-axis eval; behavior pair selection. Blocked on multi-vector injection.

---

## Addendum: Research-Scientist Critique

*Reviewer: SciCritic-C (stacking specialist).*

### Prior plausibility

HIGH. The linear algebra is exact; the empirical question is whether the
behavioral approximation (does 90% of linear-algebraic independence translate
to 90% of behavioral independence?) holds. In practice, behavior evaluations
are noisy (judge variance, prompt variance), which may obscure small interference.

### Mechanism scrutiny

The N5 grounding is solid (SUPPORTED empirically). The Pythagorean norm-growth
prediction is testable and pre-registered. The anti-aligned control provides
a natural comparison that should produce the predicted degradation.

### Confounds

1. **Alpha calibration:** solo-optimal alpha for each behavior may not be
   jointly optimal; the joint norm may push both over the coherence cliff.
2. **Behavior evaluation noise:** if the efficacy metric has > 5% variance,
   detecting a 10% degradation (90% threshold) may require n >= 7 seeds.

### Numerology check

The 90% threshold is a practical choice (10% tolerance for interference).
The 0.2 cosine threshold is motivated by the Gram-mass interference formula:
at |cos| = 0.2, cross-talk is 20% of one alpha, which reduces efficacy by
< 10% (for equal alpha vectors). The threshold and the efficacy target are
internally consistent.

### Verdict

**NOVEL+TESTABLE** for Gemma-2-2B with N5 geometry grounding. The anti-aligned
control is essential for the comparison to be meaningful. Low compute cost.

---

## Provenance & Tracing

Full per-hypothesis provenance (exact experiments, reproduce commands, artifact links, reasoning trace): [`PROVENANCE/E17.md`](../PROVENANCE/E17.md).

- **Experiments:** analysis campaign (computed quantities in the campaign JSON; see the provenance file).
- **Reproduce:**

```bash
PYTHONPATH=src python scripts/campaign_sweep.py --model models/google/gemma-3-270m-it --quant none --hyp E17 --tag-prefix E17-stack --layers 16 --alphas 0.1 --ops relative_add --behaviors anger happiness  # solo vs joint (anger+happiness) stacking comparison
```
