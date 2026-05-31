# E11 — OR-Gate Coverage vs N: Linear Scaling with Bounded False Refusal

> **One-line claim:** OR-composition of N condition vectors scales coverage
> linearly while harmless-refusal stays flat up to N ~ 5, then leaks.
>
> **Source design space:** Block B — Conditional / Gated Steering (E9–E16).
>
> **Implementation status:** `PENDING — UNTESTED`. Requires E9 CAST pipeline,
> E10 orthogonality data, and multi-category evaluation set. Not yet built.

---

## 1. Motivation (>= 100 words)

Conditional safety steering in practice must cover multiple harm categories
simultaneously. The simplest compositional strategy is OR-gating: the behavior
vector fires if ANY of the N category condition vectors exceeds its threshold.
Under ideal conditions (near-orthogonal condition vectors as tested in E10),
OR-gating should add coverage for each new category without increasing false-
refusal on benign inputs — each gate fires only on its own relevant inputs,
and the gates do not interact. But as N grows, two failure modes emerge:
(a) even near-orthogonal vectors accumulate small cross-category cosine mass,
and the union of N misfires grows; (b) the condition vector extraction may
become noisy for rarer harm categories with fewer training examples, producing
lower-fidelity gates that trigger on benign edge cases. Understanding the
N at which false-refusal begins to grow — the leakage threshold — is essential
for deploying a realistic multi-property safety stack. This experiment establishes
the OR-gating capacity of the CAST architecture on Gemma-2-2B and provides
the N * value below which multi-category conditional steering is safe.

---

## 2. Formal Hypothesis (>= 50 words)

Because near-orthogonal condition vectors (per E10: |cos| < 0.3) fire
independently on their respective harm-relevant inputs, OR-composing N such
vectors should add approximately one coverage unit per new category (linear
scaling) while the false-refusal rate accumulates only at the product of
individual false-positive rates — negligible for N < 5 at individual rates
of < 3% (per E9). Beyond N ~ 5, residual cross-category cosine mass and
noisy extraction for rarer categories compound, causing harmless-refusal to
bend upward. Formal claim: for N in {1, 2, 3, 4, 5, 6, 7}, coverage grows
linearly (slope ~ 1 category per unit N) and harmless-refusal stays below 5%
for N <= 5 on Gemma-2-2B-it, then exceeds 5% at N >= 6. Control: a single
mega-condition vector (mean of all category vectors) which should plateau
in coverage and have a different (possibly higher) false-refusal profile.

---

## 3. Falsifier (>= 30 words)

If (a) coverage grows sub-linearly for N <= 5 (slope < 0.6 per unit), indicating
condition vectors are not independently covering their categories, OR (b) harmless-
refusal exceeds 5% at N = 3 (early leakage), this hypothesis is FALSIFIED and
OR-gating requires Gram-Schmidt pre-processing (E19) before deployment. If the
mega-condition control outperforms OR-gating at all N, the orthogonality
assumption underpinning independent gating is unjustified and FALSIFIED.

---

## 4. Citations (Citation Rigor >= 80 words)

```
Wu, Yuming, et al. 2024 arXiv 'Conditional Activation Steering: Concept-Level
Control via Conditional Vectors' (arXiv:2409.05907) — introduces OR/AND/NOT
logical composition of CAST condition vectors; the OR-composition architecture
is the direct object of this hypothesis; their "composing condition vectors"
result is the inspiration for the N-scaling prediction [NEEDS VERIFICATION
on Gemma-2-2B].

Tang, Haoran, et al. 2025 arXiv 'FineSteer: Fine-Grained Steering of Large
Language Models via Subspace-Guided Conditional Activation Steering'
(arXiv:2604.15488) — SCS energy-ratio gating is a stricter gate; if SCS gates
are more orthogonal than CAST cosine gates, OR-composing SCS gates may tolerate
larger N without leakage; E12 will determine which gate to prefer for multi-N
scaling.

Korznikov, A., et al. 2026 ICML 'The Rogue Scalpel: Activation Steering
Compromises LLM Safety' (arXiv:2509.22067) — establishes that accumulated
steering vectors (their "average 20 vectors" universal attack) cause leakage;
this experiment tests the analogous leakage phenomenon for condition vectors,
not behavior vectors, but the norm-budget argument (N5) applies to the logical
union of gate firings.

Wang, Kevin, et al. 2023 arXiv 'Interpretability in the Wild: a Circuit for
Indirect Object Identification in GPT-2 Small' (arXiv:2211.00593) — motivates
the circuit-level independence of distinct behavioral directions; referenced as
background for why near-orthogonal concepts should compose without interference.
```

---

## 5. Mechanism

### 5.1 OR-gating as logical union

Define the OR-gate for N categories as:

    gate_OR(h, {v_1,...,v_N}, {theta_1,...,theta_N}) =
        max_i [ cos(h, v_i) > theta_i ]   (fires if any condition is met)

Coverage(N) = P(gate_OR fires | harmful input from any of N categories)
FalseRefusal(N) = P(gate_OR fires | benign input)

Under independence (near-orthogonal v_i, consistent with E10):
    FalseRefusal(N) = 1 - prod_i(1 - FR_i)  ≈ N * FR_single  for small FR_i

At FR_single ~ 2% (from E9), N=5 gives FalseRefusal ~ 10% — exceeding 5%
at N=5. This predicts the leakage onset at N = 3-4, slightly earlier than
the corpus estimate of N ~ 5. The pre-registration takes N ~ 5 from the
corpus but the mechanism predicts leakage may appear at N = 3-4.

### 5.2 Mega-condition control

The mega-condition vector is the (normalized) mean of all N category vectors:

    v_mega = normalize( sum_i v_i )

For near-orthogonal vectors, v_mega is a direction in the N-dimensional
subspace of the category vectors, capturing their common "harm" component.
It may achieve higher coverage on multi-category harmful inputs but also
fires on benign inputs that project onto the shared component.

---

## 6. Predicted Delta

| Metric | Predicted value | Rationale |
|---|---|---|
| Coverage slope (N = 1–5) | ~0.85 per unit N | Each new category adds ~85% of its solo coverage |
| Coverage slope (N = 5–7) | ~0.50 per unit N | Saturation + diminishing new-category coverage |
| FalseRefusal at N=1 | ~ 2% | E9 target |
| FalseRefusal at N=3 | ~ 6% | Accumulation of FR_single |
| FalseRefusal at N=5 | ~ 10% | Above 5% threshold (leakage onset) |
| FalseRefusal (mega-condition) | ~ 5-15% | Higher than N=2-3 OR, lower than N=7 OR |
| Coverage (mega-condition) | ~ 70-80% across all N categories | Shared direction captures common harm, misses category-specific |

Pre-registered; [NEEDS VERIFICATION] on Gemma-2-2B.

---

## 7. Experimental Protocol

### 7.1 Primary experiment

- **Model:** Gemma-2-2B-it, 4-bit; smoke on Gemma-3-1B-it
- **Categories:** select N categories from a harm taxonomy covering
  {hate, self-harm, illegal-advice, adult-content, personal-attack,
  medical-misinformation, financial-fraud} — N in {1, 2, 3, 4, 5, 6, 7}
- **Condition vectors:** use E10 pipeline; categories ordered by decreasing
  solo coverage so N grows by adding the next-best category
- **Evaluation:** 100 harmful prompts per category + 200 benign prompts
  (AlpacaEval), fixed seed
- **Metrics:** Coverage(N), FalseRefusal(N), and Coverage(mega-condition)
  at matched theta_c
- **Seeds:** 3 (screening); promote to 7 if N-leakage onset is unclear

### 7.2 Where it shines

This experiment shines on harm categories that are semantically well-separated
(confirmed by E10 cosine data) and have abundant contrast examples. The
leakage threshold N* is most clearly visible when individual FalseRefusal is
low (< 2%), making the compound rate increase unambiguous.

---

## 8. Cross-References

- **E9** (CAST gate): single-category baseline; gates this experiment's N=1 point
- **E10** (category orthogonality): pre-condition; if |cos| > 0.3 for any pair,
  Gram-Schmidt must be applied before OR-gating
- **E19** (Gram-Schmidt): remediation if E11 leakage appears too early
- **E15** (learned gate): the alternative to fixed-theta OR-gating that may
  tolerate larger N
- **N5** (norm budget): OR-gating accumulates gate-firings, each spending
  budget when the behavior vector fires; coverage * alpha * ||v|| grows
- **IDEA_TABLE.md** Block B row E11

---

## 9. Committee Q&A

**Q: Why not just use a multi-class classifier trained on all N categories?**

> A trained multi-class classifier is AXIS 7 (source) = learned/discriminative,
> which E15 tests. The CAST OR-gate is mechanism-grounded (same activation-space
> pipeline as behavior extraction), requires no training labels beyond the
> positive/negative split, and composes naturally with the 7-axis framework.
> Whether it beats a classifier is E15's question.

**Q: Doesn't the leakage onset at N ~ 3-4 (mechanism prediction) conflict with
the corpus estimate of N ~ 5?**

> Yes — this is a genuine tension in the pre-registration. The mechanism predicts
> leakage at N ~ 3-4 (from FR_single ~ 2% and independence). The corpus estimate
> of N ~ 5 may be optimistic. The pre-registered falsifier uses N = 5 as the
> boundary (corpus value) but the experiment will report the actual leakage onset
> wherever it occurs. If it is at N = 3, the hypothesis is PARTIALLY SUPPORTED
> (leakage confirmed, onset earlier than corpus estimate).

---

## 10. Verification Checklist

- [ ] E9 gate pipeline operational (prerequisite)
- [ ] E10 cosine matrix computed (prerequisite; if |cos| > 0.3, add GS step)
- [ ] 7 harm categories defined; 100 harmful + 200 benign prompts assembled
- [ ] Coverage and FalseRefusal logged for N in {1,...,7} and mega-condition
- [ ] Leakage onset N* identified and reported in EXPERIMENT_LEDGER.md
- [ ] Coverage slope computed (linear regression N=1–5 and N=5–7)
- [ ] IDEA_TABLE.md row E11 updated

---

## 11. Status Journal

- 2026-05-31 — Design doc created. Status: UNTESTED / PENDING. Requires:
  (a) E9 CAST pipeline (blocked); (b) E10 cosine matrix (blocked on E9);
  (c) multi-category harm dataset (7 categories × 100 prompts).

---

## Addendum: Research-Scientist Critique

*Reviewer: SciCritic-B.*

### Prior plausibility

MEDIUM. The independence assumption (OR-gate firings are independent) is the
load-bearing claim. It holds if and only if E10 confirms |cos| < 0.3. The
mechanism prediction of leakage at N = 3-4 is more conservative (and more
credible) than the corpus estimate of N = 5.

### Confounds

1. **Benign prompt diversity:** if the 200 benign prompts are not representative
   of OOD inputs, the FalseRefusal estimate will not generalize (E15 addresses).
2. **Category ordering:** coverage(N) depends on which categories are added in
   which order; the protocol addresses this by ordering on solo coverage.

### Verdict

**NOVEL+TESTABLE.** The N-scaling curve is a direct design constraint for
multi-safety-vector deployment and has not been reported for Gemma-2-2B.
The independence assumption is falsifiable and the cost is low.
