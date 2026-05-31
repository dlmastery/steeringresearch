# E12 — Energy-Ratio vs Cosine Gate PR-AUC

> **One-line claim:** Energy-ratio gating (FineSteer-SCS) gives sharper
> should-steer precision/recall than cosine-threshold CAST at equal compute.
>
> **Source design space:** Block B — Conditional / Gated Steering (E9–E16).
>
> **Implementation status:** `PENDING — UNTESTED`. Requires both CAST and SCS
> gate implementations and a shared evaluation set.

---

## 1. Motivation (>= 100 words)

CAST (arXiv:2409.05907) gates on cosine similarity between the current hidden
state and a condition vector: fire if cos(h, v_c) > theta_c. This is a
first-order test — it asks whether h has a positive projection onto v_c.
FineSteer's Subspace-guided Conditional Steering (SCS, arXiv:2604.15488) uses
an energy-ratio gate: fire if the energy of h projected onto the condition
subspace exceeds the energy projected onto a reference (neutral) subspace.
The energy ratio is a second-order quantity — it measures relative concentration
of activation energy in the condition subspace versus a baseline, which is more
robust to overall activation-norm variation. The central engineering question for
the CAST family is: which gate formulation offers the best precision-recall
trade-off for the should-steer decision? A higher PR-AUC at equal compute means
more harmful inputs correctly trigger the behavior vector, and fewer benign inputs
are incorrectly caught. This directly determines the safe operating point for
multi-category stacking (E11) and the capability tax (E16).

---

## 2. Formal Hypothesis (>= 50 words)

Because the SCS energy ratio normalizes out global activation magnitude (the
denominator is projection onto a neutral reference subspace), it produces a
gate signal that is invariant to the overall scale of h — removing one major
source of false positives in the cosine gate (prompts that happen to have
large ||h|| will artificially inflate cos(h, v_c) irrespective of their
harm-relevance). This invariance should improve precision at matched recall on
distribution-shifted inputs. Formal claim: on a held-out evaluation set of 200
prompts (100 harmful, 100 benign) drawn from the same distribution as training
but with OOD style variation, the SCS energy-ratio gate achieves strictly higher
PR-AUC than the CAST cosine gate at equal inference cost on Gemma-2-2B-it, with
the PR-AUC gap >= 0.05 at 3-seed median.

---

## 3. Falsifier (>= 30 words)

If SCS PR-AUC is not strictly higher than CAST PR-AUC at 3-seed median (PR-AUC
delta >= 0.05) on the held-out evaluation set, the hypothesis is FALSIFIED —
meaning cosine-threshold CAST is sufficient and SCS provides no gate-quality
benefit. If the two gates are within 0.03 PR-AUC (noise band from E3/E7
experience), status is NEAR-MISS, and E15 (learned gate) is the preferred
next step rather than SCS.

---

## 4. Citations (Citation Rigor >= 80 words)

```
Tang, Haoran, et al. 2025 arXiv 'FineSteer: Fine-Grained Steering of Large
Language Models via Subspace-Guided Conditional Activation Steering'
(arXiv:2604.15488) — introduces SCS energy-ratio gating; claims superiority
over cosine-threshold gates on in-distribution and OOD prompt sets [NEEDS
VERIFICATION on Gemma-2-2B]; the primary method under test.

Wu, Yuming, et al. 2024 arXiv 'Conditional Activation Steering: Concept-Level
Control via Conditional Vectors' (arXiv:2409.05907) — the CAST cosine-threshold
baseline; provides the gate formulation being compared.

Korznikov, A., et al. 2026 ICML 'The Rogue Scalpel: Activation Steering
Compromises LLM Safety' (arXiv:2509.22067) — establishes that unconditional
steering under any gate formulation can be defeated by large enough alpha;
gate precision matters because false-negatives leave harmful inputs unguarded
while false-positives add capability tax; their JailbreakBench protocol is used
as the harmful-prompt source.

Goh, Gabriel, et al. (anon) 2025 arXiv 'Selective Steering: Discriminative
Layer Identification for Targeted Activation Steering' (arXiv:2601.19375) —
discriminative-layer gate is a third gate architecture (tested in E14); comparing
all three gates (cosine, energy-ratio, discriminative-layer) on the same
evaluation set would give the full gate-comparison picture.
```

---

## 5. Mechanism

### 5.1 Cosine gate (CAST)

    gate_cos(h, v_c, theta_c) = [cos(h, v_c) > theta_c]
    cos(h, v_c) = (h · v_c) / (||h|| ||v_c||)

The cosine is scale-invariant in principle, but the empirical cos(h, v_c) is
sensitive to the effective direction of h — if h has large components in
directions unrelated to the condition, they contribute to the denominator and
dilute the gate signal.

### 5.2 Energy-ratio gate (SCS)

    E_cond = ||P_{S_c} h||^2   (energy in condition subspace S_c)
    E_ref  = ||P_{S_ref} h||^2  (energy in neutral reference subspace)
    gate_scs(h, S_c, S_ref, theta_e) = [E_cond / E_ref > theta_e]

The ratio E_cond / E_ref normalizes out prompt-level variation in activation
scale. S_ref is estimated from a neutral (general benign) prompt set. This
makes the gate invariant to global h magnitude — the key advantage over cosine.

### 5.3 PR-AUC measurement

Both gates are evaluated by sweeping their threshold (theta_c or theta_e) and
computing the precision-recall curve on the 200-prompt evaluation set. PR-AUC
is computed via sklearn-style trapezoidal integration. The comparison is at
equal compute: both gates use one dot product (or one projection) per token.

---

## 6. Predicted Delta

| Metric | Predicted value | Rationale |
|---|---|---|
| CAST PR-AUC (in-distribution) | [0.80, 0.90] | E9 target range for gate AUC |
| SCS PR-AUC (in-distribution) | [0.85, 0.93] | SCS normalization expected to help |
| PR-AUC gap (SCS - CAST) | [+0.03, +0.10] | Core claim; >= 0.05 is the falsifier threshold |
| PR-AUC gap (OOD evaluation) | [+0.05, +0.15] | SCS normalization helps more under distribution shift |
| Compute overhead (SCS vs CAST) | < 10% | Extra projection cost; comparable to CAST |

Pre-registered; [NEEDS VERIFICATION] on Gemma-2-2B.

---

## 7. Experimental Protocol

### 7.1 Primary experiment

- **Model:** Gemma-2-2B-it, 4-bit; smoke on Gemma-3-1B-it
- **Evaluation set:** 100 harmful (JailbreakBench) + 100 benign (AlpacaEval),
  held out from E9 training (no overlap)
- **OOD set:** 100 harmful + 100 benign from a different distribution
  (e.g., ToxiGen for HS, safety-aware QA for benign) — evaluates OOD robustness
- **Condition subspace (SCS):** top-k PCA components of harmful-prompt
  activations minus mean (k = 3); reference subspace from benign-prompt PCA
- **Condition vector (CAST):** DiffMean from E9 pipeline
- **Threshold grids:** theta_c in [0.0, 1.0] (100 points); theta_e in [0.5, 5.0]
  (100 points logarithmic) — sweeping full PR curve
- **Seeds:** 3

### 7.2 Where it shines

SCS benefits most when harmful and benign prompts differ in activation subspace
structure (not just mean direction) — this is the case for multi-category harm
where different sub-types cluster in the condition subspace but are not captured
by a single mean-difference direction.

---

## 8. Cross-References

- **E9** (CAST baseline): single-threshold gate being compared
- **E13** (condition layer latency): both CAST and SCS gates tested across layers
- **E14** (discriminative-layer gate): third gate architecture to compare
- **E15** (learned gate): learned logistic replaces both CAST and SCS
- **IDEA_TABLE.md** Block B row E12

---

## 9. Committee Q&A

**Q: Is SCS really a "gate" or is it a richer representation of the condition?**

> SCS is a gate in the sense that it produces a binary or soft should-steer
> decision. The energy-ratio is the gate signal, not a new direction for
> injection. It is AXIS 5 (condition) in a richer parameterization.

**Q: What if both gates fail on OOD?**

> Then E15 (learned logistic on multi-layer activations) is indicated — a richer
> gate that can generalize better via training on diverse examples.

---

## 10. Verification Checklist

- [ ] CAST gate PR-AUC computed on in-distribution and OOD sets (3 seeds)
- [ ] SCS subspace (k=3 PCA components) extracted; reference subspace extracted
- [ ] SCS gate PR-AUC computed on in-distribution and OOD sets (3 seeds)
- [ ] PR-AUC gap computed; compared to 0.05 falsifier threshold
- [ ] Compute overhead of SCS vs CAST measured (latency, VRAM)
- [ ] IDEA_TABLE.md row E12 updated; result logged in EXPERIMENT_LEDGER.md

---

## 11. Status Journal

- 2026-05-31 — Design doc created. Status: UNTESTED / PENDING. Code needed:
  (a) SCS subspace extraction and energy-ratio gate; (b) PR-AUC sweep harness;
  (c) OOD evaluation set. Blocked on E9 CAST pipeline.

---

## Addendum: Research-Scientist Critique

*Reviewer: SciCritic-B.*

### Prior plausibility

MEDIUM. The energy-ratio normalization is theoretically motivated but the
empirical benefit depends on how much of the false-positive rate in CAST cosine
gating is due to global activation-norm variation versus semantic misclassification.
If false positives are primarily semantic (benign prompts that are linguistically
similar to harmful ones), SCS offers no advantage over cosine.

### Confounds

1. **Subspace rank (k):** the SCS condition subspace rank k is a hyperparameter;
   low k may miss variation, high k may overfit. The protocol should sweep k.
2. **Reference subspace:** the neutral reference subspace affects the energy
   ratio denominator; a poorly chosen reference can inflate or deflate the gate.

### Verdict

**NOVEL+TESTABLE.** Direct gate comparison with a specific numeric criterion.
The OOD evaluation is the most informative condition — in-distribution
comparison may be insufficient to distinguish the two gate designs.
