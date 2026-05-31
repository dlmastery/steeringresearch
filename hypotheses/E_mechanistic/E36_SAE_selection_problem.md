# E36 — SAE Feature Steering: A Selection Problem, Not a Representation Problem

> **One-line claim:** SAE-feature steering underperforms raw DiffMean only
> when features are naively selected; with output-score-based feature
> selection, SAE steering matches DiffMean — the AxBench tension is a
> SELECTION problem, not a representation problem.
>
> **Block:** E — Mechanistic and interpretability-guided (E34-E40).
> **Primary axis:** A7 (HOW DERIVED — source of vector).
> **Implementation status:** `~ SCREENING-SUPPORTED` (see Status Journal).

---

## 1. Motivation (>= 100 words)

The AxBench benchmark (arXiv:2501.17148, Stanford) compares steering
methods on concept incorporation: prompting, probing, SAE-feature steering,
and DiffMean (CAA-style). AxBench's headline finding is that SAE-feature
steering underperforms both prompting and DiffMean on most concepts. This
is surprising given the interpretability promise of SAE features: if
a feature direction is the "atomic" encoding of a concept, it should be
the most efficient steering vehicle. Two competing explanations exist:
(1) the REPRESENTATION account — SAE features are not the right
representation for control; the model stores behavior information in
multi-feature superpositions that DiffMean captures but single-feature
steering misses; (2) the SELECTION account — the right SAE features DO
encode the behavior, but naive selection (e.g., by reconstruction error
or activation frequency) picks the wrong features; selecting by causal
output effect (how much does steering this feature change the desired
output) would close the gap. Hypothesis E36 tests the selection account:
if output-score-selected SAE features match DiffMean, the AxBench finding
is a methodological artifact of naive selection, not a fundamental
limitation of sparse-feature representations. Our campaign C7->C8->C9b
already established a relevant screening result: at matched fractional
alpha (relative_add), DiffMean and PCA-top1 steer near-identically
(behavior +-0.02, PPL +-8%), consistent with E4's 0.99 cosine alignment.
This confirms that the "source" (DiffMean vs PCA) is not the variable;
the dominant factor is the parameterization (raw alpha vs fractional
displacement). The next question is whether output-score-selected SAE
features also land on this same near-equivalent plateau, or whether SAE
features have a genuinely lower ceiling even with proper selection and
matched displacement.

---

## 2. Formal Hypothesis (>= 50 words)

**H:** On Gemma-2-2B-it, SAE feature vectors (from GemmaScope) selected
by causal output score (the change in concept-correct generation probability
when steering that feature) will match DiffMean behavior-success rate within
10 percentage points at matched fractional alpha, while naively selected
SAE features (selected by feature activation frequency or by cosine to
DiffMean) will underperform DiffMean by >= 20 percentage points. The
performance gap reported in AxBench (arXiv:2501.17148) is therefore
attributable to naive feature selection rather than to an inherent
representational deficit of SAE directions.

---

## 3. Falsifier (>= 30 words)

If output-score-selected SAE features STILL underperform DiffMean by >= 20
percentage points (at matched fractional alpha, relative_add parameterization,
on at least two behaviors), the representation account is supported and the
selection account is DISCARDED. Status moves to `x disproved` (the AxBench
tension is a fundamental representation problem, not a selection artifact).

---

## 4. Citations (Citation-Rigor format, >= 80 words)

```
Zhong, Zeping, et al. 2025 'AxBench: Steering LLMs? Benchmarks Matter'
arXiv:2501.17148 — the primary benchmark whose findings this hypothesis
re-interprets; AxBench reports SAE-feature steering underperforming DiffMean
and prompting; this experiment tests whether output-score selection reverses
the gap.

Lieberum, Tom, et al. 2024 'Gemma Scope: Open Sparse Autoencoders
Everywhere All at Once' arXiv:2411.02193 (GemmaScope / SAE-TS) — the
pre-trained SAE feature dictionaries on Gemma-2-2B that supply the SAE
feature vectors; paper also discusses feature selection methods and
causal intervention frameworks.

Zou, Andy, et al. 2023 'Representation Engineering: A Top-Down Approach
to AI Transparency' arXiv:2310.01405 — CAA / DiffMean baseline; the
paper that established DiffMean as a competitive steering direction source;
AxBench compares against this method.

Korznikov, et al. 2025 'The Rogue Scalpel: Activation Steering Compromises
LLM Safety' arXiv:2509.22067 — relevant because SAE-feature steering of
benign features was the attack vector (F2); output-score selection should
preferentially select behavior-relevant features with lower Rogue-Scalpel
risk if those features are more causally concentrated.
```

---

## 5. Mechanism

SAE training minimises reconstruction error subject to a sparsity penalty.
This objective maximally recovers the INPUT distribution but is
indifferent to which features cause OUTPUT behavior changes. A behavior
direction v = sum_i c_i f_i in feature space; reconstruction-optimal
selection picks features by |c_i| (amplitude in the expansion), which
correlates with feature prevalence, not with how much steering feature i
changes the output behavior score. Output-score selection replaces the
prevalence criterion with a causal criterion: run a micro-sweep of one-
feature steering interventions on a held-out contrast set, rank features
by measured delta in concept-correct output probability, and pick the top-k.
If the AxBench gap is a selection artifact, top-k causal features will
closely approximate DiffMean's direction (high cosine) and its efficacy.

Screening evidence (C9b, FINDINGS.md S-9): DiffMean and PCA-top1 steer
near-identically at matched fractional alpha (behavior +-0.02, PPL +-8%).
Since cos(DiffMean, PCA-top1) = 0.99 (S-1, S-3), source equivalence holds
for these two directions. The C7 "differences" were pure norm artifacts.
The open question is whether a causally selected SAE direction lands in
the same equivalence class.

---

## 6. Predicted Delta

| Feature selection method | Behavior success (vs DiffMean) |
|---|---|
| DiffMean (baseline) | 1.00x (reference) |
| SAE naive (activation-freq selection) | 0.55-0.75x (reproduces AxBench gap) |
| SAE cosine-to-DiffMean selection | 0.80-0.90x |
| SAE output-score selection (causal) | 0.90-1.00x (prediction: closes gap) |

If the causal-selection SAE condition reaches 0.90-1.00x of DiffMean, E36
is supported. If it stays at 0.55-0.75x, E36 is falsified.

---

## 7. Protocol

### 7.1 Primary experiment

- Model: Gemma-2-2B-it with GemmaScope SAE (arXiv:2411.02193).
- Behaviors: refusal and sycophancy (CAA pairs); optionally formality.
- Feature selection conditions:
  (A) DiffMean direction (reference);
  (B) SAE: top-k features by activation frequency on behavior-positive
      examples (naive);
  (C) SAE: top-k features by cosine similarity to DiffMean direction;
  (D) SAE: top-k features by output-score (causal) — run N=50 single-
      feature steers on contrast set, rank by |delta behavior|.
- k: 1 feature (single best), 5 features (summed), 20 features (dense).
- Alpha: relative_add at alpha = 0.10 (primary), sweep {0.05, 0.10, 0.20}.
- Eval: behavior-success rate (LLM-judge on real generated text), WikiText-
  103 PPL, MMLU-500.
- Seeds: 3 (screening), 7 for rung-3.

### 7.2 Where it shines

This experiment directly interprets the AxBench benchmark finding with a
mechanistic decomposition of the selection step. It is the minimum-cost
test of the selection vs representation accounts and determines whether
SAE feature steering is salvageable with better methodology.

---

## 8. Cross-references

- IDEA_TABLE.md Block E row E36.
- E4 (DiffMean vs PCA cosine alignment): cos=0.99 implies source
  equivalence for same-family methods; E36 tests cross-family equivalence
  (DiffMean vs SAE).
- E37 (interpretability != controllability): if output-score-selected SAE
  features match DiffMean but activation-selected ones don't, then
  "interpretable" features (high activation) are not the "controllable"
  ones — directly supporting E37.
- N8 (causal-objective dictionary): E36 tests a heuristic version of N8's
  proposal; N8 trains a full dictionary with a causal objective; E36 just
  re-ranks an existing SAE dictionary.
- arXiv:2501.17148 (AxBench): the prior to be re-interpreted.
- arXiv:2411.02193 (GemmaScope): the SAE feature source.
- C9b / FINDINGS.md S-9: screening support for source equivalence at
  matched fractional alpha.

---

## 9. Committee Q&A

**Q: Doesn't output-score selection just recover DiffMean by another name?**

> Not necessarily. Output-score selection identifies individual SAE
> features that cause the behavior change; DiffMean is a dense direction
> that mixes all features active in the contrast set. If selection finds a
> small set of causally dominant features that together approximate
> DiffMean's direction, both accounts are true (representation is fine,
> selection matters). If the small set has low cosine to DiffMean but still
> achieves high efficacy, the accounts are separable.

**Q: The C9b result (DiffMean ~ PCA at matched fractional alpha) was on
Gemma-270m with synthetic mini-data. How confident are we it transfers?**

> This is a screening observation (FINDINGS.md S-9, n=1, not an external
> claim). The norm-artifact explanation (C7 raw-alpha vs C9b relative_add)
> is mechanistically robust, but the quantitative equivalence on Gemma-2-2B
> with real AxBench evals remains to be confirmed. E36's primary experiment
> on Gemma-2-2B is the required confirmation run.

**Q: What is the 'output score' in the output-score selection?**

> The delta in probability mass on the concept-correct token (or token set)
> after single-feature steering, measured on the contrast-pair held-out
> set. This is the AxBench concept-incorporation score adapted as a
> selection criterion rather than an evaluation criterion.

---

## 10. Verification checklist

- [ ] GemmaScope SAE loaded and feature steering confirmed (hook into Gemma
      residual stream at the SAE-associated layer).
- [ ] Output-score selection calibrated: N >= 50 single-feature interventions
      per behavior, ranked by |delta concept score|.
- [ ] All four conditions (A-D) run at matched fractional alpha.
- [ ] Behavior eval on real generated text (LLM-judge or AxBench scorer).
- [ ] JailbreakBench CR baseline = 0% before any SAE steering.
- [ ] Norm-artifact check: confirm all conditions are compared at matched
      ||delta h||, not matched raw alpha (C7 lesson).
- [ ] Rung-3 gate: n >= 7, Wilcoxon + bootstrap CI (Holm corrected).
- [ ] IDEA_TABLE.md status updated post-experiment; C9b screening observation
      noted in status journal.

---

## 11. Status Journal

- 2026-05-31 — Created from corpus block E, hypothesis E36.
  Status: `~ SCREENING-SUPPORTED` (partial).

  Screening evidence (FINDINGS.md S-9, C9b, n=1, Gemma-3-270m with
  synthetic mini-data): at matched fractional alpha (relative_add), DiffMean
  and PCA-top1 steer near-identically (behavior +-0.02, PPL +-8%). This is
  consistent with E36's selection-problem account — the C7 raw-alpha
  "differences" between DiffMean and PCA (a different-source comparison)
  were norm-parameterization artifacts, not real source effects. The 0.99
  cosine alignment (E4) implies that once displacement is controlled, source
  within the DiffMean/PCA family is irrelevant.

  Key caveat: C9b tested DiffMean vs PCA (two methods in the same family);
  E36's central claim is about DiffMean vs SAE features (different families).
  The SAE output-score-selection comparison has NOT been run. The C9b result
  supports the general "norm control resolves apparent source differences"
  narrative but does not directly confirm or deny the SAE-vs-DiffMean gap.

  Next step: run the four-condition experiment on Gemma-2-2B with GemmaScope
  SAE and real AxBench evals. The screening observation is filed as
  SCREENING-SUPPORTED in the DiffMean-vs-PCA sense only; the SAE-specific
  claim remains UNTESTED.

---

## Addendum: Research-Scientist Critique

*Reviewer: SciCritic-E (mechanistic interpretability, SAE specialisation).*

### Prior plausibility
**MEDIUM-HIGH.** The selection account is the most parsimonious explanation
of the AxBench gap. GemmaScope provides the right tool. The C9b screening
result supports the meta-claim that norm-controlled source comparisons
close apparent gaps; the extension to SAE features is the logical next step.

### Mechanism scrutiny
The output-score selection is theoretically sound. The practical challenge
is computational: N single-feature steers per behavior per model requires
O(N) forward passes for selection, which could be expensive at N=1000+
features. The protocol should pre-filter to the top-100 features by cosine
to DiffMean before the causal sweep to reduce compute.

### Confounds
1. Output-score selection uses the same contrast set for selection and
   (partially) evaluation — risk of over-fitting the selection to the eval.
   Require a held-out split: select on half the contrast set, evaluate on
   the other half.
2. The relative_add parameterization normalizes by ||h||, which varies
   between DiffMean (applied at residual stream norm) and SAE features
   (applied at the SAE decoder output, which may differ in scale). Verify
   that the "matched fractional alpha" is computed from the same ||h||
   reference point for all conditions.

### Numerology
The 10pp threshold for "matching" is reasonable but the 20pp threshold for
"naive underperformance reproduces AxBench" should be verified against the
actual AxBench numbers for the specific behaviors tested. Do not assume
the AxBench gap is exactly 20pp for Gemma-2-2B without checking.

### Expected effect size
My prior: output-score selection closes ~50-70% of the AxBench gap (not
100%). Some gap likely remains because SAE features reconstruct individual
dimensions while DiffMean aggregates across the full contrast distribution,
which is a smoother estimator in high dimensions.

### Minimum-distinguishing experiment
Run one behavior (refusal) with all four selection conditions on Gemma-2-2B
at alpha=0.10 (relative_add). Report efficacy ± standard error. This is
sufficient to determine whether the selection account is viable.

### Verdict
**TESTABLE + HIGH THEORETICAL VALUE** — Resolving the AxBench SAE-vs-
DiffMean tension has direct implications for interpretability-guided
steering. The screening support from C9b (norm control resolves apparent
source gaps) is encouraging but must be confirmed on the SAE-specific
comparison. Mark as SCREENING-SUPPORTED in the DiffMean-PCA sense only;
SAE-specific claim is UNTESTED.
