# E43 — Steering Efficacy Degrades Gracefully Under Domain Shift

> **One-line claim:** Steering vectors extracted on one domain (e.g., news
> articles) retain measurable efficacy on out-of-distribution domains (e.g.,
> code, dialogue, medical text), with efficacy declining gradually as a
> function of distributional distance rather than catastrophically, implying
> that behavior directions generalise across domains at least partially.
>
> **Block:** F — Robustness, safety, and evaluation (E41-E50).
> **Primary axis:** A7 (HOW DERIVED — source/generalization).
> **Implementation status:** `o planned / UNTESTED`.

---

## 1. Motivation (>= 100 words)

Steering vectors are extracted from a specific contrast set (e.g., harmful
vs harmless news-style prompts), and their efficacy has been measured
primarily within the distribution of that contrast set. The practical
deployment question is whether the vector generalises: if I extract a
"refusal" direction on news-style prompts, will it still steer a model
toward refusal when the input is a code snippet, a medical query, or a
multi-turn dialogue? The linear representation hypothesis predicts partial
generalisation — if the "refusal" direction is a global semantic direction
in the residual stream, it should be activated (at lower magnitude) by any
input that has the relevant semantic property, regardless of surface domain.
The competing hypothesis is that the DiffMean extraction captures domain-
specific surface features along with the semantic direction, and those
surface features are what drive the steering effect; changing the domain
removes the surface features and causes the efficacy to drop catastrophically
(the vector fires on a direction the new-domain inputs don't activate).
The practical value of this distinction is large: if efficacy degrades
gracefully, a single well-extracted vector can be used across deployment
domains with calibrated confidence intervals; if degradation is
catastrophic, domain-specific extraction is required. The correlation between
domain distributional distance (measured by perplexity of the new domain
under the model finetuned on the extraction domain, or by embedding cosine
between domains) and efficacy degradation is the key falsifiable prediction:
graceful degradation means a monotone, gradual relationship; catastrophic
degradation means a step-function near the extraction distribution boundary.

---

## 2. Formal Hypothesis (>= 50 words)

**H:** For the refusal and sycophancy behavior directions extracted on the
CAA/Sorry-Bench contrast set (news/instruction style), efficacy measured on
out-of-distribution domains (code, medical, creative writing, formal dialogue)
will decline as a smooth function of embedding-space distributional distance
(cosine distance between domain centroids), with no step-function dropout
across the tested domains. Specifically, at distributional distance <= 0.5
(moderate shift), efficacy retention will be >= 50% of in-domain efficacy,
and no domain in the test suite will show zero efficacy (complete failure).

---

## 3. Falsifier (>= 30 words)

If any tested out-of-distribution domain shows complete steering failure
(< 20% efficacy retention) while others in the test suite show near-full
retention, OR if the efficacy-vs-distance relationship is step-function
(high efficacy for distance < threshold, zero efficacy for distance >
threshold), the gradual-degradation hypothesis is DISCARDED (Status
`x disproved`).

---

## 4. Citations (Citation-Rigor format, >= 80 words)

```
Rimsky, Nina, et al. 2023 'Steering Llama 2 via Contrastive Activation
Addition' arXiv:2312.06681 — CAA; the original behavior-vector extraction
methodology; cross-domain efficacy is discussed informally but not
systematically measured; E43 provides the systematic domain-shift study.

Zou, Andy, et al. 2023 'Representation Engineering: A Top-Down Approach
to AI Transparency' arXiv:2310.01405 — RepEng / DiffMean; its linear
representation hypothesis predicts domain-general encoding of semantic
directions, which is the mechanism for gradual rather than catastrophic
domain-shift degradation.

Zhong, Zeping, et al. 2025 'AxBench: Steering LLMs? Benchmarks Matter'
arXiv:2501.17148 — AxBench; measures steering efficacy across a diverse
concept set; the concept diversity implicitly tests domain generalization
but does not systematically vary distributional distance; E43 provides
the controlled version.

Korznikov, et al. 2025 'The Rogue Scalpel: Activation Steering Compromises
LLM Safety' arXiv:2509.22067 — Rogue Scalpel; the poor cross-prompt
generalisation finding (F4) is relevant: if steering is domain-specific,
then the universal attack (F5) — which averages across many prompts —
may specifically exploit domain-general directions; E43 measures the
generalizability of intended-behavior directions, as F4 measured attack
directions.
```

---

## 5. Mechanism

The mechanism for gradual degradation rests on the linear representation
hypothesis: if a behavior direction v is a global semantic axis in the model's
representation space, it should be activated (at some magnitude) by any
input with the relevant semantic content, regardless of surface domain. The
activation h(input, domain) will have a projection onto v that reflects the
semantic relevance of the input to the behavior:

    <h(input, domain), v> ≈ alpha_domain * semantic_relevance(input)

where alpha_domain is a domain-specific scaling factor (< 1 for OOD domains).
The steering effect at matched fractional alpha should then scale with
alpha_domain: higher-distance domains have lower alpha_domain and thus lower
apparent efficacy at the same fractional injection.

The distributional distance proxy: use the cosine distance between the
centroid of in-domain activations (at the injection layer) and the centroid
of out-of-domain activations. This measures how far the new domain's
representation is from the extraction distribution in activation space,
which is the relevant distance for predicting steering vector transfer.

Catastrophic degradation would occur if the extraction domain's contrast
set captures a surface-specific direction (e.g., "legal disclaimer
language" appears in harmful but not harmless news) rather than a semantic
direction. A surface-specific direction would have near-zero projection
onto out-of-domain inputs, causing step-function efficacy loss.

---

## 6. Predicted Delta

| Domain | Distributional distance (approx) | Efficacy retention |
|---|---|---|
| In-domain (news/instruction) | 0.0 | 1.00x (reference) |
| Formal dialogue | 0.1-0.2 | 0.80-0.95x |
| Creative writing | 0.2-0.3 | 0.65-0.80x |
| Medical text | 0.3-0.4 | 0.55-0.70x |
| Code (Python) | 0.4-0.6 | 0.40-0.60x |

No domain at zero efficacy (all >= 0.40x retention); smooth monotone
relationship between distance and retention.

---

## 7. Protocol

### 7.1 Primary experiment

- Model: Gemma-2-2B-it (4-bit), RTX 4090.
- Behaviors: refusal and sycophancy (DiffMean from CAA/Sorry-Bench pairs,
  news/instruction-style extraction).
- Test domains: (i) in-domain news/instruction; (ii) formal dialogue
  (DailyDialog); (iii) creative writing (ROCStories or similar); (iv)
  medical text (MedQA); (v) code (Python docstrings/comments).
- Distributional distance: compute embedding-space cosine distance between
  the centroid of activation h at the injection layer for each domain's
  test set and the extraction domain's test set centroid.
- Eval: behavior-success rate (LLM-judge) on 50 examples per domain;
  WikiText-103 PPL (domain-specific: compute PPL on domain-appropriate
  text, not just WikiText); MMLU-500 (model-wide, not domain-specific).
- Regression: fit efficacy vs distributional distance; report R^2 and
  whether the fit is significantly better linear (gradual) than step-
  function.
- Seeds: 3 (screening), 7 for rung-3.

### 7.2 Where it shines

This experiment establishes the domain generalization profile of steering
vectors — a critical characterization for any deployment use case. The
gradual-vs-catastrophic distinction shapes whether domain-specific or
universal extraction is recommended.

---

## 8. Cross-references

- IDEA_TABLE.md Block F row E43.
- E1 (pair count knee): the extraction contrast set domain composition
  affects generalisation; E43 and E1 are complementary characterizations.
- E49 (meta-reproducibility audit): cross-domain claims in source papers
  may not reproduce on Gemma-2-2B; E43 provides the systematic test.
- Rogue Scalpel F4 (poor cross-prompt generalization of attack directions):
  if attack directions are domain-specific, so may be defense directions;
  E43 tests the defense direction's domain generalization.
- arXiv:2312.06681 (CAA): extraction methodology.
- arXiv:2501.17148 (AxBench): multi-concept evaluation across diverse
  inputs — the de facto cross-domain benchmark in the steering literature.

---

## 9. Committee Q&A

**Q: Isn't "distributional distance" undefined without a specific embedding
model and metric?**

> The embedding uses the model's own activation centroid at the injection
> layer (not an external embedding model). This is the natural distance
> in the space where the steering vector operates, and is directly
> computable from the model's forward pass on each domain's test set.

**Q: What if the behavior concept is domain-specific by nature (e.g.,
"refusal" means different things in code vs dialogue)?**

> The judge prompt is adapted to each domain (e.g., for code: "does the
> model decline to generate harmful code?"; for dialogue: "does the model
> decline to comply with the harmful request?"). The semantic target is
> the same (refusal), measured in a domain-appropriate way.

**Q: How many examples per domain are enough for a meaningful efficacy
estimate?**

> 50 examples per domain gives a 95% CI of approximately ±14% for a rate
> around 70%, which is sufficient for the gradual-vs-catastrophic
> distinction. If efficacy drops from 80% (in-domain) to 30% (OOD) vs
> to 60% (OOD), the CI widths of ±14% are sufficient to distinguish
> the two scenarios.

---

## 10. Verification checklist

- [ ] Domain-specific test sets assembled (50 examples each; 5 domains;
      domain labels verified by a secondary judge).
- [ ] Activation centroids computed at injection layer for each domain.
- [ ] Distributional distance metric defined and implemented (cosine of
      centroid activations; normalized by ||centroid||).
- [ ] LLM-judge adapted per domain with domain-appropriate refusal/
      sycophancy detection prompt; calibrated on a 25-example human-
      annotated slice per domain.
- [ ] Regression (efficacy vs distance) reported with R^2 and 95% CI.
- [ ] Rung-3 gate: n >= 7, Wilcoxon + bootstrap CI (Holm corrected).
- [ ] IDEA_TABLE.md row updated.

---

## 11. Status Journal

- 2026-05-31 — Created from corpus block F, hypothesis E43.
  Status: `o UNTESTED`. Theoretically motivated by the linear
  representation hypothesis and the practical deployment question of
  domain generalisation. No prior screening run. Dependency: domain-
  specific test sets, LLM-as-judge with domain-adapted prompts, multi-
  domain activation centroid computation.

---

## Addendum: Research-Scientist Critique

*Reviewer: SciCritic-F (domain-generalisation + evaluation specialist).*

### Prior plausibility
**MEDIUM.** The gradual-degradation prediction is the optimistic version
of domain generalisation. The pessimistic prediction (catastrophic degradation
for code, which is very far from news in activation space) is equally
plausible. The outcome will depend heavily on how universal vs domain-specific
the concept directions are in Gemma-2-2B.

### Mechanism scrutiny
The mechanism (alpha_domain scaling of the semantic projection) is a specific,
falsifiable form of domain generalisation. However, the distributional
distance metric (centroid cosine) may not capture the relevant distance:
two domains with similar centroid activations might still cause different
steering effects if the variance structure differs.

### Confounds
1. The judge calibration must be domain-specific; a judge calibrated on
   news-style prompts may have lower accuracy on code-domain prompts,
   inflating apparent efficacy loss.
2. PPL on domain-specific text is affected by the domain itself (code has
   high PPL on a general-purpose model); the coherence metric must be
   computed relative to an unsteered baseline on the same domain text.

### Expected effect size
My prior: code domain shows 30-50% efficacy retention (larger drop than
the 40-60% predicted), because code activations are geometrically very
different from news activations for Gemma-2-2B. The creative writing and
dialogue domains may show 65-80% retention as predicted.

### Verdict
**TESTABLE + DEPLOYMENT-RELEVANT** — A systematic domain-shift
characterisation that is currently absent from the steering literature.
Even a negative result (catastrophic degradation for code) is highly
informative for practitioners. Strongly recommend running this experiment
early in the program.
