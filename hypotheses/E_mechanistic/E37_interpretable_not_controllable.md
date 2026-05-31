# E37 — Interpretable SAE Features Are Not the Controllable Ones

> **One-line claim:** Causally steerable SAE features are a small and
> largely non-overlapping subset of the highly interpretable SAE features;
> interpretability (reconstruction-based) does not imply controllability
> (causal output effect), and the two sets are empirically separable.
>
> **Block:** E — Mechanistic and interpretability-guided (E34-E40).
> **Primary axes:** A7 (HOW DERIVED — source) + A10 (IDENTIFIABILITY/gauge).
> **Implementation status:** `o planned / UNTESTED`.

---

## 1. Motivation (>= 100 words)

A central promise of sparse autoencoders (SAEs) is that their learned
feature directions are simultaneously interpretable (humans can assign
a concept to the feature) and causally relevant (steering the feature
changes model behaviour in the corresponding direction). These two
properties are often assumed to co-occur: if a feature has a clean
semantic label, it should be the right lever to pull for controlling that
behaviour. This assumption underlies the design of the AxBench evaluation
(arXiv:2501.17148) and much of the mechanistic interpretability agenda.
However, the assumption may be wrong. SAEs are trained with a reconstruction
objective, which rewards features that explain variance in the activation
distribution. A feature is "interpretable" in this context if it
systematically activates on a coherent category of inputs that humans can
name. But a feature is "controllable" if STEERING it (injecting along its
direction) causes the model to produce the labelled behaviour in its
outputs. These are logically distinct. A feature may activate on a concept
but be inert for output generation if the downstream circuit does not
read that feature direction. Conversely, a feature may be hard to label
but causally potent because it is positioned at a circuit bottleneck.
The hypothesis N8 in our first-principles corpus makes this prediction at
the structural level ("SAEs over-index on interpretable-but-inert
subspace"). E37 tests the empirical version: measure both interpretability
scores and causal effect sizes for a large set of GemmaScope features and
quantify the overlap of the top-K interpretable with the top-K controllable
sets. A low overlap directly motivates N8's proposal to train a causal-
objective dictionary, and explains why naive SAE-feature steering
underperforms DiffMean in AxBench (E36): DiffMean captures the controllable
subspace regardless of interpretability, while SAE feature selection by
human ratings or activation prominence captures the interpretable-but-
possibly-inert subspace.

---

## 2. Formal Hypothesis (>= 50 words)

**H:** For at least two behaviors (refusal, sycophancy) on Gemma-2-2B-it,
the Jaccard overlap between the top-50 features by interpretability score
(human label confidence, from Neuronpedia annotations or automated scoring)
and the top-50 features by causal steering effect size (output-score delta
per unit fractional displacement) will be less than 0.20 (Jaccard < 0.20).
This low overlap demonstrates that the interpretable-vs-controllable
decomposition is real and not an artifact of noisy scoring.

---

## 3. Falsifier (>= 30 words)

If Jaccard(top-50 interpretable, top-50 controllable) >= 0.35 on both
tested behaviors, the interpretability-implies-controllability assumption
holds empirically and the hypothesis is DISCARDED (Status `x disproved`).
A Jaccard >= 0.35 means the two sets are substantially co-populated and
the gap claimed in E37 does not exist at scale.

---

## 4. Citations (Citation-Rigor format, >= 80 words)

```
Bricken, Trenton, et al. 2023 'Towards Monosemanticity: Decomposing
Language Models With Dictionary Learning' (Anthropic, Transformer Circuits
Thread) — the foundational work establishing that SAE features are
interpretable; defines the interpretability scoring methodology that
provides the "interpretability" side of E37's comparison.

Lieberum, Tom, et al. 2024 'Gemma Scope: Open Sparse Autoencoders
Everywhere All at Once' arXiv:2411.02193 — GemmaScope; provides the
pre-trained SAE feature dictionaries on Gemma-2-2B with Neuronpedia
annotations that are the interpretability score source.

Zhong, Zeping, et al. 2025 'AxBench: Steering LLMs? Benchmarks Matter'
arXiv:2501.17148 — AxBench; the benchmark that measures causal steering
effect for SAE features among other methods; its underperformance of SAE-
feature steering relative to DiffMean is the empirical anomaly that E37
proposes to explain via the interpretability/controllability gap.

Yin, et al. 2024 'Selective Activation Steering' arXiv:2601.19375
(Selective) — shows that per-head targeted steering (implicitly selecting
by causal effect rather than interpretability) outperforms uniform
steering, consistent with the controllability-subset hypothesis.
```

---

## 5. Mechanism

The mechanism rests on a key observation about SAE training. The SAE
reconstruction objective is:

    min sum_i ( ||h_i - hat_h_i||^2 + lambda * ||f_i||_1 )

where f_i are feature activations. This rewards features that reconstruct
h but does not reward features whose directions causally influence output
probabilities. A feature may be "interpretable" (activates consistently
on a concept, easy to label) while being "inert" in the sense that no
downstream circuit reads that direction to produce concept-congruent
outputs. This happens when the SAE recovers a direction that is expressed
in the input distribution (hence reconstructable) but is processed by the
model in a nonlinear way that makes additive steering along that direction
ineffective.

The controllable subspace, by contrast, is spanned by directions where
the model's output circuit is approximately linear: steering along the
direction predictably shifts the output. These directions may be less
cleanly interpretable (they may be linear combinations of human-legible
concepts) but they are the load-bearing levers.

E37 measures this empirically by computing: (A) interpretability scores
from Neuronpedia annotations for a set of GemmaScope features; (B) causal
effect sizes from one-feature steering micro-sweeps (as in E36's output-
score selection); and computing the rank correlation and Jaccard overlap
between the two rankings.

---

## 6. Predicted Delta

| Metric | Predicted value |
|---|---|
| Jaccard(top-50 interp, top-50 causal) | < 0.20 (hypothesis prediction) |
| Spearman(interpretability rank, causal rank) | < 0.30 |
| Mean causal effect of top-50 interpretable features | significantly lower than top-50 causal |
| Mean interpretability of top-50 causal features | significantly lower than top-50 interpretable |

Expected finding: a two-cluster structure where "interpretable" and
"controllable" are near-orthogonal quality axes over the feature set.

---

## 7. Protocol

### 7.1 Primary experiment

- Model: Gemma-2-2B-it + GemmaScope SAE (arXiv:2411.02193).
- Feature pool: the top-500 GemmaScope features by activation frequency
  for each behavior layer (to keep compute manageable).
- Interpretability scores: Neuronpedia human annotations (confidence score
  / label clarity); or automated interpretability scoring via LLM-as-judge
  (describe the feature, rate label clarity on 1-5 scale).
- Controllability scores: single-feature steering micro-sweep on N=50
  contrast-pair prompts; measure delta in behavior-success rate; rank
  features by |delta|.
- Behaviors: refusal and sycophancy (CAA pairs; behavior-specific feature
  pools at the behavior-relevant injection layer from prior E2/layer sweep).
- Overlap metric: Jaccard(top-50 interpretable, top-50 controllable);
  Spearman rank correlation across all features.
- Seeds: 1 (screening rank correlation); 3 for Jaccard with confidence
  interval.
- Eval: JailbreakBench CR check (baseline 0%), MMLU, WikiText PPL (for any
  steering conditions run during micro-sweep).

### 7.2 Where it shines

This is the definitive test of the interpretability-implies-controllability
assumption that underlies most of the mechanistic-interpretability-guided
steering literature. If the overlap is low, it motivates a shift from
interpretability-first to controllability-first SAE training (N8).

---

## 8. Cross-references

- IDEA_TABLE.md Block E row E37.
- E36 (SAE selection problem): E37 explains WHY naive SAE selection fails —
  it selects by interpretability when controllability is the right criterion.
- N8 (causal-objective dictionary): the proposed solution to the E37 problem.
- E35 (sparse behavior vector): if behavior directions are sparse at the
  coordinate level, and SAE features are also sparse, the coordinate-sparse
  behavior basis may or may not align with the SAE basis — E37 addresses
  the alignment question.
- arXiv:2411.02193 (GemmaScope): feature source.
- arXiv:2501.17148 (AxBench): provides the causal-effect measurement
  methodology.

---

## 9. Committee Q&A

**Q: Interpretability scoring is subjective and method-dependent. Doesn't
this confound the comparison?**

> The experiment uses two interpretability scoring methods: (i) existing
> Neuronpedia human annotations (standardised); (ii) automated LLM-as-judge
> scoring (reproducible). If both yield low Jaccard overlap with the
> controllability ranking, the result is robust to scoring method. The
> Spearman correlation is also reported and is less sensitive to threshold
> choice than Jaccard.

**Q: Isn't "controllability" also noisy — depending on the behavior and the
contrast set used for the micro-sweep?**

> Yes; controllability scores are computed on a specific behavior and a
> specific contrast set. The experiment uses the same contrast set for both
> behaviors tested, and the behavior-specific nature of controllability is
> itself informative: a feature may be controllable for refusal but not
> sycophancy, further separating the two dimensions.

**Q: Could the low overlap be explained by sparse feature activation (most
features activate rarely) rather than by a controllability/interpretability
split?**

> The feature pool is pre-filtered to the top-500 by activation frequency,
> so rare-activation features are excluded. Within this pool, if low-overlap
> persists, it is not an activation-frequency artifact.

---

## 10. Verification checklist

- [ ] GemmaScope SAE loaded and feature activation indices confirmed for
      Gemma-2-2B at the injection layer.
- [ ] Neuronpedia annotations accessed or automated scoring protocol defined.
- [ ] Micro-sweep protocol: N >= 50 prompts, single-feature steering at
      relative_add alpha = 0.10, behavior-success delta computed per feature.
- [ ] Feature pool pre-filtered to top-500 by activation frequency.
- [ ] Jaccard and Spearman computed on same feature pool; threshold at k=50.
- [ ] Confidence intervals on Jaccard (3-seed bootstrap).
- [ ] IDEA_TABLE.md row updated post-experiment.

---

## 11. Status Journal

- 2026-05-31 — Created from corpus block E, hypothesis E37.
  Status: `o UNTESTED`. Theoretically motivated by N8 (controllability !=
  interpretability decomposition) and by the AxBench SAE underperformance
  finding. Dependency: GemmaScope SAE integration + Neuronpedia annotation
  access + micro-sweep infrastructure from E36.

---

## Addendum: Research-Scientist Critique

*Reviewer: SciCritic-E (SAE mechanistic interpretability specialist).*

### Prior plausibility
**MEDIUM-HIGH.** The interpretability/controllability gap is theoretically
predicted by the SAE training objective mismatch. The empirical question
is how large the gap is in practice on Gemma-2-2B. The AxBench results
suggest the gap is real but do not measure it directly.

### Mechanism scrutiny
The mechanism (reconstruction objective vs causal relevance) is sound.
The key subtlety is that "controllability" in this context measures additive
steering effect, which depends on the linearity of the readout circuit.
Features in nonlinear circuit positions will appear "uncontrollable" under
additive steering even if they are causally relevant to behavior via the
full nonlinear computation.

### Confounds
1. Interpretability and controllability may both be correlated with feature
   norm (larger-norm features are both easier to interpret and have larger
   steering effect). Partial out feature norm before computing the ranking.
2. The top-500 prefilter by activation frequency may systematically exclude
   rare but highly controllable features that are the actual load-bearing
   directions.

### Expected effect size
My prior: Jaccard(top-50 interp, top-50 causal) ~ 0.10-0.25. The gap
is likely real but the hypothesis's 0.20 threshold will be borderline.

### Verdict
**TESTABLE + HIGH IMPACT** — This experiment resolves a fundamental
question about the relationship between interpretability and control,
motivating the design of future SAE training objectives (N8). Even a
Jaccard of 0.20-0.35 (ambiguous zone) would be informative.
