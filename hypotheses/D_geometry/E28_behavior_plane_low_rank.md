# E28 — Behavior Plane Low-Rank

> **One-line claim:** The behavior plane for any steerable trait is low-rank:
> 2–3 principal components of the contrastive-difference space capture > 90% of
> steerable variance, and steering in this low-dimensional subspace is as effective
> as full-rank steering.
>
> **Source design space:** Block D — Geometry and Rotational Methods (E27–E33).
>
> **Implementation status:** UNTESTED. No screening data as of 2026-05-31.

---

## 1. Motivation (>= 100 words)

If behavior is a direction, the natural follow-up question is how many directions.
DiffMean extracts a single difference vector; PCA-top-1 extracts the principal
axis of the contrastive-difference matrix. Our screening data show that DiffMean
and PCA-top-1 are cosine-aligned at 0.994–0.996 across three models and two
architectures (S-1, S-3, S-8), which suggests that a single direction captures
the dominant steerable variance for a given trait. But this cosine alignment says
nothing about the tail: how much variance is explained by the second, third, and
fourth directions? If the behavior representation is genuinely intrinsically
one-dimensional, then the PCA spectrum should decay sharply after the first
component — a scree plot with a pronounced elbow at k=1 or k=2. If instead the
behavior is distributed across 5–10 dimensions of nearly equal variance, then
DiffMean is an arbitrary projection of a richer representation, and we are
potentially missing orthogonal directions that encode distinct aspects of the
behavior (e.g. the style vs the semantic content vs the refusal component of a
safety behavior). The low-rank hypothesis is important for three reasons. First,
if confirmed, it justifies the standard single-vector DiffMean / PCA-1 pipeline
and tells practitioners to stop at k=1. Second, if disconfirmed (the effective
rank is higher), it motivates low-rank steering methods (LoReFT, ReFT) that
explicitly operate in a k-dimensional subspace. Third, the effective rank of the
behavior plane provides a concrete estimate of the number of behavioral degrees
of freedom a model represents for a given trait, which is a basic empirical fact
about the model's representational geometry that the field currently lacks.
The CRH framework (arXiv:2605.01844) decomposes activation differences into
radial and angular components; the low-rank claim concerns the angular component
specifically — how many independent angular directions does a behavioral trait
occupy?

---

## 2. Formal hypothesis (>= 50 words)

For a set of >= 50 contrast pairs representing a single steerable behavioral
trait, the PCA of the contrastive-difference matrix (each row = positive_activation
minus negative_activation at the best steering layer) will show that the top-1
eigenvalue accounts for >= 60% of total variance, and that the cumulative variance
of the top-3 eigenvectors accounts for >= 90% of total variance. Furthermore,
a steering vector constructed as a weighted combination of the top-3 PCA components
will achieve behavior-success rate within 5% of the top-1 vector at matched
perplexity, establishing that the extra dimensions do not carry additional
steerable information. This claim must hold for at least 3 distinct behavioral
traits on at least 2 models.

---

## 3. Falsifier (>= 30 words)

If, on any single behavioral trait, the top-3 PCA components account for < 70%
of contrastive variance (i.e. the effective rank > 5), the low-rank claim is
rejected for that trait. If the full-rank steering vector achieves behavior success
> 10% higher than the top-3 steering vector at matched PPL on any tested trait,
the low-rank claim is practically rejected: the discarded dimensions carry
information the model uses for behavior. Additional falsifier: if the PCA scree
plot shows no visible elbow up to k=10, the behavior plane is diffuse and the
claim is rejected on structural grounds.

---

## 4. Citations (Citation Rigor format, >= 80 words)

```
Wu, Zhengxuan et al. 2024 'ReFT: Representation Fine-Tuning for Language Models'
(arXiv:2404.03592) — LoReFT explicitly operates in a low-rank subspace of the
residual stream; their empirical finding that rank-4 interventions often match
or beat full-rank finetuning is indirect evidence that behavior representations
are low-rank.

Park, Kiho et al. 2024 'Linear Representations of Sentiment in Large Language
Models' (arXiv:2310.15154) — demonstrates that sentiment is approximately linearly
represented along a single direction; the analogy motivates E28's low-rank
prediction for other behavioral traits.

Zou, Andy et al. 2023 'Representation Engineering: A Top-Down Approach to AI
Transparency' (arXiv:2310.01405) — RepE uses contrastive-activation PCA as its
core extraction method; the fact that PCA-top-1 is typically reported (not top-k)
is an implicit prior that k=1 suffices, which E28 makes explicit and testable.

Turner, Alex et al. 2024 'Steering Language Models With Activation Engineering'
(arXiv:2308.10248) — CAA uses a single DiffMean vector; paired with our E4
screening result (cos(DiffMean, PCA-top1) >= 0.994) this motivates asking whether
PCA-2 adds anything over DiffMean.

Cylindrical Representation Hypothesis: Gao et al. 2026 'The Cylindrical
Representation Hypothesis' (arXiv:2605.01844) — establishes that activation
differences decompose into radial (norm) and angular (direction) components;
E28 concerns the dimensionality of the angular component specifically.

Manifold Steering: Wurgaft et al. 2026 'Manifold Steering' (arXiv:2605.05115)
— frames behavior as lying on a manifold; E28 tests whether that manifold is
near-flat (low intrinsic dimension) in the angular direction for simple traits.
```

---

## 5. Mechanism

### 5.1 PCA of contrastive differences

For n contrast pairs (h_pos_i, h_neg_i) at layer L, form the matrix D ∈ R^{n x d}
where D_i = h_pos_i - h_neg_i. DiffMean = mean(D_i). The PCA of D produces
eigenvalues lambda_1 >= lambda_2 >= ... >= lambda_d. The explained variance ratio
at rank k is EVR(k) = sum(lambda_1..k) / sum(lambda_1..d). The DiffMean direction
aligns with PCA-1 at cos >= 0.994 (our data), so PCA-1 ≈ DiffMean normalized.

If EVR(1) >= 0.60 and EVR(3) >= 0.90, the behavior is intrinsically 1-to-3-
dimensional. This means the residual d - 3 dimensions of D are noise (sampling
variance from the finite contrast set) rather than signal.

### 5.2 Why this would be true

Language models are trained to represent behaviors as linear features to enable
the superposition of many behaviors in d dimensions (Johnson-Lindenstrauss allows
exp(d) near-orthogonal features). Each feature occupies approximately 1 direction.
A simple behavioral trait (e.g. "respond in French" or "be helpful vs refuse")
is encoded as a single linear feature, hence low rank. More complex behaviors
(e.g. "be creative AND safe AND factual") may occupy a small number of dimensions.

### 5.3 Why this might be false

The contrast set may include multiple sub-behaviors conflated in one label.
"Refusal" behavior, for example, conflates detection of harm (an input-sensitive
feature), the decision to refuse (a behavioral direction), and the style of
refusal (polite vs terse). Each sub-behavior may have its own direction, pushing
effective rank above 3. Similarly, multi-turn behaviors that depend on context
may spread variance across more dimensions than single-turn behaviors.

---

## 6. Predicted Delta (pre-registered)

| Metric | Predicted value | Falsifier threshold |
|---|---|---|
| EVR(1) across traits | >= 0.60 | < 0.40 falsifies |
| EVR(3) across traits | >= 0.90 | < 0.70 falsifies |
| Behavior success: top-3 vs full-rank | within 5% | > 10% gap falsifies |
| Effective rank (participation ratio) | 1.5–4 | > 8 falsifies low-rank claim |
| Cross-model consistency | EVR(1) >= 0.55 on all 3 tested models | < 0.50 on any model falsifies |

---

## 7. Protocol

### 7.1 Primary experiment

Dataset: the standard Gemma-3-270M and Gemma-3-1B contrast sets already in use
(Rogue-Scalpel direction and 2 additional behavioral traits from the harness).
For each trait and model: extract h_pos and h_neg at the best steering layer;
compute D matrix; run PCA (sklearn or torch.linalg.svd); record eigenvalue
spectrum (top 20), EVR(k) for k=1..10; compute effective rank as
participation ratio = (sum lambda_i)^2 / sum(lambda_i^2).
Then construct top-1, top-3, and full-rank steering vectors; run the steering
sweep at fractional alpha in {0.02, 0.05, 0.10, 0.20} using relative_add;
record behavior, PPL, composite per vector type.
Wall-clock: ~1 hour per (trait, model, seed) on a 4090. Budget: 3 traits * 2
models * 3 seeds = 18 runs ≈ 18 hours.

### 7.2 Where it shines

E28 is most informative for complex safety behaviors (e.g. discrimination between
hate speech sub-types) where we might expect higher effective rank. If EVR(3) < 0.90
specifically for safety-critical behaviors, that motivates low-rank steering
methods (LoReFT) for those use cases.

---

## 8. Cross-references

- Foundational result: E4 (DiffMean ≈ PCA-top-1 cosine), S-1, S-3, S-8
- Related hypotheses: E19 (Gram-Schmidt orthogonalization uses the same PCA
  machinery), E35 (sparse behavior vector — different axis, related structure)
- Corpus: N3 (orthogonal capacity = participation ratio), N12 (unified operator)
- Papers: ReFT 2404.03592, RepE 2310.01405, CRH 2605.01844

---

## 9. Committee Q&A

**Q: If DiffMean and PCA-1 are already 0.994 cosine-aligned, isn't E28 already
answered — k=1 is enough?**

> No. The cosine alignment tells us DiffMean points in the same direction as PCA-1.
> It says nothing about EVR(1). If EVR(1) = 0.30 and PCA-2 is nearly orthogonal to
> DiffMean but carries 25% of the variance, then a steering vector in the PCA-1
> direction misses a significant portion of the steerable space. The EVR question
> and the direction-alignment question are independent.

**Q: What is the expected contrast-set size needed to get a reliable PCA spectrum?**

> The PCA of D ∈ R^{n x d} is reliable when n >> rank(D). For rank 1-3, n=50 pairs
> suffices; for rank 10, n=100 is minimum. Our harness uses >= 50 pairs (E1 knee
> at ~50 pairs), so the PCA is well-posed for the predicted low-rank case.

**Q: How does this relate to N3 (orthogonal capacity)?**

> N3 predicts that the number of stackable behaviors is bounded by the participation
> ratio of the activation manifold. E28 tests the complementary claim: the intrinsic
> rank of a single behavior's representation in the contrastive-difference space.
> The two together bound the total behavioral capacity of a layer.

---

## 10. Verification checklist

- [ ] PCA of contrastive-difference matrix implemented in geometry.py
- [ ] Eigenvalue spectrum plotted and EVR(k) computed for k=1..10
- [ ] Effective rank (participation ratio) computed for each (trait, model, layer)
- [ ] Top-1, top-3, and full-rank steering vectors constructed and compared
- [ ] Behavior + PPL sweep at fractional alpha for each vector type
- [ ] Results consistent across >= 3 behavioral traits and >= 2 models
- [ ] Rows added to EXPERIMENT_LEDGER.md

---

## 11. Status journal

- 2026-05-31 — Created. UNTESTED. No screening data exists for E28. The DiffMean ≈
  PCA-1 cosine result (E4, S-1/S-3) is indirect evidence for low rank but does not
  directly test it. Priority: medium (after E27-A selective rotation and E31).

---

## Addendum: Research-Scientist Critique

*Reviewer: SciCritic-D. Critiquing the idea, not the implementation.*

### Prior plausibility

**MEDIUM-HIGH.** The linear representation hypothesis, ReFT's rank-4 success, and
the DiffMean / PCA-1 cosine alignment all point in the same direction. The
prediction of EVR(1) >= 0.60 is strong but not implausible for single, clean
behavioral traits. The claim will likely hold for simple traits and fail for
complex, multi-faceted safety behaviors.

### Mechanism scrutiny

The mechanism (superposition theory predicts single linear features for simple
behaviors) is the best-justified mechanistic prior in the field (Anthropic's
superposition work, Elhage et al. 2022). The concern is that "behavioral trait"
in practice is rarely a single atomic feature — it is a cluster label applied
to heterogeneous behaviors. The experimental design must carefully define traits
as atomic as possible to avoid conflating multiple features.

### Confounds

1. **Trait-complexity confound:** complex traits (safety, helpfulness) will
   systematically show higher effective rank than simple traits (language,
   sentiment), so aggregate claims must be stratified.
2. **Contrast-set homogeneity confound:** if positive/negative pairs are drawn
   from a narrow distribution, EVR(1) will be artificially high (the contrast
   matrix is effectively rank-1 by construction).
3. **Layer-dependence confound:** EVR(1) may be high at early layers (where
   representations are lower-dimensional) and lower at late layers (where
   residual stream accumulates more information).

### Does it specifically matter?

**YES, practically.** If EVR(3) >= 0.90, the field is correct to use single
DiffMean vectors. If EVR(3) < 0.70 for safety-relevant behaviors, low-rank
steering methods (LoReFT rank 4–8) are the correct tools and the entire
single-vector pipeline is leaving performance on the table.

### Literature precedent

ReFT (2404.03592) found rank 4 competitive with full-rank finetuning, implying
effective rank ~4 for fine-tuning objectives. PCA of contrastive activations
for sentiment (Park et al. 2310.15154) shows a dominant first component.
No paper has directly measured the contrastive-difference PCA spectrum for
refusal/safety behaviors with EVR analysis — this is a genuine empirical gap.

### Skeptical effect-size re-prediction

My prior for EVR(1): ~ Uniform[0.40, 0.80] for simple traits; ~ Uniform[0.20,
0.60] for complex safety traits. The claimed >= 0.60 threshold will likely hold
for simple traits but fail for complex ones. Prediction: mixed result, with the
hypothesis narrowed to "simple behavioral traits are low-rank; complex ones are not."

### Minimum-distinguishing experiment

Run PCA on the existing Rogue-Scalpel contrast pairs (a complex safety behavior)
and on a simple trait (e.g. "respond in French" — a linguistic register). Compare
EVR spectra. If Rogue-Scalpel shows effective rank > 5 but language shows rank ~1,
the claim is immediately qualified. Cost: < 1 hour on existing data.

### Verdict

**TESTABLE AND PRACTICALLY IMPORTANT. UNTESTED. Likely a mixed result stratified
by trait complexity. Run it; the scree plot will be informative regardless of
outcome.**
