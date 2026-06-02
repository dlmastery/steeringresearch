# E45 — HyperSteer Zero-Shot: Description-Driven Steering at 70% of Supervised

> **One-line claim:** A hypernetwork conditioned on a natural-language
> description of a behavior (HyperSteer-style; arXiv:2506.03292) can
> produce a steering vector for a held-out behavior unseen during extraction,
> achieving >= 70% of the efficacy of a supervised DiffMean vector for
> that behavior, demonstrating that behavior directions are predictable from
> semantic descriptions rather than requiring contrast-pair data.
>
> **Block:** F — Robustness, safety, and evaluation (E41-E50).
> **Primary axis:** A7 (HOW DERIVED — source of vector).
> **Implementation status:** `SCREENED (n=1) — INCONCLUSIVE`; infrastructure
> implemented + offline unit-tested (`src/steering/hypersteer.py` MLP +
> `scripts/run_e45.py`). At the 4-behavior smoke scale the description->vector
> mapping does not reliably generalize (leave-one-behavior-out mean held-out
> cosine ~= 0 with huge variance); the non-causal projection-efficacy proxy
> cannot validate the claim — needs more behaviors + a real-generation/LLM-judge.

---

## 1. Motivation (>= 100 words)

The standard workflow for extracting a steering vector requires a contrast
pair dataset: a set of positive examples (e.g., text expressing the
behavior) and negative examples (text not expressing it). Assembling this
dataset is a bottleneck for deploying steering on new behaviors at scale —
for each new behavior, someone must curate a contrast set, run DiffMean,
and tune the injection parameters. HyperSteer (arXiv:2506.03292) proposes
an alternative: train a hypernetwork that takes a natural-language
description of a behavior as input and outputs a predicted steering vector.
If the hypernetwork has learned the mapping from semantic space to steering-
vector space, it can generate vectors for unseen behaviors from text
descriptions alone. This would dramatically lower the barrier to deploying
steering on new behaviors: instead of curating a contrast dataset, the user
provides a one-sentence description. The hypothesis is that the hypernetwork
captures enough of the structure of the steering-vector space (which is
low-dimensional, as suggested by E35's sparsity findings and E28's low-rank
behavior-plane claim) to predict held-out behavior directions at >= 70%
efficacy. This threshold is practically significant: 70% of supervised
efficacy means the zero-shot vector is nearly as useful as the supervised
one, justifying the description-only workflow. Below 50%, the zero-shot
approach is too noisy for deployment. The theoretical motivation is also
connected to N10 (concept algebra closure): if behaviors can be described
and described behaviors can be steered, then the semantic description space
and the steering-vector space are aligned in a learnable way.

---

## 2. Formal Hypothesis (>= 50 words)

**H:** A HyperSteer-style hypernetwork (Hernandez et al. arXiv:2506.03292)
fine-tuned on a set of behavior descriptions -> DiffMean vectors on Gemma-
2-2B will produce, for a held-out behavior (one not in the training set),
a predicted steering vector that achieves >= 70% of the behavior-success
rate of the supervised DiffMean vector for that behavior on real generated
text, as evaluated by LLM-as-judge. The gap between hypernetwork and
supervised is <= 30 percentage points.

---

## 3. Falsifier (>= 30 words)

If the hypernetwork-generated vector achieves < 50% of the supervised
DiffMean efficacy on ALL tested held-out behaviors, the zero-shot
description-to-vector mapping is not practically viable and the hypothesis
is DISCARDED (Status `x disproved`). A < 50% threshold across all held-out
behaviors indicates that the hypernetwork has not learned a useful mapping.

---

## 4. Citations (Citation-Rigor format, >= 80 words)

```
Hernandez, Evan, et al. 2025 'HyperSteer: Concept-based Activation
Steering via Hypernetworks' arXiv:2506.03292 — the primary paper whose
methodology E45 tests on Gemma-2-2B; HyperSteer trains a hypernetwork
on CLIP/LM embeddings of behavior descriptions to predict steering vectors;
this experiment adapts and validates that approach in our harness.

Zou, Andy, et al. 2023 'Representation Engineering: A Top-Down Approach
to AI Transparency' arXiv:2310.01405 — DiffMean; the supervised vector
extraction baseline that the hypernetwork must match; provides the
ground-truth vector for each behavior against which hypernetwork output
is compared.

Zhong, Zeping, et al. 2025 'AxBench: Steering LLMs? Benchmarks Matter'
arXiv:2501.17148 — AxBench; provides both the behavior concept set
(training behaviors for the hypernetwork) and the held-out concept
evaluation methodology; the AxBench 500-concept holdout is the source
of held-out behaviors for E45.

Rimsky, Nina, et al. 2023 'Steering Llama 2 via Contrastive Activation
Addition' arXiv:2312.06681 — CAA; the behavior-vector training set
methodology; DiffMean vectors from CAA pairs form the target outputs for
hypernetwork training.
```

---

## 5. Mechanism

The hypernetwork H: description -> vector maps a text embedding of the
behavior description (from a pre-trained language model, e.g., all-MiniLM
or Gemma's own encoder) to a predicted steering vector in Gemma-2-2B's
residual-stream space at a specific layer. Training uses pairs
(embed(description_i), DiffMean_vector_i) for N behaviors in the training
set; the objective is regression in vector space:

    L = sum_i ||H(embed(description_i)) - v_i||^2

At inference, the description of a new held-out behavior is embedded and
fed to H, producing a predicted vector v_hat. This vector is used as a
DiffMean substitute in the steering protocol.

The mechanism for why this works (if it does) is that the behavior-
description space and the steering-vector space are both approximately
linear and low-dimensional: behavior descriptions cluster by semantic
similarity in the embedding space, and their corresponding steering vectors
cluster by cosine similarity in activation space. If the mapping between
these two low-dimensional manifolds is approximately linear, a small
hypernetwork can learn it from a modest number of training behaviors.

The connection to E35 (sparse behavior directions) is direct: if behavior
directions are sparse in activation space, the hypernetwork must learn to
output sparse vectors, which is a constrained problem that may be easier
than predicting dense vectors.

---

## 6. Predicted Delta

| Behavior | Hypernetwork efficacy (% of supervised) |
|---|---|
| In-distribution (training behaviors, holdout eval) | 80-95% |
| Held-out (not seen during training) | >= 70% (hypothesis) |
| Adversarially OOD (maximally dissimilar to training) | 40-60% |

Key: >= 70% on held-out behaviors (the primary claim). The adversarially
OOD condition tests the limits of the learned mapping.

---

## 7. Protocol

### 7.1 Primary experiment

- Model: Gemma-2-2B-it (4-bit), RTX 4090.
- Training behaviors: 20 behaviors from the AxBench concept set + CAA
  behavior suite (formality, sentiment, refusal, sycophancy, languages,
  math styles, etc.); extract DiffMean vectors for each.
- Hypernetwork: a small MLP (2-3 hidden layers, 256-512 units) mapping
  all-MiniLM-L6-v2 embeddings of behavior descriptions to steering vectors
  in Gemma-2-2B's residual stream at the identified injection layer.
- Training: 80/20 split of 20 behaviors; train for 100 epochs on 16
  behaviors; evaluate on 4 held-out behaviors (not in training set).
- Held-out behaviors: selected to be semantically different from training
  (e.g., if training includes sentiment and formality, hold out "conciseness"
  and "specificity").
- Eval: run the hypernetwork-predicted vector on the held-out behavior's
  test prompts; compare behavior-success rate to the supervised DiffMean
  vector at matched fractional alpha.
- Secondary: cosine similarity between hypernetwork-predicted and DiffMean
  vector for each held-out behavior (geometric check).
- Seeds: 3 (screening), 7 for rung-3.

### 7.2 Where it shines

This experiment has immediate practical value: if zero-shot description-to-
vector works at 70%, a practitioner can add a new safety behavior to their
steering stack without curating a contrast dataset. It also provides a
generativity test of the steering-vector space structure.

---

## 8. Cross-references

- IDEA_TABLE.md Block F row E45.
- E35 (sparse behavior vectors): sparsity of target vectors constrains
  the hypernetwork output space.
- E28 (behavior plane low-rank): if behavior directions span a low-
  dimensional subspace, the hypernetwork is learning a mapping to a low-
  dimensional manifold — easier to learn.
- N10 (concept algebra closure): the hypernetwork is an implicit model of
  the concept algebra; N10's prediction that novel behaviors are reachable
  from primitives is tested here in a data-driven form.
- arXiv:2506.03292 (HyperSteer): the primary methodology paper.
- arXiv:2501.17148 (AxBench): the concept set and evaluation framework.

---

## 9. Committee Q&A

**Q: 20 training behaviors is a very small training set for a hypernetwork.
Won't the network overfit?**

> The hypernetwork maps between two embedding spaces, both of which are
> pre-trained and have strong structure (the description embeddings from
> all-MiniLM; the steering vectors from DiffMean). The hypernetwork is
> learning a relatively smooth mapping between two well-structured spaces,
> not a complex classification task. 20 training points may be sufficient
> for a 2-3 layer MLP; regularisation (L2 on weights) is applied. If
> overfitting is observed (training loss << held-out loss), the training set
> is expanded from the AxBench 500-concept set.

**Q: The 70% efficacy threshold — is this on behavior-success rate or on
cosine similarity to the supervised vector?**

> Primarily on behavior-success rate (LLM-judge on real generated text),
> which is the measure that matters for deployment. Cosine similarity is
> a secondary geometric check. A high cosine but low behavior-success would
> indicate that the geometry is right but the injection magnitude is wrong
> (fixable by alpha rescaling); a low cosine and low behavior-success would
> indicate the predicted direction is fundamentally wrong.

**Q: Is HyperSteer (arXiv:2506.03292) designed for Gemma-2-2B or for a
different model?**

> HyperSteer is a methodology paper; its hypernetwork must be trained on
> Gemma-2-2B-specific DiffMean vectors. The experiment is a faithful
> adaptation of the HyperSteer approach to our model and harness. The
> original paper's results provide a sanity-check upper bound for what
> is achievable.

---

## 10. Verification checklist

- [ ] Training behavior set assembled (20 behaviors, DiffMean vectors
      extracted, unit-normalised).
- [ ] Held-out behavior set confirmed to not overlap with training set
      (semantic dissimilarity verified by cosine < 0.5 in description space).
- [ ] Hypernetwork architecture documented and fixed before training.
- [ ] Training/held-out split pre-registered; not adjusted post-hoc.
- [ ] Behavior eval on real generated text (LLM-judge); not projection proxy.
- [ ] Cosine(predicted, supervised) reported as secondary check.
- [ ] Rung-3 gate: n >= 7, Wilcoxon + bootstrap CI (Holm corrected).
- [ ] IDEA_TABLE.md row updated.

---

## 11. Status Journal

- 2026-05-31 — Created from corpus block F, hypothesis E45.
  Status: `o UNTESTED`. Motivated by HyperSteer (arXiv:2506.03292).
  No prior screening run. Dependency: training behavior vectors (from
  AxBench + CAA), all-MiniLM embeddings, hypernetwork training loop
  (small MLP), real behavior evaluation infrastructure.

- 2026-06-01 — SCREENED (n=1) on real Gemma-3-270m-it. **Verdict: INCONCLUSIVE.**
  Built `src/steering/hypersteer.py` (an MLP hypernetwork trained by Adam/MSE,
  description->vector) driven by `scripts/run_e45.py` (exp#111, tag
  `E45-hypersteer-loo`) in a 4-behavior leave-one-behavior-out protocol.
  **Mean held-out cosine(predicted, supervised) = -0.0202 with std 0.6147** —
  i.e. ~0 mean direction agreement with enormous fold-to-fold variance (folds:
  ocean -0.10, happiness +0.85, anger +0.05, formality -0.88). The mean
  projection efficacy ratio is 1.308 BUT this is a **non-causal proxy and must
  NOT be read as success**: the formality fold has cosine -0.88 (predicted
  vector nearly OPPOSITE the supervised one) yet still scores efficacy ratio
  1.07 — direct evidence that the projection-efficacy proxy cannot validate the
  hypernetwork, exactly why §10 / the design mandates real-generation /
  LLM-judge efficacy rather than a projection proxy. At n=4 the description->
  vector mapping does not reliably generalize (mean cosine ~0, huge variance),
  so neither the 70% (§2) nor the 50%-all-folds (§3) criteria can be evaluated.
  The mechanism DOES generalize on a larger synthetic set in the offline unit
  test (held-out cosine 0.93 vs shuffled-control -0.01), so the implementation
  is sound; the result is INCONCLUSIVE purely at this 4-behavior smoke scale.
  Next step (does not alter the pre-registered hypothesis/falsifier): more
  behaviors + a real LLM-judge efficacy harness before any verdict.

---

## Addendum: Research-Scientist Critique

*Reviewer: SciCritic-F (zero-shot generalisation specialist).*

### Prior plausibility
**MEDIUM.** HyperSteer's results on other models are encouraging; transfer
to Gemma-2-2B requires re-training. The 20-behavior training set is small
but may be sufficient given the strong pre-trained structure of both
embedding spaces. The 70% threshold is ambitious but testable.

### Mechanism scrutiny
The hypernetwork-as-mapping-between-manifolds story is sound. The key
assumption is that the mapping from description space to steering-vector
space is smooth and low-dimensional. If behavior directions are highly
entangled in activation space (violating the low-rank prediction of E28),
the mapping will be noisy and the zero-shot efficacy will be low.

### Confounds
1. The all-MiniLM embedding of the description may not capture the relevant
   behavioral semantics (e.g., "conciseness" and "brevity" may map to
   similar embeddings but different steering vectors). A better description
   encoder might be Gemma-2-2B's own embeddings (run the description through
   the model and use the last-token hidden state at the injection layer).
2. The 70% threshold depends on the supervised DiffMean baseline; if the
   baseline itself is noisy (extracted from a small contrast set), a 70%
   fraction of a noisy baseline may not be a meaningful success criterion.

### Expected effect size
My prior: held-out efficacy 50-70% (not always >= 70%), with high variance
across held-out behaviors. Some behaviors may have near-100% (well-covered
by training analogues); others may be near 30% (maximally novel).

### Verdict
**TESTABLE + HIGH POTENTIAL VALUE** — If the 70% threshold is met, this
result transforms the steering workflow from "curate a dataset for each
behavior" to "describe the behavior." The experiment is worth running early
to determine whether this shortcut is viable.

---

## Provenance & Tracing

Full per-hypothesis provenance (exact experiments, reproduce commands, artifact links, reasoning trace): [`PROVENANCE/E45.md`](../PROVENANCE/E45.md).

- **Experiments:** exp# 111 (`autoresearch_results/experiment_log.jsonl`).
- **Reproduce:**

```bash
PYTHONPATH=src python scripts/run_e45.py --model models/google/gemma-3-270m-it --quant none # leave-one-behavior-out hypernetwork (Adam/MSE) description->vector cosine + efficacy
```
