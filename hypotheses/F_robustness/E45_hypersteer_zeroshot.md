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

## In Plain English

**What we're testing, simply:** Normally, to steer a new behavior you must hand-
collect example sentences first. We ask whether a small helper program can skip
that step — read a one-sentence *description* of a behavior ("be more concise")
and write the steering nudge for it directly, even for a behavior it never saw.

**Key terms (defined here):**
- **Language model** — an AI that writes text one word at a time.
- **Steering** — changing the model's behavior by editing its internal state
  mid-sentence, without retraining.
- **Steering vector** — the nudge we add to push toward a behavior.
- **Residual stream** — the model's running internal scratchpad; the nudge goes
  here.
- **Layer** — one of the model's stacked processing steps.
- **alpha / strength** — how hard we push.
- **DiffMean** — the normal, example-based recipe for a nudge (average internal
  state on "yes" examples minus "no" examples). Here it's the *target* the
  helper program tries to reproduce, and the yardstick we score against.
- **Hypernetwork** — a small helper network that *writes* a steering nudge from
  a text description, instead of us building it from examples.
- **HyperSteer** — the specific published method (a hypernetwork for steering)
  this doc adapts and tests.
- **Zero-shot** — working on a behavior the helper was never trained on; the
  real test of whether it learned a general skill.
- **Coherence** — whether the steered text stays fluent and sensible.

**Why we're doing this (the point):** If it works, adding a new steerable
behavior would be as easy as describing it in a sentence — no data collection.
That would make steering far faster to deploy.

**What we found (honest status):** We built the helper program and tried it, but
only at a tiny scale (four behaviors). The result was **inconclusive**: on
behaviors it hadn't seen, the nudges it wrote were sometimes right, sometimes
pointed the *wrong way*, with no consistency. The basic machinery works on a
larger pretend dataset, so the idea isn't broken — we just can't yet say whether
it really works. A fair verdict needs many more behaviors and a proper quality
check on the actual generated text.

**What a future result would mean:** If, at larger scale, the described-into-
existence nudges reliably match the example-built ones, steering gets a powerful
shortcut. If they keep pointing the wrong way, the description-only approach
isn't trustworthy enough to use.

See [`../GLOSSARY.md`](../GLOSSARY.md) for any other term.

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

## Pseudocode & Methodology

This section specializes [`../METHODOLOGY.md`](../METHODOLOGY.md) to E45. The
shared recipe builds a vector in CLOSED FORM (`extract.diffmean_vector`); E45 is
the one exception in this block where a vector is *predicted by a trained
hypernetwork* (`src/steering/hypersteer.py`). DiffMean here is not the steer —
it is the regression TARGET.

### 1. Steering-vector recipe (description -> vector, GRADIENT-trained)

```python
# §1.3 METHODOLOGY: supervised DiffMean is the TARGET, not the steer.
# For each training behavior b: extract its supervised direction in closed form.
for b in train_behaviors:                       # ocean, happiness, anger, ... (AxBench+CAA)
    bank   = extract.build_vector_bank(model, tok, load_concept(b), layer)
    v_b    = bank[layer]["diffmean"]             # extract.diffmean_vector: mean(h_pos)-mean(h_neg)
    emb_b  = hypersteer.encode_descriptions(model, tok, [desc_b], layer)[0]
                                                 # Gemma's OWN mean-pooled residual at `layer`
                                                 # (NOT all-MiniLM — keeps the module dep-free)

# Train the hypernetwork H: emb -> vector by MSE regression (spec §5 objective)
#   L = sum_i || H(embed(description_i)) - v_i ||^2   (+ Adam weight_decay=l2)
net = hypersteer.train_hypernet(
        embeddings=E_train, target_vectors=V_train,   # [n, embed_dim] -> [n, dim]
        epochs=300, lr=1e-3, l2=1e-4, hidden=256, depth=2, seed=seed)

# ZERO-SHOT: predict a held-out behavior's vector from its DESCRIPTION ALONE.
v_hat = hypersteer.predict_vector(net, hypersteer.encode_descriptions(
            model, tok, [desc_holdout], layer)[0])    # no contrast set was curated for it
```

`HyperNet` is a 2-layer ReLU MLP (`embed_dim -> 256 -> dim`, final linear
unactivated). Everything is float32, deterministic by `seed`, runs identically
on `FakeResidualLM` (offline tests) and real Gemma.

### 2. Experiment procedure (leave-one-behavior-out)

```text
1. Pick `layer` (the injection/extraction layer) and the behavior pool.
2. For each fold = each held-out behavior b*:
   a. TARGETS: DiffMean v_b for every b != b*  (extract.diffmean_vector).
   b. INPUTS : encode_descriptions(...) for every b != b*.
   c. Train net on the (n-1) others; PREDICT v_hat(b*) from desc(b*).
3. Inject v_hat at `layer` for b*'s test prompts via hooks.apply_operation,
   operation = relative_add  (h' = h + alpha*||h||*unit(v); §2 METHODOLOGY) —
   alpha a FRACTION of ||h|| so the magnitude is source-comparable.
4. Compare to the supervised DiffMean steer for b* at matched fractional alpha.
5. MEASURE (§3 METHODOLOGY): behavior efficacy via the off-family judge
   judge.GeminiJudge on real generated text (PRIMARY, per §2/§10); PPL; MMLU;
   JailbreakBench CR (baseline 0%); geometry probes.
   SECONDARY geometric check: cosine(v_hat, v_supervised) via
   hypersteer.evaluate_holdout (NOT a substitute for real-generation efficacy).
```

### 3. Measurement & decision rule

- **PRIMARY metric:** held-out behavior-success rate (LLM-judge on real text) as
  a fraction of the supervised DiffMean rate for the same behavior.
- **Hypothesis (§2):** held-out efficacy >= 70% of supervised; gap <= 30 pp.
- **Pre-registered FALSIFIER (§3):** if the predicted vector achieves < 50% of
  supervised efficacy on ALL tested held-out behaviors, DISCARD (`x disproved`).
- **Secondary check (§6):** cosine(v_hat, v_supervised) — high cosine + low
  efficacy ⇒ direction right, magnitude wrong (alpha-fixable); low cosine + low
  efficacy ⇒ direction fundamentally wrong.

### 4. Where the code is / status — TESTED, INCONCLUSIVE

- **Driver:** `scripts/run_e45.py` (exp#111, tag `E45-hypersteer-loo`);
  hypernetwork in `src/steering/hypersteer.py`.
- **Reproduce:** `PYTHONPATH=src python scripts/run_e45.py --model
  models/google/gemma-3-270m-it --quant none`.
- **Verdict: `SCREENED (n=1) — INCONCLUSIVE`.** At the 4-behavior leave-one-out
  smoke scale, **mean held-out cosine(predicted, supervised) = -0.0202, std
  0.6147** (folds: ocean -0.10, happiness +0.85, anger +0.05, formality -0.88):
  ~0 mean direction agreement with enormous fold-to-fold variance. The mean
  projection-efficacy ratio (1.308) is a **non-causal proxy and must NOT be read
  as success** — the formality fold has cosine -0.88 (nearly OPPOSITE the
  supervised vector) yet still scores efficacy ratio 1.07, direct evidence the
  projection proxy cannot validate the hypernetwork. Neither the 70% (§2) nor the
  50%-all-folds (§3) criteria can be evaluated at n=4. The implementation IS
  sound (offline unit test: held-out cosine 0.93 vs shuffled-control -0.01); the
  result is INCONCLUSIVE purely at this smoke scale. **Missing machinery for a
  verdict:** more behaviors (20+, per §7) AND a real-generation `judge.GeminiJudge`
  efficacy harness in place of the projection proxy. This does NOT alter the
  pre-registered hypothesis or falsifier.

See [`../METHODOLOGY.md`](../METHODOLOGY.md) for the shared recipe.

---

## Provenance & Tracing

Full per-hypothesis provenance (exact experiments, reproduce commands, artifact links, reasoning trace): [`PROVENANCE/E45.md`](../PROVENANCE/E45.md).

- **Experiments:** exp# 111 (`autoresearch_results/experiment_log.jsonl`).
- **Reproduce:**

```bash
PYTHONPATH=src python scripts/run_e45.py --model models/google/gemma-3-270m-it --quant none # leave-one-behavior-out hypernetwork (Adam/MSE) description->vector cosine + efficacy
```
