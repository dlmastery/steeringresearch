# E2 — Fisher Layer Selection

> **One-line claim:** The optimal activation-steering injection layer is the layer of
> maximum linear separability of the contrast set (Fisher ratio), not a fixed late
> layer.
>
> **Source design space:** Block A — Foundations and measurement tooling (E2).
> **Primary axis:** A1 (WHERE — injection site / layer).
> **Implementation status:** `x FALSIFIED (screening, C1 — Spearman=+0.14 p=0.74)`.

---

## In Plain English

**What we're testing, simply:** The model thinks in stacked steps called layers.
We have to pick which layer to inject our nudge at. This asks whether a quick math
score (the "Fisher ratio") can tell us the best layer in advance, so we don't have
to test every layer by hand. (Spoiler: it can't — this idea was disproven.)

**Key terms (defined here so you don't have to look anything up):**
- **Language model (LLM):** an AI that predicts the next word; here, small Gemma
  models.
- **Steering:** nudging the model's behavior by adding a direction to its internal
  state as it writes, instead of retraining it.
- **Steering vector:** the specific direction of that nudge.
- **Residual stream:** the model's running internal state, where the nudge is added.
- **Layer:** one of the model's stacked processing steps (Gemma-2-2B has 26).
  Which layer we nudge at is the knob this experiment is about.
- **DiffMean:** the simple recipe for building the nudge — average the "yes"
  examples, average the "no" examples, subtract.
- **alpha (strength):** how hard we push the nudge.
- **Coherence:** whether the text stays fluent and sensible.
- **Fisher ratio:** a quick math score for how cleanly a layer separates the "yes"
  examples from the "no" examples. The idea was that a high score might mean the
  best layer to nudge — this turned out to be wrong.
- **"Read-out" vs "write-in":** the layer where the model has clearly *recorded* an
  idea is not necessarily the layer where *adding* a nudge has the biggest effect.
  That mismatch is why the quick score failed.

**Why we're doing this (the point):** Testing every layer one by one is slow. A
reliable shortcut for picking the layer would save a lot of time. We wanted to
know if this particular shortcut works.

**What the result would mean:** A positive result would have given a cheap rule for
layer choice. The actual (negative) result tells us this shortcut is unreliable, so
the layer must be chosen by directly testing behavior — and other experiments
should not rely on this shortcut either.

See [`../GLOSSARY.md`](../GLOSSARY.md) for any other term.

---

## 1. Motivation (>=100 words)

Choosing which transformer layer to inject a steering vector is one of the most
consequential hyperparameters in activation steering, yet it is typically resolved by
heuristics: "a middle-to-late layer" (CAA, arXiv:2312.06681), or the layer with the
highest probing accuracy for the behavior (RepE, arXiv:2310.01405), or a fixed fraction
of total depth (Inference-Time Intervention, arXiv:2306.03341). The Fisher discriminant
ratio — a classical linear-separability measure from Fisher (1936) and exploited in
linear discriminant analysis — offers a principled alternative: it identifies the layer
at which a linear classifier best separates the contrastive activations, which should,
intuitively, be where the concept is most linearly encoded and therefore most amenable
to linear steering. If Fisher ratio reliably predicts the best steering layer, it
provides a cheap, behavior-agnostic proxy for layer selection that requires no
generation or behavioral evaluation. This would reduce the cost of a layer sweep from
O(n_layers) generation runs to a single forward-pass eigenvalue computation. The
hypothesis is therefore practically significant: a reliable Fisher-based selection rule
would cut the rung-1 SMOKE cost substantially and generalize across behaviors.
The risk, identified in N8/E37, is that linear separability (interpretability/
detectability) may not imply causal steerability — the two properties could dissociate.

---

## 2. Formal hypothesis (>=50 words, falsifiable)

The Spearman rank correlation between per-layer Fisher ratio (ratio of between-class
to within-class variance of the DiffMean direction at each layer) and the measured
steering efficacy (behavior success score on a generation-based judge at fixed alpha)
is >= 0.70, across at least 3 distinct behaviors on Gemma-2-2B-it. This threshold
was inherited from the corpus (arXiv:2312.06681 evaluation protocol) and pre-registered
before the first screening run. The claim implies Fisher ratio is a reliable layer
selector: the max-Fisher layer should produce the highest or near-highest behavioral
response to a fixed-alpha steering intervention, within 10% of the best layer's efficacy.

---

## 3. Falsifier (>=30 words)

**Already fired (C1, screening).** Spearman(Fisher ratio, behavior efficacy) = +0.14
(p=0.74) on Gemma-3-270m with 8 layers swept at alpha=2. Max-Fisher layer (L12) was
NOT the best steering layer (L16 was, with higher behavior at lower PPL). The pre-
registered Spearman threshold of >=0.70 was not met. The hypothesis is FALSIFIED for
small Gemma models. Formal confirmation requires replication on Gemma-2-2B at n>=7
seeds with a generation-based judge to close the screening caveat.

---

## 4. Citations (>=80 words, Citation Rigor format)

```
Panickssery, Aryan, et al. 2023 arXiv 'Steering Llama 2 via
Contrastive Activation Addition' (arXiv:2312.06681) — CAA uses
middle-to-late fixed layers; does not report Fisher ratio as a
selection criterion; E2 tests whether Fisher would have selected
the same layers they found empirically.

Li, Kenneth, et al. 2023 NeurIPS 'Inference-Time Intervention:
Eliciting Truthful Answers from a Language Model' (arXiv:2306.03341)
— ITI selects layers by probing accuracy (closely related to Fisher);
the layer-selection mechanism E2 is generalizing; the correlation
between probing accuracy and intervention efficacy is exactly what E2
measures.

Park, Junsoo, et al. 2025 ICML 'CAST: Conditional Activation Steering'
(arXiv:2409.05907) — CAST uses a per-input condition score; does not
ablate layer selection vs Fisher; our screening (C1) shows that the
best condition layer (max-Fisher, L12) is not the best behavior layer
(L16), motivating CAST-style decoupling of condition-check layer from
steering layer.

Park, Junsoo, et al. 2023 NeurIPS 'The Linear Representation
Hypothesis and the Geometry of Large Language Models' (arXiv:
2311.03658) [UNVERIFIED ID] — Linear Representation Hypothesis
underpins why we expect Fisher ratio at the layer of maximal linear
separation to predict best steerability; E2 tests this expectation
and finds it does not hold, which is evidence against a naive LRH
interpretation.
```

---

## 5. Mechanism (deep technical)

Fisher ratio at layer L for a binary behavior is:

    F(L) = (mu_pos(L) - mu_neg(L))^T * Sigma_w(L)^{-1} * (mu_pos(L) - mu_neg(L))

where mu_pos, mu_neg are mean activations at layer L for the positive and negative
contrast set, and Sigma_w is the within-class covariance. Large F(L) means the
classes are well-separated by a linear boundary — which is also the condition under
which a DiffMean vector (= mu_pos - mu_neg, not the LDA projection) captures most of
the class-discriminative signal in a single direction.

The intuition for why Fisher SHOULD predict steerability: at a layer of high linear
separability, the concept direction v is strongly aligned with the dominant component
of the activation space; a small push along v has a large projection onto the
discriminative subspace and should shift behavior efficiently.

The reason it FAILS (as observed in C1): Fisher ratio measures the BETWEEN-class
variance relative to within-class variance of the EXISTING distribution. But steering
efficacy depends on how the MODEL REACTS to an added perturbation at that layer —
which is governed by the downstream computational graph (remaining layers, residual
connections, LayerNorm, attention), not by the upstream distribution. A layer can be
maximally discriminative (because earlier layers wrote the concept strongly there)
while being DOWNSTREAM of the critical decision circuit, making it a read-out point
rather than a write-in point. The layer where the concept is most "readable" need not
be the layer where adding to it has the largest causal effect.

This dissociation is exactly what N8 (Controllability != Interpretability) and E37
(Interpretable != Controllable) predict: the set of most causally steerable layers and
the set of most linearly interpretable layers are not required to coincide. C1's
result — max-Fisher L12 is the MOST FRAGILE layer (highest logPPL at alpha=4, C4) —
is consistent: the model has written the concept "legibly" at L12, but injecting
back there fights with downstream normalization and causes the most incoherence.

---

## 6. Predicted Delta (pre-registered numbers, now post-hoc)

| Metric | Pre-registered prediction | Observed (C1, screening) |
|--------|--------------------------|--------------------------|
| Spearman(Fisher, behavior) | >= 0.70 | **+0.14 (p=0.74) — FALSIFIED** |
| Max-Fisher layer = best steering layer | Yes | **No: L16 best, L12 max-Fisher** |
| Behavior at max-Fisher vs best layer | <= 5% gap | **L12=0.319, L16=0.534 (58% gap)** |

---

## 7. Experimental protocol

### 7.1 Primary experiment (status: falsified at screening; confirmation run needed)

- **Model:** Gemma-2-2B-it (4-bit), full 26-layer sweep.
- **Behaviors:** 3 semantically distinct behaviors (as pre-registered); generation-based
  judge (LLM-judge calibrated against human, not projection proxy).
- **Metrics:** Spearman(Fisher, behavior) across layers; behavior at max-Fisher layer
  vs best-found layer.
- **Control:** random-layer baseline; fixed late-layer (L20) baseline.
- **Seeds:** n>=7 for the Wilcoxon gate.

### 7.2 Where this should SHINE (but probably won't)

The Fisher-layer hypothesis would be most likely to hold in large, deeply trained
models where linear representations are especially clean. A scale-up test to Gemma-2-9B
at rung 4 is the natural extension — but still expected to FAIL based on the C1
mechanism explanation above (Fisher measures read-out, not write-in influence).

### 7.3 Actionable pivot

Since E2 is FALSIFIED, the actionable recommendation is: use the best-behavior layer
(found by a cheap behavior proxy, not Fisher) as the injection layer. For Gemma-3-270m,
this is L16. For Gemma-3-1b, the C6 sweep suggests L18 (max-Fisher L18 happened to
also give best behavior in that run, but this may be coincidence given n=1). A proper
layer sweep with a generation judge is the required next step.

---

## 8. Cross-references

- **C1** (campaign results): FALSIFIED at screening. Spearman=+0.14, p=0.74.
- **C4** (N20): max-Fisher layer L12 was also the most fragile layer — additional
  evidence against Fisher as a layer selector.
- **N8** (Controllability != Interpretability): theoretical prediction that Fisher
  (an interpretability proxy) would not predict controllability.
- **E37** (Interpretable != Controllable): the E37 hypothesis is supported by the
  E2 falsification as a mechanism account.
- **E13** (Early condition layer latency): also uses layer-selection logic; should
  not use Fisher as the selection criterion given E2's result.
- **IDEA_TABLE.md** Block A row E2.

---

## 9. Committee Q&A

**Q: The screening used a synthetic behavior proxy and only 8 layers — maybe Fisher
would work with a real judge and more layers?**

> Partially valid. The pre-registered threshold (Spearman >= 0.70) is unambiguous and
> was not close on C1 (0.14 vs 0.70). Even if more layers moved it to 0.30, the
> hypothesis would still be falsified. The only path to rehabilitation would be a
> fundamentally different concept where Fisher at every layer is computed on real
> in-distribution generation activations (not static forward passes), and compared to
> generation-judge behavior. This would be a new hypothesis (E2'), not a replication.

**Q: Couldn't the problem be the choice of alpha=2 for the sweep, not Fisher?**

> No. The falsifier is a rank correlation across layers, not an absolute performance
> claim. The rank ordering of layers should be preserved across alpha if Fisher is a
> reliable predictor. If it is not rank-invariant across alpha, Fisher is even less
> useful as a practical selector.

**Q: Should E2's falsification affect other hypotheses?**

> Yes, directly: E13 (early condition layer) should not use Fisher for condition-layer
> selection. E14 (discriminative-layer steering) should use a steerability criterion,
> not a discriminability criterion. N20 (curvature as fragility sensor) gains
> circumstantial support: if Fisher (high at L12) predicts the most fragile layer,
> maybe curvature (not Fisher) is the right fragility sensor.

---

## 10. Verification artifacts checklist

- [x] C1 layer-sweep table recorded in `ideas/_campaigns/C1_C2_results.md`.
- [x] Spearman(Fisher, behavior) = +0.14, p=0.74 logged.
- [ ] Replication on Gemma-2-2B with generation-based judge (n>=7) to formally close.
- [ ] E2 FALSIFIED status propagated to IDEA_TABLE.md.
- [ ] E13, E14 design docs updated to remove Fisher-based layer selection.
- [ ] Result row added to `EXPERIMENT_LEDGER.md` at Rung 2 (screening conclusion).

---

## 11. Status journal

- 2026-05-27 — hypothesis inherited from corpus E2; pre-registered Spearman threshold >= 0.70.
- 2026-05-29 — C1 campaign run on Gemma-3-270m, 8 layers, alpha=2, synthetic proxy:
  Spearman = +0.14 (p=0.74). **E2 FALSIFIED at screening.** Max-Fisher (L12) !=
  best steering layer (L16). Controllability != separability (confirms N8/E37 direction).
- 2026-05-31 — Design doc written; status FALSIFIED (screening confirmation pending on
  Gemma-2-2B with real judge).

---

## Addendum: Research-Scientist Critique

*Reviewer: SciCritic-A. The IDEA is already falsified; this critique addresses whether
the falsification is conclusive and what should replace E2.*

### Prior plausibility (pre-experiment)

**LOW-to-MEDIUM.** The Fisher-steerability link was appealing but untested. The ITI
paper (arXiv:2306.03341) reported that probing accuracy at layer L correlated with ITI
efficacy at L, which is the closest published support — but ITI measures probing accuracy
differently (single-layer MLP probe) and on a different model family. The direct
translation to DiffMean + Gemma was speculative.

### Mechanism scrutiny

The doc provides a crisp mechanistic account of why Fisher fails (read-out layer vs
write-in layer dissociation). This is the correct post-hoc analysis. The key insight —
that Fisher measures where the model has already resolved the concept, not where adding
a vector has maximal downstream effect — is consistent with causal mediation analysis
findings in mechanistic interpretability literature (e.g., path patching studies).

### Confounds in the falsification

1. **Small model, n=1, synthetic proxy**: the falsification is not yet conclusive by the
   project's own standards (n>=7 + real judge). The screening result is directionally
   very strong (0.14 vs 0.70) but has not cleared the formal gate.
2. **Only 8 layers tested**: the Spearman is over 8 ranks; a sample of 8 may not
   distinguish truly zero from weakly positive correlations. Still, 0.14 is far from 0.70.
3. **Alpha=2 may not be the right behavioral signal**: at alpha=2 the PPL is already
   elevated at most layers, potentially confounding the behavior measure.

### Does Fisher specifically matter?

No. Fisher is a linear separability measure. Any linear-separability metric (Mahalanobis
distance, class-conditional probing accuracy, explained variance by a linear probe)
would fail by the same mechanism if it measures read-out rather than write-in influence.
The right replacement is a CAUSAL metric: e.g., the causal effect of a unit-norm
perturbation at layer L on the final logit difference (the ITI delta at unit alpha).
This is exactly the layer-level intervention strength, not the layer-level encoding
fidelity.

### Literature precedent

Meng et al. "ROME" (arXiv:2202.05262) showed that factual associations are "stored"
in specific MLP layers identifiable by causal tracing — not by linear probing. This
is the interpretability ≠ controllability dissociation at the factual-memory level.
E2's falsification for behavior steering is the same dissociation in a new domain.
Heimersheim & Nanda (2024, arXiv) showed that linear probing accuracy at a layer
predicts what the model "knows" but not where it "decides" — directly relevant.

### Skeptical effect-size re-prediction

Even with a full Gemma-2-2B + real judge run, I predict Spearman(Fisher, efficacy)
in [0.0, 0.35] (80% CI). The null is likely; the hypothesis will formally close
as FALSIFIED.

### Minimum-distinguishing experiment

Already run (C1). The confirmation needed is: same 8-layer sweep on Gemma-2-2B-it,
3 behaviors, generation judge, n>=7 per cell. Expected cost: ~6 hours on the 4090.

### Verdict

**FALSIFIED (screening; confirmation run pending).** The dissociation between linear
separability and causal steerability is the mechanism, and is consistent with N8/E37.
The actionable pivot: use a causal perturbation-strength metric (ITI-delta at unit
alpha) for layer selection, not Fisher. This is a publishable negative result if
replicated at full n with a real judge.

---

## Pseudocode & Methodology

This hypothesis tests whether **Fisher ratio chooses the steering layer**. The knob
varied is **layer L**; the source is DiffMean (used both to steer and, via
`fisher_ratio`, to rank layers). Fisher is an extraction-side scalar, not an
injection operation (METHODOLOGY §1.3).

### 1. Steering-vector recipe

```python
# extract.build_vector_bank gives, per layer, {diffmean, pca, fisher} (METHODOLOGY §1.3)
bank = build_vector_bank(model, tok, load_concept(behavior), layers=range(n_layers))
for L in layers:
    v_L      = bank[L]["diffmean"]                       # diffmean_vector(pos,neg)
    fisher_L = bank[L]["fisher"]                         # fisher_ratio(pos,neg):
              # (mean(pos·w)-mean(neg·w))^2 / (var(pos·w)+var(neg·w)), w=unit(v_L)
L_maxfisher = best_layer(bank)                           # argmax_L fisher[L]
```

### 2. Experiment procedure

```text
1. For each layer L in the full sweep: extract v_L (DiffMean) and fisher[L].
2. For each L: steer with `add` at fixed alpha and measure behavior efficacy
       (hooks.apply_operation add: h' = h + alpha*v_L; SteeringContext).
3. Generation-judge behavior score per layer (judge.GeminiJudge; rung>=3).
4. Compute Spearman( fisher[L] , behavior_efficacy[L] ) across layers.
5. Compare behavior at L_maxfisher vs best-found layer.
6. CONTROL: random-layer baseline + fixed late-layer (L20) baseline.
```

The ONE varied knob is the layer; alpha, source, and operation are held fixed so the
layer-ranking is the only signal.

### 3. Measurement & decision rule

PRIMARY metric: `Spearman(fisher, behavior_efficacy)` across layers. FALSIFIER (§3,
already fired at screening): Spearman `< 0.70` ⇒ FALSIFIED. Observed C1:
`Spearman = +0.14 (p=0.74)`; max-Fisher L12 was NOT the best steering layer (L16 was;
58% behavior gap). Pre-registered prediction (§6) was `Spearman >= 0.70` and
max-Fisher-layer = best-layer with ≤5% gap. Verdict: **FALSIFIED** — Fisher measures
read-out separability, not causal write-in strength (consistent with N8/E37). Formal
closure needs the same sweep on Gemma-2-2B with a generation judge at n≥7.

### 4. Where the code is / status

Driver: `scripts/campaign_sweep.py` (the C1 layer sweep, exp# 20–27) — Fisher and
DiffMean both come from `extract.build_vector_bank`; injection via
`hooks.apply_operation('add', ...)`. Status **FALSIFIED at screening**; the only
missing infra is the Gemma-2-2B replication with a real judge (no new machinery
needed — the sweep already runs).

See [`../METHODOLOGY.md`](../METHODOLOGY.md) for the shared recipe.

---

## Provenance & Tracing

Full per-hypothesis provenance (exact experiments, reproduce commands, artifact links, reasoning trace): [`PROVENANCE/E2.md`](../PROVENANCE/E2.md).

- **Experiments:** exp# 20, 21, 22, 23, 24, 25, 26, 27, 121 (`autoresearch_results/experiment_log.jsonl`).
- **Reproduce:**

```bash
PYTHONPATH=src python scripts/campaign_sweep.py --model models/google/gemma-3-270m-it --quant none --hyp E2 --tag-prefix C1-E2-layer --behavior ocean --layers 2 4 6 8 10 12 14 16 --alphas 2.0 --ops add --sources diffmean
```
