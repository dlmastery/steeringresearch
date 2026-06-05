# N4 — Steering as Inverse In-Context Learning

> **One-line claim:** A DiffMean behavior vector aligns (cos > 0.6) with the activation
> delta induced by an in-context demonstration of the same behavior, making steering
> and ICL dual representations of a single operator — prompt in data-space, vector in
> activation-space.
>
> **Primary axes:** A7 (how-derived/source), A4 (operation)
> **Status:** UNTESTED

---

## In Plain English

**What we're testing, simply:** There are two ways to make a model act a certain way
without retraining: *show it examples* in the prompt ("here are three polite replies…"),
or *directly nudge its internal state* with a steering arrow. This doc asks whether
these are really the same thing — whether showing examples shoves the model's internal
"thought" in almost exactly the direction our hand-built steering arrow points.

**Key terms (defined here):**
- **Steering / steering vector** — changing behavior by adding a chosen direction to
  the model's internal "thought" mid-sentence, instead of retraining.
- **In-context learning (ICL)** — getting the behavior by putting examples in the
  prompt instead of nudging internals.
- **Residual stream** — the model's running internal thought; what we read and edit.
- **Layer** — one of the model's processing steps; a knob.
- **DiffMean** — the simplest recipe for building a steering arrow: average the
  internal state on "yes-behavior" examples, average it on "no" examples, subtract.
- **Cosine / alignment** — how closely two arrows point the same way (1 = identical
  direction, 0 = unrelated).

**Why we're doing this (the point):** If the example-prompt shove and the steering
arrow point the same way, we can sanity-check our arrows cheaply (just compare them
to what examples do) and better explain *why* steering works at all.

**What the result would mean:** A win means examples and steering are two faces of one
operation. A loss means they change the model in genuinely different ways.

See [`../GLOSSARY.md`](../GLOSSARY.md) for any other term.

---

## 1. Motivation (>= 100 words)

In-context learning (ICL) and activation steering are both methods for inducing
behavioral change in language models without gradient updates. ICL prepends examples;
steering injects a vector into the residual stream. They are treated as categorically
different: ICL is a prompting technique, steering is a mechanistic intervention. But
if both are changing the model's effective behavior, they must both be shifting
something in activation space. The natural hypothesis is that the activation shift
induced by an in-context demonstration of behavior B is approximately the same as
the DiffMean vector extracted for behavior B. This would mean: steering is just ICL
compressed into a fixed vector; or equivalently, ICL is steering distributed across
the token stream. The connection is not merely aesthetic — it has practical implications.
If the two activation shifts are high-cosine, then (1) we can validate behavior
vectors cheaply by checking ICL alignment without running steering experiments, (2)
we can explain WHY DiffMean works (it extracts the activation signature that ICL
induces), and (3) we can improve behavior vectors by optimizing for ICL alignment
as a training objective. The "function vector" line of work (Todd et al. 2023,
arXiv:2310.15213) showed that task vectors derived from ICL activations transfer
to zero-shot performance; N4 extends this to the full DiffMean-ICL alignment claim.

## 2. Formal Hypothesis (>= 50 words)

Let v_DM(B) = mean(h | behavior B) - mean(h | baseline) be the DiffMean vector for
behavior B at layer L. Let v_ICL(B) = mean(h | ICL-demo of B) - mean(h | no-demo)
be the activation shift induced by prepending k=5 demonstrations of behavior B.
The claim is:

  cos(v_DM(B), v_ICL(B)) >= 0.60

at the optimal injection layer L* (identified by max Fisher ratio), for at least 3
of the 5 tested behaviors on Gemma-3-1B. Additionally, aligning the DiffMean vector
toward v_ICL (taking the weighted average) should improve steering efficacy by at
least 5% relative to raw DiffMean.

## 3. Falsifier (>= 30 words)

If cos(v_DM, v_ICL) < 0.40 across all behaviors and all candidate layers, the
duality hypothesis is FALSIFIED. If cos is 0.40-0.59, the claim is INCONCLUSIVE.
If ICL alignment does not improve DiffMean efficacy (< 2% relative gain), the
practical implication fails even if the cosine is high.

## 4. Citations (Citation Rigor >= 80 words)

```
Todd, Ericson, Bhende, et al. 2023. 'Function Vectors in Large Language Models'
arXiv:2310.15213 (ICLR 2024). Directly relevant: shows that ICL-derived "function
vectors" at specific layers encode the task structure and causally produce task
performance when injected — the foundational evidence that ICL creates steering-like
activations. N4 extends this by testing whether the DIRECTION of the function vector
aligns with DiffMean, not just that ICL-derived vectors are causally effective.

Turner et al. 2023. 'Activation Addition' arXiv:2312.06681. The DiffMean anchor;
N4 tests whether CAA-style vectors are the same object as Todd et al.'s function vectors.

Garg, Tsipras, Liang, Valiant 2022. 'What Can Transformers Learn In-Context?'
arXiv:2208.01066 (NeurIPS 2022). Theoretical grounding: shows that transformers
implement implicit gradient descent during ICL, predicting that ICL activations will
move in the gradient direction of the behavior loss — which is what DiffMean
approximates as a first-order difference of means.

Wurgaft et al. 2026. 'Manifold Steering' arXiv:2605.05115. If ICL and DiffMean both
push activations along the activation manifold M_h toward the behavioral target region,
both vectors are manifold-tangent; their alignment follows from both approximating the
same geodesic direction on M_h.
```

## 5. Mechanism

ICL prepends k demonstrations: [demo_1, ..., demo_k, query]. The attention mechanism
aggregates the demonstration activations into the query token's key-value cache,
which shifts the residual stream at the query position. For a behavior contrast
(behavior B vs baseline), the ICL shift v_ICL is the average activation difference
at the query position between runs with and without demonstrations.

DiffMean v_DM is the average difference of activations in the behavioral condition
vs baseline condition, typically computed on short prompts that elicit the behavior.
Both quantities approximate the first-order effect of making the model "think about"
behavior B. If the behavioral representation is approximately linear in the activation
space (which is the DiffMean assumption), both methods are approximating the same
direction in representation space.

The connection to gradient descent (Garg et al. 2022): ICL acts as implicit gradient
descent on a behavior prediction loss. The gradient of L(h) w.r.t. h at the query
point is approximately in the direction of v_ICL. DiffMean approximates this gradient
as a first-difference estimate. Thus cos(v_DM, v_ICL) >= 0.60 is the prediction that
the DiffMean finite-difference approximation captures > 60% of the true gradient
direction — reasonable for well-extracted behaviors.

Layer dependence: the alignment should be strongest at the layer where the ICL signal
"converges" — typically mid-to-late layers where attention has fully aggregated the
demo context. This layer should coincide with the optimal DiffMean injection layer.

## 6. Predicted Delta

| Metric | Predicted Value | Rationale |
|---|---|---|
| cos(v_DM, v_ICL) at optimal layer | 0.60 - 0.80 | Gradient alignment argument |
| cos at non-optimal layers | 0.20 - 0.50 | ICL signal not yet converged |
| Efficacy improvement from ICL-aligned v | +5% to +15% relative | ICL alignment corrects DiffMean noise |
| Number of behaviors with cos >= 0.60 | >= 3 of 5 | Some behaviors may be non-linear |
| Layer of max alignment = Fisher layer | 60% of behaviors | Same convergence logic |

## 7. Protocol

### 7.1 Primary experiment

- Model: Gemma-3-1B-it (established steering window)
- Behaviors: 5 contrastive behaviors from the synthetic suite
- ICL demos: 5 demonstrations per behavior (DiffMean extraction format: behavioral
  example paired with baseline example)
- v_ICL extraction: forward pass with demos prepended, extract h at query token at
  layer L; forward pass without demos, extract h; difference
- v_DM extraction: standard DiffMean from 50 contrast pairs
- Cosine measurement: at layers {8, 12, 16, 20, 24}
- ICL alignment: v_blend = (1-w)*v_DM + w*v_ICL, sweep w in {0, 0.25, 0.5, 0.75, 1.0}
- Efficacy metric: behavior cosine at alpha corresponding to 5% off-shell displacement
- Seeds: 3 demo-selection seeds x 3 extraction seeds
- Wall-clock: ~3 hours on RTX 4090

### 7.2 Where it shines

Zero-shot behavior transfer: if N4 holds, a new behavior with k=5 ICL demos yields
a steering vector without any contrast-pair collection, at the cost of one forward
pass. This is practically valuable for behaviors where contrast pairs are hard to
label (e.g., subtle personality traits).

## 8. Cross-References

- N8 (controllability != interpretability): N4 provides an alternative derivation
  method (ICL-based) that may better recover the controllable subspace than DiffMean
- N10 (concept algebra closure): if behaviors are dual to ICL, algebraic composition
  of behaviors corresponds to composition of ICL demonstrations
- N12 (capstone unified operator): the "how derived" axis (A7) includes ICL as a
  vector source; N4 validates this axis choice
- Todd et al. 2023 (function vectors): direct precedent; N4 tests if the function
  vector IS the DiffMean vector
- IDEA_TABLE.md: N4 row, axes A7+A4

## 9. Committee Q&A

**Q: ICL demonstrations vary in format and the activation shift will depend on which
demos were chosen. How is v_ICL defined stably?**

> v_ICL is averaged over 3 demo-selection seeds, each drawing 5 demonstrations
> randomly from the contrast-pair pool. The variance across seeds is measured;
> if cos(v_ICL_seed1, v_ICL_seed2) < 0.70, ICL vectors are too unstable to
> compare with DiffMean and the experiment is inconclusive.

**Q: Todd et al. 2023 already showed ICL creates "function vectors" that steer
behavior — isn't N4 just replicating their result?**

> Todd et al. showed ICL-derived vectors ARE causally effective; N4 asks whether
> the DIRECTION aligns with DiffMean specifically. Their paper uses ICL to create
> vectors; N4 cross-validates DiffMean against ICL as an independent derivation
> method. The overlap in result is the point — if they agree, both methods are
> measuring the same object.

**Q: What if the alignment is high at some layers but low at others? Which layer
adjudicates the hypothesis?**

> The layer of maximum alignment is the adjudicating layer. The prediction is that
> this layer coincides with the Fisher-optimal DiffMean layer (identified in the
> C1/C2 campaigns), which is independently determined before the alignment test.

## 10. Verification Checklist

- [ ] ICL-shift extraction code implemented and validated on synthetic linear model
- [ ] Cosine measurement at all 5 layers for all 5 behaviors
- [ ] Layer of max alignment compared to Fisher-layer (pre-registered match prediction)
- [ ] ICL-blended vector efficacy measured at w in {0, 0.25, 0.5, 0.75, 1.0}
- [ ] Demo-selection seed variance measured: cos(ICL_s1, ICL_s2) reported
- [ ] Result in EXPERIMENT_LEDGER.md, IDEA_TABLE.md N4 row updated

## 11. Status Journal

- 2026-05-31 — Design doc created. Status: UNTESTED. Todd et al. 2023 provides
  strong precedent (function vectors are ICL-derived and causally effective), but
  the DiffMean-ICL alignment has not been measured on Gemma models. S-5 (E2
  FALSIFIED: Fisher layer != best layer) is a warning: the pre-registered "max
  alignment = Fisher layer" prediction has a prior failure on a related prediction.

---

## Addendum: Research-Scientist Critique

### Prior plausibility

MEDIUM-HIGH. Todd et al. 2023 provides near-direct evidence that ICL creates
directional signals in activation space. The Garg et al. 2022 implicit gradient
descent framing provides a mechanistic explanation. The 0.60 cosine threshold is
conservative and achievable if the theoretical story is correct.

### Mechanism scrutiny

The implicit gradient descent story (Garg 2022) applies to ICL on regression tasks
in a specific theoretical regime (linear transformers, i-i-d in-context examples).
For behavioral ICL on instruction-tuned LLMs, the mechanism is more complex:
instruction tuning has already shaped the representation space, and the ICL signal
may be processed via a different circuit (e.g., induction heads) than the one DiffMean
taps. The 0.60 cosine prediction may be too optimistic for instruction-tuned models.

### Confounds

1. The format of ICL demonstrations (Q&A vs completion vs instruction) affects the
   extracted v_ICL direction. Different formats may give different cosines with v_DM.
2. Instruction tuning creates a "mode switch" from base model behavior; DiffMean on
   an instruction-tuned model may primarily capture the fine-tuning residual, while
   ICL taps a different circuit.
3. The query token's position embedding differs between ICL (many preceding tokens)
   and DiffMean extraction (few tokens), which can shift the activation distribution
   independent of behavior content.

### Does the alignment claim specifically matter?

MODERATELY. The main practical payoff is ICL as a cheap behavior-vector validation
tool and zero-shot extraction. Even if cos is 0.45 (below threshold), a partially
aligned vector could still be useful for initialization. The strict 0.60 threshold
may be too demanding; a more graded conclusion (higher cos = better extraction) would
be scientifically defensible.

### Literature precedent

Hernandez et al. 2023 (arXiv:2308.09729, "Linearity of Relation Decoding") and
the broader "linear representation hypothesis" literature suggest that many
behavioral distinctions ARE linearly represented, supporting the alignment claim.
But behavioral ICL alignment specifically for DiffMean-style safety/personality
vectors has not been tested.

### Skeptical effect-size estimate

cos(v_DM, v_ICL): 0.40-0.60 at optimal layer (vs claimed 0.60-0.80). Instruction
tuning confound likely reduces alignment for safety behaviors specifically. For
factual/stylistic behaviors (less fine-tuning pressure), alignment may be higher
(0.55-0.75). Recommend stratifying by behavior type in reporting.

### Minimum distinguishing experiment

One behavior, Gemma-3-1B, L16: compute v_ICL (5 demos, 3 seeds), v_DM (50 pairs),
report cos and ICL-derived vector efficacy. Cost ~30 min. If cos > 0.50, proceed
to full protocol; if cos < 0.40, the hypothesis is likely FALSIFIED.

### Verdict

TESTABLE-MEDIUM. The theoretical story is plausible and has strong precedent.
The 0.60 threshold is somewhat aggressive given the instruction-tuning confound.
The minimum 30-minute experiment should be the first step; commit to the full
protocol only after seeing the minimum result.

---

## Pseudocode & Methodology

This section specializes [`../METHODOLOGY.md`](../METHODOLOGY.md). N4 tests whether a DiffMean vector aligns with the activation shift produced by an in-context demonstration of the same behavior. **UNTESTED** — needs an ICL-shift extractor.

### 1. Steering-vector recipe (DiffMean vs ICL shift)

Two independently derived directions are compared by cosine:

```python
v_DM  = diffmean_vector(behavior, baseline, L)          # extract.diffmean_vector, 50 contrast pairs

# ICL shift (NEW): activation delta from prepending k=5 demos (no gradients)
h_demo = forward(model, [demo_1..demo_5, query]).hidden_state[L][query_pos]
h_none = forward(model, [query]).hidden_state[L][query_pos]
v_ICL  = h_demo - h_none

alignment = cos(v_DM, v_ICL)                            # the N4 quantity
v_blend   = (1-w) * v_DM + w * v_ICL                    # inject via add (METHODOLOGY §2) to test efficacy gain
```

### 2. Experiment procedure

```text
1. For 5 behaviors, extract v_DM (50 pairs) and v_ICL (5 demos, averaged over 3 demo-selection seeds).
2. Measure cos(v_DM, v_ICL) at layers {8,12,16,20,24}; record the layer of max alignment.
3. Sweep blend w in {0,0.25,0.5,0.75,1.0}; measure behavior efficacy at 5% off-shell (E7 displacement).
4. Stability check: cos(v_ICL_seed_i, v_ICL_seed_j) > 0.70 required for a valid comparison.
```

### 3. Measurement & decision rule

- **Primary metric:** cos(v_DM, v_ICL) at the optimal layer (for ≥3 of 5 behaviors).
- **Pre-registered falsifier (§3):** cos < 0.40 across all behaviors/layers ⇒ FALSIFIED; 0.40–0.59 ⇒ INCONCLUSIVE; <2% efficacy gain ⇒ practical implication fails.
- **Verdict logic:** SUPPORTED needs cos ≥ 0.60 AND ICL-blend efficacy gain ≥ 5%.

### 4. Where the code is / status

UNTESTED. DiffMean extraction (`extract_bank`) and the efficacy bundle exist; the **ICL-shift extractor** (demo-prepended forward at the query token, with seed-averaging) is the missing machinery — that is why N4 is UNTESTED.

See [`../METHODOLOGY.md`](../METHODOLOGY.md) for the shared recipe.

---

## Provenance & Tracing

No experiments run yet — see this design doc's protocol (§7) for what would be run. Once a campaign logs rows for this hypothesis, re-run `scripts/build_provenance.py` to generate `hypotheses/PROVENANCE/N4.md`.
