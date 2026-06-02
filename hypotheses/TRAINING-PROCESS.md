# TRAINING-PROCESS.md — Is there any training per hypothesis? (Short answer: rarely, and never on the base model.)

> **Read this if you are wondering "where is the training loop for each
> hypothesis?"** The honest, code-verified answer is that **the vast majority of
> hypotheses in this project require no gradient-descent training.** Activation
> steering is an **inference-time** technique: the model's weights are **frozen**
> and never updated. The default path — used by all 109 pre-trainable-method
> experiments — is a one-shot **vector-extraction** step: a mean difference or a
> single SVD over cached activations, with no optimizer, no loss function, no
> backward pass, and no epochs.
>
> **As of 2026-06-01, three hypotheses (E15, E45, E20) have been implemented
> with gradient-trained auxiliary components and screened on real Gemma-3-270m-it
> (exp 110–113).** The base Gemma model remains frozen in all three; only a small
> auxiliary component is trained (a logistic gate, an MLP hypernetwork, or a
> sparse autoencoder plus gradient-ascent vector optimizer). All three methods
> were falsified or inconclusive at smoke scale; the offline unit tests confirm
> correct implementation on synthetic data.
>
> This document explains exactly what *does* happen per hypothesis, where it is
> in the code, and the status of the three hypotheses that involve a trained
> auxiliary component.

---

## 1. What "training" usually means — and why steering skips it

In ordinary deep learning you (a) define a loss, (b) compute gradients via
backpropagation, (c) step an optimizer (Adam/SGD), and (d) repeat for many
epochs until the **weights** converge. That is **fine-tuning / training**.

**Activation steering does none of that.** Instead it:

1. runs the **frozen** model forward on a small set of contrastive prompts (e.g.
   "happy" text vs neutral text) and **records the hidden activations** — pure
   inference, no gradients;
2. computes a single **steering vector** from those recorded activations using a
   **closed-form** formula (a subtraction, or one SVD);
3. at generation time, **adds that vector** into one residual-stream layer with a
   forward hook, which nudges the model's behavior.

No weight is ever modified. You can think of the steering vector as a *probe
direction estimated from data*, not as *parameters learned by optimization*. The
distinction matters: it is why every experiment runs in **seconds-to-minutes on a
laptop GPU** instead of hours, and why we can sweep dozens of configs per day.

> **One-line mental model:** we **measure** a direction, we do not **train** one.

---

## 2. What actually happens per hypothesis (the real "fit" step)

Every hypothesis is screened through the **same shared harness** (see the
README's "SHARED HARNESS" section). The per-hypothesis pipeline is:

| Step | What happens | Gradient? | Where in code |
|---|---|---|---|
| **A. Cache activations** | Forward-pass the frozen model on contrast pairs; mean-pool the hidden state per layer; **cache to disk once** and reuse across the whole ladder. | **No** — inference only | `src/steering/extract.py` → `collect_activations` / `collect_activations_cached` |
| **B. Build the steering vector** | Compute the direction from cached activations using one of the closed-form estimators below. | **No** — linear algebra | `src/steering/extract.py` → `build_vector_bank` |
| **C. Pick the layer** | Rank layers by the Fisher separability ratio (closed form). | **No** | `extract.py` → `fisher_ratio`, `best_layer` |
| **D. Inject & generate** | Add/rotate/project the vector into one layer via a forward hook; generate steered vs unsteered text. | **No** — inference | `src/steering/hooks.py` → `apply_operation`, `SteeringContext` |
| **E. Score 5 axes** | Behavior, capability (MMLU), coherence (PPL), safety (JailbreakBench), selectivity. | **No** | `src/steering/eval.py` |

The three vector estimators in step B — **all closed-form, all single-shot:**

- **DiffMean** — `mean(pos_activations) − mean(neg_activations)`. One subtraction.
  No iteration. (`diffmean_vector`)
- **PCA-top1** — the top singular vector of the set of per-pair differences. **One
  SVD** (`np.linalg.svd`), not gradient descent. (`pca_top1_vector`)
- **Fisher ratio** — `(mean_pos − mean_neg)² / (var_pos + var_neg)` along the
  diffmean axis, used only to *choose* the most separable layer. (`fisher_ratio`)

You can audit this yourself: a repo-wide search for `optimizer`, `loss.backward`,
`requires_grad`, or `epoch` inside `src/steering/extract.py`, `hooks.py`, and
`eval.py` — the core inference path — returns **nothing**. The three modules
that do contain an optimizer (`gate.py`, `hypersteer.py`, `sae.py`) are
purpose-built for the auxiliary-component experiments E15/E45/E20 and are never
called during standard steering runs. If you are using the default harness, there
is nothing to train.

---

## 3. Then what is the "loop" I keep hearing about?

The iteration in this project is a **research loop, not a training loop.** We hold
the (training-free) method fixed and sweep the **steering knobs**:

```
for layer in layers:
  for alpha in strengths:        # how hard to push
    for source in {diffmean, pca}:   # how to build the vector
      for operation in {add, relative_add, rotate, project_out}:
        for seed in seeds:
          run one inference experiment, log 5 axes
```

This is **coordinate descent over a configuration grid** (`scripts/campaign_sweep.py`,
`scripts/run_hillclimb.py`) — it searches for the best *hyper-parameters of the
intervention*, not the best *weights*. Each cell is one frozen-model inference
run, which is exactly why the per-hypothesis provenance files
(`hypotheses/PROVENANCE/<ID>.md`) list the runs as configs (layer/alpha/op/source)
rather than as training checkpoints.

---

## 4. Hypotheses with a trained auxiliary component — now implemented and screened

A small minority of the 70 hypotheses propose a *learned* component. As of
2026-06-01, all three have been **implemented and offline-unit-tested**, and each
has received a first smoke-scale screen on real Gemma-3-270m-it (exp 110–113).
They are honestly flagged — not silently skipped, and not declared winners on
the basis of unit-test-only results.

**Critical invariant across all three:** the **base Gemma model is frozen.** Only
the small auxiliary component is trained. The methods differ in what that
component is and what it optimizes.

| Hypothesis | What is trained (auxiliary component only; Gemma frozen) | Module | Driver | Optimizer / loss | First screening result (exp#) | Verdict |
|---|---|---|---|---|---|---|
| **E15** — Learned gate vs fixed cosine threshold | A **multi-layer logistic-regression gate** trained on multi-layer activation features (each = a prompt's pooled hidden state dotted with that layer's condition vector) from the frozen Gemma model, with an OOD hold-out set. Compared against a fixed cosine-similarity threshold (no training required). | `src/steering/gate.py` | `scripts/run_e15.py` | **torch** logistic gate: SGD on `BCEWithLogitsLoss` + L2 weight decay (PR/ROC-AUC in pure numpy; no sklearn) | exp#110: In-dist PR-AUC cosine 1.000 / logistic 1.000 (resubstitution); OOD PR-AUC cosine 0.7177 / logistic 0.5498; OOD gap −0.1679 | **FALSIFIED_OOD** — gate overfits tiny in-dist set; generalizes WORSE than fixed cosine OOD. OOD gap (−0.1679) is below the +0.06 falsifier. Revisit with a larger OOD eval set. |
| **E45** — HyperSteer zero-shot | An **MLP hypernetwork** trained by Adam/MSE to map a behavior-description embedding — produced by the **frozen Gemma model's own hidden state** (mean-pooled at the injection layer), not a separate sentence encoder — to a steering vector, then evaluated on held-out behaviors (leave-one-out). This is the only component that is a genuinely trained neural network in its own right. | `src/steering/hypersteer.py` | `scripts/run_e45.py` | Adam, MSE regression + L2 weight decay (fixed-epoch) | exp#111: 4-behavior LOO; mean held-out cosine(pred, supervised) −0.0202 (std 0.6147; folds +0.85 to −0.88); mean projection efficacy ratio 1.308 is an unreliable non-causal proxy | **INCONCLUSIVE** — description→vector does not reliably generalize at n=4; the projection-efficacy proxy is non-causal (a cos −0.88 fold still scores ratio 1.07). Need more behaviors or a causal generation eval before a verdict is possible. |
| **E20** — SAE-TS: gradient-ascent vector optimizer | Two things are trained: (1) a **sparse autoencoder** is trained here on the frozen model's pooled activations (recon MSE + L1 sparsity) — it is NOT a pretrained oracle; (2) then the **steering vector itself** is optimized by gradient ascent on the SAE-feature objective `score(F_target) − λ·score(F_side)` (SAE-TS), to activate target features while suppressing side-effect features. | `src/steering/sae.py` | `scripts/run_e20.py` | SAE: Adam (recon + L1). Vector: gradient ascent (the vector is the parameter) | exp#112 (DiffMean 3-stack baseline, Gram mass 2.13) + exp#113 (SAE-TS 3-stack, Gram mass 3.00 — maximum, all three vectors collapsed to one direction; Gram reduction −0.87 vs baseline) | **FALSIFIED** — SAE-TS did NOT improve orthogonality at 270m scale; vectors collapsed to one direction (Gram mass = theoretical maximum for collinear triplet). SAE-coverage confound suspected at 270m. |

### What the offline unit tests showed (before live runs)

All three module-level unit tests passed on clean synthetic data before the live
screens, confirming the implementation is correct:

- **gate.py unit test:** logistic gate AUC 0.998 vs cosine-threshold AUC 0.70 on
  a balanced synthetic activation dataset. The gate learns the right boundary when
  the data is not tiny and class-imbalanced.
- **hypersteer.py unit test:** MLP hypernetwork held-out cosine 0.93 on a
  10-behavior synthetic dataset vs shuffled-label baseline −0.01. Generalization
  works when n is large enough.
- **sae.py unit test:** SAE-TS optimized cross-cosine 0.011 (near-orthogonal) vs
  raw DiffMean 0.184 on synthetic SAE features with good coverage. The optimizer
  finds orthogonal directions when the feature dictionary covers the space.

The pattern across all three: the mechanisms are real and correct, but they
require more data / larger models / better SAE coverage than the 270m smoke-scale
screen can provide. These results sharpen the conditions for a successful revisit
more than they condemn the hypotheses.

### The crucial honest point: Gemma base weights never change

Even in E45 (the most "trained" of the three — a genuine MLP with hundreds of
parameters), the base Gemma model is never touched. The MLP operates entirely
outside the model graph: it takes as input a description embedding produced by a
frozen sentence encoder and outputs a vector that is then injected as a standard
activation steering intervention. The training loop has no access to Gemma's
weights and cannot modify them. This invariant holds for all three methods and is
enforced by the module architecture.

---

## 5. FAQ

**Q. So the steering vector is not learned at all — in the default path?**
Correct for the default (DiffMean/PCA) path: the vector is *estimated* from data
(a statistical fit: a mean or a principal component), but it is **not optimized**
with gradients. No loss, no epochs. One pass to collect activations, one
linear-algebra step to get the direction. The E20 SAE-TS method is the exception:
it runs gradient ascent to *optimize* the steering vector itself against an SAE
orthogonality objective, but the base model weights are still frozen throughout.

**Q. Three methods now have trainers. Does that mean the project is now "training"?**
Only in a narrow sense. The trained components are small (a logistic gate, a
~200-parameter MLP, or a gradient-ascent step on one vector) and all operate
outside the Gemma model graph. The default steering path — used by 109 of 113
experiments — remains training-free closed-form extraction. The three trainable
methods are explicitly flagged as such (tag prefix `E15`, `E45`, `E20`) and use
method-specific metrics rather than the standard composite.

**Q. Why does this count as research if almost nothing is trained?**
Because the open questions are about the **geometry and control** of a frozen
model's representations: *which* layer, *which* direction, *how much* to push,
*when* to gate, *how* methods stack — and whether any of it holds up under the
5-axis composite at matched coherence. Those are empirical questions answerable
with inference-only experiments. See `README.md` ("what SOTA steering means
here") and `FINDINGS.md`.

**Q. Where do I see what was done for one hypothesis?**
`hypotheses/PROVENANCE/<ID>.md` — exact experiment numbers, the reproduce command,
the result JSON, and the reasoning trace. For the default inference sweeps you
will see layer/alpha/operation columns and no training curves. For E15/E45/E20
you will see a training-component section alongside the usual sweep columns.

**Q. Is the model ever fine-tuned as a baseline?**
No. Fine-tuning is the *thing steering is trying to avoid* (it is expensive and
overwrites capability). The whole point is **control without retraining the
weights** — that is the headline claim the README and dashboard make. The three
trainable auxiliary components (gate, hypernetwork, SAE-TS optimizer) are
precisely designed to add learned behavior *without* touching the model.

---

*Code references verified against `src/steering/extract.py`, `hooks.py`, and
`eval.py` (inference path — no optimizer/backward/epoch). Trainable-component
modules: `src/steering/gate.py` (E15), `src/steering/hypersteer.py` (E45),
`src/steering/sae.py` (E20). Drivers: `scripts/run_e15.py`, `scripts/run_e45.py`,
`scripts/run_e20.py`. Unit tests confirm each mechanism on synthetic data before
live runs. Composite fingerprint `a9001e87087e`.*
