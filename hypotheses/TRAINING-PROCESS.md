# TRAINING-PROCESS.md — Is there any training per hypothesis? (Short answer: no.)

> **Read this if you are wondering "where is the training loop for each
> hypothesis?"** The honest, code-verified answer is that **almost nothing in
> this project is trained in the gradient-descent sense.** Activation steering is
> an **inference-time** technique: the model's weights are **frozen** and never
> updated. What looks like "training" is a one-shot **vector-extraction** step —
> a mean difference or a single SVD over cached activations, with no optimizer,
> no loss function, no backward pass, and no epochs.
>
> This document explains exactly what *does* happen per hypothesis, where it is
> in the code, and which (untested) hypotheses are the only ones that would
> require real training.

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
`requires_grad`, or `epoch` inside `src/steering/` returns **nothing**. There is
no trainer module because there is nothing to train.

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

## 4. The only hypotheses that WOULD involve training

A small minority of the 70 hypotheses propose a *learned* component. These are
**all currently `PENDING — UNTESTED`**, and the reason they are untested is
precisely that the training infrastructure for them **does not exist in this repo
yet** — they are honestly flagged, not silently skipped:

| Hypothesis | What would be trained | Status |
|---|---|---|
| **E15** — Learned gate vs fixed threshold | A small **logistic-regression gate** on multi-layer activations (supervised, needs an OOD eval set). | UNTESTED — needs gate-training infra |
| **E45** — HyperSteer zero-shot | A **hypernetwork** that emits steering vectors for unseen concepts (this is a genuinely trained network). | UNTESTED — needs a trainer |
| **E20 / SAE-based ideas** | Steering on **sparse-autoencoder features**. The SAE itself is trained — but *upstream*, by its authors; we would *load* a pretrained SAE, not train one here. | UNTESTED — needs SAE integration |

When/if these are implemented, training will be **confined to that small
auxiliary component** (a gate, a hypernetwork, or a borrowed pretrained SAE) — the
base Gemma model still stays frozen. At that point this document will be updated
with the exact trainer location, the optimizer/loss, and the training data, and
those hypotheses' `PROVENANCE/<ID>.md` files will record checkpoints alongside the
config sweep.

---

## 5. FAQ

**Q. So the steering vector is not learned at all?**
It is *estimated* from data (a statistical fit: a mean or a principal component),
but it is **not optimized** with gradients. No loss, no epochs. One pass to
collect activations, one linear-algebra step to get the direction.

**Q. Why does this count as research if nothing is trained?**
Because the open questions are about the **geometry and control** of a frozen
model's representations: *which* layer, *which* direction, *how much* to push,
*when* to gate, *how* methods stack — and whether any of it holds up under the
5-axis composite at matched coherence. Those are empirical questions answerable
with inference-only experiments. See `README.md` ("what SOTA steering means
here") and `FINDINGS.md`.

**Q. Where do I see what was done for one hypothesis?**
`hypotheses/PROVENANCE/<ID>.md` — exact experiment numbers, the reproduce command,
the result JSON, and the reasoning trace. Those runs are inference sweeps, which
is why you will see layer/alpha/operation columns and **not** training curves.

**Q. Is the model ever fine-tuned as a baseline?**
No. Fine-tuning is the *thing steering is trying to avoid* (it is expensive and
overwrites capability). The whole point is **control without retraining the
weights** — that is the headline claim the README and dashboard make.

---

*Code references verified against `src/steering/extract.py`, `hooks.py`, and a
repo-wide search confirming zero optimizer/backward/epoch usage in `src/steering/`.
Composite fingerprint `a9001e87087e`.*
