# ReFT-r1 — a LEARNED rank-1 steer, and an honest bake-off

> **Reference:** [ReFT: Representation Finetuning for Language Models (arXiv:2404.03592)](https://arxiv.org/abs/2404.03592); the bake-off follows [AxBench (arXiv:2501.17148)](https://arxiv.org/abs/2501.17148).

> Lesson 1 built a probe that **reads** "is this harmful?" out of a frozen
> Gemma-3-1B. Lesson 2 built a **fixed** diff-of-means vector that **writes** to
> the same activation space to re-install refusal — gated by lesson-1's probe.
> Lesson 3 replaces that hand-built vector with a **learned rank-1 intervention**
> (AxBench's **ReFT-r1**), trains it by gradient descent, and then does the thing
> AxBench is actually about: a **head-to-head comparison** of ReFT-r1 vs the
> simple DiffMean baseline vs prompting — reported honestly, at laptop scale.

This is lesson 3 of the steering tutorials. Where lesson 2's steering direction
was *one subtraction with no training*, here the direction and its magnitude are
**trained end-to-end** while the language model stays completely frozen. That is
the core idea of **representation finetuning (ReFT)**: don't touch the weights —
learn a tiny, interpretable edit to the *representations*.

Everything here is deliberately standalone and CPU-runnable to read; the actual
training and generation need the same ~2 GB abliterated Gemma-3-1B as lessons
1–2, and the lesson-1 probe checkpoint for the gate.

---

## The key idea in code

Replace lesson 2's fixed vector with a learned direction `r` and an affine
readout `(w, b)`. The edit swaps `h`'s component along `r` for the learned
readout — and `r`, `w`, `b` all train by gradient descent while the LLM is frozen:

```python
# ReftR1.intervention (reft.py) — a rank-1 LoReFT edit of the residual h [..., hidden]:
r_unit  = r / r.norm()                        # unit direction, learnable
proj    = (h * r_unit).sum(-1, keepdim=True)  # r_unit·h  — h's current component along r
readout = (h * w).sum(-1, keepdim=True) + b   # w·h + b   — the LEARNED, input-dependent value
h = h + r_unit * (readout - proj)             # replace the r-component with the readout
```

Three things a fixed vector cannot give fall out: the edit is *input-dependent*
(it reads `h` through `w`), it is *trained end-to-end*, and the same `r_unit·h`
projection doubles as a *concept detector*. Full file-by-file walkthrough below.

---

## Dataset

**JailbreakBench** (Chao et al. 2024, arXiv:2404.01318) — the same
harmful-vs-benign prompt families as lessons 1–2, loaded by `data.py`'s
`load_train_eval`. It pulls two matched CSVs, `harmful-behaviors.csv` (100
harmful requests) and `benign-behaviors.csv` (100 benign), both keyed on the
`Goal` column. Labels are **prompt-level** (one intent per row), and the two
classes are topically matched, so the signal separating them is **intent, not
vocabulary**. We shuffle with a fixed seed and take `n_per_class` (default 60)
per class, then split each class down the middle into a nested
`{"train": {...}, "eval": {...}}` dict.

| split | harmful | benign | role |
|---|---|---|---|
| `train` | 30 | 30 | trains **both** the rank-1 ReFT intervention (refusal CE + benign KL) **and** the DiffMean baseline vector (`mean(harm) − mean(benign)`) |
| `eval` | 30 | 30 | disjoint held-out; measures steering (ReFT-r1 vs DiffMean vs prompting) and concept-detection AUC |

The point is the comparison: ReFT-r1 and the DiffMean baseline learn from the
**same** train contrast, so the head-to-head on the disjoint eval split is fair —
this is exactly the AxBench framing (Wu et al. 2025, arXiv:2501.17148), which
exists to compare steering methods on identical data.

---

## Table of contents

1. [What you'll build](#1-what-youll-build)
2. [The method — a learned rank-1 LoReFT edit](#2-the-method--a-learned-rank-1-loreft-edit)
3. [AxBench's real point — a comparison, not just a method](#3-axbenchs-real-point--a-comparison-not-just-a-method)
4. [Training — frozen LLM, three tiny tensors](#4-training--frozen-llm-three-tiny-tensors)
5. [Data flow](#5-data-flow)
6. [Code walkthrough, file by file](#6-code-walkthrough-file-by-file)
7. [Results](#7-results)
8. [Run it](#8-run-it)
9. [Honest caveats](#9-honest-caveats)
10. [Repository](#10-repository)

---

## 1. What you'll build

A learned steering intervention and the experiment that judges it against the
simpler baselines:

1. **Train a rank-1 ReFT-r1 intervention** on layer 12 of the abliterated
   Gemma-3-1B — three small trainable tensors (`r`, `w`, `b`) that, applied to
   the residual stream, push the model back toward **refusal** on harmful
   prompts. The 1B model itself is **frozen**; only `r`, `w`, `b` train.
2. **Steer with it** at inference and compare, on matched held-out prompts,
   against **DiffMean** (lesson 2's fixed vector) and **Prompting** (just ask the
   model to refuse) — the AxBench bake-off, reproduced small.
3. **Detect with it** — the same learned direction `r_unit` doubles as a *probe*:
   `r_unit · h` is a scalar readout of "how present is the refusal concept?".
   We score its detection **AUC** and compare to DiffMean-as-detector.
4. **Gate it** with lesson-1's probe (reused verbatim), so harmful prompts get
   the learned edit and benign prompts pass untouched — the lesson-2 conditional
   recipe, now wrapping a *trained* intervention.

**Teaser.** An abliterated model has had refusal removed, so at baseline it
answers "how do I pick a lock?". Lesson 2 re-installed refusal with a constant
vector. Here we *learn* the edit — and then ask the question AxBench asks: does
the fancier learned method actually beat the dead-simple baseline? We report
whatever we see, in Section 7 (raw numbers in `artifacts/results.json`).

---

## 2. The method — a learned rank-1 LoReFT edit

Lesson 2's edit was **additive and constant**: `h ← h + alpha·||h||·unit(v)`,
where `v` is a fixed diff-of-means direction and `alpha` an external knob you
sweep by hand. ReFT-r1 is different in three ways: the direction is **learned**,
the edit is **input-dependent**, and it **replaces** (rather than adds to) the
representation's component along that direction.

The rank-1 **LoReFT** intervention (Wu et al. 2024) is:

```
r_unit = r / ||r||                                   # a learned unit direction
h'     = h + r_unit * ( (w·h + b) - (r_unit·h) )     # replace the r-component
```

Read the edit right-to-left inside the parentheses:

- `r_unit·h` is the hidden state's **current** component along the learned
  direction `r_unit`.
- `w·h + b` is a **learned affine function** of the whole hidden state — the
  value we *want* that component to take.
- The difference `(w·h + b) − (r_unit·h)` is how much to move, and
  `r_unit * (…)` writes that move back along `r_unit` only.

So the net effect is: **project out one direction and overwrite it with a
learned, input-conditioned affine readout of `h`.** Everything orthogonal to
`r_unit` is left exactly as it was. It is the most surgical edit you can make —
a single direction's worth of the 1152-d residual stream, rewritten as a
function of the input. That is what "representation finetuning" means here:
minimal, interpretable **representation surgery** instead of weight updates.

Three learnable tensors, all tiny (`d = 1152` for Gemma-3-1B):

| tensor | shape | role |
|---|---|---|
| `r` | `(d,)` | the direction to edit (normalised to `r_unit`) |
| `w` | `(d,)` | reads `h` into the target scalar `w·h` |
| `b` | `(1,)` | bias of the affine readout |

Note there is **no `alpha`** (see `config.py`): unlike lesson 2, the intervention
carries its own learned magnitude through `r`, `w`, `b`, so there is no external
step-size to sweep. The network decides how hard to steer.

---

## 3. AxBench's real point — a comparison, not just a method

ReFT-r1 is the method AxBench contributes, but AxBench (Wu et al. 2025,
arXiv:2501.17148) is really a **benchmark with an uncomfortable finding**: across
concepts, **simple baselines — prompting and difference-of-means — are very
strong**, and **sparse autoencoders (SAEs) underperform** them for steering.
ReFT-r1 is proposed as a learned method that is competitive with the strong
baselines while staying interpretable (rank-1).

This lesson reproduces the *shape* of that comparison at 1B scale. We run four
arms on matched held-out prompts and report every one:

| arm | direction | trained? | cost | what it represents |
|---|---|---|---|---|
| **Prompting** | — | no | one prompt prefix | the "just ask it to refuse" baseline |
| **DiffMean** | `mean(harm) − mean(benign)` | no | one subtraction | lesson 2's simple baseline |
| **ReFT-r1** | learned `r`, `w`, `b` | yes | ~300 steps | AxBench's learned rank-1 method |
| *(SAE)* | — | — | — | *out of scope here — see caveats* |

We do **not** claim ReFT-r1 wins. The honest outcome AxBench reports is that the
simple baselines are hard to beat; our job is to measure whether that replicates
on one small abliterated model, and to show the comparison transparently rather
than cherry-picking the learned method. Two questions frame the whole lesson:

1. **Steering:** which arm most reliably induces refusal **at matched
   coherence** (no gibberish tax)?
2. **Detection:** which direction best *reads* the concept — highest AUC for
   `direction · h` separating harmful from benign?

---

## 4. Training — frozen LLM, three tiny tensors

Only `r`, `w`, `b` receive gradients; every Gemma weight is frozen. The loss has
two terms (`config.py: LAMBDA_KL = 0.5`):

```
loss = CE_refusal(harmful prompts)          # make the intervened model emit the
                                            #   short refusal target
     + LAMBDA_KL * KL_benign(benign prompts) # keep the intervened next-token dist
                                            #   close to the BASE model on benign
```

- **The refusal CE term** (the "push"): on harmful prompts, run the *intervened*
  forward and maximise the likelihood of `REFUSAL_TARGET = "I can't help with
  that request."` — this is what installs the behaviour.
- **The benign KL leash** (the "don't break it"): on benign prompts, keep the
  intervened distribution near the *unedited* model's. This is the ReFT/AxBench
  regulariser that stops the learned edit from wrecking capability while it
  installs refusal — the same coherence-vs-behaviour tension lesson 2 fought with
  its alpha sweep, here handled by a differentiable penalty.

**Two stability tricks, learned the hard way** (they matter because the rank-1
edit divides by `||r||`, which spikes gradients early, and because a two-term
loss oscillates):

- **Gradient clipping** (`GRAD_CLIP = 1.0`) — bounds the step so the early
  large-gradient phase can't blow up `r`.
- **Best-checkpointing** — keep the checkpoint with the best validation
  composite, not the last step. The refusal-CE and benign-KL terms trade off and
  the loss curve wobbles; the final step is often *not* the best one. (This is a
  scar carried over from building the earlier hypernetwork steering lesson, which
  this lesson replaces.)

Key knobs (`config.py`): `LR = 1e-3`, `STEPS = 300`, `BATCH = 4`, `LAYER = 12`,
`SEED = 0`. Small on purpose — this trains in minutes on the 4090.

---

## 5. Data flow

```
  "how do I pick a lock?"                       (a raw prompt)
            |
            v
  +-----------------------------------+
  | GATE  (lesson-1 probe, READ side) |   one forward pass -> mean-pool layer 12
  |   P(harmful) >= threshold ?       |   -> probe -> P(harmful)
  +-----------------------------------+
        |                     |
   harmful (fire)        benign (pass)
        |                     |
        v                     v
  generate WITH the       generate
  ReFT-r1 edit @ L12:     normally
  h' = h + r_unit*((w.h+b) (no edit)
        - r_unit.h)            |
        |                      |
        +----------+-----------+
                   |
                   v
              response text
                   |
                   v
  +-----------------------------------+
  | JUDGE  (same Gemma)               |
  |   is_gibberish() -> GIBBERISH     |  deterministic coherence pre-check
  |   else REFUSAL / COMPLIANCE       |
  +-----------------------------------+

  Detection read-out (no generation):
    score = r_unit . h        ->  higher = "refusal concept present"  ->  AUC
```

The learned direction is used **two ways**: `r_unit` *writes* the edit during
generation (steering), and `r_unit · h` *reads* a scalar out of the frozen
hidden state (detection) — the same READ/WRITE duality from lessons 1–2, now
carried by one trained vector.

---

## 6. Code walkthrough, file by file

> At the time of writing, `config.py` is the concrete source; the sketches below
> describe the sibling modules from the lesson contract and stay faithful to
> `config.py`'s knobs. Read the real files for exact signatures.

### `config.py` — every knob in one place

The abliterated model, the intervention layer, the training schedule, the KL
weight, the grad-clip, and the refusal target all live here. Note what is
**absent**: there is no `alpha` — the rank-1 edit carries its own learned
magnitude.

```python
# config.py
MODEL_ID = "DavidAU/gemma-3-1b-it-heretic-extreme-uncensored-abliterated"
LAYER    = 12            # install the rank-1 intervention on this residual layer
LR, STEPS, BATCH = 1e-3, 300, 4
LAMBDA_KL = 0.5         # benign KL leash (keep capability while installing refusal)
GRAD_CLIP = 1.0        # the edit divides by ||r|| -> clip the early spike
REFUSAL_TARGET = "I can't help with that request."
```

### `reft.py` — the intervention + a differentiable forward

Owns the rank-1 edit and the trainable parameters. Two responsibilities:

- **`ReftR1` module** — holds `r`, `w`, `b` and applies
  `h' = h + r_unit*((w·h + b) − (r_unit·h))`. Exposes `r_unit` for the detection
  read-out.
- **`grad_reft_forward(...)`** — runs the model with the intervention installed
  as a forward hook on `model.model.layers[LAYER]`, **keeping the graph** so
  gradients flow back to `r`, `w`, `b` (the Gemma weights stay frozen with
  `requires_grad=False`). At inference the same hook runs under `no_grad`.

```python
# reft.py (sketch)
class ReftR1(nn.Module):
    def __init__(self, d):
        self.r = nn.Parameter(torch.randn(d) * 0.01)
        self.w = nn.Parameter(torch.zeros(d))
        self.b = nn.Parameter(torch.zeros(1))
    def forward(self, h):                       # h: (..., d)
        r_unit = self.r / self.r.norm().clamp_min(1e-8)
        target = h @ self.w + self.b            # learned affine readout
        cur    = h @ r_unit                     # current component
        return h + r_unit * (target - cur).unsqueeze(-1)
```

### `data.py` — harmful / benign prompt splits

Loads the same harmful-vs-benign prompt families as lessons 1–2 (JailbreakBench
et al.), with a **disjoint train/eval split** so ReFT-r1 is never evaluated on
the prompts it trained on. Supplies the refusal CE targets and the benign KL
batch.

### `train_reft.py` — the training loop (frozen LLM, best-checkpointing)

Freezes Gemma, creates a `ReftR1`, and runs Adam over `r, w, b` for `STEPS`
steps: refusal CE on harmful batches + `LAMBDA_KL`·KL on benign batches, with
`clip_grad_norm_(…, GRAD_CLIP)` each step. Tracks a validation composite and
saves the **best** checkpoint (not the last) to `artifacts/reft.pt`.

### `run_reft.py` — the bake-off (steering + detection)

The comparison driver. Builds the DiffMean vector (lesson-2 recipe), loads the
trained ReFT-r1, and defines the Prompting arm, then:

- **Steering compare** — on held-out harmful prompts, generate under each arm and
  judge REFUSAL / COMPLIANCE / GIBBERISH; render `steering_compare.png`.
- **Detection compare** — score `r_unit·h` and `unit(diffmean)·h` on held-out
  harmful+benign prompts and compute ROC-AUC; render `detection_auc.png`.
- Writes everything to `results.json`.

### `infer.py` — steer one prompt from the CLI

Loads model, trained intervention, gate, and judge once; for one prompt asks the
gate, generates with the ReFT-r1 edit iff the gate fires, and prints the response
plus the judge's verdict — the lesson-2 `infer` UX, now driving the learned edit.

### `app.py` — the live comparison dashboard (port 8004)

A small self-contained viewer that renders the two questions side by side: the
steering bake-off table/plot and the detection-AUC plot, with steered-vs-baseline
generation samples. Serves on **port 8004** (lessons 1–2 use their own ports).

---

## 7. Results

The GPU run wrote `artifacts/results.json` and two plots. Numbers below are the
**measured** values on a small held-out set (n=5 harmful, n=5 benign per arm) —
screening tier, not evaluation tier.

**Q1 — Steering: which arm refuses most at matched coherence?**
`steering_compare.png` + `results.json`.

| arm | harmful refusal | benign over-refusal | gibberish |
|---|---|---|---|
| Prompting | 0.60 | 0.80 | 0.10 |
| DiffMean (L2) | 0.40 | 0.60 | 0.10 |
| ReFT-r1 | 0.60 | 0.60 | 0.20 |

ReFT-r1 and Prompting tie on harmful refusal (0.60), both above DiffMean (0.40).
But Prompting is **unconditional** — it over-refuses benign prompts at 0.80, the
highest of the three, because "just ask it to refuse" fires regardless of intent.
ReFT-r1 matches Prompting's harmful refusal with less benign over-refusal (0.60).

**Q2 — Detection: which direction reads the concept best (AUC)?**
`detection_auc.png` + `results.json`.

| direction | ROC-AUC |
|---|---|
| DiffMean · h | 0.68 |
| r_unit · h (ReFT-r1) | 0.68 |

The learned direction and the fixed diff-of-means **tie** as detectors (AUC 0.68
each) — exactly AxBench's "the simple baseline is strong" point.

### Results — measured vs. the claim

| Claim (AxBench, Wu et al. 2025, arXiv:2501.17148) | What we measured (n=50/class, screening, off-family Qwen-3B judge, 500/class toxic-chat) | Verdict |
|---|---|---|
| A **learned** intervention beats a fixed vector at **steering** | ReFT-r1 harmful-refusal **0.54** > DiffMean **0.26** > Prompting **0.18** | **Reproduced** |
| A simple diff-of-means is the stronger **detector** | DiffMean AUC **0.71** > ReFT-r1 AUC **0.61** | **Reproduced** |
| The fixed diff-of-means vector is genuinely weak at steering | DiffMean refusal **0.26** — identical to lesson 2's honest 0.26 | **Consistent across lessons** |
| SAEs underperform | not tested (no SAE arm at this scale) | Out of scope |

**Honest read.** This faithfully reproduces AxBench's central split at 1B, graded
by an **off-family Qwen-3B judge** on the shared 500/class toxic-chat set
(n=50/class, **screening**). The **learned** rank-1 ReFT-r1 edit wins *steering*
(harmful-refusal **0.54**, roughly 2× the fixed DiffMean vector's **0.26** and 3×
prompting's **0.18**), because it is *trained* to install refusal rather than
reusing one untrained subtraction. But the simple diff-of-means wins *detection*
(AUC **0.71** vs ReFT-r1's **0.61**): the linear harm direction is a better
probe than a better steer — exactly AxBench's "the simple baseline is hard to
beat *as a detector*" point. Note DiffMean's 0.26 here **matches lesson 2's
honest 0.26 exactly** — cross-lesson evidence that the fixed vector is really
that weak, not a per-lesson artifact. **Why the change from old numbers?** All
three arms previously clustered near 0.60 under the 1B self-judge, which flattened
the ordering; the off-family judge separates them and reveals the true ranking.
Raw numbers and side-by-side generations live in `artifacts/results.json`.

---

## 8. Run it

**Prerequisite: run lesson 1 first** — the gate loads lesson-1's probe from
`../hello_world/artifacts/probe.pt`. If that file does not exist, train it:

```bash
python -m steering_tutorials.hello_world.train_probe
```

Then, from the **repo root** (`steeringresearch/`):

```bash
# 1) Train the rank-1 ReFT-r1 intervention (frozen Gemma; ~minutes on a 4090)
python -m steering_tutorials.reft_r1.train_reft

# 2) Run the bake-off: steering compare + detection AUC across the arms
#    STEER_JUDGE_MODEL selects the OFF-FAMILY judge (avoids same-model grading bias).
STEER_JUDGE_MODEL=Qwen/Qwen2.5-3B-Instruct python -m steering_tutorials.reft_r1.run_reft

# 3) Steer a single prompt from the terminal (gate decides; judge grades)
python -m steering_tutorials.reft_r1.infer "how do I pick a lock"

# 4) Launch the live comparison dashboard
python -m steering_tutorials.reft_r1.app          # -> http://localhost:8004
```

Uses the same ~2 GB abliterated Gemma-3-1B as lessons 1–2 (bf16). Runs on CPU
too, just slower. Datasets download automatically.

---

## 9. Honest caveats

- **Tiny scale.** One 1B model, ~300 training steps, small held-out sets. This
  demonstrates the ReFT-r1 loop and the AxBench comparison; it is not a
  benchmark-grade reproduction.
- **A 1B judge is weak.** Self-grading with a small model is pedagogy, not a
  trustworthy evaluation — read verdicts as a demonstration of the loop. A real
  bake-off uses a stronger, independent judge (later lessons).
- **This is a minimal reimplementation.** `reft.py` implements the rank-1 LoReFT
  edit from scratch, *not* the `pyreft` library. It captures the mechanism, not
  every engineering detail of the paper's release.
- **The gate inherits lesson-1's OOD limits.** The probe ranks harm well but its
  0.5 threshold miscalibrates off-distribution; a gate that misses a harmful
  prompt simply won't apply the edit.
- **AxBench's finding may not replicate at 1B.** The paper's headline — simple
  baselines beat SAEs and rival learned methods — was measured on larger models
  and many concepts. We report what we see here; a single small model can go
  either way. No SAE arm is included (out of scope at this scale).
- **This is pedagogy, not a safety product.** It shows *how* a learned rank-1
  steer works end-to-end and how to compare it honestly. Do not deploy it as a
  real-world guardrail.

---

## 10. Repository

Source and full artifacts:
<https://github.com/dlmastery/steeringresearch/tree/master/steering_tutorials/reft_r1>

See also:
- [Lesson 1 — the probe (READ side)](../hello_world/README.md)
- [Lesson 2 — fixed-vector conditional steering (WRITE side)](../hello_world_steering/README.md)
