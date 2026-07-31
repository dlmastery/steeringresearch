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


> **Instrument caveat — read before citing any rate on this page.** Every refusal /
> compliance / gibberish number here is scored by a local LLM judge. That judge family
> was calibrated against ground-truth labels and measured **ROC-AUC 0.665–0.751** — below
> the 0.85 bar this course sets for a trustworthy instrument. Small differences between
> arms therefore sit at or below the judge's own noise floor and must not be read as
> effects. See [`../JUDGE_VALIDITY.md`](../JUDGE_VALIDITY.md) for the calibration, the
> continuous-readout improvement (+0.086 AUC, 12.8x faster), and which claims survive.

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

The shared **≥500/class harmful-vs-benign** foundation,
`steering_tutorials.common.data.load_harmful_benign` (toxic-chat, prompt-level
intent labels, deduped and length-matched; built on JailbreakBench + `lmsys/toxic-chat`,
arXiv:2404.01318 / arXiv:2310.17389), loaded by `data.py`'s `load_train_eval`. We
take `N_PER_CLASS = 500` per class and split each class disjointly into a nested
`{"train": {...}, "eval": {...}}` dict.

| split | harmful | benign | role |
|---|---|---|---|
| `train` | 300 | 300 | trains **both** the rank-1 ReFT intervention (refusal CE + benign KL) **and** the DiffMean baseline vector (`mean(harm) − mean(benign)`) |
| `eval` | 200 | 200 | disjoint held-out; measures steering (ReFT-r1 vs DiffMean vs prompting) and concept-detection AUC |

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
simple baselines are hard to beat, and our job is to show the comparison
transparently rather than cherry-pick the learned method.

**One structural caveat, stated up front so §7 is not misread:** our base model is
abliterated, so its instruction-following refusal has been removed. That is the
exact faculty the **Prompting** arm depends on, and no other arm depends on it.
The steering bake-off below is therefore **not a like-for-like test of AxBench's
headline** — a learned residual-stream edit beating prompting on a model that
cannot be prompted into refusing is close to a tautology. The **detection**
comparison is unaffected and remains a fair contest.

Two questions frame the whole lesson:

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

Loads the shared ≥500/class harmful-vs-benign set (`common.data`, toxic-chat)
via `load_train_eval`, with a **disjoint train/eval split** so ReFT-r1 is never evaluated on
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
**measured** values from that file: **n = 200 harmful and n = 200 benign per arm**
(the disjoint eval split of the 500/class set), single seed — screening tier, not
evaluation tier.

**Q1 — Steering: which arm refuses most at matched coherence?**
`steering_compare.png` + `results.json` → `steering`.

| arm | harmful refusal | benign over-refusal | gibberish | n (harm / benign) |
|---|---|---|---|---|
| Prompting | 0.325 | 0.475 | 0.1375 | 200 / 200 |
| DiffMean (L2) | 0.285 | 0.475 | 0.1825 | 200 / 200 |
| ReFT-r1 | **0.605** | 0.495 | **0.090** | 200 / 200 |

ReFT-r1 refuses roughly twice as often as either baseline (0.605 vs 0.325 and
0.285) and does it with the *lowest* gibberish (0.090). Benign over-refusal is
essentially flat across all three arms (0.475–0.495), so no arm here is more
selective than another — that floor is set by the abliterated base plus the judge,
not by the method. Read the steering column with the like-for-like caveat in the
claim table below: the Prompting arm is running on a model whose instruction-
following refusal was deliberately ablated.

**Q2 — Detection: which direction reads the concept best (AUC)?**
`detection_auc.png` + `results.json` → `detection`.

| direction | ROC-AUC | n (harm / benign) |
|---|---|---|
| DiffMean · h | **0.748** | 200 / 200 |
| r_unit · h (ReFT-r1) | 0.568 | 200 / 200 |

The two directions **do not tie**. The fixed diff-of-means is much the better
detector (0.748 vs 0.568) — the learned direction is optimized to *write* refusal,
not to *read* harm, and the gap is the price of that specialization. This is
AxBench's "the simple baseline is strong" point, and it is a clean one-directional
result, not a tie.

### Results — measured vs. the claim

| Claim (AxBench, Wu et al. 2025, arXiv:2501.17148) | What we measured (train 300/class, **eval 200/class**, off-family Qwen-3B judge, 500/class toxic-chat) | Verdict |
|---|---|---|
| AxBench's headline: **prompting outperforms existing steering methods** | ReFT-r1 harmful-refusal **0.605** > Prompting **0.325** > DiffMean **0.285** — our ordering is the *opposite* | **Not a like-for-like test — see below.** Our base model is **abliterated**, which structurally cripples the prompting arm. This neither reproduces nor refutes the paper's claim. |
| A simple diff-of-means is the stronger **detector** | DiffMean AUC **0.748** > ReFT-r1 AUC **0.568** | **Consistent with AxBench** — this arm *is* like-for-like (both directions read the same frozen activations; no instruction-following involved) |
| The fixed diff-of-means vector is genuinely weak at steering | DiffMean refusal **0.285** — close to lesson 2's honest ~0.10–0.33 range | **Consistent across lessons** |
| SAEs underperform | not tested (no SAE arm at this scale) | Out of scope |

**Why the steering row is not a reproduction.** AxBench's uncomfortable finding is
that *prompting* — just telling the model to behave — beats the learned and
sparse-autoencoder steering methods it benchmarks. Our target is
`DavidAU/gemma-3-1b-it-heretic-extreme-uncensored-abliterated`: a model whose
refusal behaviour has been **deliberately ablated**. Prompting steers by invoking
exactly the instruction-following refusal circuitry that abliteration removed,
while ReFT-r1 steers by writing into the residual stream, which abliteration does
not touch. So the two arms are not competing on equal footing — we have handicapped
one of them at the model level, before any measurement. ReFT-r1 scoring 0.605
against prompting's 0.325 on this base is the expected consequence of that choice,
not evidence about the paper's claim. A like-for-like test needs a non-abliterated
instruct model, where prompting's arm is intact. Until then, treat the steering
ordering on this page as a fact about *this* model, not about the methods.

**What does survive.** The detection split is a fair comparison — both arms are
scalar read-outs of the same frozen hidden states, and abliteration gives neither
an advantage. There the simple diff-of-means clearly wins (AUC **0.748** vs
**0.568**), which is genuinely the "simple baselines are strong" spirit of AxBench.
The ordering also held at the earlier smaller run (ReFT-r1 0.54 > DiffMean 0.26 on
steering; DiffMean detection 0.71 > 0.61), so the larger n = 200/class eval firmed
it up rather than changing it. One honest cost across the board: benign
over-refusal is high in every arm (0.475–0.495, ReFT-r1 marginally the highest),
dominated by the abliterated base + judge, so no arm is meaningfully *more
selective* here. Raw numbers and side-by-side generations live in
`artifacts/results.json`.

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
- **The prompting arm is handicapped by design, so AxBench's headline is not
  tested here.** The paper's finding — prompting outperforms existing steering
  methods — presumes a model whose instruction-following refusal is intact. Ours
  is abliterated, precisely so that external refusal vectors have something to
  re-install. That makes the ReFT-r1-beats-prompting result on this page a
  property of the base model, not a verdict on the paper (see §7). The paper was
  also measured on larger models and many concepts, and no SAE arm is included
  here (out of scope at this scale).
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
