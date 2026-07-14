# HyperSteer — from a FIXED vector to a LEARNED generator

> Lesson 1 built a probe that **reads** "is this harmful?" out of a frozen
> Gemma-3-1B. Lesson 2 **writes** back: a single **fixed** diff-of-means vector,
> added to the residual stream, re-installs refusal in an abliterated model —
> gated by the lesson-1 probe. Lesson 3 replaces that hand-built vector with a
> small **hypernetwork** `H` that *generates* the steering vector from a concept:
> `v = H(concept)`. Train `H` once; then emit a steering vector for a **new**
> concept from just its exemplars — no re-extraction.

This is the "hello world" of **learned, amortized steering**. If you have worked
through [lesson 2](../hello_world_steering/README.md) you have the WRITE side
with a *fixed* vector. Here you get the GENERATE side: a network that outputs the
steering vector, trained by backpropagating **through the frozen language model**.

The three lessons form one arc:

| Lesson | Verb | What it does |
|---|---|---|
| 1 `hello_world` | **READ** | a probe reads harm from layer-12 activations |
| 2 `hello_world_steering` | **WRITE** | a *fixed* diff-of-means vector steers toward refusal, gated by the probe |
| 3 `hypersteer` | **GENERATE** | a hypernetwork `H` *learns to produce* the steering vector: `v = H(concept)` |

The payoff of lesson 3 is **amortization**: once `H` is trained, a brand-new
concept needs only a handful of exemplars to get its own steering vector — no new
contrast pass, no re-extraction — and the learned nonlinear concept→vector map
can match or beat the hand-built vector at matched coherence.

Everything here is standalone and CPU-runnable to read; the actual
training/eval needs the same ~2-3 GB abliterated Gemma-3-1B as lessons 1 and 2
(training included).

---

## Table of contents

1. [What you'll build](#1-what-youll-build)
2. [Concept: what a hypernetwork is](#2-concept-what-a-hypernetwork-is)
3. [The design](#3-the-design)
4. [Training — the key new idea](#4-training--the-key-new-idea)
5. [Data flow](#5-data-flow)
6. [Code walkthrough, file by file](#6-code-walkthrough-file-by-file)
7. [Results](#7-results)
8. [Run it](#8-run-it)
9. [Honest caveats](#9-honest-caveats)
10. [Repository](#10-repository)

---

## 1. What you'll build

A small hypernetwork `H_theta` that turns a **concept embedding** into a
**steering vector**, plus the training loop that fits it and the head-to-head
harness that grades it against lesson 2's fixed vector.

1. **Build a concept embedding** — the mean last-token layer-12 activation over a
   concept's exemplar prompts (the "refusal" concept, to start).
2. **Generate a steering vector** — `v = H(concept_embedding)`, a nonlinear map
   from the concept to a direction in the 1152-d residual stream.
3. **Steer conditionally** — add `v` at layer 12 via lesson-2's relative-add
   rule, gated by lesson-1's probe (steer harmful prompts, leave benign alone).
4. **Train `H` through the frozen model** — freeze Gemma, backprop a refusal loss
   (plus a benign regularizer) into `H` only.
5. **Judge and compare** — the same 1B Gemma grades each output; the harness asks
   two questions: does the learned vector *match/beat* the fixed CAA vector, and
   does it *generalize* to a new concept zero-shot.

**How this advances lesson 2.** Lesson 2's vector is `mean(harmful) −
mean(benign)`: one subtraction, frozen forever, one concept. Lesson 3 keeps the
*same* injection machinery but makes the vector the **output of a trained
function of the concept**. That is the whole move — from a lookup to a generator.

---

## 2. Concept: what a hypernetwork is

A **hypernetwork** is a network that outputs the parameters of *another* network
or module (Ha, Dai, Le 2016, *HyperNetworks*, arXiv:1609.09106 — [UNVERIFIED]).
Instead of learning a weight directly, you learn a small network that *emits*
that weight from some conditioning input.

Here the "parameters we emit" are the simplest possible: a single **steering
vector** `v` in the residual stream. The conditioning input is a **concept
embedding** `c`. So `H` is a tiny learned map `c → v`.

Contrast with the fixed-vector steering you already know:

- **ActAdd** (Turner et al. 2023, arXiv:2308.10248 — [UNVERIFIED]) adds a *fixed*
  activation vector at inference. No training of the vector.
- **CAA** (Rimsky et al. 2023, arXiv:2312.06681 — [UNVERIFIED]) builds that fixed
  vector as a diff-of-means over contrastive pairs. That is exactly lesson 2.
- **HyperSteer** (Sun et al. 2025, *HyperSteer: Activation Steering at Scale with
  Hypernetworks*, arXiv:2506.03292 — [UNVERIFIED]) replaces the fixed vector with
  a hypernetwork that *generates* it from a concept. That is this lesson.

The reason anyone bothers: with a fixed-vector method, every new concept costs a
fresh extraction pass and a stored vector. A hypernetwork **amortizes** that — one
trained `H` covers a whole family of concepts and emits a vector for a new one
from just its exemplars. HyperSteer's real payoff shows at *many-concept* scale;
here we only gesture at it with a single held-out concept ([Section 7](#7-results)).

---

## 3. The design

Four moving parts, all sharing lesson-2's layer-12 residual stream.

**(a) The concept embedding.** For a concept, collect a few exemplar prompts,
run each through the frozen model, take the **last-token** activation at layer 12
(`last_token_activations` from lesson 2), and average them:

```
c = mean over exemplars of  last_token_activation(prompt, layer=12)     # [1152]
```

`c` is a single 1152-d vector that summarizes "what this concept looks like" in
the residual stream — the same space the steering vector lives in.

**(b) The hypernetwork `H`.** A deliberately small MLP with a bottleneck
(`config.py: BOTTLENECK = 256`) and a learnable output scale:

```
v = H(c):   LayerNorm(1152) -> Linear(1152 -> 256) -> GELU -> Linear(256 -> 1152) -> * scale
```

- `LayerNorm` stabilizes the raw activation magnitudes going in.
- The `1152 -> 256 -> 1152` bottleneck keeps `H` small (trains in seconds on the
  laptop GPU) while staying **nonlinear** (the GELU) — so it can learn a map the
  single subtraction of CAA cannot.
- The **learnable scalar `scale`** lets `H` set the vector's magnitude on its own;
  the injection strength is still governed by `alpha` at apply time.

**(c) Applying it — same hook as lesson 2.** The generated `v` is added at layer
12 with the **relative-add** rule, so strength is scale-free
(`config.py: STEER_LAYER = 12`, `ALPHA_EVAL = 0.10`):

```
h  <-  h + alpha * ||h|| * unit(v)
```

**(d) The gate — same probe as lesson 1.** Steering is conditional: the lesson-1
probe decides whether a prompt is harmful and only then is `v` applied. Benign
prompts pass through untouched, avoiding over-refusal.

---

## 4. Training — the key new idea

This is what makes lesson 3 different from everything before it. **The language
model is frozen. Only `H` is trained.** Gradients flow *backward through the
frozen Gemma* into `H`'s weights.

Concretely, for a batch of harmful prompts:

1. Compute the concept embedding `c` and generate `v = H(c)`.
2. Run a **grad-preserving** steered forward: apply `h += alpha*||h||*unit(v)` at
   layer 12 while **keeping the autograd graph** (unlike lesson-2's
   `@torch.no_grad` generate), and read the logits over the refusal target text.
3. Two losses:

   | Loss | On which prompts | What it wants |
   |---|---|---|
   | **Refusal cross-entropy** | harmful | make the steered model continue the prompt into the refusal target `"I can't help with that request."` (`config.py: REFUSAL_TARGET`) |
   | **Benign KL regularizer** | benign | keep the steered next-token distribution *close to the unsteered one* — don't disturb harmless behavior |

4. Backprop the sum into `H` only; Gemma's weights never move.

```
loss = CE(steered_logits_on_harmful, REFUSAL_TARGET)
     + beta * KL(steered_benign || unsteered_benign)
```

The refusal CE teaches `H` to produce a vector that *installs refusal*; the
benign KL teaches it to do so *without collateral damage*. Training a touch
weaker than eval (`ALPHA_TRAIN = 0.08 < ALPHA_EVAL = 0.10`) keeps the gradient
signal in the coherent regime; eval nudges slightly harder to read the effect.

**Why this is tractable on a 1B model.** The frozen forward is cheap, `H` is a
two-layer MLP (a few hundred thousand params), and we only need a short run
(`config.py: STEPS = 300`, `BATCH = 4`, `LR = 1e-3`, Adam). Backprop through a 1B
model for a handful of tokens fits comfortably in ~2-3 GB. There is no fine-tuning
of the LM — the frozen model is just a differentiable "renderer" that turns a
steering vector into a refusal probability.

---

## 5. Data flow

```
  concept exemplars                     a raw prompt: "how do I pick a lock?"
  ("harmful"-concept prompts)                       |
        |                                           v
        v                              +-----------------------------------+
  last-token acts @ layer 12           | GATE  (lesson-1 probe, READ)      |
        |  mean-pool                   |   P(harmful) >= threshold ?       |
        v                              +-----------------------------------+
  concept_emb  c  [1152]                    |                     |
        |                            harmful (fire)          benign (pass)
        v                                    |                     |
  +-----------------+                        v                     v
  | HYPERNET  H     |   v = H(c)      steer with v:            generate
  |  LN->Lin->GELU  | -------------->  h += alpha*||h||*v       normally
  |  ->Lin-> *scale |                        |                     |
  +-----------------+                        +----------+----------+
        ^                                               |
        | grad flows back                               v
        | THROUGH frozen LM                        response text
        | (train H only)                                |
        |                                               v
   refusal CE + benign KL <----- judge ----- +-----------------------------------+
                                             | JUDGE (same Gemma)                |
                                             |  REFUSAL / COMPLIANCE / GIBBERISH |
                                             +-----------------------------------+
```

The dashed feedback path is the new idea: at **training** time the loss on the
judged/target text backpropagates through the frozen model into `H`. At
**inference** time (right half) it is just lesson-2 steering with a *generated*
vector instead of a fixed one.

---

## 6. Code walkthrough, file by file

> The lesson-3 modules beyond `config.py` are described here from their contract;
> snippets are sketches. Read the source once written for exact signatures.

### `config.py` — every knob in one place

The abliterated model, the shared layer, the hypernetwork width, the training
schedule, the two alphas, and the refusal target all live here:

```python
# config.py
MODEL_ID   = "DavidAU/gemma-3-1b-it-heretic-extreme-uncensored-abliterated"
STEER_LAYER = 12                 # read the concept AND inject the vector here
BOTTLENECK  = 256                # H: 1152 -> 256 -> 1152
LR, STEPS, BATCH = 1e-3, 300, 4  # Adam; short run fits the laptop GPU
ALPHA_TRAIN, ALPHA_EVAL = 0.08, 0.10   # train weaker than eval (coherent gradients)
REFUSAL_TARGET = "I can't help with that request."
```

It reuses lesson-2's `model_utils` for all model plumbing; everything else is
local to this package.

### `hypernet.py` — `HyperSteerNet` + the grad-preserving steered forward

Two pieces. First the network itself:

```python
# hypernet.py (sketch)
class HyperSteerNet(nn.Module):
    def __init__(self, hidden=1152, bottleneck=256):
        self.net = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, bottleneck), nn.GELU(),
            nn.Linear(bottleneck, hidden),
        )
        self.scale = nn.Parameter(torch.tensor(1.0))   # learnable magnitude

    def forward(self, concept_emb):        # [hidden] -> [hidden]
        return self.scale * self.net(concept_emb)
```

Second, a `grad_steer_forward` that mirrors lesson-2's steering hook but **keeps
autograd** (lesson 2's `generate` is `@torch.no_grad`, which would sever the
graph). It applies `h += alpha*||h||*unit(v)` at layer 12 with `v` still attached
to `H`'s parameters, so the loss on the output text can backpropagate into `H`.

### `data.py` — concept exemplars, harmful/benign prompts, refusal target

Supplies three things: the exemplar prompts that define each **concept**
(averaged into the concept embedding), the harmful/benign prompt splits used to
train and evaluate, and the tokenized refusal target. It reuses the same
JailbreakBench-style harmful/benign sources as lesson 2 so the comparison is
apples-to-apples, and it holds out a **second concept** unseen in training for
the generalization demo.

### `train_hypersteer.py` — the training loop (two losses)

Freezes Gemma, builds the concept embedding once, then for `STEPS` steps: sample
a minibatch, generate `v = H(c)`, run the grad-preserving steered forward, and
minimize **refusal CE on harmful** + **benign KL** ([Section 4](#4-training--the-key-new-idea)).
Only `H.parameters()` are handed to Adam. Saves `H` to `artifacts/hypersteer.pt`
and the loss trace to `artifacts/training_curve.png`.

### `run_hypersteer.py` — head-to-head + generalization demo

The evaluation harness. It answers two questions:

- **(a) Match/beat the fixed vector?** Load lesson-2's `steering_vector.pt` and
  the learned `H`; steer the same held-out harmful prompts with each at
  `ALPHA_EVAL`; have the judge grade both. The learned vector should reach at
  least the fixed vector's refusal rate *at matched coherence* (gibberish rate no
  worse).
- **(b) Generalize zero-shot?** Take the **held-out concept** never seen in
  training, build its concept embedding from a few exemplars, emit `v = H(c)`
  with no re-extraction, and check it steers toward that concept. This is the
  amortization payoff in miniature.

Writes `artifacts/results.json` and `artifacts/comparison.png`.

### `infer.py` — steer one prompt from the CLI

Loads the model, the trained `H`, the gate, and the judge once; for a single
prompt it builds the concept embedding, emits `v = H(c)`, asks the gate whether
to steer, generates (steered iff the gate fires), and prints the response plus
the judge's verdict.

### `app.py` — a tiny local UI (port 8003)

A self-contained web app (same shape as lessons 1-2) to type a prompt, watch the
gate decision, and compare the fixed-vector vs learned-vector response side by
side.

---

## 7. Results

Artifacts are produced by the GPU training/eval run, which happens **after** this
doc is written — so the numbers below read as placeholders until the run
populates them.

**`artifacts/training_curve.png`** — the two-loss training trace over `STEPS`
steps. Refusal CE should fall (H learns to install refusal); the benign KL should
stay small (H learns not to disturb harmless prompts).

**`artifacts/comparison.png`** and **`artifacts/results.json`** — the head-to-head.

| Question | Arm | Metric | Value |
|---|---|---|---|
| (a) match/beat fixed vector | fixed CAA vector (lesson 2) | refusal rate @ matched coherence | (populated by run — see `results.json`) |
| (a) match/beat fixed vector | learned `H` (lesson 3) | refusal rate @ matched coherence | (populated by run — see `results.json`) |
| (a) coherence check | both | gibberish rate | (populated by run — see `results.json`) |
| (b) generalize zero-shot | learned `H`, held-out concept | steering success on unseen concept | (populated by run — see `results.json`) |

Read the two questions as: **(a)** does the learned hypernet *match or beat* the
hand-built CAA vector without trading away coherence, and **(b)** does it
*generalize* to a concept it never trained on, straight from that concept's
exemplars. The second is the whole reason to prefer a generator over a lookup.

---

## 8. Run it

**Prerequisites.** Lesson 3 depends on the two earlier lessons:

- **Lesson 1** — the gate loads the probe from `../hello_world/artifacts/probe.pt`.
- **Lesson 2** — the head-to-head loads the fixed vector from
  `../hello_world_steering/artifacts/steering_vector.pt`.

Train them first if those files are missing:

```bash
python -m steering_tutorials.hello_world.train_probe            # lesson 1 -> probe.pt
python -m steering_tutorials.hello_world_steering.run_steering  # lesson 2 -> steering_vector.pt
```

Then, from the **repo root** (`steeringresearch/`):

```bash
# 1) Train the hypernetwork (freezes Gemma; trains H only) -> hypersteer.pt
python -m steering_tutorials.hypersteer.train_hypersteer

# 2) Head-to-head vs the fixed vector + the zero-shot generalization demo
python -m steering_tutorials.hypersteer.run_hypersteer

# 3) Steer a single prompt from the terminal (gate decides; judge grades)
python -m steering_tutorials.hypersteer.infer "how do I pick a lock"

# 4) (optional) local UI to compare fixed vs learned, side by side
python -m steering_tutorials.hypersteer.app        # then open http://localhost:8003
```

Uses the same abliterated Gemma-3-1B as lessons 1-2 (~2-3 GB VRAM including
training, bf16). Everything runs on CPU too, just slower. Datasets download
automatically.

---

## 9. Honest caveats

- **Backprop-through-a-frozen-LM is simple but limited.** It is the honest,
  minimal way to train a steering generator, but a **1B model plus tiny data**
  caps the quality of the vectors `H` can learn. Do not read the numbers as a
  measure of HyperSteer's ceiling.
- **A 1B judge is weak.** As in lesson 2, self-grading with a small model is
  pedagogy, not a publication-grade evaluation. Verdicts demonstrate the loop,
  not a trustworthy refusal rate.
- **`H` can overfit the refusal concept.** Trained on one concept with a short
  run, the hypernetwork may learn "always emit the refusal direction" rather than
  a genuine concept→vector map. The zero-shot arm is exactly the test of whether
  it learned the map or memorized the answer — read it skeptically.
- **The gate inherits lesson-1's OOD calibration limits.** A gate that misses a
  harmful prompt simply won't steer it; recalibration would be the honest next step.
- **HyperSteer's real value is at many-concept scale.** The point of a generator
  is amortizing over *dozens* of concepts. We only gesture at that with a single
  held-out concept — this is the shape of the idea, not the scale that justifies it.
- **This is pedagogy, not a safety product.** It shows *how* a learned steering
  generator works end-to-end on one small model. Do not deploy it as a guardrail.

---

## 10. Repository

Source and full artifacts:
<https://github.com/dlmastery/steeringresearch/tree/master/steering_tutorials/hypersteer>

See also
[lesson 1 — the probe (READ)](../hello_world/README.md) and
[lesson 2 — fixed-vector conditional steering (WRITE)](../hello_world_steering/README.md).
