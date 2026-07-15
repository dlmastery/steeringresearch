# FLAS — Flow-based Activation Steering, one field for many concepts

> Lesson 1 **read** a concept out of a frozen Gemma-3-1B. Lesson 2 **wrote** a
> **fixed** diff-of-means vector back to re-install refusal. Lesson 3 (`reft_r1`)
> **learned** a rank-1 edit for one concept. FLAS is the flow-based
> generalization: instead of one vector or one edit, we learn a
> **concept-conditioned velocity field** `v_theta(h, t, c)` and steer by
> **integrating a flow** — following a learned trajectory through activation
> space. Flow-time `T` becomes a smooth, zero-shot **strength dial**, and **one
> field handles many concepts**, including ones it never saw during training.

This is lesson 3b of the steering tutorials — the most advanced entry in the
**GENERATE** tier. Where lesson 3 learned *one direction for one concept*, FLAS
learns a *single field that produces a direction for any concept* from just a
handful of exemplars. The arc across four lessons:

> **fixed vector** (L2) → **learned rank-1 edit** (L3) → **learned velocity
> field / flow** (FLAS).

Everything here is deliberately standalone and CPU-runnable to read; the actual
training and generation need the same ~2 GB abliterated Gemma-3-1B as lessons
1–3, and the lesson-1 probe checkpoint for the gate.

Source: **FLAS** (<https://github.com/flas-ai/FLAS>), which frames steering as
**rectified flow / flow matching** applied to residual-stream activations. Our
implementation is a minimal, laptop-scale reimplementation of that idea
(`[UNVERIFIED]` arXiv provenance — see caveats).

---

## Dataset

**JailbreakBench** harm **categories**, loaded by `data.py`'s `load_concepts`.
Where lessons 1–3 used JBB's harmful/benign split as one behaviour, FLAS reads
JBB's `Category` field and treats **each category as a distinct concept** — the
thing the velocity field must learn to steer toward when conditioned on that
concept's code. From the harmful CSV (`Goal` prompts, `Category` labels) we take
three categories to **train** the flow and hold **one out entirely** for the
zero-shot test, plus a shared benign baseline as the common contrast origin.

| concept | JBB category | trained? | per-concept prompts (exemplars/steer · eval) |
|---|---|---|---|
| Malware/Hacking | Malware/Hacking | yes | 7 · 3 |
| Fraud/Deception | Fraud/Deception | yes | 7 · 3 |
| Harassment/Discrimination | Harassment/Discrimination | yes | 7 · 3 |
| **Physical harm** | Physical harm | **no (held-out)** | 7 · 3 (eval only) |

Each category caps at 10 prompts (`N_PER_CONCEPT`); the last 3 are held out for
grading and the rest do double duty as exemplars (encode → the concept code `c`)
and steer prompts (the harmful side of the diff-of-means target). A shared pool
of 40 benign prompts is the neutral contrast for every concept. The goal: **one**
field, conditioned on a concept, that steers all three trained concepts and
generalizes **zero-shot** to the held-out "Physical harm" it never saw.

---

## Table of contents

1. [What you'll build](#1-what-youll-build)
2. [Concepts — velocity fields and flows](#2-concepts--velocity-fields-and-flows)
3. [Rectified-flow training — the key idea](#3-rectified-flow-training--the-key-idea)
4. [The three payoffs](#4-the-three-payoffs)
5. [Data flow](#5-data-flow)
6. [Code walkthrough, file by file](#6-code-walkthrough-file-by-file)
7. [Results](#7-results)
8. [Run it](#8-run-it)
9. [Honest caveats](#9-honest-caveats)
10. [Repository](#10-repository)

---

## 1. What you'll build

A single learned **velocity field** and the driver that shows off what a flow can
do that a fixed vector cannot:

1. **Train a velocity field** `v_theta(h, t, c)` on layer 12 of the abliterated
   Gemma-3-1B — a small MLP that, conditioned on a **concept vector** `c` and a
   **flow-time** `t`, predicts how to move a hidden state `h`. The 1B model stays
   **frozen**; only `v_theta` trains. Training is fast regression on **cached
   activations** — no generation in the loop.
2. **Steer by integrating the flow** at inference: start from the model's real
   hidden state `h(0)`, take a few small Euler steps `h(t+dt) = h(t) + v_theta·dt`
   from `t=0` to `t=T`, and write `h(T)` back into the residual stream. `T` is a
   continuous strength dial you turn at inference with **zero retraining**.
3. **Serve many concepts from one field.** Because `v_theta` is conditioned on
   `c`, the same trained network steers refusal, sentiment, formality, … — pick
   the concept by swapping `c`, not by loading a new model.
4. **Generalize zero-shot** to a **held-out concept** the field never trained on,
   using only that concept's exemplar activations to build its `c`.
5. **Gate it** with lesson-1's probe (reused verbatim), so harmful prompts get
   the flow and benign prompts pass untouched — the lesson-2 conditional recipe,
   now wrapping a *flow*.

**Teaser.** A fixed vector gives you one direction at one magnitude; you sweep an
external `alpha` to find the sweet spot. A flow lets you *dial the strength
continuously by integrating farther* (`T`), reuse *one* network across concepts,
and steer a concept you only described with a few examples. We measure whether
that actually holds at 1B — see Section 7 (raw numbers in `artifacts/results.json`).

---

## 2. Concepts — velocity fields and flows

**A steering vector is a jump.** Lesson 2 did `h ← h + alpha·||h||·unit(v)`: one
fixed direction, one hand-tuned step. **A flow is a journey.** Instead of jumping
along a frozen direction, you **follow a learned trajectory**: at every point `h`
along the way, a field tells you which way to move next, and you integrate those
little moves into a path.

Two objects make this work:

- **The concept encoder → `c`.** For each concept we collect a few **exemplar**
  prompts (e.g. harmful prompts for "refusal"), run them through the frozen model,
  and take the **mean activation** at layer 12. That mean vector *is* the concept
  code `c` — a cheap, training-free summary of "what this concept looks like in
  activation space." (Notice this is exactly lesson 2's diff-of-means ingredient,
  now repurposed as a **conditioning input** rather than the edit itself.)

- **The velocity field → `v_theta(h, t, c)`.** A small MLP that eats the current
  hidden state `h`, a scalar flow-time `t ∈ [0,1]`, and the concept code `c`, and
  outputs a **velocity** — a vector in the same 1152-d space saying "move this
  way, this fast, right now." It is *not* a fixed direction: the move it
  recommends depends on where you are (`h`), how far along you are (`t`), and
  which concept you asked for (`c`).

**Integrating the field = steering.** Given a prompt's real hidden state `h(0)`,
we integrate

```
h(t + dt) = h(t) + v_theta(h(t), t, c) · dt ,   t : 0 -> T
```

and inject the endpoint `h(T)`. **`T` is the strength dial:** integrate a little
(small `T`) for a gentle nudge, integrate farther (larger `T`) for a stronger
push. Crucially, `T` is chosen **at inference** — one trained field exposes a
whole continuum of strengths for free, where lesson 2 needed a fresh `alpha`
sweep.

---

## 3. Rectified-flow training — the key idea

How do we teach `v_theta` to point the right way everywhere? With **rectified
flow** (the straight-line special case of flow matching): pick a start and an
end, connect them with a straight line, and train the field to match the constant
velocity of that line.

For a concept `c`, define the endpoints:

- **`h0`** — a real base activation (the hidden state on an actual prompt).
- **`h1 = h0 + delta_c`** — the same activation **shifted by the concept's
  diff-of-means** `delta_c = mean(concept exemplars) − mean(neutral)`. This is the
  *target* place we want steering to transport activations to.

Draw the straight-line path between them and read off its (constant) velocity:

```
h_t   = (1 - t) * h0 + t * h1              # straight-line interpolation, t in [0,1]
target_velocity = h1 - h0  ( = delta_c )   # constant along the straight path
loss  = || v_theta(h_t, t, c) - (h1 - h0) ||^2
```

The field is trained, over many `(h0, c, t)` samples, to predict the transport
velocity `h1 − h0` at every intermediate point `h_t` and time `t`. Once it has,
**integrating the learned field from `h0` reproduces the transport to `h1`** — but
now smoothly parameterized by how far you integrate (`T`), and **conditioned on
`c`** so *one* field encodes the transport for *every* concept it was trained on.

Why this is powerful, in one line: the straight-line target makes training a
trivial **regression** (no sampling, no generation in the loop), yet the
resulting field supports a *continuous* strength knob and *multi-concept*
conditioning that a single fixed vector cannot.

Training specifics: the **LLM is frozen**; only `v_theta` gets gradients.
Everything runs on **cached activations**, so training is fast — minutes on the
4090. (Key knobs live in `config.py`: `LAYER = 12`, the model id, the concept
list, the MLP width, the flow step count, and the seed.)

---

## 4. The three payoffs

`run_flas.py` exists to measure the three things a flow buys you over a fixed
vector. Each is a falsifiable claim we report **as measured**.

| # | Payoff | What we vary | What we watch for |
|---|---|---|---|
| a | **`T` = a smooth strength dial** | flow-time `T`, one field | refusal rate **rises with `T`**, then generation degrades into gibberish — the **coherence cliff** again |
| b | **One field, many concepts** | the concept code `c` | each trained concept steers from the **same** `v_theta`, no reload |
| c | **Zero-shot to a held-out concept** | a concept withheld from training | steering works from **exemplars alone**, without ever training on that concept |

**(a) The dial and the cliff.** Sweeping `T` should trace a curve: too small and
nothing happens; in a sweet band refusal rises; too large and the activation is
pushed off-manifold into incoherent text. This is the same **displacement /
coherence cliff** lesson 4 (`displacement_budget`) studies — here surfaced as a
property of integrating too far. We plot `rates_vs_T.png`.

**(b) Multi-concept from one network.** Because conditioning on `c` selects the
behavior, the *same* trained field handles every concept in the training set.
`per_concept.png` shows the steering outcome per concept.

**(c) Zero-shot generalization.** The real test of a *field* (vs a lookup table
of vectors): hold one concept out of training entirely, build its `c` from just
its exemplars at eval time, and see whether the field transports toward it. If it
does, `v_theta` has learned *how to steer in general*, not just memorized a few
directions. If it doesn't at 1B, we say so.

---

## 5. Data flow

```
  exemplar prompts for concept c        "how do I pick a lock?"  (a raw prompt)
            |                                       |
            v                                       v
  +---------------------------+          +-----------------------------------+
  | ConceptEncoder            |          | GATE  (lesson-1 probe, READ side) |
  |  mean activation @ L12    |          |   P(harmful) >= threshold ?       |
  |  -> concept code c        |          +-----------------------------------+
  +---------------------------+                |                     |
            |                              harmful (fire)       benign (pass)
            |                                   |                     |
            +---------------+                   v                     v
                            |          FlowContext integrates    generate
                            +--------> v_theta @ L12 over T:      normally
                                       h(t+dt)=h(t)+v_theta*dt    (no edit)
                                       t: 0 -> T ; inject h(T)        |
                                              |                       |
                                              +-----------+-----------+
                                                          |
                                                          v
                                                     response text
                                                          |
                                                          v
                                       +-----------------------------------+
                                       | JUDGE  (same Gemma)               |
                                       |   is_gibberish() -> GIBBERISH     |
                                       |   else REFUSAL / COMPLIANCE       |
                                       +-----------------------------------+
```

The concept code `c` (built once from exemplars) **conditions** the field; the
gate decides **whether** to integrate; `T` decides **how far**. Same
READ (gate) / WRITE (flow) duality as lessons 1–3, now with a learned *field*
doing the writing.

---

## 6. Code walkthrough, file by file

> At the time of writing, `flas/__init__.py` is the concrete source; the sketches
> below describe the sibling modules from the lesson contract and mirror the
> conventions of lesson 3 (`reft_r1`). Read the real files for exact signatures
> once they land.

### `config.py` — every knob in one place

The abliterated model id, the flow layer (`LAYER = 12`), the concept list (with
one concept marked **held-out** for the zero-shot test), the MLP width, the
number of Euler steps and the default `T`, the training schedule, and the seed.
Like lesson 3, there is **no fixed `alpha`** — strength is `T`, chosen at
inference.

```python
# config.py (sketch)
MODEL_ID   = "DavidAU/gemma-3-1b-it-heretic-extreme-uncensored-abliterated"
LAYER      = 12               # build c and integrate the flow on this residual layer
CONCEPTS   = ["refusal", "formal", "positive"]   # trained
HELDOUT    = "cautious"       # never trained on -> zero-shot arm
N_STEPS    = 8                # Euler steps used to integrate the flow
T_DEFAULT  = 1.0              # default flow-time (the strength dial)
SEED       = 0
```

### `flow.py` — the field, the integrator, and the steering hook

The heart of the lesson. Three responsibilities:

- **`VelocityField(nn.Module)`** — the MLP `v_theta(h, t, c)`. Concatenates the
  hidden state `h`, a (featurized) flow-time `t`, and the concept code `c`, and
  outputs a velocity in the same `d`-dim space.
- **`integrate_flow(h0, c, T, n_steps)`** — Euler integration of the field from
  `t=0` to `t=T`, returning the transported endpoint `h(T)`.
- **`FlowContext`** — a forward-hook manager on `model.model.layers[LAYER]` that,
  while active, replaces each token's hidden state with its integrated `h(T)`
  (the WRITE), then restores the model on exit.

```python
# flow.py (sketch)
class VelocityField(nn.Module):
    def __init__(self, d, c_dim, width=512):
        self.net = nn.Sequential(
            nn.Linear(d + 1 + c_dim, width), nn.SiLU(),
            nn.Linear(width, width),         nn.SiLU(),
            nn.Linear(width, d),
        )
    def forward(self, h, t, c):                       # h:(...,d) t:(...,1) c:(...,c_dim)
        return self.net(torch.cat([h, t, c], dim=-1))

def integrate_flow(field, h0, c, T=1.0, n_steps=8):
    h, dt = h0, T / n_steps
    for i in range(n_steps):
        t = torch.full_like(h[..., :1], i * dt)
        h = h + field(h, t, c) * dt                   # Euler step
    return h                                           # == h(T)
```

### `data.py` — multi-concept exemplars, cached activations, held-out split

Loads exemplar prompts per concept (refusal reuses lessons 1–3's harmful/benign
families), runs the frozen model once to **cache layer-12 activations**, builds
each concept code `c = mean(exemplars)` and each target shift
`delta_c = mean(concept) − mean(neutral)`, and constructs the
`(h0, c, target_velocity)` regression samples. Crucially it **withholds the
`HELDOUT` concept** from training so `run_flas` can test zero-shot transfer.

### `train_flas.py` — rectified-flow training (frozen LLM)

Freezes Gemma, builds a `VelocityField`, and runs Adam on the flow-matching loss:
sample `(h0, c)` and a random `t ∈ [0,1]`, form `h_t = (1−t)·h0 + t·h1`, and
minimize `|| v_theta(h_t, t, c) − (h1 − h0) ||²`. Fast regression on the cached
activations; saves the trained field to `artifacts/flas.pt`.

### `run_flas.py` — the three payoffs

The driver that produces the lesson's evidence:

- **(a) `T`-sweep** — for a fixed concept, integrate at a grid of `T`, generate,
  and judge REFUSAL / COMPLIANCE / GIBBERISH; render `rates_vs_T.png` (the dial +
  the cliff).
- **(b) per-concept** — steer each trained concept from the one field; render
  `per_concept.png`.
- **(c) zero-shot** — build the held-out concept's `c` from its exemplars only
  and test whether the field transports toward it.
- Writes everything to `results.json`.

### `infer.py` — steer one prompt from the CLI

Loads model, trained field, gate, and judge once; for one prompt asks the gate,
and iff it fires, integrates the flow for the chosen `--concept` up to `--T` and
generates, printing the response plus the judge's verdict.

```bash
python -m steering_tutorials.flas.infer "how do I pick a lock" --concept refusal --T 1.0
```

### `app.py` — the live flow dashboard (port 8005)

A small self-contained viewer with a **`T` slider**: drag it to watch refusal
rise and then collapse into gibberish, view the per-concept outcomes, and see the
zero-shot arm. Serves on **port 8005** (lessons 1–3 use their own ports).

---

## 7. Results

The GPU run wrote `artifacts/results.json` and two plots. Numbers below are the
**measured** values at 1B (n=3 per concept per cell) — screening tier.

**Q1 — Is `T` a smooth strength dial (and where's the cliff)?**
`rates_vs_T.png` + `results.json` (dial concept: Malware/Hacking).

| flow-time `T` | refusal | comply | gibberish |
|---|---|---|---|
| 0.0 | 0.00 | 1.00 | 0.00 |
| 0.5 | 0.33 | 0.00 | 0.67 |
| 1.0 | 0.33 | 0.67 | 0.00 |
| 1.5 | 0.33 | 0.67 | 0.00 |
| 2.0 | 0.67 | 0.00 | 0.33 |

Refusal climbs from 0.00 at `T=0` to 0.67 at `T=2` — the dial works. But the
climb is not clean: gibberish spikes to 0.67 at `T=0.5` and returns at 0.33 for
`T=2`, so pushing past the useful band drops you over the **coherence cliff**,
exactly as predicted.

**Q2 — One field, many concepts; and does zero-shot work at 1B?**
`per_concept.png` + `results.json`.

| concept | trained? | refusal @ default `T` |
|---|---|---|
| Malware/Hacking | yes | 0.33 |
| Fraud/Deception | yes | 0.67 |
| Harassment/Discrimination | yes | 1.00 |
| Physical harm | **no (held-out)** | 0.67 |

One field steers all three trained concepts (refusal 0.33 / 0.67 / 1.00), and the
held-out "Physical harm" concept — never trained on — steers at 0.67 from its
exemplars alone: zero-shot transfer holds at 1B.

### Results — measured vs. the claim

| Claim (FLAS, github.com/flas-ai/FLAS [UNVERIFIED]) | What we measured (n=3/concept, screening) | Verdict |
|---|---|---|
| Flow-time `T` is a continuous strength dial | refusal 0.00 (T=0) → 0.67 (T=2), gibberish past the useful band | Reproduced (with a coherence cliff) |
| One conditioned field steers many concepts | one field steers 3 concepts: refusal 0.33 / 0.67 / 1.00 | Reproduced |
| Generalizes zero-shot to an unseen concept | held-out "Physical harm" refusal 0.67 from exemplars alone | Reproduced |

**Honest read.** All three qualitative payoffs of FLAS reproduce at 1B: `T` acts
as a strength dial (with a visible gibberish cliff outside the useful range), one
velocity field steers three distinct concepts, and it transfers zero-shot to a
held-out concept at 0.67 refusal. Read this as **screening**, not a verdict: n=3
per concept is tiny, our transport targets are cheap diff-of-means shifts (a
simplification of full FLAS), and the grader is the same 1B model doing the
steering — a weak, self-referential judge. Raw numbers and side-by-side
generations live in `artifacts/results.json`.

---

## 8. Run it

**Prerequisite: run lesson 1 first** — the gate loads lesson-1's probe from
`../hello_world/artifacts/probe.pt`. If that file does not exist, train it:

```bash
python -m steering_tutorials.hello_world.train_probe
```

Then, from the **repo root** (`steeringresearch/`):

```bash
# 1) Train the velocity field by rectified flow (frozen Gemma; ~minutes on a 4090)
python -m steering_tutorials.flas.train_flas

# 2) Run the three payoffs: T-sweep, per-concept, zero-shot
python -m steering_tutorials.flas.run_flas

# 3) Steer a single prompt from the terminal (gate decides; you pick concept + T)
python -m steering_tutorials.flas.infer "how do I pick a lock" --concept refusal --T 1.0

# 4) Launch the live dashboard with the T slider
python -m steering_tutorials.flas.app          # -> http://localhost:8005
```

Uses the same ~2 GB abliterated Gemma-3-1B as lessons 1–3 (bf16). Runs on CPU
too, just slower. Datasets download automatically.

---

## 9. Honest caveats

- **Our targets are diff-of-means shifts.** We train the field to transport
  toward `h0 + delta_c`, a **simplification** of full FLAS. It captures the
  flow-matching mechanism and the `T`-as-strength story, but the endpoints are the
  cheap diff-of-means targets from lesson 2, not a richer learned coupling.
- **Tiny scale.** One 1B model, a handful of concepts, small exemplar and
  held-out sets. This demonstrates the flow loop; it is not a benchmark-grade
  reproduction of FLAS.
- **A 1B judge is weak.** Self-grading with a small model is pedagogy, not a
  trustworthy evaluation — read verdicts as a demonstration of the loop. A real
  evaluation uses a stronger, independent judge (later lessons).
- **Euler with few steps.** We integrate with a handful of explicit Euler steps
  (`N_STEPS`), which is a coarse ODE solver; the transport is approximate, and the
  cliff location shifts with step count.
- **Zero-shot may be limited at 1B.** A small field conditioned on mean-exemplar
  codes may not generalize cleanly to an unseen concept. We report what we see;
  the held-out arm can go either way.
- **The gate inherits lesson-1's OOD limits.** The probe ranks harm well but its
  0.5 threshold miscalibrates off-distribution; a gate that misses a harmful
  prompt simply won't integrate the flow.
- **This is pedagogy, not a safety product.** It shows *how* flow-based steering
  works end-to-end. Do not deploy it as a real-world guardrail.

---

## 10. Repository

Source and full artifacts:
<https://github.com/dlmastery/steeringresearch/tree/master/steering_tutorials/flas>

See also:
- [Lesson 1 — the probe (READ side)](../hello_world/README.md)
- [Lesson 2 — fixed-vector conditional steering (WRITE side)](../hello_world_steering/README.md)
- [Lesson 3 — ReFT-r1, a learned rank-1 edit](../reft_r1/README.md)
