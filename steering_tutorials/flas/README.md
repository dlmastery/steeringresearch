# FLAS — Flow-based Activation Steering, one field for many concepts

> **Reference:** [Flow Matching for Generative Modeling (arXiv:2210.02747)](https://arxiv.org/abs/2210.02747) (the flow foundation); FLAS is an own construction inspired by it (github.com/flas-ai/FLAS). Builds on lesson 3.

> Lesson 1 **read** a concept out of a frozen Gemma-3-1B. Lesson 2 **wrote** a
> **fixed** diff-of-means vector back to re-install refusal. Lesson 3 (`reft_r1`)
> **learned** a rank-1 edit for one concept. FLAS is the flow-based
> generalization: instead of one vector or one edit, we learn a
> **concept-conditioned velocity field** `v_theta(h, t, c)` and steer by
> **integrating a flow** — following a learned trajectory through activation
> space. Flow-time `T` becomes a smooth, zero-shot **strength dial**, and **one
> field handles many concepts**, including ones it never saw during training.

> ## ⚠ v2 — THE FLOW-TIME DIAL WAS MIS-SCALED. The v1 numbers below are **SUPERSEDED**.
>
> **What was wrong.** v1 defined the transport target with the **raw** diff-of-means,
> `h1 = h0 + delta_c`, and integrated an **absolute** Euler step `x <- x + dt*v`.
> Measured on the shipped `artifacts/flow.pt` at layer 12 of Gemma-3-1B:
> `||delta_c|| = 561` (sexual) / `474` (violence), against a mean activation norm
> `||h|| ≈ 5.0e3`. So the v1 flow-time grid `T ∈ {0, 0.5, 1.0, 1.5, 2.0}` actually
> injected **relative** displacements of `{0, 5.6%, 11.2%, 16.9%, 22.5%}` of `||h||`.
> Lesson 2 sweeps `alpha` only up to **0.15**, and this course has repeatedly located
> the coherence cliff at roughly 5–10% of `||h||` — so **every steered point in the v1
> grid sat at or past the cliff**, and the informative sub-cliff regime (`T < 0.05`)
> was never sampled. That is why the v1 sweep read as refusal 0.25 → 0.06 while
> gibberish climbed 0.28 → 0.75: **the lesson measured the coherence cliff, not the
> concept dial.** A mis-scaled axis, not a finding about flows.
>
> **What the fix is.** Three changes (all in `config.py`, all revertable by env var):
>
> 1. **Norm-relative transport** (`FLAS_NORM_RELATIVE=1`, default). The trainer
>    regresses the field onto the **unit** direction `delta_hat = delta_c/||delta_c||`
>    over the interpolant `h_t = h0 + t·TRAIN_T_MAX·||h0||·delta_hat`, and
>    `integrate_flow` takes norm-relative steps `x <- x + dt·||x||·v`. Since `||v|| ≈ 1`
>    by construction, integrating to `T` displaces by `≈ T·||h||` — **flow-time `T` is
>    now literally the fractional displacement, the same dial lesson 2 calls `alpha`**,
>    and the two lessons' curves are directly comparable. `run_flas` now reports the
>    **measured** `||Δh||/||h||` per `T` so the claim is auditable, not asserted.
> 2. **A grid over the informative regime**: `T ∈ {0, 0.02, 0.05, 0.10, 0.15}`
>    (sub-cliff through cliff), mirroring lesson 2's `ALPHAS`. `T_DEFAULT` 1.0 → 0.10.
> 3. **A special-token guard** (`FLAS_SKIP_SPECIAL=1`, default). v1 transported
>    BOS / `<start_of_turn>` positions, which lesson 2's `SteeringContext` explicitly
>    refuses to touch — they carry outsized residual norms and editing them derails
>    formatting for no behavioral gain.
>
> **Consequences.** The shipped `artifacts/flow.pt` is a **v1 field and must be
> retrained**; `run_flas` refuses to integrate a v1 field under the v2 convention
> (the checkpoint records `norm_relative`). `artifacts/results.json` carries a
> `SUPERSEDED` key. §7's numbers stand only as a record of the bug.
>
> **This fix does not manufacture a win.** It makes the experiment *measure the thing
> the lesson claims*. Whether refusal actually climbs with `T` at matched coherence on
> a 1B abliterated model is now an open question the re-run answers; the honest-negative
> framing stays until the new numbers land, and `run_flas`'s verdict check was tightened
> so a gibberish-dialling sweep can no longer be graded as a working dial.

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

## The key idea in code

Don't add a vector — *transport* the activation along a learned velocity field by
integrating a flow. Flow-time `T` is the strength dial; `T = 0` is the identity:

```python
# integrate_flow (flow.py) — explicit Euler integration of dx/dt = v(x, t, c):
x  = h
dt = T / n_steps                        # T = fraction of ||h|| travelled = the dial
for k in range(n_steps):
    t_norm = (k * dt) / T               # canonical clock in [0, 1), T-agnostic
    v = vfield(x, t_norm, c)            # VelocityField at state x, time t, concept c
    x = x + dt * x.norm(dim=-1, keepdim=True) * v   # NORM-RELATIVE step (v2)
h = x                                   # residual transported toward concept c
```

The `||x||` factor is the v2 fix and it is what makes `T` mean something: the field
is trained to output a **unit** direction, so integrating to `T` displaces the
residual by `≈ T·||h||`. `T = 0.10` is a 10% displacement whatever the layer, model,
or concept — the same units as lesson 2's `alpha`. Drop the factor (v1) and the
distance travelled is whatever `‖delta_c‖` happens to be, which is how this lesson
ended up sweeping 0–22% of `‖h‖` and measuring nothing but the coherence cliff.

Because the field is conditioned on a concept embedding `c` (the mean activation
of a few exemplars), one trained `vfield` steers toward *any* concept you can
encode — including held-out ones. Full file-by-file walkthrough below.

---

## Dataset

**Toxic-chat harm categories**, loaded by `data.py`'s `load_concepts` (which wraps
the shared `common.data.load_concepts`). FLAS reads toxic-chat's
`openai_moderation` harm categories and treats **each category as a distinct
concept** — the thing the velocity field must learn to steer toward when
conditioned on that concept's code. We take the **well-populated** categories to
**train** the flow and hold **one out entirely** for the zero-shot test, plus a
shared benign baseline as the common contrast origin.

Each concept splits into **disjoint** exemplar / steer / eval roles (40/30/30), so a
transport is never graded on the prompts that defined it. v2 raises
`N_PER_CONCEPT` to **500**, which is simply "take the whole pool" — the loader caps
each concept at its own availability:

| concept | toxic-chat category | trained? | pool | exemplars | steer | **eval `n`** |
|---|---|---|---|---|---|---|
| sexual (dial concept) | sexual | yes | 388 | 155 | 116 | **117** |
| violence | violence | yes | 111 | 44 | 33 | **34** |
| **harassment** | harassment | **no (held-out)** | 143 | 57 | 43 | **43** |
| benign baseline | — | contrast origin + selectivity | — | — | — | **500** |

*These are the honest maxima, not a choice.* **Concept lessons are pool-limited**
(CLAUDE.md §17.2): **no toxic-chat harm category reaches 500**, so the per-concept
eval cells are 117 / 43 / 34 — 3.2× v1's `n=36` for the dial concept, and every cell
clears the ≥30-per-concept floor, but this is **not** a ≥500/class result and must
not be reported as one. The benign side is *not* pool-limited, so
`N_BENIGN_BASELINE = 500` (v1: 120) does meet the rubric. The goal: **one** field,
conditioned on a concept, that
steers the trained concepts and generalizes **zero-shot** to the held-out concept
it never saw. (Sparse categories — hate, self_harm — are dropped by the shared
loader's `MIN_CONCEPT_AVAILABLE` gate rather than given a noise field.)

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
- **`h1`** — the same activation **shifted along the concept's diff-of-means
  direction** `delta_c = mean(concept exemplars) − mean(neutral)`. This is the
  *target* place we want steering to transport activations to.

**How far** to shift is the whole ballgame, and getting it wrong is exactly what
broke v1 (see the banner at the top). v2 measures the shift as a **fraction of the
activation's own norm**, so the distance travelled is a controlled quantity rather
than whatever magnitude the diff-of-means happened to come out at:

```
delta_hat = delta_c / ||delta_c||                       # DIRECTION only
h1        = h0 + TRAIN_T_MAX * ||h0|| * delta_hat       # far end: a FRACTION of ||h0||
h_t       = (1 - t) * h0 + t * h1                       # straight-line interp, t in [0,1]
target_velocity = delta_hat                             # UNIT, constant along the path
loss      = || v_theta(h_t, t, c) - delta_hat ||^2
```

Two things fall out of regressing onto a **unit** target. First, the integrator's
norm-relative step `dt·||x||·v` accumulates to `≈ T·||h||`, so **`T` is the fractional
displacement** — the same number lesson 2 calls `alpha`, on the same scale, comparable
across layers and models. Second, the states the field **trains** on are exactly the
states the eval flow **visits** for `T ≤ TRAIN_T_MAX`; v1 trained over a segment of
length 561 and then integrated out to 1122, so half its eval grid was off the field's
own training distribution.

The v1 formulation (`h1 = h0 + delta_c`, absolute Euler steps) is still reachable with
`FLAS_NORM_RELATIVE=0`, for reproducing the bug.

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
MODEL_ID      = "DavidAU/gemma-3-1b-it-heretic-extreme-uncensored-abliterated"
LAYER         = 12         # build c and integrate the flow on this residual layer
N_STEPS       = 8          # Euler steps used to integrate the flow
NORM_RELATIVE = True       # v2: T is a FRACTION of ||h||  (FLAS_NORM_RELATIVE=0 -> v1)
SKIP_SPECIAL  = True       # never transport BOS / <start_of_turn> (lesson-2 parity)
TRAIN_T_MAX   = 0.15       # far end of the training interpolant, as a fraction of ||h0||
T_DEFAULT     = 0.10       # default flow-time == lesson 2's mid alpha
T_SWEEP       = [0.0, 0.02, 0.05, 0.10, 0.15]   # sub-cliff through cliff
SEED          = 0
```

Every one of these is env-overridable (`FLAS_T_SWEEP`, `FLAS_T_DEFAULT`,
`FLAS_TRAIN_T_MAX`, `FLAS_N_STEPS`, …), plus the eval-size caps that let a full pass
be shrunk into one foreground window: `FLAS_N_EVAL`, `FLAS_N_BENIGN_EVAL`,
`FLAS_MAX_NEW_TOKENS`. A capped run is recorded as `"tier": "SCREENING"` in
`results.json` so it cannot be quietly reported as a full pass. The concepts and the
held-out split come from `data.py` (`FLAS_N_PER_CONCEPT`, `FLAS_N_BENIGN`), not from a
hand-written list.

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

def integrate_flow(field, h0, c, T=0.10, n_steps=8, norm_relative=True):
    h, dt = h0, T / n_steps
    for i in range(n_steps):
        t = torch.full_like(h[..., :1], i * dt)
        step = field(h, t, c) * dt
        if norm_relative:                             # v2: T = fraction of ||h||
            step = step * h.norm(dim=-1, keepdim=True)
        h = h + step                                  # Euler step
    return h                                           # == h(T)
```

`FlowContext` also carries a **special-token guard** (lesson-2 parity): it zeroes the
*displacement* at BOS / `<start_of_turn>` / pad positions, so control tokens pass
through the layer exactly as they would unsteered.

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

> **STATUS: awaiting the v2 re-run.** Everything in this section is the **v1**
> (mis-scaled-dial) run of 2026-07-22 and is **SUPERSEDED** — see the banner at the
> top of this README. It is kept because the bug is the most instructive thing in
> the lesson, not because the rates mean what they appear to mean. `flow.pt` must be
> retrained and `run_flas` re-run before any number here is cited.

The v1 GPU run wrote `artifacts/results.json` (now carrying a `SUPERSEDED` key) and
two plots. Concepts are the well-populated toxic-chat categories (sexual, violence
trained; harassment held out); the sparse hate/self_harm are dropped by the loader.

**Q1 — Is `T` a smooth strength dial (and where's the cliff)?**
`rates_vs_T.png` + `results.json` (dial concept: sexual, v1 `n=36`).

| flow-time `T` (v1) | **actual `‖Δh‖/‖h‖`** | refusal | comply | gibberish |
|---|---|---|---|---|
| 0.0 | 0.000 | 0.25 | 0.47 | 0.28 |
| 0.5 | **0.056** | 0.25 | 0.42 | 0.33 |
| 1.0 | **0.112** | 0.14 | 0.31 | 0.56 |
| 1.5 | **0.169** | 0.06 | 0.36 | 0.58 |
| 2.0 | **0.225** | 0.06 | 0.19 | 0.75 |

The second column is the diagnosis, computed after the fact from the shipped
checkpoint (`‖delta_c‖ = 561`, `‖h‖ ≈ 5.0e3`). Lesson 2's top `alpha` is **0.15**.
So `T=1.0` — v1's *default* — already displaced **11%** of the residual norm, and
`T=2.0` displaced **22%**: the sweep walked off the coherence cliff and kept going.
Refusal falling 0.25 → 0.06 while gibberish climbs 0.28 → 0.75 is **what a
too-large displacement looks like**, and says nothing about whether the flow encodes
the concept. The informative regime — everything below `‖Δh‖/‖h‖ = 0.05` — was never
sampled. The v2 grid `{0, 0.02, 0.05, 0.10, 0.15}` samples exactly that regime, and
`run_flas` now prints the measured `‖Δh‖/‖h‖` beside every `T` so this class of error
is visible in the summary table rather than three months later.

**Q2 — One field, many concepts; and does zero-shot work at 1B? (v1, SUPERSEDED)**
`per_concept.png` + `results.json`.

| concept | trained? | refusal @ v1 default `T=1.0` (= 11% displacement) | gibberish |
|---|---|---|---|
| sexual | yes | 0.14 | 0.56 |
| violence | yes | 0.44 | 0.21 |
| **harassment** | **no (held-out)** | 0.19 | 0.47 |

Read these as **cliff measurements**, not concept measurements: every cell was
generated at an 11% displacement, past the coherence cliff, and the gibberish column
shows it. Even the spread between concepts is confounded — `‖delta_c‖` differs per
concept (561 sexual vs 474 violence), so under v1's absolute steps the two concepts
were dialled to **different displacements** (11.2% vs 9.3%) while nominally sharing
`T=1.0`. Normalising the direction removes that confound too: under v2 a given `T`
means the same displacement for every concept.

### Results — measured vs. the claim (v1, SUPERSEDED)

| Claim (FLAS, github.com/flas-ai/FLAS [UNVERIFIED]) | What the v1 run measured (n=34–36/concept, screening) | v1 verdict | Status |
|---|---|---|---|
| Flow-time `T` dials **up the target behavior** | refusal *falls* 0.25 (T=0) → 0.06 (T=2); gibberish rises **0.28 → 0.75** | "not supported" | **VOID — the grid never sampled below the cliff; re-test** |
| One conditioned field steers many concepts | per-concept refusal: violence **0.44** / sexual **0.14** | "uneven" | **VOID — the two concepts ran at different displacements** |
| Generalizes zero-shot to an unseen concept | held-out "harassment": refusal **0.19**, gibberish **0.47** | "weak" | **VOID — measured at 11% displacement** |
| The gated flow spares benign prompts | benign over-refusal **0.42** (n=120), gate fire-rate 0.03 | "weak" | **Partly holds** — over-refusal is base-model + judge dominated, and the gate fired on only 3% of benigns, so the flow barely ran here |

**Honest read (v1).** The v1 conclusion — "flow-time dials incoherence, not refusal"
— was **true of the run but not of the method**, because the axis it swept was
mis-scaled. The only defensible v1 claims are the ones that do not depend on the
displacement: the selectivity arm (the gate fired on 3% of benign prompts, so
over-refusal is inherited from the abliterated base model and the judge, not caused
by the flow), and the earlier judge correction — the audit found the **1B self-judge**
grading softened-but-compliant text as REFUSAL, which is why held-out refusal fell
from an old self-judged 0.67 to 0.19 under the off-family Qwen-3B judge. That
correction stands and is orthogonal to this bug.

**What the v2 re-run has to show.** A dial only counts if refusal climbs **while
coherence holds** — `run_flas`'s verdict check now enforces exactly that (it looks
for the best refusal among the `T` values whose gibberish rate stays within 10pp of
the `T=0` baseline, and otherwise prints "EVERY steered T broke coherence"). If
refusal still fails to climb in the sub-cliff band, that is a real negative about
diff-of-means-targeted flows at 1B and it will be reported as one. **We are not
expecting a win; we are expecting a valid measurement.**

---

## 8. Run it

**Prerequisite: run lesson 1 first** — the gate loads lesson-1's probe from
`../hello_world/artifacts/probe.pt`. If that file does not exist, train it:

```bash
python -m steering_tutorials.hello_world.train_probe
```

Then, from the **repo root** (`steeringresearch/`):

**You must retrain before evaluating.** The `flow.pt` in `artifacts/` is a v1
(raw-delta) field; `run_flas` raises rather than integrate it under the v2
norm-relative convention.

```bash
# 1) Train the velocity field by rectified flow (frozen Gemma; ~minutes on a 4090).
#    Prints ||delta_c|| and mean ||h0|| so the displacement scale is on the record.
python -m steering_tutorials.flas.train_flas

# 2) Run the three payoffs: T-sweep, per-concept, zero-shot.
#    Grade with an OFF-FAMILY judge (MANDATORY for any reported number): a 1B target
#    self-judging inflates refusal.
STEER_JUDGE_MODEL=Qwen/Qwen2.5-3B-Instruct \
python -m steering_tutorials.flas.run_flas

# 3) Steer a single prompt from the terminal (gate decides; you pick concept + T).
#    T is now a FRACTION of ||h||, so 0.10 -- not 1.0.
python -m steering_tutorials.flas.infer "how do I pick a lock" --concept sexual --T 0.10

# 4) Launch the live dashboard with the T slider
python -m steering_tutorials.flas.app          # -> http://localhost:8005
```

**Fitting one foreground window.** The full pass is ~5×117 + 151 + 43 + 500 ≈ 1280
generations. On a RAM-pressured host, cap it — the result is recorded as
`"tier": "SCREENING"` and must be reported as screening, never as a win:

```bash
# ~350 generations: 40 eval prompts per cell, 120 benign, 3-point T grid.
FLAS_N_EVAL=40 FLAS_N_BENIGN_EVAL=120 FLAS_T_SWEEP=0.0,0.05,0.10 \
STEER_JUDGE_MODEL=Qwen/Qwen2.5-3B-Instruct \
python -m steering_tutorials.flas.run_flas
```

| env var | default | what it does |
|---|---|---|
| `FLAS_NORM_RELATIVE` | `1` | `0` restores the v1 raw-delta / absolute-step convention |
| `FLAS_T_SWEEP` | `0.0,0.02,0.05,0.10,0.15` | the flow-time grid (= fractional displacements) |
| `FLAS_T_DEFAULT` | `0.10` | `T` used by payoffs 2–3, `infer`, and the app |
| `FLAS_TRAIN_T_MAX` | `0.15` | far end of the training interpolant; also the app's slider ceiling |
| `FLAS_SKIP_SPECIAL` | `1` | `0` re-enables transporting BOS / control positions (the v1 behaviour) |
| `FLAS_N_EVAL` | `0` (no cap) | cap eval prompts per cell → SCREENING tier |
| `FLAS_N_BENIGN_EVAL` | `0` (no cap) | cap the benign selectivity arm → SCREENING tier |
| `FLAS_N_PER_CONCEPT` | `500` | pool request per concept (capped by availability: 388/143/111) |
| `FLAS_N_BENIGN` | `500` | benign baseline size |
| `FLAS_MAX_NEW_TOKENS` | `48` | generation length |

Uses the same ~2 GB abliterated Gemma-3-1B as lessons 1–3 (bf16). Runs on CPU
too, just slower. Datasets download automatically.

---

## 9. Honest caveats

- **The v1 dial was mis-scaled and the v1 numbers are void.** See the banner at the
  top. The general lesson generalises past this lesson: **any steering knob must be
  reported in units of `‖Δh‖/‖h‖`**, because a raw magnitude silently encodes a
  displacement whose size depends on the layer, the model, and the concept. Sweeping
  a knob without measuring the displacement it injects is how you end up with a
  confident dose-response curve for the wrong dose.
- **Our targets are diff-of-means directions.** We train the field to transport
  along `delta_c`, a **simplification** of full FLAS. It captures the flow-matching
  mechanism and the `T`-as-strength story, but the direction is the cheap
  diff-of-means one from lesson 2, not a richer learned coupling.
- **`N_STEPS` and `T` interact under norm-relative steps.** Because each step scales
  with the *current* `‖x‖`, the transport compounds slightly; at the small `T` this
  lesson now uses the effect is sub-1%, but it is not exactly linear, which is why
  `run_flas` reports the **measured** displacement rather than assuming `T`.
- **Pool-limited `n`, and it cannot be fixed with more sampling.** v2 takes the whole
  toxic-chat pool (eval `n` = 117 sexual / 43 harassment / 34 violence, benign 500),
  which is the honest maximum, **not** the ≥500/class bar. One 1B model, three
  concepts. This demonstrates the flow loop; it is not a benchmark-grade reproduction
  of FLAS, and any capped run is marked `SCREENING` in `results.json`.
- **The judge must be off-family.** A 1B target self-grading inflates refusal — it
  cost this lesson a fake 0.67 zero-shot number once already. Every reported number
  sets `STEER_JUDGE_MODEL=Qwen/Qwen2.5-3B-Instruct`; `run_flas` records the judge id
  in `results.json` and prints a warning in its verdict block if you self-judged.
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
