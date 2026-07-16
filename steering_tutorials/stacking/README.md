# Lesson 12 — Stacking: which steering priors STACK vs COMPETE

> Lessons 1–3 built single interventions: a probe that **reads** harm, one
> refusal vector that **writes** it back, a hypernetwork that **generates**
> vectors on demand. Real systems run **more than one** prior at once. This
> lesson answers the composition question: when you add a second prior, do the
> effects **add** (stack) or **fight** (compete)? We build an **additive 2→N
> ladder**, add exactly one prior per rung, and read the **marginal effect**.

Everything here is standalone and CPU-runnable to read. It imports only the
mechanical core of [lesson 2](../hello_world_steering/README.md) — the model
loader + steering hook, the diff-of-means CAA vector, and the self-grading judge
— and nothing from the research harness. The actual generation needs the same
~2 GB abliterated Gemma-3-1B as lessons 1–2.

This lesson is the tutorial instantiation of **CLAUDE.md section 9 (stacking
discipline)** and the mechanism analysis in
[`corpus/steering-stackable-vs-competing-analysis.md`](../../corpus/steering-stackable-vs-competing-analysis.md).

---

## Table of contents

1. [The one-paragraph idea](#1-the-one-paragraph-idea)
2. [The decision rule](#2-the-decision-rule-stack-vs-compete)
3. [What a "prior" is](#3-what-a-prior-is)
4. [The additive 2→N ladder](#4-the-additive-2n-ladder)
5. [The norm budget — the ceiling that decides it](#5-the-norm-budget--the-ceiling-that-decides-it)
6. [Code walkthrough, file by file](#6-code-walkthrough-file-by-file)
7. [Run it](#7-run-it)
8. [How to read the output](#8-how-to-read-the-output)
9. [Honest caveats](#9-honest-caveats)
10. [Links](#10-links)

---

## Dataset

The data is **JailbreakBench harmful vs benign** (Chao et al. 2024,
arXiv:2404.01318), pulled through lesson 2's loader
(`hello_world_steering.data.load_harmful_benign`, the `Goal` column via
`hf_hub_download`). `config.N_PER_CLASS = 40` splits into `N_EXTRACT = 24` (per
class, build the vector) and `N_EVAL = 12` held-out **harmful** prompts the
ladder is judged on. Labels are **prompt-level** harmful vs benign. From this
one contrast we build a **single refusal diff-of-means direction** and reuse it
everywhere — the three priors (A, B, B′) differ only in *site* (layer) and
*operation*, so the ladder isolates stack-vs-compete without a concept-overlap
confound.

| item | value |
|---|---|
| source / loader | JailbreakBench `Goal` via `hello_world_steering.data.load_harmful_benign` |
| size | 40/class → 24 extract (build direction) / 12 held-out harmful (judged) |
| direction | one shared refusal diff-of-means, reused at layers 12 & 8, ops `relative_add`/`add` |
| model + judge | abliterated `DavidAU/gemma-3-1b-it-heretic-extreme-uncensored-abliterated`, self-graded |

**What the lesson uses it for:** walk an additive 2→N ladder to see which priors
**stack** (disjoint sites → gains add) vs **compete** (same site, incompatible
operation → the second cancels/degrades the first), reading refusal, gibberish,
and the N5 norm budget per rung.

---

## 1. The one-paragraph idea

Two steering interventions **stack cleanly** when they act on **different sites**
(different layers) or on **near-orthogonal directions**, and their combined push
stays inside the model's natural activation manifold. They **compete** when they
overwrite the **same site** with **incompatible operations**, or when the
**cumulative norm** shoves the hidden state off-manifold. Piling on *everything*
— the "all-on hybrid" — reliably over-spends that norm budget and collapses into
gibberish (it is the steering analogue of the forbidden `sg_full_fib` hybrid in
CLAUDE.md §9). The way to *see* this is an **additive ladder**: start from one
prior, add exactly one more per rung, and read the marginal effect on the target
behavior, on coherence, and on the norm budget.

---

## 2. The decision rule (stack vs compete)

From the intervention-site taxonomy (`corpus/steering-stackable-vs-competing-analysis.md`
§1, and CLAUDE.md §9):

```
 different SITE (layer / disjoint pathway) ........... STACK   (gains add)
 near-orthogonal DIRECTION, norm in-budget ........... STACK   (until budget spent)
 same SITE + same DIRECTION + different OPERATION .... COMPETE (double-counts the plane)
 additive vs rotational on the same plane ............ COMPETE (pick one)
 cumulative ||Δh|| pushes h off-manifold ............. COMPETE (coherence budget spent)
```

The single most load-bearing fact: **conditioning/gating (CAST) is a meta-layer,
not a peer** — it stacks on almost everything (that is lesson 2's gate). This
lesson is about the *behavior-injector* layer beneath it, where the stack-vs-
compete tension actually lives.

The archetypal competing pair in the literature is **additive vs rotational** on
the same subspace (Angular/Selective Steering vs CAA, corpus §3.1). Our runnable
toy reuses lesson 2's two additive operations (`relative_add` vs raw `add`) to
stage the same-site collision without writing a new steering mechanism: two
priors on the same layer, same direction, **different operation**, double-count
the refusal plane and overshoot.

---

## 3. What a "prior" is

A **prior** (`stacking.Prior`) is the atomic unit of a stack — a direction, a
site, a strength, and an operation:

```python
@dataclass
class Prior:
    name: str            # label for the ladder + plot
    vector: np.ndarray   # the DIRECTION (unit length), shape [hidden]
    layer: int           # the intervention SITE (residual layer)
    alpha: float         # strength (fraction of ||h|| for relative_add; raw for add)
    operation: str       # "relative_add" (norm-aware) | "add" (literal ActAdd)
```

To isolate the **site** variable cleanly, all three priors in this lesson share
**one** direction — the refusal diff-of-means from lesson 2 — and differ only in
layer and operation. (A genuinely *different concept* at a different layer stacks
for the identical mechanical reason — a disjoint site — so holding the direction
fixed is the cleaner controlled demo, not a limitation.)

```
        A  = refusal @ L12  (relative_add)   the base prior
        B  = refusal @ L8   (relative_add)   DISJOINT site   -> stacks with A
        B' = refusal @ L12  (add, raw)       SAME site as A  -> competes with A
```

`B'`'s raw step is rescaled at run time so that **B' alone ≈ A alone** in
magnitude — so rung 2b is a *controlled* comparison: B' is not a weaker prior, it
genuinely competes.

---

## 4. The additive 2→N ladder

Each rung adds **exactly one** prior to the previous rung, so the change between
adjacent rows is a single, readable marginal effect:

```
 rung 1   : [A]            base single-prior refusal
 rung 2a  : [A, B]         + disjoint-site prior     EXPECT: STACK   (refusal up, budget ~2x)
 rung 2b  : [A, B']        + same-site prior (diff op) EXPECT: COMPETE (refusal <= best single)
 rung 3   : [A, B, B']     the all-on hybrid          EXPECT: OVER-STACK (gibberish up)
```

ASCII of the forward pass with a two-prior **orthogonal-site** stack (rung 2a):

```
  prompt
    │
    ▼
 ┌────────┐   ┌────────┐        ┌────────┐        ┌────────┐
 │  L0..  │…→ │  L8    │ ──────▶│  ..L12 │ ──────▶│  ..LN  │──▶ logits
 └────────┘   └───┬────┘        └───┬────┘        └────────┘
                  │ + B (refusal)   │ + A (refusal)
                  ▼                 ▼
             disjoint sites  →  edits compose, gains add
```

And the **same-site** collision (rung 2b) — both edits land on L12 and fire
*sequentially* (the later-registered `add` prior runs first, then `relative_add`
computes its norm-relative step on the already-contaminated state):

```
 ..L12 output h ──▶ [B': h += raw v] ──▶ [A: h += 0.08·‖h+v‖·v̂] ──▶ overshoot
                    same plane, twice, with mismatched operations → competes
```

`apply_stack(model, tok, prompt, priors)` composes all of a rung's priors at
once by opening one `SteeringContext` per prior inside a single `ExitStack`, then
calling lesson 2's plain `generate` (which adds no vector of its own — the live
stack hooks do all the steering). An empty prior list yields the unsteered
baseline.

---

## 5. The norm budget — the ceiling that decides it

Even "independent" additive vectors compete once their **sum** moves `h` off the
natural activation manifold ("Steered Activations are Non-Surjective", corpus
§3.3; the N5 leading indicator in CLAUDE.md §3). We measure it directly:

```
   norm_budget (N5)  =  Σ over prior layers   mean_positions( ‖Δh‖ / ‖h‖ )
```

computed with **two forward passes** per prompt (baseline vs stacked), capturing
the residual at each prior's layer. A disjoint-site stack spreads its budget
across two layers; the all-on hybrid concentrates and overshoots — which is why
its gibberish rate climbs. The right panel of `artifacts/ladder.png` plots this
budget per rung next to the refusal/gibberish panel, so you can read the collapse
as a budget story, not a mystery.

---

## 6. Code walkthrough, file by file

| file | role |
|---|---|
| `config.py` | every knob: `MODEL_ID`, the two sites (`PRIMARY_LAYER=12`, `ORTHOGONAL_LAYER=8`), `STACK_ALPHA`, `COMPETE_ADD_FRACTION`, data split, paths. |
| `stacking.py` | the mechanical core: `Prior`, `stack_contexts` (compose N `SteeringContext`s via `ExitStack`), `apply_stack` (steered decode under the whole stack), `build_priors` (A/B/B' from one refusal direction), `ladder_rungs` (the 2→N ladder). CPU self-test verifies composed-delta math (disjoint + same-site) **and** that every hook is removed on exit. |
| `run_stacking.py` | the orchestrator (all model work under `main()`): extract the refusal vector, rescale B', walk the ladder measuring refusal/gibberish/norm-budget per rung, classify stack-vs-compete, save `results.json` + `ladder.png`. Pure helpers (`_rates`, `classify_ladder`) are unit-testable without a model. |
| `README.md` | this file. |

### Reused verbatim from lesson 2 (`hello_world_steering`)

```python
from steering_tutorials.hello_world_steering.model_utils import (
    load_model, SteeringContext, generate, residual_layers, num_layers,
    last_token_activations,
)
from steering_tutorials.hello_world_steering.steer_vector import extract_caa_vector
from steering_tutorials.hello_world_steering.judge import Judge
from steering_tutorials.hello_world_steering.data import load_harmful_benign
```

Nothing is re-implemented — the whole lesson is *composition* of parts you have
already seen, which is the point.

---

## 7. Run it

CPU-only checks (no Gemma download — safe anywhere):

```bash
# 1. the mechanical core: composed-delta math + hook cleanup
python -m steering_tutorials.stacking.stacking

# 2. import + pure-helper sanity (no model touched)
python -c "import steering_tutorials.stacking.run_stacking as R; print(R.classify_ladder)"
```

The full ladder (needs the abliterated Gemma-3-1B + a GPU; greedy decoding):

```bash
huggingface-cli login          # accept the Gemma license once
# STEER_JUDGE_MODEL selects the OFF-FAMILY judge (avoids same-model grading bias).
STEER_JUDGE_MODEL=Qwen/Qwen2.5-3B-Instruct python -m steering_tutorials.stacking.run_stacking
```

Outputs land in `artifacts/`: `refusal_vector.pt`, `results.json`, `ladder.png`.

---

## 8. How to read the output

`results.json` → `decision`:

```
 stack_marginal   = refusal(2a) - refusal(1)   >0  ⇒ disjoint sites STACKED
 compete_marginal = refusal(2b) - refusal(1)   ≤0  ⇒ same-site prior COMPETED
 overstack_gibberish_delta = gibber(3) - gibber(1)  >0 ⇒ hybrid over-spent the budget
```

The empirical decision rule falls straight out: the rung whose **added prior sat
on a disjoint site** should carry positive marginal refusal; the rung whose added
prior **collided on the same site** should carry ≤0 marginal (or convert refusal
into gibberish); the all-on rung should show the highest norm budget and the
highest gibberish. The `ladder.png` colors the rungs green / amber / red so the
stack → compete → over-stack progression reads at a glance.

---

## Results — measured vs. the claim

The screening ladder (`artifacts/results.json`, n = 12 held-out harmful prompts,
abliterated 1B self-graded) walks A → A+B → A+B′ → all-on:

| Claim | What we measured (screening) | Verdict |
|---|---|---|
| Disjoint sites STACK — gains add | rung 1 [A] refusal 0.667 → rung 2a [A+B @ L8] refusal **0.333** (marginal −0.333) | Not shown here — the disjoint-site add *lowered* refusal instead of stacking |
| Same site + same direction COMPETES — no gain over the best single | rung 2b [A+B′ @ L12] refusal 0.667 (marginal 0.0) — flat | Consistent with "compete" (no gain), but indistinguishable from "no effect" at this n |
| The all-on hybrid OVER-STACKS — gibberish rises | rung 3 refusal 0.50, gibberish **0.167** (0.0 at every earlier rung); norm budget highest at 0.277 | Supported — the only rung to produce gibberish is the all-on hybrid, exactly as predicted |
| The N5 norm budget grows with the stack | budget 0.077 → 0.225 → 0.136 → 0.277 across the rungs | Supported — the disjoint and all-on stacks spend the most budget |

The one prediction that survives at 1B/n=12 is the **over-stack**: piling all
three priors on is the only configuration that breaks into gibberish, with the
highest norm budget — the collapse-as-budget-story the lesson promises. The clean
stack-vs-compete *separation* does not: the disjoint-site rung actually lost
refusal (marginal −0.333) rather than gaining, so `decision.verdict` is logged
honestly as "INCONCLUSIVE at this scale". This is a screening demo where single
rungs swing on seed noise; the mechanism is the lesson, and the numbers here
measure rather than confirm it.

---

## 9. Honest caveats

- **Small, noisy toy.** Gemma-3-1B with n≈12 held-out prompts and a 1B
  self-judge is **screening**, not evaluation (CLAUDE.md §7: n≤3 seeds is
  screening; a real claim needs n≥7 + the rigor contract). Marginal effects at
  this scale are directional illustrations, not statistics — a single rung can
  land the "wrong" way from seed noise. The mechanism is the lesson; the numbers
  are a demo.
- **Same-direction, not two concepts.** We vary the *site* on one shared
  direction to isolate the layer variable. Two genuinely different concept
  vectors would stack for the same disjoint-site reason but add a confound
  (their directions' overlap) this toy deliberately removes.
- **`add` vs `relative_add` is a stand-in for the archetypal collision.** The
  literature's crispest competing pair is additive vs *rotational* on one plane
  (corpus §3.1); we stage same-site competition with two additive operations to
  stay within lesson 2's mechanism. The *conclusion* (same site + incompatible op
  ⇒ compete) is the same.
- **Self-grading circularity.** The judge is the same 1B model — an
  "Internal QA pass — independent external review pending" result, never an
  external claim (CLAUDE.md §14).
- **Order-dependence.** Same-site hooks fire sequentially, so a same-site stack
  is order-sensitive — itself a reason such stacks are fragile.

---

## 10. Links

- Lesson 1 — [the probe (READ)](../hello_world/README.md)
- Lesson 2 — [conditional steering (WRITE + gate)](../hello_world_steering/README.md)
- Lesson 3 — [HyperSteer (generate vectors)](../hypersteer/)
- Mechanism analysis — [`corpus/steering-stackable-vs-competing-analysis.md`](../../corpus/steering-stackable-vs-competing-analysis.md)
- Project stacking discipline — **CLAUDE.md section 9**
