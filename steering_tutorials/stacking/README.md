# Lesson 12 — Stacking: which steering priors STACK vs COMPETE

> **Reference:** a composition lesson over [Contrastive Activation Addition (arXiv:2312.06681)](https://arxiv.org/abs/2312.06681) and [ActAdd (arXiv:2308.10248)](https://arxiv.org/abs/2308.10248). Data: [JailbreakBench (arXiv:2404.01318)](https://arxiv.org/abs/2404.01318).

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


> **Instrument caveat — read before citing any rate on this page.** Every refusal /
> compliance / gibberish number here is scored by a local LLM judge. That judge family
> was calibrated against ground-truth labels and measured **ROC-AUC 0.665–0.751** — below
> the 0.85 bar this course sets for a trustworthy instrument. Small differences between
> arms therefore sit at or below the judge's own noise floor and must not be read as
> effects. See [`../JUDGE_VALIDITY.md`](../JUDGE_VALIDITY.md) for the calibration, the
> continuous-readout improvement (+0.086 AUC, 12.8x faster), and which claims survive.

## The key idea in code

Composing priors is just opening one lesson-2 steering hook per prior at once;
disjoint layers stack, a same-layer collision competes (`stacking.py`):

```python
@contextmanager
def stack_contexts(model, priors):
    with ExitStack() as es:
        for p in priors:                     # one steering hook per prior...
            es.enter_context(
                SteeringContext(model, p.vector, p.layer, p.alpha, p.operation))
        yield                                # ...all live at once during one forward pass

def apply_stack(model, tok, prompt, priors):
    with stack_contexts(model, priors):      # disjoint layers -> gains add (STACK)
        return generate(model, tok, prompt)  # same layer + incompatible op -> COMPETE
```

Full file-by-file walkthrough below.

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

The data is the shared **≥500/class harmful-vs-benign** set (`common.data`,
toxic-chat, prompt-level intent labels, deduped + length-matched; built on
JailbreakBench + `lmsys/toxic-chat`, arXiv:2404.01318 / arXiv:2310.17389).
`config.N_PER_CLASS = 500` splits into `N_EXTRACT = 300` (per class, build the
vector) and a held-out **harmful** eval the ladder is judged on (`N_EVAL = 200`,
capped by `STACK_N_EVAL` for a laptop run). From this one contrast we build a
**single refusal diff-of-means direction** and reuse it everywhere — the three
priors (A, B, B′) differ only in *site* (layer) and *operation*, so the ladder
isolates stack-vs-compete without a concept-overlap confound.

| item | value |
|---|---|
| source / loader | `common.data.load_harmful_benign` (toxic-chat, ≥500/class) |
| size | 500/class → 300 extract (build direction) / up to 200 held-out harmful (judged) |
| direction | one shared refusal diff-of-means, reused at layers 12 & 8, ops `relative_add`/`add` |
| model + judge | abliterated `DavidAU/gemma-3-1b-it-heretic-...` (steered); off-family **Qwen2.5-3B** judge |

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

The ladder (`artifacts/results.json`, **n = 150 held-out harmful prompts per
rung**, extract 300/class, graded by an **off-family Qwen-3B judge** on the shared
≥500/class toxic-chat pool) walks A → A+B → A+B′ → all-on:

| Claim | What we measured (off-family Qwen-3B judge, n=150/rung) | Verdict |
|---|---|---|
| Disjoint sites STACK — gains add | rung 1 [A] refusal 0.20 → rung 2a [A+B @ L8] refusal **0.06**, gibberish 0.43 → 0.81 | Not shown here — the disjoint-site add *lowered* refusal instead of stacking |
| Same site + same direction COMPETES — no gain over the best single | rung 2b [A+B′ @ L12] refusal 0.073 (vs A's 0.20) — no gain, worse coherence (gibberish 0.77) | Consistent with "compete" — no gain over the best single prior |
| The all-on hybrid OVER-STACKS — gibberish rises | rung 3 gibberish **0.88** (highest), refusal **0.04** (lowest); norm budget highest at 0.274 | Supported — the all-on hybrid is the most degenerate rung, most gibberish + most budget spent |
| The N5 norm budget grows with the stack | budget 0.079 → 0.218 → 0.141 → 0.274 across the rungs | Supported — the disjoint and all-on stacks spend the most budget |

### The measured single prior this table used to omit — `B_refusal_rate`

`results.json` also carries `single.B_refusal_rate = 0.2533` (with
`B_gibberish_rate = 0.4133`): **prior B alone, at L8**, measured on the same 150
held-out prompts at a cost of 150 extra generations. Until now it appeared in no
table on this page, and it is read by no code — `classify_ladder()` takes only
the rungs, `_plot_ladder()` takes only the rungs. Git history shows it was
**never** consumed: it enters `run_stacking.py` in the lesson's first commit
(`1bc1192`) already orphaned, in the same shape, and is unchanged through
`9fbc8ba` and `b1ed2ec`. It is not a leftover from a comparison that was removed
— the comparison it enables was never written.

| single prior | refusal | gibberish |
|---|---|---|
| **A @L12** (the ladder's base rung) | 0.200 | 0.433 |
| **B @L8** (`single.B_refusal_rate`) | **0.253** | **0.413** |

Two consequences, both of which qualify the claims above:

1. **The ladder is anchored on the worse of its two single priors.** B alone
   nominally beats A on *both* axes. The gap itself is inside noise
   (Δ=+0.053, z=1.11 at n=150 — do not read it as "B beats A"), but it means
   the row above labelled "no gain over the **best single** prior" was never
   actually checked against the best single prior that was measured.
2. **The marginals are understated.** `decision.stack_marginal = −0.14` is
   `refusal(A+B) − refusal(A)`. Against the *best* single prior it is
   **−0.193** (z=−4.78) — a 38 % larger collapse. And A+B (0.060) is worse than
   **both** of its constituents, which is competition at B's own site, not
   merely "the disjoint-site add failed to stack".

The `hillclimb_results.json` run below re-measures every standalone prior
against an unsteered baseline for exactly this reason, and reproduces the
ordering (B alone 0.175 > A alone 0.100 at n=40 with the off-family judge).

**Honest read (robust at 500/class, n=150/rung).** The clean stack-vs-compete
*separation* still does not appear, so `decision.verdict` stays honestly
**"INCONCLUSIVE at this scale"** — but the mechanism it warns about is clearly
visible. The base prior is already near the coherence cliff (rung 1: refusal 0.20,
gibberish **0.43**), so *every* addition overspends the norm budget and tips into
word salad: refusal falls **0.20 → 0.06 → 0.04** while gibberish climbs
**0.43 → 0.81 → 0.88** and the budget grows 0.079 → 0.274. Even the nominally
"stackable" disjoint-site add (2a) degrades, because there is no coherence headroom
to add into. The surviving prediction is the **over-stack**: rung 3 spends the most
budget and is the most gibberish — exactly the "all-on hybrid is forbidden"
collapse (CLAUDE.md §9). The finding held from the earlier n≈50 run; the larger N
just firms the numbers. Screening tier (n=150/rung, single seed) — the mechanism is
the lesson; the numbers measure rather than certify it.

---

## 8b. The stacking hill-climb — a wider ladder, pre-registered

`run_stacking.py` tests two of the three clauses of the CLAUDE.md §9 decision
rule (different site; same site + same direction + different operation). It
never tests the third (**near-orthogonal direction**), never runs the two
**meta-layers** §9 says stack on almost anything (norm clamp, CAST gate), and
has no unsteered baseline — so a "marginal" can never be compared to a prior's
standalone effect, and competition is undetectable.

`run_hillclimb.py` closes that. Classifications were written to
[`PREREGISTRATION_hillclimb.md`](PREREGISTRATION_hillclimb.md) **before** the
run; they are reported below unrevised, whether or not the measurement agreed.

```bash
STEER_JUDGE_MODEL=Qwen/Qwen2.5-3B-Instruct \
  python -m steering_tutorials.stacking.run_hillclimb          # measure (resumable)
python -m steering_tutorials.stacking.run_hillclimb --report   # rebuild report, no GPU
python -m steering_tutorials.stacking.run_hillclimb --selftest # CPU, no model
python -m steering_tutorials.stacking.hillclimb                # CPU, no model
```

### The site inventory (what a prior actually *is*)

| id | site (tower · layer) | operation | training signal |
|---|---|---|---|
| **A** | residual stream, block **12** | `relative_add`, α=0.08 | CAA diff-of-means, 300 harmful vs 300 benign |
| **B** | residual stream, block **8** | `relative_add`, α=0.08 | same vector as A (only the SITE varies) |
| **B′** | residual stream, block **12** (A's site) | `add` (raw ActAdd), rescaled to A's magnitude | same vector as A |
| **C** | residual stream, block **12** (A's site) | `relative_add`, α=0.08 | top PC of the extract activations **after projecting out the refusal axis** — cos(C, v)=**3.0e-08** |
| **CLAMP** | A/B/C's layers, running **after** every injector hook | constraint: `δ ← δ·min(1, 0.10·‖h‖/‖δ‖)` | none — a geometric guard (N5) |
| **GATE** | **before** the forward pass, reading pooled L12 acts | condition: multiply the whole stack by 0/1 | lesson-1 MLP probe |

C is an **orthogonality control, not a second concept** — this lesson's pool
carries one labelled contrast, so a genuine second concept vector is not
available honestly.

### Pre-registered STACK/COMPETE matrix (unrevised)

Exactly one unconditional **COMPETE** was predicted — **A × B′** (same site,
same direction, different operation). Every other pair was predicted **STACK**;
A × C and B′ × C were predicted "STACK *until the norm budget is spent*".

### The measured ladder — n=40 held-out harmful, off-family Qwen-3B judge, SCREENING

Each rung adds exactly one prior. The forbidden all-on hybrid is not built: the
ladder contains only priors pre-classified STACK, and the COMPETE pair is a
control outside it.

| rung | added | refusal | p(refusal) | gibberish | N5 budget | **marginal** Δrefusal | added prior's **standalone** effect | benign gibberish |
|---|---|---|---|---|---|---|---|---|
| R0 | — (unsteered) | 0.275 | 0.250 | 0.500 | 0.000 | — | — | 0.000 |
| R1 | A | 0.100 | 0.125 | 0.800 | 0.079 | −0.175 | −0.175 | 0.250 |
| R2 | B (disjoint site) | 0.025 | 0.039 | 0.950 | 0.218 | **−0.075** | −0.100 | 0.800 |
| R3 | C (orthogonal dir) | 0.000 | 0.021 | 0.975 | 0.236 | **−0.025** | −0.175 | 0.900 |
| R4 | CLAMP (meta) | 0.075 | 0.071 | 0.925 | 0.223 | **+0.075** | n/a (not an injector) | 0.900 |
| R5 | GATE (meta) | 0.125 | 0.130 | 0.775 | 0.223 | **+0.050** | n/a (not an injector) | **0.000** |

Standalone controls, same 40 prompts: **B alone 0.175** / gib 0.675 · **C alone
0.100** / gib 0.700 · **B′ alone 0.175** / gib 0.725 · **[A,B′] 0.000** / gib
0.975, budget 0.141.

### What contradicted the pre-registration

1. **Direction specificity is absent — the biggest result here.** At the same
   site and the same α, the refusal diff-of-means (A, 0.100) and an **exactly
   orthogonal** direction (C, cos=3e-08, 0.100) are **indistinguishable**
   (Δ=0.000). Likewise B′ alone (0.175) equals B alone (0.175). **No rung of
   this ladder can be attributed to the refusal concept** — what is being
   measured is coherence damage, which any direction of this magnitude
   produces. This also invalidates the premise that C would be a "second,
   different prior": at α=0.08 the two are the same intervention in all but
   name.
2. **No prior helps at all.** Every single prior scores *below* the unsteered
   baseline (0.275): A 0.100, B 0.175, C 0.100, B′ 0.175. Steering "refusal
   back in" *reduces* measured refusal, because gibberish rises 0.500 → 0.800
   and takes the mass. A ladder cannot show gains adding when there are no
   gains.
3. **The pre-registered competition test is degenerate here.** *marginal <
   standalone* returns **False** for both R2 and R3 — a prior that harms *less*
   inside the stack than alone reads as "not competing". By the README's own
   criterion (*no gain over the best single prior*) every stack rung competes:
   R2 −0.150, R3 −0.175, R4 −0.100, R5 −0.050 versus B alone (all z≤−1.37,
   R2/R3 z≤−2.31). Both criteria are reported; the pre-registered one was not
   swapped out after the fact.
4. **P4 failed on magnitude, but the CLAMP classification held.** The clamp was
   predicted to cut gibberish by ≥0.10; it cut it by **0.050** (and *raised*
   refusal +0.075, so it did not compete). The mechanism is visible in the
   budget: 0.236 → 0.223 only. The clamp captures `h_base` *inside the same
   forward pass*, so at L12 its reference is already perturbed by the upstream
   L8 injector — it is a **per-site local** constraint, not a global manifold
   constraint, and it cannot claw back accumulated cross-layer drift.
5. **P5 failed on the harmful clause, and the cause is the gate, not the
   stack.** The probe fires on only **53 %** of harmful prompts (and **0 %** of
   benign), so R5 is a 53/47 mixture of steered and unsteered — harmful refusal
   moved +0.050, just over the |0.05| falsifier. The gate is **not
   behaviourally inert** on harmful here.
6. **The ladder anchor.** Replicated at n=40: the base rung [A] (0.100) is not
   the best single prior (B, 0.175).

### What the pre-registration got right

- **A × B′ = COMPETE — confirmed.** `[A,B′]` scores **0.000**, below *both*
  constituents (A 0.100, B′ 0.175; z=−2.91 vs the best), with the highest
  gibberish of any two-prior cell (0.975).
- **P3 held.** Adding the orthogonal direction cost less coherence
  (Δgibberish **+0.025**) than adding the disjoint-site prior (**+0.150**) — the
  direction clause is gentler than the site clause. Read with care: 0.950 →
  0.975 is near the ceiling.
- **GATE = STACK — upheld as a classification.** It is the only prior on this
  page that improves an axis without costing another: benign gibberish
  **0.900 → 0.000**, benign refusal restored to the unsteered 0.550, harmful
  refusal *up* 0.075 → 0.125. It stacks because it consumes no norm budget
  (0.223 unchanged) — it decides *whether*, never *what*.
- **The committed artifact regenerates.** `refusal_vector.pt` was recomputed
  from the code beside it: cosine **1.0**, norm 342.3414 identical, n=300
  identical (`vector_check.reproduces = true`).

### Reading this honestly

**SCREENING tier**: n=40 harmful / 20 benign, single seed, one α. The judge is
below its own 0.85 validity bar. The unsteered gibberish rate is already 0.500,
so this ladder is measured on a substrate with almost no coherence headroom —
which is itself the finding: **at α=0.08 on this abliterated 1B, the norm-budget
clause of §9 dominates every other clause, and the site/direction distinctions
the lesson is built to demonstrate are not resolvable.** The right next
experiment is an α sweep down to where a single prior is coherent, then rebuild
the ladder there — not more rungs at this α.

*(`AUDIT.md` on this lesson is **stale**: its check #3 verifies the README
against refusal 0.667/0.333/0.667/0.50 at n≈12, numbers that no longer exist in
`results.json` (now 0.20/0.06/0.073/0.04 at n=150). It should be re-run.)*

---

## 9. Honest caveats

- **Screening toy.** Gemma-3-1B with n=150 held-out prompts/rung and an
  off-family Qwen-3B judge is **screening**, not evaluation (CLAUDE.md §7: n≤3 seeds is
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
