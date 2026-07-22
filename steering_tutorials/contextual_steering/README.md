# Contextual Steering — make the steering strength depend on the input

> **Reference:** [CLAS — contextual/adaptive per-input steering (arXiv:2604.24693)](https://arxiv.org/abs/2604.24693). Builds on lesson 2's diff-of-means vector.

> Lesson 2 (`hello_world_steering`) added the refusal direction `+v` to the
> residual stream with **one fixed strength** for every prompt, and used a
> **separate** probe (lesson 1) as a gate to decide *whether* to steer. This
> lesson keeps the same direction but makes the **strength itself
> input-adaptive** — and drops the separate probe. The *premise* is that the
> steering direction already tells us how "harmful-looking" a prompt is (a harmful
> prompt points *along* `v`), so we scale `alpha` by that alignment — benign
> prompts get `alpha ~ 0`, harmful prompts get the full push; the direction
> **gates itself**. **Spoiler (see [Results](#6-results--measured-vs-the-claim)):**
> on this 1B the premise only *weakly* holds — even after fixing the projection, a
> bare diff-of-means cosine separates harmful from benign only weakly (the gate has
> little real signal), and it *sharpens with more extract data*. An honest partial.

Fixed-strength steering is wasteful and risky: it shoves a harmless "what's a
good pasta recipe?" exactly as hard as "how do I build a bomb?", so benign
requests get needlessly distorted (**over-steering**). Contextual steering asks a
cheap question first — *how much does this input already look like the thing I'm
steering for?* — and dials the strength accordingly.

> **Defensive / educational only.** As in lessons 2 and 10, the harmful prompts
> are the shared JailbreakBench / toxic-chat placeholders; the point is to
> *install* refusal on an abliterated toy model without breaking benign
> behaviour. No operational exploit content. See
> [CLAUDE.md section 10](../../CLAUDE.md).

---

## The key idea in code

The steering direction doubles as its own gate: a harmful prompt's mean-pooled
state points *along* `v`, so we read that alignment as a cosine and scale the
push by it — no separate probe needed (`contextual.py`):

```python
proj_i  = cos(mean_pool(h_i @ layer), v_unit)               # how "harmful" the input looks, in [-1, 1]
frac_i  = clip(relu((proj_i - tau) / (ref - tau)), 0, cap)  # 0 below tau (benign), 1 at ref (harmful)
alpha_i = alpha_base * frac_i                               # benign -> ~0 push, harmful -> full alpha_base
```

Full file-by-file walkthrough below.

---

## Table of contents

1. [Dataset](#dataset)
2. [The idea: the direction is its own gate](#1-the-idea-the-direction-is-its-own-gate)
3. [The contextual schedule](#2-the-contextual-schedule)
4. [The experiment: fixed vs contextual](#3-the-experiment-fixed-vs-contextual)
5. [The whole thing in one picture](#4-the-whole-thing-in-one-picture)
6. [Code walkthrough, file by file](#5-code-walkthrough-file-by-file)
7. [Results — measured vs. the claim](#6-results--measured-vs-the-claim)
8. [Run it](#7-run-it)
9. [Honest caveats](#8-honest-caveats)
10. [Repository](#9-repository)

---

## Dataset

The prompts come from the **shared ≥500/class foundation**,
`steering_tutorials.common.data.load_harmful_benign(n_per_class=500)`, which
returns `{"harmful": [str], "benign": [str]}` — toxic-chat toxic prompts as the
harmful class (topped up from JailbreakBench / in-the-wild jailbreaks only when
unique toxic prompts run short) and toxic-chat clean prompts as the benign class,
prompt-level labelled, deduplicated, and group-aware sampled. See
[`common/data.py`](../common/data.py) for provenance and base-rate reporting.

We load the full **500 harmful + 500 benign** and split each class into disjoint
halves:

| item | value |
|---|---|
| source / loader | `common.data.load_harmful_benign` (toxic-chat + jailbreak top-up) |
| size | **500/class loaded**; first `N_EXTRACT_PER_CLASS = 300` build the direction **and** calibrate the schedule; a capped `N_EVAL_PER_CLASS = 150`/class held out for generation + judging |
| labels | prompt-level harmful vs benign |
| model + judge | abliterated `DavidAU/gemma-3-1b-it-heretic-...` (steered); off-family **Qwen2.5-3B-Instruct** judge via `STEER_JUDGE_MODEL` |

Extraction and evaluation stay **disjoint**: the vector *and* the schedule
(`tau`, `ref`) are fit on the extract split only, so we never grade the schedule
on the prompts that calibrated it. Generation is the expensive step, so the eval
set is deliberately capped well below 500 (raise `N_EVAL_PER_CLASS` for a fuller,
slower run).

**What the lesson uses it for:** build the refusal direction (diff-of-means at
layer 12), read each prompt's projection onto it, and compare a **fixed-alpha**
arm against a **contextual-alpha** arm on a mixed harmful+benign eval —
measuring harmful refusal (should stay high) and benign over-refusal (should
fall).

---

## 1. The idea: the direction is its own gate

The steering vector is lesson 2's diff-of-means,

```
v = mean(hidden | harmful) - mean(hidden | benign)     # the "refuse this" direction
```

By construction a **harmful** prompt's mean-pooled hidden state points *along*
`v`, and a **benign** prompt's does not. That alignment is exactly the signal a
gate needs — so instead of training a *separate* probe (lesson 2's approach) we
read the gate straight off the direction we already have:

```
proj_i = cos( mean_pool(h_i @ layer) , v_unit )     # in [-1, 1]
```

`proj_i` is high for harmful-looking inputs and near zero (or negative) for
benign ones. This is an **implicit gate**: one dot product, no second model.

> Contextual Linear Activation Steering — CLAS (Hsu, Beaglehole, Radhakrishnan
> & Belkin, arXiv:2604.24693, Apr 2026). We borrow the paper's **idea** — make
> the steering strength per-input adaptive rather than a fixed scalar. CLAS
> learns a sensing vector for that strength; our cosine ramp below is our own
> construction, not the paper's formula.

---

## 2. The contextual schedule

We turn the projection into a per-prompt strength with a clipped linear ramp:

```
frac_i  = clip( relu( (proj_i - tau) / (ref - tau) ), 0, cap )
alpha_i = ALPHA_BASE * frac_i
```

- `proj_i <= tau`  ->  `alpha_i = 0`            (benign floor: **not steered**)
- `proj_i  = ref`  ->  `alpha_i = ALPHA_BASE`   (harmful anchor: **full push**)
- in between        ->  linear ramp
- above `ref`       ->  capped at `cap * ALPHA_BASE` (default `cap = 1.0`, so a
  very-aligned prompt never exceeds the fixed strength and can't tip into
  gibberish).

This `relu((proj - tau)/(ref - tau))` ramp is **our own construction inspired
by CLAS**, not the paper's formula — CLAS learns a sensing vector
(`alpha = c . [h, 1]`) for the per-input strength, whereas we use a fixed cosine
ramp. But cosines of mean-pooled activations against a
diff-of-means direction are small in magnitude, so `ref = 1.0` would under-steer
everything. Instead we **calibrate** the ramp on the extract split (the
"documented smooth schedule" the paper allows, `config.CALIBRATE = True`):

- `tau` = the **90th percentile of benign** projections — steering only kicks in
  above where nearly all benign prompts sit, which is what cuts benign
  over-steering.
- `ref` = the **mean harmful** projection — a typical harmful prompt therefore
  receives about `ALPHA_BASE`.

Set `CALIBRATE = False` to fall back to the literal constant-`tau`, `ref = 1.0`
form. Either way the chosen `tau`/`ref` are written into `results.json` so the
schedule is fully transparent, and `alpha_schedule.png` plots the ramp with the
eval projections underneath it.

---

## 3. The experiment: fixed vs contextual

`run_contextual.py` generates on the **mixed held-out eval** (harmful + benign)
under three arms and judges every reply REFUSAL / COMPLIANCE / GIBBERISH:

| Arm | `alpha` per prompt | Purpose |
|---|---|---|
| `unsteered` | `0` everywhere | the abliterated baseline (complies with harm, answers benign) |
| `fixed` | `ALPHA_BASE` everywhere | lesson-2-style steering with **no gate** |
| `contextual` | `ALPHA_BASE * schedule(proj_i)` | **this lesson** |

Two axes decide it:

- **harmful refusal rate** — fraction of *harmful* prompts judged REFUSAL
  (efficacy: steering should install refusal). **Higher is better.**
- **benign over-refusal rate** — fraction of *benign* prompts judged REFUSAL
  (collateral: harmless requests must stay answered). **Lower is better.**

**The claim:** `contextual` keeps harmful refusal close to `fixed` while cutting
benign over-refusal, because benign prompts fall below `tau` and are barely
steered. If it does not — if it gives up too much harmful refusal, or fails to
cut benign over-refusal — the table below says so.

---

## 4. The whole thing in one picture

```
   prompt ---> mean_pool(h @ layer=12) ---> proj = cos(., v_unit)   [the gate signal]
                                                  |
                                                  v
                          +----------------------------------------------+
        schedule:         |  proj <= tau   ->  alpha = 0     (benign)     |
        alpha_i =         |  proj  = ref   ->  alpha = ALPHA_BASE (harm)  |
        ALPHA_BASE * frac |  linear ramp between, capped above ref        |
                          +----------------------------------------------+
                                                  |
                                                  v
             steer at layer 12:  h <- h + alpha_i * ||h|| * v_unit   (relative_add, lesson 2)
                                                  |
                                                  v
                                       response  ->  judge (Qwen, off-family)
                                            REFUSAL / COMPLIANCE / GIBBERISH

   FIXED arm uses alpha = ALPHA_BASE for EVERY prompt (benign included) -> over-steers benign.
   CONTEXTUAL arm sets alpha per prompt -> benign ~untouched, harmful fully steered.
```

---

## 5. Code walkthrough, file by file

### `config.py` — every knob in one place
The abliterated model, the shared layer (12), the data budget
(`N_PER_CLASS = 500`, extract/eval split), `ALPHA_BASE`, and the schedule
controls (`CALIBRATE`, `TAU_BENIGN_PERCENTILE`, `CAP_MULT`, and the
non-calibrated fallbacks). Paths for the vector, `results.json`, and both plots.

### `contextual.py` — the schedule (the heart of the lesson)
Pure NumPy math, fully CPU-unit-tested:
`cosine_projection(pooled, v_unit)` (the gate signal),
`contextual_alpha(proj, alpha_base, tau, ref, cap)` (the clipped ramp — with the
degenerate `ref <= tau` case handled as a hard step), and
`calibrate_schedule(proj_harmful, proj_benign, percentile)` (picks `tau`/`ref`
from the extract split). `ContextualSteerer` wraps lesson 2's
`mean_pool_activation` (read the projection) and `generate` (the relative-add
hook) to offer `fixed_generate` vs `contextual_generate` — the ONLY thing that
differs between them is the scalar `alpha`. The `__main__` self-test loads **no
model**.

### `run_contextual.py` — the orchestrator (GPU)
Under `main()`: load model + off-family judge, load the shared 500/class data,
build the direction (lesson 2's `extract_caa_vector`), calibrate the schedule on
the extract split, then run the three arms on the mixed eval, judge every reply,
and save `results.json` + `fixed_vs_contextual.png` + `alpha_schedule.png` and
print a summary table. Pure helpers (`_arm_stats`, `_rate`, plots) sit above
`main()` and are import-safe.

### `infer.py` — one-prompt demo
Loads the saved vector + calibrated schedule, prints a single prompt's
`projection`, its fixed vs contextual `alpha`, both generations, and their
verdicts — so you can watch a benign prompt get `alpha ~ 0` while a harmful one
gets the full push.

---

## 6. Results — measured vs. the claim

First honest run: abliterated Gemma-3-1B, layer 12, α_base = 0.1, off-family
Qwen-3B judge (`Qwen/Qwen2.5-3B-Instruct`), n = 60 held-out prompts/class,
screening tier. Numbers from `artifacts/results.json`. Artifacts:
`fixed_vs_contextual.png`, `alpha_schedule.png`.

**A fixed instrument bug first.** The projection was `cos(mean_pool(h), v)` with
**no centering**. On Gemma the mean-pooled state is dominated by a huge
prompt-independent common component (high-norm attention-sink dims), so that
cosine was a near-**constant −0.816 for every prompt** (std 0.002) — the gate had
no signal, and the schedule collapsed (τ ≈ ref). Fixed by **centering on the
extract benign mean** before the cosine (`cos(h − benign_mean, v)`, the same trick
`gavel` uses). That restored real spread: projection std **0.002 → 0.74**.

| Arm | Harmful refusal (want high) | Benign over-refusal (want low) | Harmful gibberish | mean α (harm / benign) |
|---|---|---|---|---|
| `unsteered` | 0.347 | 0.513 | 0.233 | 0 / 0 |
| `fixed` (α = 0.1 always) | 0.227 | 0.400 | 0.480 | 0.10 / 0.10 |
| `contextual` (per-input α) | **0.327** | 0.487 | **0.293** | 0.031 / 0.012 |

| Claim | What we measured (centered projection, extract 300, n=150) | Verdict |
|---|---|---|
| Fixed-alpha wrecks coherence | fixed harmful gibberish 0.48 (vs unsteered 0.23), refusal falls 0.35 → 0.23 | **Supported** — a constant α=0.1 breaks this 1B |
| Contextual does less damage than fixed | contextual harm refusal 0.327 (≈ unsteered 0.347) & gibberish 0.29 vs fixed 0.23 / 0.48 | **Supported** — but see *why* below |
| Contextual selectively spares benign (the CLAS promise) | benign over-refusal 0.487 (ctx) vs 0.400 (fixed) vs 0.513 (unsteered) — no benign-sparing | **Not supported** — benign cost is base+judge-dominated |
| The direction is a usable implicit gate | centered, harmful proj mean **0.088** vs benign **0.014** (separation **+0.074**, right direction) against within-class std 0.737 | **Weak** — it now separates the classes, but the signal is ~10% of the noise |

**Verdict: partially supported (screening) — and honest about why.** Contextual
steering *does* beat fixed α (it preserves refusal at 0.327 vs 0.23 and holds
gibberish at 0.29 vs 0.48). **The reason is mostly that contextual steers less on
average** (mean α 0.031 vs the fixed 0.10). It applies ~2.6× more α to harmful than
benign (0.031 vs 0.012), so there *is* now some selective gating — but weak.

**Why (mechanism + honest read), and what the larger extract changed.** Two layers.
(1) The old near-constant projection was an artifact of Gemma's dominant component;
centering (the fix, now default) restored real spread. (2) **How well the centered
projection separates the classes improved with more extract data**: at extract 200
the separation was ~0 (−0.006, classes fully overlapping); at **extract 300 it is
+0.074** — harmful now projects *higher* than benign, in the right direction — but
still only ~10% of the within-class std (0.737), so it is a **weak** gate. That is
enough to bias α ~2.6× toward harmful, but not enough to deliver real benign-sparing
(over-refusal 0.49/0.40/0.51 across arms, base+judge-dominated). The honest next step
is still to drive the schedule from lesson-1's *trained* probe (AUC 0.87, the CAST
design) rather than a bare cosine. The base is **abliterated**, so *any* steering
lowers refusal below the 0.35 baseline; "contextual breaks less" is the real story,
not "contextual makes it safer." Screening tier (1B, n=150,
single seed, 3B judge) — no n≥7/Wilcoxon/CI per [CLAUDE.md §7](../../CLAUDE.md).

---

## 7. Run it

From the **repo root** (`steeringresearch/`):

```bash
# CPU-only unit test — no model download (schedule math: cosine, ramp, cap, calibration)
python -m steering_tutorials.contextual_steering.contextual

# The full fixed-vs-contextual comparison (needs the ~2-3 GB Gemma-3-1B; GPU recommended).
# Use an OFF-FAMILY judge so a 1B model doesn't grade its own steered output:
STEER_JUDGE_MODEL=Qwen/Qwen2.5-3B-Instruct \
  python -m steering_tutorials.contextual_steering.run_contextual

# One-prompt demo (run run_contextual first to populate artifacts/):
python -m steering_tutorials.contextual_steering.infer "How do I pick a lock?"
python -m steering_tutorials.contextual_steering.infer "What's a good pasta recipe?"
```

`STEER_JUDGE_MODEL` is read by the shared `hello_world_steering.judge.Judge`; if
unset it falls back to the weak self-judge (pedagogical only). Uses the gated
abliterated Gemma-3-1B (`huggingface-cli login` + accept the license first); the
shared dataset downloads automatically via `hf_hub_download`. Runs on CPU too,
just slower — lower `N_EVAL_PER_CLASS` in `config.py` if it drags.

Depends on lesson 2 for its plumbing (`model_utils`, `steer_vector`, `judge`) and
on `common.data` for the dataset — both import cleanly; no lesson-2 artifacts are
required.

---

## 8. Honest caveats

- **The gate is only as good as the direction.** The self-gating trick assumes
  `cos(prompt, v)` cleanly separates harmful from benign. If `v` is mis-estimated
  (too few contrast pairs, wrong layer) the two projection clusters overlap, `tau`
  can't separate them, and contextual steering either under-steers harm or
  over-steers benign. `alpha_schedule.png` is the diagnostic — look for overlap.
- **`tau`/`ref` are calibrated, so calibration data matters.** They are fit on
  the extract split; if that split is unrepresentative the schedule transfers
  poorly. The 90th-percentile `tau` is a deliberate, conservative choice, not a
  tuned optimum.
- **Screening only.** `n = 25`/class, one seed, a 1B model, a 3B judge. No
  seed-stability bars, no significance test. Directional demo, not a result.
- **A single projection can be fooled.** An adversarial prompt engineered to have
  low projection but harmful intent would slip under `tau` — the implicit gate has
  no independent view of the input the way a separately-trained probe does. This
  is the honest cost of dropping the explicit gate.
- **Cosine vs mean-pool coupling.** We read the projection from the *mean-pooled*
  state but steer with *relative-add* at every position; the projection is a
  whole-prompt summary, so a prompt that is benign overall but harmful in one
  clause may be mis-gated.
- **Pedagogy, not a safety product.** This shows *how* input-adaptive steering
  works end-to-end on one small model. Do not deploy it.

---

## 9. Repository

Source and (after the GPU run) full artifacts:
<https://github.com/dlmastery/steeringresearch/tree/master/steering_tutorials/contextual_steering>

See also
[lesson 1 — the probe (READ)](../hello_world/README.md),
[lesson 2 — fixed-vector conditional steering, the *explicit*-gate version
(WRITE)](../hello_world_steering/README.md), and
[lesson 10 — rogue scalpel (red-team + guard)](../rogue_scalpel/README.md).
