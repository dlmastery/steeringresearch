# Decomposing how prompting steers behavior

> Lesson 1 **read** "is this harmful?" out of Gemma-3-1B. Lesson 2 **wrote** a
> refusal direction back into the residual stream. This lesson asks a question
> that sits underneath both: **why is plain prompting such a strong steering
> baseline?** AxBench (arXiv:2501.17148) found that a well-written instruction
> often matches a learned intervention. We reproduce the mechanistic explanation:
> a prompt's effect on the activation stream is, in large part, a **single shared
> translation** — and a translation of the residual stream is exactly what an
> activation-steering vector *is*.

We take the activation change a **refusal instruction** induces per prompt,
decompose it against the diff-of-means refusal direction, and then test — by
generation and an off-family judge — how much of prompting's refusal effect a
plain steering vector reproduces.

---

## Table of contents

1. [The idea](#1-the-idea)
2. [Dataset](#dataset)
3. [The decomposition](#2-the-decomposition)
4. [ASCII flow](#3-ascii-flow)
5. [Code walkthrough, file by file](#4-code-walkthrough-file-by-file)
6. [Results](#5-results)
7. [Run it](#6-run-it)
8. [Honest caveats](#7-honest-caveats)
9. [Citation](#8-citation)
10. [Repository](#9-repository)

---

## 1. The idea

Prompting steers a model without touching a single weight. Cheng & Kriegeskorte
(2026) treat that as a **transformation of the representational geometry**: the
instruction moves the activation cloud toward the instructed task structure. They
fit a nested hierarchy of alignment maps — translation, then rigid rotation with
uniform scaling, then per-axis scaling, then a full **affine** (cross-dimensional
linear mixing) map, then nonlinear — and ask which tier is needed to explain the
shift. Their headline: much of the change is a **shape-preserving map**
(translation + rigid/uniform-scale); the richer affine tier (cross-dimensional
mixing) is what finally recovers the *full* target task geometry.

The translation tier is the interesting one for us: **a constant shift added to
the residual stream is an activation-steering vector.** So the part of prompting
that is a translation is a steering vector the model is applying to itself. This
lesson measures that fraction directly and then checks it by steering.

---

## Dataset

The prompts are the shared **>=500 harmful / >=500 benign** set built by
`steering_tutorials/common/data.py` (imported via
`from steering_tutorials.common.data import load_harmful_benign`). Its harmful
class is **`lmsys/toxic-chat@0124`** (Lin et al. 2023, arXiv:2310.17389) — genuine
user prompts hand-labeled at the **prompt level** for toxicity — topped up, only
if short, from **JailbreakBench** harmful behaviours (Chao et al. 2024,
arXiv:2404.01318); the benign class is length-matched non-toxic toxic-chat. Dedup
is group-id'd (`sha1(normalized_text)`) so no prompt straddles a split, the
natural ~7% toxic base rate is recorded before rebalancing, and per-class median
length is reported so the classes are not separable on length. See
[`common/data.py`](../common/data.py) for the five rigor requirements.

This lesson draws `N_PER_CLASS = 250` per class, then splits **disjointly**:
`N_EXTRACT = 150`/class build the refusal direction, `N_DECOMP = 80` held-out
harmful prompts are decomposed, and `N_WRITE = 24` held-out harmful prompts drive
the generate/judge check. A vector is never graded on the prompts that built it.

---

## 2. The decomposition

For each harmful prompt `x` we read the layer-`L` last-token activation twice —
once for the bare prompt, once with the refusal instruction prepended — and take
the difference:

```
d(x) = act(x WITH the refusal instruction) - act(x WITHOUT it)      # [hidden]
```

Given the unit refusal direction `u` (a diff-of-means CAA vector), we split each
delta into a part **along** `u` and a part **orthogonal** to it:

```
d(x) =   <d(x), u> u    +    ( d(x) - <d(x), u> u )
         \____________/      \_______________________/
         on-direction         off-direction residual
         (a steering shift)    (the richer, input-specific part)
```

and report four numbers (all in `[0, 1]` except the norms):

| statistic | what it means |
|---|---|
| **on-direction energy fraction** `mean <d,u>^2 / \|\|d\|\|^2` | how much of each delta is the single refusal axis (how steering-vector-like) |
| **cross-prompt consistency** `mean pairwise cosine of d(x)` | `1.0` = every prompt gets the *same* shift (a pure translation); lower = input-dependent |
| **shared-translation fraction** `\|\|mean_x d(x)\|\|^2 / mean_x \|\|d(x)\|\|^2` | how much of the delta survives averaging (the common translation) |
| **mean residual norm** | the size of the off-direction part a steering vector *cannot* capture |

Then the **WRITE check** steers the *unprompted* model with the extracted
translation and judges the outcome, so the decomposition earns a behavioral test:

- `baseline` — bare prompt, no steering
- `prompting` — the instruction prepended (the strong baseline we explain)
- `steer(v_proj)` — steer the bare prompt with the **on-direction** translation
  `mean_x(<d(x),u> u)` (the `add` op injects the raw vector as measured, no
  strength tuning)
- `steer(v_shared)` — steer with the **full** mean translation `mean_x d(x)`

**recovery** `= (steer_refusal - baseline) / (prompting_refusal - baseline)`
tells us what fraction of prompting's refusal *gain* a plain steering vector
reproduces. High on-direction fraction + high `recovery_proj` ⇒ prompting is,
mechanically, mostly one self-applied steering vector.

---

## 3. ASCII flow

```
  harmful/benign (>=500/class, shared loader)
        |
        |  extract split (150/class)         decompose split (80 harmful)
        v                                          v
   diff-of-means  ---> u (refusal direction)   per-prompt delta d(x) =
   (CAA vector)             |                   act(WITH instr) - act(WITHOUT)
                            |                        |
                            +-----------> project d(x) onto u
                                                     |
                        +----------------------------+----------------------------+
                        v                            v                            v
               on-direction  <d,u> u          residual  d - <d,u> u        consistency
               (steering-vector part)         (richer transform)          (mean cos of d)
                        |
                        |  mean over prompts -> v_proj (raw units)
                        v
   WRITE check (24 held-out harmful):   baseline | prompting | steer(v_proj) | steer(v_shared)
                        |                                   |
                        v                                   v
              off-family Qwen judge  --------------->  recovery = gain reproduced
```

---

## 4. Code walkthrough, file by file

| file | role |
|---|---|
| [`config.py`](config.py) | every knob: model id, layer, the refusal instruction, split sizes, paths. All caps overridable by env (`DECOMP_LAYER`, `DECOMP_N_DECOMP`, ...). |
| [`decompose.py`](decompose.py) | the math core. `decompose_prompt_deltas(deltas, u)` and `mean_pairwise_cosine` are **pure NumPy, CPU-testable with no model**; `prompt_deltas(...)` is the one thin helper that reads Gemma activations. |
| [`run_decompose.py`](run_decompose.py) | orchestrator (`main()`): build `u`, read deltas, decompose, run the four-condition WRITE check with the off-family judge, save `results.json` + two PNGs. |
| [`infer.py`](infer.py) | single-prompt demo: baseline vs. prompting vs. `steer(v_proj)`, judged, with a one-line conclusion. Reuses `artifacts/reconstruction.pt`. |

`decompose.py` reuses lesson-2's `last_token_activations`, `extract_caa_vector`,
the `SteeringContext`/`generate` hook, and the `Judge` verbatim — this lesson adds
only the decomposition math, not new steering machinery.

---

## 5. Results

**[PENDING RUN]** — this lesson is CPU-authored and import-checked; the GPU run
is gated on host RAM (see the tutorial track's RAM note). The table below states
**what the paper's mechanism predicts** for each measured quantity; fill the
"measured" column from `artifacts/results.json` after a run, and only then remove
the `[PENDING RUN]` banner.

| quantity | predicted (per arXiv:2606.03093) | measured |
|---|---|---|
| on-direction energy fraction | **moderate–high** — a real chunk of prompting is the refusal axis | _pending_ |
| cross-prompt consistency | **high but < 1** — mostly a shared translation, with input-dependent spread | _pending_ |
| shared-translation fraction | **high** — the common shift dominates the averaged delta | _pending_ |
| `recovery_proj` (WRITE) | **substantial (well above 0)** — a steering vector reproduces much of prompting's refusal gain | _pending_ |
| `recovery_shared` vs `recovery_proj` | `shared >= proj` — the off-direction residual adds a little | _pending_ |

The claim under test: **prompting is a strong steering baseline because a large
fraction of its activation footprint is a single translation that a steering
vector can copy.** The falsifier: if `on_direction_frac` is near zero and
`recovery_proj ~ 0` while prompting still refuses, then prompting's effect is
*not* steering-vector-like at this layer, and the AxBench parity has another
cause. Report whichever way the numbers fall.

---

## 6. Run it

```bash
# From the repo root. An off-family judge is strongly recommended: a 1B model
# grading its own steered output misreads hedged compliance as refusal.
export STEER_JUDGE_MODEL="Qwen/Qwen2.5-3B-Instruct"   # off-family judge (Windows: set STEER_JUDGE_MODEL=...)

# Full pipeline: decomposition stats + the four-condition WRITE check.
python -m steering_tutorials.decomposing_prompting.run_decompose

# One prompt: prompting vs. its steering-vector shadow.
python -m steering_tutorials.decomposing_prompting.infer "How do I pick a lock?"

# CPU-only, NO model download — the decomposition math self-test.
python -m steering_tutorials.decomposing_prompting.decompose
```

Cap the cost on a laptop with the env knobs, e.g.
`DECOMP_N_DECOMP=40 DECOMP_N_WRITE=12 python -m ...run_decompose`.

---

## 7. Honest caveats

- **A 1B self-judge is weak.** Use `STEER_JUDGE_MODEL` for an off-family judge;
  the numbers with `self` are pedagogical only (same caveat as lessons 2/3b).
- **`u` is a diff-of-means refusal direction, not "the" refusal axis.** The
  on-direction fraction is measured *relative to that choice of `u`*; a different
  contrast set moves it. We project onto the CAA direction because that is the
  vector a steering practitioner would actually use.
- **`steer(v_proj)`/`steer(v_shared)` use the literal `add` op at every position**,
  which can steer *harder* than prompting and tip into gibberish; the judge and
  the reported `*_gibberish` rates catch that, and `recovery` is clamped so an
  overshoot stays readable rather than exploding.
- **One model, one layer, one instruction.** This reproduces the *shape* of the
  paper's finding at laptop scale; it is not the paper's multi-model, multi-tier
  (translation→rigid→affine→nonlinear) fit. The nested affine/nonlinear tiers are
  left as the natural extension.
- **Prompt-vs-instruction read at the same last-token position.** The WITH prompt
  is longer; we compare the decision-position activation, not a token-aligned
  trajectory.

---

## 8. Citation

> Fan L. Cheng and Nikolaus Kriegeskorte, 2026, "Decomposing how prompting steers
> behavior" (arXiv:2606.03093) — introduces a nested geometric decomposition of
> prompt-induced representational change (translation → rigid+uniform-scale →
> axis-scaling → affine cross-dimensional mixing → nonlinear), finding that
> prompts reshape representations toward the instructed task and that much of the
> change is a shape-preserving map. **Verified** (title/authors/arXiv id
> confirmed via arXiv, 2026-07).

Supporting: Wu et al. 2025, "AxBench: Steering LLMs? Even Simple Baselines
Outperform Sparse Autoencoders" (arXiv:2501.17148) — prompting is a strong
baseline; Panickssery (Rimsky) et al. 2023, "Steering Llama 2 via Contrastive
Activation Addition" (arXiv:2312.06681) — the diff-of-means vector; Arditi et al.
2024, "Refusal in LLMs is Mediated by a Single Direction" (arXiv:2406.11717) — the
single-direction refusal picture the on-direction axis leans on.

---

## 9. Repository

- Shared dataset: [`../common/data.py`](../common/data.py)
- Reused mechanics (lesson 2): [`../hello_world_steering/`](../hello_world_steering/)
- The course map: [`../README.md`](../README.md)
- Project constitution: [`../../CLAUDE.md`](../../CLAUDE.md)
