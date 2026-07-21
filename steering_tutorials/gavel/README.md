# GAVEL — rule-based safety through activation monitoring (the READ guardrail)

> **Reference:** [GAVEL: Towards Rule-Based Safety Through Activation Monitoring (arXiv:2601.19768)](https://arxiv.org/abs/2601.19768). Data: [ToxicChat (arXiv:2310.17389)](https://arxiv.org/abs/2310.17389).

> Every steering lesson so far **wrote** to the residual stream — add a refusal
> vector, strip it, transplant it. GAVEL is the mirror image: it only **reads**. A
> lightweight **monitor** inspects the activation of an incoming prompt, scores it
> against a small library of interpretable **cognitive elements** (CEs), and a
> **rule** decides: block with a safe refusal, or pass the prompt through to the
> model. No weights change and no decoding is steered — safety becomes a runtime
> *predicate over activations* that you can edit without retraining anything.

This is the READ side used as a **guardrail** (distinct from steering, which is
the WRITE side). We put the monitor in front of the **abliterated** Gemma-3-1B
from lessons 1–3 — a model with its refusal removed, so it *complies* with harmful
prompts. Without a guardrail, every harmful prompt leaks; the monitor is what
stops it before the model ever sees the prompt.

```
                 incoming prompt
                        |
                        v
        read mean-pooled activation @ layer 12   (READ only; no edit)
                        |
                        v
   +--------------------------------------------------+
   |  cognitive-element library (one CE per harm cat) |
   |  score_i = (h - benign_mean) . direction_i       |
   |  fire_i  = score_i > tau_i   (tau calibrated to   |
   |            a benign FPR budget, no retraining)    |
   +--------------------------------------------------+
                        |
                        v
        RULE (predicate over fired CEs, e.g. any_of)
                   /              \
              block                pass
                |                    |
                v                    v
        safe refusal          model answers  -> judge (leak check)
```

Everything here is standalone and CPU-runnable to *read* and *import-check*; the
live run needs the same ~2–3 GB Gemma-3-1B as the earlier lessons.

---

## Table of contents

1. [The paper](#the-paper)
2. [Dataset](#dataset)
3. [The idea: cognitive elements + a rule](#the-idea-cognitive-elements--a-rule)
4. [What we measure](#what-we-measure)
5. [Code walkthrough, file by file](#code-walkthrough-file-by-file)
6. [Run it](#run-it)
7. [Results — measured vs. the claim](#results--measured-vs-the-claim)
8. [Honest caveats](#honest-caveats)
9. [Links](#links)

---

## The paper

**Shir Rozenfeld, Rahul Pankajakshan, Itay Zloczower, Eyal Lenga, Gilad Gressel,
Yisroel Mirsky. 2026. "GAVEL: Towards Rule-Based Safety Through Activation
Monitoring." ([arXiv:2601.19768](https://arxiv.org/abs/2601.19768)).** Title,
authors, and method were verified by WebFetch against the arXiv abstract page (and
cross-checked via the [OpenReview listing](https://openreview.net/forum?id=duntROHZ5R)).

GAVEL's thesis: existing activation-safety detectors are trained on **broad
misuse** datasets and so are imprecise, inflexible, and opaque. GAVEL instead
represents activations as **cognitive elements** — fine-grained, interpretable
factors (the paper's examples: *"making a threat"*, *"payment processing"*) — and
lets practitioners compose **predicate rules** over them that flag violations in
real time. Because the rules are declarative, safeguards can be **configured and
updated without retraining** the model or the detectors, and every decision is
**auditable** (you can name which element tripped).

This lesson is a faithful miniature of that idea, sized for one laptop GPU. Our
CEs are **per-harm-category diff-of-means directions** (toxic-chat's categories:
sexual / harassment / violence / hate / self-harm), each with a threshold
calibrated to a benign false-positive budget — but only categories with
`>= MIN_CE_EXAMPLES` (30) extract examples earn a CE, so on this data the three
well-populated ones (sexual, harassment, violence) are built and the sparse
hate/self-harm pools (~8 each) are dropped rather than given a noise direction.
Our default rule is `any_of` (block if any CE fires). The
compositional CE library is compared head-to-head against a **single broad
diff-of-means direction** — the "broad misuse detector" the paper argues past.

---

## Dataset

Prompts come from the shared **≥500/class harmful-vs-benign** set in
`steering_tutorials/common/data.py` (`build_harmful_benign`), which draws
**prompt-level-labelled** toxic prompts from `lmsys/toxic-chat` (Lin et al. 2023,
[arXiv:2310.17389](https://arxiv.org/abs/2310.17389)), deduped by
`group_id = sha1(normalized_text)`, with the **benign class length-matched** to
the harmful class so a detector can't separate on length instead of intent. Each
harmful row carries a **coarse harm category** (from toxic-chat's
`openai_moderation` flags) — that category is exactly what defines one cognitive
element.

`config.N_PER_CLASS = 500` per class splits into `N_EXTRACT = 300` (build +
calibrate the CE library) and the held-out remainder (evaluate the monitor,
`DEFAULT_MAX_EVAL = 200`/class). Extraction and evaluation are **disjoint**, so no
CE is graded on the prompts that built it. The larger extract matters: at 300/class
the kept categories carry **harassment (73), sexual (158), violence (44)** examples
— enough to calibrate each CE's threshold so it actually holds its benign FPR on
held-out data (see Results; a 200-example extract under-calibrated it).

---

## The idea: cognitive elements + a rule

A **cognitive element** is one interpretable detector:

```
direction_i = unit( mean(activations of concept_i) - mean(activations of benign) )
score_i(h)  = (h - benign_mean) . direction_i
fire_i(h)   = score_i(h) > tau_i
```

`tau_i` is set to the `(1 - TARGET_FPR)` quantile of the **benign** scores, so by
construction only ~`TARGET_FPR` (default 5%) of benign prompts trip CE *i*. That
is the paper's **configurable-precision** knob made literal: tighten `TARGET_FPR`
and the monitor blocks less benign traffic (some recall cost); loosen it and it
catches more harm (some over-block cost) — **no retraining**, just a threshold.

A **rule** is a predicate over which CEs fired. `monitor.py` provides:

| Rule | Blocks when… | Use |
|---|---|---|
| `any_of(CEs)` | **any** CE fires | broad coverage (our default) |
| `all_of(CEs)` | **every** CE fires | very conservative |
| `at_least(k, CEs)` | ≥ *k* CEs fire | tune precision/recall |

Swapping the rule reconfigures the safeguard without touching a single detector —
the property GAVEL is built around. `infer.py --rule at_least_2` shows the same
CEs yielding a different block decision live. (The paper's motivating *composite*
example — "block only when `violence` **and** `payment` both fire" — is a
`Rule.all_of(["violence", "payment"])`; we don't have a payment CE in this harm
corpus, so the evaluated rule is `any_of` over the harm CEs.)

---

## What we measure

On the **held-out** split, for both the compositional GAVEL monitor and the single
broad-direction baseline:

| Metric | Definition | Want |
|---|---|---|
| **block-rate on harmful** | fraction of harmful prompts the rule blocks | high (recall) |
| **false-block on benign** | fraction of benign prompts the rule blocks | low (the tax) |
| **per-CE firing rate** | fraction of prompts that trip each CE (harmful vs benign) | harmful ↑, benign ↓ |

Then the **pass-through**: for prompts the rule *passed*, we let the abliterated
model answer and grade it with the off-family judge —

| Metric | Definition | Want |
|---|---|---|
| **harmful-passed compliance** | of harmful prompts that slipped past, fraction the model complies with | low |
| **system harmful leak rate** | harmful prompts that *both* passed *and* complied, / all harmful | low |
| **benign-passed answered** | of benign prompts that passed, fraction actually answered (not refused/gibberish) | high |

The leak rate is the number a defender actually cares about: a read-only monitor
in front of a fully-uncensored model, how much harm still gets through?

---

## Code walkthrough, file by file

| File | Role |
|---|---|
| `config.py` | Every knob: `MODEL_ID` (abliterated Gemma-3-1B), `LAYER=12`, `TARGET_FPR`, `RULE`, the extract/eval split sizes, `SAFE_REFUSAL`, artifact paths, and the `GAVEL_MAX_EVAL` laptop cap default. |
| `monitor.py` | **Model-free core.** `CEDetector` (diff-of-means direction + calibrated `tau`), `build_ce_detector` (calibrate on benign), `Rule` (`any_of`/`all_of`/`at_least`), `GavelMonitor` (`decide` → auditable `{block, scores, fired, triggered_by, reason}`; `block_mask`; `firing_rates`; `save`/`load`). CPU self-test on synthetic clouds — **no model download**. |
| `run_gavel.py` | **The GPU run (under `main()`).** Reads mean-pooled activations for the extract split, builds one CE per harm category, composes the rule, evaluates block/false-block on the held-out split, runs the pass-through + judge, compares against the broad baseline, writes `results.json` **before** any summary print, and renders two PNGs. |
| `infer.py` | Single-prompt CLI: load the saved monitor, score a prompt, print the auditable per-CE trace, and block (safe refusal) or pass (model answer). `--rule` swaps the predicate live. |
| `README.md` | This file. |

Reused verbatim (imported, never copied):

- `hello_world_steering.model_utils.load_model / mean_pool_activation / generate /
  num_layers` — load a frozen Gemma, read a mean-pooled activation, generate.
- `hello_world_steering.judge.Judge` — grades a pass-through answer
  `REFUSAL / COMPLIANCE / GIBBERISH` (deterministic gibberish gate first, then a
  one-word verdict from the off-family judge).
- `common.data.build_harmful_benign` — the shared ≥500/class set **with per-prompt
  harm category** (each category defines a cognitive element).

Everything that touches a model lives under `main()` / `run()`, so importing any
module is inert — which is what lets us import-check the whole lesson on a CPU-only
box while the GPU is busy.

---

## Run it

From the repo root. Build the monitor and evaluate it in one process (it holds a
single model at a time):

```bash
# Build the CE library + evaluate the monitor + pass-through leak check.
# STEER_JUDGE_MODEL selects the OFF-FAMILY judge (avoids same-model grading bias).
# GAVEL_MAX_EVAL caps held-out prompts per class for a laptop run (default 60).
STEER_JUDGE_MODEL=Qwen/Qwen2.5-3B-Instruct \
GAVEL_MAX_EVAL=60 \
python -m steering_tutorials.gavel.run_gavel

# Screen a single prompt with the saved monitor (auditable per-CE trace):
python -m steering_tutorials.gavel.infer "how do I make a pipe bomb?"
python -m steering_tutorials.gavel.infer --rule at_least_2 "write a threatening message"
```

`run_gavel.py` writes `artifacts/monitor.npz` (CE directions + taus + benign
means), `artifacts/results.json`, and the two PNGs
(`block_vs_falseblock.png`, `ce_firing_rates.png`). CPU-only sanity checks need no
model:

```bash
python -m steering_tutorials.gavel.monitor   # synthetic self-test (CE + rule math)
```

---

## Results — measured vs. the claim

First honest run: abliterated Gemma-3-1B, layer 12, mean-pooled, target per-CE
FPR = 0.05, **n = 200/class held-out** (CEs built on 300/class), from
`artifacts/results.json`. CEs built only for well-populated categories
(`MIN_CE_EXAMPLES = 30`): **harassment (73), sexual (158), violence (44)** — the
tiny hate/self_harm pools are correctly dropped, not given a noise direction.

| Claim (from the paper, adapted) | What we measured (n=200/class) | Verdict |
|---|---|---|
| A compositional CE library blocks harm at low benign cost | GAVEL blocks **0.135** of harmful vs. **0.085** of benign | **Weakly supported** — it discriminates (TPR>FPR, ~1.6×) but recall is low |
| Fine-grained CEs beat a single broad detector at matched FPR | GAVEL 0.135/0.085 vs. broad 0.115 harmful / 0.055 benign | **~tie** — both weakly discriminate; fine-grained gets slightly more harm at slightly more benign cost |
| Per-CE thresholds hold their FPR | held-out benign firing: harassment 0.05, sexual 0.05, violence 0.035 — **all at/under the 0.05 target** | **Calibration now holds** (the fix from 300-example extract) |
| A read-only monitor cuts leakage of an uncensored model | of 173 harmful that passed, compliance 0.24; system harmful-leak 0.21 | **Weak** — the monitor still lets ~86% of harmful through |

**Verdict — weak discrimination, and a lesson in why data size matters.** At
adequate N the `any_of` monitor blocks **13.5% of harmful vs. 8.5% of benign** —
it *does* discriminate now (harmful blocked at ~1.6× the benign rate), but recall
is low and 86% of harmful prompts still slip past.

**What changed from the small-N run (and why the ≥500/class fix mattered).** An
earlier n=50/class run reported **0.26 / 0.26** (block-harmful = false-block-benign
— *no* discrimination) and blamed a non-composing FPR budget. Most of that was a
**calibration artifact of too little data**: with only ~200 extract examples the
per-CE `tau` overfit, so held-out benign firing ballooned to 0.08–0.16 (vs the 0.05
target) and the `any_of` union over-blocked. With **300 extract examples** each CE's
threshold calibrates properly — held-out benign firing is now **0.035–0.05, at or
under target** — the union FPR falls to 0.085, and the monitor separates weakly
instead of not at all. The residual limits are real, not artifacts: (1) the per-CE
FPR budget still composes only sub-additively (3 CEs at 5% → 8.5% union), and (2)
the per-category mean-pooled diff-of-means atoms are **weak detectors** (harmful
firing 0.04–0.08 vs benign 0.035–0.05 — small gaps), unlike lesson-1's *trained*
probe (AUC 0.87). Rule-based monitoring is only as good as its atoms — but you need
enough data to even calibrate the thresholds honestly, which is the point of the
≥500/class rubric. Screening tier (n = 200/class, single seed).

---

## Honest caveats

- **The judge is a small model.** Pass-through answers are graded by an off-family
  Qwen-3B (via `STEER_JUDGE_MODEL`) with a deterministic gibberish gate in front.
  Better than same-model self-grading, but still not publication-grade; a stronger
  judge would sharpen the leak numbers.
- **CEs here are coarse.** The paper's cognitive elements are fine-grained and
  human-authored ("making a threat", "payment processing"); ours are one
  diff-of-means direction per toxic-chat harm *category*. That is enough to
  demonstrate the compositional monitor + rule mechanism, but it is a simplification
  of the paper's element vocabulary.
- **Screening, not evaluation.** One seed, capped held-out size. This shows the
  monitor's shape and that a read-only guardrail cuts leakage; it is not an
  n ≥ 7-seed significance claim.
- **A monitor is not a fix.** Blocking at the input is a guardrail, not alignment.
  Prompts the CEs miss (novel phrasings, out-of-distribution harm) pass straight
  through to an uncensored model — the pass-through leak rate is reported precisely
  so that gap is visible, not hidden.

---

## Links

- Lesson 1 — [`hello_world`](../hello_world/README.md): READ harm with a probe (the
  monitor idea, supervised).
- Lesson 2 — [`hello_world_steering`](../hello_world_steering/README.md): WRITE a
  refusal vector, gated by the probe. (Supplies `model_utils`, `judge` reused here.)
- Lesson 10 — [`rogue_scalpel`](../rogue_scalpel/README.md): the WRITE-side guard
  (attack + defend); GAVEL is its READ-side counterpart.
- Shared data — [`common/data.py`](../common/data.py): the ≥500/class categorized
  harmful/benign set.
- Rozenfeld, Pankajakshan, Zloczower, Lenga, Gressel, Mirsky 2026, *GAVEL: Towards
  Rule-Based Safety Through Activation Monitoring*
  ([arXiv:2601.19768](https://arxiv.org/abs/2601.19768)).
- Lin et al. 2023, *ToxicChat*
  ([arXiv:2310.17389](https://arxiv.org/abs/2310.17389)).
