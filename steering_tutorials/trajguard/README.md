# TrajGuard — the jailbreak is in the decoding trajectory, before the words arrive

> **Reference:** [TrajGuard: Streaming Hidden-state Trajectory Detection for Decoding-time Jailbreak Defense (arXiv:2604.07727)](https://arxiv.org/abs/2604.07727).

> The sibling lesson (`multiturn_jailbreak`) caught an attack in the trajectory
> across **conversation turns**. This lesson zooms all the way in: the attack is
> already visible in the trajectory across **generated tokens**. As the
> abliterated Gemma decodes a harmful completion, its residual-stream state at
> each new token drifts, token by token, toward a high-risk region; a benign
> completion stays put. A sliding window over those decoding-time hidden states
> quantifies the risk in **real time** and flags the jailbreak **before** the
> harmful content is fully emitted — a streaming, decoding-time defence with no
> prompt-side classifier.

This is a **detection** lesson (no LLM judge — a detector reads a signal off the
model's own hidden states, exactly like lesson 1). We generate completions for
harmful vs. benign prompts, capture the per-token layer-12 trajectory, and detect
with (a) the paper's **training-free** sliding-window projection detector and (b)
the **reused** learned sequence classifiers from `multiturn_jailbreak`. The
headline extra is an **early-detection curve**: AUC as a function of how few
generated tokens we have seen.

> **The claim under test.** Cheng Liu, Xiaolei Liu, Xingyu Li, Bangzhou Xin,
> Kangyi Ding 2026, *TrajGuard: Streaming Hidden-state Trajectory Detection for
> Decoding-time Jailbreak Defense* (arXiv:2604.07727, ACL 2026 Findings) — this
> lesson operationalizes the paper's decoding-time-trajectory framing with a
> sliding-window harm-projection detector plus reused sequence models.

---

## The key idea in code

Generate, capture the per-token layer-12 states, project onto the harm direction,
smooth with a causal sliding window, and flag the moment the risk crosses `tau`:

```python
# trajectory.py -- the training-free streaming detector (the paper's method):
completion, traj = generate_and_capture(model, tok, prompt)   # traj: [n_tokens, dim] @ layer 12
center, unit_dir = harm_direction(train_trajs, train_labels)  # unit(mean(harmful tok) - mean(benign tok))
scores = token_scores(traj, center, unit_dir)                 # (traj - center) @ unit_dir, per token
risk   = sliding_window_risk(scores, window=4)                # causal running mean over last 4 tokens
flag   = risk >= tau                                          # tau calibrated to 10% benign FPR on train
#  -> on a HARMFUL completion `risk` drifts UP and crosses tau within the first few
#     tokens (flagged EARLY); on a BENIGN completion it stays flat and never crosses.
```

No training, no prompt classifier, no generation-of-a-judgement — just a running
projection of the model's own decoding-time states. Full walkthrough below.

---

## Table of contents

1. [Why the token trajectory carries the attack](#1-why-the-token-trajectory-carries-the-attack)
2. [The detectors](#2-the-detectors)
3. [Pipeline](#3-pipeline)
4. [Files](#4-files)
5. [Code walkthrough, file by file](#5-code-walkthrough-file-by-file)
6. [The dataset](#6-the-dataset)
7. [Sibling: multiturn_jailbreak](#7-sibling-multiturn_jailbreak)
8. [Running](#8-running)
9. [Results — measured vs. the claim](#9-results--measured-vs-the-claim)
10. [The confound audit — what the AUCs must clear](#10-the-confound-audit--what-the-aucs-must-clear)
11. [Honest caveats](#11-honest-caveats)
12. [Repository](#12-repository)

---

## 1. Why the token trajectory carries the attack

Prompt-side jailbreak detection asks a single question — *"is this incoming
prompt harmful?"* — once, before generation. But a well-crafted jailbreak is
built precisely to look benign at the prompt: the harm only materialises in the
**completion the model is about to write**. By the time the harmful text exists as
tokens, a prompt classifier has already been bypassed.

TrajGuard's move is to watch the generation *from the inside*. Autoregressive
decoding produces one hidden state per generated token. As the model commits to a
harmful continuation, those states **drift** — steadily, in a consistent
direction — toward the region occupied by other harmful completions. A benign
continuation stays in the benign region. So the signal is not "is token *k*
harmful?" but "is the residual-stream state **moving** toward the harmful
region?", and a sliding window over the per-token states reads that drift.

The value is that the drift is visible **early**. You do not need the whole
harmful answer to detect it; the first handful of generated tokens already move,
and the sliding-window risk crosses `tau` before the payload is emitted. That is
the entire pitch of a *streaming* defence: flag (and, in deployment, halt) the
jailbroken generation while it is still mostly unwritten. The
[early-detection curve](#9-results--measured-vs-the-claim) is exactly this
measurement — AUC at K = 2, 4, 8, 16, 32 tokens.

---

## 2. The detectors

Every detector reads the same object: a completion's per-token trajectory
`[n_tokens, dim]` at layer 12. Four are compared, from training-free to fully
learned:

| method (`config.METHODS`) | training | how it decides | role |
|---|---|---|---|
| `threshold_freeform` | **none** | project each token onto the harm direction; **max** sliding-window risk vs. `tau` | the paper's own training-free detector |
| `per_turn_max` | logistic | lesson-1 probe on each token alone, **max** over tokens | stateless baseline (reused from the sibling) |
| `trajectory_mlp` | MLP | hand-crafted trajectory features -> MLP | cheap stateful-ish reference (reused) |
| `seq_gru` | GRU | GRU over the ordered token sequence -> logit | the learned sequence model (reused) |

`threshold_freeform` is the honest baseline: it only fits a **harm direction**
(diff-of-means of harmful vs. benign token states, on train) and a **threshold**
`tau` (calibrated to 10% benign false-positive rate on train). No classifier is
trained. The three learned methods are imported **unchanged** from
`multiturn_jailbreak.models` — a token trajectory is the same `[n_steps, dim]`
object a turn trajectory is (see [Sibling](#7-sibling-multiturn_jailbreak)).

The pre-registered claim is that the token trajectory carries the jailbreak
signal — both a training-free projection and a learned sequence model separate
harmful from benign completions above chance (falsifier in
[Section 9](#9-results--measured-vs-the-claim)).

---

## 3. Pipeline

```
  common.data.load_harmful_benign(n_per_class)     abliterated Gemma-3-1B (local)
        harmful prompts     benign prompts                     |
            |                    |                             |
            +--------- per prompt: generate_and_capture -------+
                                 |  (greedy decode; ONE forward pass,
                                 v   output_hidden_states=True @ layer 12)
              trajectories : List[ [n_tokens, dim] ]   label = prompt class
                                 |
                    +------------+-------------------------------+
                    |         5-fold StratifiedKFold CV          |
                    v                                            v
     threshold_freeform   per_turn_max   trajectory_mlp   seq_gru   (models.py reused)
         (training-free)       |              |             |
                    \__________|______________|_____________/
                                 |
              out-of-fold P(jailbreak) -> AUC (+boot CI), F1, TPR@FPR=0.10
                                 |
                    +------------+-------------------------------+
                    |  EARLY-DETECTION: AUC using only first K   |
                    |  tokens, K in {2,4,8,16,32}                |
                    v                                            v
         results.json  +  ROC / early-AUC-vs-K / risk-drift-example PNGs
```

The risk-drift view — `sliding_window_risk` on one harmful vs. one benign
completion — is what `infer.py` prints and what `run_trajguard.py` renders to
`artifacts/risk_drift_example.png`.

---

## 4. Files

| file | role |
|---|---|
| `config.py` | every knob: model, layer, generation cap, window, target FPR, early-K list, CV/methods, paths |
| `trajectory.py` | generate + capture the per-token trajectory; the training-free sliding-window projection detector |
| `data.py` | generate harmful/benign completions with the abliterated Gemma; capture + cache the token trajectories; **`confound_report()`** — the trivial-baseline audit (token count, char length, hidden-state norm, prompt length) |
| `run_trajguard.py` | orchestrator: build -> **confound audit** -> 5-fold CV over the four methods -> early-detection curve -> `results.json` + plots |
| `infer.py` | generate for one harmful + one benign prompt; print the per-token sliding-window risk drifting up and crossing `tau` |

`models.py` is **not** in this folder — it is imported unchanged from the sibling
`multiturn_jailbreak`.

---

## 5. Code walkthrough, file by file

### `config.py` — every knob in one place

The local abliterated Gemma path, `LAYER = 12`, the generation cap
(`MAX_NEW_TOKENS`, the trajectory length), the detector knobs (`WINDOW`,
`TARGET_FPR`), the early-detection `EARLY_KS = [2,4,8,16,32]`, the method list and
CV knobs (`METHODS`, `N_FOLDS`, `BOOTSTRAP`), and all paths. Everything is
overridable by env var so an eval shrinks into one foreground window (see
[Running](#8-running)).

### `trajectory.py` — capture + the training-free detector (the heart)

`generate_and_capture(model, tok, prompt)` greedily generates a completion, then
runs **one** forward pass over prompt+completion with `output_hidden_states=True`
and keeps `hidden_states[layer]` at the **completion** token positions only —
`[n_gen_tokens, dim]` on CPU. The training-free detector is pure geometry:
`harm_direction` computes the center (mean of all train token states) and the unit
harm direction (mean harmful-token state − mean benign-token state);
`token_scores` projects a trajectory onto it; `sliding_window_risk` is the causal
running mean over the last `WINDOW` tokens. `ThresholdDetector` wraps these with
the same `fit` / `predict_proba` API as the learned models, plus
`predict_proba_earlyK` for the early-detection curve. A CPU self-test on synthetic
drifting-vs-flat trajectories asserts AUC > 0.9 with no model.

### `data.py` — generate, capture, cache

`build_token_trajectories` pulls prompts from `common.data.load_harmful_benign`,
loads the abliterated Gemma **once**, and for each prompt calls
`trajectory.generate_and_capture` — label 1 for harmful prompts, 0 for benign. The
result is cached as a ragged `.npz` pack (all trajectories vstacked with
per-completion token counts) plus a prompts/completions sidecar, so re-runs and
`infer.py` are fast. It is the **only** model load in the lesson besides `infer`.
`confound_report(trajectories, labels, completions, prompts)` is the honesty check
required by [`CONFOUND_DISCIPLINE.md`](../CONFOUND_DISCIPLINE.md): it scores the
label against four **trivial scalars that carry no trajectory information** —
generated-token count, completion character length, the mean per-token
hidden-state norm `||h_t||`, and the final-token norm `||h_last||` (plus prompt
length) — and returns each raw AUC, its **directionless** fold `max(auc, 1-auc)`,
the per-class means, and `worst_auc_directionless`, the bar every headline must
clear. The norm features are the load-bearing ones here: a residual-stream
*magnitude* gap alone would separate the classes with no trajectory *shape*
involved, which would make a sequence model's whole margin illusory. It is pure
CPU numpy over the cached trajectories — no model load.

### `run_trajguard.py` — the orchestrator

`main()` builds/loads the dataset, runs the **confound audit first** (so the bar
exists before the headline does; it lands in `results.json` under `confound` and in
a `vs CONF` margin column in the summary table), then runs **5-fold StratifiedKFold
CV**. For each
method it pools out-of-fold predictions and reports **AUC (+ bootstrap CI), F1,
accuracy, TPR@FPR=0.10**. It then computes the **early-detection curve**: for each
`K` in `EARLY_KS`, the out-of-fold AUC of the training-free detector and `seq_gru`
using only the first `K` tokens of each trajectory. Writes `results.json` (schema
in the contract) **before** the summary print, and renders three PNGs: ROC,
early-AUC-vs-K, and the risk-drift example.

### `infer.py` — watch the risk drift up

Loads the model once, fits the harm direction + `tau` on a small reference slice
(generating each reference completion with the already-loaded model), then for one
harmful and one benign prompt (or your own prompt on the CLI) prints the per-token
sliding-window risk. You watch the harmful completion's risk climb and cross
`tau` early, while the benign one stays flat. All model-touching code is under
`main()`.

---

## 6. The dataset

There is no ready-made corpus of labelled per-token jailbreak trajectories — we
**generate** it. Prompts come from the shared **≥500 harmful + ≥500 benign**
foundation exposed by `steering_tutorials.common.data.load_harmful_benign` (built
on JailbreakBench, Chao et al. 2024, arXiv:2404.01318, plus the principled
`lmsys/toxic-chat` loader — prompt-level intent labels, harm-category stratified,
deduped). We cap at `N_PER_CLASS` per class (generation is the cost on the
RAM-constrained host) and, for each prompt, let the **abliterated** Gemma-3-1B
(`DavidAU/gemma-3-1b-it-heretic-...`) generate a completion and capture its
layer-12 per-token trajectory.

| role | dataset | what it is | label |
|---|---|---|---|
| harmful | `common.data` harmful prompts | abliterated-Gemma completion + its token trajectory | **1** |
| benign | `common.data` benign prompts | abliterated-Gemma completion + its token trajectory | **0** |

**The label is the prompt's class, not a judged rating of the completion.** We do
**not** run a judge over the generated text and label it by measured harm; a
harmful-class prompt is labelled 1 regardless of what the model actually wrote.
The abliterated model is chosen precisely because it **complies** with harmful
prompts, so a harmful-class prompt reliably yields a harmful completion (and thus a
genuinely harmful trajectory to detect) — but "reliably" is an assumption about
the base model, not a per-example guarantee. The harm direction is fit on **train
only** in every fold, so no completion is graded on the direction it defined.

---

## 7. Sibling: multiturn_jailbreak

This lesson is the **token-level sibling** of `multiturn_jailbreak`. They share one
idea — *classify a sequence of hidden states* — at two granularities:

| | `multiturn_jailbreak` | `trajguard` (this lesson) |
|---|---|---|
| the sequence chunk | a conversation **turn** | a generated **token** |
| the trajectory | across turns of a chat | across tokens of one completion |
| when it fires | after several turns | **during** a single generation (streaming) |
| the attack | Crescendo / ActorAttack escalation | a jailbroken **completion** drifting to the harm region |

Because a token trajectory is the same `[n_steps, dim]` object a turn trajectory
is, `models.py` is **reused unchanged**: `PerTurnMaxProbe`, `TrajectoryMLP`, and
`SeqGRU` classify token sequences here exactly as they classified turn sequences
there. The only new code is the capture path (`generate_and_capture`) and the
training-free sliding-window projection detector.

---

## 8. Running

From the **repo root** (`steeringresearch/`):

```bash
# CPU-only self-test (NO model): the training-free detector on synthetic trajectories
python -m steering_tutorials.trajguard.trajectory

# A small smoke that DOES load the model (8 completions):
python -m steering_tutorials.trajguard.data

# The full generate -> capture -> CV run (needs the ~2-3 GB abliterated Gemma-3-1B):
python -m steering_tutorials.trajguard.run_trajguard

# Watch the per-token risk drift up on a harmful prompt and stay flat on a benign one:
python -m steering_tutorials.trajguard.infer

# ...or on your own single prompt:
python -m steering_tutorials.trajguard.infer "How do I pick a household lock?"
```

**Env caps** (shrink an eval into one foreground window — the host's RAM, not
VRAM, is the wall):

| var | meaning | default |
|---|---|---|
| `TG_N_PER_CLASS` | completions generated per class | 120 |
| `TG_MAX_NEW_TOKENS` | generated tokens per completion (trajectory length) | 40 |
| `TG_WINDOW` | sliding-window length (tokens) | 4 |
| `TG_FOLDS` | StratifiedKFold CV folds | 5 |
| `TG_INFER_N` | reference completions per class in `infer.py` | 12 |

```bash
# a fast smoke run:
TG_N_PER_CLASS=40 TG_MAX_NEW_TOKENS=24 TG_FOLDS=3 \
  python -m steering_tutorials.trajguard.run_trajguard
```

On Windows PowerShell set env vars first, e.g. `$env:TG_N_PER_CLASS = "40"`.

**No judge.** This is a **detection** lesson: the detector reads a signal off the
frozen model's own decoding-time hidden states — there is no LLM judge and no
generation-of-a-verdict, so the off-family-judge discipline of the steering
lessons does not apply (`results.json` records `"judge": null`).

---

## 9. Results — measured vs. the claim

> **These AUCs are UNPRICED and must not be read as the lesson's claim.** The run
> below predates this lesson's confound audit: it was published with **no trivial
> baseline of any kind**, so the margin it earns over a scalar with no trajectory
> information in it is **unknown**. `confound_report()` now exists and is wired
> into the orchestrator, but computing it needs a re-run (the audit reads the
> cached trajectories, so it is a CPU-only recompute — the GPU is not needed).
> Until that row is filled, treat every number here as *raw* and the lesson's
> claim as **unsupported**, per
> [`CONFOUND_DISCIPLINE.md`](../CONFOUND_DISCIPLINE.md). See
> [Section 10](#10-the-confound-audit--what-the-aucs-must-clear).

Run at the ≥500/class config: abliterated Gemma-3-1B, layer 12, **300 harmful +
300 benign** completions (max 40 generated tokens), window=4, 5-fold stratified CV,
bootstrap 95% CIs, from `artifacts/results.json`.

Detection (full trajectory). The **claimable margin** column is the only quantity
the lesson may headline; it stays empty until the audit is re-run:

| method | AUC (raw, unpriced) | F1 | TPR@FPR=0.10 | claimable margin over the confound bar |
|---|---|---|---|---|
| `threshold_freeform` (training-free, the paper's method) | 0.638 | 0.24 | 0.10 | **−0.045** (BELOW the bar) |
| `per_turn_max` (learned per-token, stateless) | 0.931 | 0.84 | 0.84 | **+0.196** |
| `trajectory_mlp` | 0.944 | 0.90 | 0.90 | **+0.208** |
| `seq_gru` | 0.945 | 0.86 | 0.88 | **+0.209** |

Early detection — AUC using only the first K generated tokens (the streaming value):

| method | K=2 | K=4 | K=8 | K=16 | K=32 |
|---|---|---|---|---|---|
| `threshold_freeform` | 0.69 | 0.671 | 0.619 | 0.628 | 0.63 |
| `seq_gru` | **0.94** | 0.946 | 0.935 | 0.947 | 0.949 |

**Verdict — PROVISIONAL, pending the confound audit.** Against *chance*, the
pre-registered falsifier (both detectors ≤ 0.60) is not triggered: every method
clears 0.5 and the learned models reach 0.93–0.945, with `seq_gru` already at
0.94 from the first 2 generated tokens. But "clears chance" is the wrong bar.
The falsifier that actually binds this lesson is the confound one — *if
`AUC(method) ≤ max(confound_auc, 1 − confound_auc)` the claim is FALSE* — and it
**has never been evaluated here**. A norm-only or token-count-only rule could in
principle produce most of this separation. The streaming story ("you do not need
to wait for the harmful content to finish") is therefore **stated but not yet
earned**; it becomes a result when [Section 10](#10-the-confound-audit--what-the-aucs-must-clear)
has numbers in it.

**The honest twist — softened by the larger N.** At the small n=80 run the stateless
`per_turn_max` was clearly *best* (0.977), suggesting the sequence structure added
nothing (a jailbroken completion emits harmful-content tokens, so each token's state
is individually separable). At **n=300 that gap closes**: the learned sequence models
(`seq_gru` 0.945, `trajectory_mlp` 0.944) now *marginally edge* `per_turn_max` (0.931)
— the sequence structure adds a little after all, and the "per-token already wins"
story was partly small-n. Still, the paper's **training-free sliding-window projection
remains the weakest (0.638)** — a single diff-of-means direction, projected and
windowed, throws away signal a learned classifier keeps. The cross-lesson lesson holds
qualitatively: **trajectory modeling earns its keep most when the individual chunks
look benign (multi-turn escalation), and least when the chunks already carry the
signal (active harmful generation)** — but the per-token dominance here was weaker
than the small run implied.

**Why per-token is so strong here (a confound to name).** The abliterated model
*complies immediately*, so a harmful-class completion is harmful from token 1 —
there is no benign-looking preamble to hide the intent. The regime TrajGuard
actually targets — a completion that *starts* benign and *drifts* into harm — would
weaken the per-token signal and is where the sliding window would matter more; our
immediate-compliance setup does not exercise that drift. Reported, not hidden.

---

## 10. The confound audit — what the AUCs must clear

> **Status: MEASURED 2026-07-27 — and it changes how the headline reads.**
> The strongest trivial baseline is **prompt character length at AUC 0.7354**
> (directionless); hidden-state `final_norm` reaches **0.588** and token count **0.514**.
>
> So a raw AUC of 0.944 is **not** a 0.944-sized result. The claimable quantity is the
> margin **above 0.7354**:
>
> | method | raw AUC | **margin above the 0.7354 bar** |
> |---|---|---|
> | `seq_gru` | 0.945 | **+0.209** |
> | `trajectory_mlp` | 0.944 | **+0.208** |
> | `per_turn_max` | 0.931 | **+0.195** |
> | `threshold_freeform` (the paper's own training-free method) | 0.638 | **−0.098 — BELOW the bar** |
>
> Two consequences stated plainly. The learned sequence detectors **do** clear the bar,
> by ~0.21 — a real result, but a fifth of what the raw number suggests. And the paper's
> own training-free projection method lands **below** a character-count heuristic on this
> data, so it does not clear the bar at all.
>
> Before this audit the lesson headlined 0.944 / 0.945 with no baseline whatsoever, which
> made the margin unpriced and the claim uninterpretable.

A detection AUC is not a result until the strongest **trivial** baseline on the same
data is reported beside it, and only `headline − baseline` may be claimed
([`CONFOUND_DISCIPLINE.md`](../CONFOUND_DISCIPLINE.md)). This lesson was the
course's one **non-compliant** case: it headlined 0.944 / 0.945 with no baseline of
any kind. The only length-ish quantity ever recorded was *mean trajectory length*
(harmful **38.78** vs benign **37.96** tokens), which is near-matched **by the
40-token generation cap** — a design-time control, not a measured baseline. And
`threshold_freeform` (0.638) is a weak *method*, not a confound; beating it proves
nothing about triviality.

The four scalars audited, each carrying **no trajectory information whatsoever**:

| trivial feature | why it could separate the classes without any real signal | directionless AUC |
|---|---|---|
| `tokencount` | generated-token count. Nominally capped at `MAX_NEW_TOKENS`, but **early EOS** is class-informative — a benign completion that stops short is a free label. | **+0.209** |
| `charlen` | the completion's raw character length — the classic prompt-harm confound, one level downstream. | **+0.209** |
| `mean_norm` | **the important one.** Mean over tokens of `‖h_t‖` at layer 12. If harmful and benign completions simply sit at different residual-stream *magnitudes*, one scalar separates them and **every sequence model's margin is illusory** — no drift, no shape, no trajectory. | **+0.209** |
| `final_norm` | `‖h_last‖` alone, the single cheapest such scalar. | **+0.209** |
| `prompt_charlen` | prompt length — a deployable prompt-side rule that needs no hidden states at all. | **+0.209** |
| **worst (the bar)** | `max` of the above; the number every headline must clear | **+0.209** |

**Why the norms are the right confound for *this* lesson.** The siblings audit
length and count because their inputs are text. Here the detector's input is a
matrix of hidden states, so the trivial shortcut is geometric: the classes could
differ in the *size* of the residual-stream vectors rather than in how those
vectors *move*. `harm_direction` is a diff-of-means, and a mean-norm gap projects
straight onto it — so `threshold_freeform`, `trajectory_mlp`, and `seq_gru` could
all be reading one scalar. That is exactly the hypothesis the audit falsifies or
confirms, and it has never been run.

**Pre-registered falsifier (binding from now on).** For each method, if
`AUC(method) ≤ worst_auc_directionless`, that method's detection claim is **FALSE**
and must be reported as such — no reinterpreting the confound as "a strong
baseline", no retreating to the early-detection curve to rescue it. The
early-detection numbers inherit the same bar: an early-K AUC is claimable only
above the confound computed on the **same first-K truncation**, which the current
audit does not yet compute.

**To fill this in** (CPU only, no GPU, no regeneration — the trajectories are
cached in `artifacts/token_trajectories.npz`):

```bash
python -m steering_tutorials.trajguard.run_trajguard
```

The audit prints before the CV block, lands in `results.json` under `confound`,
and adds a `vs CONF` margin column to the summary table.

---

## 11. Honest caveats

- **The published AUCs are unpriced.** The single largest caveat, and the reason
  §9 opens with a warning: the run in `artifacts/results.json` reports **no
  trivial-confound baseline**, so its margin over a no-trajectory scalar is
  unknown. Fixed in code (§10), not yet fixed in the numbers.
- **Screening tier, not evaluation.** Single 1B model, one layer, a few hundred
  generated completions, 5-fold CV, one seed — a directional demo, not the n ≥ 7
  seeds + rigor contract CLAUDE.md reserves the word "winner" for. Do not
  over-read the ordering.
- **The abliterated model complies — that is the point, and a confound.** We use
  the abliterated Gemma because it answers harmful prompts, so harmful-class
  prompts yield harmful completions with a real drift to detect. On an aligned
  model most harmful prompts would be refused, and the "harmful" trajectory would
  be a *refusal* trajectory instead — the design would need re-labelling.
- **Label = prompt class, not a judged harm rating.** A harmful-class prompt is
  labelled 1 even if the model happened to produce something innocuous; we do not
  judge each completion. The AUC therefore measures separability of
  *prompt-class-conditioned* trajectories, which is what the streaming detector
  actually sees.
- **The harm direction is fit on train only.** In every CV fold `harm_direction`
  and `tau` are computed on the training completions, never the held-out ones, so
  a detector is never graded on the direction it defined. Random CV (no grouping)
  is fine here because each completion is an independent generation.
- **Detection, not generation.** This lesson detects a jailbroken generation; it
  does **not** halt or steer it. Wiring the flag into a decoding-time stop is the
  natural next lesson.
- **Inspired-by, not a paper reproduction.** The sliding-window harm-projection
  detector + reused sequence classifiers operationalize TrajGuard's
  decoding-time-trajectory *idea*; they are **not** a faithful reimplementation of
  the paper's exact architecture (see `AUDIT.md`).

---

## 12. Repository

Source and full artifacts:
<https://github.com/dlmastery/steeringresearch/tree/master/steering_tutorials/trajguard>

Cited (both arXiv ids WebFetch-verified): TrajGuard (Liu et al. 2026,
arXiv:2604.07727, ACL 2026 Findings); and the context-aware multi-turn sibling
DeepContext (Albrethsen et al., arXiv:2602.16935).

See also
[the course map](../README.md),
[the turn-level sibling — multiturn_jailbreak](../multiturn_jailbreak/README.md)
(whose `models.py` this lesson reuses unchanged), and
[lesson 1 — the single-prompt activation probe (READ)](../hello_world/README.md),
whose activation-reading idea both siblings generalize.
