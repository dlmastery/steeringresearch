# TrajGuard — the jailbreak is in the decoding trajectory, before the words arrive

> **Reference:** [TrajGuard: Streaming Hidden-state Trajectory Detection for Decoding-time Jailbreak Defense (arXiv:2604.07727)](https://arxiv.org/abs/2604.07727)
> — Cheng Liu, Xiaolei Liu, Xingyu Li, Bangzhou Xin, Kangyi Ding; ACL 2026 Findings,
> submitted 2026-04-09. *Relevance:* a training-free, streaming, decoding-time defence
> that aggregates hidden-state trajectories over a sliding window to quantify risk in
> real time. Its central claim is **comparative** — that hidden states during decoding
> carry **stronger** risk signals **than input prompts** — evaluated over 12 jailbreak
> attacks. This lesson is an *inspired-by* reconstruction, and §3 is about the fact that
> until 2026-08-08 it did not test that comparison at all.

> The sibling lesson (`multiturn_jailbreak`) caught an attack in the trajectory
> across **conversation turns**. This lesson zooms all the way in: the attack is
> already visible in the trajectory across **generated tokens**. As the
> abliterated Gemma decodes a harmful completion, its residual-stream state at
> each new token drifts, token by token, toward a high-risk region; a benign
> completion stays put. A sliding window over those decoding-time hidden states
> quantifies the risk in **real time** and flags the jailbreak **before** the
> harmful content is fully emitted.

This is a **detection** lesson (no LLM judge — a detector reads a signal off the
model's own hidden states, exactly like lesson 1; `results_*.json` records
`"judge": null`).

> ### Read §3 first.
> On 2026-08-08 this lesson **retracted its headline inference** and was **re-based on
> a different substrate**. Most of the value here is the correction, not a new number.

---

## Table of contents

1. [Why the token trajectory carries the attack](#1-why-the-token-trajectory-carries-the-attack)
2. [The detectors](#2-the-detectors)
3. [**RETRACTION, and the re-basing it forced**](#3-retraction-and-the-re-basing-it-forced)
4. [Pipeline](#4-pipeline)
5. [Files](#5-files)
6. [Code walkthrough, file by file](#6-code-walkthrough-file-by-file)
7. [The dataset — two substrates and an OOD arm](#7-the-dataset--two-substrates-and-an-ood-arm)
8. [Sibling: multiturn_jailbreak](#8-sibling-multiturn_jailbreak)
9. [Running](#9-running)
10. [Pre-registered falsifiers](#10-pre-registered-falsifiers)
11. [Results](#11-results)
12. [The confound audit — four bars and one rival](#12-the-confound-audit--four-bars-and-one-rival)
13. [Artifact discipline and reproduction](#13-artifact-discipline-and-reproduction)
14. [Honest caveats](#14-honest-caveats)
15. [Repository](#15-repository)

---

## 1. Why the token trajectory carries the attack

Prompt-side jailbreak detection asks a single question — *"is this incoming
prompt harmful?"* — once, before generation. A well-crafted jailbreak is built
precisely to look benign at the prompt: the harm only materialises in the
**completion the model is about to write**. By the time that text exists as
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
and the sliding-window risk crosses `tau` before the payload is emitted.

**That whole story has a precondition**, and §3 is about what happens when it is not
met: the completion must *start* somewhere other than where it *ends*. If the prompt is
overtly toxic and the model complies from token 1, there is no drift to read, and the
prompt-side classifier the argument dismisses is in fact the strongest thing available.

### The key idea in code

```python
# trajectory.py -- the training-free streaming detector (the paper's method):
completion, traj = generate_and_capture(model, tok, prompt)   # traj: [n_tokens, dim] @ layer 12
center, unit_dir = harm_direction(train_trajs, train_labels)  # unit(mean(harmful tok) - mean(benign tok))
scores = token_scores(traj, center, unit_dir)                 # (traj - center) @ unit_dir, per token
risk   = sliding_window_risk(scores, window=4)                # causal running mean over last 4 tokens
flag   = risk >= tau                                          # tau calibrated to 10% benign FPR on train
```

No training, no prompt classifier, no generation-of-a-judgement — just a running
projection of the model's own decoding-time states.

---

## 2. The detectors

Every detector reads the same object: a completion's per-token trajectory
`[n_tokens, dim]` at layer 12. Four are compared, from training-free to fully
learned:

| method (`config.METHODS`) | training | how it decides | role |
|---|---|---|---|
| `threshold_freeform` | **none** | project each token onto the harm direction; **max** sliding-window risk vs. `tau` | the paper's own training-free detector |
| `per_turn_max` | logistic | lesson-1 probe on each token alone, **max** over tokens | **stateless** baseline (reused from the sibling) |
| `trajectory_mlp` | MLP | hand-crafted trajectory features -> MLP | cheap stateful-ish reference (reused) |
| `seq_gru` | GRU | GRU over the ordered token sequence -> logit | the learned sequence model (reused) |

`per_turn_max` is the load-bearing control, not a makeweight: it sees **one token at a
time and no ordering**. If it matches the sequence models, there is no *trajectory*
signal on that substrate — only a per-token one — regardless of how high all the AUCs
are.

The three learned methods are imported **unchanged** from `multiturn_jailbreak.models`
— a token trajectory is the same `[n_steps, dim]` object a turn trajectory is (see
[§8](#8-sibling-multiturn_jailbreak)).

---

## 3. RETRACTION, and the re-basing it forced

### 3.1 The retracted claim

Until 2026-08-08 this README (§10) and `CLAUDE.md` §18.3 item 4 both asserted:

> ~~"prompt char-length is at chance, 0.5032 — **you cannot call this from the prompt
> alone**"~~

**Retracted. Both halves are false, and this lesson's own course proves it.**

**(a) 0.5032 was designed in, not measured.** The shared loader
[`common/data.py`](../common/data.py) draws the benign class **length-matched to the
harmful length histogram, decile-bin stratified** (`_length_matched_sample`, defined
line 246, invoked line 442; the header field `length_confound_note` at line 466 says so
in as many words). A prompt-length AUC near 0.50 is the sampler doing exactly what it
was written to do. It is **a control that was applied, not a discovery about the data**
— and a control cannot be cited as evidence for the thing it was applied to guarantee.

**(b) Length is not content.** That prompt *length* is uninformative says nothing about
whether prompt *text* is. It is. Three independent measurements, all on the same
toxic-chat pool:

| measurement | value | source |
|---|---|---|
| `hello_world`'s prompt-only probe | **AUC 0.965** | [`CONFOUND_DISCIPLINE.md`](../CONFOUND_DISCIPLINE.md) §3 |
| prompt-only bag-of-words, **overt** substrate, 500/class | **AUC 0.8779** | measured here, `python -m steering_tutorials.trajguard.data` |
| prompt-only bag-of-words, **disguised** substrate, 181/class | **AUC 0.9688** | same |

Compare that to what this lesson's *trajectory* detectors reached on the superseded
run: best **0.945**.

**The honest statement, which replaces the retracted one:**

> On toxic-chat, a prompt-side classifier **beats** the decoding-time detector. A
> bag-of-words model over the prompt alone reaches 0.878–0.969 depending on substrate,
> and `hello_world`'s prompt probe reaches 0.965; the best trajectory detector here
> reached 0.945. The paper's comparative claim — decoding-time hidden states carry
> *stronger* risk signals than input prompts — is **not supported on this dataset**,
> and this lesson never tested it, because it built no prompt-side classifier at all.

### 3.2 There is no trajectory signal on this substrate, only a per-token one

This is not speculation either — the superseded run's own numbers say it:

- `threshold_freeform`, the **drift** detector, the paper's own method, scored
  **0.638**, which is **below** the completion-character-length confound bar of
  **0.735**. The detector that reads movement lost to counting characters.
- The **stateless** per-token probe, which sees no sequence at all, scored **0.931** —
  essentially level with `trajectory_mlp` (0.944) and `seq_gru` (0.945).

Read together: the classes are separable **token by token**, and modelling the sequence
adds ~0.014. That is what "no trajectory signal, only a per-token one" means, stated as
the lesson's finding rather than buried as a caveat.

**Not "further work needed".** The claim was wrong, the substrate was wrong, and both
are now stated as such in every surface.

### 3.3 The mechanism, and why the substrate was the problem

TrajGuard needs prompts whose **surface is benign** and whose **completion turns
harmful** — that gap is the entire reason a decoding-time monitor could beat a
prompt-side one. toxic-chat's `toxicity==1` prompts are **overtly toxic user inputs**
from a live Vicuna demo. The abliterated model complies from token 1, so:

- the drift never happens (hence `threshold_freeform` below a character count), and
- the intent is on the prompt surface (hence the prompt-side classifier winning).

The paper evaluates on **12 jailbreak attacks**. The superseded lesson used **zero**
attack wrappers. There was no jailbreak in the jailbreak-detection lesson.

### 3.4 The re-basing — and the substrate was already in the pool

toxic-chat 0124 ships **two** human prompt-level binary labels: `toxicity` **and
`jailbreaking`**. `common/data.py` reads only `toxicity`, so the disguised-attack
subset was being discarded at load time.

`common/` is shared and this lesson does not edit it. `trajguard/data.py` instead reads
the same CSV through the same download path and applies `common/data.py`'s **own**
primitives (`_clean` / `_norm_key` / `_group_id` / `_stratified_sample` /
`_length_matched_sample`, `MIN_CHARS` / `MAX_CHARS`), so dedup, grouping,
ambiguity-dropping and length-matching are bit-identical to every other lesson — only
the label expression differs. The lesson now ships **two substrates and teaches the
contrast**:

| `TG_SUBSTRATE` | positive class | unique available | rule 1 (≥500) | what it is |
|---|---|---|---|---|
| **`overt`** (default) | `toxicity==1 AND jailbreaking==0` | **512** | **PASS at 500/class** | overtly toxic user input. Expect: no drift, prompt-side wins |
| **`disguised`** | `jailbreaking==1` | **181** | **FAIL — pool-limited** | an attack wrapper (roleplay / DAN-style) around harmful intent. The only stratum where drift can exist |
| `mixed` | `toxicity==1` | 693 | PASS at 500/class | **LEGACY**: the superseded run's substrate, both pooled. Kept only for reproducibility |

Benign is always `toxicity==0 AND jailbreaking==0` (8,889 unique available),
length-matched to whichever harmful class is in play.

**The disguised arm is genuinely pool-limited**, and CLAUDE.md §17 rule 2 requires
saying so rather than implying a compute choice: toxic-chat carries **204
`jailbreaking==1` annotations in the entire dataset**, which is **181** after dedup and
the shared length/ambiguity filters. 181/class clears the ≥30 floor and misses the ≥500
floor. **Every number from that arm is PROVISIONAL and labelled so.** The `overt` arm
carries the rule-1-compliant headline at 500/class.

### 3.5 What the re-basing predicts — pre-registered before any run

See [§10](#10-pre-registered-falsifiers). The short version, registered **2026-08-08**:
on `disguised` we predict `threshold_freeform` finally clears its confound bar (F1) and
that the drift margin is **larger** on `disguised` than on `overt` (F3). We
**explicitly do not** predict that the trajectory detector beats the prompt-side rival
(F2): the measured prompt-content bar on the disguised substrate is **0.9688**, higher
than on the overt one, because DAN-style wrappers are lexically stereotyped and
unigrams catch them. F2 failing on **both** substrates would be a clean, publishable
negative about toxic-chat as a substrate — not a failure of the lesson.

---

## 4. Pipeline

```
  data.select_prompts(substrate)          abliterated Gemma-3-1B (local)
        harmful prompts     benign prompts                     |
            |                    |                             |
            |   prompt-channel confound audit (CPU, no model)   |
            |   -> length / content / shuffle bars              |
            |                    |                             |
            +--------- per prompt: generate_and_capture -------+
                                 |  (greedy decode; ONE forward pass,
                                 v   output_hidden_states=True @ layer 12)
              trajectories : List[ [n_tokens, dim] ]   label = prompt class
                                 |
                    fingerprint asserted against the config + prompt ids
                                 |
                  completion-channel confound audit (BEFORE the methods)
                    length / count / content / shuffle / mean_norm / final_norm
                                 |
                    +------------+-------------------------------+
                    |         5-fold StratifiedKFold CV          |
                    v                                            v
     threshold_freeform   per_turn_max   trajectory_mlp   seq_gru   (models.py reused)
         (training-free)     (STATELESS)      |             |
                    \__________|______________|_____________/
                                 |
        out-of-fold P(jailbreak) -> AUC (+boot CI), F1, TPR@FPR=0.10,
        margin vs the confound bar (paired bootstrap CI), margin vs the PROMPT rival
                                 |
                    +------------+-------------------------------+
                    |  EARLY-DETECTION: AUC using only first K   |
                    |  tokens, each vs its OWN first-K bar       |
                    v                                            v
              OOD arm (jackhhao/jailbreak-classification)
                                 |
         results_<substrate>.json  +  ROC / early-AUC-vs-K / risk-drift PNGs
```

---

## 5. Files

| file | role |
|---|---|
| `config.py` | every knob: model, layer, generation cap, **substrate**, the one authoritative `N_PER_CLASS`, window, target FPR, early-K list, OOD config, CV/methods, per-substrate paths |
| `trajectory.py` | generate + capture the per-token trajectory; the training-free sliding-window projection detector |
| `data.py` | substrate-aware prompt selection (`select_prompts`), the **config fingerprint**, generation + cache, `confound_report()` on the shared spine, `prompt_confound_report()` (the rival), `early_k_confound()` |
| `ood.py` | the out-of-distribution arm: `jackhhao/jailbreak-classification`, length-capped and length-matched |
| `run_trajguard.py` | orchestrator: fingerprint check -> **requested-vs-achieved gate** -> confound audit -> 5-fold CV -> early-detection + its own bars -> OOD -> falsifier verdicts -> `results_<substrate>.json` + plots |
| `infer.py` | generate for one harmful + one benign prompt; print the per-token sliding-window risk drifting up and crossing `tau` |

`models.py` is **not** in this folder — it is imported unchanged from the sibling
`multiturn_jailbreak`. `confound_report` is no longer this lesson's own: it is
[`common/confound.py`](../common/confound.py), the one shared spine.

---

## 6. Code walkthrough, file by file

### `config.py` — every knob in one place, and exactly one `N`

The local abliterated Gemma path, `LAYER = 12`, `MAX_NEW_TOKENS`, the detector knobs
(`WINDOW`, `TARGET_FPR`), `EARLY_KS = [2,4,8,16,32]`, `METHODS`, `N_FOLDS`,
`BOOTSTRAP = 10000` (CLAUDE.md §7's floor; it is seconds of CPU), the OOD block, and
**per-substrate paths** so two arms can never overwrite each other's numbers.

`N_PER_CLASS = 500` is now **the single authoritative number**. It previously appeared
as 500 here, 120 in §8 of this README and 300 in the shipped artifact — see
[§13](#13-artifact-discipline-and-reproduction).

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
`predict_proba_earlyK`. A CPU self-test on synthetic drifting-vs-flat trajectories
asserts AUC > 0.9 with no model.

### `data.py` — select, fingerprint, generate, audit

`select_prompts(substrate, n_per_class, seed)` is **CPU-only, deterministic and loads
no model**. It returns the chosen rows plus a header that states the pool ceiling, the
requested `n`, the effective target, and — critically — whether a shortfall is
`pool_limited`. That flag is the rule-2 distinction between a documented corpus ceiling
and a silent bug, and the caller must not have to infer it.

`dataset_fingerprint` is sha256 over the config snapshot **and the sorted sampled
prompt group-ids**. Including the ids is what makes it a real guard: a change to the
shared loader, the seed or the pool reshuffles *which* prompts were used while leaving
every scalar knob identical — the exact bug `cross_trajectory` shipped. `load_or_build`
recomputes the expected fingerprint (seconds of CPU) and **raises `CacheMismatch` with
a field-by-field diff** rather than falling through to a silent rebuild or returning
stale arrays.

`confound_report` delegates to [`common/confound.py`](../common/confound.py) and adds
the two bars the spine cannot know about — `mean_norm` and `final_norm`. Those are the
load-bearing ones here: the detector's input is a matrix of hidden states, so the
trivial shortcut is *geometric*. `harm_direction` is a diff-of-means, and a mean-norm
gap projects straight onto it — if the classes simply sit at different residual-stream
magnitudes, one scalar separates them and every sequence model's margin is illusory.

`prompt_confound_report` is the retraction in executable form: it measures what a
bag-of-words model gets from the prompt alone. It is reported **separately and never
folded into the confound bar**, because a prompt-side classifier is a rival **method**,
not a trivial confound.

`early_k_confound(K)` computes the bar on the same first-K truncation the early
headline is read from. §10 of the superseded README promised this and never delivered
it, which left the lesson's entire pitch as its one unpriced number.

### `ood.py` — the out-of-distribution arm

`jackhhao/jailbreak-classification` (ungated, cached): **629 unique jailbreak / 1,323
unique benign** after the shared dedup. Rendered through the *same* `common/data.py`
primitives as the in-domain set — rendering two channels differently is how
`biencoder_guard` manufactured a 0.72 length AUC out of nothing. Its jailbreak prompts
are far longer than its benign ones (median 1,544 vs 234 chars), so `select_ood_prompts`
caps the positives at the benign class's own p90 length ceiling before length-matching.
Measured effect, CPU-only:

| arm | n/class | length bar | prompt-content bar |
|---|---|---|---|
| p90 length cap (default) | 274 | **0.5624** | 0.9873 |
| no cap | 400 | **0.6504** | 0.9914 |

An OOD number under a 0.65 length bar is not an OOD number, so the capped arm is the
default and both are reported.

### `run_trajguard.py` — the orchestrator

Prints the pre-registered falsifiers **first**, loads the fingerprint-checked dataset,
runs `assert_n_achieved` **before any metric is computed** (see §13), runs the confound
audit **before the methods** so the bar exists before the headline, then 5-fold
StratifiedKFold CV. Per method: AUC + bootstrap CI, F1, accuracy, TPR@FPR=0.10, the
margin over the binding bar **with a paired bootstrap CI over the same resample
indices**, and the margin over the prompt-side rival. Then the early-detection curve
with per-K bars, the OOD arm, and the falsifier verdicts. `results_<substrate>.json` is
written **before** the summary print.

### `infer.py` — watch the risk drift up

Loads the model once, fits the harm direction + `tau` on a small reference slice, then
for one harmful and one benign prompt prints the per-token sliding-window risk.

---

## 7. The dataset — two substrates and an OOD arm

There is no ready-made corpus of labelled per-token jailbreak trajectories — we
**generate** it. Prompts come from `lmsys/toxic-chat` 0124 (Lin et al. 2023,
arXiv:2310.17389, EMNLP'23 Findings), the same pool the shared loader uses, read
through `common/data.py`'s own primitives so the discipline is identical.

| role | selection | pool | label |
|---|---|---|---|
| harmful (`overt`) | `toxicity==1 AND jailbreaking==0` | 512 | **1** |
| harmful (`disguised`) | `jailbreaking==1` | **181 — pool-limited** | **1** |
| benign (both) | `toxicity==0 AND jailbreaking==0`, length-matched | 8,889 | **0** |
| OOD | `jackhhao/jailbreak-classification`, p90-length-capped + length-matched | 274/class | 1 / 0 |

**The label is the prompt's class, not a judged rating of the completion.** We do
**not** run a judge over the generated text; a harmful-class prompt is labelled 1
regardless of what the model actually wrote. The abliterated model is chosen precisely
because it **complies** — but §3 is the discovery that this very property is what
destroys the drift the lesson set out to detect. The harm direction is fit on **train
only** in every fold, so no completion is graded on the direction it defined.

**Natural base rates**, recorded before rebalancing to 1:1: toxic ≈ 7.3% of raw rows,
`jailbreaking` ≈ 2.0%. PR-AUC and calibration must be read against those, not against
the balanced set.

### Why not `intrinsec-ai/cstm-bench`

It is the benchmark CLAUDE.md §17 rule 8 names for this lesson *family*, and it is the
right OOD for `cross_trajectory` / `meerkat`. It is the wrong one **here**: 108 rows of
multi-session **agent traces**, whose unit is a session, not a generated token. It is
also not in this host's HuggingFace cache and this host has no HF token. Stated rather
than quietly skipped.

The best-matched corpus in existence for this lesson is `allenai/wildjailbreak`
(`adversarial_harmful` vs `adversarial_benign` — "looks like a jailbreak, isn't", ~82k
and ~79k rows). It is **gated** and unusable without a token.

---

## 8. Sibling: multiturn_jailbreak

| | `multiturn_jailbreak` | `trajguard` (this lesson) |
|---|---|---|
| the sequence chunk | a conversation **turn** | a generated **token** |
| the trajectory | across turns of a chat | across tokens of one completion |
| when it fires | after several turns | **during** a single generation (streaming) |
| the attack | Crescendo / ActorAttack escalation | a jailbroken **completion** drifting to the harm region |

Because a token trajectory is the same `[n_steps, dim]` object a turn trajectory
is, `models.py` is **reused unchanged**. The only new code is the capture path
(`generate_and_capture`) and the training-free sliding-window projection detector.

The cross-lesson finding, now stated sharply by §3: **trajectory modelling earns its
keep when the individual chunks look benign (multi-turn escalation) and earns nothing
when the chunks already carry the signal (active harmful generation).**

---

## 9. Running

From the **repo root** (`steeringresearch/`):

```bash
# 1. CPU-only, NO model: the training-free detector on synthetic trajectories
python -m steering_tutorials.trajguard.trajectory

# 2. CPU-only, NO model: prompt selection + the PROMPT-CHANNEL confound bars for
#    every substrate. This is where the retraction in section 3 is reproduced.
python -m steering_tutorials.trajguard.data

# 3. CPU-only, NO model: the OOD arm's selection and its measured length/content bars
python -m steering_tutorials.trajguard.ood

# 4. The full generate -> capture -> CV run (needs the ~2-3 GB abliterated Gemma-3-1B).
#    Run BOTH arms; the lesson is the contrast.
TG_SUBSTRATE=overt      python -m steering_tutorials.trajguard.run_trajguard
TG_SUBSTRATE=disguised  python -m steering_tutorials.trajguard.run_trajguard

# 5. Watch the per-token risk drift up on a harmful prompt and stay flat on a benign one:
python -m steering_tutorials.trajguard.infer
```

**Env caps** (shrink an eval into one foreground window — the host's RAM, not
VRAM, is the wall):

| var | meaning | default |
|---|---|---|
| `TG_SUBSTRATE` | `overt` \| `disguised` \| `mixed` | `overt` |
| `TG_N_PER_CLASS` | **the one authoritative n**; auto-clamped to the substrate's pool | **500** |
| `TG_MAX_NEW_TOKENS` | generated tokens per completion (trajectory length) | 40 |
| `TG_LAYER` | residual layer read per token | 12 |
| `TG_WINDOW` | sliding-window length (tokens) | 4 |
| `TG_FOLDS` | StratifiedKFold CV folds | 5 |
| `TG_BOOTSTRAP` | bootstrap resamples | 10000 |
| `TG_REBUILD` | `1` = regenerate the trajectory cache (needs the GPU) | off |
| `TG_OOD` | `0` = skip the OOD arm | on |
| `TG_OOD_LENCAP` | quantile of the benign length distribution capping OOD positives | 0.90 |
| `TG_INFER_N` | reference completions per class in `infer.py` | 12 |

On Windows PowerShell set env vars first, e.g. `$env:TG_SUBSTRATE = "disguised"`.

**No judge.** This is a **detection** lesson: the detector reads a signal off the
frozen model's own decoding-time hidden states — there is no LLM judge and no
generation-of-a-verdict, so the off-family-judge discipline of the steering
lessons does not apply.

---

## 10. Pre-registered falsifiers

Registered **2026-08-08**, before any run on the new substrates. The runner **prints
them at startup and evaluates every one of them** into
`results_<substrate>.json → falsifier_verdicts`, so they cannot be quietly dropped from
the write-up. A falsifier is a condition under which a claim is **false**, not a hope
about what will happen.

| tag | claim | FALSE if |
|---|---|---|
| **F0** | leakage control | `confound.shuffle.auc > 0.60`. If this trips the run is **invalid** and nothing below means anything |
| **F1** | the drift detector clears its own bar | `AUC(threshold_freeform) <= confound.worst_auc` on the **disguised** substrate |
| **F2** | **decoding beats the prompt — THE PAPER'S CLAIM** | `max_method_auc <= prompt_channel.content.auc` |
| **F3** | the substrate contrast (the pedagogy) | `margin(threshold_freeform, disguised) <= margin(threshold_freeform, overt)` |
| **F4** | streaming earns its keep | `early_auc(K) <= early_confound(K).worst_auc` for **every** K |
| **F5** | the OOD arm transfers | `ood_max_method_auc <= ood.prompt_channel.content.auc` |

**Our registered predictions, stated as numbers where we already have them:**

- **F1 — we predict it HOLDS on `disguised` and FAILS on `overt`.** On `overt` the
  model complies from token 1, so there is nothing to drift; the superseded mixed-arm
  run put `threshold_freeform` at 0.638 against a 0.735 bar.
- **F2 — we predict it FAILS on both substrates.** The prompt-content bars are already
  measured: **0.8779** (overt) and **0.9688** (disguised). Beating 0.9688 with a
  40-token trajectory from a 1B model is not what we expect. This is the substantive
  pre-registration: we are predicting the paper's headline comparison **does not
  reproduce on toxic-chat**, and if it does hold we will have been wrong in the
  direction that favours the paper.
- **F3 — genuinely open.** It is the one prediction the re-basing exists to test.
- **F4/F5 — open.**

A pre-registration that only predicts wins is not a pre-registration.

### 10.1 Outcome — the F2 prediction above was WRONG

> **The four predictions above are reproduced verbatim and are NOT edited to match the
> outcome.** A pre-registration that gets quietly corrected after the data lands is
> worth nothing; the whole value is in leaving the wrong call on the page.

**F2 was predicted to FAIL. It HELD.** From
[`artifacts/results_disguised.json`](artifacts/results_disguised.json)
(`falsifier_verdicts.F2_decoding_beats_prompt`):

```
best_method_auc    0.9855   (trajectory_mlp)
prompt_content_auc 0.9688
holds              true
```

So the decoding channel beat the prompt channel by **+0.0167**, and TrajGuard's headline
comparative claim — that decoding-time hidden states carry signal the prompt does not —
**reproduces on this substrate**. §10 said in advance that *"if it does hold we will have
been wrong in the direction that favours the paper."* That is exactly what happened. We
were wrong, in the direction that favours the paper, and the correct record of that is
this paragraph rather than a rewritten bullet.

**Three things keep this from being a win to celebrate.**

1. **The margin is +0.0167 — smaller than the uncertainty on either side of it.**
   `trajectory_mlp`'s own 95% CI is [0.9731, 0.9948], i.e. **0.0217 wide**, and the margin
   does not clear that. On **181/class** at **one seed**. No paired CI on the margin
   exists either (§12.3: the content bar is a CV-pooled centroid model exposing no
   per-item score, so no resample pairing is possible). This is a screening-tier
   reproduction, not a measured effect.
2. **Two of the four methods do not clear the rival at all.** `trajectory_mlp` (+0.0167)
   and `per_turn_max` (+0.0161) beat it; `seq_gru` is **−0.0145** and
   `threshold_freeform` is **−0.2119**. F2 is defined on the *max* over methods, so it
   passes on the strength of the single best row.
3. **Do not conflate the two bars.** §11.2's binding **confound bar** is the
   completion-channel `content` bar at **0.9103** — a trivial-baseline floor every method
   must clear. F2's **0.9688** is the prompt-channel content **rival**: a competing
   method, deliberately never folded into `worst_auc` (§12.1), because burying a rival
   inside a confound bar would let the lesson "clear the bar" without facing the
   comparison. A method can clear 0.9103 and still lose to 0.9688 — `seq_gru` does
   exactly that (+0.0441 over the bar, −0.0145 against the rival).

**F1 and F5 went the other way**, and the same no-editing rule applies to them: F1 was
predicted to HOLD on `disguised` and **FAILED** (`threshold_freeform` 0.7570 vs the 0.9103
bar); F5 was open and **FAILED** (OOD best 0.8653 vs a 0.9873 prompt-content rival). F3 is
still open because the `overt` arm has not been run. Of the falsifiers with a verdict on
this substrate, **two failed, two held, and one of the two that held did so against our
own prediction.**

---

## 11. Results

> **Status: HALF RUN.** The **`disguised`** substrate ran on 2026-08-08
> ([`artifacts/results_disguised.json`](artifacts/results_disguised.json), fingerprint
> `b34e4b2e85bb…`, `models/google/gemma-3-1b-it`, layer 12, 40 new tokens, 5-fold,
> 10,000 bootstrap resamples, seed 0) and §§11.2–11.4 and §12.2 below are **measured**.
> The **`overt`** substrate has **not** been run — §11.1 is still `[PENDING RUN]`, and
> with it **F3**, the substrate contrast that is the entire reason the lesson was
> re-based (`substrate_comparison` in the artifact says so in as many words: *"F3 needs
> BOTH arms; run the other substrate with TG_SUBSTRATE=overt"*). The CPU-only
> prompt-channel numbers in [§3](#3-retraction-and-the-re-basing-it-forced) and
> [§12.2](#122-measured-now-cpu-only-no-model) are reproducible right now with
> `python -m steering_tutorials.trajguard.data`.

### 11.1 `overt` substrate — 500/class, rule-1 compliant

| method | AUC | 95% CI | F1 | TPR@FPR=0.10 | margin vs confound bar | **margin vs prompt-only (0.8779)** |
|---|---|---|---|---|---|---|
| `threshold_freeform` | `[PENDING RUN]` | | | | | |
| `per_turn_max` (stateless) | `[PENDING RUN]` | | | | | |
| `trajectory_mlp` | `[PENDING RUN]` | | | | | |
| `seq_gru` | `[PENDING RUN]` | | | | | |

### 11.2 `disguised` substrate — 181/class, **POOL-LIMITED, PROVISIONAL** — MEASURED

The binding confound bar on this substrate is `content` at **0.9103** (§12.2); the
prompt-content **rival** is **0.9688** and is never folded into the bar (§12.1).

| method | AUC | 95% CI | F1 | TPR@FPR=0.10 | margin vs confound bar (0.9103) | **margin vs prompt-only (0.9688)** |
|---|---|---|---|---|---|---|
| `threshold_freeform` | 0.7570 | [0.7042, 0.8067] | 0.675 | 0.547 | **−0.1533 — BELOW the bar** | −0.2119 |
| `per_turn_max` (stateless) | 0.9850 | [0.9732, 0.9941] | 0.856 | 0.978 | **+0.0747** | +0.0161 |
| `trajectory_mlp` | **0.9855** | [0.9731, 0.9948] | 0.951 | 0.983 | **+0.0752** | +0.0167 |
| `seq_gru` | 0.9544 | [0.9313, 0.9742] | 0.906 | 0.901 | **+0.0441** | −0.0145 |

**Read both halves of this table.** Three of the four methods clear the confound bar —
`per_turn_max`, `trajectory_mlp` and `seq_gru` beat TF-IDF unigrams on the same
completions by +0.075/+0.075/+0.044. But **the paper's own method, `threshold_freeform`,
lands 0.153 BELOW the bar** — the drift detector this lesson is named after cannot beat
unigram content on the substrate where it should be strongest. That is **F1 FAILING**
(`falsifier_verdicts.F1_drift_clears_bar.holds = false`), exactly as a falsifier is
supposed to work, and it is the more important half of the row.

`vs_confound_paired_ci` is `null` on all four: the spine's `content` bar is a CV-pooled
centroid model that exposes no per-item score, so no paired interval is fabricated
(§12.3).

**F2 HOLDS — against our own registered prediction**, by **+0.0167** over the
prompt-content **rival** (0.9688), which is *not* the 0.9103 confound bar this table's
margin column is priced against; the two are different quantities and §12.1 explains why
they are kept apart. §10 predicted F2 would FAIL on both substrates. The full write-up —
the verbatim prediction, why it was wrong, and why the margin does not support a strong
claim — is in [§10.1](#101-outcome--the-f2-prediction-above-was-wrong).

### 11.3 Early detection, each K against its own first-K bar — MEASURED

Each K is scored against a bar recomputed on that same K-token prefix (there is no
character or content bar at a prefix — the first-K tokens were never decoded to text —
so the prefix bar is `count` + the two geometry bars).

| | K=2 | K=4 | K=8 | K=16 | K=32 |
|---|---|---|---|---|---|
| `threshold_freeform` | 0.7332 | 0.7380 | 0.7483 | 0.7496 | 0.7569 |
| `seq_gru` | **0.9684** | 0.9595 | 0.9694 | 0.9680 | 0.9570 |
| **first-K confound bar** | 0.6249 (`mean_norm`) | 0.5765 (`mean_norm`) | 0.5826 (`final_norm`) | 0.5742 (`final_norm`) | 0.5479 (`count`) |

**F4 HOLDS**, 10/10 cells clearing their own bar
(`falsifier_verdicts.F4_streaming_clears_its_own_bar.holds = true`). Note what the row
does *not* say: `seq_gru` is already at 0.968 by **K=2** and does not improve with more
tokens, so what the early-K curve demonstrates is that two tokens suffice on this
substrate — not that the method accumulates evidence over the trajectory.

### 11.4 OOD — `jackhhao/jailbreak-classification`, 274/class — MEASURED

Pool-limited (629 unique jailbreak rows → 274 after the 1,311-char length cap),
`pool_limited: true`, 548 completions captured, 0 skipped.

| method | AUC | 95% CI | degradation vs in-domain |
|---|---|---|---|
| `threshold_freeform` | 0.7832 | [0.7450, 0.8198] | **+0.0263** (better OOD than in-domain) |
| `per_turn_max` | **0.8653** | [0.8344, 0.8938] | −0.1196 |
| `trajectory_mlp` | 0.7854 | [0.7470, 0.8218] | **−0.2001** |
| `seq_gru` | 0.8202 | [0.7839, 0.8547] | −0.1341 |

**Every method fails OOD, and the failure is reported as prominently as the in-domain
win.** The OOD completion-channel binding bar is `content` at **0.9341**, which *no*
method reaches; the prompt-content rival is **0.9873**, so **F5 FAILS**
(`ood_best_auc 0.8653` vs `0.9873`, `holds: false`). The in-domain result at §11.2 does
not transfer: `trajectory_mlp`, the in-domain winner, loses 0.20 AUC and drops below
`per_turn_max`. F5 is a high bar by construction and that is stated here, per
CLAUDE.md §17.

### 11.5 The superseded run

The 2026-07-27 run (substrate `mixed`, **300/class**, bootstrap 2,000, a two-bar
confound audit with no content bar and no shuffle control) is quarantined under
[`artifacts/superseded_2026-07-27/`](artifacts/superseded_2026-07-27/) with a note
recording exactly why. **No number in it may be quoted as a current result.** Its
direction is corroboration for §3 and nothing more:

| method | AUC | F1 (from `results.json`) | margin vs the 0.7354 bar |
|---|---|---|---|
| `threshold_freeform` | 0.638 | 0.172 | **−0.098 — BELOW the bar** |
| `per_turn_max` | 0.931 | 0.781 | +0.195 |
| `trajectory_mlp` | 0.944 | 0.899 | +0.208 |
| `seq_gru` | 0.945 | 0.894 | +0.209 |

Two corrections carried from the audit: the previous README printed
`threshold_freeform`'s margin as **−0.045** where the arithmetic is
`0.63778 − 0.735439 = −0.09766`, i.e. **−0.098**; and three F1 cells (0.24 / 0.84 /
0.86) disagreed with the artifact beside them (0.172 / 0.781 / 0.894). Both are fixed
above. `trajectory_mlp`'s F1 is 0.899, printed 0.90 — the only cell that was right.

---

## 12. The confound audit — four bars and one rival

A detection AUC is not a result until the strongest **trivial** baseline on the same
data is reported beside it, and only `headline − baseline` may be claimed
([`CONFOUND_DISCIPLINE.md`](../CONFOUND_DISCIPLINE.md), CLAUDE.md §17 rule 7).

This lesson's own `confound_report` folded correctly (`max(auc, 1−auc)`) and ran
**before** the CV block, which is the discipline the rule actually asks for. It was
missing two of the four controls. It now runs on
[`common/confound.py`](../common/confound.py), the one shared spine, keeping the good
habits and gaining the missing bars:

| bar | what it asks | channel |
|---|---|---|
| `length` | can raw character count separate the classes? | completion + prompt |
| `count` | can the number of generated tokens separate them? Nominally capped at `MAX_NEW_TOKENS`, but **early EOS is class-informative** | completion |
| `content` | **the bar no detection lesson in this course had.** Can TF-IDF unigrams under 5-fold CV separate them? A "trajectory" detector that cannot beat unigrams is not reading trajectories | completion + prompt |
| `shuffle` | with labels permuted, does the pipeline still score above chance? Far from 0.5 means **leakage, not signal** | both |
| `mean_norm` | mean over tokens of `‖h_t‖` at layer 12. If the classes sit at different residual-stream *magnitudes*, one scalar separates them and every sequence model's margin is illusory | completion (this lesson's addition) |
| `final_norm` | `‖h_last‖` alone, the single cheapest such scalar | completion (this lesson's addition) |
| `multivariate` | **the bar that makes the set complete.** Can the four scalars *combined* separate them, when no one of them can alone? Logistic regression under the same folds as the methods — see [§12.3](#123-the-three-controls-that-were-missing--now-in-code) | completion (this lesson's addition) |

`worst_auc` over all of those **except `shuffle`** is the binding bar. The shuffle
control is a leakage diagnostic and is never a bar to clear. Every bar in that list is a
single feature except `multivariate`, which is exactly why it had to be added: a method
that beats each trivial feature individually can still be beaten by their combination.

### 12.1 The prompt channel is a RIVAL, not a bar

This is the structural half of the retraction. A prompt-side classifier is not a
trivial confound — it is the **competing method** the paper claims decoding-time states
beat. So `prompt_confound_report` is computed, reported side by side, given its own
column in the summary table (`vs PROMPT`) and its own falsifier (**F2**) — and it is
**never** folded into `worst_auc`, because burying a rival method inside a confound bar
would let the lesson "clear the bar" without ever facing the comparison.

### 12.2 Measured now, CPU-only, no model

Reproduce with `python -m steering_tutorials.trajguard.data`:

| substrate | n/class | prompt `length` | prompt `content` | prompt `shuffle` |
|---|---|---|---|---|
| `overt` | 500 | 0.5047 | **0.8779** | 0.5077 |
| `disguised` | 181 | 0.5065 | **0.9688** | 0.5702 |
| `mixed` (legacy) | 500 | 0.5012 | **0.8842** | 0.5302 |

The `length` column is ~0.50 in every row **because the shared loader made it so**.
That is the number the retracted claim was built on. The `content` column is what it
was silently standing in for.

The `disguised` shuffle control at **0.5702** is the highest of the three and sits
below the 0.60 invalidation threshold but not comfortably; at 181/class the shuffle
statistic is itself noisy. Recorded, not smoothed.

The completion-channel bars require the generated text and therefore the GPU run. That
run landed for `disguised` on 2026-08-08 and they are **measured**
(`results_disguised.json:confound`, 181/class, 362 completions):

| substrate | `length` | `count` | `content` | `shuffle` | `mean_norm` | `final_norm` | **binding bar** |
|---|---|---|---|---|---|---|---|
| `disguised` | 0.5064 | 0.5515 | **0.9103** | 0.5404 | 0.5463 | 0.5753 | **`content` 0.9103** |
| `overt` | `[PENDING RUN]` | | | | | | |

`content` binds by a wide margin — every scalar bar, including both geometry bars this
lesson added, is within 0.08 of chance. So the classes are **not** separated by
completion length, token count, or residual-stream magnitude; they are separated by what
the model actually wrote, and that is the only bar a trajectory method has to beat.
`shuffle` at 0.5404 is below the 0.60 invalidation threshold, so **F0 passes** and the
run is valid.

### 12.3 The three controls that were missing — now in code

The three gaps this section used to list were **code** gaps: the controls did not exist,
so "worst of univariate" was standing in for "the strongest trivial baseline". All three
are implemented in [`controls.py`](controls.py) and wired into the runner: the block
lands in `results_<substrate>.json` under `controls`, and each method cell gains
`matched_bin` and a labelled `vs_confound_paired_ci`.

**The shipped `results_disguised.json` predates them.** It was written 2026-08-08 and
still carries `vs_confound_paired_ci: null` on all four methods and no `controls` block;
populating those cells needs a re-run of `run_trajguard`, which needs the GPU. Control 1
is the exception — it is measurable today from the committed text-free sidecar with no
model at all, and is measured in §12.3.2. Do not read the absence of a `controls` block
in the current artifact as the controls being absent from the lesson.

| control | what it asks | status |
|---|---|---|
| **1. multivariate trivial baseline** | can `{charlen, tokencount, mean_norm, final_norm}` *combined* separate the classes, when no one of them can alone? Logistic regression, **same folds** as the methods, scaler fit on the training fold only, pooled out-of-fold, folded directionless — and folded **into** `worst_auc`, so if it binds, it binds | **MEASURED** (below) |
| **2. matched-bin** | does the separation survive at approximately fixed completion length? AUC recomputed within `charlen` quantile bins and pooled by pair count (the stratified Mann-Whitney statistic), CI resampled **within** bins | **CODE READY** — needs the runner |
| **3. paired margin CI vs the binding bar** | the four `null`s in the shipped artifact | **SOLVED** (below) |

#### 12.3.1 The paired-CI blocker was solvable

The shipped `results_disguised.json` carries `vs_confound_paired_ci: null` on all four
methods, with a note explaining that the spine's `content` bar "exposes no per-item
score". The note was accurate about the symptom and wrong about the cause.
`common.confound.content_bar` **does** compute a per-item out-of-fold score for every
item — `scores[i] = <tfidf_i, centroid_pos> − <tfidf_i, centroid_neg>` — and then
discards the vector, returning only the pooled AUC.

`controls.content_bar_scores` reproduces that loop through the spine's **own**
primitives and returns the vector, then **asserts** its pooled AUC equals the spine's to
`1e-9`. If `common/confound.py` ever changes, the assertion raises rather than pairing
the methods against a different model than the one that set the bar — the §18.8 rule
("assert your anchors"). The clean long-term fix is a `return_scores=True` flag on the
spine itself; `common/` is shared and this lesson does not edit it.

Consequence: a paired bootstrap CI is now computable against `content` **and** against
`multivariate`, not only against the four scalar bars, and every CI records
`against_bar` — a CI against `final_norm` and one against `content` are different
claims and must never print under one header.

#### 12.3.2 Control 1, measured on the disguised substrate — CPU, no GPU

Reproduce with `python -m steering_tutorials.trajguard.controls --from-meta` (reads the
committed text-free sidecar; no model, no hidden-state blob):

| feature | univariate (in-sample) | LR coefficient |
|---|---|---|
| `charlen` | 0.5064 | +0.384 |
| `tokencount` | 0.5515 | −0.553 |
| `mean_norm` | 0.5463 | +0.061 |
| `final_norm` | 0.5752 | −0.237 |
| **joint (out-of-fold)** | **0.5940** [0.5345, 0.6525] | — |
| joint (in-sample, optimistic) | 0.6145 | — |

**The multivariate baseline does not change the verdict here.** At 0.5940 it is far
below `content` (0.9103), so `content` remains the binding bar and every margin in
§11.2 stands unchanged. That is the honest outcome of computing it: the control was
worth having and it did not fire. It is now priced rather than assumed.

Two things the table is careful about. The univariate column is **in-sample** — a fixed
feature has no parameters, so its AUC is unbiased — while the joint column fits five
numbers and is therefore reported **out-of-fold**. The gain over the best univariate is
`+0.0188` out-of-fold and `+0.0393` like-for-like in-sample; only the out-of-fold figure
competes for the binding bar. And a *negative* out-of-fold gain, which is possible at
small `n`, would mean the combination buys nothing beyond one feature — not that it is
worse than its parts.

#### 12.3.3 What control 2 can and cannot do

The matched-bin control's power is **bounded by bin width**: stratifying a continuous
confound into 4 bins leaves a residual inside each bin. On a planted length confound
(`controls._selftest` case A, a "detector" whose score *is* length) the raw AUC is
0.9647 and the within-bin AUC falls to 0.8711 / 0.7231 / 0.7086 / 0.5772 / 0.5357 at
2 / 4 / 8 / 12 / 20 bins. So a within-bin AUC is uninterpretable without the bin count
beside it, and `n_bins_achieved` is reported separately from `n_bins_requested` because
ties in the stratifier can collapse edges. `MATCHED_BINS` defaults to 4: on 362 items
that is ~90 per bin (~45/class), and refining further on this pool trades the control
for noise.

The same self-test checks the converse — case B plants a *real* signal alongside the
same length gap and asserts the control leaves it intact (0.9936 → 0.9906). A
stratification that flattens genuine signal is not a control, it is a bug.

#### 12.3.4 Why control 1 exists at all

`controls._selftest` case C constructs the failure that worst-of-univariate cannot see:
two trivial features sharing a large-variance noise term with the class signal entering
at opposite sign, so each is at chance marginally while their difference separates
perfectly. Measured: `length` 0.6178, `count` 0.5000, `content` 0.5000, `mean_norm`
0.5438, `final_norm` 0.5438 — **and the multivariate baseline at 1.0000**, which then
correctly becomes the binding bar. A method beating every univariate bar on that data
would be beating nothing.

Run the controls' own CPU self-tests with
`python -m steering_tutorials.trajguard.controls` (seconds, no model, no GPU).

### 12.4 A defect found while measuring: the committed sidecar was the OOD arm's

`data._save_cache` wrote `write_meta(C.META_PATH, ...)` with the path **hardcoded**. The
OOD builder calls the same `_save_cache` with `C.OOD_CACHE`, so building the OOD arm
silently overwrote the **in-domain** committed sidecar with the OOD arm's 548 records.
The shipped `trajectory_meta_disguised.json` was byte-for-byte the OOD file: 548
completions, a `substrate: "ood:jackhhao/..."` snapshot, the OOD fingerprint, and a
`cache_file` field naming the in-domain npz it did not describe.

Nothing crashed. Both files were well-formed JSON of plausible size — the CLAUDE.md
§18.8 pattern exactly. The cost was §13's central reproduction promise: *"the numeric
confound bars recompute from this file alone with no GPU"* was false for the arm this
README headlines.

Fixed three ways: `data._meta_path_for` derives the sidecar path from the cache being
written; `write_meta` takes the `cache_file` name from its caller instead of hardcoding
`C.TRAJ_CACHE`; and `write_meta` now **asserts** that an `ood:` snapshot only ever lands
in `OOD_META_PATH` and vice versa, so the two can never merge again. The redundant
second `write_meta` in `ood.py` — which wrote the *correct* file while `_save_cache` had
already written the same records to the wrong one — is removed.

Both sidecars were regenerated from the intact `.npz` caches (CPU only, no model). The
in-domain sidecar now reproduces all four scalar bars in `results_disguised.json`
**exactly**: `length` 0.506395, `count` 0.551540, `mean_norm` 0.546290, `final_norm`
0.575227, with the class means matching to every printed digit.

---

## 13. Artifact discipline and reproduction

### 13.1 The three-different-N defect, and what replaced it

`config.py` said **500** ("this is 1000 completions"), §8 of this README said **120**,
and the shipped artifact said **300**. All three were "true" simultaneously because
`load_or_build()` returned the cache whenever its `labels` array was non-empty —
checking no `n_per_class`, no `max_new_tokens`, no `LAYER`, no `MODEL_ID`, no pool. A
run configured for 500/class silently consumed a 600-completion cache and wrote
`n_harmful: 300`. This was not hypothetical; it is how the artifact came to contradict
its own config.

Three changes, all enforced in code:

1. **One authoritative `N`** — `config.N_PER_CLASS = 500`, auto-clamped to the
   substrate's measured pool, surfaced in the env table above and nowhere else.
2. **A config fingerprint** — sha256 over `{substrate, n_per_class, seed,
   max_new_tokens, layer, model_id, greedy}` **and the sorted sampled prompt
   group-ids**, stored inside the npz and in the sidecar. `load_or_build` recomputes it
   and raises `CacheMismatch` with a field-by-field diff. It does not fall through to a
   silent rebuild and it never returns the stale arrays. *An anchor that matches
   nothing must fail, not pass* (CLAUDE.md §18.8).
3. **A requested-vs-achieved gate** — `assert_n_achieved` runs **before any metric**
   and raises unless the shortfall is explained by a `pool_limited` corpus ceiling. The
   accounting (`requested / effective / achieved / pool / skipped / rule1_compliant`)
   lands in `results_<substrate>.json → sizes`, so the rule-1 status is readable from
   the artifact rather than inferable only from the loader.

### 13.2 The 106 MB artifact — what we chose and why

`artifacts/token_trajectories.npz` was **106,095,984 bytes = 101.18 MiB**, i.e. **over
GitHub's 100 MiB hard per-file limit**. A push containing it is rejected outright. It
was gitignored — correctly — but that made §10's old instruction ("*CPU only, no GPU,
no regeneration — the trajectories are cached in `artifacts/token_trajectories.npz`*")
**false for every reader except whoever generated them**.

**The choice made here: keep the raw cache out of git, and commit a text-free
reproduction sidecar in its place.** Considered and rejected: sharding it under the
limit (a 500/class cache is ~90 MB even at float16 — sharding puts ~90 MB of
regenerable floats in git history forever to reproduce numbers a 100 kB file already
reproduces).

What ships:

| artifact | committed? | why |
|---|---|---|
| `token_trajectories_<substrate>.npz` | **no** | ~90–100 MB of regenerable float16 hidden states. Now `savez_compressed` + float16 (was float32 uncompressed) |
| `token_trajectories_<substrate>.json` | **no** | it holds the abliterated model's **completions on harmful prompts**. Those are new harmful generations and are deliberately not republished |
| **`trajectory_meta_<substrate>.json`** | **yes** | ~100 kB, **text-free**: per-completion label, prompt group-id, token count, completion character length, `mean_norm`, `final_norm`, plus the dataset fingerprint |
| `results_<substrate>.json` | **yes** | every measured number |

So from a fresh clone, with **no GPU**: `select_prompts()` reproduces the exact prompt
set (deterministic, from the public dataset), the **prompt-channel** bars recompute in
full, and the **numeric completion-channel** bars (`length`, `count`, `mean_norm`,
`final_norm`) recompute from the meta sidecar. What genuinely needs the GPU is
regenerating the hidden states — and the fingerprint then proves the regenerated cache
is the same set. The completion **content** bar needs the generated text and therefore
the re-run; that is stated rather than implied away.

**Regeneration cost** (single RTX 4090 Laptop, abliterated Gemma-3-1B, greedy, 40 new
tokens): 1,000 completions for the `overt` arm, 362 for `disguised`, 548 for the OOD
arm. Wall-clock is `[PENDING RUN]` — the honest number is the one the run reports, and
CLAUDE.md §18.6 is explicit that latency figures formed across two machine states are
meaningless.

```bash
TG_SUBSTRATE=overt TG_REBUILD=1 python -m steering_tutorials.trajguard.run_trajguard
```

---

## 14. Honest caveats

- **The headline inference was retracted, not softened** ([§3](#3-retraction-and-the-re-basing-it-forced)).
  "You cannot call this from the prompt alone" was false, and the number it rested on
  was a control the shared loader applies by construction.
- **The paper's headline comparison was out of scope and is now in scope, untested.**
  TrajGuard's claim is that decoding-time hidden states beat input prompts. The
  superseded lesson built no prompt-side classifier, so it could neither support nor
  refute that. It now does — and the measured prompt bars (0.878 / 0.969) say the
  comparison is likely to go **against** the paper on this substrate.
- **`disguised` is pool-limited at 181/class** against a 500/class floor. Genuinely
  pool-limited (204 annotations exist in all of toxic-chat), stated everywhere,
  PROVISIONAL everywhere. The `overt` arm carries the rule-1-compliant headline.
- **Screening tier, not evaluation.** Single 1B model, one layer, one seed, 5-fold CV
  — a directional demo, not the n ≥ 7 seeds + rigor contract CLAUDE.md reserves the
  word "winner" for.
- **The abliterated model complies — that is the point, and the problem.** On an
  aligned model most harmful prompts would be refused and the "harmful" trajectory
  would be a *refusal* trajectory; the design would need re-labelling. On the
  abliterated one it complies from token 1, which is precisely what removes the drift.
- **Label = prompt class, not a judged harm rating.** The AUC measures separability of
  *prompt-class-conditioned* trajectories, which is what a streaming detector actually
  sees.
- **The harm direction is fit on train only** in every CV fold, so a detector is never
  graded on the direction it defined. Random CV (no grouping) is fine here because each
  completion is an independent generation and `common/data.py`'s `group_id` already
  collapsed surface near-duplicates.
- **Not actually streaming.** `generate_and_capture` generates fully, then re-forwards
  the whole sequence. For a causal decoder the layer-12 state at position *t* depends
  only on tokens ≤ *t*, so the values are equivalent to a true streaming read — but no
  halting path exists. This lesson detects; it does not stop.
- **Inspired-by, not a paper reproduction.** The sliding-window harm-projection
  detector + reused sequence classifiers operationalize TrajGuard's decoding-time
  *idea*; they are not a faithful reimplementation of its architecture. See `AUDIT.md`.
- **`intrinsec-ai/cstm-bench` is not used, deliberately** — wrong granularity
  (multi-session agent traces) and absent from this host's cache. `allenai/wildjailbreak`,
  the best-matched corpus in existence for this lesson, is gated.

---

## 15. Repository

Source and full artifacts:
<https://github.com/dlmastery/steeringresearch/tree/master/steering_tutorials/trajguard>

**Citations** (every arXiv id WebFetch-verified; see `AUDIT.md`):

- **arXiv:2604.07727** — *TrajGuard: Streaming Hidden-state Trajectory Detection for
  Decoding-time Jailbreak Defense.* Cheng Liu, Xiaolei Liu, Xingyu Li, Bangzhou Xin,
  Kangyi Ding. ACL 2026 Findings, 2026-04-09. *The lesson's subject; §3 is about the
  comparative claim in its abstract that this lesson had not tested.*
- **arXiv:2602.16935** — *DeepContext: Stateful Real-Time Detection of Multi-Turn
  Adversarial Intent Drift in LLMs.* Justin Albrethsen, Yash Datta, Kunal Kumar,
  Sharath Rajasekar. 2026-02-18. *The multi-turn sibling of the same idea.*
- **arXiv:2404.01318** — *JailbreakBench: An Open Robustness Benchmark for Jailbreaking
  Large Language Models.* Chao, Debenedetti, Robey, Andriushchenko, Croce, Sehwag,
  Dobriban, Flammarion, Pappas, Tramèr, Hassani, Wong. NeurIPS 2024 D&B, 2024-03-28.
  *The harmful top-up source inside `common/data.py`.*
- **arXiv:2310.17389** — *ToxicChat: Unveiling Hidden Challenges of Toxicity Detection
  in Real-World User-AI Conversation.* Zi Lin, Zihan Wang, Yongqi Tong, Yangkun Wang,
  Yuxin Guo, Yujia Wang, Jingbo Shang. EMNLP 2023 Findings. *The prompt pool, and the
  source of the `jailbreaking` column §3.4 re-bases on.*

See also
[the course map](../README.md),
[the turn-level sibling — multiturn_jailbreak](../multiturn_jailbreak/README.md)
(whose `models.py` this lesson reuses unchanged),
[lesson 1 — the single-prompt activation probe (READ)](../hello_world/README.md)
(whose prompt-only probe at AUC 0.965 is the rival §3 measures this lesson against), and
[the shared confound spine](../common/confound.py).
