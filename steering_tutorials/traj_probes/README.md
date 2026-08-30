# traj_probes — does the model's own residual stream know an agent trajectory is doomed, or is it just counting turns?

> **Reference:**
> - [Doomed from the Start: Early Abort of LLM Agent Episodes via a Recall-Controlled Probe Cascade (arXiv:2607.06503)](https://arxiv.org/abs/2607.06503)
>   — Zhihao Ruan, Xinyu Huang, Zhiyuan Zhou, Rongkai Wei, Jiayu Lin, Zhengyu Wang, Han Sun,
>   7 Jul 2026. WebFetch-verified 2026-08-27. *Relevance:* a linear probe on one fixed layer,
>   read from the very first round, predicts eventual episode failure; a recall-controlled
>   cascade then aborts doomed episodes early to save tokens (60.2% / 54.9% reported on
>   TextCraft / WebShop at 90% recall). This is **reproduction A** below.
> - [Tool-Call Dependency Structure is Linearly Decodable in LLM Agent Residual Streams (arXiv:2605.25310)](https://arxiv.org/abs/2605.25310)
>   — Kaiyue Sun, Alina Kazakov, 25 May 2026. WebFetch-verified 2026-08-27. *Relevance:* a
>   low-capacity linear probe decodes which earlier tool output feeds which later tool
>   argument, checked against a Hewitt-Liang random-label ceiling and a positional baseline.
>   This is **reproduction C** below.
> - "Goal-Drift Probes: Anticipating Multi-Turn LLM Agent Failure From Mid-Network
>   Activations", Chen, ICML 2026 AIWILD workshop. **[UNVERIFIED]** — no arXiv id was supplied
>   for this one and it has NOT been WebFetch-confirmed; cite it as unverified until someone
>   resolves it to a real record. *Relevance (as reported to us, unconfirmed):* failure
>   predicted 3 steps ahead at AUC 0.989, falling to >= 0.939 after step-index
>   residualisation. This is **reproduction B** below.
> - Dataset: [`AI45Research/ATBench`](https://huggingface.co/datasets/AI45Research/ATBench)
>   on HuggingFace, `apache-2.0` (dataset-card front matter). Paper:
>   [ATBench: A Diverse and Realistic Agent Trajectory Benchmark for Safety Evaluation
>   and Diagnosis (arXiv:2604.02022)](https://arxiv.org/abs/2604.02022) — Li, Luo, Xie, Fu,
>   Yang, Shao, Ren, Qu, Fu, Yang, Shao, Hu, Liu; cs.AI, submitted 2 Apr 2026 (v1), latest
>   revision 20 Aug 2026 (v4). WebFetch-verified 2026-08-29.

This lesson is the **residual-stream** half of the long-trajectory
guardrail-detection family (CLAUDE.md section 17 rule 8). Its sibling,
[`streaming_trajectory_aggregation`](../streaming_trajectory_aggregation/README.md),
embeds trajectory **text** with a sentence encoder and classifies the embedding —
the question there is "is this text separable?" Here we replay the trajectory
through a decoder and read **its own internal state**. The question is different
and harder to earn: does the model's residual stream carry the failure signal
*earlier and more generalisably* than anything visible on the surface, or does a
probe on it just rediscover position or vocabulary and dress it up as insight?

We run **Gemma-3-1B** (the aligned base, not the abliterated build — we are
reading how the model behaves normally, not steering it) on **one 4090**, over
**ATBench**. The reproduction papers use Qwen-2.5-7B / Qwen3-32B / Llama-3.3-70B
over TextCraft, WebShop, ALFWorld and tau-bench. **We do not reproduce their
numbers and this README will never print ours beside theirs.** What can transfer
is the *method and its controls* — not the magnitude of any AUC.

> **Status: the GPU pass has run, and the result is NEGATIVE.** Gemma-3-1B
> layer-12 activations over the pinned ATBench corpus (8,341 turn-rows, 997
> trajectories, group-aware 5-fold CV, 594.5s wall clock), 2026-08-29 —
> `artifacts/results_atbench_gemma-3-1b-it_L12.json`. The pre-registered
> content-bar falsifier (Section 8) **FIRED**: the activation probe is real
> (it clears both the step-index floor and the random-label ceiling by a wide
> margin) but it loses to a plain bag-of-words bar at every horizon measured
> (Section 7). **This is not a refutation of arXiv:2607.06503** — it is one
> model (1B vs 7B–70B), one layer, one pooling scheme, one corpus; the honest
> statement is that we cannot claim the reproduction, not that the method
> fails (Section 7's limitations subsection). The per-layer sweep is
> **COMPLETE**: six layers, every one below the bar, best is layer 12.

---

## 1. The three reproductions

| | paper | what it claims | what we build against it |
|---|---|---|---|
| A | Ruan et al., 2607.06503 | one layer, one linear probe, first round → predicts eventual failure; a recall-controlled cascade aborts doomed episodes | `probes.LinearTrajProbe` scored at every step k, and (not yet built) a cascade that aborts under a global recall target, matching `types.CascadeResult` |
| B | Chen, ICML 2026 AIWILD **[UNVERIFIED]** | failure predictable 3 steps ahead, survives step-index residualisation | `probes.StepResidualiser` + `auc_residualised` on every cell |
| C | Sun & Kazakov, 2605.25310 | tool-call argument dependencies are linearly decodable, above a random-label ceiling | `probes.random_label_control` (Hewitt & Liang, EMNLP 2019, arXiv:1909.03368); ATBench itself carries **no** dependency-edge annotation, so `Turn.consumes_from` is empty on this corpus and reproduction C's edge-decoding arm is out of scope here, not silently faked |

Reproduction A is the one the rest of this README is really about, because an
early-abort cascade is *defined* by operating at high precision on a small
high-confidence subset — which turns out to be exactly where this corpus's worst
leak lives (Section 3). It has now run, and the short version is: real signal,
still a loss (Section 7) against its own pre-registered bar (Section 8).

---

## 2. The corpus, honestly

Loaded via `data.ATBenchLoader` (`types.TrajLoader`). Measured on the run that
built `artifacts/corpus_atbench_n500_s0_t16.json.gz`, 2026-08-29:

- **997 trajectories loaded** (500 safe / 497 unsafe). `N_PER_CLASS=500` asks for
  500 of each; the unsafe pool is 3 short, and the loader records the **requested**
  and **achieved** counts separately rather than reporting the request as if it
  had been delivered.
- **The label is an outcome, not a marker of risk presence.** Only 248 of the 500
  safe trajectories are tagged `risk_source=benign` — the other 252 carry a real
  threat tag (a genuine risk was present in the scenario) but the episode still
  resolved safely. A probe that keys off "was there a threat" rather than "did it
  go wrong" is answering a different question than the label asks.
- **No grouping field exists**, and that is not a defect to work around: 1000/1000
  unique ids, 1000/1000 unique first-user messages, 998/1000 unique tool suites
  (the one 3-row tool-suite family is three unrelated scenarios that happen to
  reuse two tools). So **one group per trajectory is the correct choice, not a
  degenerate one** — `ActivationBundle` rows are *turns*, and grouping by
  trajectory is what stops turn 3 of an episode training a probe that is then
  scored on turn 7 of the *same* episode. See `config.GROUP_BY` and the GROUPING
  section of `data.py`.

### Anchor check: the parser reproduces the paper's own numbers

ATBench's arXiv:2604.02022 abstract states the corpus is "1,000 trajectories
(503 safe and 497 unsafe), averaging 9.01 turns" — and on a full, uncapped load
of the 1,000-row pool our loader independently measures **503 safe / 497
unsafe** and a mean of **9.009 turns** (delta 0.001). The 997-trajectory
sampled corpus this lesson actually runs on gives **9.008** -- the two differ
because `N_PER_CLASS=500` short-falls the unsafe class by 3, so quote whichever
pool a number came from rather than treating them as one. (`data.py`'s `PAPER_STATS` +
`verify_against_paper` record this comparison; it is not merely a coincidence
worth citing, it is a check on the parser itself.)

Why it is a real check and not a formality: ATBench's turn schema comes in two
shapes that never mix — `{role, content}` for user/environment turns, and
`{role, thought, action}` for **agent** turns, which carry **no `content` key
at all**. A parser that reads `turn["content"]` and quietly skips a
`KeyError` drops every agent turn — 4,492 of the corpus's 9,009 turns, and
precisely the half that carries the failure signal — and would report a mean
turn count near 4.5, not 9.01. Reproducing the published mean is therefore
direct evidence the agent turns survived parsing, using a statistic the paper
itself published, not a synthetic sanity check invented after the fact. (We do
not compare token counts: the paper's "3.95k tokens" is not something we have
measured, and character counts are not tokens.)

---

## 3. The section that matters most: a perfect, invisible, local leak

Every reported AUC in this series must survive two controls beside the raw
number (`types.ProbeResult`): step-index residualisation and a bag-of-words
content bar. Building the leakage gate (`leakage.py`) surfaced a defect that
**neither control catches**, measured on all 997 loaded trajectories, 2026-08-29:

- longest **safe** trajectory: **18 turns**. Longest **unsafe** trajectory: **62
  turns**.
- So **every turn at 0-based step index >= 18 is unsafe by construction** — its
  label follows from its position, before any activation is read.
- **32 trajectories** (all unsafe) run past 18 turns, contributing **570 of 8,981
  turn-rows (6.3%)**, and that region is **100% label=1**.
- The leak is **invisible globally**: step-index AUC over the *whole* corpus is
  **0.5686**, and fractional position (step / trajectory length) is **exactly
  0.5000**. Both look harmless.
- But it is **local and perfect**: `step >= 8` gives `P(unsafe) = 0.633`;
  `step >= 16` gives `0.997`; `step >= 24` gives `1.000`.
- **`StepResidualiser` (CONTROL 1) cannot remove this.** It regresses out
  *linear* step-index information; this leak is a *threshold*, not a trend.
  `probes.py`'s own self-test measures exactly this gap: a planted quadratic
  position code survives degree-1 residualisation at AUC **0.6232**. A
  probe reading this region would sail through CONTROL 1 and look like a
  finding.

The fix had to be in the **data**, because the control cannot reach it:
`config.MAX_TURNS` defaults to **16**, not 0 (unlimited), and `data.load_corpus`
gates on it via `leakage.assert_no_deterministic_region` so the corpus cannot be
loaded uncapped by accident. Verified both ways:

| corpus | rows | step-index AUC | deterministic region |
|---|---|---|---|
| uncapped (`MAX_TURNS=0`) | 8,981 | 0.5686 | present, 570 rows (6.3%), 100% label=1 |
| capped at 16 (`MAX_TURNS=16`) | 8,341 (92.9% kept) | 0.5037 | **empty** |

The sharp part: dropping only 6.3% of rows takes step-index AUC all the way to
chance. The global 0.5686 figure **was** the deterministic region, diluted by the
93.7% of rows that carry no position signal at all — not a weak, diffuse
correlation spread evenly across the corpus. And the cap is not merely a
concession to the leak: reproduction A's own claim is that failure is
predictable *early*, so evaluating on early turns is the faithful test of it, not
a weakened one. It bites reproduction A hardest for exactly that reason — a
cascade is defined by operating at high precision on a subset, and that subset is
where the region lives.

---

## 4. Other measured facts, reported the way they measured, not the way they first looked

- **Trajectory length is a weaker confound than the class means suggest.**
  Character-count AUC is **0.5930**, turn-count AUC is **0.5095** — modest, not
  the story. The class means (unsafe 17,101 chars vs safe 3,881 chars) are
  **outlier-driven**: the **medians** are close (3,948 vs 3,582). Reporting the
  mean here would have implied a length confound the AUC does not support; say
  plainly that the mean was the misleading statistic.
- **The label is trajectory-level only.** ATBench carries no per-step
  annotation — `mistake_step` / `mistake_agent` are `None` on every trajectory,
  and `TrajCorpus.step_label_provenance` says so explicitly. Any claim about
  *which turn* went wrong is a claim about a label this corpus does not have.

---

## 5. Pipeline, file by file

| file | role |
|---|---|
| `types.py` | the fixed spine: `Turn`, `AgentTrajectory`, `TrajCorpus`, `ActivationBundle`, `ProbeResult`, `CascadeResult`, and the `TrajLoader` / `ActivationExtractor` / `TrajProbe` protocols. Owned by the lead; sub-lessons implement against it, never edit it. |
| `config.py` | the config anchor — corpus choice, model id/layer, sampling, `MAX_TURNS`, and every artifact path via `common.artifact_paths.keyed_path` (never a bare `results.json` — see `meerkat` / `rogue_scalpel` / `cross_trajectory`'s near-misses with exactly that bug). |
| `data.py` | `ATBenchLoader`: downloads/parses ATBench's nested `contents` field (agent turns carry `thought`+`action`, never a bare `content` key — a naive parser silently drops half the corpus), builds `AgentTrajectory` objects, caches by pool fingerprint, refuses a disagreeing cache. |
| `leakage.py` | measures and gates the step-index deterministic region described in Section 3. A build-time check, not a runtime hope. |
| `probes.py` | `LinearTrajProbe`, `StepIndexProbe`, group-aware CV, the clustered bootstrap CI, and all three controls (step-index residualisation, content bar, Hewitt-Liang random-label ceiling). |
| `activations.py` | `HFActivationExtractor`: one causal forward pass per trajectory (not one per prefix — see the module docstring's O(n) vs O(n^2) argument, and `verify_prefix_equivalence`, which *measures* causality rather than assuming it), a resumable row journal (a reap costs one trajectory, not the run), and a two-fingerprint cache (behaviour + data). |
| `extract_acts.py` | the GPU half, split out on its own so a reap is cheap (~10s/trajectory, ~2.8h for the full 997): loads the corpus, runs `HFActivationExtractor`, journals every trajectory as it completes, caches by (behaviour, data) fingerprint. |
| `run_traj_probes.py` | reproduction A/B: fits all four arms (`linear`, `step_only`, `content_bar`, `random_label`) on one cached activation bundle and one fixed set of group-aware CV folds, then the early-abort curve at k in (1,2,3,4,6,8) reusing that SAME fold assignment (recomputing folds per k would silently compare different train/test partitions and call the difference "early abort"). Writes `artifacts/results_atbench_<model>_L<layer>.json`. |
| `sweep_layers.py` | rules the layer axis in/out before the layer-12 negative is reported as a property of the residual stream rather than of the layer: one model load per layer, journalled, scored against the SAME trajectory-level content bar. Writes `artifacts/layer_sweep_atbench_<model>.json`. |

The "Run it" commands below are the self-tests of the individual modules; the
end-to-end GPU commands (extract → fit → sweep) are given separately at the end
of this section.

---

## 6. Run it

Every module is CPU-only to import and self-test — none of them load a model
except `activations.py`, and its self-test uses a tiny synthetic toy model, not
Gemma. All commands run from the repo root with the pinned interpreter.

```
# fixed spine: dataclasses + protocols
C:/Users/evija/anaconda3/python.exe -m steering_tutorials.traj_probes.types

# config anchor: keyed artifact paths, env overrides, preflight
C:/Users/evija/anaconda3/python.exe -m steering_tutorials.traj_probes.config

# the loader: downloads a small (n=25/class) smoke pool, exercises the cache
C:/Users/evija/anaconda3/python.exe -m steering_tutorials.traj_probes.data

# the leakage gate: plants the ATBench-shaped long-tail leak and shows the
# residualiser cannot remove it, then shows capping turns does
C:/Users/evija/anaconda3/python.exe -m steering_tutorials.traj_probes.leakage

# the probe controls: group-aware CV, CONTROL 1/2/3, clustered bootstrap CI
C:/Users/evija/anaconda3/python.exe -m steering_tutorials.traj_probes.probes

# the extractor: prefix-equivalence proof, resumable journal, cache fingerprints
C:/Users/evija/anaconda3/python.exe -m steering_tutorials.traj_probes.activations
```

To materialise the full pinned corpus (500/class request, capped at
`MAX_TURNS=16`) without running Gemma at all:

```
C:/Users/evija/anaconda3/python.exe -c "from steering_tutorials.traj_probes.data import load_corpus, summarise; print(summarise(load_corpus()))"
```

The GPU pass has run. To reproduce it end to end (extract, then fit, then the
optional layer sweep):

```
# extract Gemma-3-1B layer-12 activations over the pinned corpus (foreground,
# resumable — a reap costs one trajectory, not the run)
C:/Users/evija/anaconda3/python.exe -u -m steering_tutorials.traj_probes.extract_acts

# fit all four arms + the early-abort curve, write results_atbench_*.json
C:/Users/evija/anaconda3/python.exe -u -m steering_tutorials.traj_probes.run_traj_probes

# layer sweep (layer 4 done; TP_SWEEP_LAYERS=4,8,16,20,24 to pick layers)
C:/Users/evija/anaconda3/python.exe -u -m steering_tutorials.traj_probes.sweep_layers
```

Results: `artifacts/results_atbench_gemma-3-1b-it_L12.json`,
`artifacts/content_bar_by_k.json`, `artifacts/layer_sweep_atbench_gemma-3-1b-it.json`.
See Section 7.

---

## 7. Results: the probe is real, it still loses, and the margin is SMALL

> **All numbers below are measured at the corrected config**
> (`MAX_TURN_CHARS=8000`, `MAX_TOKENS=16384`; 1.46% of turns clipped, 0 rows
> dropped). An earlier pass ran at `MAX_TURN_CHARS=1200`, which clipped ~10% of
> turns; those results were **discarded, not reported**. See bug 4 in
> "What broke". The correction moved the headline materially, and in the
> direction that should worry an author: the probe got **better** (0.7453 ->
> 0.8070 at the trajectory unit) and the bar got **worse** (0.8671 -> 0.8418),
> so most of the gap I first measured was my own truncation, not the method.

**The binding comparison, at the matched trajectory unit:**

| quantity | value |
|---|---|
| activation probe (trajectory, max-pooled) | **0.8070** |
| TF-IDF unigram content bar (same text) | **0.8418** |
| margin | **-0.0347** |
| paired bootstrap on the margin, 10k resamples of trajectories | mean **-0.0345**, 95% CI **[-0.0656, -0.0036]** |
| P(probe > bar) | 0.014 |
| clears_content_bar | **False** |

The CI excludes zero, so the bar really does beat the probe -- but its upper
bound is **-0.0036**, a hair from zero. At `c1200` the same margin read -0.1218.
**The effect is real and it is small**, and anyone quoting the earlier -0.12
would be quoting an artifact of over-clipping.

### The four arms (row-level headline, for reference)



Gemma-3-1B, layer 12, ATBench, 8,341 turn-rows over 997 trajectories, group-aware
5-fold CV, 594.5s wall clock, 2026-08-29
(`artifacts/results_atbench_gemma-3-1b-it_L12.json`). **This is a negative
result and it is stated as one here, not in Section 8.**

| arm | AUC | 95% CI |
|---|---|---|
| linear activation probe (raw, row-level) | 0.7694 | [0.7526, 0.7859] |
| after step-index residualisation (CONTROL 1) | 0.7681 | unchanged by residualisation |
| step-index floor (`StepIndexProbe`) | 0.5036 | [0.4871, 0.5208] |
| random-label ceiling (CONTROL 3, Hewitt-Liang) | 0.5124 | — |
| TF-IDF unigram content bar (CONTROL 2) | 0.8418 | — |
| `clears_content_bar` | **FALSE** | — |

The probe measures something real, not an artefact of the two things most
likely to fake one: it is not **capacity** (the random-label ceiling sits at
0.5124, near chance, while the real probe scores 0.7694 — far outside that
ceiling's own CI width); and it is not **position** (residualising out linear
step-index moves the AUC from 0.7694 to 0.7681, i.e. not at all — the
`MAX_TURNS=16` cap from Section 3 already did its job, since the step-index
floor itself sits at chance, 0.5036). It is simply **worse than unigrams over
the same text**: 0.7694 vs 0.8418. The one apples-to-apples unit match the
falsifier actually calls for — content bar is trajectory-level, so the linear
probe must be pooled to trajectory level too (`max`-pooled, `auc_residualised`'s
sibling in the results file) rather than compared at its raw row-level headline
— gives **0.8070 vs 0.8418**, margin **-0.0347**. A paired bootstrap over
trajectories (10k resamples) puts that margin at mean **-0.0345**, 95% CI
**[-0.0656, -0.0036]**, P(probe > bar) = 0.014. The CI excludes zero, so the bar
genuinely wins -- but its upper bound is a hair from zero, and the same margin
read **-0.1218** before the truncation was corrected. The verdict survives; the
*size* of it did not.

**The early-abort curve**, each point benchmarked against a content bar computed
on the SAME first-k turns (`artifacts/content_bar_by_k.json`), so no cell is
compared against a bar that read more text than the probe did:

Both sides are at the **trajectory** unit, so each cell is a like-for-like
comparison:

| k (turns visible) | 1 | 2 | 3 | 4 | 6 | 8 | 16 |
|---|---|---|---|---|---|---|---|
| probe AUC (trajectory) | 0.5733 | 0.6128 | 0.6227 | 0.6410 | 0.7353 | 0.7999 | **0.8070** |
| content bar AUC | 0.7272 | 0.7438 | 0.7392 | 0.7784 | 0.8331 | 0.8462 | 0.8418 |
| margin | -0.154 | -0.131 | -0.117 | -0.137 | -0.098 | **-0.046** | **-0.035** |

The probe still loses at every k. But the margin **shrinks monotonically with
horizon** from k=4 onward: -0.137 -> -0.098 -> -0.046 -> -0.035. The residual
stream closes most of its gap on the surface as the trajectory gets longer,
which is exactly the regime this lesson exists to study.

**This trend was invisible at the over-clipped config, and I reported its
absence as a finding.** The `c1200` README said the margin was "stable -0.117 to
-0.169 at every k -- not an aggregate effect hiding a horizon where it wins".
That sentence is now **withdrawn**: it was false, and it was false *because of
the cap*. A per-turn character cap clips long trajectories hardest -- they have
both more turns and the giant tool dumps -- so over-clipping damaged precisely
the long-horizon cells where the probe's advantage accrues, and flattened the
trend into a flat line. That is a mechanism consistent with the data, not a
separately verified claim; what is verified is that the trend appears once the
cap is loosened.

Read plainly: on this corpus the internal state buys little over unigrams for
short trajectories, and closes to near-parity by 16 turns. Whether it would
*overtake* the bar past 16 turns is **not measured here** -- `MAX_TURNS=16` is
capped for the leak reason in Section 3, so the question is open, and it is the
obvious next experiment rather than an implication of this table.

**What does and does not reproduce.** The paper's *qualitative* claim
reproduces: failure is predictable above chance from the very first turn
(0.5733, k=1) and rises monotonically with horizon. What does not reproduce is
the residual stream beating the surface — every paper here reports the probe
as the interesting number; on this run the unigram bar is the better detector
at every horizon.

**Do not read this as a refutation of arXiv:2607.06503.** That paper runs
Qwen-2.5-7B / Qwen3-32B / Llama-3.3-70B over TextCraft and WebShop; this run is
Gemma-3-1B, layer 12, last-token pooling, over ATBench. Model scale, layer,
pooling, or corpus could each carry the reproduction on their own. **One of the
four is now ruled out**: the layer sweep below covers six depths and every one
falls below the bar. Model scale, pooling and corpus remain untested. The
honest statement is **"we cannot claim the reproduction here,"** not "the
method fails." This README will not print these numbers beside the papers'.

### Known limitations of this run

- **One model.** Gemma-3-1B-it, 1B parameters, against 7B–70B in the
  reproduction papers.
- **One pooling scheme.** Last-token activation at each turn boundary; no other
  pooling (mean, max) has been tried.
- **One corpus.** ATBench only; the reproduction papers use TextCraft, WebShop,
  ALFWorld and tau-bench.
- **The layer axis is ruled out.** Six layers, each scored against the SAME
  layer-independent content bar (0.8418) and re-scored at the matched
  trajectory unit
  (`artifacts/layer_sweep_unitfixed_atbench_gemma-3-1b-it.json`):

  | layer | 4 | 8 | **12** | 16 | 20 | 24 |
  |---|---|---|---|---|---|---|
  | probe AUC (trajectory) | 0.7467 | 0.7753 | **0.8070** | 0.7817 | 0.7810 | 0.7731 |
  | margin vs bar | -0.095 | -0.067 | **-0.035** | -0.060 | -0.061 | -0.069 |

  Every layer is below the bar, so the negative is **not** an artefact of
  probing the wrong depth. Layer 12 — the config default, which this README
  previously flagged as "a default, not a finding" — turns out to be the best
  of the six, so the headline was not depth-unlucky either.

  Note this is a *different shape* from the over-clipped run, which produced a
  flat profile (spread 0.041, no clear peak). At the corrected config the
  spread is 0.060 with a distinct peak at layer 12. The truncation was
  flattening the layer structure as well as the horizon trend.

### What broke, and how it was caught

Four instrument bugs surfaced during this run. None of them crashed — each
produced confident, well-formed, wrong output, the same shape as every prior
defect in this program (Section 18.8 of the project's `CLAUDE.md`):

1. **The 4,096-token extraction cap silently dropped rows in a
   label-correlated way.** 327 unsafe rows were truncated away against 13 safe
   — from trajectories that were 86% unsafe against a 49.8% corpus base rate,
   i.e. precisely the late turns of the failing class. The run's own log said
   "28 turns truncated," understating a 340-row loss by 12x. Fixed by moving
   the cap into the loader as `config.MAX_TURN_CHARS` (so the model and the
   content bar read the exact same string), at which point 0 rows are dropped.
2. **That fix appeared to do nothing.** `ATBenchLoader.cache_path` was keyed on
   `(corpus, n, seed, max_turns)` and not on the new character cap, so it kept
   serving a pre-cap cache while the config stamp printed
   `MAX_TURN_CHARS=1200` right beside it.
3. **The early-abort curve was silently empty at every k.** `fold_of` was built
   keyed by `traj_uid` (e.g. `'1'`) but read by `group_id` (e.g.
   `'atbench::1'`) — a total key mismatch — and the failure reported itself as
   "fewer than 2 usable folds," which reads like sparse data, not a bug. Its
   own self-test passed *with the bug present*, because the synthetic fixture
   set `traj_uid == group_id` and so was structurally incapable of catching a
   key-format mismatch. A total key miss now raises instead of silently
   returning nothing.
4. **The fix for (1) was itself set far too aggressively, and that was the
   worst of the four** — because it did not merely lose data, it could have
   biased the headline comparison in the probe's disfavour. The first cap was
   `MAX_TURN_CHARS=1200`, chosen only so trajectories would fit a **hard-coded**
   4,096-token `ExtractSettings` default that was never a config knob, never
   stamped, and never swept. Per-turn length is p50 347 / p90 1,088 / p99
   50,915 / max 53,255 chars, so a 1,200 cap clips roughly **10% of all turns**
   when its only target was a handful of giant tool dumps. That is data
   truncation for its own sake, and it is not symmetric between the two arms:
   the probe reads the **last token** of each turn, so clipping changes what it
   reads, while TF-IDF still recovers the discriminative unigrams from the
   surviving text. A negative result measured that way is partly an artifact of
   the cap. Corrected: `MAX_TURN_CHARS=8000` with `MAX_TOKENS=16384` (now a
   real, stamped, fingerprinted config value), which clips **1.46% of turns
   (122 of 8,341)**, drops no rows, and sits well inside the model's 32,768
   context. **Every number in Section 7 is measured at the corrected config.**
   The `c1200` results were discarded, not reported.

The common shape: right output format, no error, wrong content — the failure
mode this whole program keeps finding, never the loud kind. Bug 4 adds a second
lesson the others do not carry: a *control* can be over-tightened until it
damages the thing it was protecting, and a cap chosen to satisfy a hidden
default is a cap chosen for the wrong reason.

---

## 8. What would falsify this (pre-registered)

- **Reproduction A/B fail** if `LinearTrajProbe`'s `auc_residualised` at early
  step indices (k <= 4, on the `MAX_TURNS=16`-capped corpus, so Section 3's
  region cannot contaminate the read) does not clear `StepIndexProbe`'s own AUC
  by a margin outside the bootstrap CI. If the residualised probe is
  indistinguishable from a probe that only ever sees the step counter, the
  residual stream has added nothing reproduction A/B did not already get from
  position. **Did not fire**: at k=1 the probe already reads 0.5741, outside
  the step-index floor's CI ([0.4871, 0.5208]); the gap widens with k
  (Section 7).
- **The content bar sinks the claim** if `content_bar_control`'s bag-of-words AUC
  (pooled to one score per trajectory, the same unit the activation probe is
  compared at) matches or beats the activation probe's residualised AUC. A
  "trajectory" probe that cannot beat unigrams over the same text has not shown
  the internal state carries anything the surface didn't. **FIRED**:
  trajectory-pooled probe 0.8070 (row-level headline 0.7694) vs bar 0.8418, at
  every k in the early-abort curve (Section 7). This is the headline verdict of
  this lesson.
- **The random-label ceiling disqualifies a raw number** if `LinearTrajProbe`'s
  real AUC does not clear `random_label_control`'s ceiling by more than the
  ceiling's own CI width — that would mean the probe's capacity alone explains
  the score at this n, independent of any real signal. **Did not fire**: 0.7694
  vs a ceiling of 0.5124 (Section 7) — capacity alone does not explain this
  probe's score.
- **Reproduction C is not evaluable at all here** unless a future pass adds
  synthetic (and honestly-labelled-as-invented) tool-dependency edges — ATBench
  itself carries none, and `Turn.consumes_from` must not be filled from
  argument-string pattern matching, which would be inventing the very label
  being measured (`data.py`'s `_render_turn` docstring is explicit about this).
  Still out of scope; unchanged by this run.

---

## 9. Repository

Lesson package: [`steering_tutorials/traj_probes/`](.)
Sibling (text-embedding side of the same family):
[`streaming_trajectory_aggregation`](../streaming_trajectory_aggregation/README.md)
Course map: [`steering_tutorials/README.md`](../README.md)
