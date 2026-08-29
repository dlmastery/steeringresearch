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

> **Status: no results yet.** The data loader, the leakage gate, the probe
> controls and the activation extractor are built and CPU-self-tested. The GPU
> pass (extract Gemma-3-1B activations over the pinned corpus, fit the probes,
> write `artifacts/results_*.json`) has **not been run**. Every number below that
> would come from that pass is marked `[PENDING RUN]`. Nothing in this file is
> invented to fill that gap.

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
leak lives (Section 3).

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

No orchestrating runner (`run_traj_probes.py` or similar, tying load → extract →
probe → `results_*.json` into one command) exists yet in this package as of this
writing. The "Run it" commands below are the self-tests of the modules that do
exist.

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

The GPU pass — `HFActivationExtractor(config.MODEL_ID).extract(corpus, config.LAYER)`
followed by `LinearTrajProbe().fit_predict_cv(bundle)` at every step k — is
**`[PENDING RUN]`**; no `artifacts/results_*.json` exists yet.

---

## 7. What would falsify this (pre-registered)

- **Reproduction A/B fail** if `LinearTrajProbe`'s `auc_residualised` at early
  step indices (k <= 4, on the `MAX_TURNS=16`-capped corpus, so Section 3's
  region cannot contaminate the read) does not clear `StepIndexProbe`'s own AUC
  by a margin outside the bootstrap CI. If the residualised probe is
  indistinguishable from a probe that only ever sees the step counter, the
  residual stream has added nothing reproduction A/B did not already get from
  position.
- **The content bar sinks the claim** if `content_bar_control`'s bag-of-words AUC
  (pooled to one score per trajectory, the same unit the activation probe is
  compared at) matches or beats the activation probe's residualised AUC. A
  "trajectory" probe that cannot beat unigrams over the same text has not shown
  the internal state carries anything the surface didn't.
- **The random-label ceiling disqualifies a raw number** if `LinearTrajProbe`'s
  real AUC does not clear `random_label_control`'s ceiling by more than the
  ceiling's own CI width — that would mean the probe's capacity alone explains
  the score at this n, independent of any real signal.
- **Reproduction C is not evaluable at all here** unless a future pass adds
  synthetic (and honestly-labelled-as-invented) tool-dependency edges — ATBench
  itself carries none, and `Turn.consumes_from` must not be filled from
  argument-string pattern matching, which would be inventing the very label
  being measured (`data.py`'s `_render_turn` docstring is explicit about this).

---

## 8. Repository

Lesson package: [`steering_tutorials/traj_probes/`](.)
Sibling (text-embedding side of the same family):
[`streaming_trajectory_aggregation`](../streaming_trajectory_aggregation/README.md)
Course map: [`steering_tutorials/README.md`](../README.md)
