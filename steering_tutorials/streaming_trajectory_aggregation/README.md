# streaming_trajectory_aggregation (STA) — how do you aggregate 126 steps of an agent trajectory, and can you flag the scheming before it lands?

> **References** (every id below was WebFetch-verified in
> [`CITATIONS_VERIFIED.md`](CITATIONS_VERIFIED.md) — title, authors and subject class read off
> `arxiv.org/abs/<id>` itself, never from a search snippet; that file also records **two
> DO-NOT-CITE traps** that resolve to unrelated fields):
>
> - [Trajectory Guard -- A Lightweight, Sequence-Aware Model for Real-Time Anomaly Detection in Agentic AI (arXiv:2601.00516)](https://arxiv.org/abs/2601.00516)
>   — Laksh Advani, AAAI TrustAgent 2026, submitted 2 Jan 2026. *Relevance:* states the claim
>   this entire ladder exists to test, verbatim in its own abstract — *"mean-pooling embeddings
>   dilutes anomalous steps, while contrastive-only approaches ignore sequential structure."*
>   Its architecture is a Siamese Recurrent Autoencoder (F1 0.88-0.94, 32 ms); our `gru` is a
>   plain 1-layer GRU, **inspired-by, not a reproduction**.
> - [Tracing the Dynamics of Refusal: Exploiting Latent Refusal Trajectories for Robust Jailbreak Detection (arXiv:2605.02958)](https://arxiv.org/abs/2605.02958)
>   — Xulin Hu, Che Wang, Wei Yang Bryan Lim, Jianbo Gao, Zhong Chen, 2 May 2026. *Relevance:*
>   the mean-pooling **prediction** this lesson checks. **The supporting text is BODY text, not
>   the abstract** — Sec. 5.4 (Ablation Study): *"Mean pooling dilutes this strong 'needle' into
>   the vast 'haystack' of irrelevant background noise"*; Table 2 caption, which makes
>   **global max-pooling the paper's own sparsity-aware default** and mean the ablation *of* it;
>   Appendix D: *"sequence-level mean aggregation can achieve high recall on several attack sets
>   but yields near-random XSTest AUROC."* Anyone checking this citation against the abstract
>   alone will not find it. (Inspired-by; we do not implement SALO.)
> - [Real-Time Detection and Repair of LLM Agent Failures (arXiv:2608.02464)](https://arxiv.org/abs/2608.02464)
>   — Sunny Dubey, submitted 3 Aug 2026. *Relevance:* the mechanism behind our `esn_cusum`
>   monitor — a one-class echo-state-network **ensemble** with CUSUM alarms, AUROC 0.872 at a 5%
>   false-alarm budget, ~200 microseconds/step. **The paper is NOT called
>   "AgentTrajectorySentinel"**; that name appears in the brief that spawned this lesson and is
>   not this paper's title. Our monitor is a **single** reservoir over embeddings, not an
>   ensemble over telemetry.
> - [SafetyDrift: Predicting When AI Agents Cross the Line Before They Actually Do (arXiv:2603.27148)](https://arxiv.org/abs/2603.27148)
>   — Aditya Dhodapkar, Farhaan Pishori, submitted 28 Mar 2026. **Submitted to COLM, NOT
>   accepted** — never describe it as a COLM paper. *Relevance:* the mechanism behind our
>   `safetydrift` monitor — agent safety modelled as an absorbing Markov chain, catching 94.7% of
>   violations with 3.7 steps of advance warning. Our adaptation departs from it in three ways,
>   one of them fatal to the mechanism (see [Known defects](#8-known-defects-open-not-fixed)).
> - [TRACE: Trajectory Risk-Aware Compression for Long-Horizon Agent Safety (arXiv:2606.00611)](https://arxiv.org/abs/2606.00611)
>   — Zhepei Hong, Lin Wang, Liting Li, Haokai Ma, Junfeng Fang, Fei Shen, Dan Zhang, Xiang Wang,
>   submitted 30 May 2026. *Relevance:* the mechanism behind `query_token_compressor` — compress
>   a long trajectory into a compact latent evidence state, then judge from it (+12.6 pp on
>   ASSEBench / Pre-Ex-Bench / R-Judge). We implement the Compressor half only; **there is no
>   Reader**, and **K=16 is a harness convenience, not the paper's stated value** (K=16 does not
>   appear in its abstract).
> - [TRACE: Trajectory Reasoning through Adaptive Cross-Step Evidence Aggregation for LLM Agents (arXiv:2606.07054)](https://arxiv.org/abs/2606.07054)
>   — Vijitha Mittapalli, Shreyaa Jayant Dani, Satya Srujana Pilli, Snigdha Ansu, Mohammadreza
>   Teymoorianfard, Franck Dernoncourt, Hongjie Chen, Yu Wang, Ryan A. Rossi, Nesreen K. Ahmed,
>   submitted 5 Jun 2026. **A different system that shares the TRACE acronym with the paper
>   above** — always disambiguate by full title + id. *Relevance:* establishes SHADE-Arena as the
>   benchmark for exactly this problem (aggregate F1 0.713, recall 0.844 across ten domains), and
>   frames the failure mode our corpus embodies: a hidden malicious objective pursued through
>   individually benign actions.

---

## Read this first

**This is a SCREENING-tier result: one corpus, one embedder, one seed, five CV groups.** Under
CLAUDE.md §7 nothing here is EVALUATION-tier and no sentence below uses the word "winner".

Four of the five applicable pre-registered falsifiers HOLD. But the four honest sentences that
matter more than the verdicts are:

1. **A content-only rival is about as good as the best deployable monitor.** The binding confound
   bar — a bag-of-words model on the trajectory text — is **0.9039**. Exactly one *causal*
   method reaches it: `gru` at **0.9051**, clearing by **0.0012**, with a 95% CI
   [0.8859, 0.9236] that **straddles the bar**. The one method that clearly beats the bar,
   `query_token_compressor_k16` at 0.9392, is **`is_causal=FALSE`** — it attends over the whole
   trajectory including the future, so it is **not a deployable streaming monitor**.
2. **The streaming half of the lesson failed.** Both causal monitors score at chance
   (`esn_cusum` 0.511, `safetydrift` 0.459 against a 0.904 bar). F4 FAILS, and it fails hard.
3. **F5 "HOLDS" is a hollow pass.** The 11-step median lead is computed over **3 detected
   positives out of 132 eligible**, at a **0.294 false-alarm rate**. `safetydrift` detected
   **zero** and is flagged `unreliable_calibration`. A monitor already shown to be at chance by
   F4 cannot furnish a meaningful lead time; F5's threshold ("median lead > 0") was simply too
   weak to notice.
4. **The central claim contradicts this repo's own prior result**, which is why it was
   pre-registered as under test rather than assumed. See §5.

---

## 1. What this lesson is

The trajectory-detection family in this course has climbed from turns to tokens to agents to
many-traces. This is the long-horizon rung: **one agent, one task, ~126 steps, a covert
secondary objective**, and two questions that only appear at that length.

| question | who answers it here |
|---|---|
| Does mean-pooling *dilute* a sparse anomalous step as the horizon grows? | the 8-method offline aggregator ladder |
| Can a **causal** monitor — one that has only seen steps `0..t` — flag it **before** the point of no return? | the two streaming monitors + the lead-time arm |

The second question is what makes this corpus special: SHADE is the only pairing in
[`DATASETS_VERIFIED.md`](DATASETS_VERIFIED.md)'s sweep with per-step judge labels **and** an
oracle flagging step **and** a point-of-no-return step, so `lead = point_of_no_return_step -
ideal_flagging_step` is directly computable rather than inferred.

**Predecessors:** `trajguard` (streaming, across generated tokens) and `cross_trajectory`
(aggregation, across agents). This lesson is streaming *and* aggregation, across steps.

---

## 2. The corpus

`adityaasinha28/control_arena_shade` (1500 rows, full trajectories) joined by row index to
`aksh-n/control_arena_shade-monitor-labels` (per-row Gemini-3.1-Pro judge logs).

| property | value |
|---|---|
| admitted | **1000 trajectories, 500 positive / 500 negative** (clears CLAUDE.md §17 rule 1) |
| dropped | 75 (`verification.passed == False`), 0 with no usable per-step judge records |
| steps per trajectory | min 9 · **median 126** · mean 125.6 · max 422 |
| CV groups | **`n_distinct_groups = 5`** |
| licence | **unstated** by both repos (the upstream SHADE-Arena *code* is MIT; these community re-dumps assert nothing) |
| label provenance | **`gemini-3.1-pro-preview` judge log — LLM-generated, not human ground truth** |
| pool fingerprint | `7c7d372875af1a0e...` (stamped in `results_shade.json`) |

Two of those rows are load-bearing caveats, not metadata:

- **The GroupKFold rests on five groups.** With `n_distinct_groups = 5` and `N_FOLDS = 5`, each
  fold holds out one group. Group-aware CV is doing real work (no straddles were measured), but
  five is a small number of independent units for a 1000-trajectory result, and every CI below
  should be read with that in mind.
- **Every per-step label is an LLM judge's output.** Gemini is off-family from Gemma
  (§17 rule 3), which is necessary but not sufficient: lead time is measured against a judge
  **oracle**, never against truth. `results_shade.json` carries
  `"oracle_note": "lead measured against a judge oracle, NOT ground truth"` on every lead-time
  row for exactly this reason.

---

## 3. The confound bar — read this before any AUC

Every headline AUC is priced against `binding_bar = max(full_trajectory.worst_auc,
first_step_rival.content.auc)`, computed **before** the ladder runs.

| bar | AUC | what it is |
|---|---|---|
| character length | 0.5952 | directionless; can raw length separate the classes? |
| step count | 0.5431 | directionless; can trajectory *length in steps* separate them? |
| **content (bag-of-words)** | **0.9039** | **the binding bar** — a TF-IDF model on the trajectory text |
| label shuffle | 0.5217 | the leakage control (F0) |
| first-step rival | 0.6826 | a bag-of-words model on the **first step only** — a rival method, not a bar to fold in quietly |

**0.9039 is a high bar and it is the whole story of §5.** A trajectory-aware aggregator that
lands at 0.90 has not demonstrated that trajectory structure matters; it has demonstrated that
it matched a model which never looked at the structure at all.

---

## 4. Results — the offline ladder

SHADE, n=1000, 5-fold group-aware CV, 10,000-resample bootstrap CI, seed 0,
EmbeddingGemma-300m (dim 768). Wall clock 5362 s.

| method | AUC | 95% CI | us/step | causal? | vs bar 0.9039 |
|---|---|---|---|---|---|
| `query_token_compressor_k16` | **0.9392** | [0.9253, 0.9526] | 8.33 | **NO** | +0.0353, CI clear of the bar |
| `gru` | **0.9051** | [0.8859, 0.9236] | 39.23 | yes | **+0.0012, CI straddles the bar** |
| `max_pool` | 0.8581 | [0.8344, 0.8811] | 2.82 | yes | below |
| `mean_max_std_pool` | 0.8532 | [0.8290, 0.8770] | 4.05 | yes | below |
| `last_step` (control) | 0.8427 | [0.8180, 0.8659] | 2.10 | yes | below |
| `deviation_weighted` | 0.7540 | [0.7241, 0.7839] | 7.15 | **NO** | below |
| `mean_pool` | 0.7397 | [0.7090, 0.7703] | 2.98 | yes | below |
| `first_4_steps` | 0.6366 | [0.6022, 0.6703] | 2.06 | yes | below |

**How to read this table.** The only row that clearly beats the content bar cannot be deployed:
`query_token_compressor_k16` is `is_causal=False`, cross-attending over the entire trajectory
including steps that have not happened yet. Strip it out and the best deployable monitor,
`gru`, is statistically indistinguishable from a bag-of-words model over the same text. **On
this corpus, at this scale, trajectory-aware aggregation has not yet earned its cost.**

### Horizon-truncation curve (`FirstKSteps(k)`)

| k | 2 | 4 | 8 | 16 | 32 | 64 |
|---|---|---|---|---|---|---|
| AUC | 0.632 | 0.637 | 0.560 | 0.600 | 0.604 | 0.700 |

Non-monotone, and every point far below the full-trajectory ladder. The signal is **not** in an
early prefix — which is consistent with covert scheming that only becomes visible late, and is
also the reason the lead-time result is as thin as it is. Note the shape of this control before
over-reading it: see §8, defect (d).

---

## 5. The central claim — F1, and why it contradicts this repo

**F1 HOLDS.** `mean_pool` 0.7397 sits **0.1185 below `max_pool`** (0.8581) and **0.1654 below
`gru`** (0.9051), against a pre-registered margin of 0.02. On a median-126-step corpus, a true
mean over step embeddings loses a large amount of discriminative signal that a max over the same
embeddings keeps — exactly what arXiv:2601.00516's abstract and arXiv:2605.02958's Sec. 5.4
predict.

**This contradicts `cross_trajectory`, in this same course**, where `mean_agg` **topped** the
ladder at 0.9358, above `attn_pool` (0.8635) and `gnn_agg` (0.8121). That is why F1 was
pre-registered as genuinely under test rather than assumed. Two things reconcile the two
results, and both belong in the reading:

1. **`cross_trajectory`'s `mean_agg` is not a mean pool.** It concatenates `[mean, max, std]`
   (`cross_trajectory/models.py:198`) — a max term was never absent there. This lesson keeps
   `MeanPool` (a true mean) and `MeanMaxStdPool` (the exact `cross_trajectory` recipe) as two
   separately-scored rows precisely so that conflation cannot recur. Here
   `mean_max_std_pool` scores **0.8532** while the true `mean_pool` scores **0.7397** — the
   0.11 gap between them *is* the max term.
2. **The horizons are not comparable.** `cross_trajectory` aggregates K per-trajectory latents;
   this lesson aggregates ~126 per-step embeddings. Dilution is a function of how much you are
   averaging over.

**What F1 does NOT establish: that the HORIZON is the cause.** See §7.

---

## 6. The streaming half — F4 fails, F5 is hollow

| monitor | offline AUC of peak score | 95% CI | vs bar 0.9039 |
|---|---|---|---|
| `esn_cusum` | 0.5115 | [0.4754, 0.5475] | fails by 0.39 |
| `safetydrift` | 0.4593 | [0.4239, 0.4954] | fails by 0.44 — **below chance** |

**F4 FAILS.** Neither causal streaming monitor separates the classes at all. This is the
lesson's clearest negative and it should not be softened: the mechanisms adapted from
arXiv:2608.02464 and arXiv:2603.27148, ported from telemetry/hand-designed risk states onto
Gemma step embeddings, do not transfer to this substrate.

### Lead time (F5)

| monitor | eligible positives | detected | median lead | false-alarm rate | calibration |
|---|---|---|---|---|---|
| `esn_cusum` | 132 | **3** | 11.0 steps (leads: 18, 11, 1) | **0.294** | ok |
| `safetydrift` | 132 | **0** | n/a | 0.0 | **UNRELIABLE** |

**F5 is recorded as HOLDS and it is a hollow pass.** Say the whole sentence out loud: *the
median lead over the three positives that a chance-level monitor happened to alarm on, while
false-alarming on 29% of benign trajectories, was 11 steps.* That is not advance warning; it is
an artifact of a threshold. `safetydrift`'s threshold **ties** its train-benign peak max
(0.610636 over 313 benign trajectories with only **4 distinct values**), so its silence is
**by construction**, not evidence of separation — the harness detects this and stamps
`unreliable_calibration: true` rather than reporting a quiet monitor as a clean one.

The pre-registered falsifier asked only "is the median lead > 0". It should have required a
detection rate and a false-alarm budget. **Treat F5 as unanswered.**

A corpus-level caveat that was known before the run and still applies:
[`DATASETS_VERIFIED.md`](DATASETS_VERIFIED.md) measured `lead = ponr - ideal_flag` on a 188-row
sample at **median 0**, with the mean carried almost entirely by a five-point tail, and **60% of
positives offering zero or negative advance warning**. Advance warning on this corpus is scarce
before any monitor is involved.

---

## 7. Pre-registered falsifier verdicts

Registered **2026-08-11** in `config.py:FALSIFIERS`, evaluated on the 2026-08-17 run. No
verdict was redefined after the numbers came in.

| tag | verdict | number | reading |
|---|---|---|---|
| `F0_shuffle_control` | **HOLDS** | shuffle AUC 0.5217 in [0.45, 0.55] | no leakage in the confound instrument (but see §8(e) — it does not certify the ladder) |
| `F1_mean_pool_collapses_long` | **HOLDS** | mean 0.7397 vs max 0.8581 (margin 0.1185), gru 0.9051 | dilution is real on a 126-step corpus |
| `F2_mean_pool_survives_short` | **PENDING** | — | AgentDojo-only. **The AgentDojo run is in flight as of 2026-08-17; F2 is unanswered, not answered N/A.** |
| `F3_sequence_beats_laststep` | **HOLDS** | best 0.9392 vs last_step 0.8427 (margin 0.0966) | holds as written — but read the caveat below |
| `F4_streaming_clears_bars` | **FAILS** | esn_cusum 0.511, safetydrift 0.459 vs bar 0.9039 | the streaming half does not work |
| `F5_lead_time_positive` | **HOLDS (hollow)** | median 11.0 over **3/132** detections | see §6 |

**F3's caveat, derived by arithmetic from the table above (not a re-run).** F3's `best` is a max
over all eight ladder rows, which crosses two boundaries at once: it takes a **non-causal**
method, and it takes one from the **end-to-end-trained** arm to beat a control fitted with the
shared logistic-regression classifier. Restricted to the apples-to-apples comparison — best
pooled, shared-classifier, causal method vs `last_step` — the margin is
**`max_pool` 0.8581 - `last_step` 0.8427 = +0.0154**, which is **below the pre-registered 0.02
bar**. So the honest statement is: *a trained sequence encoder beats the last step; fixed
pooling over the trajectory does not clearly beat it.*

**What F1 + F2 could not have told you even if F2 had landed.** `types.py` and the run's console
output claim that "F1 holds + F2 holds means the horizon is the cause". **That inference is not
licensed by the design**, and this lesson's own fidelity audit says so
([`AUDIT_FIDELITY.md`](AUDIT_FIDELITY.md) §2.1). SHADE and AgentDojo hold only the label
machinery constant; task suite, tool inventory, environment, generating agent, step prose style,
base difficulty, and the *kind* of positive (a covert secondary objective vs an injected
instruction) all vary together with the horizon. A joint F1/F2 pass would be equally consistent
with "AgentDojo's prompt injections are blatant enough for any pooled representation" — nothing
to do with horizon length. The within-corpus fix (a `Truncate(k, inner)` wrapper giving
`AUC[MaxPool@k] - AUC[MeanPool@k]` as a function of k, on SHADE alone) is specified in the audit
and **is not built**.

---

## 8. Known defects (open, not fixed)

These come from this lesson's own two audits. They are open as of this run and the numbers above
must be read through them.

**(a) `SafetyDriftMonitor` does not implement drift.** Its `update()` returns
`self._absorb_prob_table[cluster(step_t)]` — a pure function of the **current step alone** —
and the trajectory score is a running `max` over those lookups. `self._current_state` is written
and never read. Measured with the class imported unmodified: `score(trajectory) ==
score(SHUFFLED) == score(REVERSED)`, to the last digit. The Markov structure is a *state-
labelling device*, not a drift accumulator, and arXiv:2603.27148's entire thesis is composition
along a path. **This monitor cannot express the mechanism it is named for.**

**(b) Both streaming monitors are monotone in trajectory length by construction.** Both score a
trajectory as `self._peak`, a running maximum, which is non-decreasing in step count. Measured
on trajectories drawn from **one identical benign distribution differing only in length**:
`ESNCusum` gives "longer = positive" AUC **0.9222** (30 vs 120 steps) and `SafetyDriftMonitor`
**0.9024** (20 vs 100 steps). On this corpus the step-count confound bar is a benign 0.5431,
so the effect is not currently paying out — but the property is structural and would bite any
corpus where positives run longer.

**(c) `esn_cusum` drops the paper's ensemble** (single reservoir) and uses a **one-sided** CUSUM,
so a channel drifting *below* its healthy mean is invisible.

**(d) The horizon control is built in the wrong shape.** `FirstKSteps._pool` is
`prefix.mean(axis=0)` — it is `MeanPool` restricted to a prefix, with no `MaxPool@k` and no
`GRU@k`. The §4 curve therefore answers *"how much signal sits in an early prefix"* and
**cannot** answer *"does the mean-vs-max gap widen as the horizon grows"*, which is the actual
dilution claim. A falling curve is what you would get from *less information*; dilution and
information loss are indistinguishable in the curve as built.

**(e) F1, F2 and F3 never consult the confound bar.** Only F4 compares against `binding_bar` in
the machine-readable verdict. The §4 and §5 pricing above was done by hand, in this document.
Relatedly, F0's shuffle runs **inside `common.confound` on bag-of-words features** — it
certifies the confound module, not the embedding-to-aggregator ladder. No shuffle control is run
on the ladder itself.

**(f) The 10,000 bootstrap resamples never enter a verdict.** Every falsifier is an unpaired
point-estimate threshold. `AggregatorResult.scores` is per-trajectory and index-aligned across
all methods, so a **paired** bootstrap on `AUC_max - AUC_mean` (what CLAUDE.md §7 asks for) is
one extra loop away and would give each delta its own CI. F3's `best` is additionally a max over
eight rows — winner's-curse biased upward, with no Holm correction.

**(g) The run's own console header prints an over-strong reading** of the F1/F2 conjunction (see
§7). Believe this README, not that line.

---

## 9. The discarded AgentDojo run

`artifacts/agentdojo_DISCARDED_RUN_2026-08-11.log` is **not evidence** and its AUCs
(`gru` 0.9537, `last_step` 0.9196, `first_4` 0.6582) must never be quoted. Commit `581dd98`
voided that run after a code audit found three CRITICAL bugs, each reproduced with a CPU script
rather than reasoned:

1. **Step TEXT was silently misaligned to step LABELS.** An `aligned` guard was computed and
   then `del`-ed without ever being read, so in the normal case (where `messages` carries
   system/user/tool turns while `step_analyses` covers only analysed steps) judge step *k* was
   bound to `raw_texts[k]` — reproduced: an `on_path=True` label landing on the **system
   prompt**. Every embedding and therefore every AUC in that run was computed over text with no
   verified correspondence to its labels.
2. **The `-1` benign sentinel leaked into `point_of_no_return_step`**, so affected positives
   counted toward the denominator while `alarm_step <= -1` could never be satisfied — silently
   booked as undetected.
3. **The corpora cache fingerprint was tautological** — it recomputed a hash from the very file
   it had just read, so it could never detect staleness.

All three are fixed in `f1d9d44`, each verified by before/after reproduction, and the stale
cache, embeddings, PNGs and `results_agentdojo.json` were **deleted rather than salvaged**. The
log is kept only as a record of what was discarded.

The log's header also reads **`n=0/5`**, which was a separate bug: the summary printed the
**seed** under the label `n`, so a 500-trajectory run announced itself as n=0. Fixed in
`a473cf4`. **The SHADE log carries the same stale header** — that run was already in flight when
the fix landed, so its process held the old code. The real n is **1000 trajectories, seed 0,
5-fold**, as recorded in `results_shade.json`.

Seven of ten audited bug classes came back **clean**, which is also a result: no train/test
leakage in any scaler, centroid, TF-IDF, k-means or CUSUM null; zero measured group straddles;
no `is_causal` flag lying; `state_bytes()` genuinely measures rather than returning a constant.

---

## 10. Run it

```
# The 4090 does one job at a time. Check nothing else is on the GPU first.
nvidia-smi --query-compute-apps=pid,used_memory --format=csv

# SHADE (the deciding corpus, ~90 min end to end, foreground)
set STA_CORPUS=shade
C:\Users\evija\anaconda3\python.exe -m steering_tutorials.streaming_trajectory_aggregation.run_sta

# AgentDojo (the short-horizon control that answers F2)
set STA_CORPUS=agentdojo
C:\Users\evija\anaconda3\python.exe -m steering_tutorials.streaming_trajectory_aggregation.run_sta
```

Every knob is env-overridable with an `STA_` prefix; `config.py` is the single authority
(`STA_N_PER_CLASS`, `STA_SEED`, `STA_EMBEDDER`, `STA_N_FOLDS`, `STA_BOOTSTRAP`, `STA_EARLY_KS`,
`STA_QUERY_TOKEN_K`, `STA_MAX_STEPS`). Results are written to a **per-corpus**
`artifacts/results_<corpus>.json` — never one shared path, which is the defect that let
`cross_trajectory`'s gemma arm overwrite its MiniLM headline. `results_*.json` and the PNGs are
written **before** the summary print, so a cp1252 console crash still leaves the data on disk.

`config.py`, `types.py`, `aggregators/`, `streaming/` and `corpora/` are all **CPU-only to
import**: they load no model, download nothing, and touch no network at import time.

---

## 11. Files

| path | what |
|---|---|
| `config.py` | every knob, one authoritative value each; the pre-registered falsifier text |
| `types.py` | the fixed spine — Pydantic data models + Protocol interfaces |
| `corpora/` | `shade.py` (the deciding corpus) + `agentdojo.py`, `assebench.py`, `atbench.py` |
| `aggregators/pooling.py` | `MeanPool`, `MaxPool`, `MeanMaxStdPool`, `DeviationWeighted`, `LastStep`, `FirstKSteps` — all scored by an identical shared classifier |
| `aggregators/sequence.py` | `GRUAggregator`, `QueryTokenCompressor` — trained end to end |
| `streaming/esn_cusum.py` | `ESNCusum` — reservoir + per-channel CUSUM, one-class |
| `streaming/markov.py` | `SafetyDriftMonitor` — absorbing chain; **see §8(a)** |
| `embed.py` | EmbeddingGemma-300m step embeddings, cached and fingerprinted against the live corpus |
| `evaluate.py` | group-aware CV, bootstrap CIs, confound bars, lead time |
| `run_sta.py` | the orchestrator; evaluates the falsifiers against the numbers just produced |
| `artifacts/results_shade.json` | **the run** — every AUC, CI, per-trajectory score, and verdict |
| `artifacts/shade_run_2026-08-17.log` | its console output |
| `artifacts/*.png` | ROC by method, horizon-truncation curve, lead-time distribution |
| [`CITATIONS_VERIFIED.md`](CITATIONS_VERIFIED.md) | every arXiv id fetched and checked; 2 DO-NOT-CITE traps |
| [`DATASETS_VERIFIED.md`](DATASETS_VERIFIED.md) | the four-corpus sweep, measured before any loader existed |
| [`AUDIT_CODE.md`](AUDIT_CODE.md) | the bug hunt; 3 CRITICAL findings, each reproduced |
| [`AUDIT_FIDELITY.md`](AUDIT_FIDELITY.md) | paper-vs-implementation fidelity + whether the experiment tests its hypothesis |

---

*Screening tier: one corpus, one embedder, one seed, five CV groups. Internal QA pass —
independent external review pending (CLAUDE.md §14 circularity disclosure: the auditors share a
model family with the lesson authors).*
