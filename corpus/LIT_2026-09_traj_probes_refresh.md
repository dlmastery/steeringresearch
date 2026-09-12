# Literature refresh — long-horizon trajectory probes, Aug–Sep 2026

*Compiled 2026-09-11. Every arXiv id below was WebFetch-verified on that date
(title + authors + submission date read from the abstract page). Nothing here is
cited from memory. Ids that could not be verified are marked.*

## Why this refresh

`steering_tutorials/traj_probes/` was built against July 2026 papers
(arXiv:2607.06503, 2605.25310) on ATBench (2604.02022). The question was whether
August–September 2026 changed the picture — new methods, or new corpora that
would make reproduction C (tool-call dependency edges) evaluable.

**Short answer:** one directly relevant method paper (2609.09448), which was
reproduced on the lesson's cached activations the same day; two new ATBench
variants (one usable as OOD); and **still no corpus with tool-dependency
annotations**, so reproduction C stays declared-not-evaluable.

## Verified papers

| id | title (verbatim) | authors | date | relevance |
|---|---|---|---|---|
| **2609.09448** | Do Agents Know When They Succeed? Calibrating Agent Confidence from Internal Representations | Mammen, Joswin, Medicherla | 8 Sep 2026 | **The direct successor to this lesson.** Two residual-stream probes for success prediction (below). Reproduced here. |
| 2609.01466 | Parsing the Stream: A Live Trace Model for Long-Horizon Agents and Their Observers | Pakhomov, Nijkamp (Salesforce) | 1 Sep 2026 | A trace **formalism** (append-only event ledger folded into typed state), not a detector. Releases `Salesforce/tracelab-comprehend` (CC-BY-4.0, 12 synthetic sessions, no safety labels). |
| 2609.11216 | Legible Failures: Detecting and Repairing In-Context Binding Errors | (not fetched in full) | Sep 2026 | Linear probes recover a query-specific hidden-state signal on failed binding trials under a "leak-free protocol". Adjacent: probe-as-repair rather than probe-as-detector. |
| 2608.02464 | Real-Time Detection and Repair of LLM Agent Failures | — | Aug 2026 | Already cited by `streaming_trajectory_aggregation`. |
| 2604.14858 | Benchmarks for Trajectory Safety Evaluation and Diagnosis in OpenClaw and Codex: ATBench-Claw and ATBench-Codex | Yang, Li, Zhu, Zhou, Xie, Luo, Shao, Hu, Liu | 16 Apr 2026 (v2 29 Apr) | Two new ATBench variants. **No dependency-structure annotation.** |
| 2604.07223 | (TraceSafe-Bench; COLM 2026) | CyCraft | Apr 2026 | 1,170 mutated tool-use traces, 12 failure categories. **Gated**, Apache-2.0. No dependency annotations. |
| 2603.18245 | Who Tests the Testers? Systematic Enumeration and Coverage Audit of LLM Agent Tool Call Safety | Chen, Yan, Zhang, Zhang | 18 Mar 2026 (rev 15 Aug) | A search snippet attributed "partial tool-dependency graph annotations" to this line of work. **The abstract page does not support that** — no mention of dependency graphs, Gorilla, or BFCL. Treated as unconfirmed. |

## 2609.09448 in detail, and what was done with it

**Method, as fetched from the full text:**

- **LTD (Latent Trajectory Dynamics)** — 28 "kinematic" features from how the
  final-layer residual state *moves* across turns: cosine displacement
  `1 − cos(h_t, h_{t+1})`, relative displacement `‖h_{t+1}−h_t‖/‖h_t‖`, their
  mean/std/min/max stratified by transition type (reasoning / commitment /
  action / feedback), plus path efficiency and drift ratios → L2 logistic.
- **ARP (Action Representation Probe)** — final-layer state at each action
  endpoint, mean-pooled over actions (terminal excluded), standardised on train
  statistics, PCA-64, L2 logistic, monotone Platt calibration.
- Models Qwen-14B / Qwen-7B / DeepSeek-6.7B; environments Bash / SQL / Python.
- Headline: ARP 0.842 AUROC on Qwen-14B/SQL vs 0.743 for their best baseline.

**The evaluation gap.** Their baselines are *"Calibrated Logprob"* and *"HTC"*
(48 features from token probabilities). **There is no bag-of-words / TF-IDF
text baseline in the paper.** That is the control that decides every verdict in
`traj_probes`, and on ATBench it beat the per-turn snapshot probe.

**Reproduction on this lesson's cached activations** (`traj_probes/ltd.py`;
Gemma-3-1B, ATBench, 997 trajectories, group-aware 5-fold CV, trajectory unit,
paired bootstrap over trajectories vs the TF-IDF bar at 0.8418):

| layer | LTD (28 feats) | ARP (PCA-64+Platt) | LTD+ARP |
|---|---|---|---|
| 12 | 0.7217 (−0.120, excl. 0) | **0.8143** (−0.028, **incl. 0**) | 0.8169 (−0.025, incl. 0) |
| 16 | 0.7154 (−0.126) | 0.7869 (−0.055) | 0.7894 |
| 20 | 0.7057 (−0.136) | 0.7878 (−0.054) | 0.7924 |
| 24 | 0.6538 (−0.188) | 0.7781 (−0.064) | 0.7773 |

For comparison, the lesson's earlier arms on the same corpus and bar:
last-token snapshot L12 **0.8070** (loses), `mean_turn` L12 **0.8565** (parity).

**Reading:**

1. **LTD loses to unigrams at every layer**, by 0.12–0.19, CI excluding zero
   everywhere. The trajectory-*dynamics* idea — the paper's headline — is the
   weakest probe run on this corpus, below even the plain snapshot. The
   turn-count feature is included and priced (0.4839 alone, chance), so this is
   not the step-index confound.
2. **ARP reaches parity at L12** and never beats the bar — the same outcome as
   `mean_turn`, which it closely resembles (both mean-pool over turns).
3. "Consistently outperform surface baselines" is true of the baselines they
   ran and false of the cheapest surface baseline there is.

**Stated deviations:** L24 used as near-final (Gemma-3-1B has 26 layers); role
transitions stratified by (user/assistant/tool) pairs rather than their four
types; their logprob baselines not reproduced (token probabilities were never
cached). 1B model here vs 7–14B there — magnitudes do not transfer; the method
and its control do.

## Datasets

| dataset | rows | labels | licence | gated | dependency edges | use |
|---|---|---|---|---|---|---|
| `AI45Research/ATBench` | 1,000 | 503 safe / 497 unsafe | apache-2.0 | no | **no** | current substrate |
| `AI45Research/ATBench-Claw` | 500 | 204 safe / 296 unsafe | apache-2.0 | no | **no** | **OOD candidate** — OpenClaw sessions, 7–21 events, tool calls as structured `toolCall`/`toolResult` |
| `AI45Research/ATBench-Codex` | ? | ? | CC-BY-4.0 (paper) | ? | **no** | not fetched |
| `CyCraftAI/TraceSafe` | 1,170 | 12 failure cats + benign | apache-2.0 | **yes** | no | refetch-only per licence gate |
| `Salesforce/tracelab-comprehend` | 12 sessions | none (synthetic, comprehension Qs) | CC-BY-4.0 | no | no | not a safety corpus |

**Reproduction C (arXiv:2605.25310, tool-call dependency edges) remains not
evaluable.** No public corpus found annotates which earlier tool output feeds
which later argument. The one lead (a snippet attributing "partial dependency
graph annotations" to 2603.18245) did not survive fetching the abstract.

## What changed in the lesson as a result

- `traj_probes/ltd.py` — LTD + ARP implemented and run; `artifacts/ltd_arp_L{12,16,20,24}.json`.
- README section 7d added; CLAUDE.md §17 entry updated.
- Nothing about the lesson's verdict softened. The Sept 2026 method, tested
  against the control it lacked, lands at parity at best.
