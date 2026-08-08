# Novelty sweep: the CONFORMAL GATE for conditional activation steering

**Date run:** 2026-08-07
**Agent:** conformal-novelty (subagent of team-lead)
**Question:** Is it novel to apply conformal prediction / distribution-free risk
control to the GATE of a conditionally-applied activation steering vector
(CAST-style conditional steering), so that WHEN the steering fires carries a
calibrated, distribution-free guarantee?

**Status:** IN PROGRESS — appended after each search round.

---

## Searches run

(appended live; a query that returned nothing is logged too — negative coverage
is evidence)

| # | query | tool | outcome |
|---|---|---|---|
| 1 | `conformal prediction activation steering LLM` | WebSearch | Steering hits and conformal hits are DISJOINT sets. Only bridge surfaced: arXiv:2604.19775 (conformal interpretability of probes). |
| 2 | `conformal risk control LLM safety gate guardrail arxiv 2026` | WebSearch | Rich 2026 CRC-for-LLM-safety literature: 2605.20270, 2606.29054, 2607.24343, 2506.00911, 2602.04056, 2512.24587, 2605.30693. None mention steering. |

---

## Findings

### Round 1 (queries 1-2) — raw hits, not yet primary-verified

**Steering side (no conformal content in the hit list):**
- arXiv:2409.05907 — "Programming Refusal with Conditional Activation Steering" (CAST). The
  method our gate would wrap.
- arXiv:2606.11599 "When is Your LLM Steerable?"; 2606.11172 "Predicting Future Behaviors
  in Reasoning Models Enables Better Steering"; 2506.07022 AlphaSteer; 2512.16602 Refusal
  Steering; 2509.00373 activation steering + preference optimization (VLM).

**Conformal side (no steering content in the hit list):**
- arXiv:2605.20270 — "Conformal Selective Acting: Anytime-Valid Risk Control for
  RLVR-Trained LLMs" (May 2026).
- arXiv:2606.29054 — "When Can Conformal Risk Control Certify LLM Outputs? Bounds,
  Impossibility, and Adaptation for Structured Generation" (Jun 2026).
- arXiv:2607.24343 — "Beyond Aggregate Risk: Role-Stratified Conformal Risk Control for
  LLM Tool Calls" (Jul 2026).
- arXiv:2506.00911 — "Conformal Arbitrage: Risk-Controlled Balancing of Competing
  Objectives in Language Models".
- arXiv:2602.04056 — modular safety guardrails for robots, conformal monitoring layer.
- arXiv:2512.24587 MultiRisk; 2605.30693 Triaging Threats to Specialized Guardrails.

**Candidate nearest-neighbour to verify:**
- arXiv:2604.19775 — "From Actions to Understanding: Conformal Interpretability of
  Temporal Concepts in LLM Agents" (Apr 2026). Combines step-wise reward modelling +
  conformal prediction to label INTERNAL REPRESENTATIONS, with linear probes on
  activation-space directions. This is conformal-over-a-probe — the READ half of the
  idea. Needs primary fetch to see whether it ever WRITES (steers).

*Preliminary read: the two literatures are adjacent but the hit sets do not overlap.*

### Round 2 (queries 3-4)

| # | query | tool | outcome |
|---|---|---|---|
| 3 | WebFetch `arxiv.org/abs/2604.19775` | WebFetch | VERIFIED. See below. |
| 4 | `calibrated gate conditional activation steering condition vector threshold guarantee` | WebSearch | Surfaced arXiv:2603.14623 "Proactive Routing to Interpretable Surrogates with Distribution-Free Safety Guarantees" — MUST verify, engine returned a sentence about "the conformal guarantee holds for any gate score". Also 2602.04521, 2605.06342, 2606.12299, 2606.20852 (steering, no conformal). |

**arXiv:2604.19775 — VERIFIED via primary abstract (https://arxiv.org/abs/2604.19775)**
"From Actions to Understanding: Conformal Interpretability of Temporal Concepts in LLM
Agents". Padhi, Kaur, Agarwal, Cobb, Elenius, Acharya, Samplawski, Berenbeim, Bastian,
Jha, Kursuncu, Roy. Submitted **27 Mar 2026** (v1), revised 1 Jul 2026 (v2).
- Combines step-wise reward modelling with conformal prediction to **statistically label
  the model's internal representation at each step as successful or failing**. Linear
  probes are then trained on those labels to find temporal-concept directions.
- **It does steer:** "preliminary results on improving an LLM agent's performance by
  leveraging the proposed framework for steering the identified successful directions
  inside the model."
- **How close:** this is the closest paper found that has BOTH conformal prediction AND
  activation steering. But the roles are different from our idea: conformal is used
  UPSTREAM, to produce *labels for probe training* (a labelling device), not as a
  calibrated *firing threshold* on the gate at inference. The steering it does is
  described as preliminary and is not reported as conditionally gated with a
  distribution-free risk guarantee on when it fires.

**Note on CAST's own gate (arXiv:2409.05907):** the condition vector fires on a
cosine-similarity threshold vs the hidden state, tuned by **grid search / data-driven
tuning** — i.e. an uncalibrated, no-guarantee threshold. That is exactly the hole the
conformal gate would fill.

### Round 3 (queries 5-6)

| # | query | tool | outcome |
|---|---|---|---|
| 5 | WebFetch `arxiv.org/abs/2603.14623` | WebFetch | VERIFIED. Conformal gate — but it gates MODEL ROUTING, not steering. See below. |
| 6 | `distribution-free guarantee representation engineering intervention when to intervene arxiv` | WebSearch | NEGATIVE for the target idea. Hits are causal-representation-learning (2507.05412, 2209.11924, 2205.15196), a RepE survey (2502.17601), 2505.18672, and 2602.04935 ASA (router-conditioned steering, NO conformal). No distribution-free guarantee on a steering gate. |

**arXiv:2603.14623 — VERIFIED via primary abstract (https://arxiv.org/abs/2603.14623)**
"Proactive Routing to Interpretable Surrogates with Distribution-Free Safety Guarantees".
Iqtedar Uddin, Mazin Khider, André Bauer. Submitted **15 Mar 2026**.
- A **lightweight gate** selects a model before either runs; the **routing threshold is
  chosen by Clopper-Pearson conformal calibration** on a held-out set, guaranteeing the
  routed-set violation rate is at most alpha w.p. 1-delta. Derives a **feasibility
  condition** linking the base safe rate pi, the risk budget alpha, and **sufficient AUC
  thresholds** for feasible routing to exist. 35 OpenML datasets.
- **Explicitly NOT about model internals** (verified answer: "This work does not involve
  intervening on model internals").
- **How close:** it is the *methodological template* for a conformal gate — same
  Clopper-Pearson-on-a-gate-score machinery, same feasibility-vs-AUC framing — but what
  it gates is **which model runs**, on tabular OpenML data. It does not touch activation
  steering, LLM residual streams, or CAST. This confirms the CLAUDE.md 18.2 pre-flight
  note: it is feasibility-conditioned on gate AUC, so a probe-AUC result *feeds* it
  rather than being scooped by it.

### Round 4 (queries 7-8) — arXiv API full-text, the strongest negative evidence so far

| # | query | tool | outcome |
|---|---|---|---|
| 7 | arXiv API `all:"conformal" AND all:"activation steering"` (40 results, date desc) | WebFetch | **ONLY 2 HITS IN THE ENTIRE ARXIV CORPUS**, and both are false positives: 2605.26192 (protein co-folding) and 2604.13213 (rare-event stochastic control). **Zero papers in all of arXiv contain both phrases meaningfully.** |
| 8 | arXiv API `abs:"conformal" AND abs:"steering"` (50 results, date desc) | WebFetch | 82 total results, overwhelmingly beam-steering antennas, protein conformations, and diffusion guidance. Only 4 flagged as LLM representation steering: 2606.26924, 2606.12299, 2604.19775, 2505.14617 — and of those only 2604.19775 genuinely pairs conformal with activation-space steering. |

This is the single most important negative result in this sweep: an arXiv full-text
conjunction of "conformal" and "activation steering" returns **nothing on-topic**.

Adjacent robotics results worth noting (conformal gating *when to intervene*, but not on
activations) [UNVERIFIED — API summary only]:
- arXiv:2602.22474 "When to Act, Ask, or Learn: Uncertainty-Aware Policy Steering"
  (2026-02-25) — conformal prediction calibrates VLM verifiers that gate robot policy
  selection. The closest *conceptual* analogue of a calibrated firing decision.
- arXiv:2606.12299 "Learning What to Say to Your VLA: Mostly Harmless Vision Language
  Action Model Steering" (2026-06-10) — conformal prediction to safely steer a VLA, but
  by language feedback, not by residual-stream injection.
- arXiv:2505.00779 "Uncertainty-aware Latent Safety Filters", arXiv:2501.04823 "Learning
  Robot Safety from Sparse Human Feedback using Conformal Prediction" — conformal
  thresholds gating a safety filter in robotics.

**arXiv:2602.04935 (ASA) [UNVERIFIED — from search snippet only]:** "Activation Steering
Adapter", training-free inference-time controller, single-shot mid-layer intervention,
**router-conditioned mixture of steering vectors**. This is a *gated* steering method —
i.e. it occupies the CAST slot — but the snippet shows no conformal / distribution-free
component. Flagged as adjacent prior art on the gating axis, not on the guarantee axis.

### Round 5 (queries 9-10)

| # | query | tool | outcome |
|---|---|---|---|
| 9 | arXiv API `all:"conformal" AND all:"steering vector"` (40 results) | WebFetch | **2 hits in all of arXiv, both antenna/audio signal processing** (2312.14482 beamforming, 2212.02076 conformer speech separation). Zero LLM hits. Confirms round-4 negative. |
| 10 | `conformal abstention jailbreak detection classifier language model distribution-free guarantee 2026` | WebSearch | Conformal-abstention-for-LLM-safety is an ACTIVE area (2506.13593 time-to-unsafe-sampling, 2606.11949 online shift detection + conformal adaptation, 2507.03279 conformal information pursuit, ICML 2026 workshop on agentic uncertainty). Jailbreak-detection hits (2602.16520, 2604.16824) carry NO conformal component. **The conformal-abstention line abstains from OUTPUTS; it never gates an activation intervention.** |

### Round 6 (queries 11-12)

| # | query | tool | outcome |
|---|---|---|---|
| 11 | `CAST conditional activation steering condition vector calibration false positive rate control follow-up 2026` | WebSearch | NEGATIVE for a conformal follow-up. CAST's own reported control is *empirical* (harmful refusal 83-90%, benign false-refusal <6%) with **no distribution-free guarantee**; commentary notes the gate is "highly sensitive to the choice of hyperparameters" and "may require periodic recalibration". Named extensions are multitask/continual steering and hypernetworks — not calibration. |
| 12 | arXiv API `all:"conformal" AND all:"residual stream"` | WebFetch | **ZERO RESULTS IN ALL OF ARXIV** (`totalResults=0`). |

Query 12 is the cleanest negative available: no arXiv paper, ever, contains both the
phrase "conformal" and the phrase "residual stream".

### Round 7 (queries 13-16) — THE TWO NEAREST WORKS FOUND

| # | query | tool | outcome |
|---|---|---|---|
| 13 | arXiv API `all:"risk control" AND all:"steering vector"` | WebFetch | **ZERO RESULTS IN ALL OF ARXIV** (`totalResults=0`). |
| 14 | `"conformal" steering vector LLM intervention calibrated firing threshold safety guarantee arxiv 2607 2608` | WebSearch | Surfaced the two nearest works: 2606.12299 and 2605.14746. Also 2603.24543 "Analysing the Safety Pitfalls of Steering Vectors", 2505.22637 "Understanding (Un)Reliability of Steering Vectors", 2502.18862 — all steering-safety, no conformal. |
| 15 | WebFetch `arxiv.org/abs/2606.12299` | WebFetch | VERIFIED. |
| 16 | WebFetch `arxiv.org/abs/2605.14746` | WebFetch | VERIFIED. |

#### NEAREST WORK #1 — arXiv:2606.12299 (VERIFIED, https://arxiv.org/abs/2606.12299)
"Learning What to Say to Your VLA: Mostly Harmless Vision Language Action Model
Steering". Hyun Joe Jeong, Gokul Swamy, Andrea Bajcsy. **10 Jun 2026**.
Verbatim, the load-bearing sentence: *"...learns an improvement head that predicts when
language steering will improve performance. **We conformalize this improvement head to
prevent harmful steering interventions**, where the LFP decreases task performance
relative to the original instruction on out-of-distribution scenarios."*
- **This is the closest structural match in the literature: a conformalized GATE that
  decides WHEN a steering intervention fires.** Same shape as the idea.
- **What differs (the gap):** (i) the intervention is a **language feedback policy** —
  prompt-level text, explicitly *not* an activation-space / residual-stream vector;
  (ii) the model is a frozen **VLA robot policy**, not an LLM being conditionally
  steered; (iii) the risk being controlled is **task-performance degradation**, not
  safety-refusal behaviour; (iv) no CAST-style condition vector, no per-layer injection
  site, no coherence/capability axes.

#### NEAREST WORK #2 — arXiv:2605.14746 (VERIFIED, https://arxiv.org/abs/2605.14746)
"Selective Safety Steering via Value-Filtered Decoding". Bat-Sheva Einbinder, Hen
Davidov, Yee Whye Teh, Yarin Gal, Yaniv Romano. **14 May 2026** (v1), rev. 12 Jul 2026.
- States the exact problem our gate targets: *"existing decoding-time steering methods
  often intervene unnecessarily, modifying generations that would have been safe under
  the base model... provides an **explicit bound on the probability of false
  interventions**. A single threshold hyperparameter controls this bound."*
- **What differs (the gap):** the intervention is **decoding-time token filtering**
  (value-filtered sampling), NOT an activation steering vector; and the verified read of
  the abstract is that it **does not use conformal prediction / distribution-free risk
  control** — the bound comes from the value criterion. Note the author list is the
  Romano conformal group, so a conformal follow-up on this line is a live scoop risk.
- Verified answer to the direct question: "The document does not mention either
  approach [conformal prediction or distribution-free risk control]."

### Round 8 (queries 17-18)

| # | query | tool | outcome |
|---|---|---|---|
| 17 | arXiv API `abs:"conformal" AND abs:"intervention" AND abs:"language model"` (40 results) | WebFetch | The full census of "conformal gates an LLM intervention". 7 flagged: 2606.16667 (budgeted conformal evidence — gates zoom/crop in VLMs), 2606.12299, **2604.09155 CORA: Conformal Risk-Controlled Agents (10 Apr 2026, execute/abstain boundary for GUI agent ACTIONS)**, 2604.19775, 2602.22474, 2511.06575 CoFineLLM, 2504.21022 ConformalNL2LTL. **Every one of them gates an OUTPUT, ACTION, or TOOL CALL. None gates an activation-space edit.** |
| 18 | `selective prediction conformal probe linear classifier decides when to apply steering vector LLM refusal` | WebSearch | NEGATIVE for conformal. Returns the "when does steering work" line instead: 2604.15557 "Predicting Where Steering Vectors Succeed", 2606.22686 "The Geometry of Refusal: Linear Instability in Safety-Aligned LLMs", 2605.28553 "Refusal Before Decoding: Detecting and Exploiting Refusal Signals in Intermediate LLM Activations", 2512.16602. All predict steering success from probes/geometry with **no distribution-free calibration**. |

Round 8 matters because it is a *census* rather than a keyword shot: it enumerates the
papers that conformalize an LLM intervention decision. The intervention being gated is,
in every case, at the **output/action layer** — abstain, re-look, re-ask, execute-or-not,
say-this-instead. The **write-to-activations** slot is empty.

### Round 9 (queries 19-22) — final recency + last two candidates

| # | query | tool | outcome |
|---|---|---|---|
| 19 | `arxiv 2607 2608 conformal prediction steering vector activation intervention language model guarantee August 2026` | WebSearch | The two literatures are still parallel in Jul 2026: 2607.24562 "Hierarchical Group-Conditional Conformal Risk Control for **Selective Prediction** in Language Models" (CRC, no steering) vs 2607.27574 "Policy Gradient Steering", 2607.25270 "Where Steering Signals Come From", 2607.19364 (steering, no conformal). **No 2608 hit of any kind.** |
| 20 | arXiv API `abs:"conformal" AND abs:"activation" AND abs:"LLM"` (40 results) | WebFetch | 3 flagged: 2606.04141, 2604.19775, 2505.14617. Nothing else. |
| 21 | WebFetch `arxiv.org/abs/2606.04141` | WebFetch | VERIFIED — NOT a scoop, see below. |
| 22 | WebFetch `arxiv.org/abs/2607.19364` | WebFetch | VERIFIED — NOT a scoop, see below. |

**arXiv:2606.04141 (VERIFIED, https://arxiv.org/abs/2606.04141)** "Caught in the
Act(ivation): Toward Pre-Output and Multi-Turn Detection of Credential Exfiltration by
LLM Agents". Kargi Chauhan, Pratibha Revankar. 2 Jun 2026. Has activation probes AND
split conformal prediction — but the verified read is that the **conformal calibration is
applied to the honeytoken detector, not to the activation probe**, and the whole paper is
**detection-only** (no intervention, no steering).

**arXiv:2607.19364 (VERIFIED, https://arxiv.org/abs/2607.19364)** "Statistically Grounded
Sparse-Feature Interventions for Activation-Space Control in Large Language Models".
Siddique, Alam, Rafy, Raiyan, Mahmud, Hasan. 5 Jun 2026. Gemma-family SAE steering with a
six-condition reliability filter + Borda consensus over F-test / KSG-MI / Cohen's d.
"Statistically grounded" here means **statistic-based feature RANKING with no formal
guarantee** — verified: "no mention of conformal prediction or distribution-free risk
control". Its gate selects *which features* to steer with, not *when* to fire on an input.

---

## Verdict

# STILL OPEN

**Epistemic status of this claim.** This is "I could not find prior work" — and the
strength of that statement rests mostly on the three arXiv full-text API conjunctions,
which are exhaustive over the arXiv corpus rather than ranked-and-truncated like a web
search:

| exhaustive arXiv conjunction | total results | on-topic |
|---|---|---|
| `all:"conformal" AND all:"activation steering"` | 2 | **0** (protein folding, rare-event control) |
| `all:"conformal" AND all:"steering vector"` | 2 | **0** (beamforming, conformer ASR) |
| `all:"conformal" AND all:"residual stream"` | **0** | **0** |
| `all:"risk control" AND all:"steering vector"` | **0** | **0** |

I am NOT claiming "there is no prior work" in an absolute sense. Residual risk: the
arXiv API matches phrases, so a paper using different vocabulary (e.g. "calibrated
context gate", "certified representation control", "PAC-Bayes trigger") would be missed;
non-arXiv venues, ICML/NeurIPS 2026 workshop tracks, and anything posted in the last few
days are outside this sweep. Two workshop venues are live in this exact space
(ICML 2026 "Statistical Frameworks for Uncertainty in Agentic Systems"; TrustNLP@ACL 2026)
and were not enumerated paper-by-paper.

### The 3 nearest published works

1. **arXiv:2606.12299** — "Learning What to Say to Your VLA: Mostly Harmless Vision
   Language Action Model Steering". Hyun Joe Jeong, Gokul Swamy, Andrea Bajcsy.
   **10 Jun 2026.** *Nearest by structure.* Conformalizes an "improvement head" that
   predicts when steering will help, explicitly "to prevent harmful steering
   interventions". Same idea shape — a conformal gate on a steering decision — but the
   steering is **language/prompt feedback to a frozen VLA robot policy**, and the
   controlled risk is **task-performance degradation**, not activation-space editing or
   safety-refusal behaviour.

2. **arXiv:2603.14623** — "Proactive Routing to Interpretable Surrogates with
   Distribution-Free Safety Guarantees". Iqtedar Uddin, Mazin Khider, André Bauer.
   **15 Mar 2026.** *Nearest by machinery.* Clopper-Pearson conformal calibration of a
   **gate threshold**, with a feasibility condition tying the risk budget to the base
   safe rate and to sufficient **gate AUC**. But it gates **which model runs** on tabular
   OpenML data and explicitly does not touch model internals.

3. **arXiv:2604.19775** — "From Actions to Understanding: Conformal Interpretability of
   Temporal Concepts in LLM Agents". Padhi, Kaur, Agarwal, Cobb, Elenius, Acharya,
   Samplawski, Berenbeim, Bastian, Jha, Kursuncu, Roy. **27 Mar 2026** (rev 1 Jul 2026).
   *Nearest by substrate.* The only paper found that does conformal prediction AND
   activation-space steering in one work — but conformal is used **upstream, to label
   representations for probe training**, and the steering is unconditional and reported
   as "preliminary". No calibrated firing threshold, no distribution-free guarantee on
   when the intervention applies.

Runner-up worth watching: **arXiv:2605.14746** "Selective Safety Steering via
Value-Filtered Decoding" (Einbinder, Davidov, Teh, Gal, Romano; 14 May 2026, rev 12 Jul
2026) names the exact problem — "existing decoding-time steering methods often intervene
unnecessarily" — and gives "an explicit bound on the probability of false interventions",
but at the **decoding/token-filtering** layer and (verified) **without** conformal
prediction. Given the author list is the Romano conformal group, a conformalized sequel
on this line is the most likely route by which this idea gets scooped.

### The precise gap that remains open

Every conformal-gates-an-LLM-intervention paper found gates at the **output / action /
token / prompt** layer: abstain, re-look, re-ask, execute-or-not, say-this-instead
(2606.16667, 2604.09155 CORA, 2602.22474, 2511.06575, 2504.21022, 2607.24562, 2606.12299).
Every conditional-activation-steering paper found sets its firing threshold by **grid
search or empirical tuning** with no finite-sample validity (2409.05907 CAST itself,
2602.04935 ASA, 2607.19364).

**Nobody has put a distribution-free, finite-sample guarantee on the decision to WRITE to
the residual stream.** Specifically open:

- a conformal / Clopper-Pearson calibration of the **CAST condition-vector similarity
  threshold**, converting "cosine > tau, tau found by grid search" into "the false-fire
  rate on benign prompts is at most alpha with probability 1-delta";
- the two-sided version — controlling **over-refusal (gate fires on benign)** and
  **miss (gate silent on harmful)** simultaneously, which is the selectivity axis of the
  program's composite;
- the **feasibility condition** of 2603.14623 instantiated on a *steering* gate: what
  probe AUC is required for a given (alpha, base harmful rate) to admit a feasible
  threshold at all. This is the point at which the program's judge-free probe-AUC
  measurements become the input to a novel result rather than a repetition of one.

This is consistent with, and now much better evidenced than, the CLAUDE.md 18.2 pre-flight
note: 2603.14623 is feasibility-conditioned on gate AUC, so a probe-AUC result
**strengthens** it rather than being scooped by it.

---

## Coverage summary

**22 distinct queries**: 10 WebSearch, 6 exhaustive arXiv-API conjunctions, 6
primary-abstract WebFetch verifications. Every paper characterised above as a
near-neighbour was read from its own arXiv abstract page, not from a search snippet.
Anything read only from a snippet is marked `[UNVERIFIED]` inline (2602.04935,
2602.22474, 2606.16667, 2604.09155).
