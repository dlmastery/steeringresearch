# arXiv ID verification — streaming_trajectory_aggregation tutorial

Status: IN PROGRESS (appended incrementally as each fetch completes).
Method: WebFetch of `https://arxiv.org/abs/<ID>` — title + author + subject checked, never ID resolution alone.

## Verification table

| id | REAL? | actual title | date | brief's claim |
|---|---|---|---|---|
| 2608.02464 | **REAL** | Real-Time Detection and Repair of LLM Agent Failures | 3 Aug 2026 | **TITLE WRONG** ("AgentTrajectorySentinel" is not this paper's name); technical claims SUPPORTED |
| 2603.27148 | **REAL** | SafetyDrift: Predicting When AI Agents Cross the Line Before They Actually Do | 28 Mar 2026 | SUPPORTED (one number unverified) |
| 2601.00516 | **REAL** | Trajectory Guard -- A Lightweight, Sequence-Aware Model for Real-Time Anomaly Detection in Agentic AI | 2 Jan 2026 | SUPPORTED |
| 2606.00611 | **REAL** | TRACE: Trajectory Risk-Aware Compression for Long-Horizon Agent Safety | 30 May 2026 | SUPPORTED (K=16 unverified) |
| 2606.07054 | **REAL** | TRACE: Trajectory Reasoning through Adaptive Cross-Step Evidence Aggregation for LLM Agents | 5 Jun 2026 | SUPPORTED (exact) |
| 2606.11680 | **REAL** | Organize then Retrieve: Hierarchical Memory Navigation for Efficient Agents | 10 Jun 2026 | **TITLE WRONG** ("HORMA" is the system, not the title); "3-22%" OVERSTATED at the low end |
| 2511.20639 | **REAL** | Latent Collaboration in Multi-Agent Systems | 25 Nov 2025, v4 3 Aug 2026 | SUPPORTED (title is not "LatentMAS") |

---

## Per-paper detail

### 2608.02464 — REAL, but the brief's NAME IS WRONG

- **Actual title:** *Real-Time Detection and Repair of LLM Agent Failures*
- **Author:** Sunny Dubey (single author)
- **Submitted:** 3 August 2026. No revision listed.
- **Subjects:** cs.AI; cs.LG; cs.SE
- **Comments:** 16 pages, 5 figures. Code, data, demo on GitHub; walkthrough video.
- **What it actually claims (from its abstract):** mid-episode LLM-agent failures (loops, cascading
  tool errors, goal drift, fabrication, silent corruption) can be detected from observable telemetry
  alone with lightweight monitors, without step-by-step LLM verification. 2,823 episodes, three
  frameworks/models (qwen2.5, llama3.1, gemini-2.5-flash). One-class **echo-state-network ensemble
  with CUSUM alarms**: 0.71 detection at a 5% false-alarm budget, AUROC 0.872. Deterministic
  verification (recomputing totals from actual tool results) catches 60% of failures (96% with
  coverage checks) at zero false positives, transfers across models unchanged. Closed into a
  rollback/re-execution repair loop it recovers 45% of failures and lifts task success 52% -> 73%
  at ~200 microseconds/step.
- **Brief's characterisation:** the mechanism ("ESN-CUSUM reservoirs"), "~200us/step" and
  "52%->73%" are **SUPPORTED and appear in the abstract**. But the brief calls the paper
  **"AgentTrajectorySentinel"** — that string is **not the title** and does not appear on the
  abstract page. Verdict: **name WRONG, technical content SUPPORTED**. Cite by the real title.
- Note the brief says "booking agent" for the 52%->73% figure; the abstract states it as
  **overall task success**, not scoped to a booking agent -> `[NEEDS VERIFICATION]` on that scoping.

### 2603.27148 — REAL, correctly characterised

- **Actual title:** *SafetyDrift: Predicting When AI Agents Cross the Line Before They Actually Do*
- **Authors:** Aditya Dhodapkar, Farhaan Pishori
- **Submitted:** 28 March 2026
- **Subjects:** cs.CR; cs.AI
- **Comments:** "Sent to COLM conference" (9 pages, 7 figures) — i.e. **submitted, NOT accepted**.
  Do not describe it as a COLM paper.
- **What it actually claims:** "safety drift" — individually benign actions composing into a policy
  violation (read a confidential file -> summarize -> email externally). Models agent safety
  trajectories as **absorbing Markov chains** to predict violation probability within a horizon.
  Risk thresholds vary by task type: communication tasks ~85% violation likelihood within five
  steps from mild-risk states; technical tasks below 5%. The monitor catches **94.7%** of violations
  with **3.7 steps** of advance warning, beating keyword matching and per-step LLM judges.
- **Brief's characterisation: SUPPORTED.** "absorbing Markov chain", "94.7%", "3.7 steps warning"
  all appear in the abstract. **"<0.001ms/step" does NOT appear in the abstract** — the abstract
  says only "lightweight". Mark `[NEEDS VERIFICATION — not in abstract]`.

### 2601.00516 — REAL, correctly characterised

- **Actual title:** *Trajectory Guard -- A Lightweight, Sequence-Aware Model for Real-Time Anomaly
  Detection in Agentic AI*
- **Author:** Laksh Advani (single author)
- **Submitted:** 2 January 2026
- **Subjects:** cs.LG; cs.AI
- **Venue:** "Accepted to AAAI Trustagent 2026" — stated on the page, so the brief's venue is right.
- **What it actually claims:** multi-step agent plans fail from misalignment or structural problems;
  "mean-pooling embeddings dilutes anomalous steps, while contrastive-only approaches ignore
  sequential structure." Proposes a **Siamese Recurrent Autoencoder** with a hybrid
  contrastive + reconstruction loss to catch both contextual and structural plan failures.
  Synthetic and real-world data: **F1 0.88-0.94**, inference latency **32 ms**.
- **Brief's characterisation: SUPPORTED** on every point — architecture, venue, F1 range, and the
  ~32 ms latency all appear in the abstract.
- **Repo note:** this is the same paper the existing `steering_tutorials/trajguard` lesson is built
  on. Reuse that lesson's citation block rather than re-deriving it.

### 2606.00611 — REAL, correctly characterised

- **Actual title:** *TRACE: Trajectory Risk-Aware Compression for Long-Horizon Agent Safety*
- **Authors:** Zhepei Hong, Lin Wang, Liting Li, Haokai Ma, Junfeng Fang, Fei Shen, Dan Zhang,
  Xiang Wang
- **Submitted:** 30 May 2026 (so it carries a 2606 id despite the May date — normal for
  end-of-month submissions)
- **Subjects:** cs.AI
- **Venue:** none stated.
- **What it actually claims:** frames long-horizon agent safety detection as **trajectory-level
  evidence compression**. Risks in long trajectories are "sparse, delayed, and compositional" and
  are missed by turn-level detection. A **Compressor** condenses the full trajectory into a
  "compact latent evidence state" under trajectory-level supervision; a **Reader** judges safety
  from that compressed state. Up to **+12.6 percentage points** across **ASSEBench, Pre-Ex-Bench,
  R-Judge**, with better resilience as context length grows. Code released.
- **Brief's characterisation: SUPPORTED.** "+12.6pp" and all three benchmark names appear in the
  abstract. **"K=16 query tokens" does NOT appear in the abstract** — mark
  `[NEEDS VERIFICATION — not in abstract]` (it is plausibly a body/implementation detail; read the
  paper before quoting it).
- **NAME COLLISION WARNING:** the brief lists a *second* paper also called "TRACE" at 2606.07054.
  Two different systems sharing an acronym in the same month is exactly the kind of thing that
  produces a mis-citation. Disambiguate by full title + id everywhere in the lesson.

### 2606.07054 — REAL, correctly characterised (the OTHER TRACE)

- **Actual title:** *TRACE: Trajectory Reasoning through Adaptive Cross-Step Evidence Aggregation
  for LLM Agents*
- **Authors:** Vijitha Mittapalli, Shreyaa Jayant Dani, Satya Srujana Pilli, Snigdha Ansu,
  Mohammadreza Teymoorianfard, Franck Dernoncourt, Hongjie Chen, Yu Wang, Ryan A. Rossi,
  Nesreen K. Ahmed
- **Submitted:** 5 June 2026 (v1 only)
- **Subjects:** cs.CL; cs.AI; cs.CR; cs.LG
- **Venue:** none stated.
- **What it actually claims (verbatim abstract on file):** autonomous LLM agents can pursue hidden
  malicious objectives through sequences of individually benign actions, defeating standard
  trajectory-level monitoring. Existing work either scores a whole trajectory in one pass or splits
  it into independently scored windows, so it cannot connect evidence across temporally distant
  actions. TRACE runs a **TIJ (Triage-Inspect-Judge) loop**: identify high-signal regions, inspect
  them while carrying accumulated evidence across reasoning steps, then synthesize a
  trajectory-level verdict. Evaluated on **ten task domains from SHADE-Arena**:
  **aggregate F1 0.713, recall 0.844**, largest gains on tasks needing long-range evidence linking.
- **Brief's characterisation: SUPPORTED, exactly.** "Triage-Inspect-Judge", "recall 0.844" and
  "SHADE-Arena" are all verbatim in the abstract. This is the cleanest citation in the set.
- **Distinct from 2606.00611** despite the shared acronym: different authors, different mechanism
  (evidence aggregation loop vs. learned latent compression), different benchmarks (SHADE-Arena vs.
  ASSEBench/Pre-Ex-Bench/R-Judge).

### 2606.11680 — REAL, but the brief cites the SYSTEM NAME as the title

- **Actual title:** *Organize then Retrieve: Hierarchical Memory Navigation for Efficient Agents*
- **Authors:** Hao-Lun Hsu, Nikki Lijing Kuang, Boyi Liu, Zhewei Yao, Yuxiong He
- **Submitted:** 10 June 2026 (v1 only)
- **Subjects:** cs.AI; cs.CL; cs.LG
- **Venue:** none stated.
- **What it actually claims:** LLM agents are stateless, so long tasks force large context encoding.
  **HORMA** (the system, named in the abstract) "organizes experience into a file-system-like
  hierarchical structure, where summarized entities are linked to" the original trajectory data.
  Two phases: structured memory construction, then navigation-based retrieval. Construction
  distinguishes failures caused by *insufficient* information from those caused by *misleading*
  context. A lightweight RL agent traverses the hierarchy to pick minimal-yet-sufficient
  information. Evaluated on **ALFWorld, LoCoMo, LongMemEval**: better task performance under
  constrained context, using **at most 22.17% of baseline tokens** on long conversations.
- **Brief's characterisation: PARTLY WRONG.** "HORMA" and "file-system-like hierarchy" are correct
  and both appear in the abstract — but **HORMA is not the paper's title**; citing it as such will
  not resolve for a reader. The token figure "**3-22%**": the **22%** upper end matches (22.17%),
  but the **3% lower end does NOT appear in the abstract** ->
  `[NEEDS VERIFICATION — not in abstract]`. Quote it as "at most 22.17% of baseline tokens", which
  is what the paper actually says.
- Also note: this is a **memory/efficiency** paper, not a safety-monitoring paper. It belongs in a
  lesson section about latent compression for cost, not about risk detection.

### 2511.20639 — REAL, correctly characterised on the numbers

- **Actual title:** *Latent Collaboration in Multi-Agent Systems*
- **Authors:** Jiaru Zou, Ruizhong Qiu, Gaotang Li, Xiyuan Yang, Katherine Tieu, Pan Lu, Ke Shen,
  Hanghang Tong, Yejin Choi, Jingrui He, James Zou, Mengdi Wang, Ling Yang
- **Submitted:** 25 November 2025. **v1** 25 Nov 2025 · **v2** 8 Dec 2025 · **v3** 1 Jun 2026 ·
  **v4** 3 Aug 2026 (current). The brief's "v4, updated Aug 2026" is **correct**.
- **Subjects:** cs.CL; cs.AI; cs.LG
- **Venue:** **ICML 2026 Spotlight** (stated on the page — the brief omitted this; worth adding).
- **What it actually claims:** **LatentMAS** (the system, named in the abstract) lets LLM agents
  collaborate in continuous latent space instead of via text. Agents emit latent thoughts as hidden
  embeddings; a **shared latent working memory** carries information between agents losslessly
  without re-encoding. Claims higher expressiveness and lossless preservation at lower complexity
  than text-based MAS. Across nine benchmarks: up to **14.6% higher accuracy**,
  **70.8%-83.7% fewer output tokens**, **4x-4.3x faster** end-to-end inference.
- **Brief's characterisation: SUPPORTED.** "70-84% fewer tokens" is a fair rounding of 70.8-83.7%,
  and "4x faster" matches 4x-4.3x. Both are in the abstract. Two refinements: the title is
  *Latent Collaboration in Multi-Agent Systems*, **not** "LatentMAS" (that is the system name); and
  the abstract says **shared latent working memory**, which is not literally "KV-cache sharing" —
  the brief's phrase "latent KV-cache sharing" is a mechanism gloss that does **not** appear in the
  abstract -> `[NEEDS VERIFICATION — not in abstract]`. Read the method section before asserting the
  KV-cache framing.

---

## Systems the brief named WITHOUT an id

Each entry below was resolved by search and then **verified by fetching the arXiv abstract page
itself** — search-engine summaries were never accepted as evidence (see the STAIR entry for why).

### BRT-Align — FOUND: **arXiv:2509.21528**

- **Title:** *Preemptive Detection and Steering of LLM Misalignment via Latent Reachability*
- **Authors:** Sathwik Karnik, Somil Bansal
- **Submitted:** 25 September 2025 (v1 only). Subjects: cs.LG; cs.AI. No venue stated.
- **BRT-Align is the system name, confirmed verbatim in the abstract:** "We propose BRT-Align, a
  reachability-based framework that brings control-theoretic safety tools to LLM inference."
- Models autoregressive generation as a dynamical system in latent space and learns a safety value
  function via **backward reachability**. Two mechanisms: a runtime monitor that forecasts unsafe
  completions several tokens ahead, and a **least-restrictive steering filter** that minimally
  perturbs latent states away from unsafe regions.
- **Brief's characterisation ("backward reachability tube latent safety filter"): SUPPORTED.**
  Cite as the full title, with BRT-Align as the system name. Note the date: this is **Sep 2025**,
  not a 2026 paper.

### SentinelAgent — FOUND: **arXiv:2505.24201**

- **Title:** *SentinelAgent: Graph-based Anomaly Detection in Multi-Agent Systems*
- **Authors:** Xu He, Di Wu, Yan Zhai, Kun Sun
- **Submitted:** 30 May 2025 (v1 only). Subjects: cs.AI. No venue stated.
- Models agent interactions as **dynamic execution graphs**, doing semantic anomaly detection at
  **node, edge, and path** levels; plus a pluggable LLM-powered oversight agent that observes,
  analyzes and intervenes per security policy. Targets single-point faults and prompt injections
  *and* multi-agent collusion / latent exploit paths. Case studies: an email assistant and
  Microsoft's **Magentic-One**. Claims explainable root-cause attribution.
- **Brief's characterisation ("multi-agent execution graphs"): SUPPORTED**, and here the brief's
  name IS the title. Again note the date: **May 2025**.

### ATLAS — FOUND: **arXiv:2601.03093** (title changed between versions — see warning)

- **Current title (v3/v4):** *ATLAS: Verifier-Guided Adaptive Latent Activation Steering for
  Efficient LLM Reasoning*
- **v1 title was different:** *ATLAS: Adaptive Test-Time Latent Steering with External Verifiers for
  Enhancing LLMs' Reasoning*. **Cite the current title**, and be aware secondary sources still
  carry the old one.
- **Authors:** Tuc Nguyen, Thai Le. Submitted 6 Jan 2026; latest version 20 Jun 2026.
- Trains a lightweight **verifier over latent states** that predicts the quality of ongoing
  reasoning and adaptively picks the steering action per-example and per-step, against the failure
  mode of fixed policies / static intervention strength causing over- or under-steering.
- **Brief's characterisation ("verifier-guided latent steering"): SUPPORTED.** Caveat: this is a
  **reasoning-efficiency** paper, not a safety-monitoring paper — do not present it as a guardrail.
- **CONFIRMED by direct abstract-page fetch:** authors **Tuc Nguyen, Thai Le**; v1 6 Jan 2026,
  v2 7 Jun 2026, v3 9 Jun 2026, **v4 20 Jun 2026 (current)**; subjects cs.LG, cs.CL; comments
  "21 pages, 6 figures"; no venue stated. Abstract verbatim opening: "We propose Adaptive Test-time
  Latent Steering (ATLAS), a lightweight framework that dynamically controls steering decisions at
  inference time using a trained, lightweight verifier over the latent states." Reported to
  outperform baselines on math and coding reasoning **while reducing test-time token consumption**.

### TraF-Align — **RESOLVES, BUT TO A COMPLETELY UNRELATED FIELD. DO NOT CITE.**

- The only arXiv paper by that name is **arXiv:2503.19391**, *TraF-Align: Trajectory-aware Feature
  Alignment for Asynchronous Multi-agent Perception* — Song, Yang, Wen, Li; **CVPR 2025**;
  code at `github.com/zhyingS/TraF-Align`.
- It is an **autonomous-driving cooperative-perception** paper. "Multi-agent" here means
  **multiple vehicles/sensors sharing LiDAR features**, not LLM agents. It compensates for
  inter-vehicle network latency by predicting feature-level object trajectories; evaluated on
  **V2V4Real** and **DAIR-V2X-Seq** with AP50/70 gains.
- **This is the exact failure mode the brief warned about** — the same class of error as
  `2606.20852` resolving to a chest-X-ray paper. The id resolves, the name matches, the *subject is
  a different research area entirely*. There is **nothing here about LLM trajectory monitoring or
  activation steering**. Verdict: **DO NOT CITE in this lesson.**

### STAIR — **NO VERIFIED arXiv ID.** (and a hallucinated id was caught here)

- Search returned "*Reusing Past Repairs Through Hierarchical Trajectory Abstraction for Coding
  Agents*" attributed to **arXiv:2607.29657**. I fetched that id. **2607.29657 is actually
  *Development of FDD-ON: an Ontology for VAV HVAC System Fault Detection and Diagnostics***
  (Chen, Fricke, Shen et al., 31 Jul 2026) — an **HVAC fault-diagnosis ontology**, nothing to do
  with coding agents, and the abstract does **not** contain "STAIR".
- **The search summary invented that id-to-title pairing.** This is precisely why the rule is
  fetch-the-abstract-page, never trust a search snippet.
- Other near-matches, all of which are **different systems** and none named STAIR:
  `2605.06642` StraTA (Strategic Trajectory Abstraction for agentic RL — note the confusable name),
  `2604.04225` (Temporal Behavior Tree-guided trajectory repair),
  `2511.05931` (Self-Abstraction from Grounded Experience).
- **Verdict: no arXiv id found for "STAIR (repair trajectory abstraction trees)".** Do not cite it
  until someone produces a verifiable id. If the intent was StraTA, that is a different paper with
  a different claim.

### StTP / StMP — FOUND: **arXiv:2604.08169**

- **Title:** *Activation Steering for Aligned Open-ended Generation without Sacrificing Coherence*
- **Authors:** Niklas Herbster, Martin Zborowski, Alberto Tosato, Gauthier Gidel, Tommaso Tosato
- **Submitted:** 9 Apr 2026 (v1); **revised 2 Jul 2026 (v2)**. Subjects: cs.AI. No venue stated.
- **Confirmed verbatim:** "Steer-to-Target-Projection (**StTP**) and Steer-to-Mirror-Projection
  (**StMP**), that use a logistic regression decision boundary to selectively intervene only on
  tokens" whose activations fall below distribution-derived thresholds — i.e. already-aligned
  tokens are left untouched. Tested against dishonesty and dismissiveness induced by malicious
  system prompts, on **Llama-3.3-70B-Instruct** and **Qwen3.6-27B**, evaluated on **MMLU, MT-Bench,
  AlpacaEval**.
- **Brief's characterisation ("StTP/StMP projection-aware steering"): SUPPORTED.** Both acronyms are
  real and belong to this one paper. Cite the full title — "StTP/StMP" alone will not resolve.
- **Directly relevant to this repo:** it is a coherence-preserving steering paper evaluated on
  capability benchmarks, which is the same multi-axis framing as CLAUDE.md §3/§6.

### "Latent Sentinel" — **NO arXiv ID. It is an OpenReview submission only.**

- Found as *Latent Sentinel: Real-Time Jailbreak Detection with Layer-wise Probes*,
  **OpenReview `id=tuFRx6Ww2n`** (~Oct 2025). I could not locate any arXiv posting.
- Reported content: tiny linear probes on frozen hidden states across multiple Transformer layers,
  scores aggregated online; <0.003% added parameters, ~0.1-0.13% latency overhead on one A100;
  98-100% detection on JailbreakBench/AdvBench; trained on 50k adversarial + 50k benign prompts.
- **Verdict: no arXiv id found.** It may be cited as an OpenReview preprint **if** the lesson states
  that it is unrefereed and not on arXiv — but per §17 rule 4 the README's `> **Reference:**` block
  wants `arxiv.org/abs/...` links, so this one does not qualify for that block. Also note: every
  number above is from a search summary, **not** from a fetched primary page, so all of it is
  `[UNVERIFIED]`.

### COMPACT-VA — **NO arXiv ID FOUND.**

- Two searches (`"COMPACT-VA" arXiv`, and a trajectory-scoped variant) returned nothing by that
  name. Near-misses are all different systems: `2606.23525` (Self-Compacting Language Model
  Agents), `2605.08580` (Slipstream: trajectory-grounded compaction validation), `2509.23586`
  (Reducing Cost of LLM Agents with Trajectory Reduction).
- **Verdict: no arXiv id found. Do not cite.** If the lesson needs a trajectory-compaction citation,
  `2605.08580` or `2606.23525` are real candidates — but they must be fetched and verified first;
  I have not done so, and their titles above come from search snippets.

### Sigma-Mem / reliability matrices — **NO arXiv ID FOUND.**

- Searched both `Sigma-Mem`/`SigmaMem` and the `ΣMem` glyph. Nothing matches.
- Distinct real papers that a careless search might collapse into it: `2602.16666`
  (*Towards a Science of AI Agent Reliability*), `2601.22290` (*The Six Sigma Agent* — note "Sigma"
  here refers to Six Sigma methodology, **not** a memory system), `2606.25161` (*TrustMem*).
- **Verdict: no arXiv id found. Do not cite.** The "Six Sigma Agent" name overlap is a live
  mis-citation risk — it is an enterprise-reliability architecture paper, not a memory-reliability
  matrix.

### Trait-space drift monitoring (June 2026) — **AMBIGUOUS; no paper matches that exact phrase.**

- No arXiv paper is titled "trait-space drift monitoring". The closest real June-2026 paper,
  **verified by direct fetch**, is:
  - **arXiv:2606.21843** — *Measuring What Persists: Conditioning Mechanisms and a Geometric
    Framework for AI Agent Identity*, **Andrew Tanner** (single author), submitted **20 June 2026**,
    cs.AI/cs.CL, 29 pages. Models identity as "non-geodesic structure" and behavioral drift as "its
    relaxation toward the geodesic"; reports 55 distinct response patterns vs 1 baseline and "no
    measurable deformation through 150K tokens". **Read this caveat before citing:** the paper
    itself concedes "the magnitude homology framework's full diagnostic promise... remains
    empirically unconfirmed." Cite it as a *framework proposal*, never as an established result.
- Other real candidates in the same space, **id + title from search only, NOT fetched -> all
  `[UNVERIFIED]`**: `2605.09863` (Nautilus Compass, black-box persona drift, ROC AUC 0.83 —
  May 2026, not June), `2601.04170` (Agent Drift).
- **Verdict: the brief's label does not map to a single paper.** Pick one of the above by its real
  title and verify it, or drop the reference.

---

## SAFE TO CITE

Verified real, subject-matter correct, and the brief's technical characterisation holds:

| id | cite as |
|---|---|
| **2608.02464** | Real-Time Detection and Repair of LLM Agent Failures — **NOT "AgentTrajectorySentinel"** |
| **2603.27148** | SafetyDrift: Predicting When AI Agents Cross the Line Before They Actually Do (submitted to COLM, not accepted) |
| **2601.00516** | Trajectory Guard -- A Lightweight, Sequence-Aware Model for Real-Time Anomaly Detection in Agentic AI (AAAI TrustAgent 2026) |
| **2606.00611** | TRACE: Trajectory Risk-Aware Compression for Long-Horizon Agent Safety |
| **2606.07054** | TRACE: Trajectory Reasoning through Adaptive Cross-Step Evidence Aggregation for LLM Agents |
| **2606.11680** | Organize then Retrieve: Hierarchical Memory Navigation for Efficient Agents — **NOT "HORMA"** (that is the system) |
| **2511.20639** | Latent Collaboration in Multi-Agent Systems (ICML 2026 Spotlight, v4) — **NOT "LatentMAS"** |
| **2509.21528** | Preemptive Detection and Steering of LLM Misalignment via Latent Reachability (system: BRT-Align) |
| **2505.24201** | SentinelAgent: Graph-based Anomaly Detection in Multi-Agent Systems |
| **2601.03093** | ATLAS: Verifier-Guided Adaptive Latent Activation Steering for Efficient LLM Reasoning (v4; v1 had a different title) |
| **2604.08169** | Activation Steering for Aligned Open-ended Generation without Sacrificing Coherence (systems: StTP, StMP) |

## DO NOT CITE

| item | reason |
|---|---|
| **2503.19391 (TraF-Align)** | **RESOLVES TO AN UNRELATED FIELD.** Autonomous-driving cooperative perception (CVPR 2025). "Multi-agent" = vehicles sharing LiDAR features, not LLM agents. Same trap as the 2606.20852 chest-X-ray incident. |
| **2607.29657** | Search claimed this was a coding-agent repair paper. It is *Development of FDD-ON: an Ontology for VAV HVAC System Fault Detection and Diagnostics*. **The id-to-title pairing was fabricated by the search summariser.** |
| **STAIR** | No arXiv id found. Do not substitute StraTA (2605.06642) — different system, different claim. |
| **COMPACT-VA** | No arXiv id found. |
| **Sigma-Mem / reliability matrices** | No arXiv id found. Beware "The Six Sigma Agent" (2601.22290) — unrelated. |
| **Latent Sentinel** | OpenReview only (`id=tuFRx6Ww2n`), no arXiv posting. Fails the §17 rule-4 arXiv-link requirement. |
| **"trait-space drift monitoring"** | Not a paper title. Use 2606.21843 by its real title, with its own "empirically unconfirmed" caveat, or drop. |

## Numbers that are NOT in the abstract — mark `[NEEDS VERIFICATION]`

| claim in brief | id | status |
|---|---|---|
| "<0.001ms/step" | 2603.27148 | **Not in abstract.** Abstract says only "lightweight". |
| "K=16 query tokens" | 2606.00611 | **Not in abstract.** Likely a method-section detail; read the paper first. |
| "3-22% token usage" (the **3%** lower bound) | 2606.11680 | **Not in abstract.** Abstract states "at most 22.17% of baseline tokens" — quote that instead. |
| "latent KV-cache sharing" | 2511.20639 | **Not in abstract.** Abstract says "shared latent working memory". Mechanism gloss unverified. |
| "52%->73% **booking agent**" | 2608.02464 | Numbers are in the abstract; the **"booking agent" scoping is not** — abstract states it as overall task success. |
| all Latent Sentinel figures (98-100%, <0.003% params, 0.1-0.13% latency) | n/a | `[UNVERIFIED]` — from a search summary, no primary page fetched. |
| Nautilus Compass "ROC AUC 0.83"; Agent Drift | 2605.09863 / 2601.04170 | `[UNVERIFIED]` — search snippet only, abstract pages not fetched. |

Numbers that **DO** appear verbatim in their abstracts and are safe to quote: `~200 microseconds/step`,
`52% -> 73%`, `AUROC 0.872` (2608.02464); `94.7%`, `3.7 steps` (2603.27148); `F1 0.88-0.94`, `32 ms`
(2601.00516); `+12.6 pp`, ASSEBench/Pre-Ex-Bench/R-Judge (2606.00611); `F1 0.713`, `recall 0.844`,
SHADE-Arena (2606.07054); `22.17%` (2606.11680); `14.6%`, `70.8-83.7%`, `4x-4.3x` (2511.20639).

---

*Method note: every SAFE-TO-CITE id above was confirmed by fetching `arxiv.org/abs/<id>` directly and
reading title + authors + subject class. Search-engine summaries were used only to locate candidate
ids, never as evidence of what a paper claims — the 2607.29657 case shows exactly why.*
