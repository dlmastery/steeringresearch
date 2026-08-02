# SYNTH 2026-08 — Persona Simulation & Long-Horizon Multi-Turn Data Synthesis

**Scan date:** 2026-08-02
**Hard date filter:** arXiv IDs `2606`, `2607`, `2608` ONLY (published 2026-06-02 or later).
Anything `2605` or earlier is OUT OF SCOPE and is named as lineage only, never cited.

**Verification discipline:** every arXiv id below was WebFetched and its title + author list
confirmed before it shipped. Numbers are quoted only where read directly from the paper page.
Where a field could not be read from the fetched content, it says `not stated in fetched content`
rather than a guess.

**Consumer context:** this feeds the activation-steering course lessons on multi-turn jailbreak
detection (`multiturn_jailbreak`), cross-session/fractured-intent attacks (`cross_trajectory`),
and trajectory guards (`trajguard`), which already use SafeMTData Attack_600 and CSTM-Bench.
The three questions that decide whether a method is usable here are therefore carried as
columns: **can it run on a local 1B–4B**, **does it emit per-turn labels**, and
**was it validated against real attack data rather than its own synthetic test set**.

---

## Summary table

| # | Paper | arXiv | Date | Method (one line) | Needs frontier model? | Per-turn labels? | Validated vs REAL attacks? |
|---|---|---|---|---|---|---|---|
| 1 | Context-Fractured Decomposition Attacks on Tool-Using LLM Agents | 2606.09084 | 2026-06-08 | Plant benign artifacts in early sessions, trigger harmful composition in a fresh context | **No** — attacker is uncensored Llama-3-8B by default | **Yes** — every setup step must individually score Accept=1 | Partial — real agent benchmark (AgentDojo), synthetic attacks |
| 2 | MJ: Multi-turn LLM Jailbreaking via Decomposed Credit Assignment | 2607.11070 | 2026-07-13 | DC-GRPO gives each turn its own group-relative advantage (immediate + future credit) | **No** — attacker is Qwen3-4B-Instruct-2507 | **Yes** — that is the entire contribution | Real harmful-behavior benchmarks (HarmBench/StrongREJECT/JailbreakBench); attacks themselves synthetic |
| 3 | GPT-Red: Automated Red Teaming via Self-Play at Scale | 2607.26115 | 2026-07-28 | Attacker self-plays against a *population* of simultaneously-trained defenders | **Yes** — GPT-5.5/5.6 class throughout | No — environment-level reward only | **Yes** — human red-teamers + IPI 2025 human prompt-injection dataset |
| 4 | Learning to Attack and Defend (AdvGRPO) | 2606.09701 | 2026-06-08 | Attacker↔defender GRPO co-training with a single→multi-turn curriculum | **No** — Qwen2.5-7B/14B attacker | Dense multi-channel reward; not per-turn safety labels | Real benchmarks (AdvBench/HarmBench/WildJailbreak) |
| 5 | Scalable Hierarchical Attention Transformers for Multi-Turn Jailbreak Detection | 2606.21082 | 2026-06-19 | Encode each turn, then a conversation-level cross+self-attention transformer | **No** — ~278M total | **Head exists but is DISABLED** (aux loss weight = 0) | Partly — real corpora (WildChat/ShareGPT/XGuard/Red Queen) + LLM-generated attacks |
| 6 | TRACE: Turn-level Reward Assignment via Credit Estimation | 2607.13988 | 2026-07-15 | Per-turn reward = TD change in log-ratio state value from a frozen reference model | **No** — Qwen3-4B and Qwen3-30B-A3B | **Yes** — dense per-action, no critic training | N/A (agentic search, not safety) |
| 7 | Beyond Individual Personas (GroupPersona) | 2606.07893 | 2026-06-05 | Condition user agents on *behavioral groups* so the corpus matches a reference distribution | Not stated in fetched content | No | **Yes** — aligns to real reference corpora (assistant-style + Reddit) |
| 8 | State-Grounded Multi-Agent Synthetic Data Generation (StateGen) | 2606.16307 | 2026-06-15 | 4-role loop: persona user sim + agent + state-grounded tool sim + multi-axis judge | Model-agnostic via config registry; not stated | **No** — one 8-criterion score per finished conversation | **No** — explicitly no production-data comparison |
| 9 | WRIT: Write-Read Intensive Trajectory Synthesis | 2606.02908 | 2026-06-01 | Synthesize trajectories along write-decision count × evidence burden | Generation implied frontier; **trained model is 4B** | Not stated in fetched content | No — τ²-bench only |
| 10 | Drowning in Routine: Signal Dilution in Multi-Turn Agent Training | 2606.22164 | 2026-06-20 | Decision density ρ (fraction of turns that affect return) governs multi-turn RL difficulty | No | Theory *about* per-turn signal | Controlled synthetic experiments |
| 11 | Learning from Synthetic Data without Model Collapse (KITE) | 2607.17043 | 2026-07-19 | Failure-guided generation + boundary-aware uncertainty curation | No — open-source LLMs | No | No — instruction-tuning, not attacks |
| 12 | Simulated Customers Never Walk Away | 2606.20708 | 2026-06-16 | Teacher-forced probe: do simulated users reproduce REAL decision outcomes? | No | No | **Yes** — 2,790 real sales conversations, 793 with verified payment outcomes |
| 13 | Enjoy Your Talk | 2607.10428 | 2026-07-11 | Decoupled persona user-sim + target + independent judge ensemble | Judges are LLM ensemble | Intent-tracking measured per turn | Not stated in fetched content |
| 14 | Synthetic minority data is redundant or invalid | 2607.20787 | 2026-07-22 | Validity is a population quantity; de-biased test on withheld real data | No | N/A | **Tabular, not LLM text** — method transfers, domain does not |
| 15 | Simulating Eating Disorder Patients with LLMs | 2606.26109 | see note | Persona stability across/within multi-turn conversations vs psychometric ground truth | No — six unnamed LLMs | No | **Yes** — published case vignettes with known EDE-Q scores |

> **Note on #15:** the arXiv id carries the `2606` prefix (in-window) but the fetched page
> reported a submission date of 12 May 2026. The discrepancy is unresolved; treat this row as
> `[DATE-AMBIGUOUS]` and do not use it as a load-bearing citation without re-checking.

---

## 1. Context-Fractured Decomposition Attacks on Tool-Using LLM Agents (arXiv:2606.09084)

**Xiaofeng Lin, Yukai Yang, Daniel Guo, Sahil Arun Nale, Charles Fleming, Guang Cheng — 8 June 2026.**
*Exploiting Artifact Provenance Gaps.*

**Method.** Two phases. **Semantic decomposition:** an unrestricted attacker LLM recursively
splits a harmful objective G into sub-queries, with *rejection sampling* — candidates are
regenerated up to R retries until each qᵢ is (i) artifact-oriented (writes to or reads from a
store 𝒮) and (ii) does not lexically reference G. **Fractured execution:** a "planting" phase
writes benign primitives (generic templates, encrypted strings) across *separate sessions*; a
"triggering" phase in a fresh context issues a benign command that reads and recombines them.
Decomposition is fully automated with configurable depth Dₘₐₓ and branching Wₘₐₓ.

**Numbers read.** ASR across six models — GPT-4.1 **66.03%** (vs Tree-of-Attacks 38.36%,
Crescendo 56.60%, Role-Play 47.17%); Gemini-2.5-pro **76.25%** (vs 58.36/48.11/42.10);
Claude-3.5-Sonnet **31.44%** (vs 3.14/7.55/19.50); Gemini-2.0-flash **36.85%**; Mistral-Small-3.2
**37.11%**; GPT-5-nano **20.12%**. Best average rank **1.17**. Abstract claims up to **+28.3 pp**
over SOTA baselines. Attacker: **jailbreak-tuned uncensored Llama-3-8B** by default (GPT-4.1 as
ablation). Benchmark: AgentDojo filtered to 53 data-exfiltration cases (Banking 11, Slack 7,
Travel 19, Workspace 16).

**Detector results — the important part for us.** A GPT-4o-mini monitor judging the conversation
log detects Crescendo at **60.37%** but CFD leaf-only at **47.17%**, dropping to **26.42%** when
benign noise is added. No complete defense is implemented; "provenance lineage tagging" is
proposed but unvalidated.

**What this project could use.** This is the closest 2026 analogue to our `cross_trajectory`
lesson and it is *stronger* than SafeMTData: the fracture is across **sessions and agent
instances**, not just turns, so a per-conversation guard cannot see it by construction. Two
directly reusable pieces: (a) the **rejection-sampling constraint** — regenerate a sub-query
until it is artifact-oriented AND lexically G-free — is a cheap, local-model-runnable recipe for
manufacturing genuinely benign-looking turns, which is exactly the hard-negative problem our
detection lessons keep hitting; (b) the **per-step Accept=1 success criterion** gives us real
per-turn benign labels for free, which is what §2 of our rubric needs and what most trajectory
datasets lack. The 26.42%-under-noise detector number is also a ready-made *bar* for
`trajguard` — a residual to beat, in the spirit of the completion-length confound bar.

---

## 2. MJ: Multi-turn LLM Jailbreaking via Decomposed Credit Assignment (arXiv:2607.11070)

**Junyoung Park, Namgyu Park, Sechan Lee, Yoon-Chan Jhi, Jihoon Cho, Sangdon Park — 13 July 2026.**

**Method.** DC-GRPO, a turn-level credit assignment framework for GRPO. Instead of broadcasting
one trajectory-level score to every turn, each turn gets its own group-relative signal combining
an **immediate credit** term `(rᵢ,ₜ − μₜʳ)/σₜʳ` and a **future credit** term
`(Rᵢ,ₜ₊₁ − μₜ₊₁ᴿ)/σₜ₊₁ᴿ`. Two weightings: *static* (fixed α) and *dynamic*
(rollout-dependent, `σₜʳ/σₜᴿ` and `γσₜ₊₁ᴿ/σₜᴿ`).

**Numbers read.** Dynamic **98.26%** ASR₅@₃ average, static **97.88%**, vs SEMA **86.58%** and
TROJail **86.23%** (~11–12 pp gain). Attacker: **Qwen3-4B-Instruct-2507** (Qwen2.5-3B-Instruct in
appendices). Victims: Llama-3.1-8B-Instruct (training victim), Qwen2.5-7B-Instruct, Gemma-2-9B-IT,
Mistral-7B-Instruct-v0.3, plus GPT-OSS-20B. Training on AdvBench; eval on HarmBench (200),
StrongREJECT† (288 after dedup), JailbreakBench† (55 after overlap filtering). A DPP diversity
bonus and a "Diversity vs. ASR" trade-off analysis appear in the appendices; mode-collapse
quantification is limited in the main text. **SafeMTData is not used.**

**What this project could use.** This is the single most implementable paper in the scan for our
hardware. A **Qwen3-4B attacker** fits the 16 GB 4090 ceiling and the "smallest first" rule, and
**Gemma-2-9B-IT is already a victim in their table** — so a replication lands directly on a model
family we run. The concrete borrow is DC-GRPO's decomposition: our detection lessons currently
have trajectory-level labels only, and DC-GRPO is a principled, published way to *derive* per-turn
importance from a trajectory outcome without ground-truth turn labels. That is precisely the gap
flagged in the `multiturn_jailbreak` and `trajguard` lessons. Note the honest caveat: the credit
signal is optimised to make attacks succeed, not to mark which turn is *harmful* — using it as a
turn-label proxy is an assumption that needs its own falsifier.

---

## 3. GPT-Red: Automated Red Teaming via Self-Play at Scale (arXiv:2607.26115)

**Eric Wallace, Christopher A. Choquette-Choo, Nikhil Kandpal, Sam Toyer, Dylan Hunn,
Stephanie Lin, Yuxin Wen, Xiangyu Qi, Christopher Wolff, Zizhao Wang, Milad Nasr, Sicheng Zhu,
Chuan Guo, Juan Felipe Cerón Uribe, Kaiwen Wang, Aiden Low, Kai Xiao, Kai Chen — 28 July 2026.**

**Method.** A scalable self-play algorithm in which the attacker is tasked with attacking a
**diverse population of simultaneously-trained defender agents** rather than a single frozen one.
The defender-model tool is stateful, letting the attacker build multi-turn conversations.
Rewards are environment-specific and asymmetric (check attacker success and defender task
completion).

**Numbers read.** Attacker initialised from **GPT-5.5**; defenders are GPT-5-series through
GPT-5.5 during training, with **GPT-5.6** as the held-out robustness evaluation. Fake
chain-of-thought direct prompt injection improved **5.2% → 95.9%**. Held-out IPI scenarios against
GPT-5.6: **ASR below 4%**. GPT-5.6 reached **100% robustness** on the IPI 2025 Challenge dataset
when rerun, with one misgraded success validated by human inspection. Parameter counts are not
disclosed.

**On mode collapse — read carefully.** The claim is *qualitative* in the fetched text: attacker-only
training was "better narrowly, i.e., mainly against the defender they trained against and worse on
held-out ones," and Figure 7 shows attacks trained against nine frozen defenders transfer poorly
because "the attacker learned to inspect the behavior of its opponents and tailor its attack
accordingly." **No exact numerical diversity comparison between multi-defender and attacker-only
conditions was readable.** Do not quote a diversity number from this paper.

**What this project could use.** The transferable *idea* is that population-of-defenders is the
anti-mode-collapse mechanism, and it is model-scale-agnostic — a 1B–4B attacker against a small
pool of differently-prompted or differently-steered Gemma defenders is a faithful scaled-down
instantiation. This is also the only paper in the scan with a **genuine external-validity check**:
comparison against human red-teamers and a dataset of human-authored function-calling prompt
injections. That is the standard our own synthesis lessons should be held to. Caveat for the
course: as published it is entirely frontier-scale and cannot be reproduced here; cite it as
motivation, not as a reproduction target.

---

## 4. Learning to Attack and Defend: Adaptive Red Teaming of Language Models via GRPO (arXiv:2606.09701)

**Blake Bullwinkel, Eugenia Kim, Amanda Minnich, Mark Russinovich — 8 June 2026.**

**Method.** AdvGRPO makes GRPO viable for joint attacker-defender optimisation using **dense
multi-channel rewards** and **decoupled advantage normalisation**. Three-stage curriculum:
(1) single-turn attacker (K=1), (2) multi-turn closed-loop (K>1, max 5 turns), (3) attacker-defender
co-training with alternating updates every N=10 steps.

**Numbers read.** Attackers: Qwen2.5-7B-Instruct, Qwen2.5-14B-Instruct, Qwen3.5-9B; GPT-4.1 as
training-time defender; Qwen2.5-7B-Instruct as co-training defender. Qwen2.5-14B multi-turn:
**90.0%** on AdvBench, **91.0%** on HarmBench. Transfer to Llama-3.1-8B: **92.5%** AdvBench,
**88.5%** HarmBench. Evaluation includes WildJailbreak.

**The negative result is the valuable part.** The authors document **entropy collapse in attacker
prompts over training**, found entropy regularization hard to tune, and report in Table 5 that
attacks "vary surface-level phrasing rather than explore structurally different approaches."
No quantitative diversity metric beyond token-level entropy is provided.

**What this project could use.** Two things. First, the **curriculum ordering**
(single-turn → closed-loop multi-turn → co-train) is a cheap, directly copyable recipe and the
paper states it prevents the defender from dominating. Second, and more useful to us: this is a
**published, honest admission that self-play attackers collapse to surface-level paraphrase**.
That is exactly the failure mode our program worries about — it is the synthesis analogue of the
`cos(diffmean,pca)=0.966` lesson, where two things look interchangeable at the surface and are not.
Any synthesis lesson we build must therefore measure *structural* diversity, not lexical, and this
paper is the citation for why. It also warns that a synthesis pipeline seeded from a collapsed
attacker will produce a detector that only learns one phrasing family.

---

## 5. Scalable Hierarchical Attention Transformers for Multi-Turn Jailbreak Detection in Long Conversations (arXiv:2606.21082)

**Chenhui Hu, Muhammed Salih, Sudipto Guha, Subramanian Srinivasan — 19 June 2026.**

**Method.** A TurnEncoder produces compact per-turn representations; a ConvTransformer
(4 decoder layers, 8 heads, 2048 FFN) combines cross-attention and self-attention over the turn
sequence. TurnEncoder is **multilingual E5-base, 278M params**; total ~278M.

**Numbers read.** Benchmark: **14,038 conversations**, hybrid — real corpora (UltraChat, WildChat,
ShareGPT, OASST2, XGuard, Red Queen, SafeDialBench) plus LLM-generated synthetic conversations
covering gradual escalation, sudden adversarial insertion, and context narrowing (generator model
not disclosed). Justification: "No single jailbreak corpus covers the full range of known
multi-turn attack strategies."

| Model | Recall | Precision | F1 | FPR |
|---|---|---|---|---|
| **This model (τ=0.50)** | 92.10% | 95.90% | **0.9394** | **2.80%** |
| Claude Opus 4.7 | 78.30% | 90.60% | 0.8691 | 5.80% |
| Claude Sonnet 4.5 | 66.10% | 91.70% | 0.7682 | 4.30% |
| Claude Sonnet 4 | 59.10% | 91.70% | 0.7189 | 3.80% |
| ToT Majority | 37.00% | 89.10% | 0.5226 | 3.20% |
| ToT OR | 47.10% | 85.30% | 0.6072 | 5.80% |
| Concatenation | 36.20% | 88.50% | 0.5136 | 3.40% |

Ablation at L=4: **AUC-ROC 98.66%, accuracy 95.04%**. Combining attention mechanisms cut false
positives by **2.26 pp** vs self-attention alone.

**The finding that matters most to us.** The architecture **has a turn-level classification head,
but the auxiliary loss weight is set to zero, so the head has no effect on training or inference.**
Only conversation-level binary labels are used, and the paper **does not explain how per-turn
labels would be assigned from conversation-level annotations.** This is the exact open problem our
`trajguard` and `multiturn_jailbreak` lessons face, and the SOTA detector paper in-window simply
declines to solve it.

**What this project could use.** A 278M hierarchical detector is squarely within our budget and
beats a frontier judge on F1 at half the FPR — a strong argument for the "small local guard"
thesis the course already makes. The `Concatenation` baseline (F1 0.5136) is a ready-made,
honestly-weak bar. Most importantly: the disabled turn head is a **documented gap we could
actually close**, e.g. by using DC-GRPO-style (#2) or TRACE-style (#6) credit decomposition to
manufacture the turn labels this model architecturally wants but was never given. That is a real,
publishable-shaped hole, not a reproduction.

---

## 6. TRACE: Turn-level Reward Assignment via Credit Estimation for Long-Horizon Agents (arXiv:2607.13988)

**Leitian Tao, Baolin Peng, Wenlin Yao, Tao Ge, Hao Cheng, Mike Hang Wang, Jianfeng Gao,
Sharon Li — 15 July 2026.**

**Method.** Represent a rollout as state transitions at tool-call boundaries; obtain gold-answer
log-probabilities from a **frozen reference model**; transform them into log-ratio state values;
derive per-action rewards as **temporal-difference changes** in those values. **No additional
critic training is required.**

**Numbers read.** Qwen3-4B: **7.2 → 35.6** on BrowseComp-Plus. Qwen3-30B-A3B: **8.4 → 42.6**.
Specific baseline comparisons beyond these were not readable in the fetched content.

**What this project could use.** This is the cleanest general-purpose answer in the scan to
"how do you assign turn-level labels when only the trajectory is labelled," and it is
**critic-free and runs on a 4B**. Because it only needs a frozen reference model and
log-probabilities, it is directly portable to our setting: substitute the harmful-completion
target for the gold answer and the TD-delta becomes a per-turn "how much did this turn advance the
harmful objective" score. That would give `trajguard` genuine per-turn supervision derived from
Attack_600's trajectory labels. It is safety-agnostic as published (agentic search), so any use
here is an **adaptation, not a reproduction** — and it needs a falsifier: does the TD-delta
actually correlate with human turn-level harm judgments, or does it just track fluency?

---

## 7. Beyond Individual Personas: Aligning Synthetic Dialogue to Population-Level Behavior Distributions (arXiv:2606.07893)

**Xinyi Liu, Rinat Khaziev, Hooshang Nayyeri, Emine Yilmaz, Charith Peris, Hari Thadakamalla — 5 June 2026.**

**Method.** GroupPersona. The stated failure mode is that dialogue generators produce individually
plausible conversations while **distorting overall corpus composition**. GroupPersona "turns
population statistics into generation controls": it separates each dialogue's core behavioral
signature from predictable side effects, and conditions user agents on **behavioral groups** drawn
from a reference population's interaction patterns.

**Numbers read.** 12 behavior attributes. Alignment measured by **Jensen-Shannon divergence**,
quality calibration by **mean absolute deviation**. JSD reduced **0.234 → 0.177** vs the strongest
average baseline, a **24.4% reduction**. Quality-score MAD from the reference-conversation profile
**0.63** vs **0.91** for next-best. Four corpora across two dialogue sources (assistant-style and
Reddit-derived), with structure-preserving and variation-enhanced construction variants. Model
sizes not stated in fetched content.

**What this project could use.** This is the direct answer to the subtopic question *"does persona
diversity provably increase data diversity or just surface variation?"* — and its answer is that
**per-dialogue plausibility and corpus-level realism are different objectives, and optimising the
first distorts the second.** The reusable machinery is the evaluation, not the generator:
**JSD against a real reference corpus** is a cheap, judge-free, defensible diversity check that we
could run over any synthetic multi-turn set we build (including our own Attack_600 augmentations)
to prove the synthetic corpus matches a real one distributionally. Judge-free matters here given
our judge calibration failure (AUC 0.7508, under the 0.85 gate) — JSD over behavior attributes
sidesteps the broken instrument entirely.

---

## 8. State-Grounded Multi-Agent Synthetic Data Generation for Tool-Augmented LLMs (arXiv:2606.16307)

**Rahul Khedar, Eshita, Sneha Teja Sree Reddy Thondapu, Mayank Malhotra, Arup Das, Jitesh Chandra,
Yun-Shiuan Chuang, Chaitanya Kulkarni, Arun Menon, Linsey Pang, Avinash Karn, Mouli V,
Prakhar Mehrotra — 15 June 2026.**

**Method.** StateGen runs a four-role LLM loop — a **persona-conditioned user simulator**, an agent
under test, a **state-grounded tool simulator**, and a **multi-axis LLM judge** — around an
authoritative **state manager** holding a structured world-state object across turns. Each role is
bound to a model independently via a configuration registry (concrete model identities are not
disclosed).

**Numbers read.** **64,698** evaluated conversations across three production datasets. Tool-call
hallucination score **9.66/10** (49,331 samples). Overall mean score **6.54** (median 7.0, σ=2.25).
Persona vector is **23-dimensional**: 6 categorical demographics (jurisdiction, age bracket,
channel, device type, language proficiency, time availability), 12 continuous behavioral traits
(cost sensitivity, patience, assertiveness, verbosity, politeness, domain knowledge, risk
tolerance, compliance tendency, platform trust, digital literacy, slang usage, emoji usage), and
5 emotional states (frustration, anxiety, trust, confidence, stress). Goal-achievement spreads
**51.2% → 67.0%** across eight persona profiles, a **15.8 pp spread**. Compared against eight
external systems.

**Honest limitations, read directly.** Persona diversity is **not formally quantified** beyond
showing trait variation produces measurable behavioral differences. Scoring is **holistic after
conversation completion** — eight criteria on the full transcript, **no per-turn scores**. And
there is **no empirical comparison against real production conversations**; the paper's stated
motivation is that real logs contain PII and are unusable, which is a reason for synthesis, not
evidence of fidelity.

**What this project could use.** The 23-dimensional trait vector is a well-specified, copyable
persona schema, and the **15.8 pp goal-achievement spread is a genuine "personas do something"
result** — behavioral, not lexical, which is the right kind of evidence. The **state manager**
pattern is the transferable architecture: an authoritative world-state object across turns is
exactly what makes a multi-turn trajectory internally consistent, and it is what our
`cross_trajectory` synthesis lacks. But by our own §4 criterion this paper is a **generator of its
own priors** — the judge that scores the data comes from the same family as the generator, and
there is no external anchor. Under our R17.3 off-family-judge rule, its 9.66/10 would be
inadmissible as a headline number. Use the architecture; discard the scores.

---

## 9. WRIT: Write-Read Intensive Trajectory Synthesis for Multi-Turn User-Facing Agents (arXiv:2606.02908)

**Hengrui Gu, Xiaotian Han, Kaixiong Zhou — 1 June 2026.**

**Method.** Synthesises trajectories along two explicit complexity dimensions: **quantity of write
decisions** and **evidence burden per decision**. The pipeline generates write-intensive and
read-heavy tasks, diversifies user behavior instructions for conversational realism, then simulates
agent-user interactions in executable environments.

**Numbers read.** "With only **2K** synthesized trajectories, a **4B** model trained on WRIT
outperforms **GPT-5.1 no-think** on **τ²-bench** and substantially reduces inference-time token
usage." Per-turn/step labelling is not stated in the fetched content; no validation against real
user data is reported.

**What this project could use.** The headline is a data-efficiency existence proof that matters for
our compute budget: **2K trajectories, 4B model, beats a frontier baseline.** That is the scale our
lessons operate at. The transferable design choice is **parameterising synthesis along named
difficulty axes** rather than sampling blind — the direct analogue for us is (number of
decomposition steps × how much harmful intent each step carries), which would give
`cross_trajectory` a principled curriculum instead of a flat pool. Caveat: executable environments
do the verification work here, and safety has no equivalent oracle, so the guarantee does not
carry over.

---

## 10. Drowning in Routine: Signal Dilution in Multi-Turn Agent Training (arXiv:2606.22164)

**Yann Pernot, Vi Retault — 20 June 2026.**

**Method / finding.** Formalises **signal dilution**: in multi-turn agents some decisions affect
downstream reward while others are "necessary but reward-equivalent." When consequential decisions
are sparse, routine turns inject gradient noise without learning signal into trajectory-level
methods such as GRPO. The governing quantity is **decision density ρ** — the fraction of turns
whose actions affect returns.

**Numbers read.** In controlled experiments with tunable decision density, "the predicted scaling
is recovered with **R² = 0.999**," and the training-step gap widens significantly as ρ → 0.
Model architectures and parameter counts are not specified.

**What this project could use.** This is the theory under the practice of #2 and #6, and it is
directly diagnostic for our data. Fractured/decomposed jailbreaks are **the low-ρ regime by
construction** — most turns are deliberately inert, and only one or two carry the payload. ρ gives
us a **single measurable statistic to characterise an attack dataset** and to predict, before
spending GPU, whether a trajectory-level guard can learn from it at all. Computing ρ over
Attack_600 and CSTM-Bench would be a cheap, judge-free, CPU-only piece of work and would give the
`trajguard` lesson a principled reason for per-turn supervision rather than an intuition.

---

## 11. Learning from Synthetic Data without Model Collapse in Iterative Instruction Tuning (arXiv:2607.17043)

**Xiaonan Luo, Yue Huang, Kehan Guo, Ping He, Chuan Zou, Ting Hua, Xiangliang Zhang — 19 July 2026.**

**Method.** KITE, a two-stage framework combining **failure-guided data generation** with
**boundary-aware uncertainty curation** — targeting a model's specific weaknesses rather than
generating generically.

**Key finding.** Collapse in iterative instruction tuning "is not simply uniform performance
degradation, but can appear as a **polarization of competence**, where synthetic training
reinforces already strong skills while further degrading weak ones."

**Numbers read.** None. The fetched content states only that "experiments across several datasets
and multiple open-source LLMs show that KITE yields more stable improvement than strong
synthetic-data baselines." **No model sizes or numerical results were readable — do not quote
numbers from this paper.**

**What this project could use.** The polarization finding is a measurement warning with teeth: if
we iterate a synthesis loop and track only an aggregate score, collapse is **invisible** — the
average holds while the tails diverge. That is the same silent-failure shape catalogued in §18.8
of our constitution. The operational rule to import: **report per-capability (or per-attack-family)
breakdowns on every synthesis iteration, never the mean alone.** For our attack synthesis this
means tracking ASR per harm category and per decomposition depth, not a pooled ASR.

---

## 12. Simulated Customers Never Walk Away: Decision Fidelity of LLM User Simulators Measured Against Real Purchase Outcomes (arXiv:2606.20708)

**Liang Chen — 16 June 2026.**

**Method.** A **teacher-forced probe protocol** that holds context and instrument fixed, run
against a production dataset of **2,790 real sales conversations including 793 with verified
payment outcomes**. It tests whether simulated users reproduce actual decision patterns when facing
consequential choices.

**Numbers read.** The failure is named the **"disengagement deficit"** and it is *conditional*, not
uniform. For eventual buyers, simulators matched real behavior (**+0.09** depth bias). For eventual
**non-buyers**, simulators showed inflated interest (**+0.40** depth bias, **d=0.38, p<0.001**).
Expressed resistance was **halved, 25.1% → 13.5%**. Fabricated deliberation **nearly doubled,
21.9% → 40.1%**. Replicated on DeepSeek (**d=0.41, p=0.002**).

**What this project could use.** This is the methodological centrepiece of the whole scan for
subtopic 4, and it generalises far beyond sales. The design is: **take real trajectories with
verified outcomes, teacher-force the simulator into the same context, and check whether it diverges
differentially by outcome class.** Applied to us, the question becomes: does a simulated attacker
give up when a real attacker gives up? If simulated adversaries "never walk away" the way simulated
customers never walk away, then **synthetic attack corpora systematically under-represent
abandoned and partial attacks**, and every guard trained on them is calibrated on a population of
implausibly persistent adversaries. That is a concrete, testable defect in the data our detection
lessons already use. The stratified-by-outcome comparison is cheap and judge-free and is the single
best validity check available in-window.

---

## 13. Enjoy Your Talk: A Human-Centered Benchmark for Multi-Turn Dialogue (arXiv:2607.10428)

**Jinglan Gong, Jiefan Lu, Hewei Guo, Kehan Li, Zhiyuan Han, Jihang Jiang, Wenwen Tong,
Lewei Lu — 11 July 2026 (v1), revised 24 July 2026 (v2).**

**Method.** Decouples three roles that are usually entangled: a **persona-grounded user simulator**,
a **target model evaluated separately on intent perception and response generation**, and an
**independent, configurable ensemble of LLM judges**.

**Numbers read.** **3,400 dialogues** across **17 target models**. SOTA models separate by up to
**9×** on objective intent-tracking. Final-intent completion rate (FICR) ranges **0.53–0.88** on
PersonaMem-v2 but **saturates above 0.95** on Nemotron-USA. A "warm-up effect" appears in **16 of
17** models. Model sizes and whether real human data grounds the benchmark were not stated in the
fetched content.

**What this project could use.** Two structural lessons. First, **decoupling the simulator from the
judge is the architectural fix for the self-judging defect** we found across 8 UNKNOWN-provenance
lessons — this paper treats judge independence as a first-class design constraint, and it is a
clean citation for our R17.3 off-family rule. Second, the benchmark-saturation contrast
(0.53–0.88 vs >0.95) is a warning we should heed directly: **a synthetic multi-turn benchmark can
saturate and stop discriminating**, which is the failure our own dashboards guard against by
requiring at least one dominated row. Worth checking whether CSTM-Bench is in the saturated regime.

---

## 14. Synthetic minority data is redundant or invalid (arXiv:2607.20787)

**Ahmad B. Hassanat, Ahmad S. Tarawneh, Ghada A. Altarawneh — 22 July 2026.**
*A data-dependent validity theory and a de-biased test.*

**Domain caveat, stated up front: this is imbalanced TABULAR classification (medical, finance),
not LLM text generation.** It is included for its *method*, and any transfer is an analogy that
would need its own validation.

**Method.** Reframes validity as a **population quantity** — the probability that a synthetic point
truly belongs to the minority class — rather than as performance on the data that generated it.
Validity is then determined by **class overlap in the underlying distribution, not by the choice of
generation method**. The de-biased test estimates true validity using **withheld real data**.

**Numbers read.** The classical evaluation (scoring synthetic examples against the data that
created them) **underestimates true invalidity in 96–99% of method-by-imbalance-ratio cells**.
Across **91 methods and three classifiers**, gains over trivial baselines have a **median below
0.01 F1**, most approaches damage calibration, and gains fall within "a decision threshold's reach"
of noise. The dichotomy: where classes overlap, no faithful generator can produce truly valid
synthetic minorities (**invalid**); where classes separate, oversampling adds nothing over baseline
(**redundant**).

**What this project could use.** The 96–99% figure is the sharpest available statement of the
subtopic-4 thesis: **evaluating synthetic data against its own generating distribution
systematically flatters it.** The importable rule is the de-biased test's shape — hold out **real**
data and score synthetic validity against that, never against the generator's own pool. For us that
means any synthetic multi-turn attack set must be scored against held-out real attacks
(CSTM-Bench, MHJ) and never against a synthetic test split from the same pipeline. The
redundant-or-invalid dichotomy is also a useful pre-registration prompt: before synthesising, ask
whether the classes overlap (synthesis cannot help) or separate (synthesis is unnecessary), and
state which regime you believe you are in.

---

## 15. Simulating Eating Disorder Patients with LLMs (arXiv:2606.26109) `[DATE-AMBIGUOUS]`

**Jennifer Haase, Jana Gonnermann-Müller, See Heng Yim, Nicolas Leins, Jan Mendling,
Sebastian Pokutta.** The arXiv id prefix is in-window (`2606`) but the fetched page reported
**12 May 2026**; the discrepancy is unresolved. Treat as provisional.

**Method.** A **dual-assessment framework** (persona self-report **+** independent observer
ratings) using the **EDE-Q** psychometric instrument, grounded in **five published case vignettes
with known ground-truth scores**. Experiment I tests *between*-conversation stability;
Experiment II tests *within*-conversation stability.

**Numbers read.** All six tested models (unnamed) **systematically overshoot ground-truth severity
by 12–30% of the scale range (0.7–1.8 points on a 0–6 scale)**.

**What this project could use.** Relevant purely as **persona-fidelity methodology**. The design —
anchor personas to cases with *externally known* ground-truth scores, then measure drift with an
independent observer rather than self-report — is the persona analogue of #12's outcome-anchored
probe, and it is the right template for asking whether a simulated attacker persona stays in
character across a long trajectory. Persona drift over turns is a plausible and unmeasured
confound in every long-horizon attack corpus we use. The systematic-overshoot direction is also
suggestive: simulated personas exaggerate their defining trait, which would predict that simulated
attackers are *more* overtly adversarial than real ones — consistent with #12 and directly
contrary to what a fractured-intent corpus needs.

---

## What I searched for and did NOT find

Recorded so the gaps are visible rather than silently absent. All searches used the
2606/2607/2608 prefix filter; results below were rejected as out-of-window or nonexistent.

1. **A persona-bank-at-scale paper in-window.** The scaling line is all older — Persona Hub
   (2406.20094, 1B personas), Persona Generators (2602.03545), DeepPersona (2511.07338),
   Population-Aligned Persona Generation (2509.10127). **Nothing 2606+ builds a large persona bank.**
   The in-window persona work has moved from *scale* to *distributional alignment* (#7) and
   *fidelity measurement* (#15) — which is arguably the more useful turn, but it means there is no
   current citation for "more personas ⇒ more diversity."

2. **A paper proving persona diversity yields semantic rather than surface diversity.** The
   closest statements are out-of-window: "Measuring Lexical Diversity of Synthetic Data Generated
   through Fine-Grained Persona Prompting" (2505.17390), which found fine-grained persona detail
   yields *minimal* gains over simply specifying a length cutoff, and 2602.03545's remark that
   embedding metrics "struggle to disentangle meaningful semantic diversity from mere stylistic
   variance." **In-window, the question is answered only obliquely** — by #7's JSD-vs-reference
   framing and #8's 15.8 pp behavioral spread. No in-window paper settles it.

3. **Any in-window paper using SafeMTData / Attack_600.** Searched repeatedly. **Zero hits.**
   #2 explicitly does not use it. Our lessons appear to be ahead of the in-window literature on
   this specific corpus, which cuts both ways — no external baseline to check ourselves against.

4. **An in-window paper on synthesising *cross-session* (not merely multi-turn) attacks with
   per-turn labels.** #1 is the only one operating across sessions, and its labels are a byproduct
   of its success criterion rather than a designed annotation. No dedicated cross-session synthesis
   method exists in-window.

5. **A quantitative multi-defender-vs-attacker-only diversity number.** #3 makes the claim
   qualitatively and #4 documents collapse without a metric beyond token entropy. **No in-window
   paper reports a clean numerical diversity comparison for self-play red teaming.** The
   quantitative statements on this belong to out-of-window work (Self-RedTeam 2506.07468's
   17.8%-more-diverse figure; Quality-Diversity Red-Teaming 2506.07121; Evolving Diverse Red-team
   LMs 2310.00322) and are **not cited here** under the date filter.

6. **User-simulator realism/Sim2Real validation in-window.** This entire literature clusters just
   outside the window — realsim (2605.02624), distributional-gap measurement (2605.07847),
   Sim2Real for agentic tasks (2603.11245, 451 human participants), RealUserSim (2605.20204).
   **#12 is the only in-window member of this family**, which makes it disproportionately important
   for subtopic 4.

7. **Persona-conditioned red teaming in-window.** PCAP (2605.11730) and PersonaTeaming (2605.05682,
   2509.03728) are all out-of-window. **No in-window paper combines persona conditioning with
   adversarial generation** — a live gap, and one this project is positioned to fill.

8. **An in-window method that validates synthetic multi-turn attacks against real attack
   distributions.** Nothing does this directly. #3 comes closest (human red-teamers, human-authored
   IPI dataset) but is frontier-scale; #14 supplies the right *statistical* machinery but for
   tabular data. **The honest summary of subtopic 4 is that in-window synthesis methods largely do
   not verify that their output resembles real attacks** — which is precisely the failure mode the
   task description anticipated.

9. **Numbers for #11 (KITE) and model sizes for #7 (GroupPersona) and #8 (StateGen).** Fetches
   returned abstract-level content only. Marked as not-stated above rather than guessed; re-fetch
   the full HTML/PDF before any of these is used as a load-bearing citation.

---

*All 15 arXiv ids above were WebFetch-verified for title and author list on 2026-08-02. No id is
marked `[UNVERIFIED]`; one is marked `[DATE-AMBIGUOUS]` (#15). Numbers appear only where read
directly from the fetched paper; absences are stated explicitly. Per the project's inheritance
rule, every number here is `[NEEDS VERIFICATION]` until reproduced on our own ladder.*
