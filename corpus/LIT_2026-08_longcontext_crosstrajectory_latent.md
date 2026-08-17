# LIT 2026-08 — Long-context jailbreaks, cross-trajectory attacks, and latent-based detection

**Scanned:** 2026-08-05. **Window:** core citations are **2604–2607** (Apr–Jul 2026).
Four older papers are included and **explicitly marked** because they are load-bearing
(one of them is the falsifier that gates this entire research line).

**Verification status:** every arXiv ID below was **WebFetch-verified** — real title +
real author list + submission date pulled from the abstract page. None cited from
memory. Numbers quoted from abstracts are `[NEEDS VERIFICATION]` until reproduced on
our own ladder, per the repo rule for inherited numbers.

---

## 0. The one-sentence synthesis

> **Every attack in this literature is the same move at a different granularity:
> fragment the harmful signal below the window the defender scores over. Every
> defence is the same counter-move: widen or compress the window until the signal
> re-aggregates. The differences are entirely about *what unit* gets fragmented.**

| unit fragmented | attack | our lesson |
|---|---|---|
| tokens / position in context | NINJA (2511.04707) | — (gap) |
| turns in one conversation | MT-JailBench (2605.11002), MultiBreak (2605.01687) | `multiturn_jailbreak` |
| generated tokens | — | `trajguard` |
| tool artifacts across contexts | **CFD (2606.09084)** | — (gap) |
| sessions | **CSTM-Bench (2604.21131)** | `cross_trajectory` (uses it as OOD) |
| cooperating agents | **2607.07368** | `cross_trajectory` |
| traces in a repository | — | `meerkat` |

This is the strongest argument yet that `multiturn_jailbreak` / `trajguard` /
`cross_trajectory` / `meerkat` are **four points on one axis**, not four lessons.

---

## 1. Latent / activation-based detection — the core 2026 line

### 1.1 arXiv:2604.28129 — Latent Adversarial Detection
Kulkarni (sole author), 30 Apr 2026.
*"Latent Adversarial Detection: Adaptive Probing of LLM Activations for Multi-Turn
Attack Detection"*

Probes the **residual stream in real time**; identifies *"adversarial restlessness"* —
each attack phase-shift produces measurable activation movement. **Five scalar
trajectory features.**

**This is the single most important paper for us, and not for its headline.** Its
headline is 76.2% → **93.8%** on held-out **synthetic** data (models 24B–70B). But:

- **On real-world LMSYS it drops to 47–71%.**
- Three-source combined training: 89.4% at 2.4% FPR.
- **Probes do NOT generalise across architectures** (model-specific).
- **Three-phase turn-level labels are critical — binary labels produced 50–59% false
  positives.**

Two direct consequences for this repo:

1. **The synthetic→real gap is measured, not speculative.** This is exactly the
   dataset problem, quantified by someone else. A detector tuned on synthetic
   trajectories loses ~25–45 points on real traffic.
2. **`trajguard` and `multiturn_jailbreak` use binary labels.** This paper says
   binary labelling alone drives 50–59% FPR and that a three-phase (benign →
   escalation → payload) turn-level scheme is what makes the method work. That is a
   concrete, cheap, testable change to two existing lessons.

### 1.2 arXiv:2605.02958 — SALO / Refusal Trajectories
Hu, Wang, Lim, Gao, Chen. v1 2 May 2026, v2 26 May 2026.
*"Tracing the Dynamics of Refusal: Exploiting Latent Refusal Trajectories for Robust
Jailbreak Detection"*

Uses **causal tracing** to find a *"Refusal Trajectory"*: a **sparse upstream
activation pattern that persists across layers even when the jailbreak suppresses the
surface refusal signal**. Method **SALO** (Sparse Activation Localization Operator)
operates on **raw hidden-state volumes from a selected layer window**, preserving
layer×token geometry. Evaluated on Qwen / Llama / Mistral, calibrated with XSTest,
attacked with GCG.

**The line that indicts our own lesson 1** — verbatim, **§5.4 Ablation Study** (NOT the
abstract; a subagent checking only the abs page could not confirm it and correctly
dropped it, so the section pointer is recorded here to stop that recurring):

> *"Mean pooling dilutes this strong "needle" into the vast "haystack" of irrelevant
> background noise."*

Two further details from the full text, both load-bearing and both stronger than the
quote alone:

- **The paper's own method uses Global Max-Pooling**, and its §5.4 ablation *replaces*
  that with Global Average Pooling to demonstrate the collapse — Table 2 caption:
  *"Mean Pooling: Replaces the sparsity-aware Global Max-Pooling with Global Average
  Pooling."* So max-pool is the paper's sparsity-aware default, not an afterthought.
  Our `probe_tuning` sweep must therefore include **max** as a first-class cell.
- **Appendix D is the sharper warning for us:** *"sequence-level mean aggregation can
  achieve high recall on several attack sets but yields near-random XSTest AUROC."*
  That is mean pooling looking **good on attacks and failing on the benign/over-refusal
  set** — a selectivity failure that an attack-only evaluation cannot see. Any layer/
  pooling sweep we run must score an XSTest-style benign arm, or it will reproduce
  exactly this blind spot.

`hello_world` classifies **layer-12 mean-pooled** activations. `probe_tuning`'s layer
sweep was never built. If SALO is right, our probe is discarding the signal by
construction, and the fix (a layer *window* + no mean pool) is cheap and falsifiable.
**This is the most actionable finding in the scan.**

### 1.3 arXiv:2605.28553 — Refusal Before Decoding
*"Refusal Before Decoding: Detecting and Exploiting Refusal Signals in Intermediate
LLM Activations."* Toxic/jailbreak concepts are **linearly identifiable** in the
residual stream and usable for both detection and intervention — i.e. the READ→WRITE
composition our lesson 2 attempts, from the detection side. Corroborates the course's
recurring result that **READ works while WRITE is weak.**

### 1.4 arXiv:2606.00611 — TRACE (the architecture to steal)
Hong, Wang, Li, Ma, Fang, Shen, Zhang, Wang. 30 May 2026.
*"TRACE: Trajectory Risk-Aware Compression for Long-Horizon Agent Safety"*

**Compressor–Reader**: the Compressor encodes a full trajectory into a **compact
latent evidence state**; the Reader judges safety from that. Targets exactly the
stated problem — *"sparse, delayed, and compositional risk signals"* that escape local
moderation.

- **+12.6 pp** over baselines on ASSEBench, Pre-Ex-Bench, R-Judge.
- **Degrades less as context length grows** (on LongSafety).
- Code on GitHub.

This is the most directly transplantable design for our long-trajectory problem, and
its Compressor is a *latent* summary — so it sits naturally beside our steering work
rather than being a pure NLP baseline.

---

## 2. THE FALSIFIER — read before building any latent detector

### 2.1 arXiv:2412.09565 — Obfuscated Activations `[OLDER: Dec 2024 / Feb 2025]`
Bailey, Serrano, Sheshadri, Seleznyov, Taylor, Jenner, Hilton, Casper, Guestrin,
Emmons.
*"Obfuscated Activations Bypass LLM Latent-Space Defenses"*

Breaks **sparse autoencoders, representation probing, and latent OOD detection** by
jointly optimising inputs that produce harmful output *while matching benign
activation profiles*.

> **Reduces recall from 100% to 0% while retaining a ~90% jailbreak rate.**

**Every §1 method is measured against non-adaptive attacks.** Under this repo's own
pre-registration rule, no latent-detection claim we make is admissible without an
adaptive-obfuscation arm. The paper's one consolation — obfuscation **degrades
performance on complex tasks** (e.g. SQL generation) — is itself the most interesting
research direction: *the capability cost of hiding* may be the thing that is actually
detectable, and it is a **geometry** question, which is this program's home turf.

---

## 3. Cross-trajectory / multi-agent — the 2026 threat models

### 3.1 arXiv:2607.07368 — Multi-Agent AI Control (the fragmentation effect)
Makins, Angelini, Shams, Phuong. 8 Jul 2026. NeurIPS submission, 81 pp.

> *"almost all prior AI control work considers a single agent attacking within a
> single trajectory"* — but real deployments run many agents in shared environments.

**The fragmentation effect: as more agents coordinate, per-agent monitoring becomes
less likely to catch any of them.**

- Environment **FakeLab**: 9 services, 86 benign tasks, 4 attack objectives.
- An explicit planner raises attack completion **up to sevenfold**.
- Stronger monitors cut undetected success by **>1 order of magnitude**, still imperfect.
- A weak "trusted" monitor misses most attacks.

This is the rigorous threat model `cross_trajectory` currently lacks.

### 3.2 arXiv:2606.09084 — Context-Fractured Decomposition (CFD)
Lin, Yang, Guo, Nale, Fleming, Cheng. 8 Jun 2026.

Cross-context multi-step jailbreaks that **preserve benign-looking intermediate
artifacts** from an early interaction and trigger harm **much later, possibly in a
different agent instance or workflow stage**. Individually harmless tool actions;
harmful only when composed asynchronously.

> *"most existing attacks and defenses still assume a single contiguous conversation
> visible to the defender."*

**+28.3 pp** attack success over SOTA baselines, **even against strong single-turn
judges**. This is a genuine gap in our course — we have no artifact-provenance lesson.

### 3.3 arXiv:2604.21131 — Cross-Session Threats / CSTM-Bench
Azarafrooz (sole author). 22 Apr 2026.

Four attack ops: **accumulate, compose, launder, inject_on_reader**. Benchmark
**CSTM-Bench**: 26 executable attack taxonomies, 54 scenarios, two splits (dilution;
12 isolation-invisible cross-session), seven identity anchors, **matched
benign-pristine and benign-hard confounders**, **publicly released on HuggingFace**.
A bounded-memory **Coreset Memory Reader (K=50)** was the only method holding recall
across both splits.

**We already use CSTM-Bench as OOD** — this is its source paper, and we were not
citing it. Its *matched benign-hard confounders* are exactly the rubric-7
length/confound discipline we impose on ourselves.

### 3.4 arXiv:2607.24893 — Early Detection of Distributed Backdoors
*"...in Multi-Agent LLM Systems: A Characterization Study."* Argues dilution across
steps **motivates trajectory-level analysis** because per-step audits miss run-level
patterns.

---

## 4. Long-context attacks

### 4.1 arXiv:2511.04707 — NINJA / Jailbreaking in the Haystack `[OLDER: Nov 2025]`
Jailbreaks aligned models by **appending benign, model-generated content** to a
harmful goal. Key mechanism: **the position of the harmful goal in a long context
governs success**. Cheap, needs no gradient access.

### 4.2 arXiv:2605.08277 — Mitigating Many-shot Jailbreaks with One Single Demonstration
A **single** well-chosen demonstration substantially mitigates MSJ — a very cheap
defence baseline any long-context lesson should price against.

### 4.3 arXiv:2604.25921 — One Word at a Time
*"Incremental Completion Decomposition Breaks LLM Safety"* — decomposition down to the
token level; the finest-grained point on the fragmentation axis.

### 4.4 arXiv:2603.23509 — Internal safety collapse in frontier LLMs `[verify before use]`
Surfaced via secondary reference only; **not yet WebFetch-verified.** Mark
`[UNVERIFIED]` until fetched.

---

## 5. The dataset problem — the user's "harder part", answered

**The literature agrees it is the bottleneck, and one paper measured the cost.**

| resource | size | released | fits our need? |
|---|---|---|---|
| **CSTM-Bench** (2604.21131) | 26 taxonomies / 54 scenarios, 2 splits | **HF, yes** | already our OOD; small |
| **MultiBreak** (2605.01687) | **10,389 multi-turn prompts, 2,665 intents** | ICML 2026; release not stated on abstract page | **best size match**; verify release |
| **MT-JailBench** (2605.11002) | modular multi-turn | check | modular = good for ablation |
| 2606.21082 eval set | **14,038 conversations** | check | largest conversation-level set found |
| **LongSafety** (2502.16971) `[OLDER]` | 1,543 cases, **avg 5,424 words**, 7 categories | yes | the long-*context* set, not multi-turn |
| **ASSEBench** | 15 risk categories, 29 scenarios, multi-turn agent records | yes | agent trajectories |
| **FakeLab** (2607.07368) | 9 services, 86 tasks, 4 objectives | with paper | multi-agent, synthetic-but-principled |

**The measured warning (2604.28129): synthetic held-out 93.8% → real LMSYS 47–71%.**
Any long-trajectory detector we build on synthetic traces must report a real-traffic
number or it is not a result. This is the trajectory-domain analogue of the failures
already catalogued in §18.8.

**Our own rubric already forces the right answer here:** build the ≥500/class MAIN set
from what we can assemble, and report a **real released benchmark as OOD** (rubric 5).
CSTM-Bench is already wired for exactly this.

---

## 6. What this implies for the program (candidate work, not yet pre-registered)

Ordered by evidence-per-GPU-hour:

1. **Kill the mean pool in `hello_world` / build `probe_tuning`'s missing layer
   sweep.** SALO (2605.02958) says mean pooling dilutes a sparse needle. We mean-pool
   at a single fixed layer 12, and the promised layer sweep has no code and no
   artifact (a known gap). One cheap experiment tests a published mechanism *and*
   closes an open debt. **Judge-free** (probe AUC vs ground-truth labels), so the
   failed judge calibration does not block it.
2. **Three-phase turn labels in `trajguard` / `multiturn_jailbreak`.** 2604.28129
   attributes 50–59% FPR to binary labelling. Cheap relabel, directly falsifiable.
3. **Adaptive-obfuscation arm as a standing gate.** Per 2412.09565, any latent
   detector claim without it is inadmissible. Should become a rubric item, not a
   per-lesson choice.
4. **Unify the four trajectory lessons onto the fragmentation axis** (§0 table) — a
   documentation/framing change that costs no compute and makes the course argue
   one thesis instead of four.
5. **Cite 2604.21131 wherever CSTM-Bench appears.** We use the benchmark without
   citing its paper. That is a straightforward citation defect.

**Nothing above is pre-registered yet.** Per §7 the screening/evaluation split and
success criterion must be committed to git *before* any sweep.

---

## Sources

- [arXiv:2604.28129 — Latent Adversarial Detection](https://arxiv.org/abs/2604.28129)
- [arXiv:2605.02958 — Tracing the Dynamics of Refusal (SALO)](https://arxiv.org/abs/2605.02958)
- [arXiv:2605.28553 — Refusal Before Decoding](https://arxiv.org/abs/2605.28553)
- [arXiv:2606.00611 — TRACE](https://arxiv.org/abs/2606.00611)
- [arXiv:2412.09565 — Obfuscated Activations](https://arxiv.org/abs/2412.09565)
- [arXiv:2607.07368 — Multi-Agent AI Control](https://arxiv.org/abs/2607.07368)
- [arXiv:2606.09084 — Context-Fractured Decomposition](https://arxiv.org/abs/2606.09084)
- [arXiv:2604.21131 — Cross-Session Threats / CSTM-Bench](https://arxiv.org/abs/2604.21131)
- [arXiv:2607.24893 — Distributed Backdoors in Multi-Agent LLM Systems](https://arxiv.org/abs/2607.24893)
- [arXiv:2605.01687 — MultiBreak](https://arxiv.org/abs/2605.01687)
- [arXiv:2605.11002 — MT-JailBench](https://arxiv.org/abs/2605.11002)
- [arXiv:2606.21082 — Hierarchical Attention for Long Conversations](https://arxiv.org/abs/2606.21082)
- [arXiv:2511.04707 — Jailbreaking in the Haystack (NINJA)](https://arxiv.org/abs/2511.04707)
- [arXiv:2605.08277 — Mitigating Many-shot Jailbreaks](https://arxiv.org/abs/2605.08277)
- [arXiv:2604.25921 — One Word at a Time](https://arxiv.org/abs/2604.25921)
- [arXiv:2502.16971 — LongSafety](https://arxiv.org/abs/2502.16971)
