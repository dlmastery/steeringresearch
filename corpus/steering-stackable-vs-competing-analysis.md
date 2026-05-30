# LLM Steering Methods: Stackability vs. Competition Analysis
### Focus: Conditional / Activation Steering for Small Gemma Models (1B/2B), reproducible on RTX 4090
### Synthesized from captured Grok research transcripts (Conditional Steering, SAE-vs-Baselines, Gemma-Scope) — SOTA as of late May 2026

> **Provenance & caveat.** This document synthesizes the two verbatim Grok transcripts harvested on 2026-05-29. Grok's *paper titles and arXiv IDs* were largely real on independent arXiv spot-checks, but Grok **fabricates model-usage and quantitative details** (e.g. it invented "Gemma-4-E4b-it" for the MAGS paper — Gemma-4 does not exist; newest open family is Gemma-3). Treat every quantitative number and "uses Gemma-X" claim below as **[NEEDS VERIFICATION]** until checked against the actual PDF. The *stackability/competition logic* here is derived from each method's intervention site and mechanism, which is the most trustworthy axis to reason about.

---

## 1. The Core Organizing Principle

Whether two steering methods **stack** (compose cleanly) or **compete** (interfere / are mutually exclusive) is governed almost entirely by **WHERE in the forward pass they intervene** and **HOW they modify the signal there**. Two methods stack when they (a) act on disjoint intervention sites, or (b) act on the same site through an operation that is mathematically composable (additive, or projection onto orthogonal subspaces). They compete when they (a) overwrite the same site with incompatible transformations (e.g. one adds a vector, another rotates the same plane), or (b) consume the same "budget" (norm, attention mass, decoding distribution) such that applying both degrades each.

### 1.1 The intervention-site taxonomy (the axis that decides everything)

| Layer of the stack | Intervention site | Representative method families |
|---|---|---|
| **Decoding / logits** | output token distribution | DoLa & contrastive decoding |
| **Residual stream** | hidden state \(h\) (additive) | ActAdd, CAA, RepE, ITI, refusal-direction, SAE-feature steering, FLAS |
| **Residual stream** | hidden state \(h\) (rotational) | Angular Steering, Spherical Steering, Selective Steering |
| **Attention internals** | Q / K / V vectors | DISCO (Q/V), SKOP (key-orthogonal) |
| **Attention scores** | post-/pre-softmax weights | PASTA, SpotLight |
| **KV cache** | cached keys/values | KV-Cache Steering, Memory Inception, MAGS |
| **Weights (frozen + low-rank)** | learned interventions on \(h\) | ReFT/LoReFT, HyperSteer, BiPO/RePS |
| **Gating / conditioning (meta-layer)** | a switch that decides *whether* to apply any of the above | CAST, FineSteer (SCS), Selective Steering (layer gate), persona-vector preventative steering |

The single most important insight from the corpus: **conditional steering (CAST and its descendants) is not a peer method — it is a meta-layer.** It wraps any of the additive/rotational behavior methods and gates them on input context. This makes the conditioning machinery *orthogonal* (and therefore stackable) with almost every behavior-injection method below it.

---

## 2. STACKABLE COMBINATIONS (compose cleanly)

### 2.1 Conditioning meta-layer × any behavior vector — the canonical stack
- **CAST (Conditional Activation Steering, 2409.05907)** computes a *condition vector* (PCA on contrastive prompt activations) and fires a behavior vector only when \(\text{sim}(h, \text{proj}(h\mid c)) > \theta_c\). Because the condition check is a read-only similarity probe at an early layer and the behavior injection is the standard additive CAA-style edit at a later layer, **CAST stacks on top of essentially any additive behavior vector** (ActAdd, CAA, refusal-direction, SAE-feature, FLAS). This is the primary mechanism for "apply refusal vector only when input is about hate speech."
- **FineSteer's SCS (Subspace-guided Conditional Steering, 2604.15488)** is a direct CAST successor using energy-ratio gating; same stacking property.
- **Selective Steering's discriminative-layer gate (2601.19375)** restricts steering to layers where class means have opposite sign — another orthogonal gate that composes with the underlying edit.

**Why they stack:** the gate operates on a *different operation type* (decide) than the injection (modify), and consumes no norm/attention budget.

### 2.2 Multiple NON-INTERACTING behavior vectors for distinct safety properties
This is the user's explicit interest: can you run several steering vectors for different safety issues simultaneously without them fighting?
- **Conceptors (2410.16314)** represent activation sets as ellipsoidal regions and provide **AND / OR / NOT Boolean composition** of multiple steering goals; the transcript reports "the AND-combined conceptor outperformed the mean-combined steering vectors" — i.e. conceptors are an explicit *compositional algebra* for stacking vectors. **[NEEDS VERIFICATION]**
- **CAST logical composition (OR/AND/complement of condition vectors)** lets you build "if hate-speech OR adult-content, then refuse" — multiple conditions, one or more behaviors.
- **EasyEdit2 vector merging (2504.15133)** implements Linear / TIES / DARE-TIES merges to combine e.g. safety + sentiment vectors into a single intervention.
- **Persona Vectors (2507.21509, Anthropic)** define separate directions for distinct traits (evil, sycophancy, hallucination) — by construction a *bank* of vectors intended to be monitored/applied for different properties.

**Why they stack — and the catch:** Independent behavior vectors stack additively **only while they remain near-orthogonal and the summed norm stays in-distribution.** Once vectors overlap in direction or the cumulative norm pushes \(h\) off-manifold, they start competing (see §3.3). Conceptors and orthogonalized/targeted vectors (SAE-TS) exist precisely to *preserve* non-interaction.

### 2.3 Different intervention sites — naturally orthogonal
- **Residual-stream addition (CAA) × KV-cache steering (2507.08799)** — one edits \(h\), the other edits cached K/V; different sites, compose in principle.
- **Residual addition × attention-score reweighting (PASTA 2311.02262 / SpotLight 2505.12025)** — \(h\)-edit vs. attention-mass edit; orthogonal sites.
- **Residual addition × contrastive decoding (DoLa 2309.03883)** — DoLa acts at the logits by contrasting layers; it sits *after* all residual edits and composes with them.
- **SAE-feature steering × KV/attention methods** — feature clamp on residual, plus an attention/KV lever, hit different sites.

### 2.4 SAE-targeting to make additive vectors cleaner (side-effect reduction)
- **SAE-TS (2411.02193)** uses SAEs to *measure side effects* of any steering vector and optimize a replacement that activates only the desired feature while suppressing others. This is a "make-it-stack-better" wrapper: it actively orthogonalizes a vector against unwanted features so that it interferes less with other behaviors.
- **FGAA (2501.09929)** combines CAA + SAE-TS in SAE latent space; reported to outperform CAA / SAE-decoder / SAE-TS individually. **[NEEDS VERIFICATION]**

### 2.5 Weight-level low-rank interventions × inference-time vectors
- **ReFT/LoReFT (2404.03592)** learns low-rank interventions on frozen-model representations; **HyperSteer (2506.03292)** generates steering vectors from a hypernetwork conditioned on a natural-language prompt. Both produce residual-stream edits and can in principle layer with an inference-time additive vector, though see §3 for the norm-budget caveat.

---

## 3. COMPETING / MUTUALLY-EXCLUSIVE METHODS (interfere)

### 3.1 Additive vs. Rotational edits on the SAME residual subspace — directly competing
- **CAA / ActAdd (additive)** push \(h\) by adding a fixed vector (increasing norm).
- **Angular Steering (2510.26243)** and **Spherical Steering** instead **rotate** \(h\) within a 2D plane *without increasing norm*; **Selective Steering (2601.19375)** is a norm-preserving rotation that explicitly fixes Angular Steering's norm-violation flaw on small (<7B) models.
These two philosophies target the *same* refusal/behavior plane with *incompatible* operations. You pick one: either you add along the direction or you rotate toward it. Applying both to the same plane double-counts and tends to push \(h\) off-manifold → incoherence. **They compete by construction.**

### 3.2 SAE-feature steering vs. simple baselines (prompting / probes / finetuning) — empirically competing for the same job
This is the headline tension in the corpus:
- **AxBench (2501.17148, Stanford)** verdict (verbatim from transcript): *"On both evaluations [steering and concept detection], SAEs are not competitive."* Prompting beats everything for steering; linear probes beat everything for detection.
- **DeepMind GDM negative results** ("SAEs underperformed linear probes").
- **Kantamneni et al. / Farrell et al.** — SAE probes win <3% of the time vs. strong baselines.
These are **competing approaches to the same goal**: if a prompt or a finetune already achieves the behavior, an SAE-feature steer is largely redundant and usually worse. They are not stacked in practice — you choose the cheaper/stronger one. The nuance: **Arad/Mueller/Belinkov (2505.20063)** show SAE steering becomes competitive *only if you select the right features* (output-score filtering), implying naïve SAE steering competes-and-loses but curated SAE steering can re-enter the additive stack.

### 3.3 Multiple additive vectors competing for the NORM / in-distribution budget
Even "independent" safety vectors compete once their sum moves \(h\) outside the natural activation manifold:
- **"Steered LLM Activations are Non-Surjective" (2604.09839)** proves steered states can lie off the manifold reachable by any prompt — stacking too many additive edits compounds this. **[NEEDS VERIFICATION of the Gemma-3-1B/Llama-3.2-1B/Qwen-0.5B claims]**
- **In-Distribution Steering / IDS (2510.13285)** exists specifically to keep stacked edits in-distribution; its existence is evidence that naïve multi-vector stacking competes for coherence budget.
- **O'Brien et al. (2411.11296)** report SAE refusal-clamping reduces jailbreaks **but causes MMLU regression and over-refusal** — a direct safety-vs-capability competition. Stacking more safety strength competes against general capability.

### 3.4 KV-cache contamination — residual steering competing with attention routing
- Residual-stream steering can **contaminate the KV cache** for downstream tokens. **SKOP (2605.06342, key-orthogonal projection)** and **GCAD (2605.10664, attention-delta gating)** are explicit fixes — meaning unmitigated residual steering *competes with* clean attention routing over long generations. **[NEEDS VERIFICATION; GCAD flagged [UNVERIFIED] in source]**

### 3.5 Static vs. dynamic attention steering — substitutes, not complements
- **PASTA (2311.02262)** profiles heads offline and applies a *static* reweight. **SpotLight (2505.12025)** monitors attention mass and applies *dynamic, only-when-needed* bias. They solve the same problem two ways; you would not run both on the same heads — they compete as design choices.

---

## 4. DECISION MATRIX (pairwise)

Legend: ✅ stack · ⚠️ stack with care (norm/coherence budget) · ❌ compete (pick one)

| | CAA / ActAdd (additive) | Angular/Selective (rotational) | SAE-feature steer | KV-cache steer | Attention-score (PASTA/SpotLight) | DoLa (decoding) | CAST gate |
|---|---|---|---|---|---|---|---|
| **CAA / ActAdd** | ⚠️ (orthogonalize; watch norm) | ❌ same plane | ⚠️ (SAE-TS to de-conflict) | ✅ diff site | ✅ diff site | ✅ after-stack | ✅ gated |
| **Angular/Selective** | ❌ | ⚠️ multi-plane only | ❌ same subspace | ✅ | ✅ | ✅ | ✅ |
| **SAE-feature steer** | ⚠️ | ❌ | ⚠️ (multi-feature) | ✅ | ✅ | ✅ | ✅ |
| **KV-cache steer** | ✅ | ✅ | ✅ | ⚠️ (SKOP to avoid contamination) | ⚠️ shared attention pathway | ✅ | ✅ |
| **Attention-score** | ✅ | ✅ | ✅ | ⚠️ | ❌ PASTA vs SpotLight | ✅ | ✅ |
| **DoLa (decoding)** | ✅ | ✅ | ✅ | ✅ | ✅ | — | ✅ |
| **CAST gate** | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ⚠️ compose conditions via AND/OR |

---

## 5. RECOMMENDED STACK FOR MULTIPLE NON-INTERACTING SAFETY VECTORS (Gemma-2-2B on a 4090)

A concrete, mechanism-grounded recipe (numbers to be verified empirically on your harness):

1. **Meta-layer (gating):** CAST condition vectors, one per safety property (hate-speech, self-harm, illegal-advice…), composed with OR. Read-only early-layer probes — near-zero interference.
2. **Behavior layer (injection):** one additive behavior vector per property, **orthogonalized via SAE-TS** so the vectors are mutually near-orthogonal → preserves non-interaction (§2.2/§2.4).
3. **Coherence guard:** apply IDS-style in-distribution constraint (or cap cumulative norm) so the summed edits stay on-manifold (§3.3).
4. **Long-generation hygiene:** if generations are long, route the edit through a key-orthogonal projection (SKOP) to avoid KV-cache contamination (§3.4).
5. **Do NOT** also rotate the same planes (Angular/Selective) — choose additive *or* rotational, not both (§3.1).
6. **Optional, free, composable:** DoLa at decode time for a factuality bump — it sits after everything (§2.3).

**What competes with this stack:** plain prompting and finetuning often match or beat naïve SAE steering (AxBench), so use SAE features only when curated/orthogonalized; and avoid stacking so many vectors that capability regresses (O'Brien over-refusal).

---

## 6. SUMMARY TABLE — Methods by composability role

| Role | Methods | Composability |
|---|---|---|
| **Meta-gates (wrap others)** | CAST, FineSteer-SCS, Selective layer-gate, persona preventative steering | Stack on almost everything |
| **Additive behavior injectors** | ActAdd, CAA, RepE, ITI, refusal-direction, FLAS | Stack with each other only if orthogonal + norm-bounded |
| **Rotational injectors** | Angular, Spherical, Selective | Compete with additive on same plane |
| **Composition algebras** | Conceptors, EasyEdit2 merges, SAE-TS targeting | Purpose-built to make injectors non-interacting |
| **Disjoint-site levers** | KV-cache steering, DISCO (Q/V), PASTA/SpotLight (attn scores), DoLa (logits) | Stack with residual methods (mind shared attention/KV) |
| **Learned low-rank** | ReFT/LoReFT, HyperSteer, BiPO/RePS | Stack with gates; mind norm budget |
| **Competing baselines** | prompting, linear probes, finetuning | Substitutes for SAE steering (often win) |

---

*Generated from verbatim Grok transcripts (2026-05-29). Stackability reasoning is mechanism-based; all quantitative/Gemma-usage claims carry [NEEDS VERIFICATION] and should be confirmed against source PDFs before use in a reproduction harness.*
