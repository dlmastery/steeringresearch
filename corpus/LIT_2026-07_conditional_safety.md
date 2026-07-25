# Literature scan — Conditional / Gated Steering and Steering-Based Safety Guardrails

**Verified on 2026-07-25.** Every arXiv id below was checked by fetching its `arxiv.org/abs/` page
and confirming title + authors + submission date, unless explicitly marked `[UNVERIFIED]`
(surfaced by search but the abs page was not fetched in this pass — do NOT cite these until fetched).
Scope: prioritized June–July 2026 (`2606.*`, `2607.*`), with earlier 2026 work included where it
directly bears on our novelty claims.

---

## Executive summary

1. **Gating is no longer a contribution — it is the baseline.** By mid-2026 the field assumes
   steering is conditional. A linear probe on hidden states as the gate, plus per-input adaptive
   strength, is the default architecture. `2606.29441` (28 Jun 2026) already publishes the first
   *unified head-to-head* of unconditional steering vs probe-gated defenses on identical models,
   attacks and metrics. Our "probe gates the steering vector" composition is a reproduction of
   published work, not a novel contribution.

2. **Our WRITE-side negative is now the predicted outcome, not a surprise.** `2607.02714`
   (2 Jul 2026, 24 open models) reports that refusal occupies a **multi-dimensional subspace**
   distributed across layers. A single diff-of-means direction is not expected to restore refusal
   in an abliterated model. Publishing our result as an anomaly would be a mistake; publishing it
   as a *scale-checked confirmation with the read/write asymmetry quantified* (probe AUC ~0.99 vs
   WRITE-side gibberish-only) is defensible and consistent with `2606.29441`.

3. **Novelty triage of our three proposed contributions:**
   - **(i) Conformal / certified gate with a provable over-refusal bound → STILL OPEN.** No paper
     found puts split-conformal calibration on an *activation-space* gate for LLM refusal with an
     FPR/over-refusal guarantee. All ingredients are published separately. This is our strongest
     claim, and it is a ~3-month window, not a 12-month one.
   - **(ii) Displacement / norm budget as a coherence controller → LARGELY SCOOPED. Demote from
     headline.** `2602.02712` (strength theory, non-monotone laws, 11 models) + `2606.06735`
     (angle–norm decomposition, norm↔perplexity tradeoff) + `2604.09839` (off-manifold proof)
     collectively own this. A "Δ‖h‖ correlates with incoherence" result in 2026 is a replication.
   - **(iii) Direction fungibility (real vs shuffled/random norm-matched control) → SUBSTANTIALLY
     SCOOPED unconditionally, SURVIVES under a gate.** Rogue Scalpel (`2509.22067`), the
     non-identifiability paper (`2602.06801`) and the radiology random-vector controls
     (`2606.20852`) already show unconditional direction non-specificity. Nobody has tested
     fungibility **under a gate**, nor in the abliterated-model refusal-restoration setting.

4. **Practical warning for our harness:** `2606.15980` shows activation probes go **stale** under
   fine-tuning-style model updates (quantization is mostly safe, QLoRA+quant is worst). Our
   AUC-0.99 probe has a shelf life and any certified bound must be re-calibrated after a model
   update, or the guarantee is void. This is a concrete constraint on claim (i) and should be
   pre-registered as a limitation.

---

## (a) Table of most relevant recent work

| Title | arXiv id | Date | Contribution | Relevance to us |
|---|---|---|---|---|
| Closing the Activation-Cone Blind Spot: Response-Time Probing and Unified Defense — Subhadip Mitra | 2606.29441 | 28 Jun 2026 | First unified evaluation of unconditional steering vs **probe-gated** defenses on identical models/attacks/metrics. Shows prompt-time activation defenses structurally cannot see prefilling attacks; a response-time linear probe on the first generated tokens gets AUROC 0.97–1.00 across 7 models; composed with AlphaSteer gives DSR 0.983 (Mistral) / 0.994 (Llama). Code + data released. | **Our closest competitor.** Probe-gates-steering is now a published, benchmarked baseline. Read for what it does NOT do: no conformal calibration, no over-refusal bound, no direction-fungibility control. |
| Not All Refusals Are Equal: How Safety Alignment Fails Cybersecurity at Scale — Hadetskyi, Pasquini, Sorokin | 2607.02714 | 2 Jul 2026 (v2 7 Jul) | 24 open-source LLMs: **refusal occupies a multi-dimensional subspace within layers** and is distributed widely across layers (esp. trillion-param MoE). Demonstrates *domain-specific* abliteration; classifies models into 3 abliteration-susceptibility tiers; safety-training type + architecture are the strongest predictors. | Directly explains our diff-of-means WRITE negative at 1B and 4B. Gives the model taxonomy needed to state which models our result generalizes to. |
| On the Non-Identifiability of Steering Vectors in Large Language Models — Venkatesh, Kurapath | 2602.06801 | 6 Feb 2026 (v4 1 Apr) | Steering vectors are not uniquely identifiable: orthogonal perturbations achieve **near-equivalent efficacy with negligible effect sizes** across multiple models, validated by semantic classifiers. Establishes equivalence classes of behaviorally indistinguishable interventions. | Pre-empts a naive "the direction doesn't matter" claim. Our fungibility test must be framed *under the gate* to survive this. |
| Towards Understanding Steering Strength — Taimeskhanov, Vaiter, Garreau | 2602.02712 | 2 Feb 2026 (v2 8 Jul 2026) | First theoretical analysis of steering **magnitude**: precise qualitative laws for next-token probability, concept presence, and cross-entropy; establishes **non-monotonic** effects of strength. Validated on 11 models from small GPT to modern LLMs. | Pre-empts a naive norm-budget claim. Any displacement budget we propose must engage these non-monotonicity laws explicitly or it dies in review. |
| A Geometric Account of Activation Steering through Angle–Norm Decomposition — Aparin, Gaintseva | 2606.06735 | 4 Jun 2026 (v2 8 Jun) | Decomposes interventions into angular alignment + radial magnitude. Concepts live primarily in **angular** structure (justifying spherical steering), but hidden-state norm still governs stability and downstream effects. Argues against single-coefficient α that entangles the two. 7 models. | Our Δ‖h‖ budget is only a contribution if it is angle/radius-decomposed and improves on this parameterization. Otherwise it is a re-derivation. |
| Steered LLM Activations are Non-Surjective — Mishra, Khashabi, Liu | 2604.09839 | 10 Apr 2026 (v2 7 May) | Proves activation steering pushes the residual stream **off the manifold of states reachable from discrete prompts**; almost surely no prompt reproduces the steered internal behavior. Formal separation between white-box steerability and black-box prompting. Validated on 3 LLMs. | Formalizes the off-manifold premise our displacement-budget hypothesis rests on. Cite as the foundation, not as our finding. |
| Adaptive Probe-based Steering for Robust LLM Jailbreaking — Chen, Dong, Xie (ICML 2026) | 2605.20286 | 19 May 2026 | Attack. Uses model-extraction to guide learned steering vectors toward the ideal one; **automatic strength tuning from contrastive activation statistics** (no manual α, no extra contrastive prompts). Raises harmful response rate 6% → 70% on fortified LLMs. Code released. | The attack our gate must survive at Rung 4. Also shows adaptive per-input strength is now standard on the *attack* side too. |
| Contextual Linear Activation Steering (CLAS) — Hsu, Beaglehole, Radhakrishnan, Belkin | 2604.24693 | 27 Apr 2026 | Dynamically adapts linear steering to **context-dependent steering strengths** rather than uniform application. Competitive with ReFT and LoRA under limited labeled data across benchmarks and model families. | Per-input adaptive α is the 2026 default. Our gate must be compared against CLAS, not against fixed-α CAA. |
| Do Activation Monitors Survive Model Updates? Benchmarking, Predicting, and Repairing Activation-Monitor Staleness — Evan Duan | 2606.15980 | 14 Jun 2026 (v2 18 Jun) | Quantization-style updates largely preserve frozen probes; **fine-tuning-style updates frequently make probes stale**. QLoRA+quantization is worst. Privacy probes degrade more than refusal-compliance probes. Degradation is predictable from pre-deployment features; label-free activation realignment repairs every repair-relevant stale cell. | Hard constraint on any certified gate: the conformal guarantee is void after a fine-tune. Must be a pre-registered limitation. |
| Multi-Adapter Representation Interventions via Energy Calibration — Yu, Li, Chen, Li, Singh, Cao, Hu | 2605.28722 | 27 May 2026 | Competitive multi-adapter mechanism where experts adaptively set intervention direction and strength per sample; **energy-based gating module** using internal propagation dynamics decides which inputs are applicable for intervention. Evaluated on TruthfulQA, BBQ, MMLU, safety benchmarks. | Learned/energy gate — **no** distribution-free guarantee. This is the nearest "gate" work that leaves our conformal slot open. |
| Learning What to Say to Your VLA: Mostly Harmless Vision Language Action Model Steering — Jeong, Swamy, Bajcsy | 2606.12299 | 10 Jun 2026 | Learns an improvement predictor for when language steering helps, then **conformalizes the improvement head to prevent harmful steering interventions** OOD. +24.7% sim / +65.0% hardware on frozen models. | **The single closest thing to a conformal steering gate that exists.** But it is VLA/robotics, language-level steering, and the guarantee is against performance degradation — not an over-refusal bound on an LLM safety gate. |
| AgentLens: Interpretable Safety Steering via Mechanistic Subspaces for Multi-Turn Coding Agent — Luo, Zhang, Quan, Jin, Cai, Xiao, Niu, Xiang | 2606.22673 | 21 Jun 2026 | Detects harmful execution states from step-level hidden representations and mitigates by intervening in a **10-dimensional subspace**; introduces the Mechanistic Agent Safety benchmark (194 tasks, annotated multi-turn trajectories, 3 models). Uses counterfactual augmentation so the linear probe does not memorize malicious-command syntax. | Multi-dimensional (not rank-1) intervention + probe detection, agentic setting. Confirms the field has moved past single-direction writes. |
| Analysing the Safety Pitfalls of Steering Vectors — Li, Fastowski, Zaradoukas, Prenkaj, Kasneci | 2603.24543 | 25 Mar 2026 | Systematic JailbreakBench audit of CAA steering vectors: steering can amplify jailbreak success by up to **57%** or reduce it by **50%**, attributed to overlap between steering vectors and latent refusal directions. | Rogue-Scalpel-family confirmation that steering is a two-edged safety primitive. Supports our §10 mandate. |
| Activation Steering Induces Emergent Misalignment: A More Comprehensive Evaluation — Cao, Lou, Liu, Feng, Li, Ng, Luu | 2606.08682 | 7 Jun 2026 | Steered models show **broader emergent misalignment than finetuned ones**, with harmful responses of stronger semantic relevance and higher coherence. Examines steering magnitude, low-rank structure, and layer selection across families and scales. | Strengthens the case that steering-as-defense needs a certified gate rather than a bigger α. Note: coherence here is an observed property, not a formal metric — do not cite it as a coherence-budget result. |
| Translating Inference-Time Control to Radiology VLMs: Activation Steering for Pneumonia Classification on Chest X-rays — de Mattos Farina, Esmeraldo, Matsuoka, Kuriki, Kitamura | 2606.20852 | 18 Jun 2026 | Applies activation steering to radiology VLMs with **reverse-vector controls, random-vector controls, and conditional bootstrap CIs**. Reports conditions where the selected steering vector does not separate from the distribution of 20 random unit vectors at matched layer and absolute scale. | The methodological template for a random-direction control done properly (matched norm + bootstrap CI). Our fungibility test should adopt this protocol. |
| The Rogue Scalpel: Activation Steering Compromises LLM Safety (ICLR 2026, Trustworthy AI) | 2509.22067 | Sep 2025; v2 Mar 2026 | Steering **random** directions raises harmful compliance 0% → 1–13%; steering benign SAE features is comparably harmful; combining 20 random vectors that jailbreak a single prompt yields a **universal attack** on unseen requests with no harmful data, weights, gradients, or logits. | Our red-team probe (§10). Also the original direction-non-specificity result — our fungibility framing must differentiate from it. |
| Obfuscated Activations Bypass LLM Latent-Space Defenses | 2412.09565 | Dec 2024 | SAE-based, representation-probing and latent-OOD defenses are all vulnerable to obfuscated activations; against harmfulness probes, recall can drop 100% → 0% while retaining ~90% jailbreak success. | The standing attack against *our* probe gate. Any certified bound is exchangeability-conditional and this attack breaks exchangeability. Must be stated. |
| When Can Conformal Risk Control Certify LLM Outputs? Bounds, Impossibility, and Adaptation for Structured Generation | 2606.29054 | Jun 2026 | `[UNVERIFIED]` — search-surfaced. Studies when conformal procedures succeed under task-specific structured losses and the minimum abstention required when they cannot. | Directly relevant impossibility results for claim (i). **Fetch the abs page before citing.** |
| Conformal Selective Acting: Anytime-Valid Risk Control for RLVR-Trained LLMs | 2605.20270 | May 2026 | `[UNVERIFIED]` — search-surfaced. Reported as the only method among ten satisfying pathwise validity and non-refusing deployment on every cell under adversarial distribution shift. | Output-level, not activation-level. The nearest conformal-safety competitor conceptually. **Verify before citing.** |
| Proactive Routing to Interpretable Surrogates with Distribution-Free Safety Guarantees | 2603.14623 | Mar 2026 | `[UNVERIFIED]` — search-surfaced. Reported to show conformal guarantees hold for any gate score independent of calibration, with calibration affecting only efficiency/coverage. | If accurate, this is a *theoretical* precedent that weakens the "our gate score is special" framing — the guarantee is score-agnostic. Important. **Verify.** |
| AEGIS: A Backup Reflex for Physical AI | 2606.06660 | Jun 2026 | `[UNVERIFIED]` — search-surfaced. Reported to use split-conformal calibration to convert per-step scores into triggers, with an early-harm gate. | Conformal-calibrated safety trigger, but physical AI / robotics domain. **Verify.** |
| Selective Steering: Norm-Preserving Control Through Discriminative Layer Selection | 2601.19375 | Jan 2026 | `[UNVERIFIED]` — search-surfaced. Norm-preserving steering with discriminative layer selection. | Directly overlaps our norm-budget claim (ii). **Verify — this one matters for our novelty assessment.** |
| BarrierSteer: LLM Safety via Learning Barrier Steering | 2602.20102 | Feb 2026 | `[UNVERIFIED]` — search-surfaced. Reported comparison of linear probes to nonlinear safety barriers: replacing neural barriers with linear probes raised harmful rate 0.86% → 2.89%. | If accurate, evidence that a *linear* probe gate is not the ceiling — relevant to how we specify our gate. **Verify.** |
| Latent-space Attacks for Refusal Evasion in Language Models | 2605.21706 | May 2026 | `[UNVERIFIED]` — search-surfaced. | Attack surface for gated defenses. **Verify.** |
| Steering Beyond the Support: Adversarial Training on Unsupervised Jailbroken Activation Simulation | 2605.24535 | May 2026 | `[UNVERIFIED]` — search-surfaced. | Defense-side adversarial training in activation space. **Verify.** |

---

## (b) STATE OF CONDITIONAL STEERING 2026

### What is now standard (assumed, not argued)

- **Conditioning is the default; unconditional steering is the ablation.** CAST (arXiv:2409.05907,
  ICLR 2025 Spotlight) is no longer a method to be extended — it is the framing everyone inherits.
  Papers now say "conditional variants gate when interventions fire to improve selectivity" as
  background, not as a claim.
- **A linear probe on hidden states is the default gate.** `2606.29441`, `2606.22673`, `2605.20286`
  all use one. Probe AUROC in the 0.97–1.00 range on harmful/benign is a *reported baseline number*,
  not a headline result. Our AUC ~0.99 is on-trend, not exceptional.
- **Per-input adaptive strength is standard.** CLAS (`2604.24693`), adaptive-probe steering
  (`2605.20286`), atomic-unit steering (AUSteer, `2602.04428` `[UNVERIFIED]`). Fixed-α CAA is a
  strawman baseline in 2026.
- **Steering is accepted to be off-manifold** (`2604.09839`, proved) and to have **non-monotonic
  strength effects** (`2602.02712`, theory + 11 models). Norm inflation → perplexity/capability
  degradation is a known, published mechanism (`2606.06735`).
- **Geometric constraints on refusal steering are expected** — null-space projection (AlphaSteer,
  arXiv:2506.07022 `[UNVERIFIED]`), on-manifold pullback / encoder-Jacobian methods, angle-vs-norm
  parameterization. A bare additive diff-of-means vector is a 2024 baseline.

### What is contested

1. **Is refusal one direction?** `2607.02714` says no: multi-dimensional subspace, distributed
   across layers, with model-dependent susceptibility tiers. The single-direction result
   (arXiv:2406.11717) is now treated as a special case for certain aligned models, not a general
   law. Defenses explicitly aim to *disperse* refusal across dimensions (extended-refusal training,
   arXiv:2505.19056 `[UNVERIFIED]`) precisely to break single-direction abliteration.
2. **Are steering vectors identifiable at all?** `2602.06801` says no — behaviorally
   indistinguishable equivalence classes exist. This undercuts direction-centric interpretability
   claims across the board.
3. **Is steering even a safe primitive?** Rogue Scalpel (`2509.22067`) + `2603.24543` (±57%/−50%
   jailbreak swing) + `2606.08682` (emergent misalignment broader than finetuning) form a coherent
   case that steering *creates* risk. The field's answer is increasingly "gate it, constrain it
   geometrically, or don't write at all — just detect."
4. **Prompt-time vs response-time monitoring.** `2606.29441` shows prompt-time activation defenses
   structurally miss prefilling attacks; response-time probing on the first generated tokens is
   proposed as the fix. This is a live architectural split.
5. **Detect vs intervene.** The "just use a probe" position is gaining ground: probes are cheap,
   accurate, and don't damage the model. The counter-arguments are (a) probes go stale
   (`2606.15980`), (b) probes are evadable by obfuscated activations (`2412.09565`), and (c) a
   detector cannot *repair* behavior, only block it. Our program sits exactly on this fault line
   and should say so explicitly.

---

## (c) IS OUR PROPOSED CONTRIBUTION STILL NOVEL? — the critical section

Blunt assessment of the three legs of "certified conditional displacement."

### (i) Conformal-calibrated gate with a provable over-refusal bound — **STILL NOVEL, window closing**

**Nothing found that does this.** No paper applies split-conformal / conformal risk control to an
**activation-space** gate for LLM refusal with a distribution-free bound on over-refusal
(harmless-refusal / false-positive rate).

Nearest work, in decreasing order of threat:

- **`2606.12299` (Learning What to Say to Your VLA, 10 Jun 2026)** — genuinely conformalizes an
  intervention gate ("conformalize this improvement head to prevent harmful steering
  interventions"). **But:** VLA/robotics, the steering is *language-level* not activation-level,
  and the guarantee bounds performance degradation OOD, not over-refusal on a safety gate. Distance
  from us: same machinery, different domain, different controlled quantity. This is the paper a
  reviewer will cite at us. We must cite it first and state the delta.
- **`2605.28722` (Multi-Adapter via Energy Calibration, 27 May 2026)** — a per-sample
  *applicability* gate for representation interventions, i.e. exactly our gate's job, but calibrated
  by an energy score with **no distribution-free guarantee**. This is the "learned gate" incumbent
  our certified gate must beat.
- **`2606.29054`, `2605.20270`, `2603.14623`, `2606.06660` [all UNVERIFIED]** — conformal risk
  control applied to LLM outputs, routing, selective acting, and physical-AI triggers. All
  *output-level or non-LLM*. `2603.14623`'s reported claim that conformal guarantees hold for **any**
  gate score independent of calibration is the most dangerous to us: if true, the theory is
  score-agnostic and our contribution reduces to "we applied a known guarantee to an activation
  probe" — an application paper, not a theory paper. **Verify 2603.14623 before committing.**

**Remaining gap:** the *composition* — conformal-calibrated activation-space gate + over-refusal
bound + empirical validation on a steering defense under attack. Every ingredient is published;
the assembly is not. Two caveats that must be pre-registered as limitations, not discovered later:
(a) exchangeability is broken by obfuscated-activation attacks (`2412.09565`), so the bound is
conditional on a non-adaptive adversary; (b) the calibration is void after a fine-tuning-style
model update (`2606.15980`).

**Verdict: NOVEL+TESTABLE, but shallow-moat and time-sensitive. Make this the headline.**

### (ii) Displacement / norm budget as a coherence controller — **LARGELY SCOOPED. Demote.**

This is the weakest leg. The specific claim "off-manifold displacement magnitude predicts
incoherence, therefore budget it" is substantially already in print:

- **`2602.02712` (v2 8 Jul 2026)** — first *theory* of steering strength: precise laws for
  next-token probability, concept presence, and **cross-entropy**, with non-monotonic effects,
  validated on 11 models. Cross-entropy is our coherence proxy. This paper owns the
  strength→degradation relationship.
- **`2606.06735` (4 Jun 2026)** — explicitly separates angle from norm, states that norm carries no
  concept information but governs stability and downstream damage, and reports the
  norm-preservation vs perplexity/capability tradeoff at high angular strength. This paper owns the
  norm→coherence relationship.
- **`2604.09839` (10 Apr 2026)** — *proves* the off-manifold premise. We cannot claim the premise.
- **`2601.19375` (Selective Steering: Norm-Preserving Control) [UNVERIFIED]** — if this is what the
  title suggests, norm-preserving control is already an implemented method. **Verify this id; it is
  the one most likely to fully close leg (ii).**

**Remaining gap (narrow):** the budget survives only in an *operational* form —
(a) as the resource a conformal gate *spends per input* (budget as a control variable inside the
certified loop, not as a standalone empirical correlation), or
(b) as an angle/radius-decomposed budget that measurably improves on `2606.06735`'s
parameterization at matched behavior. Framing (a) is the defensible one because it is only
meaningful *given* the gate; framing (b) is a head-to-head against a June-2026 paper and is a fight
we would probably lose on a 4090.

**Verdict: DERIVATIVE+TESTABLE. Keep as supporting machinery for (i). Do NOT headline it.
A standalone "Δ‖h‖ predicts incoherence" result is a 2026 replication.**

### (iii) Direction fungibility (real vs shuffled/random norm-matched control) — **SCOOPED UNCONDITIONALLY; SURVIVES UNDER THE GATE**

The unconditional version of this result is published three times over:

- **Rogue Scalpel `2509.22067`** — random directions raise harmful compliance 0% → 1–13%; 20 random
  vectors compose into a universal attack. Direction non-specificity, demonstrated, at scale.
- **`2602.06801` (Non-Identifiability)** — orthogonal perturbations achieve near-equivalent efficacy
  with negligible effect sizes; behaviorally indistinguishable equivalence classes. This is the
  theory-shaped version of our claim.
- **`2606.20852` (radiology VLM)** — runs the exact experiment we proposed: 20 random unit vectors,
  matched layer and absolute scale, bootstrap CIs; finds selected vectors falling *inside* the
  random distribution for several conditions.

Note the counter-evidence too: `2602.06801` reports orthogonal perturbations produce logit shifts
27–53% *smaller* than same-norm random directions, and the alignment-faking analysis
(`2605.27681` `[UNVERIFIED]`) finds targeted directions significantly above the random null on
OLMo-32B but indistinguishable from random at some strengths on Gemma-27B. So "fungible" is
model- and strength-dependent, not a clean universal. Any claim we make must be per-model and
per-strength, with the `2606.20852` protocol (norm-matched, layer-matched, bootstrap CI).

**Remaining gap (real):** *nobody has tested fungibility under a gate.* The question
"if a calibrated gate supplies the selectivity, is the written direction interchangeable with a
norm-matched random one at matched coherence?" is unoccupied, and it is the scientifically
interesting version — it directly tests whether the gate or the direction carries the information.
The abliterated-model refusal-restoration setting (where the direction demonstrably fails) is also
unoccupied.

**Verdict: NOVEL+TESTABLE only in the conditional framing.** Frame as *"under a calibrated gate,
the direction is a knob and the gate carries the information"* — never as *"steering directions are
arbitrary"* (published, ×3).

### Summary novelty table

| Claim | Status | Closest prior work | What survives |
|---|---|---|---|
| (i) Conformal gate + over-refusal bound | **Open** | `2606.12299` (VLA, conformalized steering gate); `2605.28722` (uncalibrated energy gate) | The full composition on an LLM safety gate. Headline this. |
| (ii) Displacement/norm budget for coherence | **Scooped** | `2602.02712`, `2606.06735`, `2604.09839`, `2601.19375`[U] | Only as a control variable *inside* the certified gate. Demote to supporting. |
| (iii) Direction fungibility | **Scooped unconditionally** | `2509.22067`, `2602.06801`, `2606.20852` | Fungibility *under the gate*, in the abliterated-model setting. Reframe. |

### Corollary: how to position our existing negative result

Our measured result (diff-of-means refusal vector does not restore refusal on abliterated
Gemma-3-1B/4B — only gibberish as α rises — while a probe on the same activations separates at
AUC ~0.99) should be positioned as:

- **Confirmation** of `2607.02714`'s multi-dimensional-subspace prediction at small scale, in a
  model tier (1B/4B abliterated) that paper does not cover, with a cross-scale check.
- **Quantification** of the READ≫WRITE asymmetry, which `2606.29441` implies (probe-gated wins)
  but does not isolate as a measurement.
- **NOT** as a surprise. It is the predicted outcome as of 2 Jul 2026.

---

## (d) The 3 papers to read in full

1. **`2606.29441` — Closing the Activation-Cone Blind Spot: Response-Time Probing and Unified
   Defense** (Mitra, 28 Jun 2026).
   *Why:* it is the paper we are competing with. It already benchmarks probe-gated vs unconditional
   steering head-to-head on identical models/attacks/metrics, already reports probe AUROC 0.97–1.00,
   and already composes a probe halt with AlphaSteer for DSR 0.98–0.99. Read it to extract the
   precise delta: it has no conformal calibration, no over-refusal bound, no direction-fungibility
   control. That delta *is* our paper. It also releases code and datasets — use them as our
   baseline rather than re-implementing, and adopt its prompt-time vs response-time distinction
   (our gate is prompt-time and therefore structurally blind to prefill attacks unless we fix it).

2. **`2607.02714` — Not All Refusals Are Equal: How Safety Alignment Fails Cybersecurity at Scale**
   (Hadetskyi, Pasquini, Sorokin, 2 Jul 2026).
   *Why:* the 24-model evidence that refusal is a multi-dimensional, layer-distributed subspace.
   It reframes our WRITE-side negative from anomaly to prediction, and its three-tier abliteration-
   susceptibility taxonomy tells us which model families our result generalizes to and which it
   does not. Without this citation our negative reads as a failed experiment; with it, it reads as
   a scale-checked confirmation.

3. **`2602.02712` — Towards Understanding Steering Strength** (Taimeskhanov, Vaiter, Garreau,
   v2 8 Jul 2026).
   *Why:* the theory of magnitude. If we propose a displacement budget without engaging its
   non-monotonicity laws — and `2606.06735`'s angle/norm split — a reviewer closes leg (ii) in one
   line. Read it to decide whether to demote the budget to a control variable (recommended) or to
   attempt a genuine improvement on the parameterization.

**Runners-up if time permits:** `2606.12299` (the only extant conformalized steering gate — read
before writing the conformal section), `2602.06801` (non-identifiability — read before writing the
fungibility section), `2606.15980` (probe staleness — read before claiming any durable guarantee).

---

## Verification ledger

**Verified by fetching `arxiv.org/abs/<id>` on 2026-07-25** (title + authors + date confirmed):
2606.06735, 2606.15092, 2606.29441, 2607.02714, 2606.20852, 2605.20286, 2604.24693, 2603.24543,
2604.09839, 2602.02712, 2602.06801, 2605.28722, 2606.12299, 2606.22673, 2606.15980, 2606.08682.

**Cited from prior corpus / widely-known, not re-fetched this pass:** 2409.05907 (CAST),
2406.11717 (single-direction refusal), 2509.22067 (Rogue Scalpel — abs page not fetched this pass
but corroborated by multiple independent sources incl. OpenReview and HF papers), 2412.09565
(Obfuscated Activations).

**`[UNVERIFIED]` — surfaced by search only, abs page NOT fetched. Do not cite until verified:**
2606.29054, 2605.20270, 2603.14623, 2606.06660, 2601.19375, 2602.20102, 2605.21706, 2605.24535,
2605.27681, 2602.04428, 2506.07022 (AlphaSteer), 2505.19056 (extended-refusal defense),
2603.22061, 2605.17413, 2607.05842, 2605.24154 (Palette), 2509.13450 (SteeringSafety),
2605.05892 (FLAS), 2605.06342.

**Also noted but lower priority:** `2606.15092` (High-Dimensional Random Projection for Activation
Steering / HiDRA, Pham, Do, Abdullaev, Nguyen, Than, 13 Jun 2026) — verified. Uses random
projection to a high-dimensional space to capture discriminative structure beyond linear
diff-of-means; note this is a *complement* to learned directions, **not** evidence that random
directions suffice. Do not misread it as supporting fungibility.
