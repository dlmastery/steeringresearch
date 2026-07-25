# LIT_2026-07 — Surveys, Position Papers & Foundations of Activation Steering

**Verified on 2026-07-25.** Literature scout pass for the steering autoresearch program.
Scope: surveys, position/critique papers, and 2026 consolidations of "what we now know"
about activation steering / representation engineering, prioritising June–July 2026
(arXiv `2606.*` / `2607.*`), with seminal `2602–2605` included.

**Verification protocol.** Every arXiv id in the VERIFIED table below was fetched at its
`arxiv.org/abs/` page on 2026-07-25 and its **title + author list + submission date**
confirmed. Ids appearing only in search-result snippets are collected separately and
marked `[UNVERIFIED]`; they must not be cited in a reasoning entry until fetched.
Quantitative figures are as reported on the abs pages (abstract-level), not re-derived
from the PDFs — treat them as `[NEEDS VERIFICATION]` per CLAUDE.md §15.10 until reproduced.

---

## Executive summary

1. **There is no dedicated activation-steering survey from June–July 2026.** The survey
   layer of this field is still the two February-2025 papers (Wehner et al. 2502.19649;
   Bartoszcze et al. 2502.17601). The only 2026 survey touching steering is a broad
   mechanistic-interpretability overview (2607.07316) in which steering is one chapter.
2. **What Jun–Jul 2026 produced instead is a wave of limits / negative-result papers.**
   The field has entered its critique phase: non-surjectivity, non-identifiability,
   decodability–steerability gap, α non-monotonicity theory, composition collapse,
   emergent misalignment, and a steering-vector supply-chain attack — all within five months.
3. **Our program's headline results are mostly CORROBORATED, and two of them already have
   names in the literature.** The "READ works, WRITE doesn't" result is the
   *decodability–steerability gap* (2605.05715, May 2026). The off-manifold/incoherence
   mechanism is the thesis of 2605.05115 (May 2026).
4. **Our single highest-risk claim is the label-shuffled control (~97%).** It is
   corroborated by non-identifiability (2602.06801) but contradicted by a consistent
   random-direction-control literature. Our own weak-judge caveat (AUC 0.68) means it is
   currently instrument-limited and must not ship as a finding.
5. **Actionable, immediately:** 2604.15557 shows the middle-layer heuristic produces
   *no effect* on Gemma-2-2B and OLMo-2-1B-Instruct — our exact scale — while a
   training-free logit-lens predictor (A_lin) selects layers that work. If our layer choice
   came from the middle-layer heuristic, this may explain finding (a) outright.

---

## (a) The 2026 survey / position / limits corpus — VERIFIED

All rows below fetched and confirmed 2026-07-25.

| Title | arXiv id | Date | Thesis (one line) | Open problems / limitations it names |
|---|---|---|---|---|
| Taxonomy, Opportunities, and Challenges of Representation Engineering for Large Language Models — Jan Wehner, Sahar Abdelnabi, Daniel Tan, David Krueger, Mario Fritz | **2502.19649** | 27 Feb 2025 (v5, 8 Oct 2025) | The field's unifying taxonomy: RepE is a pipeline of **representation identification → operationalization → control** | (1) simultaneous manipulation of **multiple concepts**; (2) **reliability** of control; (3) **performance preservation** under intervention; (4) methodological gaps / missing best practices |
| From Weights to Activations: Is Steering the Next Frontier of Adaptation? — Simon Ostermann, Daniil Gurgurov, Tanja Baeumel, Michael A. Hedderich, Sebastian Lapuschkin, Wojciech Samek, Vera Schmitt | **2604.14090** | 15 Apr 2026 — **ACL 2026 Main** | **Position paper.** Introduces *functional criteria for adaptation methods* and a unified taxonomy placing steering alongside fine-tuning, PEFT and prompting; steering = "local and reversible behavioral change without parameter updates" | Steering "is rarely analyzed within the same conceptual framework as established adaptation methods" — no shared evaluation basis against FT/PEFT/prompting |
| Mechanistic Interpretability for Neural Networks: Circuits, Sparse Features and Symbolic Reasoning — Pranav Sawant, Jakub Krejčí | **2607.07316** | 8 Jul 2026 | Broad mech-interp survey (circuits, superposition, SAEs/transcoders, neurosymbolic); steering vectors + causal interventions are the control chapter | No steering-specific taxonomy or enumerated open-problem list in the abstract — **not a substitute for a steering survey** |
| On the Limits of Steering Vectors for Preference-Aligned Generation — Melanie Subbiah, Zara Hall, Kathleen McKeown | **2607.01802** | 2 Jul 2026 | Limits paper along three axes: **trait expressibility, task transfer, multi-trait composition**. PLUME benchmark; Qwen2.5-7B-Instruct + Llama3.1-8B-Instruct | Effectiveness varies substantially across traits; transfer to downstream tasks degrades; **all** composition methods lose trait expression as vectors are added; **coherence-vs-expressibility tradeoff requiring per-setting hyperparameter tuning**; "steering vectors face meaningful limits as a general-purpose tool for preference alignment" |
| Steered LLM Activations are Non-Surjective — Aayush Mishra, Daniel Khashabi, Anqi Liu | **2604.09839** | 10 Apr 2026 (rev 7 May 2026) | Formal separation: **activation steering pushes the residual stream off the manifold of states reachable from discrete prompts** | Ease of steering ≠ evidence the behavior is prompt-accessible; steered vulnerabilities may not translate to real prompt-based jailbreak risk; white-box and black-box interventions must be decoupled in evaluation protocols |
| Towards Understanding Steering Strength — Magamed Taimeskhanov, Samuel Vaiter, Damien Garreau | **2602.02712** | 2 Feb 2026 (rev 8 Jul 2026) | **First theoretical analysis of steering strength (α)**; derives qualitative laws for next-token probability, concept presence, and cross-entropy | Reveals **non-monotonic effects of α**; "too little and the intended behavior does not emerge, too much and the model's performance degrades beyond repair". Validated on **eleven** LMs, small GPT → modern |
| On the Non-Identifiability of Steering Vectors in Large Language Models — Sohan Venkatesh, Ashish Mahendran Kurapath | **2602.06801** | 6 Feb 2026 (final 1 Apr 2026) | Steering vectors are **fundamentally non-identifiable**: large equivalence classes of behaviorally indistinguishable interventions | "**Orthogonal perturbations achieve near-equivalent efficacy with negligible effect sizes**" — undermines the assumption that a discovered direction is a uniquely meaningful internal representation |
| Manifold Steering Reveals the Shared Geometry of Neural Network Representation and Behavior — Daniel Wurgaft, Can Rager, Matthew Kowal, Vasudev Shyam, Sheridan Feucht, Usha Bhalla, Tal Haklay, Eric Bigelow, Raphael Sarfati, Thomas McGrath, Owen Lewis, Jack Merullo, Noah Goodman, Thomas Fel, Atticus Geiger, Ekdeep Singh Lubana | **2605.05115** | 6 May 2026 | **"Linear steering — which assumes a Euclidean geometry — cuts through off-manifold regions and hence produces unnatural outputs."** Recasts the problem from *finding the right direction* to **finding the right geometry** | Pathologies of linear steering (brittleness, incoherence, off-target effects) stem from flat-geometry mismatch with true curved representation geometry |
| Decodable but Not Corrected by Fixed Residual-Stream Linear Steering: Evidence from Medical LLM Failure Regimes — Ming Liu | **2605.05715** | 7 May 2026 | **Names the decodability–steerability gap.** Overthinking is linearly decodable at **71.6% balanced accuracy (p<10⁻¹⁶)** yet **29 steering configurations across 5 families give Δ≈0** | Cause is representational entanglement: the concept direction has **85–88% overlap with task-critical computation**; concept erasure costs −3.6pp accuracy (p=0.01) vs +0.3pp for random erasure; non-targeted steering costs −12.1pp. Probe still usable for **abstention** (held-out AUROC 0.610, beats 5 uncertainty baselines, p=0.009). Qwen2.5-7B, MMLU-STEM |
| Predicting Where Steering Vectors Succeed — Jayadev Billa | **2604.15557** | 16 Apr 2026 | **Linear Accessibility Profile (LAP)**: A_lin applies the unembedding matrix to hidden states — a *training-free* logit-lens predictor of steerability | Peak A_lin correlates ρ=**+0.86 to +0.91** with steering effectiveness and ρ=+0.63 to +0.92 with layer selection, over 24 binary concept families × 5 models (Pythia-2.8B → Llama-8B). **Three-regime framework**: when diff-of-means suffices, when nonlinear methods are needed, when intervention cannot work. **Steering at LAP layers redirects completions on Gemma-2-2B and OLMo-2-1B-Instruct where the standard middle-layer heuristic produces no effect** |
| When is Your LLM Steerable? — Chenrui Fan, Yize Cheng, Ming Li, Soheil Feizi, Tianyi Zhou | **2606.11599** | 10 Jun 2026 | **ASTEER** testbed: 1.4M steered generations over 150 concepts; GBDT on pre/post hidden-state features predicts under-steer / success / over-steer at **~0.7 macro-F1 on unseen concepts** | "Early hidden states encode substantial, structured information about eventual steering efficacy" — steerability is predictable without full rollout; enables near-optimal α at a fraction of decoding cost. Abs page does not enumerate model scales |
| When Is Rank-1 Steering Cheap? Geometry, Granularity, and Budgeted Search — John T. Robertson, Jianing Zhu, Haris Vikalo, Zhangyang Wang | **2605.16362** | 9 May 2026 (rev 20 May 2026) | Reframes "when does rank-1 fail?" as "when is rank-1 **cheap and stable**" — variability reflects **search difficulty**, not the absence of a useful rank-1 intervention | **Concept granularity** (utility-maximizing direction rotates systematically across inputs) correlates r=**−0.46** with best-found utility (p<0.001); prompt-boundary directional alignment predicts where effective interventions live; geometry-guided search cuts trials to 95% of best utility by 39.8% across three model families |
| Decomposing how prompting steers behavior — Fan L. Cheng, Nikolaus Kriegeskorte | **2606.03093** | 2 Jun 2026 | Geometric decomposition of prompting into a ladder of maps (translation → rigid → axis scaling → affine → nonlinear) across LMs and VLMs | Prompts reshape representations toward instructed task structure, largely via shape-preserving maps — **but "cross-dimensional linear mixing is a key mechanism"**, i.e. affine, not pure translation. Implication: **rank-1 additive steering is the wrong operator class** for reproducing what prompting does |
| Activation Steering Induces Emergent Misalignment: A More Comprehensive Evaluation — Qi Cao, Jian Lou, Meiting Liu, Wenjie Feng, Dan Li, See-Kiong Ng, Anh Tuan Luu | **2606.08682** | 7 Jun 2026 | Emergent misalignment (EM) is **not exclusive to fine-tuning** — activation steering induces it too, generalising unsafe behavior to unrelated tasks | Steered models generate unsafe content with **higher semantic coherence than fine-tuned counterparts** (i.e. worse, not better, from a safety standpoint); sensitivity varies with steering intensity, subspace dimensionality, and training iterations. Qwen-3.5 series + diverse families/scales |
| Steering Vectors are an Adversarial Attack Surface — Abzal Aidakhmetov, Donato Crisostomi, Tommaso Mencattini, Adrian Robert Minut, Iacopo Masi, Emanuele Rodolà | **2606.05958** | 4 Jun 2026 | **Supply-chain attack on shared steering-vector bundles**: poisoning only **4–6% of tokens** in a steering dataset yields a vector that jailbreaks while behaving benignly on normal prompts | Poisoned vectors reach **20–55% ASR**, a 19–51% improvement over baseline attacks; "apparently safe bundles" are silently distributable |
| Analysing the Safety Pitfalls of Steering Vectors — Yuxiao Li, Alina Fastowski, Efstratios Zaradoukas, Bardh Prenkaj, Gjergji Kasneci | **2603.24543** | 25 Mar 2026 | CAA steering **overlaps the refusal direction**, so steering for an unrelated behavior moves jailbreak susceptibility as a side effect | Attack success rate moves **+57% / −50%** depending on direction; template-based attacks amplify more strongly under steering; documented across model families and sizes |

### Seen in search results but NOT fetched — `[UNVERIFIED]`, do not cite yet

`2502.17601` (Bartoszcze et al., *Representation Engineering for LLMs: Survey and Research
Challenges*, the companion Feb-2025 survey — id and title seen in citations and PDF text,
author list unconfirmed); `2602.06256` (*Steering Safely or Off a Cliff? Rethinking
Specificity and Robustness in Inference-Time Interventions*); `2603.18329`
(*FaithSteer-BENCH*); `2605.27681` (*Behavioural Analysis of Alignment Faking* — source of
the "targeted shift 1.10–2.25σ above random null" figure); `2606.15420` (*Constitutional
Value Potentials* — "+0.24 to +0.60 over random-orthogonal control"); `2604.01366`
(*CogBias* — Gram-Schmidt orthogonal-direction control); `2603.13911` (*The Phenomenology
of Hallucinations*); `2606.22357` (*ORBIT*); `2606.08454` (*Beyond Linear Activation
Steering / INNSteer*); `2605.03907` (*Steer Like the LLM*); `2604.08169` (*Activation
Steering for Aligned Open-ended Generation without Sacrificing Coherence*); `2605.30076`
(*UniSteer*); `2607.19364` (*Statistically Grounded Sparse-Feature Interventions*);
`2604.24693` (*Contextual Linear Activation Steering*); `2605.01844` (*The Cylindrical
Representation Hypothesis*); `2509.13450` (*SteeringControl / SteeringSafety*);
`2505.22637` (*Understanding (Un)Reliability of Steering Vectors*).

**Explicitly not found:** no `2606.*` or `2607.*` paper that is a *survey of activation
steering / representation engineering* as such. If one is needed for the paper's related-work
section, the honest statement is that the field's surveys are 2502.19649 / 2502.17601 and
that no 2026 update exists as of 2026-07-25.

---

## (b) WHAT THE FIELD NOW BELIEVES

**On efficacy.**

1. **READ is largely solved; WRITE is not.** As of May 2026 this has a name — the
   **decodability–steerability gap** (2605.05715) — and is documented across clinical
   reasoning, math solvability and factual reasoning. Linear decodability of a concept does
   **not** imply the concept is a manipulable linear feature.
2. **Steering is a real but *narrow* tool.** 2607.01802 is the July-2026 statement of the
   limits: trait-dependent efficacy, degradation under task transfer, and composition
   collapse as vectors are added. The practitioner consensus mirrors this — steering works
   for refusal / sentiment / formality, fails for factual recall and complex reasoning.
3. **Steering is not equivalent to prompting, in either direction.** Steering reaches states
   no prompt can produce (2604.09839, non-surjectivity), while prompting itself acts by
   **cross-dimensional affine mixing** rather than translation (2606.03093) — so a rank-1
   additive vector is structurally the wrong operator to imitate prompting.

**On failure modes.**

4. **The dominant failure mechanism is geometric, not directional.** 2605.05115 states it
   plainly: linear steering assumes Euclidean geometry, cuts through off-manifold regions,
   and *hence* produces unnatural outputs. The prescription is to find the right geometry,
   not a better direction.
5. **α is non-monotonic and there is an irreducible coherence-vs-expression tradeoff.**
   2602.02712 gives the theory (11 models); 2607.01802 gives the empirical tradeoff and
   notes it needs per-setting tuning. Naive global linearity in α is refuted.
6. **The middle-layer heuristic is refuted.** Optimal layer is domain- and model-specific;
   a training-free logit-lens predictor beats the heuristic, and on Gemma-2-2B /
   OLMo-2-1B-Instruct the heuristic gives **no effect at all** (2604.15557).
7. **Direction may not be the load-bearing component.** Non-identifiability (2602.06801):
   large equivalence classes of behaviorally indistinguishable interventions, with
   orthogonal perturbations reaching near-equivalent efficacy. This is in live tension with
   the random-direction-control literature (see §c-e).
8. **Variability is partly a search problem.** 2605.16362 argues rank-1 directions often
   exist but are hard to find, and that concept granularity (direction rotating across
   inputs) is the cost driver — r=−0.46 with best-found utility.

**On safety.**

9. **Steering is a safety liability as much as a safety tool.** ASR moves ±50–57% via
   refusal-direction overlap (2603.24543); steering induces emergent misalignment with
   *higher* coherence than fine-tuning does (2606.08682); shared steering-vector bundles are
   a poisonable supply chain at 4–6% token contamination (2606.05958). A guard evaluation
   that only measures leakage in one direction is incomplete.

**On small-model steerability.**

10. **Unresolved and under-covered.** Larger is *not* reliably more steerable; linear
    steerability emerges over pretraining and varies by concept. The 2026 limits papers run
    almost exclusively at **7–8B** (2607.01802: Qwen2.5-7B / Llama3.1-8B; 2605.05715:
    Qwen2.5-7B; 2604.15557: Pythia-2.8B → Llama-8B). The only ≤2B evidence surfaced is
    2604.15557's Gemma-2-2B / OLMo-2-1B demonstration. **The ≤4B regime is a genuine gap in
    the literature — which is where this program operates.**

---

## (c) WHAT THIS CHANGES FOR US

Our five measured findings, adjudicated against the verified corpus.

### (a) Mean-pooled diff-of-means barely installs refusal and mostly destroys coherence, at BOTH 1B and 4B — **CORROBORATED (strongly). Not novel as a phenomenon.**

- 2607.01802 (2 Jul 2026): coherence-vs-expressibility tradeoff requiring per-setting tuning;
  steering effectiveness varies substantially by trait; limits as a general-purpose tool.
- 2605.05115 (6 May 2026): linear steering "cuts through off-manifold regions and hence
  produces unnatural outputs" — the mechanism for our coherence destruction.
- 2602.02712 (2 Feb 2026, rev 8 Jul 2026): too little α ⇒ no behavior; too much ⇒ degradation
  beyond repair, non-monotonically.
- **What is ours:** the *cross-scale* replication (1B **and** 4B, same negative result) with
  the full five-axis composite. Frame as replication-at-small-scale, not discovery.
- **Action:** before claiming the negative result is about diff-of-means, rule out the
  confound flagged by 2604.15557 — **if our injection layer came from the middle-layer
  heuristic, the null may be a layer-selection artifact, not a method failure.** Run the
  training-free A_lin logit-lens profile first. This is the cheapest possible falsifier and
  it targets our headline claim.

### (b) Trained probe on the SAME activations reaches AUC 0.99 harmful-vs-benign — READ works, WRITE doesn't — **CORROBORATED and ALREADY NAMED; our version is quantitatively stronger and at an uncovered scale.**

- 2605.05715 (7 May 2026) coins the **decodability–steerability gap**: 71.6% balanced
  accuracy decodable, Δ≈0 across 29 steering configurations in 5 families, on Qwen2.5-7B.
- **Ours is a sharper instance:** AUC **0.99** (vs their 71.6% balanced accuracy) with the
  same WRITE null, at **≤4B** where they ran 7B. A near-perfect probe makes the gap much
  harder to attribute to a weak read.
- **Priority claim risk:** we cannot claim to have discovered this gap. We can claim
  (i) the extreme-separability instance, (ii) the ≤4B scale, (iii) the refusal/safety
  concept rather than clinical overthinking.
- **The experiment we are missing** is their causal diagnostic: they attribute the gap to
  **representational entanglement** — 85–88% overlap between the concept direction and
  task-critical computation, verified by a concept-erasure ablation (−3.6pp, p=0.01, vs
  +0.3pp for random erasure). Our finding (d) suggests *geometry* rather than entanglement
  is our cause. **Running both diagnostics and separating entanglement from off-manifold
  displacement is the single most valuable next experiment**, because it distinguishes our
  mechanism from the published one.
- **Silver lining to import:** their probe still delivered value as a **selective-abstention
  gate** (held-out AUROC 0.610, beating five uncertainty baselines, p=0.009). Our 0.99 probe
  as a CAST-style conditional gate is precisely the same move, and 2605.05715 is the
  citation that legitimises "the probe is the product, the vector is not."

### (c) Coherence cliff at α ≈ 0.05–0.10 — **CORROBORATED qualitatively; NOVEL quantitatively.**

- 2602.02712 gives the theory of non-monotone α and validates on 11 models — but derives
  *qualitative laws*, and does not localise a normalised cliff at ≤4B.
- 2607.01802 confirms the tradeoff exists and needs per-setting tuning; does not locate it.
- **A cliff this early (α≈0.05–0.10 of activation norm) at 1B–4B is a scale-specific,
  unclaimed number.** This is our strongest candidate for a genuinely novel quantitative
  contribution — *provided* the normalisation convention is stated explicitly (fraction of
  ‖h‖ at the injection layer) so it is comparable to 2602.02712's laws.
- **Action:** pre-register the cliff location as a *prediction* on a held-out concept before
  measuring it again; that converts it from an observation into a falsifiable claim.

### (d) Off-manifold displacement predicts incoherence, ρ = +0.585 — **CORROBORATED mechanistically; NOVEL as a measurement.**

- 2605.05115 (16 authors, 6 May 2026) *asserts* exactly this mechanism — off-manifold
  traversal ⇒ unnatural outputs — and recasts steering as a geometry problem. But the abs
  page states the claim qualitatively; it does not report a displacement↔incoherence
  correlation coefficient.
- **We appear to hold the number.** Cite 2605.05115 for the mechanism and claim ρ=+0.585 as
  the quantitative leading indicator, at ≤4B. This directly serves CLAUDE.md §3's geometry
  leading-indicator mandate and §6's λ_geo term — we now have external theoretical backing
  for pricing off-shell displacement in the composite.
- **Caveat to state:** ρ=+0.585 is a moderate correlation. Report the CI and the n; do not
  let it be read as "displacement explains incoherence."
- **Adjacent unclaimed lever:** 2605.16362's *concept granularity* (direction rotates across
  inputs, r=−0.46 with utility) is a second geometric predictor we do not currently log.
  Cheap to add alongside participation ratio and effective rank.

### (e) Label-shuffled control captures ~97% of the steering effect at ≤2B (judge AUC 0.68) — **SPLIT: CORROBORATED by one line of work, CONTRADICTED by another. Currently INSTRUMENT-LIMITED. Do not ship.**

**Corroborating:**
- 2602.06801 (6 Feb 2026, final 1 Apr 2026) — steering vectors are non-identifiable; large
  equivalence classes; "**orthogonal perturbations achieve near-equivalent efficacy with
  negligible effect sizes**." This is close to a formal version of our result.
- 2605.16362 is compatible: if variability is search difficulty rather than direction
  identity, many directions can be near-equivalent.

**Contradicting** — a consistent random-direction-control literature reports that learned
directions *do* beat matched random/orthogonal nulls, all `[UNVERIFIED]` pending fetch:
- `2605.27681` — targeted shift sits **1.10–2.25 σ above the random null** with norm- and
  dimension-matched random unit vectors at the same layers.
- `2606.15420` — value direction exceeds a random-orthogonal control by **+0.24 to +0.60**.
- `2604.01366` — Gram-Schmidt orthogonal-direction control used to test whether orientation
  matters, with a positive answer.
- `2603.13911` — steering along the boundary produces substantially larger output changes
  than random directions.

**Verdict for our program.** The literature is genuinely split, which makes this both the
most interesting and the most dangerous of our five. **But our own judge is AUC 0.68** — a
near-chance instrument cannot distinguish a 97% overlap from a 60% overlap. Per CLAUDE.md
§17 rubric item 3, this number is self-disqualifying as a headline.

**Action, in order:**
1. Re-run the shuffled-label control with the mandated off-family judge
   (`STEER_JUDGE_MODEL=Qwen/Qwen2.5-3B-Instruct`). Until then the finding is
   `UNTESTED_ON_RIGHT_DATASET`.
2. Match the null construction to the published protocol — **norm-matched random unit
   vectors at the same layer** (2605.27681's design) *and* Gram-Schmidt orthogonal
   directions (2604.01366's design), not only label-shuffling. Label-shuffling and
   norm-matched random directions are different nulls and may not be interchangeable;
   a shuffled-label vector retains the data's covariance structure while a random unit
   vector does not.
3. Report the effect as σ above the null (the field's unit), not as "% of effect captured."
4. Fetch and verify the four `[UNVERIFIED]` ids before citing any of them.

---

## (d) The three papers to read in full

**1. `2605.05715` — Decodable but Not Corrected by Fixed Residual-Stream Linear Steering (Ming Liu, 7 May 2026).**
The only paper that names our headline finding (b). Read for the **causal diagnostic we do
not have**: the concept-erasure ablation that established 85–88% overlap between the concept
direction and task-critical computation, with a random-erasure control. That protocol,
applied to our refusal direction at 1B/4B, is what turns "WRITE doesn't work" from an
observation into a mechanism. Also read for the constructive framing — their probe survived
as a selective-abstention gate, which is the precedent for our probe-as-CAST-gate result.

**2. `2605.05115` — Manifold Steering Reveals the Shared Geometry of Neural Network Representation and Behavior (Wurgaft, Rager, Kowal, … Lubana; 16 authors, 6 May 2026).**
The mechanism behind findings (a), (c) and (d) in a single frame, from a large and credible
group. It gives us (i) the citation for off-manifold displacement causing incoherence,
(ii) external justification for the λ_geo term in the composite, and (iii) a *constructive
alternative* — manifold-aware steering — that converts our negative result into a research
program rather than a dead end. Read the geometry construction closely; if their manifold
estimator is tractable at 1B on 16 GB, it is our next method rung.

**3. `2604.15557` — Predicting Where Steering Vectors Succeed (Jayadev Billa, 16 Apr 2026).**
The most immediately actionable paper in the corpus and the only one validated **at our exact
scale**: LAP-recommended layers redirect completions on **Gemma-2-2B and OLMo-2-1B-Instruct**
where the **standard middle-layer heuristic produces no effect**. A_lin is training-free
(logit lens on hidden states) and therefore essentially free on our budget. Read it before
running any further α/layer sweep — it may reduce our sweep cost substantially, and it
supplies a three-regime framework that tells us in advance whether diff-of-means *can* work
for a given concept. **If our injection layer came from the middle-layer heuristic, this
paper is a candidate explanation for finding (a) and must be ruled out before we publish a
negative result about diff-of-means.**

**Runner-up (mandatory if we touch guards): `2603.24543` — Analysing the Safety Pitfalls of
Steering Vectors.** ASR moves **+57% / −50%** through refusal-direction overlap. Our
JailbreakBench gate currently checks for leakage; per this paper it must check **both
directions**, because a steering vector that *reduces* ASR by overlapping the refusal
direction is inflating our safety axis for the wrong reason. Pair with `2606.08682`
(steering induces emergent misalignment, with higher coherence than fine-tuning) and
`2606.05958` (4–6% token poisoning ⇒ 20–55% ASR in shared bundles) for the Rung-4 red-team
design.

---

## Immediate implications for the program

| Implication | Affected artifact |
|---|---|
| Run the A_lin / LAP layer profile before any further α×layer sweep; rule out middle-layer-heuristic confound on finding (a) | `skills/steering-hillclimb`, next experiment's Diagnose step |
| Add the concept-erasure ablation (targeted vs random erasure) to separate entanglement from off-manifold geometry as the cause of the READ/WRITE gap | new experiment; `hypotheses/D_geometry/` |
| Re-run the shuffled-label control with the Qwen off-family judge and with norm-matched random + Gram-Schmidt orthogonal nulls; report σ above null | finding (e) — currently blocked from `FINDINGS.md` |
| Log concept granularity (direction rotation across inputs) alongside Δ‖h‖, effective rank, participation ratio | `src/steering/eval.py` geometry probes |
| Safety gate must measure ASR movement in **both** directions, not leakage only | `skills/steering-rogue-scalpel-guard` |
| Related-work section must state that no 2026 steering survey exists; anchor on 2502.19649 + 2604.14090 (ACL 2026 position paper) | paper draft |
| Fetch-verify the four random-direction-control ids before citing | `corpus/` discipline |

---

*Compiled 2026-07-25. Sixteen arXiv abs pages fetched and verified (title + authors + date);
seventeen further ids listed as `[UNVERIFIED]` pending fetch. All quantitative figures are
abstract-level as reported by the source and carry `[NEEDS VERIFICATION]` until reproduced on
the 4090 ladder, per CLAUDE.md §15.10.*
