# The 2026 Evaluation Standard for Activation Steering

**Verified on 2026-07-25.** All arXiv ids below were WebFetch-verified against their
`arxiv.org/abs/` page (title + authors + submission date confirmed) unless explicitly
marked `[UNVERIFIED]`. Nothing here is cited from model memory.

Scope: evaluation science for activation/conditional steering — benchmarks, LLM-judge
reliability, reproducibility/negative results, required controls, statistical practice.
Prioritised June–July 2026 (arXiv `2606.*`, `2607.*`).

---

## EXECUTIVE SUMMARY

1. **Our headline is already scooped, in part.** A norm-matched random-direction null
   that *matches or exceeds* the targeted steering effect is already published
   (arXiv:2605.27681, Llama-70B layer 58). "Shuffled/random control captures the
   effect" is no longer a novel claim on its own. See section (d).
2. **The field moved from "does it steer?" to "is the direction load-bearing?"**
   Non-identifiability (arXiv:2602.06801) says behaviourally-indistinguishable
   equivalence classes of steering vectors are large; orthogonal perturbations reach
   near-equivalent efficacy. Behavioural matching alone can no longer identify a direction.
3. **LLM judges are now presumed guilty until validated.** The largest 2026 study
   (arXiv:2606.19544, ~541k judgments) shows chance-corrected agreement collapses
   33–41 pp below exact-match, and defines a *Minimum Viable Validation Protocol* that
   any credible pipeline must satisfy. Single-trial judging is inadmissible
   (arXiv:2606.13685: 11 trials needed to recover a reference verdict at 95%).
4. **Consequence for us.** Our AUC-0.68 judge fails every 2026 bar. The re-run must
   ship a *judge card* as a separate pre-registered artifact BEFORE any steering number
   is computed, and must report the real direction's effect as **sigma above a
   random-direction null**, not versus no-steering.
5. **Reframe the contribution.** The defensible novel claim is the *scale and
   conditionality boundary* of direction-fungibility (where it holds, where it breaks,
   how the conditional gate interacts) — measured with a validated judge and a proper
   null. A bare fungibility null would be scooped on arrival.

---

## (a) KEY PAPERS

| Title | arXiv id | Date | Contribution | What it demands of an evaluation |
|---|---|---|---|---|
| Behavioural Analysis of Alignment Faking — Mitrani Hadida, Karty, Williams-King, Cooney | arXiv:2605.27681 | 26 May 2026 | 41 norm- and dimension-matched random unit vectors as a per-layer/per-alpha null; reports targeted effects in sigma-above-null; finds the null *swallows* the targeted effect on Llama-70B | Report the targeted effect as **sigma above a random-direction null distribution**, per layer and per alpha — never versus the no-steer baseline alone |
| On the Non-Identifiability of Steering Vectors in Large Language Models — Venkatesh, Kurapath | arXiv:2602.06801 | 6 Feb 2026 (rev. 1 Apr 2026) | Large equivalence classes of behaviourally indistinguishable interventions under single-layer white-box access; orthogonal perturbations achieve near-equivalent efficacy with negligible effect sizes; SVD of activation covariance sizes the equivalent-intervention subspace | Behavioural output matching cannot establish that *this* direction is the mechanism; structural constraints beyond behavioural testing are required |
| Reliability without Validity: A Systematic, Large-Scale Evaluation of LLM-as-a-Judge Models Across Agreement, Consistency, and Bias — Norman, Rivera, Hughes | arXiv:2606.19544 | 17 Jun 2026 | 21 judges, 9 providers, 118 runs, ~541k judgments; kappa deflation vs exact-match is universal (33.8–41.2 pp on MT-Bench); judge rankings shift up to 14 positions across benchmarks; defines the Minimum Viable Validation Protocol | Chance-corrected kappa as the headline number; paired AB+BA position-bias measurement; >=3 runs; >=2 benchmark label distributions |
| The Coin Flip Judge? Reliability and Bias in LLM-as-a-Judge Evaluation — Yagubyan | arXiv:2606.13685 | 23 Apr 2026 | Pairwise preferences flip 13.6% of the time across runs (28% of questions exceed 20% flip rate); cross-judge agreement 76% (kappa = 0.51); GPT-4o-mini first-position bias 72% A-majority (p = 0.024) | **11 repeated trials** (15 for high-variance items) for a majority vote to recover the 50-trial reference verdict with 95% probability; position randomisation; explicit uncertainty reporting |
| FaithSteer-BENCH: A Deployment-Aligned Stress-Testing Benchmark for Inference-Time Steering — Ding, Hu, Zhang, Yao, Li, Liu, Hu | arXiv:2603.18329 | 18 Mar 2026 | Three gate-wise criteria (controllability, utility preservation, robustness) at a fixed deployment-style operating point; exposes "illusory controllability" and shows many methods induce prompt-conditional alignment rather than a stable latent directional shift | Evaluate at a fixed operating point; stress-test with instruction perturbations, role prompts, encoding transformations, data scarcity |
| Statistically Grounded Sparse-Feature Interventions for Activation-Space Control in LLMs — Siddique, Alam, Rafy, Raiyan, Mahmud, Hasan | arXiv:2607.19364 | Jun 2026 | Six-condition reliability filter; unweighted Borda consensus over F-test, KSG mutual information, and Cohen's d; Fisher-LDA grounding; evaluated over 3 Gemma-family models, 4 behavioural domains, **356 layer-strength configurations** | Report Cohen's d; sweep the full layer x strength grid rather than reporting a tuned point; use multiple independent statistics, not one |
| Mechanistic Indicators of Steering Effectiveness in Large Language Models — Jafari, Xue, Salim | arXiv:2602.01716 | 2 Feb 2026 (rev. 12 Mar 2026) | Normalized Branching Factor (entropy-derived) + KL divergence between steered activations and the target concept in vocabulary space predict steering success without a judge | Corroborate judge-based efficacy with judge-independent internal signals |
| Steering LLMs? Actually, Sparse Autoencoders can outperform simple baselines — Jorgensen, Hansen | arXiv:2605.31183 | 29 May 2026 | Overturns the AxBench result that SAEs underperform simple baselines; with a supervised feature-selection pipeline SAEs reach near-LoRA reference performance on AxBench | AxBench conclusions are pipeline-sensitive: state the selection pipeline explicitly; a method-level ranking without it is not reproducible |
| Analyzing the Generalization and Reliability of Steering Vectors — Tan et al. | arXiv:2407.12404 | Jul 2024 (NeurIPS 2024; v6) | The foundational reliability result: steering vectors have substantial in- and out-of-distribution limitations; for several concepts they are brittle to reasonable prompt changes | Report per-concept variance, not aggregate means; test prompt-format sensitivity |
| Steering Vectors are an Adversarial Attack Surface | arXiv:2606.05958 | Jun 2026 | Security framing of steering interventions | Relevant to the safety axis; not fetched in full — treat details as `[UNVERIFIED]` |
| AxBench: Steering LLMs? Even Simple Baselines Outperform Sparse Autoencoders — Wu et al. | arXiv:2501.17148 | Jan 2025 | The reference concept-control benchmark this line builds on | Prompting and diff-of-means baselines are mandatory comparators |

Additional benchmarks surfaced but **not** individually abs-verified in this pass —
treat as `[UNVERIFIED]` pending a fetch: CLaS-Bench (arXiv:2601.08331, cross-lingual
steering), SwordBench (arXiv:2605.16372, orthogonality of steering image
representations), UniSteer (arXiv:2605.30076), SteeringSafety / SteeringControl
(arXiv:2509.13450), Steer-Bench (arXiv:2505.20645), PsySET (arXiv:2510.04484),
Who Drifted: the System or the Judge? (arXiv:2606.15474).

---

## (b) THE 2026 EVALUATION STANDARD FOR STEERING

A concrete checklist. A steering result that misses any of these is not credible in
July 2026.

### B1. Judge validation (do this FIRST, as a standalone artifact)
- [ ] Judge is **off-family** from the model being steered.
- [ ] Agreement with human labels reported as **chance-corrected Cohen's kappa (or
      Krippendorff's alpha)** as the *headline* number. Exact-match percentage alone is
      disqualifying — the deflation is 33–41 pp (arXiv:2606.19544).
- [ ] **Position bias**: paired AB+BA evaluations, report `|P(A wins) - 0.5|`; must be
      **< 0.10**.
- [ ] **Test-retest**: >=3 independent runs at temperature 0, response caching disabled.
- [ ] **Paradox audit**: if test-retest > 0.95, explicitly verify position bias < 0.10
      before claiming reliability (high reproducibility != validity).
- [ ] **Multi-trial verdicts**: single-trial judging is inadmissible;
      **>=11 trials** with majority vote (15 for high-variance items), position
      randomised (arXiv:2606.13685).
- [ ] Cross-validate the judge on **>=2 label distributions** (preference-style and
      correctness-style).
- [ ] Publish a **judge card** with all of the above before any downstream number.

### B2. Required controls
- [ ] **Norm- and dimension-matched random-direction null**: 30–41 random unit vectors,
      same layer, same alpha grid, multiple seeds. Report the targeted effect as
      **sigma above this null** (arXiv:2605.27681).
- [ ] **Label-shuffled / permuted-label direction** built through the identical pipeline.
- [ ] **Orthogonal-complement direction** (the non-identifiability control,
      arXiv:2602.06801).
- [ ] **Prompting baseline** (AxBench: simple baselines are strong).
- [ ] **No-steer baseline** at matched decoding settings.
- [ ] Explicit statement of which equivalence class the claim is about — you are
      claiming a *behavioural* effect, not a unique mechanism, unless you add structural
      evidence.

### B3. Matched coherence and utility
- [ ] Comparisons made at **matched perplexity / matched coherence**, not at matched alpha.
- [ ] **Utility preservation** measured on unrelated capabilities at a *fixed deployment
      operating point* (arXiv:2603.18329).
- [ ] **Gibberish / degeneration rate** reported as its own axis — an incoherent model
      can score "safe" for free.
- [ ] **Robustness stress**: instruction perturbation, role prompts, encoding
      transformations. Prompt-conditional alignment must be distinguished from a stable
      latent shift.

### B4. n, seeds, splits
- [ ] **>=7 seeds** for any evaluation-tier claim (n<=3 is screening only).
- [ ] Substantial eval sets — hundreds of prompts per class, not tens.
- [ ] **Group-aware, deduplicated** train/eval splits; scaler and direction fitted on
      train only.
- [ ] Held-out **concepts** as well as held-out prompts (in-distribution steering
      success does not imply concept generalisation, arXiv:2407.12404).
- [ ] A real released benchmark as an **OOD** test where one exists.

### B5. Statistics
- [ ] **Paired** tests (Wilcoxon signed-rank / paired t / McNemar as the metric type
      dictates), with the *same seeds* across conditions.
- [ ] **Bootstrap 95% CI** on the delta (>=10k resamples).
- [ ] **Multiple-comparison correction** (Holm-Bonferroni) across the layer x alpha x
      method family.
- [ ] **Cohen's d** per cell; report the full layer x strength grid, not the tuned point
      (arXiv:2607.19364).
- [ ] Per-concept variance reported, not just the aggregate mean.
- [ ] Pre-registered screening-vs-evaluation classification and success criterion.

### B6. Confounds and corroboration
- [ ] **Length / token-count confound audit** on judged outputs (length_auc); claim only
      the margin above the larger of {baseline, confound}.
- [ ] Judge-independent corroboration: NBF / KL-to-concept, or geometry probes
      (arXiv:2602.01716).
- [ ] Instrument limitations stated as prominently as wins (base-model quirks,
      abliteration, judge ceiling).

---

## (c) OUR RE-RUN PROTOCOL (direction-fungibility, abliterated Gemma + Qwen3-4B judge)

The re-run is publishable only if all of the following ship together.

### C1. Judge gate — a separate, pre-registered, blocking artifact
- Hand-label **>=200 generations**, stratified across REFUSAL / COMPLIANCE / GIBBERISH,
  drawn from the actual steered and unsteered distributions (not a clean proxy set).
- Compute Qwen3-4B agreement vs. those labels. **Ship gate: Cohen's kappa >= 0.6 AND
  AUC >= 0.85** on that human-labelled set. Our previous AUC-0.68 judge **fails** —
  do not reuse it for any headline number.
- Judge run config: temperature 0, **3 repeat runs**, verdict-order/position randomised,
  majority vote over >=11 trials for the headline condition, caching disabled.
- Report test-retest, position bias `|P(A)-0.5| < 0.10`, and the kappa-vs-exact-match
  deflation. Publish this as the judge card **before** computing any steering delta.
- If the gate fails, escalate the judge (larger off-family model) — do **not** proceed
  and caveat.

### C2. Control battery (all five, every layer, every alpha)
1. Real diff-of-means direction.
2. **Label-shuffled direction** (the original null).
3. **Norm- and dimension-matched random unit vectors**, >=30 per (layer, alpha) cell,
   forming the null distribution. Headline statistic = **sigma of the real direction
   above this null**.
4. **Orthogonal-complement direction** (non-identifiability control).
5. Prompting baseline + no-steer baseline.

### C3. Design
- **n >= 7 seeds**, paired across all conditions (identical seeds, identical prompts).
- **>= 300 eval prompts per class** held out, group-aware split, deduplicated; the build
  pool sized so the >=500/class rule is satisfied upstream.
- Full **layer x alpha grid**, reported in its entirety — no tuned-point reporting.
- Cross-scale: repeat at >=2 model scales, since 2605.27681 shows direction-specificity
  is **model-dependent** (present at OLMo-32B / Gemma-27B, absent at Llama-70B). Our
  claim lives or dies on the scale boundary.

### C4. Statistics
- Paired Wilcoxon signed-rank, real vs. each control.
- Bootstrap 95% CI on every delta, >=10k resamples.
- Holm-Bonferroni across the layer x alpha x control family.
- Cohen's d per cell; sigma-above-null per cell.
- Ordinal gate for any EXTERNAL-READY claim: worst evaluation seed beats best baseline seed.

### C5. Confounds and corroboration
- Length / token-count AUC on judged outputs; PPL-matched comparison; gibberish rate as
  a separate axis.
- Geometry leading indicators (off-shell displacement, effective-rank drop, norm budget,
  participation ratio) as judge-independent corroboration — this is our version of the
  NBF/KL check.
- State plainly that the abliterated base inflates compliance and that this is the
  instrument, not the method.

### C6. Framing
- Do **not** headline "a shuffled control captures the effect" — that exists (see (d)).
- Headline the **boundary**: at which scale / layer / gating condition does direction
  specificity emerge, and how much of the effect is generic perturbation at <=2B. Frame
  as a *replication-plus-extension* of arXiv:2605.27681 and arXiv:2602.06801 into the
  small-model and conditional-gate regime, with a validated judge.

---

## (d) BLUNT ANSWER: DOES A SHUFFLED/RANDOM-CONTROL NULL FOR STEERING ALREADY EXIST?

**Yes. Precisely, and it directly pre-empts the bare version of our headline.**

**arXiv:2605.27681 — "Behavioural Analysis of Alignment Faking"** (Mitrani Hadida,
Karty, Williams-King, Cooney; 26 May 2026) applies **41 random unit vectors, matched in
norm and dimensionality**, at the identical layers and alpha ranges used for targeted
interventions, across multiple random seeds, and reports:

- **OLMo-32B, layer 22** — targeted animal-welfare direction moved the compliance gap
  from 0.370 to 0.510–0.720; norm-matched random directions left the gap near baseline.
  Targeted effect sat **1.10–2.25 sigma above the random null** (direction-specific).
- **Gemma-27B, layer 21** — targeted intervention **0.68–0.98 sigma above the random
  null** at certain alphas (weakly direction-specific).
- **Llama-70B, layer 58** — random directions **reliably decreased** the compliance gap,
  and the targeted direction's effect fell **well within the random-direction null
  distribution**: an explicit "**not direction-specific**" finding, attributed to generic
  perturbation rather than directional steering.

**arXiv:2602.06801 — "On the Non-Identifiability of Steering Vectors in LLMs"**
(Venkatesh, Kurapath) independently establishes the same conclusion analytically and
empirically: orthogonal perturbations achieve **near-equivalent efficacy with negligible
effect sizes**, so there exist large equivalence classes of behaviourally
indistinguishable interventions.

Supporting, weaker precedents: label-permutation controls collapsing supervised metrics
to ~0 are described as routine practice in the interpretability literature surfaced in
this pass (e.g. the confound-aware and geometric-stability lines), and
**arXiv:2603.18329 (FaithSteer-BENCH)** independently reports "illusory controllability"
— steering that is prompt-conditional rather than a stable latent directional shift.

### What this means for us
1. "A label-shuffled control captures ~97% of the real direction's effect" is **not a
   novel headline**. It is a replication of a published, model-dependent phenomenon.
2. Our result at <=2B with an AUC-0.68 judge is, as it stands, **uninterpretable**: we
   cannot distinguish "direction is irrelevant" from "instrument is blind," and the
   literature now supplies both a positive and a negative precedent, so either
   conclusion is a priori plausible.
3. The genuinely open, publishable questions we can own:
   - **The scale boundary.** 2605.27681 covers 27B–70B. Nobody has mapped where
     direction-specificity emerges in the **1B–4B** regime. Our ladder is built for
     exactly this.
   - **The conditional-gate interaction.** Does gating (CAST-style) restore
     direction-specificity that is absent unconditionally? No paper found addresses this.
   - **A validated-judge replication.** Every prior null is judge-mediated; a
     kappa->=0.6 / AUC->=0.85 validated off-family judge with an 11-trial protocol would
     be a methodological contribution in its own right.

**Recommendation: reframe now, before the re-run, so the pre-registration states the
boundary claim rather than the fungibility claim.**

---

## Verification status

Abs-page verified (title + authors + date confirmed 2026-07-25): arXiv:2605.27681,
2602.06801, 2606.19544, 2606.13685, 2603.18329, 2607.19364, 2602.01716, 2605.31183.

Cited from search-result metadata only, not individually abs-fetched in this pass —
`[UNVERIFIED]`: arXiv:2606.05958, 2407.12404 (well-established NeurIPS 2024 paper;
abs page listed in results but not fetched), 2501.17148 (AxBench; pre-existing project
citation), 2601.08331, 2605.16372, 2605.30076, 2509.13450, 2505.20645, 2510.04484,
2606.15474, 2604.02608, 2604.17698.
