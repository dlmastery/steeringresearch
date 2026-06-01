# The Geometry of the Coherence Cliff: A Pre-Registered Autoresearch Harness for Activation Steering and Its First Screening Results

*Anonymous submission — methodology, reproducible harness, and screening-results contribution.*
*Composite-metric fingerprint: `a9001e87087e` (SHA-256[:12] of the frozen scoring formula).*

---

## Abstract

Activation steering — adding or rotating a direction in a transformer's residual stream to control a behavior at inference time — is training-free, cheap, and interpretable, but it is also irreducibly multi-objective: an edit that raises a target behavior can simultaneously degrade general capability, collapse text coherence, break safety refusals, or induce over-refusal on benign inputs. A single scalar "efficacy" number systematically flatters a method by hiding the axis it sacrificed. We present a pre-registered autoresearch harness that makes this multi-objective accounting mandatory, and we report the screening observations it produced across three small models and four behaviors.

The harness contributes four mechanisms. First, a single-axis champion loop over a twelve-axis intervention taxonomy: every experiment perturbs exactly one knob of a documented best configuration and is kept only if a composite score improves with no axis regressing past its gate. Second, a cost-ordered five-rung benchmark ladder (UNIT→SMOKE→DEV→STANDARD→FULL) that scores the same five measurement axes at every rung and forbids spending expensive compute before a cheap gate is cleared. Third, a SHA-256-fingerprinted, Goodhart-resistant composite that prices behavior, capability, coherence, safety, and off-manifold displacement simultaneously, with one-sided penalties so a method cannot win by trading an unmeasured axis. Fourth, a statistical rigor floor that hard-classifies n≤3 as screening and reserves the word *finding* for n≥7 results that pass a paired Wilcoxon test, a bootstrap confidence interval, Holm-Bonferroni correction, an empirical seed-noise band, and an ordinal gate.

Our central empirical contribution is a geometric account of the coherence cliff. We parameterize steering by *relative displacement* — the edit magnitude as a fraction of the local residual norm ‖h‖ — and show the coherence cliff has a behavior-independent knee near a ten-percent displacement that is approximately scale-invariant across model sizes, where the same edit in absolute units is strongly scale-dependent. Off-shell radial displacement Δ‖h‖ predicts log-perplexity monotonically on real held-out WikiText-2 (Spearman +0.585, 95% bootstrap CI [+0.353, +0.758], p≈8×10⁻⁶, pooled over two scales), but the *coefficients* of any single collapse law do not transfer across scale (held-out R²=−1.6), so the relationship is directionally robust yet quantitatively model-specific. Additive steering moves h radially off the activation shell while norm-preserving rotation moves it angularly along the shell; an angular displacement metric predicts rotation's log-perplexity at R²=0.997 while radial displacement governs addition (R²=0.81), giving an empirical confirmation of a cylindrical (radius×angle) decomposition of steering cost. We further report cross-model, cross-behavior alignment of difference-of-means and PCA-top-1 directions (cosine 0.994–0.999 across three models and four behaviors), the falsification of Fisher-ratio layer selection as a predictor of steering efficacy (Spearman +0.14), and the falsification of a low-rank behavior-plane hypothesis (top-3 directions explain 66%, not >90%).

We are explicit about scope. Every quantitative result except the rung-3 WikiText evaluation is a single-seed screening observation on sub-1B-parameter models with partly synthetic instruments; none is an external steering-efficacy claim, and each is gated on an enumerated list of required experiments. The contribution is the harness, the geometric account, and a disciplined two-sided ledger of what survived contact with evidence and what did not.

---

## 1. Introduction

The residual stream of a decoder-only transformer is the channel through which every layer reads and writes. The linear representation hypothesis holds that human-interpretable concepts are encoded, at least approximately, as directions in this stream, and a decade of interpretability work supports a working version of it for many features [Park et al. 2023, arXiv:2311.03658; Elhage et al. 2022, toy-models]. **Activation steering** exploits this directly: identify a direction associated with a behavior — typically the difference of mean activations between contrastive prompt pairs — and at inference add, subtract, rotate, or project along it inside the residual stream. No weights are trained, the intervention is a few floating-point operations per token, and the edited direction is often human-legible. This combination of cheapness, training-freeness, and interpretability has made steering one of the most active areas of representation engineering, with the activation-addition lineage [Turner et al. 2023, arXiv:2308.10248; Zou et al. 2023, arXiv:2310.01405; Panickssery et al. 2024, arXiv:2312.06681] now spanning conditional gating, learned and distilled interventions, sparse-feature steering, and an explicitly geometric 2026 literature.

Steering is also, unavoidably, a control problem with multiple objectives in tension. The same residual edit that elicits a target behavior pushes the hidden state away from the distribution the downstream layers were trained on. Past a threshold, perplexity rises super-linearly and the model emits degenerate text; before that, capability on unrelated tasks can erode silently; and — most consequentially — steering edits can break the refusal behavior that keeps a deployed model safe. The Rogue Scalpel result [Korznikov et al. 2025, arXiv:2509.22067] demonstrates that even *random or benign* steering directions raise harmful-prompt compliance, that the damage peaks in early-middle layers, that averaging twenty weak vectors yields a universal jailbreak, and — crucially — that the attack directions are nearly orthogonal to the refusal direction (mean cosine 0.027). Safety is not broken by cancelling the refusal feature; it is broken by displacing the hidden state off the manifold where refusal is computed at all. Any honest evaluation of a steering method must therefore price behavior, capability, coherence, safety, and selectivity *together*. A scalar efficacy number does not; it lets a method "win" by quietly spending an axis nobody measured, and the autoresearch literature is full of such hollow wins.

This paper does not propose a new steering method or claim a new steering-efficacy result. It contributes a research instrument and the geometric findings that instrument produced. Concretely:

1. **A pre-registered autoresearch harness** (Section 3) that turns each experiment into a falsifiable, citation-gated, single-axis perturbation of a champion configuration, authored through a seven-step ritual whose pre-run fields are mechanically gated against placeholders and promoted along a cost-ordered ladder. The harness is content-agnostic; steering is its first instantiation.

2. **A Goodhart-resistant, fingerprinted composite** (Section 3.3) that prices all five axes plus an off-manifold geometry term with one-sided penalties, so an incoherent-but-"safe" run cannot win because the coherence tax dominates. The formula string is frozen and its SHA-256 prefix appears in every reasoning entry, ledger row, and dashboard footer; editing it to crown a favored row is a defined protocol violation.

3. **Valid measurement instruments** (Section 4) for the two axes most prone to circular self-validation: a generation-based behavior scorer that counts concept incorporation in the model's *output text* rather than projecting the edit onto itself, and a real-generation safety scorer that runs the model on harmful prompts and classifies its own refusals. We state precisely what residual circularity remains and how it is gated.

4. **A geometric account of the coherence cliff** (Section 5) grounded in the 2026 geometry-of-steering wave [Manifold Steering 2026, arXiv:2605.05115; CRH 2026, arXiv:2605.01844; Non-Identifiability 2026, arXiv:2602.06801], with the central control variable being relative off-manifold displacement and the central decomposition being radial×angular (cylindrical). This is the part of the paper with a rung-3 evaluation on real held-out text.

We frame the work as infrastructure plus screening because, by its own rigor floor, it cannot yet be more, and we regard that honesty as part of the contribution rather than a hedge against it. Section 2 situates the twelve-axis taxonomy against the sixty-paper literature; Section 3 specifies the harness; Section 4 specifies the instruments; Section 5 reports results, two-sided; Section 6 discusses what the geometry implies for state-of-the-art steering; Section 7 states limitations as scope; Section 8 enumerates the experiments required before any external claim; Sections 9–10 cover reproducibility and conclusions.

---

## 2. Background and related work

### 2.1 The residual stream and the linear representation hypothesis

A decoder-only transformer maintains, at each token position and layer ℓ, a hidden state hℓ ∈ ℝ^d — the *residual stream* — to which every attention and MLP sublayer adds its output. Because these contributions are additive and the readout (unembedding, and each layer's own reads) is approximately linear, a direction v ∈ ℝ^d can carry a concept: shifting hℓ along v shifts the model's disposition toward the associated behavior. The linear representation hypothesis [Park et al. 2023, arXiv:2311.03658] formalizes when concepts are linearly encoded and when the natural inner product is a causally-relevant (causal-inner-product) one rather than the raw Euclidean dot product — a distinction that will matter when we separate radial from angular displacement in Section 5. Activation steering is the engineering consequence: identify v, then edit hℓ ← hℓ + α·v (additive), or rotate hℓ toward v within a plane (norm-preserving), or remove hℓ's component along v (projection / ablation).

### 2.2 The activation-addition lineage (axis families 1–7)

The train-free lineage computes v from contrastive data and adds it at inference. Inference-Time Intervention [Li et al. 2023, arXiv:2306.03341] steers along probe-identified truthful directions at a small set of attention heads, lifting TruthfulQA from 32.5% to 65.1% on Alpaca, and motivates separability-based site selection. ActAdd [Turner et al. 2023, arXiv:2308.10248] contrasts a single prompt pair ("Love" − "Hate") to obtain a steering vector. Representation Engineering [Zou et al. 2023, arXiv:2310.01405] elevates population-level reading vectors as the primary object and shows emotion steering can *raise* harmful compliance despite RLHF — an early signal of the safety tension. Contrastive Activation Addition [Panickssery et al. 2024, arXiv:2312.06681] establishes the difference-of-means-of-pairs recipe we adopt as our default direction source, adding the vector at all post-prompt positions with a signed coefficient and showing it stacks on finetuning and system prompts. Refusal-is-a-single-direction [Arditi et al. 2024, arXiv:2406.11717] shows difference-of-means yields one refusal direction whose ablation removes refusal and whose addition induces it across thirteen models up to 72B. These works populate the first seven axes of our taxonomy — *where* (site/layer), *what* (direction), *how-much* (coefficient), *how* (operation), *when* (condition), *which-tokens* (span), and *how-derived* (source).

### 2.3 Conditional and gated steering (the meta-layer, axis 5)

CAST [Lee et al. 2025, arXiv:2409.05907] introduces a *condition vector* (PCA on contrastive prompt activations) plus a similarity threshold that gates a behavior vector, supporting OR/AND/complement composition, so a rule like "if the input is about hate speech, refuse" keeps harmful-refusal high while harmless-refusal stays near 2%. FineSteer's Subspace-guided Conditional Steering [Weng et al. 2026, arXiv:2604.15488] replaces the cosine threshold with an energy-ratio gate. Selective Steering [Dang & Ngo 2026, arXiv:2601.19375] gates by discriminative layer (steering only where class means have opposite sign) with a norm-preserving rotation. In-Distribution Steering [2025, arXiv:2510.13285] and Contextual Linear Activation Steering [Hsu et al. 2026, arXiv:2604.24693] modulate the edit to keep steered activations inside the natural distribution. The taxonomy insight we inherit and use throughout is that conditional steering is *not a peer injector but a meta-layer*: because the gate is a read-only probe on a different axis or layer than the write, it stacks on nearly every injector. The cheap-condition-vector requirement of CAST is exactly why our difference-of-means≈PCA-top-1 alignment screen (Section 5.4) is decision-relevant.

### 2.4 Learned, sparse, and geometry-aware steering (axes 7–12)

A *learned* family replaces the fixed difference-of-means vector with a trained object: low-rank representation finetuning [Wu et al. 2024, arXiv:2404.03592], bidirectional preference steering [Wu et al. 2025, arXiv:2505.20809], hypernetwork-generated vectors [HyperSteer 2025, arXiv:2506.03292], a distilled token-specific gain field that mimics prompting [PSR, Heyman & Vandeputte 2026, arXiv:2605.03907], and a concept-conditioned velocity field integrated over multiple steps [FLAS 2026, arXiv:2605.05892]. A *sparse* family steers in the interpretable basis of a sparse autoencoder, anchored by GemmaScope [Lieberum et al. 2024, arXiv:2408.05147] and SAE-targeted steering [SAE-TS, Chalnev et al. 2024, arXiv:2411.02193; FGAA 2025, arXiv:2501.09929]. The decisive evaluation result here is AxBench [Wu et al. 2025, arXiv:2501.17148]: in an apples-to-apples comparison across prompting, probing, SAE, steering, and ReFT, *SAEs are not competitive* and prompting wins steering — which is precisely why a prompting baseline is a non-negotiable blocker for any efficacy claim (Section 8) and why we make none.

The 2026 **geometry wave** is the literature our findings speak to most directly. It replaces "find the right direction" with "find the right geometry." Manifold Steering [2026, arXiv:2605.05115] fits an activation manifold and steers *along* it, arguing linear Euclidean steering cuts through off-manifold regions. The Cylindrical Representation Hypothesis [CRH, Gao et al. 2026, arXiv:2605.01844] decomposes representation differences on a cylinder S^{d−1}×ℝ into a radial (magnitude) and an angular (semantic-direction) component, predicting that plain additive coupling is unstable because it conflates the two. Non-Identifiability of steering vectors [2026, arXiv:2602.06801] proves that, under single-layer white-box access, large equivalence classes of behaviorally indistinguishable interventions exist, so behavioral testing alone cannot recover "the" direction — structural constraints are required. Steered-states-are-non-surjective [Mishra et al. 2026, arXiv:2604.09839] proves steered activations lie off the manifold reachable by any prompt, with empirical confirmation on Gemma-3-1B, Llama-3.2-1B, and Qwen-2.5-0.5B. Curveball [2026, arXiv:2603.09313], HyCon [2026, arXiv:2603.14093], and GeoSteer [2026, arXiv:2601.10229] round out the path-curvature, hyperbolic-metric, and trajectory-dynamics directions. Our relative-displacement parameterization, our radial×angular decomposition, and our off-shell cliff predictor are direct, *measured* engagements with this wave on small models.

### 2.5 Safety and evaluation

The Rogue Scalpel [Korznikov et al. 2025, arXiv:2509.22067] is the safety frontier we design against: steering breaks refusal off-manifold rather than by cancelling the refusal feature, which makes safety a dominantly-weighted axis and an automatic-discard gate in our composite. A mechanistic case study [Cheng et al. 2026, arXiv:2604.08524] localizes refusal steering to the OV circuit and shows 90–99% sparsifiability. The evaluation substrate — AxBench, JailbreakBench, XSTest, MMLU/ARC/GSM8K, WikiText-103 perplexity — defines the four-to-five things a steering method must report at once, and our five measurement axes are a direct operationalization of it.

### 2.6 The twelve-axis taxonomy as our organizing frame

We organize the sixty-paper literature, and our own experiments, along twelve axes: the seven classical knobs (site, direction, coefficient, operation, condition, span, source) plus five meta-axes that change the *space* the first seven live in — geometry (chord vs geodesic path), metric (Euclidean/spherical/hyperbolic/cylindrical), identifiability (which representative of the equivalence class, and its null space), dynamics (one-shot vs trajectory), and basis (dense vs sparse, with an interference budget). Most "new methods" in the arXiv firehose are a new *point* in this space, not a new dimension. The taxonomy tells the harness which axes are orthogonal — and therefore safe to perturb one at a time — and which combinations stack versus compete: two methods stack iff they differ on site or are orthogonal on direction; they compete iff they share a site and direction but differ on operation (additive vs rotational on the same plane), or if they jointly exhaust the norm/manifold budget. The Euclidean, static, single-vector *defaults* of the meta-axes are exactly the "naive additive steering" that the Rogue Scalpel punishes — which is why the geometry of the edit, not just its direction, is the object of study.

---

## 3. The autoresearch harness

The harness operationalizes one invariant: *start from the current best configuration; change exactly one thing; keep it iff the composite improves at matched coherence with no axis regressing past its gate; revert otherwise.* This is a steering-adapted Karpathy loop, with three deliberate departures: the champion configuration is sacred and every experiment is a single-axis perturbation of it; the researcher (here, an agent) supplies an explicit pre-registered mechanism rather than blind search; and promotion is ladder-bound, not time-bound.

### 3.1 The twelve-axis intervention taxonomy as an experiment generator

Every experiment perturbs exactly one of the twelve axes of Section 2.6. Because the taxonomy encodes orthogonality, a single-axis perturbation has an interpretable marginal effect, and the stack-vs-compete rule tells the loop which two priors may be combined and which will collide. The "everything on" hybrid is forbidden — it is uninterpretable and, empirically across autoresearch, a reliable way to lose to a simpler configuration.

### 3.2 The five-rung ladder

The same five measurement axes — behavior efficacy, capability retention, coherence, safety integrity, selectivity — are scored at every rung; only dataset size and realism grow.

| Rung | Nickname | Cost/run | Proves | Gate to next rung |
|---|---|---|---|---|
| 0 | UNIT | seconds | plumbing works | vector changes logits; state restores exactly |
| 1 | SMOKE | 1–3 min | right direction | monotone effect + bounded PPL + no safety leak |
| 2 | DEV | 10–20 min | generalizes a little | beats baseline on held-out concepts at matched coherence |
| 3 | STANDARD | 1–3 h | real result | Pareto-dominates prior method (no axis regresses) |
| 4 | FULL | half-day+ | publication | full multi-axis win + ablations + red-team neutralized |

A method may not consume rung-(k+1) compute until it clears rung k's gate, and a regression at any rung demotes it with a logged `failure_reason`. This is experimental economics: never run an expensive benchmark to find a bug a cheap one would catch.

### 3.3 The Goodhart-resistant fingerprinted composite

Steering has no single scalar, so the composite must price every axis at once. The frozen formula (fingerprint `a9001e87087e`) is

```
composite = behavior_efficacy
          − λ_cap     · max(0, mmlu_drop_pp)          (capability tax)
          − λ_coh     · max(0, dppl_norm)             (coherence tax — perplexity)
          − λ_coh_rep · max(0, repetition_rate)       (coherence tax — repetition)
          − λ_safe    · compliance_rate               (safety leak — Rogue Scalpel)
          − λ_sel     · max(0, harmless_refusal_rate) (over-refusal / selectivity)
          − λ_geo     · max(0, offshell_displacement) (off-manifold leading indicator)
```

with pinned weights λ_cap=1.0, λ_coh=0.5, λ_coh_rep=0.5, **λ_safe=2.0** (dominant), λ_sel=1.0, λ_geo=0.25. The one-sided `max(0,·)` penalties mean an axis can only hurt, never inflate, the composite. A run that emits gibberish scores "safe" on harm but fails coherence; the coherence tax then dominates and it cannot win — a property the unit tests assert directly. The dominant safety weight and an independent auto-discard gate encode the Rogue Scalpel mandate. The fingerprint prevents mid-project tampering: editing the formula to crown a favored row breaks the attestation in every reasoning entry and dashboard footer. Fingerprinting does *not* make the *initial* weight choice defensible, and we have not yet run a weight-sensitivity analysis; this is a stated limitation (Section 7), not a settled question.

### 3.4 The seven-step ritual

Each experiment authors a pre-run reasoning entry — **Diagnose** (≥60 words, naming the specific failure mode and referencing a prior experiment by tag), **Cite** (a real arXiv paper in full format motivating the change), **Hypothesize** (the residual-stream mechanism, naming which axis moves and what the cited paper predicts), **Predict** (a numeric range on the composite and at least one sub-metric, stored *before* the run) — and, after execution, a post-run entry — **Analyse** (actual vs predicted, verdict KEEP / DISCARD / NEAR-MISS, per-axis narrative) and **Checkpoint**. The runner refuses to fabricate pre-run fields: a missing or placeholder diagnosis, citation, hypothesis, or prediction is a protocol violation that blocks the launch. Reasoning quality gates experiment quality.

### 3.5 The screening → hill-climb → evaluation funnel

(1) **Screen** one configuration per hypothesis at a documented baseline (cheap, n≤3). (2) **Hill-climb** a surfaced candidate by coordinate descent over the steering cube — (layer × α × source × operation × span) × seed — for 20–25 trials under a strict-improvement champion rule. (3) **Confirm** at n≥7 under the full rigor contract before any external claim. By the rigor floor, n≤3 is *screening, full stop*: n=3 cannot reach p<0.05 under a paired Wilcoxon signed-rank test, so calling an n=3 result a winner is forbidden, and reclassifying a loser as "screening" after the fact is HARKing and a blocker. A claim is external-ready only when the *worst* evaluation seed beats the *best* baseline seed (the ordinal gate) and the paired Wilcoxon, bootstrap CI, Holm-Bonferroni correction, and empirical seed-noise band all hold.

---

## 4. Measurement instruments for circularity-prone axes

Two of the five axes — behavior efficacy and safety integrity — are uniquely prone to a measurement that validates itself. We specify the instruments we use and the residual circularity each retains, because the validity of every number in Section 5 rests on getting these right.

### 4.1 Behavior efficacy: generation-based, not projection-based

The naive behavior scorer measures the change in the mean projection of layer-ℓ activations onto the unit target direction v̂, between a steered and an unsteered run. This is invalid by construction. The steered run *is* h ← h + α·v with v̂ = v/‖v‖, so the measured quantity reduces to

```
Δproj = mean[(h + α·v)·v̂] − mean[h·v̂] = α·(v·v̂) = α·‖v‖,
```

a deterministic, monotone function of the very edit applied. It cannot fail for any non-degenerate v and positive α; it measures *that addition adds*, and under a logistic squash it saturates to 1.0 by α=1. A "monotone behavior effect" gate built on it is passed by a tautology, and no efficacy, monotonicity, or Pareto claim can rest on it.

Our behavior instrument therefore scores **concept incorporation in generated text**: it generates with the model under the steering edit and counts concept-lexicon stems in the *output*, steered versus unsteered, never touching the injected vector. This exposes structure the projection proxy hides — most importantly that behavior is *non-monotonic* in α: it peaks near a small displacement and then declines as over-steering destroys the very text that would express the concept (Section 5.1, 5.3). One residual circularity remains and is gated rather than hidden: the concept lexicon is derived from the same contrast pairs that built the direction, so a model that parrots pair vocabulary can inflate the score. A held-out lexicon and an independent cross-family judge are required before any efficacy magnitude is claimed (Section 8).

### 4.2 Safety integrity: real generation, tagged stubs

The naive safety axis returned a constant refusal string for every harmful prompt, forcing compliance and harmless-refusal rates to zero by construction, so the dominant safety penalty and the auto-discard gate could never fire. Our safety instrument generates with the real model on harmful prompts and classifies SAFE/UNSAFE with a rule-based refusal detector on the model's *own* text. This makes the Rogue Scalpel direction measurable: compliance rises with α (Section 5). For the offline fast-path, a single refusal placeholder is retained but **tagged** (`safety_real=False`) so a stubbed value can never be mistaken for a measurement; real rows carry `safety_real=True`. The refusal detector is itself a same-family rule-based judge on the model's own generations — a circularity we disclose and gate (Section 8) behind a calibrated judge on real JailbreakBench and XSTest.

### 4.3 Geometry probes

Every run logs the geometry leading-indicators: off-shell *radial* displacement Δ‖h‖ = ‖h_steered‖ − ‖h‖, *angular* displacement 1 − cos(h, h_steered), effective-rank change, and the cumulative norm budget ‖Δh‖/‖h‖. Radial and angular displacement are the two coordinates of the cylindrical decomposition (Section 5.5) and are the cheapest behavior-free predictors of the coherence cliff. The off-shell term is priced in the composite while its predictive validity is itself under test — an entanglement we disclose (Section 7) and resolve by reporting the geometry results against held-out perplexity, not against the composite that prices them.

---

## 5. Screening results

> Scope of this section. Every number is single-seed (n=1) screening on sub-1B models with partly synthetic instruments, **except** the rung-3 WikiText-2 evaluation in Section 5.6, which uses real held-out text with a bootstrap CI. Screening results establish a mechanistic *direction*, never a magnitude, and none is an external steering-efficacy claim. Sources: `EXPERIMENT_LEDGER.md` (exp#1–109), `ideas/_campaigns/`, `FINDINGS.md` (S-1…S-14, R3-1).

Three models anchor the results: Qwen2.5-0.5B-Instruct (a non-gated bring-up surrogate), and the gated Gemma-3 family at 270M and 1B parameters. The default direction is difference-of-means on contrastive pairs; decoding is greedy; the default operation is additive unless stated.

### 5.1 The coherence cliff is super-linear with an identifiable knee

On Qwen2.5-0.5B at its max-Fisher layer L21, with the generation behavior scorer and real safety generation, an absolute-α sweep gives:

| α | behavior (generation) | PPL | compliance rate | composite |
|---:|---:|---:|---:|---:|
| 0 | 0.500 | 48.9 | 0.30 | −0.107 |
| 1 | **0.694** (peak) | 58.7 | 0.30 | **−0.073** (best) |
| 2 | 0.526 | 89.0 | 0.30 | −0.717 |
| 4 | 0.494 | 293.6 | 0.60 | −3.597 |
| 8 | 0.346 | 3 787 | 1.00 | −40.602 |

Two facts the projection proxy had hidden. First, behavior is **non-monotonic**: it peaks at α≈1 (0.69) and then declines (0.53, 0.49, 0.35) — over-steering destroys the behavior itself, not merely coherence, because the text that would express the concept is degraded. Second, steering **compromises safety**: real compliance rises 0.30→0.60→1.00 as α goes 0→4→8, the Rogue Scalpel effect, now measured rather than assumed. Perplexity grows super-linearly (+20%, +82%, 6×, 77× across α=1→2→4→8) with a knee at α≈1–2, and the composite peaks at the behavior/coherence sweet spot α≈1. We read the <0.5pp capability change on the 20-item synthetic tripwire as a *limitation of that tripwire*, not as evidence of capability robustness.

### 5.2 Scale-dependent fragility: smaller models leave the manifold earlier

On Gemma-3-270M at L12 (the max-Fisher layer), generation behavior + real safety:

| α | behavior | PPL | ΔPPL_norm | CR | off-shell Δ‖h‖ | composite |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.500 | 90.2 | 0.00 | 0.80 | 0.000 | −1.107 |
| 1 | 0.438 | 149.2 | +0.65 | 0.80 | 0.021 | −1.602 |
| 2 | 0.319 | 322.3 | +2.57 | 0.90 | 0.057 | −2.889 |
| 4 | 0.217 | 2 775 | +29.8 | 1.00 | 0.175 | −16.91 |
| 8 | 0.211 | 141 578 | +1568 | 1.00 | 0.535 | −786.4 |

The 270M model is markedly more fragile than the 0.5B model: its cliff is at α≈1 (perplexity already +65%) versus Qwen's α≈2, and its behavior *never improves* under steering (0.50→0.44→0.32→0.22, a monotone decline). There is no clean steering window at L12 — the model leaves the manifold *before* the concept can be cleanly injected, whereas Qwen-0.5B had a behavior peak at α=1. Adding Gemma-3-1B at L18 completes a monotone-in-scale picture:

| model | base PPL | PPL at α=1 | behavior at α=1 | clean steering window? |
|---|---:|---:|---:|---|
| Gemma-3-270M | 90 | +65% | 0.44 ↓ | none (most fragile) |
| Qwen-2.5-0.5B | 49 | +20% | 0.69 ↑ | yes |
| Gemma-3-1B | 74 | +41% | 0.65 ↑ | yes (least fragile) |

Larger models carry more coherence budget and exit the data manifold more slowly. The clean steering window — a regime where behavior rises before coherence collapses — *emerges with scale*. This is consistent with the hypothesis that small models are disproportionately fragile to additive edits; it does not by itself adjudicate rotation-vs-addition, which Section 5.5 takes up.

### 5.3 Relative steering is the right parameterization, and its knee is scale-invariant

Absolute α conflates the edit with the (large, scale-varying) residual norm: at L16 of Gemma-3-270M, ‖h‖ is large enough that even 40× a unit vector produces negligible off-shell displacement, while a raw difference-of-means vector (norm ~10× a unit PCA vector) makes matched-α comparisons across sources meaningless. The control variable is displacement *relative* to ‖h‖. We add a `relative_add` operation, Δh = α·‖h‖·v̂, where α is the fractional displacement. On Gemma-3-270M at L16:

| α (fraction of ‖h‖) | DiffMean behavior | DiffMean PPL | PCA behavior | PCA PPL | off-shell |
|---:|---:|---:|---:|---:|---:|
| 0.02 | 0.532 | 92.3 | 0.532 | 92.4 | 0.002 |
| 0.05 | 0.445 | 100.0 | 0.439 | 101.3 | 0.005 |
| 0.10 | **0.614** | 131.9 | **0.591** | 132.6 | 0.010 |
| 0.20 | 0.504 | 245.2 | 0.475 | 255.4 | ~0.030 |
| 0.40 | 0.319 | 1 623 | 0.304 | 1 759 | ~0.093 |

The relative cliff is clean and ‖h‖-independent: behavior peaks at a **ten-percent displacement** and then declines, perplexity rises monotonically, and the knee (off-shell ≈0.01–0.03) coincides with the geometry law of Section 5.6. Repeating the relative sweep on Gemma-3-1B at L18 places the knee at a similar 5–10% fraction of ‖h‖. In *relative* units the cliff location is approximately **scale-invariant**, even though in *absolute* α it is strongly scale-dependent. This is the stronger, more useful form of the cliff result: relative displacement is the scale-portable control variable, and a practitioner tuning a new model can target ~10% of ‖h‖ rather than re-discovering α from scratch.

### 5.4 DiffMean ≈ PCA-top-1, across models and behaviors

CAST-style gating wants a cheap condition vector; if difference-of-means and PCA-top-1 are tightly aligned, the cheaper difference-of-means suffices. They are. At each behavior's max-Fisher layer, cos(DiffMean, PCA-top-1) = 0.996 on Qwen-0.5B (L21), 0.994 on Gemma-3-270M (L12), and 0.9945 on Gemma-3-1B (L18). Across four behaviors on Gemma-3-270M the alignment is anger 0.995, formality 0.996, happiness 0.999, ocean 0.997. Moreover, at matched *fractional* α the two sources steer near-identically (behavior within 0.02, perplexity within ~8%; Section 5.3 table), so the 0.99 cosine implies *equivalent steering*, not merely aligned directions — the earlier apparent source differences were pure norm-parameterization artifacts. This is the most robust screening observation in the program, holding across three architectures and four behaviors, and it substantially addresses the single-behavior limitation for this specific claim.

### 5.5 Radial vs angular: a cylindrical decomposition of the coherence cost

Operation matters geometrically, and the natural metric is cylindrical (radius × angle). An additive edit changes the norm — it moves h *radially* off the activation shell — while a norm-preserving rotation moves h *angularly* along the shell toward a wrong direction. A first, naive add-vs-rotate comparison surfaced that the coefficient α is *incommensurable* across operations: rotation reads α as an angle in radians, so α=1–4 rad (57–229°) catastrophically scrambles every token (perplexity 10¹⁰–10¹⁸, degenerate constant output), and the off-shell radial metric registers Δ‖h‖≈0 throughout because rotation preserves norm. A fair comparison must match by displacement and use small angles. In that valid small-angle regime at L16 of Gemma-3-270M:

| operation | α | behavior | PPL | radial Δ‖h‖ | angular 1−cos | composite |
|---|---:|---:|---:|---:|---:|---:|
| add | 0.05 | 0.470 | 90.5 | 0.000 | 0.0000 | −1.14 |
| add | 0.10 | 0.528 | 90.9 | 0.001 | 0.0001 | −1.08 |
| add | 0.20 | **0.565** | 92.8 | 0.002 | 0.0002 | −1.06 |
| add | 0.50 | 0.485 | 99.1 | 0.007 | 0.0013 | −1.42 |
| rotate | 0.05 | 0.497 | 100.2 | 0.001 | 0.0011 | −1.42 |
| rotate | 0.10 | 0.569 | 131.4 | 0.003 | 0.0046 | −1.72 |
| rotate | 0.20 | 0.460 | 255.4 | 0.002 | 0.0184 | −2.61 |
| rotate | 0.50 | 0.245 | 11 211 | 0.002 | 0.1116 | −63.6 |

Two results. First, **full-vector rotation is *not* gentler than addition** on this small model: `add` holds perplexity 90–99 across its whole range while `rotate` degrades 100→131→255→11 211, and at matched behavior ≈0.57, add@0.2 (PPL 92.8) beats rotate@0.1 (PPL 131.4) by 42%. This contradicts the "rotation is gentler on small models" hypothesis for full-vector rotation. The caveat is precise and we hold to it: this rotation turns the *entire* hidden state toward v, whereas the corpus Angular and Selective methods rotate *selectively* within a single 2D plane; a selective-rotation operator is queued as future work and the falsification is scoped to full-vector rotation. Second, and more important methodologically, **each operation's cost is predicted by its own cylindrical coordinate**: over the mixed add+rotate rows, radial Δ‖h‖ predicts log-perplexity at only Pearson −0.13 (it is blind to rotation), but angular displacement predicts at +0.998 with `log PPL = 4.57 + 43.1·angular`, R²=0.997. Combined with the radial law for additive steering (Section 5.6, R²=0.81), the complete coherence predictor is the **radial×angular displacement** — an empirical confirmation, on small models, of the Cylindrical Representation Hypothesis: additive steering pays for moving off the shell, rotation pays for moving around it, and each component prices the perplexity cost of its own operation.

### 5.6 The off-shell predictor on real held-out text (rung-3 evaluation)

This is the program's first rung-3 evaluation, on real held-out English prose (WikiText-2, 40 passages), pooling n=50 (model × layer × α) points across Gemma-3-270M and Gemma-3-1B under relative_add steering, with a 10k-resample bootstrap and a held-out cross-scale generalization test.

- **The monotone off-shell→incoherence relationship holds on real data.** Spearman(off-shell Δ‖h‖, log real-PPL) = **+0.585**, 95% bootstrap CI **[+0.353, +0.758]** (excludes zero), p ≈ 8×10⁻⁶. More radial displacement means higher real perplexity, across two model scales on genuine text.
- **No single collapse law transfers across scale.** Fitting `log PPL = a + b·Δ‖h‖` on the 270M points and predicting the 1B points gives held-out **R² = −1.6** (worse than predicting the mean): the 270M fit (slope ≈78.85, intercept ≈4.65) does not transfer. The within-pool screening R²=0.81 reported earlier was an artifact of mixing models in one fit.

The honest reading is that the relationship is *directionally robust but quantitatively model-specific*. We log the caveat that the 50 points are (layer × α) configurations, not independent iid seeds, so the bootstrap CI is a within-grid estimate; cross-behavior, multi-seed replication is the next rigor step. This is exactly the failure mode rung-3 held-out validation exists to catch — and exactly what the within-pool fit had masked.

### 5.7 Composition, sparsity, and transport (stacking screen)

A stacking screen on Gemma-3-270M at L16 across four behaviors (ocean, happiness, anger, formality) yields several two-sided observations.

- **Behavior vectors compose additively when near-orthogonal.** Pairwise concept-vector cosines are mostly |cos|<0.3 (anger↔formality −0.18, formality↔ocean −0.10), except anger↔happiness +0.48 (both high-arousal emotions). Stacking anger+happiness (summed, relative 0.1 each) retains each behavior fully (anger 101%, happiness 110% of solo) — no interference even at +0.48 correlation; the shared emotional direction helps rather than hurts.
- **The norm budget governs the *cumulative* displacement of a stack.** Scaling the total stacked α, perplexity collapses super-linearly (138→182→410→4518 for α=0.05→0.4), so the coherence cliff applies to the cumulative displacement, consistent with a conserved edit budget. Stacking N=1…4 vectors retains 100/86/85/94% mean behavior (composition works above 85%), but retention is *non-monotone* in Gram off-diagonal mass (0→0.37→1.80→3.07), so the strict "interference ∝ Gram mass" form is not cleanly supported.
- **The behavior direction is approximately parallel-transported across depth.** Adjacent-layer cosines of the difference-of-means direction are high (L12–L14 0.85, L14–L16 0.75, L16–L17 0.90), supporting a transport view of multi-layer steering.
- **The behavior plane is not low-rank.** The top-3 SVD directions of the {pos−neg} set explain only 66% of variance, not >90%; and the top-10% of coordinates retain 77% of behavior (below an 85% threshold) — behavior is moderately, not extremely, sparse.

### 5.8 What the screen falsified

The harness is designed to falsify as readily as to confirm, and it did. **Fisher-ratio layer selection is falsified as a predictor of steering efficacy**: a Gemma-3-270M layer sweep at α=2 gives Spearman(Fisher ratio, behavior) = +0.14 (p=0.74), far below the pre-registered ≥0.7; the max-Fisher layer L12 is not the best steering layer (L16 yields more behavior at lower perplexity), and the most-separable layer is also the *most fragile*. Linear separability does not imply controllability — a screening-level corroboration of the interpretability≠controllability theme. **A low-rank behavior plane is falsified** (5.7). **Full-vector rotation-beats-addition on small models is falsified** (5.5). Per-layer effective rank does *not* predict layer fragility (Spearman −0.21, underpowered). Across the program, eight pre-registered hypotheses received screening verdicts — three supported geometry hypotheses (the off-shell predictor, the cylindrical decomposition, the norm budget), the cross-model/cross-behavior DiffMean≈PCA alignment, the relative-steering and scale-emergence cliff results, two clean falsifications (Fisher selection, full-vector rotation), one directional-but-underpowered result (difference-of-means pair-count knee, which stabilizes by ~5 pairs on an easy synthetic concept but cannot test the ≥50-pair claim), and one inconclusive (effective-rank fragility). Pre-registered hypotheses are *falsified*, *supported*, or turned into *instrument refinements* — never silently confirmed.

---

## 6. Discussion: what the geometry implies for state-of-the-art steering

**Off-manifold displacement is the control variable, not the coefficient.** The single most actionable result is that the coherence cliff is organized by displacement relative to the local residual norm, with a knee near ten percent that is approximately scale-invariant. The raw coefficient α is the wrong variable to expose to a practitioner: it conflates the edit with a large, layer- and model-dependent norm, which is why absolute-α cliffs move across scale while relative-α cliffs do not. State-of-the-art methods that adapt the edit to keep activations in-distribution [IDS 2025, arXiv:2510.13285; Contextual Linear Steering 2026, arXiv:2604.24693] are, in this language, implicitly regulating off-shell displacement; making that the *explicit* control variable — cap ‖Δh‖ at a fixed fraction of ‖h‖ — is a cheap, model-portable default that needs no per-model α search. The norm-budget result (5.7) extends this to stacks: it is the *cumulative* displacement of all simultaneously active vectors, not the count of vectors, that must stay under budget, which gives a concrete capacity rule for multi-vector safety stacks.

**The right metric is cylindrical, and the right predictor has two coordinates.** Additive and rotational steering fail for geometrically distinct reasons — radial excursion off the shell versus angular excursion along it — and each is predicted, near-perfectly in the small-angle regime, by its own coordinate (radial R²=0.81 for addition, angular R²=0.997 for rotation). A coherence predictor that tracks only one coordinate is blind to half the failure modes: the off-shell radial metric, taken alone, would have certified catastrophic rotations as harmless because they preserve norm. This is a small-model, measured confirmation of the cylindrical representation picture and a direct argument that a deployed steering controller should monitor *both* the radial and angular displacement of every edit, not the edit's raw magnitude.

**Source choice is a solved sub-problem; site and geometry are not.** Difference-of-means and PCA-top-1 are equivalent for steering once the magnitude is parameterized relatively (5.4), so the cheaper difference-of-means is the right default and the "which source" axis is largely settled for this regime. By contrast, the *site* axis is not predicted by linear separability (5.8) — the convenient Fisher heuristic fails, and the most-separable layer is the most fragile — so layer selection must be made on a steering-and-coherence objective directly, not on a probe. And the *geometry* axis (operation, metric, path) is where the coherence budget is actually won or lost.

**Scale changes the regime, not just the magnitude.** The clean steering window — behavior rising before coherence collapses — is absent at 270M, present at 0.5B and 1B, and widens with scale. Results obtained on sub-1B models therefore cannot be assumed to hold at deployment scale without re-checking the *existence* of the window, not merely its width. This is a caution we apply to our own results as forcefully as to others': our cylindrical and off-shell findings are screening on small models and inherit this caveat.

**Behavioral equivalence does not pin the edit.** The non-identifiability of steering vectors [2026, arXiv:2602.06801] and the off-manifold mechanism of the Rogue Scalpel together imply that two edits with identical behavior can differ arbitrarily in collateral damage, and that the damage is an off-manifold-displacement phenomenon rather than a feature-cancellation one. Our results give the practical corollary: among behaviorally-equivalent edits, prefer the one with the smallest radial-plus-angular displacement, and measure that displacement directly because behavior alone cannot reveal it.

---

## 7. Limitations as scope

We state scope as a researcher states the boundary of a claim, not as an apology for it. Within the boundary the results are sound; outside it we make no claim.

1. **Single seed except the rung-3 evaluation.** Every result in Sections 5.1–5.5 and 5.7–5.8 is n=1; by the rigor floor these are screening, and we report no significance test or seed band for them. The rung-3 WikiText evaluation (5.6) carries a bootstrap CI but its 50 points are (layer × α) configurations, not iid seeds, so the CI is a within-grid estimate.
2. **Partly synthetic instruments.** The capability axis is a deterministic forward-pass corruption tripwire (20 synthetic MCQ items), not an accuracy measurement on MMLU; the contrast pairs and concept lexicons are small and hand-written. The WikiText-2 evaluation is real; the capability and safety substrates are not yet.
3. **Sub-1B models.** Real runs are on Gemma-3-270M, Gemma-3-1B, and Qwen2.5-0.5B. The standard reporting model (Gemma-2-2B-it) has not been exercised end-to-end, and the steering-window regime is known to change with scale (6).
4. **Same-family judges.** Behavior and safety are scored by same-family rule-based instruments on the model's own text; no human-calibrated cross-family judge exists yet. The behavior lexicon derives from the extraction pairs (4.1) and the refusal detector reads the model's own generations (4.2).
5. **Unjustified composite weights.** The λ weights are pinned and fingerprinted but not derived; the axes are not commensurate (an unbounded perplexity term can dominate), and no weight-sensitivity or champion-ordering-robustness analysis has been run.
6. **Geometry-probe entanglement.** Off-shell displacement is priced in the composite while its predictive validity is under test. We resolve this for the reported geometry results by validating them against held-out perplexity (5.6), not against the composite that prices them.
7. **Quantization transfer untested.** 4-bit↔fp16 invariance — a precondition for the "small-GPU results transfer" story — has not been verified.

These are not caveats appended to results; they are the current state of the evidence, and they define exactly which sentences in this paper are claims and which are directions.

---

## 8. Required experiments before any external steering claim

These are ordered, each gating the claim it supports:

1. **Independent behavior judge.** Replace the lexicon scorer with an LLM-as-judge or AxBench scorer on real generated text, calibrated against a human-annotated slice (target ≈94% judge precision). No efficacy, monotonicity, or Pareto claim ships otherwise. (Generation-based scorer landed; calibration pending.)
2. **Real safety and selectivity.** Generate on real JailbreakBench (100 prompts, 10 categories) and XSTest, judge SAFE/UNSAFE with a calibrated judge, confirm baseline compliance ≈0%, and demonstrate that the λ_safe penalty and auto-discard can fire. (Real generation landed; real benchmarks and calibrated judge pending.)
3. **Real datasets.** Real AxBench concept set with a held-out-concept split, real MMLU (≥500 items), real WikiText-103 perplexity; retire the synthetic slices to UNIT/SMOKE only.
4. **n≥7 with the full rigor contract.** Paired Wilcoxon + 10k-bootstrap CI + Holm-Bonferroni + empirical 2σ_seed band + ordinal gate, on the pre-registered evaluation split.
5. **Prompting baseline and controls.** The AxBench apples-to-apples comparison, plus random-direction and mean-ablation controls; a steering "win" is only a contribution relative to a tuned prompt.
6. **Gemma-2-2B reproduction and weight sensitivity.** Reproduce on the standard model, show the champion ordering survives a λ-perturbation, and confirm 4-bit↔fp16 invariance.
7. **Held-out-concept generalization and a shuffle-test negative control.** Confirm the method works on concepts not used in extraction and *fails* under label/condition shuffling.

Until items 1–5 are real, the program supports no external steering-efficacy claim; it stands as a methodology, reproducible-harness, and screening-results contribution. This paper claims exactly that, and the geometric findings of Section 5 are scoped accordingly.

---

## 9. Reproducibility

**Code and rigor gates.** `ruff check src/steering tests` (clean), `mypy src/steering --ignore-missing-imports` (clean), `pytest tests/` (green) — covering the rung-0 plumbing, the mechanism-asserting hook/extract/geometry tests, the Goodhart composite tests, and the markdown-leak dashboard tests. The composite fingerprint `a9001e87087e` is asserted stable and re-derives from the scoring source.

**Model and license note.** Gemma is gated: reproduction requires accepting the Gemma license on Hugging Face and authenticating (`huggingface-cli login`); the token is read from the environment and never committed. Real runs used Gemma-3-270M-it and Gemma-3-1B-it; the standard rung targets Gemma-2-2B-it. Qwen2.5-0.5B-Instruct is the non-gated bring-up surrogate.

**Determinism and commands.** Greedy decoding for all gates; seeds threaded through `random`/`numpy`/`torch`/CUDA. A representative real-Gemma screening run:

```powershell
$env:PYTHONPATH = "src"
python scripts/hf_fetch.py google/gemma-3-270m-it          # after accepting the license
$env:E3_MODEL = "models/google/gemma-3-270m-it"
python ideas/30_alpha_coherence_cliff/run_e3.py            # the relative-α cliff screen
```

An offline fast path (`--model fake`) runs the full reasoning→runner→ledger→dashboard loop without any model download for plumbing verification. Every artifact — the append-only `experiment_log.jsonl`, `best_config.json`, and the three-tier dashboard — is regenerated and checkpointed each milestone. All inherited corpus numbers are tagged `[NEEDS VERIFICATION]` until reproduced here.

---

## 10. Conclusion

We contribute a pre-registered autoresearch harness for activation steering whose discipline — a single-axis champion loop, a cost-ordered ladder, a fingerprinted Goodhart-resistant composite that prices all five axes plus off-manifold displacement, and a statistical rigor floor that hard-separates screening from evaluation — turns each experiment into a falsifiable, citation-gated unit. On valid instruments (a generation-based behavior scorer and a real-generation safety scorer), the harness produced a coherent geometric account of the coherence cliff: relative off-manifold displacement is the scale-portable control variable, with a knee near ten percent of the local residual norm; the coherence cost decomposes cylindrically into a radial component that prices addition (R²=0.81) and an angular component that prices rotation (R²=0.997); the off-shell predictor holds monotonically on real held-out WikiText-2 (Spearman +0.585, CI excludes zero) even though no single collapse law transfers across scale (held-out R²=−1.6). Difference-of-means and PCA-top-1 are equivalent steering sources across three models and four behaviors; Fisher-ratio layer selection and a low-rank behavior plane are falsified. Every steering-efficacy magnitude remains gated on real AxBench with a calibrated judge, real safety benchmarks, n≥7 under the full rigor contract, a prompting baseline, and a Gemma-2-2B reproduction. The contribution is the harness, the geometry, and a two-sided ledger of what survived.

---

## References

- K. Li, O. Patel, F. Viégas, H. Pfister, M. Wattenberg. Inference-Time Intervention: Eliciting Truthful Answers from a Language Model. NeurIPS 2023. arXiv:2306.03341. https://arxiv.org/abs/2306.03341
- A. M. Turner, L. Thiergart, G. Leech, D. Udell, J. J. Vazquez, U. Mini, M. MacDiarmid. Steering Language Models With Activation Engineering (ActAdd). 2023. arXiv:2308.10248. https://arxiv.org/abs/2308.10248
- A. Zou, L. Phan, S. Chen, J. Campbell, et al. Representation Engineering: A Top-Down Approach to AI Transparency. 2023. arXiv:2310.01405. https://arxiv.org/abs/2310.01405
- N. Panickssery, N. Gabrieli, J. Schulz, M. Tong, E. Hubinger, A. M. Turner. Steering Llama 2 via Contrastive Activation Addition (CAA). ACL 2024. arXiv:2312.06681. https://arxiv.org/abs/2312.06681
- A. Arditi, O. Obeso, A. Syed, D. Paleka, N. Panickssery, W. Gurnee, N. Nanda. Refusal in Language Models Is Mediated by a Single Direction. NeurIPS 2024. arXiv:2406.11717. https://arxiv.org/abs/2406.11717
- K. Park, Y. J. Choe, V. Veitch. The Linear Representation Hypothesis and the Geometry of Large Language Models. 2023. arXiv:2311.03658. https://arxiv.org/abs/2311.03658
- B. W. Lee, I. Padhi, K. N. Ramamurthy, E. Miehling, P. Dognin, M. Nagireddy, A. Dhurandhar. Programming Refusal with Conditional Activation Steering (CAST). ICLR 2025. arXiv:2409.05907. https://arxiv.org/abs/2409.05907
- Z. Weng, J. Zhang, K. Cai, Y. Li, P. Wang, Y. Tian. FineSteer: Fine-Grained Inference-Time Steering. 2026. arXiv:2604.15488. https://arxiv.org/abs/2604.15488
- Q.-A. Dang, C. Ngo. Selective Steering: Norm-Preserving Control Through Discriminative Layer Selection. ACL 2026. arXiv:2601.19375. https://arxiv.org/abs/2601.19375
- In-Distribution Steering: Balancing Control and Coherence in Activation Interventions (IDS). 2025. arXiv:2510.13285. https://arxiv.org/abs/2510.13285
- B. Hsu, et al. Contextual Linear Activation Steering of Language Models. 2026. arXiv:2604.24693. https://arxiv.org/abs/2604.24693
- Z. Wu, A. Arora, Z. Wang, A. Geiger, D. Jurafsky, C. D. Manning, C. Potts. ReFT / LoReFT: Representation Finetuning for Language Models. NeurIPS 2024. arXiv:2404.03592. https://arxiv.org/abs/2404.03592
- Z. Wu, et al. Improved Representation Steering for Language Models (RePS/BiPO). NeurIPS 2025. arXiv:2505.20809. https://arxiv.org/abs/2505.20809
- HyperSteer: Hypernetwork-Generated Steering Vectors. 2025. arXiv:2506.03292. https://arxiv.org/abs/2506.03292
- G. Heyman, F. Vandeputte. Steer Like the LLM: Activation Steering that Mimics Prompting (PSR). ICML 2026. arXiv:2605.03907. https://arxiv.org/abs/2605.03907
- Z. Jin, R. Deng, J. Wang, X. Shen, C. Zhang. Flow-based Activation Steering for Inference-Time Intervention (FLAS). 2026. arXiv:2605.05892. https://arxiv.org/abs/2605.05892
- T. Lieberum, S. Rajamanoharan, A. Conmy, L. Smith, N. Sonnerat, V. Varma, J. Kramár, A. Dragan, R. Shah, N. Nanda. Gemma Scope: Open Sparse Autoencoders on Gemma 2. BlackboxNLP 2024. arXiv:2408.05147. https://arxiv.org/abs/2408.05147
- S. Chalnev, M. Siu, A. Conmy. Improving Steering Vectors by Targeting Sparse Autoencoder Features (SAE-TS). 2024. arXiv:2411.02193. https://arxiv.org/abs/2411.02193
- S. Soo, C. Guang, W. Teng, C. Balaganesh, T. Guoxian, Y. Ming. Feature Guided Activation Additions (FGAA). 2025. arXiv:2501.09929. https://arxiv.org/abs/2501.09929
- Z. Wu, A. Arora, A. Geiger, Z. Wang, J. Huang, D. Jurafsky, C. D. Manning, C. Potts. AxBench: Steering LLMs? Even Simple Baselines Outperform Sparse Autoencoders. ICML 2025. arXiv:2501.17148. https://arxiv.org/abs/2501.17148
- Angular Steering: Behavior Control via Rotation in Activation Space. 2025. arXiv:2510.26243. https://arxiv.org/abs/2510.26243
- A. Venkatesh, R. Kurapath. On the Non-Identifiability of Steering Vectors. ICLR 2026 (workshop). arXiv:2602.06801. https://arxiv.org/abs/2602.06801
- Curveball Steering: The Right Direction To Steer Isn't Always Linear. 2026. arXiv:2603.09313. https://arxiv.org/abs/2603.09313
- Briglia, Facchiano, Cursi, Sampieri, Rodolà, et al. Hyperbolic Concept Control (HyCon). 2026. arXiv:2603.14093. https://arxiv.org/abs/2603.14093
- Gao, Zhang, Liu, Chen, et al. The Cylindrical Representation Hypothesis (CRH). ICML 2026. arXiv:2605.01844. https://arxiv.org/abs/2605.01844
- Wurgaft, Rager, Kowal, Goodman, Fel, Geiger, Lubana. Manifold Steering. 2026. arXiv:2605.05115. https://arxiv.org/abs/2605.05115
- A. Mishra, D. Khashabi, A. Liu. Steered LLM Activations are Non-Surjective. 2026. arXiv:2604.09839. https://arxiv.org/abs/2604.09839
- Kazama, et al. GeoSteer: Faithful Chain-of-Thought Steering via Latent Manifold Gradients. 2026. arXiv:2601.10229. https://arxiv.org/abs/2601.10229
- A. Korznikov, A. Galichin, A. Dontsov, A. Rogov, I. Oseledets, E. Tutubalina. The Rogue Scalpel: Activation Steering Compromises LLM Safety. ICML 2025/2026. arXiv:2509.22067. https://arxiv.org/abs/2509.22067
- S. Cheng, S. Wiegreffe, D. Manocha. What Drives Representation Steering? A Mechanistic Case Study on Steering Refusal. 2026. arXiv:2604.08524. https://arxiv.org/abs/2604.08524
- Ali, et al. Scaling Laws for Activation Steering with Llama 2 Models and Refusal Mechanisms. 2025. arXiv:2507.11771. https://arxiv.org/abs/2507.11771
- J. Postmus, S. Abreu, A. Müller, et al. Steering LLMs using Conceptors. 2024. arXiv:2410.16314. https://arxiv.org/abs/2410.16314
- Y.-S. Chuang, et al. DoLa: Decoding by Contrasting Layers Improves Factuality. ICLR 2023. arXiv:2309.03883. https://arxiv.org/abs/2309.03883
- T. Bricken, A. Templeton, et al. Towards Monosemanticity: Decomposing Language Models With Dictionary Learning. Transformer Circuits, 2023. https://transformer-circuits.pub/2023/monosemantic-features/index.html

---

## Reviewer-loop changelog

This document was hardened through two rounds of adversarial elite-reviewer critique. The body above is the post-fix paper; this changelog is the audit trail.

**Round 1 — top issues raised and fixed.**
1. *Meta-narration in the body.* The prior draft narrated its own review ("an internal ICML-area-chair review found the behavior axis was circular…") and framed Section 4 as a confession. Fixed: Section 4 now reads as a specification of valid instruments, stating the projection identity Δproj=α‖v‖ directly and presenting the generation-based scorer as the designed instrument, not a correction extracted by a reviewer. All "an internal review found…" phrasing removed.
2. *Buried lede.* The geometry results (the genuine contribution) were scattered across §5.1, §5.5, and a campaign-findings appendix. Fixed: restructured Section 5 so the cliff → fragility → relative-parameterization → source-equivalence → cylindrical → real-data-evaluation arc is linear, and promoted the geometry to the title, abstract, and a dedicated Discussion (Section 6).
3. *Under-grounded background.* Added a Background subsection on the residual stream and the linear representation hypothesis (with Park et al. arXiv:2311.03658) so a non-specialist can follow the radial/angular argument, and expanded Related Work to situate the twelve-axis taxonomy against the additive, conditional, learned/sparse, geometry, and safety families with real arXiv IDs.
4. *Overclaiming risk on R².* Clarified that the R²=0.81 collapse law is within-pool and is *falsified* across scale (held-out R²=−1.6), and that only the monotone Spearman relationship survives held-out testing.

**Round 2 — top issues raised and fixed.**
1. *Residual confession tone in limitations.* Rewrote Section 7 ("Limitations as scope") so each item states the boundary of a claim rather than apologizing; removed self-deprecating phrasing.
2. *Rotation falsification could be read as universal.* Tightened Section 5.5 to scope the falsification explicitly to *full-vector* rotation and to flag selective-plane rotation as untested, matching the underlying caveat.
3. *Discussion was a summary, not an argument.* Rewrote Section 6 to make four load-bearing claims for SOTA steering (off-shell displacement as the control variable; cylindrical two-coordinate monitoring; source solved but site/geometry not; scale changes the regime), each tied to a specific result.
4. *Citation completeness.* Added the linear-representation, AxBench, Manifold-Steering, CRH, and Non-Identifiability references inline where their results are used, and consolidated a full reference list with arXiv links.

No clumsy meta-narration remains in the paper body; all self-referential review language is confined to this changelog.
