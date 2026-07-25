# LIT_2026-07 — Geometry & Mechanism of Activation Steering

**Scout pass date:** 2026-07-25
**Scope:** manifold / off-distribution view, SAE-vs-diffmean status, learned-vs-fixed
directions, scale dependence, multi-vector composition.
**Verification protocol:** every arXiv id below was checked by fetching its
`arxiv.org/abs/` page and confirming title + authors + submission date. Ids that
could only be confirmed via search-snippet are marked `[SNIPPET-ONLY]`. Nothing
here is cited from model memory.

**Headline for the lead:** the displacement/norm-budget leg is **substantially
closed** by prior work. See §C.0 — read that first.

---

## (a) Key papers

### Core geometry / off-manifold line

| Title | arXiv | Date | Contribution | Relevance to us |
|---|---|---|---|---|
| Selective Steering: Norm-Preserving Control Through Discriminative Layer Selection | 2601.19375 | 2026-01-27 | Norm-preserving **planar rotation** `R_θ^P = I − (b₁b₁ᵀ+b₂b₂ᵀ) + [b₁ b₂]R_θ[b₁ b₂]ᵀ` + layer selection where class means are **opposite-signed** (`μ̃_pos·μ̃_neg < 0`). Zero PPL-threshold violations, ~100% capability retention on 5 benchmarks. 8 models incl. Llama-3.2-1B/3B, Qwen2.5-1.5B/3B, Gemma-2-2B/9B | **The single most important paper for us.** Closes much of both our geodesic leg and our displacement-budget leg. See §C.0/§C.2 |
| Why Steering Works: Toward a Unified View of LM Parameter Dynamics | 2602.02343 | 2026-02-02 (rev 2026-04-12) | "Preference–utility" analysis on a shared log-odds scale; utility declines when interventions push representations **off the valid-generation manifold**; introduces a validity-decay factor | Our finding (a) already exists as a framework |
| Steered LLM Activations are Non-Surjective | 2604.09839 | 2026-04-10 (rev 2026-05-07) | Proves steering pushes the residual stream **off the manifold of prompt-reachable states**; no prompt reproduces a steered internal state | Off-manifold is structural, not an artifact of bad α |
| Minimizing Collateral Damage in Activation Steering | 2605.01167 | 2026-05-01 | Steering as constrained optimization minimizing expected squared collateral change; weights perturbations by the empirical **second-moment matrix** (anisotropic, not isotropic) | A scalar ‖Δh‖ budget is the wrong norm; use Σ-weighted |
| Manifold Steering Reveals the Shared Geometry of Neural Network Representation and Behavior | 2605.05115 | 2026-05-06 | Fits an **activation manifold M_h** to representations and a **behavior manifold M_y** to output distributions; linear steering "cuts through off-manifold regions" and produces unnatural output; manifold-following steering does not | **On-manifold projection rescue, already published** |
| Prompt–Activation Duality: Improving Activation Steering via Attention-Level Interventions (GCAD) | 2605.10664 | 2026-05-11 (rev 2026-05-14) | Identifies **KV-cache contamination** — steered token states are cached and reused, turning a local perturbation into *cumulative* coherence degradation. GCAD steers at the attention level w/ token gating. Coherence drift −18.6 → −1.9; turn-10 trait 78.0 → 93.1 | Mechanistic explanation for why our *cumulative* ‖Δh‖/‖h‖ predicts incoherence |
| A Geometric Account of Activation Steering through Angle–Norm Decomposition | 2606.06735 | 2026-06-04 (rev 2026-06-08) | Steering methods differ mainly in how they couple **angular** alignment to a concept direction vs change in **hidden-state norm**. Concepts live in angular structure; norm governs stability and downstream effects. Recommends parameterizing by angle + radius instead of one additive α | Splits our single displacement budget into two independently-controllable dials. **Basis of the proposed experiment (§d)** |
| Curveball Steering: The Right Direction To Steer Isn't Always Linear | 2603.09313 | 2026-03 | Nonlinear steering via polynomial-kernel PCA; beats linear PCA steering especially under strong geometric distortion | Nonlinear ≠ ambient geodesic; see §C.2 |
| Predicting Where Steering Vectors Succeed (LAP) | 2604.15557 | 2026-04-16 | **Linear Accessibility Profile**: `A_lin` (unembedding applied to intermediate hidden states) predicts steering success, ρ = +0.86 to +0.91. **Three-regime framework**: where diff-of-means works, where nonlinear is required, where nothing works. 5 models Pythia-2.8B→Llama-8B; validated on Gemma-2-2B and OLMo-2-1B-Instruct; 24 binary concept families | **Directly explains our finding (c)** |

### Composition / multi-vector

| Title | arXiv | Date | Contribution | Relevance |
|---|---|---|---|---|
| GEMS: Geometric Constraints Enable Multi-Semantic Superposition in LLMs | 2606.19946 | 2026-06-18 | Two named failure modes: **distributional deviation** ("additive perturbations accumulate in norm across layers and drive activations outside the training distribution") and **directional interference** (non-orthogonal vectors mutually dampen). Fixes: norm-preserving weighted superposition + attention-pathway injection + real-time orthogonalization. **3B–31B.** GSM8K 98% w/ 3 concurrent directions vs 92% baseline and 4% unconstrained; WikiText-2 PPL +2.2% | **Prior art on the norm budget as a coherence controller.** Also validates our §9 stacking discipline |
| ORBIT: Training-Free Multi-Attribute Behavioral Steering via Orthogonal Subspace Rotation | 2606.22357 | 2026-06-21 | Joint subspace from per-attribute steering planes via SVD; a single **norm-preserving rotation** within it; adaptive per-token gating. Llama-3.2-3B, Qwen-2.5-7B, Llama-3.1-8B | Our rotation arm generalized to N vectors; also our probe-gate as per-token gating |
| Activation Steering with a Feedback Controller (PID Steering) | 2510.04309 | rev v3 2026-05-16, ICLR 2026 poster | Casts existing steering as a **P** controller with the steering vector as feedback signal; adds **I** (persistent cross-layer correction) and **D** (overshoot damping) | Control-theoretic framing of any budget controller we build. Read before designing ours |
| UniSteer: Text-Guided Flow Matching in Activation Space for Versatile LLM Steering | 2605.30076 | 2026-05-28 | Learns a **universal conditional velocity field** over residual-stream activations from natural-language conditions; one model, many behaviors | Learned generative alternative to fixed directions; relevant to `flas/` lesson |
| Activation Steering for Aligned Open-ended Generation without Sacrificing Coherence | 2604.08169 | 2026-04-09 (rev **2026-07-02**) | StTP/StMP "projection-aware" methods steer **only tokens whose activations fall below a logistic-regression decision boundary**. Llama-3.3-70B, Qwen3.6-27B | **External validation of our probe-gated (CAST-style) design.** Newest revision in the corpus |

### Safety / red-team / scale

| Title | arXiv | Date | Contribution | Relevance |
|---|---|---|---|---|
| Activation Steering Induces Emergent Misalignment: A More Comprehensive Evaluation | 2606.08682 | 2026-06-07 | Systematic study of AS-induced emergent misalignment across model families, **scales**, target tasks and intervention layers; mid-to-late layers give the strongest/most stable EM induction | Rogue Scalpel axis at scale; supports our §10 guard-layer C (avoid fragile mid-layers) |
| Steering Vectors are an Adversarial Attack Surface | 2606.05958 | 2026-06-04 | Poisoning **4–6% of tokens** in a shared steering dataset yields 20–55% absolute ASR (+19–51% over clean). A **refusal-direction orthogonalization defense** recovers ~82% of the gap without degrading benign behavior | New red-team probe for Rung 4; the defense *is* our guard layer A |
| The Rogue Scalpel: Activation Steering Compromises LLM Safety | 2509.22067 | rev 2026-02 `[SNIPPET-ONLY]` | Random-direction steering raises harmful compliance 0% → 1–13% | Already in our corpus; id re-confirmed only by snippet here |

### SAE / learned-vs-fixed

| Title | arXiv | Date | Contribution | Relevance |
|---|---|---|---|---|
| AxBench: Steering LLMs? Even Simple Baselines Outperform Sparse Autoencoders | 2501.17148 | ICML 2025 | On Gemma-2-2B/9B, **SAEs are not competitive**; diff-in-means best for concept *detection*; LoReFT/LoRA top the *steering* table | Our baseline. Note what it actually says — see §C.4 |
| Steering LLMs? Actually, Sparse Autoencoders can outperform simple baselines | 2605.31183 | 2026-05-29 | SAEs reach **close to reference LoRA performance on AxBench** when features are selected and labelled by a supervised pipeline | The 2026 rebuttal — conditional, and the condition is supervision |
| ReFT: Representation Finetuning for Language Models | 2404.03592 | NeurIPS 2024 | LoReFT: low-rank subspace intervention, 15×–65× more parameter-efficient than LoRA | Our `reft_r1/` lesson basis |

---

## (b) THE 2026 GEOMETRIC PICTURE

The field has converged on a single causal story, and then spent Q2 2026 refining it.

**The story.** Steering degrades output because it displaces activations off the
manifold of states the model can validly generate from, and degradation rises
monotonically with that displacement. This is now asserted at three levels:
*empirically* (2602.02343 — utility declines as interventions push representations
off the valid-generation manifold), *geometrically* (2605.05115 — linear steering
cuts a chord through off-manifold regions while the data manifold curves away),
and *formally* (2604.09839 — the steered residual stream provably leaves the set of
prompt-reachable states; steering is non-surjective).

**Four refinements landed this quarter, and they matter more than the headline.**

1. **Displacement is not one number — it is angle plus norm** (2606.06735). Concepts
   are encoded primarily in *angular* structure; the *norm* component is what governs
   stability and downstream damage. A single additive coefficient α entangles the two.
   The recommendation is explicit: parameterize interventions by interpretable
   angular and radial components.
2. **Displacement is not isotropic** (2605.01167). Collateral damage depends on
   *which* direction you move in, weighted by the empirical second-moment matrix of
   activations. Penalizing all directions equally is the wrong objective.
3. **Displacement is not per-token — it compounds** (2605.10664, 2606.19946). Two
   independent accumulation channels: across *turns* via KV-cache contamination
   (steered states get cached and reused), and across *layers* via additive norm
   growth. Both convert a bounded local perturbation into unbounded global drift.
4. **The layer matters as much as the magnitude** (2601.19375, 2606.08682). Steering
   is only well-posed at layers where the target feature is linearly present with
   *opposite-signed class alignment*; outside that band you spend budget for no
   behavioral return.

**How the field now prevents degradation — five moves, all published:**

- **Constrain the geometry**: norm-preserving planar rotation instead of addition
  (2601.19375, ORBIT 2606.22357, GEMS 2606.19946).
- **Project onto a fitted manifold**: steer along M_h rather than through it
  (2605.05115).
- **Gate the tokens**: intervene only where a learned boundary says it is needed
  (StTP/StMP 2604.08169, GCAD 2605.10664, ORBIT per-token gating).
- **Select the layer**: discriminative selection by opposite-signed class alignment
  (2601.19375).
- **Close the loop**: control-theoretic P/I/D modulation of steering strength
  (2510.04309).

The synthesis: **coherence is bought by constraining *where* and *when* you steer,
not by shrinking α.** Global magnitude reduction is now regarded as the crude
instrument; selectivity and geometry-respecting parameterization are the sharp ones.

---

## (c) WHAT THIS CHANGES FOR US

### C.0 — CRITICAL: has the displacement/norm-budget-as-coherence-controller been published?

**Yes — largely. This leg is mostly closed.** Answering the lead's question directly,
in order of how much each closes it:

| Our intended contribution | Prior art | Status |
|---|---|---|
| Off-manifold displacement → coherence loss | 2602.02343 (validity decay), 2604.09839 (formal), 2605.05115 (geometric) | **Closed.** Do not claim the observation |
| Norm budget as a *coherence controller* | **2601.19375** — norm-preserving rotation, **zero PPL-threshold violations** across all models and angles, ~100% capability retention. GEMS 2606.19946 — norm-preserving superposition against cross-layer norm accumulation, WikiText-2 PPL +2.2% | **Closed for the "hold the norm fixed" version.** Both ship it and report the coherence win |
| On-manifold projection rescue | 2605.05115 fits M_h and steers along it; PCA/autoencoder projection is the stated mechanism | **Closed** |
| Closed-loop strength modulation | 2510.04309 PID Steering (ICLR 2026) | **Closed** |
| Anisotropic / Σ-weighted budget | 2605.01167 | **Closed** |

**2601.19375 in particular is the paper the lead flagged, and it does close that
leg — but not entirely.** Verified: *Selective Steering: Norm-Preserving Control
Through Discriminative Layer Selection*, Quy-Anh Dang & Chris Ngo, 2026-01-27.
Eight models across Llama/Qwen/Gemma including **four in our 1B–4B band**
(Llama-3.2-1B, Qwen2.5-1.5B, Qwen2.5-3B, Gemma-2-2B). Code at
`github.com/knoveleng/steering`.

**Three gaps survive, and our contribution must live in them:**

1. **No matched-budget comparison.** Selective Steering compares three *rotation*
   methods (SAS, AAS, Selective) to each other, and dismisses additive methods
   (ActAdd) as suffering "catastrophic degradation on smaller models" — **without a
   controlled equivalence test**. It never holds displacement fixed and asks whether
   the rotation path is better *per unit of displacement spent*. Our finding (b) is
   exactly that experiment. This is a real, defensible hole.
2. **Nobody uses displacement as a calibrated online *predictor*.** Every published
   method *constrains* geometry a priori (rotate, project, clamp). None measures
   per-token displacement at inference and modulates against a coherence setpoint
   with a *calibrated* effect size. PID Steering is closed-loop but its feedback
   signal is the steering vector itself, not a coherence proxy.
3. **The angle/norm decomposition (2606.06735) is asserted, not measured as a
   controller, and not reported at ≤4B.** This is the opening we should take (§d).

**Action:** stop describing a displacement budget as novel. Reframe the program
around *matched-budget attribution* — which component of displacement carries the
coherence tax — and cite 2601.19375, 2606.19946, 2605.05115 and 2510.04309 as the
methods we are decomposing rather than the ideas we are inventing.

### C.1 — Off-manifold displacement ↔ incoherence, Spearman ρ = +0.585 (WikiText-2, two model sizes)

**CORROBORATED, and pre-empted at the level of the claim.**

The monotone relationship is the stated premise of 2602.02343 (utility declines as
interventions push representations off the valid-generation manifold) and the formal
result of 2604.09839. GEMS 2606.19946 names our exact mechanism — "additive
perturbations accumulate in norm across layers and drive activations outside the
training distribution" — and reports the coherence consequence on the same corpus we
used (WikiText-2). 2605.10664 supplies the second accumulation channel (KV cache).

Two cautions, both material:

- **ρ = +0.585 is weak relative to the competition.** LAP's `A_lin` predicts steering
  outcomes at ρ = +0.86 to +0.91 (2604.15557). If we are marketing displacement as a
  *predictor*, we are currently offering an inferior one. We should measure `A_lin` on
  our setup as a baseline predictor before building a controller on ρ = 0.585.
- **Scalar ‖Δh‖ is the wrong statistic** per 2605.01167 (anisotropy) and 2606.06735
  (angle/norm entanglement). A Σ-weighted or radial-only displacement may well be the
  statistic that lifts ρ toward the LAP range. That is a cheap, high-value re-analysis
  of data we already have — no new GPU time.

### C.2 — Norm-preserving geodesic / rotated path NOT more coherent than the straight chord at matched budget (curved was worse)

**NOVEL — this is our strongest result, and it is a genuine negative that the
literature has not tested.** But it must be framed precisely or it will be wrong.

The apparent contradiction: Selective Steering (2601.19375) reports zero perplexity
violations from norm-preserving rotation; ORBIT (2606.22357) and GEMS (2606.19946)
both adopt norm-preserving rotation specifically for stability; Curveball
(2603.09313) and Manifold Steering (2605.05115) both report curved/nonlinear paths
beating linear ones.

**Why our result is compatible with all of them, and therefore interesting:**

1. **None of them controls the budget.** 2601.19375 explicitly does *not* compare
   rotation vs addition at matched displacement (verified from the paper body). Its
   rotation win is confounded with a smaller effective displacement and with its
   layer-selection criterion. We ran the controlled version.
2. **Their curve is fitted to data; ours was not.** 2605.05115 steers along an
   *activation manifold fitted to representations*. 2601.19375 rotates within a **2D
   plane spanned by two data-derived basis vectors** (b₁, b₂), leaving the orthogonal
   complement untouched. Curveball works in a **kernel-PCA feature space**. A
   norm-preserving geodesic on the ambient hypersphere is *none of these* — it is
   curvature without data.

**The claim we can defend:** *norm preservation alone does not buy on-manifold-ness;
a path must be curved toward the data manifold, not merely constrained to constant
radius.* That is a clean, falsifiable, previously-unstated result and it explains why
the field's rotation methods work (they rotate in a fitted subspace) without our
geodesic working (it rotated in an arbitrary one).

**Required before we ship it:** rerun the curved arm with (i) the 2601.19375 planar
rotation confined to the (b₁,b₂) span, and (ii) a PCA-fitted manifold à la 2605.05115.
If the null survives *those*, it is a strong publishable negative. If it does not, we
have found the boundary condition — which is equally publishable and more useful.
Publishing the ambient-geodesic null alone, without the fitted-subspace comparison,
would be a straw man and a reviewer would say so.

### C.3 — Diff-of-means WRITE fails at 1B AND 4B while probe READ hits AUC 0.99

**CORROBORATED and mechanistically explained. Do not claim discovery.**

This is precisely LAP's three-regime framework (2604.15557): read-linearity and
write-efficacy dissociate, and there is a middle regime where a concept is linearly
*decodable* but diff-of-means steering fails and nonlinear methods are required. LAP
validates on Gemma-2-2B and OLMo-2-1B-Instruct — adjacent to but not identical with
our Gemma-3-1B/4B setup.

2601.19375 supplies a second, sharper candidate explanation we should test
immediately because it is nearly free: their **discriminative layer selection**
criterion, `μ̃_pos · μ̃_neg < 0`. Layers split into three regimes — early (feature
absent), middle (robust opposite-signed separation), late (weakening separation).
A probe can succeed at a layer where the class means are *same-signed but
different-magnitude*; diff-of-means steering at that layer will be weak or
incoherent. **If our injection layer fails the opposite-sign test, our WRITE
negative is a layer-selection artifact, not a property of diff-of-means.** This is a
one-hook diagnostic on cached activations, no generation required, and it is a
BLOCKER on the current framing of finding (c).

**What remains genuinely ours:** the result holding on Gemma-3 at both 1B and 4B
(cross-scale, our commit `0dab3bc`), which is a scale point neither paper covers,
*conditional on* passing the layer-selection check above.

### C.4 — Learned ReFT rank-1 beats diff-of-means for behavior

**CORROBORATED, and we have been mis-citing AxBench.**

AxBench (2501.17148) finds that *simple baselines outperform **sparse autoencoders***
— it does **not** find that simple baselines beat *learned* interventions. LoReFT and
LoRA sit at the top of its steering table; diff-in-means wins at concept *detection*,
which is the READ task, not the WRITE task. Our ReFT-r1 > diff-of-means result is
therefore squarely consistent with AxBench, not a challenge to it.

**Correct the framing everywhere it appears in `steering_tutorials/reft_r1/` and the
theme tables.** "AxBench: simple baselines are strong" is accurate only against SAEs.
Note also the 2026 rebuttal (2605.31183): SAEs reach near-LoRA AxBench performance
*given a supervised feature-selection pipeline* — so even the SAE verdict is now
conditional on supervision rather than settled.

Read together with C.3 this is one coherent story: **at 1B–4B, the concept is
linearly readable but not linearly writable, so a learned intervention beats a fixed
direction.** That is a defensible cross-lesson thesis and it is exactly LAP's
middle regime, instantiated at a scale LAP did not test.

### Scale (context for all of the above)

Evidence is real but thinner than the geometry line, and much of it is
`[SNIPPET-ONLY]` — treat as `[NEEDS VERIFICATION]` until fetched:

- Steerable-concept count grows with model size and **plateaus around 2.8B–6.9B**
  (Pythia family) `[SNIPPET-ONLY, attributed to 2507.11771]`.
- Larger models shift *less* per unit α with the same vector — steering effect size
  decreases with scale `[SNIPPET-ONLY]`.
- **Validity/coherence degrades more under strong steering on smaller models**
  `[SNIPPET-ONLY]`.
- 2606.08682 (verified) confirms AS-induced misalignment varies substantially with
  scale and layer, mid-to-late layers strongest.

**Read for our program:** 1B–4B is a **high-efficacy, low-coherence-margin** regime.
Large effect sizes are available cheaply; the coherence budget is what binds. That is
a favorable setting for a budget/controller result and an unfavorable one for naive
α-scaling — it is the strongest available justification for why this work belongs at
our scale rather than being a compute-constrained compromise.

---

## (d) The single most promising mechanism-level experiment at 1B–4B on a 16 GB laptop

### Decompose the coherence tax: does it live in the ANGLE or the NORM?

**The open question.** 2606.06735 (June 2026) asserts that concepts are carried by
*angular* structure while the *norm* governs stability and downstream damage, and
recommends parameterizing steering by separate angular and radial components. It
asserts this; it does **not** build the controller, does **not** test it at ≤4B, and
does **not** hold displacement constant while varying the split. Meanwhile
2601.19375, ORBIT and GEMS all ship norm-preserving methods *without* ever
demonstrating that the norm is the component that actually costs coherence — they
assume it. Nobody has run the matched-budget decomposition.

**Design.** Fixed injection layer (chosen by the 2601.19375 opposite-sign criterion),
Gemma-3-1B and Gemma-3-4B. Independently vary:

- **θ** — angular rotation of `h` toward the concept direction, within the 2D plane
  spanned by the data-derived basis (b₁, b₂), per the verified 2601.19375 formula
  `R_θ^P = I − (b₁b₁ᵀ + b₂b₂ᵀ) + [b₁ b₂] R_θ [b₁ b₂]ᵀ`.
- **r** — radial scaling of ‖h‖.

Sweep a grid whose cells are grouped onto **iso-‖Δh‖ contours**, so every point on a
contour spends the identical displacement budget with a different angle/norm split.
Measure behavior efficacy and WikiText-2 PPL at every cell. ~30 cells × 2 models,
greedy decoding, cached contrast activations — comfortably a single foreground
session inside the 16 GB / RAM-pressure constraints.

**Pre-registered falsifier.** *Iso-displacement contours are iso-coherence.* If PPL
is flat along a contour, displacement magnitude is the whole story, our current
budget framing is correct, and 2606.06735's decomposition adds nothing at this scale.

**Predicted result.** Coherence tracks Δr almost alone; efficacy tracks θ almost
alone. Contours are strongly non-iso-coherent.

**Why this is the right experiment.**

- It is the **only one of our four findings' follow-ups that is not already closed**
  by C.0. Every other candidate (build a budget controller, project on-manifold,
  close the loop) has 2026 prior art shipping the same idea.
- It **converts our weakest asset into our strongest**. If the tax is radial, then
  ρ = 0.585 for scalar displacement is low *because scalar displacement is the wrong
  statistic* — and Δr alone should predict incoherence far better. That is a
  re-analysis of existing data plus a confirmation sweep.
- It **resolves finding (b) mechanistically rather than leaving it a bare null.** A
  norm-preserving geodesic spends its entire budget on θ. If θ were free, that path
  should have been *more* coherent — it was not. Either the coherence kernel has an
  angular term too, or our ambient geodesic left the (b₁,b₂) plane and incurred
  off-plane displacement the scalar budget did not price. The grid distinguishes
  these two explanations; nothing else we can run does.
- It **ships an actionable artifact**: if confirmed, the deliverable is a *radial
  clamp with free angular travel* — one forward hook, no training, no learned
  manifold — plus the first measurement at 1B–4B of a decomposition the field
  currently takes on faith.

**Sequencing note for the lead — two cheap prerequisites, both BLOCKERs:**

1. Run the `μ̃_pos · μ̃_neg < 0` layer diagnostic (2601.19375) on our cached
   activations. If our injection layer fails it, finding (c) is a layer-selection
   artifact and must be re-framed before anything is claimed. Cost: minutes, no
   generation.
2. Re-score our existing displacement↔incoherence data with radial-only and
   Σ-weighted displacement (2605.01167). If ρ jumps toward LAP's 0.86–0.91, the
   grid experiment's prediction is half-confirmed before we spend GPU time.

---

## Verification ledger

**Verified by `arxiv.org/abs/` fetch (title + authors + date confirmed):**
2601.19375, 2602.02343, 2604.08169, 2604.09839, 2604.15557, 2605.01167, 2605.05115,
2605.10664, 2605.30076, 2605.31183, 2606.05958, 2606.06735, 2606.08682, 2606.19946,
2606.22357, 2510.04309.

**Additionally verified by full-text HTML fetch:** 2601.19375 (models, rotation
formula, layer-selection criterion, and the absence of a matched-budget comparison).

**`[SNIPPET-ONLY]` — title/authors from search results, abs page not fetched. Treat as
`[NEEDS VERIFICATION]`:** 2603.09313 (Curveball), 2507.11771 (steering scaling laws),
2510.10205 (PIXEL), 2606.11489 (CLAE), 2602.06801 (non-identifiability),
2509.22067 (Rogue Scalpel — already in corpus from prior passes).

**Known-good from prior corpus work, not re-fetched this pass:** 2501.17148 (AxBench),
2404.03592 (ReFT).

**Negative result on recency:** no 2607.* paper in the steering-geometry line was
found. July 2026 is effectively empty so far; the newest relevant item is the
**2026-07-02 revision of 2604.08169**. The frontier for this topic is June 2026
(2606.06735, 2606.19946, 2606.22357, 2606.08682, 2606.05958) — five papers in four
weeks, all geometric. The field is moving fast in exactly our direction, which raises
the cost of delay on the §d experiment.
