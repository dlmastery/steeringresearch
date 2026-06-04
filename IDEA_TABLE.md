# IDEA_TABLE — Hypothesis Registry

> **Canonical source:** `corpus/50-steering-experiments-autoresearch.md` (E1–E50, N1–N12)
> and `corpus/steering-missed-dimensions-and-highdim-algebra.md` (N13–N20).
> All quantitative thresholds are pre-registered predictions inherited from the
> corpus and marked **[NEEDS VERIFICATION]** until reproduced on the 4090 ladder.
> Do NOT edit a row's hypothesis or metric after the corresponding idea dir is
> created — that constitutes post-hoc goal-post shifting (HARKing).

**Primary axis codes (12-axis framework):**
A1=WHERE(site) | A2=WHAT(direction) | A3=HOW-MUCH(coeff) | A4=HOW(operation) |
A5=WHEN(condition) | A6=WHICH-TOKENS(span) | A7=HOW-DERIVED(source) |
A8=GEOMETRY(path) | A9=METRIC(space) | A10=IDENTIFIABILITY(gauge) |
A11=DYNAMICS(trajectory) | A12=BASIS/SUPERPOSITION

---

## BLOCK A — Foundations and measurement tooling (E1–E8)

| ID | Short title | Block | Primary axis | Falsifiable hypothesis (1 line) | Primary metric + success threshold [NEEDS VERIFICATION] | Status | Rung reached | Idea dir |
|----|-------------|-------|-------------|--------------------------------|--------------------------------------------------------|--------|-------------|----------|
| E1 | DiffMean pair-count knee | A | A7 (how derived) | DiffMean steering vectors from >=50 contrast pairs reach >=90% of asymptotic behavior-shift effect; more pairs give diminishing returns. | Behavior success vs pair count; knee at >=50 pairs reaching >=90% asymptote [NEEDS VERIFICATION] | DIRECTIONAL(scr,underpowered): knee~5 pairs cos>0.95 on easy synthetic concept; can't test n=50 (C5) | 2 | `ideas/10_diffmean_paircount_knee/` |
| E2 | Fisher layer selection | A | A1 (site/layer) | The optimal injection layer is the layer of maximum linear separability of the contrast set (Fisher ratio), not a fixed late layer. | Spearman correlation between Fisher ratio and measured efficacy >= 0.7 [NEEDS VERIFICATION] | FALSIFIED(scr): rho=+0.14 p=0.74 (C1, Gemma-270m); on REAL AxBench (2B, 20 concepts, off-family judge AUC 0.68) the layer curve is NEARLY FLAT — behavior 0.163-0.184 across L6-22, shallow peak L18 (coherence best L20), weak mid-late preference, no sharp optimum (exp#121) | 2 | `ideas/20_fisher_layer_selection/` |
| E3 | Alpha coherence cliff | A | A3 (coeff) | Coefficient alpha has a behavior-specific coherence cliff: below it MMLU drop <2pt, above it perplexity rises super-linearly. | Alpha inflection point identified; super-linearity verified; MMLU drop <2pt below cliff [NEEDS VERIFICATION] | SUPPORTED(scr): super-linear cliff + cross-scale window-emergence (E3/C6); CONFIRMED on REAL AxBench (concept500 sample, 30 concepts, 2B/layer 20, off-family judge AUC 0.68): clear coherence cliff, behavior+coherence peak at alpha~0.10 (beh 0.163, coh 0.677), roughly flat to alpha~0.20 (coh 0.665), then super-linear collapse — coh 0.68→0.46→0.11 at alpha 0.20→0.40→0.80, knee ~0.10-0.20 (exp#120) — the cliff generalizes to real benchmark (unlike E7's directional claim, S-16 vs S-17) | 3 | `ideas/30_alpha_coherence_cliff/` |
| E4 | DiffMean vs PCA cosine alignment | A | A7 (how derived) | DiffMean and PCA-top-1 condition vectors are >0.95 cosine-aligned, so cheaper DiffMean suffices for CAST gating. | Cosine(DiffMean, PCA-top-1) >= 0.95 across layers, per behavior [NEEDS VERIFICATION] | SUPPORTED(scr): cos(dm,pca)=0.994-0.999 on 3 models x 4 behaviors (C6/C11); on REAL AxBench (2B, layer 20, 100 concepts) DiffMean vs PCA-top1 only MODERATELY aligned — mean |cos| 0.65, median 0.73, only 16% >=0.90, p5 0.19 (exp#122) — the synthetic cos~0.99 did NOT generalize (paired-data artifact: clean true pairs inflate cos; AxBench unpaired differences diverge) | 2 | — |
| E5 | 4-bit vs fp16 invariance | A | A7 (how derived) | Steering efficacy is invariant (within noise) to 4-bit vs fp16, so 4090-scale results transfer to full precision. | Behavior delta between quantizations <3% [NEEDS VERIFICATION] | PENDING | — | — |
| E6 | Over-steering linear probe | A | A5 (condition) | A linear probe on layer-L activations predicts pre-generation whether an input will be over-steered (incoherent), AUC>=0.8. | AUC of probe predicting over-steering >= 0.80 [NEEDS VERIFICATION] | PENDING | — | — |
| E7 | Norm-relative alpha | A | A3 (coeff) | Normalizing alpha by ||h|| (relative steering) reduces cross-prompt variance vs absolute alpha. | Behavior-shift variance lower for relative vs absolute alpha [NEEDS VERIFICATION] | WEAK / SCALE-DEPENDENT on REAL AxBench (concept500, n=500, off-family judge AUC 0.68): 270M real<shuf NEGATIVE/floor (real 0.046 vs shuf 0.056, delta −0.010, CI [−0.0142,−0.0066], p=1.4e-6; exp#118); 2B real>shuf but TINY +0.004 p=0.011 CI[+0.0003,+0.0077], ordinal-fail (both ~0.135, shuf captures ~97%; exp#119) — weak, scale-dependent, mostly-generic; synthetic-ocean +0.135 did NOT generalize. Prior SYNTHETIC "ocean" run was PROVISIONAL directional win (real>shuffled both scales Holm-corrected, 270M +0.135 p=4e-4; 1B +0.096 p=.014; exp#116/117, S-15); sign-blind 270M auto-label "DIRECTIONAL" corrected to NEGATIVE; screening: relative_add clean cliff, knee~10% ||h|| (C9b) | 3 | — |
| E8 | IT->base transfer | A | A7 (how derived) | Behavior vectors from instruction-tuned Gemma transfer to base model with <20% efficacy loss (direction is pretraining-induced). | Cross-model efficacy loss <20% IT->base; also test base->IT [NEEDS VERIFICATION] | PENDING | — | — |

---

## BLOCK B — Conditional / gated steering, CAST family (E9–E16)

| ID | Short title | Block | Primary axis | Falsifiable hypothesis (1 line) | Primary metric + success threshold [NEEDS VERIFICATION] | Status | Rung reached | Idea dir |
|----|-------------|-------|-------------|--------------------------------|--------------------------------------------------------|--------|-------------|----------|
| E9 | CAST harmless refusal gate | B | A5 (condition) | CAST thresholding keeps harmless-input refusal <3% while raising harmful-input refusal by >=50pt vs unconditional steering. | Harmless refusal <3%; harmful refusal delta >=50pt vs unconditional [NEEDS VERIFICATION] | PENDING | — | — |
| E10 | Category condition orthogonality | B | A2 (direction) | Condition vectors for distinct safety categories (hate, self-harm, illegal) are near-orthogonal (\|cos\|<0.3), enabling independent OR-gating. | Pairwise cosine matrix \|cos\|<0.3 across categories [NEEDS VERIFICATION] | PARTIAL(scr): concepts mostly near-orthogonal except anger-happiness +0.48 | 2 | — |
| E11 | OR-gate coverage vs N | B | A5 (condition) | OR-composition of N condition vectors scales coverage linearly while harmless-refusal stays flat up to N~5, then leaks. | Coverage linear; harmless-refusal flat up to N~5 then leaks [NEEDS VERIFICATION] | PENDING | — | — |
| E12 | Energy-ratio vs cosine gate PR-AUC | B | A5 (condition) | Energy-ratio gating (FineSteer-SCS) gives sharper should-steer precision/recall than cosine-threshold (CAST) at equal compute. | Gate PR-AUC higher for SCS vs cosine threshold [NEEDS VERIFICATION] | PENDING | — | — |
| E13 | Early condition layer latency | B | A1 (site) | The condition check can move to an earlier layer than behavior injection with no gating-accuracy loss, cutting latency. | Gate AUC unchanged at earlier layer; latency reduced [NEEDS VERIFICATION] | PENDING | — | — |
| E14 | Discriminative-layer steering | B | A1 (site) | Discriminative-layer selection (steer only opposite-signed-mean layers) preserves capability better than all-layer steering. | MMLU retention at fixed behavior better than all-layer [NEEDS VERIFICATION] | PENDING | — | — |
| E15 | Learned gate vs fixed threshold | B | A5 (condition) | A learned logistic gate on multi-layer activations beats a fixed cosine threshold under distribution shift. | Gate AUC on OOD prompts higher for learned vs fixed-theta [NEEDS VERIFICATION] | FALSIFIED_OOD(scr): learned gate overfits tiny data, OOD AUC gap -0.17 (logistic 0.55 vs cosine 0.72), both perfect in-dist (exp#110) | 1 | — |
| E16 | Conditional gate capability tax | B | A5 (condition) | Conditional gating cuts the capability tax of always-on safety steering by >=80% (capability spent only when gate fires). | MMLU on benign set gated vs always-on; tax reduction >=80% [NEEDS VERIFICATION] | PENDING | — | — |

---

## BLOCK C — Stacking and multi-vector composition (E17–E26)

| ID | Short title | Block | Primary axis | Falsifiable hypothesis (1 line) | Primary metric + success threshold [NEEDS VERIFICATION] | Status | Rung reached | Idea dir |
|----|-------------|-------|-------------|--------------------------------|--------------------------------------------------------|--------|-------------|----------|
| E17 | Near-orthogonal stacking | C | A12 (superposition) | Two near-orthogonal behavior vectors (\|cos\|<0.2) added together keep >=90% of solo effects, no cross-degradation. | Joint vs solo success rate; joint >=90% of solo for each behavior [NEEDS VERIFICATION] | SUPPORTED(scr): stack anger+happiness retains 101%/110% (E10_E17) | 2 | — |
| E18 | Interference vs Gram mass | C | A12 (superposition) | Interference grows monotonically with summed off-diagonal Gram mass of the stacked vectors. | Fit interference vs Gram off-diagonal is monotone across 2-5 vectors [NEEDS VERIFICATION] | PARTIAL(scr): stacking retains 85-94% but interference non-monotone in Gram | 2 | — |
| E19 | Gram-Schmidt orthogonalization | C | A2 (direction) | Gram-Schmidt orthogonalizing a new vector against the active set preserves the new behavior while removing interference with existing ones. | Old-behavior retention higher for orthogonalized vs raw add [NEEDS VERIFICATION] | PENDING | — | — |
| E20 | SAE-TS vs DiffMean 3-stack coherence | C | A7+A12 | SAE-TS targeting makes stacked vectors more orthogonal in feature space than raw activation space, improving multi-behavior coherence. | 3-stack coherence SAE-TS vs DiffMean; SAE-TS higher [NEEDS VERIFICATION] | FALSIFIED(scr): SAE-TS LESS orthogonal, Gram 3.00 vs DiffMean 2.13 (vectors collapsed); tiny-real-SAE coverage confound (exp#112/113) | 1 | — |
| E21 | Conceptor AND vs sum | C | A4 (operation) | Conceptor AND-composition beats summed vectors for >=3 simultaneous goals at fixed coherence. | Multi-goal success at matched PPL; Conceptor AND > mean-sum [NEEDS VERIFICATION] | PENDING | — | — |
| E22 | Norm budget and collapse | C | A3+A12 | Total steering budget is how far sum(alpha_i v_i) pushes h outside the in-distribution shell; capping ||delta h|| at the empirical activation-norm quantile prevents collapse. | PPL vs cumulative ||delta h||; identify quantile; capped better than uncapped [NEEDS VERIFICATION] | SUPPORTED(scr): cumulative-displacement cliff PPL 138->4518 (E18_E22) | 2 | — |
| E23 | Same-plane vs orthogonal-plane composition | C | A4+A8 | Additive + rotational edits on the same plane interfere destructively, but on orthogonal planes they stack; site/operation governs composability. | Success same-plane vs orthogonal-plane; orthogonal superior [NEEDS VERIFICATION] | PENDING | — | — |
| E24 | Residual + KV-cache composition | C | A1 (site) | Residual + KV-cache steering (disjoint sites) compose with >=85% of each effect on short generations but degrade on long ones from KV contamination. | Effect vs generation length; >=85% short, degrades long [NEEDS VERIFICATION] | PENDING | — | — |
| E25 | DoLa stacking on residual steer | C | A4 (operation) | DoLa (decoding-time) stacks on any residual steer for additive factuality gain at no coherence cost. | TruthfulQA+PPL steer+DoLa vs DoLa alone; additive gain, no PPL cost [NEEDS VERIFICATION] | PENDING | — | — |
| E26 | Gate-before-behavior injection order | C | A6 (span) | Injecting the gating-relevant vector before the behavior vector (in layer order) improves selectivity vs reverse. | Selectivity by injection order; gate-first better [NEEDS VERIFICATION] | PENDING | — | — |

---

## BLOCK D — Geometry and rotational methods (E27–E33)

| ID | Short title | Block | Primary axis | Falsifiable hypothesis (1 line) | Primary metric + success threshold [NEEDS VERIFICATION] | Status | Rung reached | Idea dir |
|----|-------------|-------|-------------|--------------------------------|--------------------------------------------------------|--------|-------------|----------|
| E27 | Rotation beats addition on small models | D | A4+A8+A9 | Norm-preserving rotation beats additive steering specifically on small (<3B) models where additive edits more easily exit the manifold. | Coherence gap additive vs rotational larger for 1B than 9B at matched behavior [NEEDS VERIFICATION] | FALSIFIED(scr): add gentler than full-rotation @L16 (+42% PPL); caveat=full-vector!=selective (C3b); NOT SUPPORTED on REAL AxBench (2B, L20, 20 concepts × 8 eval, relative_add vs relative_rotate, alpha=0.10, exp#123): rotate-vs-add delta = −0.003 behavior / 0.000 coherence (diffmean: add 0.169/0.681 vs rotate 0.166/0.681) — rotation gives no benefit, operations roughly equal; difference tiny, not dramatic (S-20) | 3 | `ideas/_campaigns/` |
| E28 | Behavior plane low-rank | D | A2 (direction) | The behavior plane is low-rank: 2-3 dims capture >90% of steerable variance for a trait. | Variance explained by top-k of contrastive-difference space; >90% at k=2-3 [NEEDS VERIFICATION] | FALSIFIED(scr): top-3 var=66% not >90% (not low-rank) | 2 | — |
| E29 | Geodesic vs linear alpha monotonicity | D | A8 (geometry) | Geodesic (spherical) interpolation gives more monotone behavior control than linear alpha-scaling. | Monotonicity of behavior vs control parameter; geodesic more monotone [NEEDS VERIFICATION] | PENDING | — | — |
| E30 | Adaptive rotation on partial-aligned tokens | D | A6+A4 | Adaptive rotation (rotate only partially-aligned activations) preserves coherence better than rotating all tokens. | PPL at matched behavior; adaptive rotation better [NEEDS VERIFICATION] | PENDING | — | — |
| E31 | Rotation preserves activation norm | D | A4+A9 | Rotational steering leaves activation norm (hence LayerNorm stats) unchanged, explaining capability preservation. | Post-LN variance additive vs rotational; rotational closer to zero-edit [NEEDS VERIFICATION] | PENDING | — | — |
| E32 | Refusal vs detection direction separability | D | A2 (direction) | The refusal direction and the harmfulness-detection condition direction are distinct (low cosine): detection and execution are separable subspaces. | Cosine(condition, behavior) is low across layers [NEEDS VERIFICATION] | PENDING | — | — |
| E33 | Flow vs linear on convex vs non-convex | D | A4+A8 | Curved flow-based transport (FLAS-style) beats single-step addition mainly when the steered manifold is non-convex; on convex traits the gain vanishes. | Flow vs linear gain stratified by convexity proxy; gain only for non-convex [NEEDS VERIFICATION] | PENDING | — | — |

---

## BLOCK E — Mechanistic and interpretability-guided (E34–E40)

| ID | Short title | Block | Primary axis | Falsifiable hypothesis (1 line) | Primary metric + success threshold [NEEDS VERIFICATION] | Status | Rung reached | Idea dir |
|----|-------------|-------|-------------|--------------------------------|--------------------------------------------------------|--------|-------------|----------|
| E34 | Refusal via OV circuit | E | A1 (site) | Refusal steering acts mainly through the OV (attention-output) circuit; freezing attention scores costs <10% efficacy. | Efficacy frozen vs live attention; frozen cost <10% [NEEDS VERIFICATION] | PENDING | — | — |
| E35 | Sparse behavior vector | E | A2+A12 | A behavior vector can be sparsified to <10% of dims (top-magnitude coords) with <15% efficacy loss; behaviors live in sparse coordinate sets. | Efficacy vs sparsity; <15% loss at <10% dims [NEEDS VERIFICATION] | PARTIAL(scr): top-10% retains 77% (below 85%) | 2 | — |
| E36 | SAE selection problem | E | A7 (how derived) | SAE-feature steering underperforms raw DiffMean unless features are output-score-selected; the AxBench tension is a selection problem, not a representation problem. | SAE-steer naive vs selected vs DiffMean; selected SAE matches DiffMean [NEEDS VERIFICATION] | SUPPORTED(scr): diffmean~=pca at matched fractional alpha (C9b); WEAK/WASH on REAL AxBench (2B, L20, 20 concepts × 8 eval, alpha=0.10, exp#124): pca+add behavior 0.178 vs diffmean+add 0.169 (+0.009 margin) but pca coherence 0.644 vs diffmean coherence 0.681 (−0.037) — a tradeoff not a win; source barely matters; PCA marginally higher behavior, DiffMean marginally higher coherence; despite divergent directions on real concepts (E4, S-19), steering outcomes similar (S-21) | 3 | — |
| E37 | Interpretable != controllable | E | A7+A10 | Causally-steerable features are a small subset of interpretable SAE features; interpretability does not imply controllability. | Overlap of high-interpretability and high-causal-effect features is small [NEEDS VERIFICATION] | PENDING | — | — |
| E38 | Task + style in one edit | E | A2 (direction) | Function/task vectors (ICL-derived) and DiffMean behavior vectors share subspace; composing them transfers task + style in one edit. | Joint task+style success composing both; better than either alone [NEEDS VERIFICATION] | PENDING | — | — |
| E39 | Persona-vector drift monitoring | E | A5+A11 | Persona-vector monitoring predicts behavioral drift before it shows in outputs, enabling pre-emptive gating. | Lead-time between persona-projection shift and output shift; positive lead [NEEDS VERIFICATION] | PENDING | — | — |
| E40 | Procrustes cross-layer alignment | E | A1+A11 | The same behavior direction across layers is the parallel transport of one underlying direction; Procrustes-aligning across layers improves multi-layer steering. | Multi-layer steer with vs without cross-layer alignment; with better [NEEDS VERIFICATION] | SUPPORTED(scr): cross-layer cos 0.75-0.90 (transport) | 2 | — |

---

## BLOCK F — Robustness, safety, evaluation (E41–E50)

| ID | Short title | Block | Primary axis | Falsifiable hypothesis (1 line) | Primary metric + success threshold [NEEDS VERIFICATION] | Status | Rung reached | Idea dir |
|----|-------------|-------|-------------|--------------------------------|--------------------------------------------------------|--------|-------------|----------|
| E41 | Activation-based jailbreak resistance | F | A5 (condition) | Conditional refusal steering resists jailbreak suffixes better than system-prompt refusal, because the condition reads activations not tokens. | Refusal under adversarial suffixes; activation-based higher [NEEDS VERIFICATION] | PENDING | — | — |
| E42 | Gate cuts over-refusal | F | A5 (condition) | Over-steering for safety causes over-refusal on benign look-alikes (XSTest-style); a gate cuts this by >=70%. | Benign-refusal rate gated vs ungated; reduction >=70% [NEEDS VERIFICATION] | PENDING | — | — |
| E43 | Efficacy under domain shift | F | A7 (how derived) | Steering degrades gracefully under domain shift — efficacy decay is gradual, not catastrophic. | Efficacy across domain shifts; gradual not step-function decline [NEEDS VERIFICATION] | PENDING | — | — |
| E44 | Safety-capability Pareto frontier | F | A3 (coeff) | Multi-property safety stacks have a discoverable safety-vs-capability Pareto frontier; the knee is identifiable from per-vector alpha. | Pareto frontier traced over alpha-grid; knee identified [NEEDS VERIFICATION] | PENDING | — | — |
| E45 | HyperSteer zero-shot | F | A7 (how derived) | A held-out behavior can be steered by a hypernetwork from its description at >=70% of supervised efficacy. | Zero-shot steer efficacy >=70% of supervised vector [NEEDS VERIFICATION] | INCONCLUSIVE(scr): LOO held-out cos~=0 (mean -0.02, std 0.61) at n=4; projection-efficacy proxy unreliable, needs more behaviors+real judge (exp#111) | 1 | — |
| E46 | Iso-behavior capability curve | F | A3 (coeff) | Efficacy and capability tax are separately tunable: some (layer, alpha, sparsity) settings hold behavior fixed while sweeping capability tax. | Iso-behavior capability curve exists; settings identified [NEEDS VERIFICATION] | PENDING | — | — |
| E47 | Gate + ortho-stack + norm-cap combination | F | A5+A2+A3 | Gate (CAST) + orthogonalized stack (E19) + norm cap (E22) gives the best multi-safety-vector coherence of any tested combination. | Aggregate safety x capability score; best vs each component alone [NEEDS VERIFICATION] | PENDING | — | — |
| E48 | Prefill vs per-token steering | F | A6 (span) | Steering directions are temporally stable within a generation; per-token recomputation gives no benefit over once-at-prefill. | Per-token vs prefill steering effect; no measurable difference [NEEDS VERIFICATION] | PENDING | — | — |
| E49 | Meta-reproducibility audit | F | A7 (how derived) | Source-corpus Gemma quantitative claims reproduce within +/-20% on our 4090 for >=70% of tested papers. | Reproduction rate >=70% of tested claims within +/-20% [NEEDS VERIFICATION] | PENDING | — | — |
| E50 | Minimal-stack SOTA recipe | F | A5+A2+A3+A4 | A minimal stack recipe (gate + 3 orthogonal vectors + norm cap + DoLa) reproduces SOTA multi-property control on Gemma-2-2B within 24 GB end-to-end. | Full-bundle SOTA match; all components ablated; fits 24 GB [NEEDS VERIFICATION] | PENDING | — | — |

---

## NOVEL FIRST-PRINCIPLES HYPOTHESES N1–N12

| ID | Short title | Block | Primary axis | Falsifiable hypothesis (1 line) | Primary metric + success threshold [NEEDS VERIFICATION] | Status | Rung reached | Idea dir |
|----|-------------|-------|-------------|--------------------------------|--------------------------------------------------------|--------|-------------|----------|
| N1 | Steering manifold tangent hypothesis | Novel | A8+A9 | Effective steering is locally a tangent-space translation on the activation manifold; projecting the steering vector onto the local tangent space restores additive steering to rotational-method coherence. | Tangent-projected additive vs Angular Steering PPL at matched behavior; near-equal [NEEDS VERIFICATION] | PENDING | — | — |
| N2 | Conditioning = curvature algebraic factorization | Novel | A5+A8 | Any steering operation factorizes into a where-to-act scalar field g(h) (curvature/gating) and a what-to-do direction; CAST/SCS/discriminative-layer are the same gate operator parameterized differently. | One learned g(h) recovers CAST theta, SCS energy ratio, and selective layer-mask as special cases [NEEDS VERIFICATION] | PENDING | — | — |
| N3 | Orthogonal capacity theorem | Novel | A12 (superposition) | The number of behaviors stackable without interference equals the effective local dimensionality (participation ratio) of the activation manifold at the injection layer. | Stacking degrades sharply when N exceeds participation ratio; ratio predicts E18 knee across layers/models [NEEDS VERIFICATION] | PENDING | — | — |
| N4 | Steering as inverse ICL | Novel | A7 (how derived) | DiffMean(behavior) aligns with the activation delta induced by an in-context demonstration of the same behavior (cos>0.6). | Cosine(DiffMean, activations_ICL - activations_no_ICL) > 0.6 [NEEDS VERIFICATION] | PENDING | — | — |
| N5 | Norm-budget conservation law | Novel | A3+A12 | There is a conserved edit budget B ~ q-quantile(||h||); coherence collapses when ||sum alpha_i v_i|| > B regardless of how it is spent. | Re-plotting E17-E22 against ||delta h||/||h|| produces a single master collapse curve [NEEDS VERIFICATION] | SUPPORTED(scr): logPPL=5.40+2.87*offshell R2=0.81 (C2, 23 rows) | 2 | `ideas/_campaigns/` |
| N6 | Gate in read not write | Novel | A5+A2 | Forcing cos(condition, behavior)=0 reduces over-refusal without lowering true-positive refusal. | XSTest over-refusal drop when condition orthogonalized vs behavior; true-positive unchanged [NEEDS VERIFICATION] | PENDING | — | — |
| N7 | Parallel transport across layers | Novel | A1+A11 | Transport-aligned multi-layer steering beats independent vectors with fewer parameters. | Efficiency + coherence gain for transport-aligned vs independent per-layer vectors [NEEDS VERIFICATION] | SUPPORTED(scr): cross-layer cos 0.75-0.90 (E40) | 2 |
| N8 | Controllability != interpretability decomposition | Novel | A7+A10+A12 | A causal-objective dictionary yields fewer, more causal atoms than reconstruction SAEs and beats AxBench SAE results. | Causal-objective dictionary vs GemmaScope SAE on steering; causal better [NEEDS VERIFICATION] | PENDING | — | — |
| N9 | Steering as closed-loop dynamical control | Novel | A3+A11 | A cheap proportional feedback controller (adjust alpha from current projection error) beats fixed-alpha on long generations and resists drift. | P-controller drift over 512 tokens vs fixed-alpha; P-controller lower [NEEDS VERIFICATION] | PENDING | — | — |
| N10 | Concept algebra closure | Novel | A2+A4 | A behavior with no contrast data can be synthesized as a learned combination of primitive vectors and steered successfully at >=60% of supervised efficacy. | Zero-shot efficacy via algebraic combination >=60% of supervised [NEEDS VERIFICATION] | PENDING | — | — |
| N11 | Curvature-aware alpha per prompt | Novel | A3+A8 | A per-prompt curvature estimate (local PCA spectrum decay) predicts that prompt's cliff alpha, enabling per-prompt adaptive alpha with lower variance than E7's norm-relative scheme. | Correlation of local curvature with per-prompt cliff; variance reduction vs E7 [NEEDS VERIFICATION] | PENDING | — | — |
| N12 | Single-operator unification (capstone) | Novel | A1-A12 | Implementing a unified conditioned, tangent-projected, norm-budgeted flow operator recovers each named method (CAST/Angular/CAA/SAE-TS/FLAS/KV) on ablation and Pareto-dominates all on Gemma-2-2B. | Component ablation recovers each named method; fully-on Pareto-dominates all [NEEDS VERIFICATION] | PENDING | — | — |

---

## NOVEL FIRST-PRINCIPLES HYPOTHESES N13–N20
*(From `corpus/steering-missed-dimensions-and-highdim-algebra.md`)*

| ID | Short title | Block | Primary axis | Falsifiable hypothesis (1 line) | Primary metric + success threshold [NEEDS VERIFICATION] | Status | Rung reached | Idea dir |
|----|-------------|-------|-------------|--------------------------------|--------------------------------------------------------|--------|-------------|----------|
| N13 | Geodesic > chord | Novel | A8 (geometry) | For matched behavior change, manifold/geodesic steering yields strictly lower off-manifold displacement and lower rogue-compliance than linear add. | Off-manifold displacement and rogue-compliance lower for geodesic vs linear add at matched behavior [NEEDS VERIFICATION] | PENDING | — | — |
| N14 | Metric-matched operation | Novel | A9 (metric) | The best operation is determined by the concept's geometry: hierarchical concepts -> hyperbolic; polar/intensity concepts -> cylindrical; directional traits -> spherical; metric-mismatched operation costs coherence. | Coherence cost of metric-mismatched vs metric-matched operation on same trait [NEEDS VERIFICATION] | PENDING | — | — |
| N15 | Coset min-collateral post-processing | Novel | A10 (identifiability) | Among all behaviorally-equivalent vectors (Non-ID coset), the one with minimal projection onto fragile mid-layer subspaces has the least alignment damage at equal efficacy. | Alignment damage reduction of coset-min-collateral rep vs naive DiffMean at equal efficacy [NEEDS VERIFICATION] | PENDING | — | — |
| N16 | Radius/angle decoupling (CRH) | Novel | A9 (metric) | Steering that moves only the angle (fixed radius) is more coherent than steering that moves both; rogue damage scales with the radius excursion, not the angle. | Coherence of angle-only vs angle+radius steering at matched behavior; rogue damage correlation with radius excursion [NEEDS VERIFICATION] | SUPPORTED(scr): angular predicts rotation logPPL R2=0.997; radial predicts add R2=0.81 (C3b) | 2 | `ideas/_campaigns/` |
| N17 | Concentration penalty | Novel | A9+A12 | Off-shell displacement |delta(||h||)| predicts incoherence better than raw ||alpha*v||; norm-preserving steers beat norm-changing steers at equal angle change. | Correlation of off-shell displacement with incoherence vs raw alpha*v; norm-preserving better [NEEDS VERIFICATION] | SUPPORTED(scr): spearman(offshell,logPPL)=+0.71 (C2) | 2 | `ideas/_campaigns/` |
| N18 | Interference-budget additivity | Novel | A12 (superposition) | Stacking k near-orthogonal safety vectors is safe iff the summed interference budget (sum of |cos| overlaps) stays below a threshold; degradation predicted by sum of |cos| overlaps, not by k. | Degradation predicted by sum-of-|cos| better than by k alone; threshold identified [NEEDS VERIFICATION] | PENDING | — | — |
| N19 | Trajectory beats endpoint | Novel | A11 (dynamics) | Distributing a fixed total intervention budget across the forward trajectory (small nudges per layer/step) produces less collateral than one large endpoint shift. | Collateral at matched total budget; distributed trajectory less than single endpoint [NEEDS VERIFICATION] | PENDING | — | — |
| N20 | Curvature as fragility sensor | Novel | A8+A1 | Local manifold curvature (or local effective-rank collapse) at a layer predicts that layer's rogue-fragility, giving a behavior-free way to pick safe injection layers. | Correlation of local curvature/effective-rank with rogue-fragility across layers [NEEDS VERIFICATION] | INCONCLUSIVE(scr): eff-rank rho=-0.21 w/ fragility, underpowered; but max-Fisher L12=most fragile (C4) | 2 | `ideas/_campaigns/` |

---

## Suggested execution order

*(Transcribed faithfully from `corpus/50-steering-experiments-autoresearch.md`)*

1. **E1–E8** (tooling + measurements) — unlocks everything.
2. **N5, N3, N1** early — they define the budget/dimensionality/tangent constraints that make later stacking interpretable.
3. **E9–E26** (conditioning + stacking) interleaved with **N2, N6, N10**.
4. **E27–E40** (geometry + mechanism) with **N7, N8, N11, N4**.
5. **E41–E50** (robustness/eval) with **N9**.
6. **N12** last — the capstone unification consuming results from all prior blocks.

**N13–N20** (geometry-wave extensions) slot alongside their thematically related predecessors:
- N13, N14, N16, N17 alongside E27–E33 (geometry block).
- N15 alongside E4/E36/E37 (identifiability).
- N18 alongside E17–E22 (stacking).
- N19 alongside E40/N7/N9 (trajectory/dynamics).
- N20 alongside E2/E20 (layer selection).

---

> **Note:** All quantitative thresholds above are **[NEEDS VERIFICATION]** — they are
> pre-registered predictions transcribed from the corpus, not established facts.
> The harness confirms or falsifies every threshold on the 4090 ladder.
> Gemma-specific numbers inherited from source papers are additionally subject to
> the corpus discipline: mark `[UNVERIFIED]` on any individual claim until a
> reproduction run is recorded in `EXPERIMENT_LEDGER.md`.
