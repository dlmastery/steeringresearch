# 50 Incremental Experiments for Autonomous Research on LLM Steering
### Target: small Gemma models (Gemma-2-2B-it / Gemma-3-1B / 2B class), single RTX 4090 (24 GB)
### Each experiment = a self-contained, falsifiable unit for a Claude-Code autoresearch harness
### Synthesized from the captured steering corpus + first-principles meta-analysis of the underlying high-dimensional algebra

---

## How to read this document

Each experiment has: **H#** one clean falsifiable hypothesis; **Setup** the minimal intervention; **Metric** the primary measurement and success criterion; **Ablation/Control** what to vary to isolate the effect. Earlier experiments build tooling that later ones depend on. The final block (**N1-N12**) holds first-principles novel hypotheses: step-back claims about the geometry/algebra of steering, each with a concrete test.

**Standing infrastructure (build once, E1-E3):** a hook-based intervention library on the model gemma-2-2b-it (4-bit); contrast-pair extraction (DiffMean / PCA); a layer/coefficient sweep harness; and an eval bundle: behavior success rate, MMLU (capability retention), perplexity (coherence), and an off-target side-effect probe. Every later experiment reports against this bundle.

---

## BLOCK A - Foundations and measurement tooling (E1-E8)

**E1.** *Hypothesis:* On Gemma-2-2B, DiffMean steering vectors from >=50 contrast pairs reach >=90% of asymptotic behavior-shift effect; more pairs give diminishing returns. *Setup:* extract vectors at pair counts {10,25,50,100,200}. *Metric:* behavior success vs pair count, locate knee. *Control:* random-pair baseline.

**E2.** *Hypothesis:* The optimal injection layer is the layer of maximum linear separability of the contrast set (Fisher ratio), not a fixed late layer. *Setup:* per-layer Fisher ratio vs measured efficacy. *Metric:* Spearman >=0.7. *Ablation:* 3 behaviors.

**E3.** *Hypothesis:* Coefficient alpha has a behavior-specific coherence cliff: below it capability holds (MMLU drop <2pt), above it perplexity rises super-linearly. *Setup:* alpha sweep, record MMLU+PPL. *Metric:* inflection point, verify super-linearity. *Control:* zero-vector inject.

**E4.** *Hypothesis:* DiffMean and PCA-top-1 condition vectors are >0.95 cosine-aligned, so cheaper DiffMean suffices for CAST gating. *Metric:* cosine across layers. *Ablation:* per behavior.

**E5.** *Hypothesis:* Steering efficacy is invariant (within noise) to 4-bit vs fp16, so 4090-scale results transfer to full precision. *Metric:* behavior delta <3%. *Control:* same seed/prompts.

**E6.** *Hypothesis:* A linear probe on layer-L activations predicts pre-generation whether an input will be over-steered (incoherent), AUC>=0.8. *Setup:* label coherent/incoherent, train probe on pre-gen hidden state. *Metric:* AUC.

**E7.** *Hypothesis:* Normalizing alpha by ||h|| (relative steering) reduces cross-prompt variance vs absolute alpha. *Metric:* behavior-shift variance, relative vs absolute. *Control:* fixed alpha.

**E8.** *Hypothesis:* Behavior vectors from instruction-tuned Gemma transfer to base model with <20% efficacy loss (direction is pretraining-induced). *Metric:* cross-model efficacy. *Ablation:* reverse (base->it).

---

## BLOCK B - Conditional / gated steering, CAST family (E9-E16)

**E9.** *Hypothesis:* CAST thresholding keeps harmless-input refusal <3% while raising harmful-input refusal by >=50pt vs unconditional steering. *Metric:* harmful vs harmless refusal. *Control:* unconditional vector at matched alpha.

**E10.** *Hypothesis:* Condition vectors for distinct safety categories (hate, self-harm, illegal) are near-orthogonal (|cos|<0.3), enabling independent OR-gating. *Metric:* pairwise cosine matrix. *Ablation:* within vs cross category.

**E11.** *Hypothesis:* OR-composition of N condition vectors scales coverage linearly while harmless-refusal stays flat up to N~5, then leaks. *Metric:* coverage and false-refusal vs N. *Control:* single mega-condition.

**E12.** *Hypothesis:* Energy-ratio gating (FineSteer-SCS) gives sharper should-steer precision/recall than cosine-threshold (CAST) at equal compute. *Metric:* gate PR-AUC. *Control:* cosine gate.

**E13.** *Hypothesis:* The condition check can move to an earlier layer than behavior injection with no gating-accuracy loss, cutting latency. *Metric:* gate AUC vs condition-layer; latency. *Ablation:* same-layer.

**E14.** *Hypothesis:* Discriminative-layer selection used as a gate (steer only opposite-signed-mean layers) preserves capability better than all-layer steering. *Metric:* MMLU retention at fixed behavior. *Control:* all-layer.

**E15.** *Hypothesis:* A learned logistic gate on multi-layer activations beats a fixed cosine threshold under distribution shift. *Metric:* gate AUC on OOD prompts. *Control:* fixed theta.

**E16.** *Hypothesis:* Conditional gating cuts the capability tax of always-on safety steering by >=80% (capability spent only when gate fires). *Metric:* MMLU on benign set, gated vs always-on. *Control:* no steer.

---

## BLOCK C - Stacking and multi-vector composition (E17-E26)

**E17.** *Hypothesis:* Two near-orthogonal behavior vectors (|cos|<0.2) added together keep >=90% of solo effects, no cross-degradation. *Metric:* joint vs solo success. *Control:* anti-aligned pair.

**E18.** *Hypothesis:* Interference grows monotonically with summed off-diagonal Gram mass of the stacked vectors. *Metric:* fit interference vs Gram off-diagonal. *Ablation:* 2-5 vectors.

**E19.** *Hypothesis:* Gram-Schmidt orthogonalizing a new vector against the active set preserves the new behavior while removing interference with existing ones. *Metric:* old-behavior retention, orthogonalized vs raw. *Control:* raw add.

**E20.** *Hypothesis:* SAE-TS targeting makes stacked vectors more orthogonal in feature space than raw activation space, improving multi-behavior coherence. *Metric:* 3-stack coherence, SAE-TS vs DiffMean. *Control:* raw.

**E21.** *Hypothesis:* Conceptor AND-composition beats summed vectors for >=3 simultaneous goals at fixed coherence. *Metric:* multi-goal success at matched PPL. *Control:* mean-sum.

**E22.** *Hypothesis:* Total steering budget is how far sum(alpha_i v_i) pushes h outside the in-distribution shell; capping ||delta h|| at the empirical activation-norm quantile prevents collapse. *Metric:* PPL vs cumulative ||delta h||, locate quantile. *Control:* uncapped.

**E23.** *Hypothesis:* Additive + rotational edits on the same plane interfere destructively, but on orthogonal planes they stack - site/operation, not method name, governs composability. *Metric:* success same-plane vs orthogonal-plane. *Control:* additive-only.

**E24.** *Hypothesis:* Residual + KV-cache steering (disjoint sites) compose with >=85% of each effect on short generations but degrade on long ones from KV contamination. *Metric:* effect vs generation length. *Control:* SKOP-projected version.

**E25.** *Hypothesis:* DoLa (decoding-time) stacks on any residual steer for additive factuality gain at no coherence cost. *Metric:* TruthfulQA+PPL, steer +/- DoLa. *Control:* DoLa alone.

**E26.** *Hypothesis:* Injecting the gating-relevant vector before the behavior vector (in layer order) improves selectivity vs reverse. *Metric:* selectivity by injection order. *Control:* single-layer.

---

## BLOCK D - Geometry and rotational methods (E27-E33)

**E27.** *Hypothesis:* Norm-preserving rotation beats additive steering specifically on small (<3B) models where additive edits more easily exit the manifold. *Metric:* coherence gap additive vs rotational, 1B vs 9B. *Control:* matched behavior.

**E28.** *Hypothesis:* The behavior plane is low-rank: 2-3 dims capture >90% of steerable variance for a trait. *Metric:* variance explained by top-k of contrastive-difference space. *Ablation:* per trait.

**E29.** *Hypothesis:* Geodesic (spherical) interpolation gives more monotone behavior control than linear alpha-scaling. *Metric:* monotonicity of behavior vs control parameter. *Control:* linear.

**E30.** *Hypothesis:* Adaptive rotation (rotate only partially-aligned activations) preserves coherence better than rotating all tokens. *Metric:* PPL at matched behavior. *Control:* all-token rotation.

**E31.** *Hypothesis:* Rotational steering leaves activation norm (hence LayerNorm stats) unchanged, explaining capability preservation. *Metric:* post-LN variance additive vs rotational. *Control:* zero-edit.

**E32.** *Hypothesis:* The refusal direction and the harmfulness-detection condition direction are distinct (low cosine): detection and execution are separable subspaces. *Metric:* cosine(condition, behavior). *Ablation:* across layers.

**E33.** *Hypothesis:* Curved flow-based transport (FLAS-style) beats single-step addition mainly when the steered manifold is non-convex; on convex traits the gain vanishes. *Metric:* flow vs linear gain, stratified by convexity proxy. *Control:* linear.

---

## BLOCK E - Mechanistic and interpretability-guided (E34-E40)

**E34.** *Hypothesis:* Refusal steering acts mainly through the OV (attention-output) circuit; freezing attention scores costs <10% efficacy. *Metric:* efficacy frozen vs live attention. *Control:* full steer.

**E35.** *Hypothesis:* A behavior vector can be sparsified to <10% of dims (top-magnitude coords) with <15% efficacy loss; behaviors live in sparse coordinate sets. *Metric:* efficacy vs sparsity. *Control:* random-coord keep.

**E36.** *Hypothesis:* SAE-feature steering underperforms raw DiffMean unless features are output-score-selected - the AxBench tension is a selection problem, not a representation problem. *Metric:* SAE-steer (naive vs selected) vs DiffMean. *Control:* prompting.

**E37.** *Hypothesis:* Causally-steerable features are a small subset of interpretable SAE features; interpretability does not imply controllability. *Metric:* overlap of high-interpretability and high-causal-effect features.

**E38.** *Hypothesis:* Function/task vectors (ICL-derived) and DiffMean behavior vectors share subspace; composing them transfers task + style in one edit. *Metric:* joint task+style success. *Control:* either alone.

**E39.** *Hypothesis:* Persona-vector monitoring predicts behavioral drift before it shows in outputs, enabling pre-emptive gating. *Metric:* lead-time between persona-projection shift and output shift. *Control:* output-only detector.

**E40.** *Hypothesis:* The same behavior direction across layers is the parallel transport of one underlying direction; Procrustes-aligning across layers improves multi-layer steering. *Metric:* multi-layer steer with vs without cross-layer alignment. *Control:* independent per-layer.

---

## BLOCK F - Robustness, safety, evaluation (E41-E50)

**E41.** *Hypothesis:* Conditional refusal steering resists jailbreak suffixes better than system-prompt refusal, because the condition reads activations not tokens. *Metric:* refusal under adversarial suffixes. *Control:* prompt-based refusal.

**E42.** *Hypothesis:* Over-steering for safety causes over-refusal on benign look-alikes (XSTest-style); a gate cuts this by >=70%. *Metric:* benign-refusal rate gated vs ungated. *Control:* no steer.

**E43.** *Hypothesis:* Steering degrades gracefully under domain shift - efficacy decay is gradual, not catastrophic. *Metric:* efficacy across domain shifts. *Control:* in-domain.

**E44.** *Hypothesis:* Multi-property safety stacks have a discoverable safety-vs-capability Pareto frontier; the knee is identifiable from per-vector alpha. *Metric:* trace frontier over alpha-grid. *Control:* single property.

**E45.** *Hypothesis:* A held-out behavior unseen during extraction can be steered by a hypernetwork (HyperSteer-style) from its description at >=70% of supervised efficacy. *Metric:* zero-shot steer efficacy. *Control:* supervised vector.

**E46.** *Hypothesis:* Efficacy and capability tax are separately tunable: some (layer, alpha, sparsity) settings hold behavior fixed while sweeping capability tax. *Metric:* iso-behavior capability curve. *Control:* naive alpha-scaling.

**E47.** *Hypothesis:* Gate (CAST) + orthogonalized stack (E19) + norm cap (E22) gives the best multi-safety-vector coherence of any tested combination. *Metric:* aggregate safety x capability. *Control:* each component alone.

**E48.** *Hypothesis:* Steering directions are temporally stable within a generation; per-token recomputation gives no benefit over once-at-prefill. *Metric:* per-token vs prefill steering. *Control:* prefill-only.

**E49.** *Hypothesis:* Source-corpus Gemma quantitative claims reproduce within +/-20% on our 4090 for >=70% of tested papers (meta-reproducibility audit). *Metric:* reproduction rate. *Control:* documented per-paper.

**E50.** *Hypothesis:* A minimal stack recipe (gate + 3 orthogonal vectors + norm cap + DoLa) reproduces SOTA multi-property control on Gemma-2-2B within 24 GB end-to-end. *Metric:* full-bundle SOTA match. *Control:* ablate each component.

---

## NOVEL FIRST-PRINCIPLES HYPOTHESES (N1-N12)
*Step-back claims about the underlying high-dimensional algebra of steering - not incremental tweaks. Each is risky, falsifiable, and motivated by the geometry the corpus only gestures at.*

**N1 - The Steering Manifold Tangent Hypothesis.** *Claim:* Effective steering is locally a tangent-space translation on the activation manifold; additive steering fails on small models because the additive vector leaves the tangent plane (the manifold curves faster in low dimension). *Prediction:* projecting the steering vector onto the local tangent space (from a kNN patch of natural activations) restores additive steering to rotational-method coherence, unifying additive and rotational steering as flat vs curved approximations of one operation. *Test:* tangent-projected additive vs Angular Steering at matched behavior; predict near-equal PPL.

**N2 - Conditioning = Curvature, Behavior = Direction (Algebraic Factorization).** *Claim:* Any steering operation factorizes into a where-to-act field (a function of h: curvature/gating) and a what-to-do vector (direction). CAST, FineSteer-SCS, and discriminative-layer selection are the same gate operator parameterized differently. *Prediction:* one learned scalar field g(h) times a fixed direction reproduces all three. *Test:* fit g(h); show it recovers CAST theta, SCS energy ratio, and Selective layer-mask as special cases.

**N3 - The Orthogonal Capacity Theorem (empirical).** *Claim:* The number of behaviors stackable without interference equals the effective local dimensionality (participation ratio) of the activation manifold at the injection layer, not the hidden size. *Prediction:* stacking degrades sharply once N exceeds the participation ratio of the activation covariance. *Test:* measure participation ratio per layer; show it predicts the E18 stacking knee across layers/models.

**N4 - Steering as Inverse In-Context Learning.** *Claim:* A behavior vector is the gradient of the model's own ICL update for that behavior; steering and prompting are dual representations of one operator (prompt = data-space, vector = activation-space). *Prediction:* DiffMean(behavior) aligns with the activation delta induced by an in-context demonstration of the same behavior (cos>0.6). *Test:* compare DiffMean to (activations_with_ICL minus activations_without); predict high cosine, explaining why prompt-mimicking steering works.

**N5 - The Norm-Budget Conservation Law.** *Claim:* There is a conserved edit budget B ~ q-quantile(||h||); coherence collapses when ||sum alpha_i v_i|| > B regardless of how it is spent (one strong vector or many weak). *Prediction:* a single collapse curve vs normalized edit magnitude collapses all multi-vector experiments onto one master curve. *Test:* re-plot E17-E22 against ||delta h||/||h||; predict data collapse.

**N6 - Gating Belongs in the Read, Not the Write (Separation Principle).** *Claim:* Detection (condition) and execution (behavior) are more robust in orthogonal subspaces; entangling them in one vector is the root cause of over-refusal. *Prediction:* forcing cos(condition, behavior)=0 reduces over-refusal without lowering true-positive refusal. *Test:* orthogonalize condition vs behavior; measure XSTest over-refusal drop.

**N7 - Behaviors are Parallel-Transported, Not Re-Learned, Across Layers.** *Claim:* The same behavior direction at different depths is one object transported along the residual stream; correct multi-layer steering is a transported field, not independent per-layer vectors. *Prediction:* transport-aligned multi-layer steering beats independent vectors with fewer parameters. *Test:* E40 with explicit transport operator; predict efficiency + coherence gain.

**N8 - The Controllability != Interpretability Decomposition.** *Claim:* Activation space splits into an interpretable-but-inert subspace and a controllable subspace; SAEs over-index on the former. A steering-optimal dictionary should be learned with a causal (intervention) objective, not reconstruction. *Prediction:* a causal-objective dictionary yields fewer, more causal atoms than reconstruction SAEs and beats AxBench SAE results. *Test:* train causal-objective dictionary on Gemma-2-2B; compare steering to GemmaScope SAE.

**N9 - Steering as Control of a Latent Dynamical System.** *Claim:* Token-by-token generation is a discrete dynamical system in activation space; a steering vector is open-loop control, and the right formulation is closed-loop (feedback proportional to gap from a target manifold). *Prediction:* a cheap proportional feedback controller (adjust alpha from current projection error) beats fixed-alpha on long generations and resists drift. *Test:* P-controller on the behavior projection; compare drift over 512 tokens vs fixed-alpha.

**N10 - The Concept Algebra Closure Hypothesis.** *Claim:* Behavior directions form an approximately closed algebra under linear combination plus a few nonlinear composition operators (conceptor AND/OR are its linearized shadow). Novel behaviors are reachable as algebraic expressions of primitives. *Prediction:* a behavior with no contrast data can be synthesized as a learned combination of primitive vectors and steered successfully. *Test:* hold out behavior B; express B = f(primitives); steer; predict >=60% of supervised efficacy.

**N11 - Curvature-Aware Conditioning Predicts the Coherence Cliff.** *Claim:* The alpha coherence cliff (E3) occurs where steering displacement crosses a local curvature threshold of the manifold; high-curvature regions tolerate less steering. *Prediction:* a per-prompt curvature estimate (local PCA spectrum decay) predicts that prompt's cliff alpha, enabling per-prompt adaptive alpha. *Test:* correlate local curvature with per-prompt cliff; deploy adaptive alpha; predict variance reduction beyond E7's norm-relative scheme.

**N12 - The Single-Operator Unification (capstone).** *Claim:* CAST (gate), Angular (rotate), CAA (add), SAE-TS (orthogonalize), FLAS (flow), KV-steering (cache-edit) are special cases of one operator: a conditioned, tangent-projected, norm-budgeted flow on the activation manifold, h <- h + g(h) * Proj_T(Phi_t(v)) capped at budget B. *Prediction:* implementing this single parameterized operator and ablating its terms recovers each named method, and the fully-on configuration Pareto-dominates all of them on Gemma-2-2B. *Test:* build the unified operator; run a component ablation reconstructing each method; verify Pareto dominance - the capstone the whole program builds toward.

---

## Suggested execution order for the autoresearch harness
1. **E1-E8** (tooling + measurements) - unlocks everything.
2. **N5, N3, N1** early - they define the budget/dimensionality/tangent constraints that make later stacking interpretable.
3. **E9-E26** (conditioning + stacking) interleaved with **N2, N6, N10**.
4. **E27-E40** (geometry + mechanism) with **N7, N8, N11, N4**.
5. **E41-E50** (robustness/eval) with **N9**.
6. **N12** last - the capstone unification consuming results from all prior blocks.

*All quantitative thresholds are starting targets / pre-registered predictions, not established facts; the harness should confirm or falsify them. Gemma-usage and numeric claims inherited from the source corpus remain [NEEDS VERIFICATION].*
