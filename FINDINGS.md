# FINDINGS — Steering Research Program

---

## How to read this document

### What this project is doing

This research program tests whether small language models can be steered — nudged
to behave differently — by injecting carefully constructed vectors directly into
their hidden layers at inference time, without any fine-tuning or retraining. This
technique is called "activation steering." The models under study are Gemma-3-270m,
Gemma-3-1B (both from Google), and Qwen-2.5-0.5B (from Alibaba). All are small,
running on a single RTX 4090 Laptop with 16 GB of video memory.

We are testing 70 pre-registered hypotheses about how activation steering works,
under what conditions it fails, and which mathematical approaches are most efficient
and safe. All hypotheses, metrics, and success thresholds were written down and
committed to version control before the experiments ran.

### Identifier glossary — every code defined here, inline

**E1 through E50** are the 70 pre-registered hypotheses from the project's
hypothesis registry (stored in IDEA_TABLE.md). They cover: foundational measurement
methods (E1–E8), conditional or gated steering (E9–E16), stacking multiple steering
vectors simultaneously (E17–E26), geometry and rotational methods (E27–E33),
mechanistic interpretability (E34–E40), and robustness and safety evaluation
(E41–E50). Example: "E4" is the specific hypothesis that two methods for constructing
a steering vector — DiffMean and PCA-top1 (both defined below) — produce nearly the
same direction (cosine similarity above 0.95).

**N1 through N20** are twenty novel first-principles hypotheses, also pre-registered,
probing deeper geometric and algebraic structure. Example: "N17" is the hypothesis
that a cheap geometric measurement — how far a steering operation pushes the model's
hidden state off the surface of its normal activation sphere (called "off-shell
displacement") — predicts how incoherent the model's outputs will become.

**S-1 through S-21** are the screening/confirmation observations recorded in this
document. S-1 through S-14 are directional findings from single experimental runs
(n=1). S-15 is different: it is the program's first controlled, multi-seed (n=20),
off-family-judged, cross-scale result — measured the way the earlier single-seed
screens were not. It was the strongest and most rigorous observation in the program
on a SYNTHETIC single concept ("ocean"). S-16 is the program's first evaluation on a
REAL external benchmark (AxBench, 500 concepts), run at TWO model scales: it tests
the very same E7 directional effect that looked supported in S-15. At 270M the effect
is NEGATIVE/at the floor (the model cannot express the concepts at all); at 2B the
concepts become expressible (~0.135, 3× higher) and a SMALL, statistically-significant
real-direction advantage appears (+0.004, p=0.011) — but it is tiny, fragile (CI barely
excludes 0), fails the strict ordinal gate, and a label-shuffled vector still captures
~97% of the steering effect. So the +0.135 directional effect the synthetic "ocean"
concept suggested (S-15) does NOT generalize: on a real population of concepts the
advantage is weak, scale-dependent, and mostly generic. S-16 therefore corrects and
contextualizes S-15. S-17 returns to a DIFFERENT question — the alpha-coherence cliff
(E3, originally confirmed on synthetic data in S-2, S-4, S-8, S-9, S-11) — and
tests it on the REAL AxBench benchmark with 30 concepts and an off-family local fluency
judge measuring both behavior and coherence simultaneously. The cliff is SUPPORTED: behavior
and coherence both peak at alpha~0.10 and collapse super-linearly past alpha~0.20. This
generalizes the program's geometry/coherence results (E3, N17) to a real benchmark, in
contrast to E7's directional claim (S-16), which did not generalize. S-18 asks the
complementary E2 question — which layer is best to inject the steering vector? — but
on the REAL AxBench benchmark with 20 concepts at 2B (26-layer Gemma-2-2B-it), sweeping
layers 6–22. The layer curve is NEARLY FLAT: behavior ranges only 0.163–0.184 (~13%
relative) with a shallow peak at layer 18; coherence ranges 0.606–0.669 with a shallow
peak at layer 20. There is a weak mid-to-late layer preference but no sharp single optimum.
This extends the E7/E3 emerging theme: DIRECTION and LAYER matter little; the dominant
control variable is alpha (the E3 coherence cliff, S-17). S-19 returns to the E4 question —
whether DiffMean and PCA-top1 are interchangeable — but on 100 REAL AxBench concepts
(Gemma-2-2B-it, layer 20, exp#122): the mean |cos(DiffMean, PCA-top1)| is only 0.65
(median 0.73, p5 0.19, only 16% of concepts >=0.90). The synthetic cos~0.99 result did NOT
generalize; on real concepts the two extraction methods are only MODERATELY aligned. S-20
tests E27 (rotation vs additive steering) on the same real AxBench grid (Gemma-2-2B-it,
layer 20, 20 concepts × 8 eval instructions, relative_add/rotate at alpha=0.1, exp#123):
the rotate-vs-add delta is only -0.003 (diffmean: add 0.169/0.681 vs rotate 0.166/0.681) —
rotation gives no benefit; the operations are roughly equal. This supersedes the
synthetic E27 FALSIFICATION with a real-benchmark NOT SUPPORTED verdict: rotation is
neither dramatically worse nor better than addition on real abstract concepts at this scale.
S-21 tests E36 (PCA vs DiffMean source) on the same grid (exp#124): pca+add yields behavior
0.178 vs diffmean+add 0.169 — a +0.009 margin — but pca coherence (0.644) is lower than
diffmean coherence (0.681). A wash: source barely matters, PCA marginally higher behavior,
DiffMean marginally higher coherence, differences tiny.
None of S-1 through S-21 meets the full six-part bar required for an external claim.

**"rung-3"** refers to the third level on a five-rung evaluation ladder this project
uses. Rung 0–2 experiments are cheap, fast, and run on small synthetic datasets.
Rung 3 (called "STANDARD") means real held-out benchmark data, proper statistical
tests, and cross-scale validation. This program has completed one rung-3 evaluation
attempt, described in detail below.

**Verdict tiers** — how each hypothesis is classified after testing:
- SUPPORTED (scr) — points in the predicted direction in a screening (n=1) run; not yet externally citable
- FALSIFIED (scr) — contradicts the pre-registered prediction in a screening run
- PARTIAL (scr) — mixed result: some predictions confirmed, others not
- INCONCLUSIVE (scr) — underpowered or confounded; no directional conclusion possible
- PENDING — not yet tested

**Composite score** — a single number (reported to 4 decimal places) that prices
all five measurement axes simultaneously, so no method can appear to "win" by
sacrificing one axis. The five axes are: (1) behavior efficacy — did the steering
actually change the model's behavior in the intended direction? (2) capability
retention — did scores on reasoning benchmarks like MMLU stay stable? (3) coherence
— is the output still grammatical, measured by perplexity? (4) safety integrity —
did steering create any harmful-compliance leak? (5) selectivity — does the method
fire only on the right inputs? The composite formula is cryptographically fingerprinted
(fingerprint: `a9001e87087e`) so it cannot be silently changed.

**Perplexity (PPL)** — a standard measure of how surprised a language model is by
text. A steered model that has been pushed off its normal behavior distribution
assigns higher surprise scores (higher PPL) to normal English sentences, signaling
incoherent outputs. Baseline PPL for these small models on WikiText-2 prose is
roughly 74–90. PPL of 1,000+ means the model is essentially generating garbage.

**Off-shell displacement (written Δ‖h‖)** — after a steering vector is added to
a hidden layer's activation, this measures how much the vector's length changed.
Think of the model's activations as normally living on a roughly spherical surface
(the "shell"). Steering that pushes the activation far off that sphere correlates
strongly with output degradation. It can be computed cheaply without running the
model to generate text.

**DiffMean** — a simple, cheap method to construct a steering vector: take the
average hidden-layer activation for a set of positive-class examples (e.g., sentences
expressing "anger"), subtract the average for negative-class examples (neutral
sentences), and use the difference as the steering direction.

**PCA-top1** — an alternative vector construction method: run Principal Component
Analysis on the set of positive-minus-negative activation differences and use the
top principal component as the steering direction. More principled but more
computationally expensive than DiffMean.

**Alpha (α)** — the multiplier on the steering vector. Larger alpha = stronger push.
In "relative steering," alpha is expressed as a fraction of the activation's natural
magnitude (e.g., alpha=0.1 means "push 10% of the activation's current length in
the steering direction"). In "absolute steering," alpha is a raw number.

**Behavior efficacy score** — how well the steered model's outputs incorporate the
target concept, measured either by cosine similarity between output activations and
the steering vector (the fast but circular proxy used in screening runs) or by an
independent LLM judge scoring real generated text (the real measure, required for
external claims but not yet wired in).

**"SCREENING ONLY — not an external claim"** — this phrase means the observation
was made in a single run (n=1) and has not cleared the six-part statistical gate
described in the Rigor Contract below. It cannot be cited as a research finding.

---

## What counts as an external-ready finding — the Rigor Contract

A result may be promoted from "screening observation" to "finding" only when ALL
six conditions hold simultaneously:

1. At least 7 independent random seeds were run for the winning configuration.
2. A paired Wilcoxon signed-rank test on the improvement delta passes p < 0.05.
3. A 95% bootstrap confidence interval (10,000 resamples) on the delta excludes zero.
4. Holm-Bonferroni multiple-comparisons correction has been applied across the whole
   family of experiments in the sweep (to guard against false positives from testing
   many configurations).
5. The worst evaluation seed still beats the best baseline seed (ordinal gate —
   ensures the result is not driven by a single lucky run).
6. The result is logged in EXPERIMENT_LEDGER.md with verdict "KEEP" at rung 3 or above.

Additionally, every finding must price all five axes via the fingerprinted composite.
A method that achieves perfect behavior but makes outputs incoherent, or causes a
safety leak, does not qualify — no single-axis wins allowed.

Moving a result from screening to findings without meeting this full contract is
a HARKing violation (Hypothesizing After Results are Known) and is a project BLOCKER.

---

## Current status

**No external-ready findings yet — 124 experiments run.**

### Real-benchmark synthesis (S-16 through S-21): alpha is the only strong knob

Across all six testable hypotheses re-evaluated on the real AxBench benchmark
(pyvene/axbench-concept500) with an off-family judge (Qwen2.5-7B-Instruct, AUC 0.68
disclosed), the ONLY steering knob that STRONGLY matters is alpha — the E3 coherence
cliff (S-17, confirmed). Every other design choice has a WEAK or marginal effect: the
specific concept DIRECTION barely beats a shuffled control (E7, S-16 — weak,
scale-dependent, mostly-generic), the LAYER is nearly flat (E2, S-18), the SOURCE
(DiffMean vs PCA-top1) is a wash on steering even though the two methods produce only
moderately aligned directions on real concepts (E36, S-21 and E4, S-19), and the
OPERATION (add vs rotate) gives no benefit (E27, S-20). Synthetic single-concept
evaluation systematically OVERSTATED all of these effects; only the
geometry/coherence finding (alpha cliff, consistent with N17 off-shell displacement
predicts incoherence) generalized to a real benchmark. Practical takeaway: tune alpha
(~0.10) and stop optimizing direction, layer, source, or operation — they barely move
the needle on real abstract concepts at this scale. The external-ready count remains
zero.

On the REAL AxBench benchmark, across two model scales (exp#118 at 270M, exp#119 at
2B), E7's directional steering advantage over a label-shuffled control is
SCALE-DEPENDENT and WEAK: NEGATIVE/at the floor at 270M, and
statistically-significant-but-tiny (+0.004, p=0.011) at 2B where the ordinal gate
still fails and a shuffled vector captures ~97% of the effect — the synthetic-"ocean"
+0.135 win (S-15) did NOT generalize (S-16). The E3 ALPHA-COHERENCE CLIFF is
SUPPORTED on real AxBench (exp#120): behavior and coherence both peak at alpha~0.10
and collapse super-linearly past alpha~0.20 (S-17). The E2 LAYER SWEEP finds a NEARLY
FLAT layer curve — behavior 0.163–0.184 across layers 6–22, shallow peak at layer 18
(S-18). E4 (DiffMean vs PCA-top1 alignment) on 100 real AxBench concepts finds only
MODERATE alignment — mean |cos| 0.65, only 16% of concepts >=0.90 (S-19). E27
(rotation vs addition) on the (source × operation) 4-cell grid (exp#123) finds
rotate-vs-add delta = −0.003 at identical coherence — rotation gives no benefit, NOT
SUPPORTED (S-20). E36 (PCA vs DiffMean as steering source) on the same grid (exp#124)
finds pca+add marginally higher behavior (+0.009) but lower coherence (−0.037) vs
diffmean+add — a wash, source barely matters (S-21).

124 experiments have run on real Gemma-3-270m, Gemma-3-1B, Gemma-2-2B, and
Qwen-2.5-0.5B. They produced 21 screening/confirmation observations (S-1 through
S-21) spanning 19 of the 70 hypotheses. Six rung-3-style real-benchmark evaluations
have been completed: N17 (off-shell displacement predicts incoherence), E7 on
SYNTHETIC data (exp#116/117 — PROVISIONAL), E7 on REAL AxBench at two scales
(exp#118/119 — S-16), E3 coherence cliff on REAL AxBench (exp#120 — S-17), E2 layer
sweep on REAL AxBench (exp#121 — S-18), and the combined E27/E36 (source × operation)
grid on REAL AxBench (exp#123/124 — S-20/S-21). A pure geometry check on E4
(exp#122 — S-19) completes the real-benchmark correction body.

The E7 AxBench evaluation is the important reality check: it replaced ALL prior
synthetic/hand-authored data with AxBench's 500 real concepts, contrast text, and
held-out eval instructions, scored by an off-family local judge (Qwen2.5-7B-Instruct,
4-bit; judge validated at ROC-AUC 0.68 vs AxBench ground truth — WEAK but UNBIASED,
disclosed). The replication unit is the CONCEPT (n=500). On AxBench the real DiffMean
direction's advantage over a matched-displacement shuffled-label control is
scale-dependent: at 270M the real direction is significantly WORSE and both conditions
are at the floor (real 0.046 vs shuffled 0.056; paired delta −0.010, 95% CI
[−0.0142, −0.0066], p=1.4×10⁻⁶) — the model essentially cannot express the abstract
concepts at all. At 2B (the scale AxBench was built for, layer 20) the concepts become
expressible (both conditions ~0.135, 3× higher), and the real direction significantly
BEATS shuffled — but the effect is substantively TINY (+0.004 on a 0–1 scale, ~3%
relative; 95% CI [+0.0003, +0.0077] barely excludes 0; Wilcoxon p=0.011), the ordinal
gate still FAILS, and the label-shuffled vector captures ~97% of the steering effect.
This is the most rigorous evaluation of E7 and it is, honestly, a WEAK/negative result
for DiffMean steering: the real concept direction carries only a weak concept-specific
signal; most of the steering effect is generic, not direction-specific.

The exp#120 E3 AxBench evaluation (S-17) is a complementary result: it tests a
DIFFERENT claim — whether the coherence cliff shape is real on a real benchmark — and
finds a clear positive confirmation. The geometry/coherence findings (E3 cliff, N17
off-shell predicts incoherence) generalize to AxBench; the direction-specificity
finding (E7) does not. The exp#121 E2 AxBench evaluation (S-18) extends this to the
layer-selection question: the layer curve is nearly flat. The exp#122 E4 geometry
check (S-19) extends the pattern to E4: on 100 real AxBench concepts the DiffMean and
PCA-top1 directions are only moderately aligned (mean |cos| 0.65). The exp#123 E27
operation check (S-20) and exp#124 E36 source check (S-21) complete the (source ×
operation) grid and extend the pattern further: neither operation type nor source
selection provides a meaningful advantage on real abstract concepts.

By the program's own rigor floor there are STILL zero external-ready findings.
(S-17 caveats: 30 concepts only — a curve, not a population test; single model 2B +
single layer 20; judge AUC 0.68 (disclosed, unbiased); the cliff alpha may shift with
model/layer, but the qualitative shape — peak ~0.10, super-linear collapse past ~0.20
— is robust across the 30 concepts and consistent with S-2, S-4, S-8, S-9, S-11.
S-18 caveats: 20 concepts, single model/2B, single knee alpha=0.1, judge AUC 0.68.
S-19 caveats: 100 concepts, single model/layer, unpaired AxBench data. S-20/S-21
caveats: 20 concepts, single model/layer, 4-cell grid, judge AUC 0.68; the wash verdict
holds for the means but individual-concept variation is not captured.) The screening
observations motivate the next required experiments but are NOT citable claims.

---

## Summary table — every hypothesis tested so far

All verdicts are SCREENING (single-run, n=1) unless marked [rung-3].

| Hypothesis | Plain-English question | Verdict | One-line result |
|---|---|---|---|
| E1: DiffMean pair-count knee | How many contrast pairs do you need before the steering vector stabilizes? | DIRECTIONAL (underpowered) | Vector reaches >0.95 cosine of full at ~5 pairs on a simple concept; the corpus's 50-pair threshold cannot be tested yet |
| E2: Fisher layer selection | Is the most linearly separable layer the best one to inject the steering vector? | FALSIFIED; additionally FLAT on real AxBench | Spearman(Fisher ratio, efficacy) = +0.14, p=0.74; the most separable layer on Gemma-270m (layer 12) is not the best steering layer; on REAL AxBench (2B, 20 concepts, exp#121) the layer curve is NEARLY FLAT — behavior 0.163–0.184 across L6–22, shallow peak L18, no sharp optimum (S-18) |
| E3: Alpha coherence cliff | Does output quality collapse super-linearly once steering strength exceeds a threshold? | SUPPORTED — now confirmed on REAL AxBench (exp#120, S-17) | Confirmed on all three models across synthetic data; additionally confirmed on 30 real AxBench concepts (2B, layer 20, off-family judge) — behavior + coherence peak at alpha~0.10 and collapse super-linearly past alpha~0.20 |
| E4: DiffMean vs PCA-top1 alignment | Do the cheap and expensive vector construction methods produce the same direction? | PARTIAL — synthetic cos~0.99 did NOT generalize to real AxBench (S-19) | Synthetic screening: cosine 0.994–0.999 across 3 models and 4 behaviors. On 100 REAL AxBench concepts (2B, layer 20, exp#122): mean |cos| 0.65, median 0.73, p5 0.19, only 16% >=0.90. The two methods are only MODERATELY aligned on real concepts; the near-equivalence was a paired-data artifact |
| E7: Norm-relative alpha | Does expressing steering strength as a fraction of activation magnitude reduce variability — and does the real concept direction beat a matched-displacement control? | WEAK / SCALE-DEPENDENT on REAL AxBench [rung-3]; the synthetic-only +0.135 win did NOT generalize | On the synthetic single concept "ocean" the real direction beat the shuffled control on both scales (S-15, PROVISIONAL, +0.135). On the REAL AxBench benchmark (500 concepts, off-family local judge AUC 0.68) the advantage is weak and scale-dependent: at 270M (exp#118) real 0.046 vs shuffled 0.056, delta −0.010, CI [−0.0142, −0.0066], p=1.4e-6 — significant but NEGATIVE, both at the floor (~0.05/1.0); at 2B (exp#119) concepts become expressible (both ~0.135) and real BEATS shuffled but only by +0.004 (CI [+0.0003, +0.0077] barely excludes 0, p=0.011), ordinal gate FAILS, shuffled captures ~97% of the effect. Weak, scale-dependent, mostly-generic — see S-16 |
| E10: Category orthogonality | Do steering vectors for distinct concepts point in different directions? | PARTIAL | Mostly orthogonal (|cosine| < 0.3) except anger and happiness share a component (cosine = +0.48) |
| E17: Near-orthogonal stacking | Can two behavior vectors be applied simultaneously without mutual interference? | SUPPORTED | Stacking anger + happiness retains 101%/110% of solo behavior despite +0.48 cosine overlap |
| E18: Interference vs Gram mass | Does interference between stacked vectors grow proportionally to their geometric overlap? | PARTIAL | Stacking retains 85–94% behavior, but retention is not monotone in Gram off-diagonal mass |
| E22: Norm budget and collapse | Does cumulative displacement across all stacked vectors jointly govern coherence collapse? | SUPPORTED | Four stacked concepts drive PPL from 138 to 4,518 as total displacement scales; consistent with a shared norm budget |
| E27: Rotation beats addition on small models | Does norm-preserving rotation outperform additive steering on small models? | FALSIFIED (full-vector, synthetic); NOT SUPPORTED on real AxBench — rotate ~ add | Full-vector rotation costs +42% PPL at matched behavior vs additive on synthetic (caveat: falsifies full-vector, not selective-plane rotation); on REAL AxBench (2B, layer 20, 20 concepts × 8 eval, exp#123) rotate-vs-add delta = −0.003 (diffmean: add 0.169/coh 0.681 vs rotate 0.166/coh 0.681) — rotation gives no benefit, operations roughly equal (S-20) |
| E28: Behavior plane low-rank | Does the steering direction live in a 2–3 dimensional subspace? | FALSIFIED | Top-3 SVD dimensions explain only 66% of variance; not low-rank |
| E35: Sparse behavior vector | Can the steering vector be compressed to 10% of its coordinates with minimal loss? | PARTIAL | Top-10% coordinates retain 77% behavior; below the 85% target |
| E36: DiffMean equals PCA at matched displacement | Do the two construction methods steer identically when compared fairly (equal displacement)? | SUPPORTED (synthetic); WEAK/WASH on real AxBench — source barely matters | At matched fractional displacement on synthetic data, behavior within 0.02 and PPL within 8%; earlier apparent gaps were a parameterization artifact. On REAL AxBench (2B, layer 20, 20 concepts × 8 eval, exp#124): pca+add behavior 0.178 vs diffmean+add 0.169 (+0.009 margin) but pca coherence 0.644 vs diffmean coherence 0.681 (−0.037) — a wash, source barely matters, PCA marginally higher behavior, DiffMean marginally higher coherence (S-21) |
| E40: Cross-layer direction continuity | Is the same steering direction approximately preserved as you move through network depth? | SUPPORTED | Cross-layer cosine 0.75–0.90 for adjacent layers; the direction is approximately parallel-transported through depth |
| N5: Universal norm-budget law | Is there one equation predicting coherence from displacement that works across all models? | FALSIFIED across scale [rung-3] | The monotone direction holds but the law's coefficients are model-specific; held-out R² = −1.6 when predicting across scales |
| N7: Parallel transport across layers | Is the same behavior direction preserved through the network? | SUPPORTED | Same evidence as E40; cross-layer cosine 0.75–0.90 |
| N16: Cylindrical Representation Hypothesis | Does coherence cost decompose into independent radial (norm change) and angular (direction change) components? | SUPPORTED | Angular displacement predicts rotation's log-PPL at R²=0.997; radial displacement predicts additive steering's log-PPL at R²=0.81; the two components are independent predictors |
| N17: Off-shell displacement predicts incoherence | Does measuring how far steering pushes the activation off the sphere predict output incoherence without generation? | SUPPORTED [rung-3] | Spearman = +0.585, 95% CI [+0.353, +0.758], p=8e-6, on real WikiText-2 across two model sizes |
| N20: Curvature as fragility sensor | Does per-layer activation geometry predict which layers are most fragile to steering? | INCONCLUSIVE | Effective-rank correlates weakly with fragility (Spearman −0.21); underpowered at 8 layers |

---

## The strongest result — confirmed on real held-out data (rung-3 evaluation)

### How far you push the activation off the sphere predicts how incoherent the output will be — confirmed on real text across two model sizes

**What hypothesis this tests:** "N17 — Concentration penalty." The pre-registered
prediction: the cheap geometric quantity called off-shell displacement (how much the
activation's magnitude changes after steering) should correlate positively with output
incoherence (perplexity), and this relationship should hold across model sizes and
injection layers.

**Why this matters:** If true, you can screen hundreds of steering configurations
for coherence impact in seconds by measuring activation geometry — without generating
a single word of text. This is a major practical tool for safe steering.

**What "off-shell displacement" means:** In a trained language model, the hidden
vector at each layer (the "activation") tends to have a roughly consistent length
across normal inputs. Imagine all normal activations living on the surface of a
high-dimensional sphere. Additive steering pushes the activation outward or inward
— off the sphere. The off-shell displacement is the absolute change in the
activation vector's length: |length after steering − length before steering|. The
hypothesis is that larger displacement → higher perplexity (more incoherent outputs).

**Earlier screening evidence (n=1, within-pool):** In 23 pooled data points across
Gemma-3-270m and Qwen-2.5-0.5B, 8 layers, and a range of steering strengths:
Spearman(off-shell displacement, log-perplexity) = +0.705; Pearson = +0.899.
A single linear equation — log PPL = 5.40 + 2.87 × off-shell displacement — fit
the pooled data with R² = 0.81 across both architectures and all layers. This
suggested a universal predictive law.

**Rung-3 evaluation on real held-out text:** Using 40 passages from WikiText-2
(real English prose, not synthetic data), pooling 50 data points across
Gemma-3-270m and Gemma-3-1B, multiple injection layers, and a range of steering
strengths (all using relative steering, where alpha = fraction of activation norm):

- Spearman(off-shell displacement, log real-perplexity) = **+0.585**
- 95% bootstrap confidence interval (10,000 resamples): **[+0.353, +0.758]** — excludes zero
- p-value: 8.1 × 10⁻⁶

The monotone relationship holds on genuine English text, across two different model
sizes, and across many different experimental configurations. This is the program's
first rung-3 evaluation and its strongest, most defensible result.

**Remaining caveat — why this is still not an external-ready finding:** The 50
data points come from a grid of layer × steering-strength configurations, not from
50 independent random seeds or 50 independent behaviors. The bootstrap confidence
interval is a within-grid estimate. A fully external claim still requires replication
across independent prompt sets and independent behaviors, with n≥7 seeds under the
full six-part statistical contract.

---

### The universal numerical law does NOT transfer across model sizes

**What was also tested:** Whether the specific coefficients of the
log-PPL = a + b × displacement equation, fit on the smaller Gemma-3-270m, correctly
predict the larger Gemma-3-1B's perplexity values on the same type of data.

**Result:** Held-out R² = **−1.6**. An R² below zero means the transferred equation
performs worse than simply predicting the mean perplexity for all configurations —
it actively mispredicts the 1B model's behavior. The 270m slope (78.85) and intercept
(4.65) do not transfer.

**What this means in plain language:** "Steer more → worse outputs" is true and
robust across model sizes (that is N17, confirmed above). But "exactly how much
worse as a function of displacement" is model-specific and requires separate
calibration per model. The earlier within-pool R²=0.81 mixed both models in one
regression, which inflated the apparent universality. This is precisely the failure
mode that rung-3 held-out testing is designed to catch.

**Status:** N17 (directional, monotone claim) is SUPPORTED at rung-3 level with
appropriate caveats. N5 (single universal numerical law across all models) is
FALSIFIED at the across-scale level.

---

## Screening observations (SCREENING ONLY — not external claims)

Observations S-1 through S-14 below come from n=1 runs on models of 270M–1B
parameters using synthetic mini-datasets and an activation-projection behavior
proxy. They establish directional evidence and motivate the required next
experiments. None can be cited as a research finding.

S-15 is in a different class: it is the program's first controlled confirmation —
n=20 seeds, real matched-displacement directional controls, an off-family LLM
judge, and cross-scale replication — but on a SYNTHETIC single concept ("ocean").
It is nonetheless PROVISIONAL (not external-ready), so it cannot be cited as a
finding. S-16 then takes the same E7 directional question to a REAL external
benchmark (AxBench, 500 concepts) at TWO model scales and finds the effect is weak
and scale-dependent: NEGATIVE/at the floor at 270M, and significant-but-tiny (+0.004,
ordinal-gate-fail, ~97% of the effect captured by a shuffled vector) at 2B. The
synthetic +0.135 directional win does NOT generalize. S-16 corrects and contextualizes
S-15. Both are documented in full below, after S-14.

---

### S-1: The two main vector construction methods point in nearly the same direction

**Testing hypothesis E4** — "Do DiffMean and PCA-top1 produce the same steering
direction (cosine similarity > 0.95)?" DiffMean is fast and simple; PCA-top1 is
more principled but more expensive. If they point in the same direction, DiffMean
suffices for all practical purposes.

**Result on Qwen-2.5-0.5B** (injecting at layer 21, the layer of maximum linear
separability on this model): cosine(DiffMean, PCA-top1) = **0.996** — well above
the pre-registered 0.95 threshold.

**Model / data:** Qwen-2.5-0.5B-Instruct, single "ocean" concept, n=1.

**SCREENING ONLY.** Gate to external: n≥7, multiple behaviors, Gemma reproduction.

---

### S-2: Steering has a "coherence cliff" — small doses tolerable, large doses rapidly destroy output quality

**Testing hypotheses E3 (coherence cliff exists) and N17 (off-shell displacement
predicts the cliff):** Alpha is the multiplier on the steering vector. Does
perplexity rise super-linearly with alpha?

**Result on Qwen-2.5-0.5B at layer 21:** Perplexity relative to baseline at
alpha = 1, 2, 4, 8: +20%, +82%, ×6, ×77. The cliff is sharply super-linear with
a knee around alpha ≈ 1–2. The composite score (all five axes combined) peaks at
alpha ≈ 1. Off-shell displacement rises monotonically alongside perplexity
throughout the sweep, consistent with N17's prediction.

**Note on the behavior proxy:** The "behavior efficacy" number here uses an
activation-projection proxy — the cosine similarity between the steered output
activation and the steering vector. This is circular (same activations used to
build the vector). Perplexity and geometry readings are real; behavior numbers are
not externally valid.

**Model / data:** Qwen-2.5-0.5B, n=1.

**SCREENING ONLY.** Gate to external: real generated-text behavior scoring, real
safety measurement, n≥7 seeds, reproduction on Gemma.

---

### S-3: DiffMean ≈ PCA-top1 equivalence holds on Gemma, not just Qwen

**Testing hypothesis E4 cross-model:** Repeating S-1 on a different model
architecture to check robustness.

**Result on Gemma-3-270m** (injecting at layer 12, the layer of maximum linear
separability on this model): cosine(DiffMean, PCA-top1) = **0.994** — very close
to the 0.996 seen on Qwen and well above the 0.95 threshold. The near-equivalence
holds across both architectures.

**Model / data:** Gemma-3-270m-it, n=1.

**SCREENING ONLY.**

---

### S-4: On the smallest model (270m), steering never improves behavior — it only hurts, at any strength

**Testing hypotheses E3 (coherence cliff) and E27 (small models are more fragile)
and N17 (geometry predicts the cliff):**

**Result on Gemma-3-270m at layer 12:** The behavior proxy score started at 0.50
(baseline) and declined monotonically to 0.22 as alpha increased — behavior never
improved. Perplexity rose sharply from alpha=1 onwards (+65% at alpha=1). This
contrasts with the Qwen-0.5B model, which had a window at alpha ≈ 1 where behavior
improved before quality degraded. The 270m model appears to exit normal behavior
distribution before clean concept injection can occur. Off-shell displacement tracked
perplexity super-linearly throughout, consistent with N17.

**Why this matters:** It suggests there may be a minimum model size below which
activation steering cannot operate cleanly at any safe injection strength. This
motivated the cross-scale comparison in S-8.

**Model / data:** Gemma-3-270m, n=1.

**SCREENING ONLY.**

---

### S-5: The most linearly separable layer is not the best steering layer

**Testing hypothesis E2** — "Does the layer of maximum linear separability of the
contrast set (measured by the Fisher discriminant ratio) give the best steering
efficacy?" Pre-registered prediction: Spearman correlation between Fisher ratio and
steering efficacy across layers ≥ 0.7.

**Result on Gemma-3-270m, sweeping 8 layers at alpha=2:** Spearman(Fisher ratio,
behavior efficacy) = **+0.14** (p=0.74) — far below the predicted ≥ 0.7.

The layer with the highest Fisher ratio is layer 12 (ratio = 30.6), but the best
steering layer by behavior score is layer 16 (efficacy 0.534 vs 0.319 at layer 12;
perplexity 205 vs 322). Maximum linear separability does not predict maximum
steering efficacy on this model.

**Implications:** (a) The earlier coherence-cliff experiments ran at layer 12 —
a suboptimal layer. All subsequent Gemma-270m experiments use layer 16. (b) This
falsification reinforces a broader principle: interpretability (how cleanly a
representation separates categories) does not equal controllability (how well you
can steer with it). Also: layer 12, the most separable layer, turned out to also
be the most fragile to steering (highest perplexity at alpha=4), suggesting that
high linear separability may reflect, rather than confer, brittleness.

**Model / data:** Gemma-3-270m, 8 layers, alpha=2, n=1 per cell.

**SCREENING ONLY — E2 FALSIFIED.**

---

### S-6: A cheap geometric probe predicts incoherence across models and layers

**Testing hypotheses N17 (off-shell displacement predicts incoherence) and N5
(a single universal equation describes the relationship):**

**Result, pooled over 23 steered configurations** (Gemma-3-270m + Qwen-2.5-0.5B,
8 layers, alpha ranging from 1 to 24):

- Spearman(off-shell displacement, log perplexity) = **+0.705** (p = 1.7 × 10⁻⁴)
- Pearson = **+0.899**
- A single linear equation — log PPL = 5.40 + 2.87 × off-shell displacement —
  fits the pooled data with **R² = 0.81**

This is the cheapest possible measure (no text generation required) and it predicts
incoherence across two different architectures and all tested layers.

**Important caveat:** The R²=0.81 was computed within the pooled dataset — both
models mixed in the same regression fit. The rung-3 held-out test (described above)
later showed that the specific equation coefficients do not transfer across model
sizes, even though the directional relationship does. The screening R²=0.81 was
partly an artifact of mixing models in one fit.

**Model / data:** 23 pooled data points, 2 models, 8 layers, alpha range 1–24, n=1 per cell.

**SCREENING ONLY. N17 SUPPORTED; N5 universal-law claim later FALSIFIED at rung-3.**

---

### S-7: Additive steering is gentler than full-vector rotation; two different geometry metrics are needed for the two operation types

**Testing hypothesis E27** ("Does norm-preserving rotation outperform additive
steering on small models?") and **hypothesis N16** (the Cylindrical Representation
Hypothesis — that coherence cost decomposes into two independent components):

**Setup:** On Gemma-3-270m layer 16, comparing additive steering vs full-vector
rotation at small angles (0.05 to 0.5 radians), logging both radial displacement
(change in activation length) and angular displacement (1 − cosine between original
and steered activation direction).

**Result on E27 (rotation vs addition):** At matched behavior score (≈ 0.57),
additive steering at alpha=0.2 gives perplexity = 92.8; full-vector rotation at
matched angle = 0.1 radians gives perplexity = 131.4 — a **+42% coherence
penalty for rotation**. Additive steering holds perplexity 90–99 across its full
safe range; full-vector rotation degrades rapidly from 100 to 11,211 as angle
increases.

**Result on N16 (the two-component geometry predictor):** Radial displacement
(off-shell Δ‖h‖) predicts additive steering's log-PPL well (R²=0.81 from S-6),
but is nearly blind to rotation's damage (Pearson = −0.13 for rotation). Conversely,
angular displacement (1 − cosine between steered and original activation) predicts
rotation's log-PPL at **R²=0.997**, with the equation: log PPL = 4.57 + 43.1 ×
angular displacement. The two components are independent — each governs its own
operation type. This is the empirical confirmation of the Cylindrical Representation
Hypothesis: think of the activation as a point in a cylindrical coordinate system
(radius and angle), and each type of steering moves it along a different cylinder axis.

**Caveat on E27 falsification:** The rotation tested here rotates the entire hidden
state vector toward the steering direction. Published Angular Steering and Selective
Steering methods rotate only within a carefully chosen 2D plane (a much more surgical
operation). This result falsifies full-vector rotation, not necessarily those
selective-plane methods. Selective-plane rotation is queued as a future experiment.

**Model / data:** Gemma-3-270m, layer 16, alpha range 0.05–0.5, n=1 per cell.

**SCREENING ONLY. E27 FALSIFIED for full-vector rotation; N16 SUPPORTED.**

---

### S-8: A safe steering window emerges with model scale — the 270m model is uniquely fragile

**Testing hypothesis E27 cross-scale** — does the ability to steer cleanly emerge
as models get larger?

**Result across three models on the coherence cliff:**

| Model | Param count | PPL increase at alpha=1 | PPL increase at alpha=2 | Behavior at alpha=1 | Safe window? |
|---|---|---|---|---|---|
| Gemma-3-270m | 270M | +65% | +257% | 0.44 (declining from baseline) | None — most fragile |
| Qwen-2.5-0.5B | 500M | +20% | +82% | 0.69 (peaked above baseline) | Yes |
| Gemma-3-1B | 1B | +41% | +181% | 0.65 (peaked above baseline) | Yes — least fragile |

The 270m model has no steering strength at which behavior improves before coherence
collapses. Both larger models have a window (around alpha=1) where behavior peaks
above baseline before the cliff. The 1B model's cliff is the gentlest of the three.

Also confirmed on Gemma-3-1B: cosine(DiffMean, PCA-top1) at layer 18 = **0.9945**,
confirming the near-equivalence of the two construction methods on a third model.

**Model / data:** Three models, two architectures, n=1 per cell.

**SCREENING ONLY. E27 cross-scale mechanism SUPPORTED; E4 confirmed on third model.**

---

### S-9: Expressing steering strength as a fraction of activation magnitude eliminates cross-prompt variability and resolves the DiffMean vs PCA-top1 comparison

**Testing hypotheses E7 (norm-relative alpha is more consistent) and E36 (DiffMean
and PCA-top1 are equivalent when compared at matched displacement):**

**The problem with absolute alpha:** The earlier steering coefficient "alpha" was
multiplied onto the raw steering vector. But activation vector lengths vary
substantially across prompts and layers, so the same alpha produces very different
actual displacements in different contexts. Also, DiffMean's raw vector has a length
roughly 10× larger than the unit-normalized PCA-top1 vector, so comparing them at
the same raw alpha is meaningless — it is like comparing distances in miles to
distances in kilometers without converting.

**The fix:** Define alpha as a fraction of the activation's current length — alpha=0.1
means "push the activation 10% of its current length in the steering direction."
This is called "relative steering."

**Results on Gemma-3-270m, layer 16, relative steering:**

| Alpha (fraction of activation length) | DiffMean behavior score | PCA-top1 behavior score | Perplexity |
|---|---|---|---|
| 0.02 (2% push) | 0.532 | 0.532 | 92 |
| 0.10 (10% push) | 0.614 | 0.591 | 132 |
| 0.20 (20% push) | 0.504 | 0.475 | 245 |
| 0.40 (40% push) | 0.319 | 0.304 | 1,623 |

Behavior peaks at **alpha ≈ 0.10** (10% displacement), then declines as coherence
breaks down. The cliff shape is clean and stable. DiffMean and PCA-top1 produce
results within 0.02 behavior score and 8% perplexity of each other at every matched
alpha — effectively identical steering. The earlier apparent differences between the
methods were entirely due to mismatched raw-alpha comparisons.

**Scale invariance:** On Gemma-3-1B, the cliff knee sits at approximately alpha ≈
0.05–0.10 as well — the safe window in relative-displacement terms is roughly the
same fraction of activation length on both model sizes.

**Model / data:** Gemma-3-270m (and cross-scale check on Gemma-3-1B), n=1 per cell.

**SCREENING ONLY. E7 SUPPORTED; E36 SUPPORTED and resolved.**

---

### S-10: DiffMean and PCA-top1 equivalence holds across four different behavior concepts

**Testing hypothesis E4 cross-behavior** — is the near-equivalence specific to one
concept, or does it generalize?

**Results on Gemma-3-270m, four concepts (each tested at its own most-separable layer):**

| Concept | cosine(DiffMean, PCA-top1) |
|---|---|
| Anger | 0.995 |
| Formality | 0.996 |
| Happiness | 0.999 |
| Ocean (abstract concept) | 0.997 |

All four are well above the pre-registered 0.95 threshold. DiffMean and PCA-top1
produce nearly identical directions regardless of which behavior you are steering
toward, and across three different model architectures and sizes.

**Model / data:** Gemma-3-270m, n=1 per concept.

**SCREENING ONLY. E4 SUPPORTED across behaviors.**

---

### S-11: The coherence cliff appears for every behavior tested, but concrete emotions steer more strongly than abstract concepts

**Testing hypotheses E3 (coherence cliff is general, not concept-specific) and
N17 (off-shell displacement predicts incoherence as a function of steering strength):**

**Results on Gemma-3-270m, layer 16, four behaviors, relative steering (10% displacement = alpha=0.1):**

Perplexity rises with steering strength for every concept tested (anger: 90 → 293;
formality: 90 → 602; happiness: 90 → 246; ocean: 90 → 370). The coherence cliff
is not a quirk of one concept — it appears universally.

Behavior efficacy at the same steering strength (alpha = 10% displacement) varies
by concept: anger 0.77 > happiness 0.63 > ocean 0.53 ≈ formality 0.53. Concrete
emotions steer more strongly than abstract or formal register concepts, likely
because emotional states are more localized and distinct in activation space.

**An important scope clarification for N17:** When comparing different behavior
concepts at a fixed steering strength (instead of comparing different strengths
within one behavior), the correlation between off-shell displacement and log-PPL
is weak (Spearman +0.17, n=8). N17 is primarily an alpha-sweep relationship — it
predicts how coherence degrades as you increase steering strength within a
single configuration, not how different concepts compare to each other at the
same strength.

**Model / data:** Gemma-3-270m, 4 concepts, layer 16, n=1 per cell.

**SCREENING ONLY. E3 cross-behavior SUPPORTED; N17 scope clarified.**

---

### S-12: Two behavior vectors can be stacked simultaneously with no interference, even when they are somewhat correlated

**Testing hypotheses E17 (near-orthogonal stacking retains both behaviors fully)
and E10 (concept vectors are near-orthogonal to each other):**

**Pairwise cosine similarities between the four steering vectors** (anger, formality,
happiness, ocean) on Gemma-3-270m layer 16:

Most pairs are near-orthogonal (|cosine| < 0.3): anger↔formality = −0.18,
formality↔ocean = −0.10. But anger↔happiness = **+0.48** — meaningfully correlated
because both are high-arousal emotions that activate similar circuits.

**Stacking test:** Applying anger + happiness simultaneously, each at 10%
displacement, retains **101% of anger's solo behavior and 110% of happiness's solo
behavior** — zero interference, and even slight improvement. The shared emotional
component apparently reinforces rather than dilutes each behavior.

**What this means:** Near-orthogonality between steering vectors is not required
for clean simultaneous steering — at least not at moderate strengths and small vector
counts. Two behavior vectors compose nearly additively.

**Model / data:** Gemma-3-270m, layer 16, 2-vector stack, n=1.

**SCREENING ONLY. E17 SUPPORTED; E10 PARTIAL (one exception to near-orthogonality,
but stacking still works in that exception).**

---

### S-13: The steering direction is preserved across adjacent layers; the behavior subspace is not especially low-dimensional or sparse

**Testing three hypotheses:** E40 (parallel transport — is the same direction
maintained as you move through network depth?), E28 (is the behavior subspace
low-rank, i.e., compressible to 2–3 dimensions?), and E35 (is the behavior vector
sparse, concentrated in 10% of coordinates?):

**E40 result (cross-layer cosine similarities for the DiffMean anger direction on
Gemma-3-270m):** Layer 12→14 = 0.85, layer 14→16 = 0.75, layer 16→17 = 0.90.
The same steering direction is approximately maintained as you traverse the network
depth, consistent with a single underlying direction being "parallel transported"
through layers by the network's weight matrices. This explains why single-layer
injection can be effective even when not targeting the precise optimal layer.

**E28 result:** The top 3 singular-value-decomposition dimensions of the set of
positive-minus-negative activation differences explain only **66%** of variance —
far below the pre-registered >90% threshold. The behavior direction is not
concentrated in a 2–3 dimensional subspace. It is spread across more dimensions
than the "low-rank" hypothesis predicted.

**E35 result:** Retaining only the top 10% of coordinates (by magnitude) of the
steering vector and zeroing the rest yields **77%** of the original behavior score —
below the pre-registered 85% threshold. The vector is moderately sparse but not
as sparse as predicted.

**Model / data:** Gemma-3-270m, anger concept, n=1.

**SCREENING ONLY. E40 SUPPORTED; E28 FALSIFIED; E35 PARTIAL.**

---

### S-14: Cumulative steering displacement from stacked vectors governs collapse; interference between stacked vectors does not follow a simple geometric rule

**Testing hypotheses E22 (cumulative norm budget governs collapse across stacked
vectors) and E18 (interference grows proportionally to the geometric overlap of
the stacked vectors, measured by the Gram matrix):**

**E22 result — cumulative displacement and PPL collapse:**
Scaling the total steering displacement across 4 stacked concept vectors on
Gemma-3-270m layer 16 produces super-linear perplexity collapse:

| Total displacement alpha | Perplexity |
|---|---|
| 0.05 (5% of activation length, 4 vectors stacked) | 138 |
| 0.10 | 182 |
| 0.20 | 410 |
| 0.40 | 4,518 |

The coherence cliff applies to the cumulative push from all stacked vectors combined.
Each individual vector contributes to a shared "norm budget," and when the total
crosses the budget threshold, coherence collapses. This is consistent with the
N5 norm-budget hypothesis.

**E18 result — interference and Gram matrix:** Stacking 1 → 4 vectors retains
100% / 86% / 85% / 94% of mean per-behavior efficacy (composition works —
all above 85%). However, the retention is NOT monotonically related to the
off-diagonal Gram matrix mass (which increases from 0 → 0.37 → 1.80 → 3.07 as
more vectors are stacked). The strict pre-registered prediction that
"interference ∝ Gram off-diagonal mass" is not supported.

**Model / data:** Gemma-3-270m, layer 16, 4 concepts, n=1.

**SCREENING ONLY. E22 SUPPORTED; E18 PARTIAL.**

---

### S-15: E7 relative-steering is a real, scale-replicated DIRECTIONAL effect — but not yet EXTERNAL-READY

**This is the program's first controlled, multi-seed, off-family-judged, cross-scale
result.** Every prior observation (S-1 through S-14) used a single run (n=1) and an
activation-projection behavior proxy — a circular measure that scores how well a
steered output's activation aligns with the very vector used to steer it. S-15 was
measured the way the earlier 113 single-seed screens were not: with real controls,
n=20 seeds, an independent off-family judge, and replication across two model scales.
It is the strongest and most rigorous observation in this program. It is still
PROVISIONAL — not external-ready — for the two precise reasons given at the end of
this entry.

---

**What hypothesis this tests:** "E7 — Norm-relative alpha." The pre-registered
prediction is that expressing steering strength as a fraction of the activation's
own length (rather than as a raw absolute coefficient) gives a clean, norm-independent
steering effect. The controlled test here asks the sharper question that no proxy run
could answer: **at a fixed displacement magnitude, does the real concept direction
actually steer behavior better than a content-free direction of the same length?**

**Every term defined inline (assume the reader cross-references nothing):**

- **relative_add (the operation under test).** Ordinary additive steering adds
  `alpha × v` to a hidden-layer activation `h`, where `v` is the steering vector and
  `alpha` is a raw coefficient. The trouble: activation lengths `||h||` vary a lot
  across prompts and layers, so the same `alpha` produces a wildly different actual
  push in different contexts. `relative_add` fixes this by (1) normalizing the
  steering vector to unit length (a pure direction), and (2) setting the push size to
  `alpha × ||h||` along that unit direction. So `alpha = 0.1` means "push the
  activation 10% of its own current length in the steering direction," regardless of
  prompt or layer. Crucially, because the direction is always a unit vector, **every
  condition compared below receives the identical displacement magnitude `alpha × ||h||`
  — only the DIRECTION differs.** This is what makes the controls fair.

- **The DiffMean concept direction (the "real" condition).** Take the average
  hidden-layer activation over a set of positive-class examples (here, sentences about
  the "ocean" concept), subtract the average over negative-class examples, and use that
  difference, normalized to unit length, as the steering direction.

- **The shuffled-label control (the PRIMARY directional control).** Take the exact
  same pooled set of activations used to build the real direction, but randomly
  re-partition them into two groups whose labels have nothing to do with "ocean," then
  compute DiffMean on that random split. This destroys the semantic contrast while
  keeping the data, the dimensionality, the procedure, and — via relative_add — the
  displacement magnitude all identical. If the real direction beats this control, the
  effect is attributable to the *meaning* in the contrast, not to the mechanics of
  pushing the activation around. This is the matched-coherence directional control.

- **The random-direction control (a SECONDARY control).** A random unit direction,
  unrelated to any data, pushed at the same displacement magnitude. This is a weaker,
  sanity-check control: a content-free push of equal size.

- **The off-family LLM judge (the behavior instrument).** The generator is Gemma; the
  judge is Google's Gemini (gemini-2.5-flash-lite), a DIFFERENT model family. This
  matters because the project's own audits disclose a same-model-family circularity
  risk: if the judge and the generator share a family, the judge may reward
  family-typical text rather than genuine on-concept behavior. Using an off-family
  judge breaks that circularity. The judge rates each generated text from 0 to 10 on
  behavior (does the text express the ocean concept?) AND, separately, on coherence
  (is the text fluent and sensible?), at temperature 0 (deterministic), and the
  ratings are cached. **Judge validation:** real ocean prose scores behavior 8/10;
  off-topic tax text scores 0/10; and keyword-stuffing ("ocean ocean sea sea") scores
  only 2/10 — i.e., the judge is NOT fooled by keyword soup, unlike the old lexicon
  proxy. Behavior is reported below as judge_behavior/10, rescaled into [0,1].

- **The four rigor legs (the four-part contract applied here).** A directional win is
  asserted only if all four hold: (1) a **paired Wilcoxon signed-rank test** on the
  per-seed real-minus-control deltas gives p < 0.05; (2) a **bootstrap 95% confidence
  interval** on the paired delta (≥10,000 resamples) excludes zero; (3) **Holm-Bonferroni
  correction** across the family of alphas tested {0.05, 0.10, 0.15} still rejects the
  null (guards against picking the one lucky alpha); and (4) the **ordinal gate** — the
  WORST real seed must beat the BEST control seed (ensures the win is not driven by a
  few lucky seeds at the distribution's edge). An **extraction-stability gate** is also
  reported: the bootstrap cosine of the DiffMean direction to the full-data direction,
  which checks the concept direction itself is stable under resampling.

**Design specifics.** Behavior concept: "ocean." n=20 seeds with stochastic generation
(temperature 0.8); each seed's behavior is the mean of the judge's scores over 4
evaluation prompts. Two scales: gemma-3-270m-it (injection layer 16) and gemma-3-1b-it
(injection layer 18). Run by `scripts/confirm_e7.py`, composing
`src/steering/controls.py` (the matched-displacement controls),
`src/steering/stats.py` (the rigor_report contract), and `src/steering/judge.py` (the
off-family judge). The primary comparison is **real vs shuffled-label** at the knee
alpha = 0.10.

**Results at the knee (alpha = 0.10), real concept direction vs the shuffled-label control:**

| Quantity | 270M (exp#116) | 1B (exp#117) |
|---|---|---|
| Real behavior (judge/10, in [0,1]) | 0.730 | 0.549 |
| Shuffled-label behavior | 0.595 | 0.453 |
| Paired delta (real − shuffled) | **+0.135** | **+0.096** |
| Bootstrap 95% CI on the delta | **[+0.084, +0.184]** | **[+0.025, +0.163]** |
| Paired Wilcoxon p | **0.0004** | **0.014** |
| Holm-rejected across alpha family {0.05,0.10,0.15} | TRUE | TRUE |
| Extraction-stability (bootstrap cosine to full-data direction) | 0.94 | 0.92 |
| Matched-coherence at the knee? | TRUE (real 0.614 vs shuffled 0.715) | **FALSE** (real 0.345 vs shuffled 0.699) |
| Ordinal gate (worst real seed > best shuffled seed)? | **FALSE** | **FALSE** |
| Secondary: real vs random-direction, paired Wilcoxon p | 0.0001 | 0.0004 |

On the secondary random-direction control, the random direction at the same
displacement magnitude does not merely lose — it collapses coherence to ~0.002 with
perplexity around 52,000 (effectively gibberish), confirming that a content-free push
of equal size destroys the output while the real direction does not.

**Cross-scale conclusion.** The real concept direction SIGNIFICANTLY beats the
matched-displacement shuffled-label control on BOTH scales — Holm-corrected on both,
with bootstrap confidence intervals excluding zero on both. The directional effect of
relative-steering therefore REPLICATES across model scale. This is the program's first
cross-scale, controlled, off-family-judged directional win.

**Honest verdict — PROVISIONAL, not a null.** This is NOT external-ready, but for an
INFORMATIVE reason, not because the effect failed. The effect is statistically robust
and scale-replicated. It falls short of ONLY the two strictest legs of the four-part
contract:
1. **The ordinal gate fails on both scales** — the worst real seed does not exceed the
   best shuffled seed.
2. **Matched-coherence fails on 1B** — at the knee, the real direction trades away more
   coherence (0.345) than the shuffled control (0.699).

The mechanistic reason the ordinal gate fails is itself informative: **the shuffled-label
control is unexpectedly strong.** A random split of a small, concept-dominated contrast
set still recovers roughly 0.45–0.60 of the ocean direction — because the pooled
activations are so dominated by the concept that even a random re-partition picks up a
large fraction of it. As a result the per-seed distributions of the real and shuffled
conditions overlap at their extremes, so the worst real seed and the best shuffled seed
cross. On 1B, the real direction additionally spends more coherence than the shuffled
control at the knee, so the matched-coherence leg fails there (alpha ≈ 0.05 is more
coherence-matched on 1B). Neither failure indicates the effect is absent; both indicate
the *control* and the *knee tuning* need refinement to separate the distributions cleanly.

**Why this result matters — the proxy-vs-judge contrast (the measurement-validity
lesson).** The SAME experiment, run earlier with the OLD activation-projection lexicon
proxy, produced a noise-level result: on 270m (exp#114) the proxy gave a real-minus-shuffled
delta of only +0.022, and on 1B (exp#115) the proxy gave shuffled BEATING real by −0.019.
The proxy's conclusion was that the effect "does not replicate across scale." That
conclusion was an ARTIFACT of an invalid instrument. Swapping in the validated off-family
judge amplified the behavior signal roughly 6× and revealed that the effect DOES replicate
across scale and DOES beat the matched control. This is concrete, within-program evidence
for why an unvalidated proxy must never back a claim: the same data yielded opposite
scientific conclusions under the two instruments, and only the validated instrument
(the one not fooled by keyword soup) is trustworthy.

**Next steps to push E7 toward EXTERNAL-READY:**
1. A cleaner null control than label-shuffling on a tiny, concept-dominated set — e.g.
   an orthogonalized direction (the real direction with its concept component removed)
   or a genuine random-concept direction — so the control no longer leaks ~0.5 of the
   real direction.
2. More evaluation prompts and more seeds to tighten the per-seed distributions and
   close the ordinal separation.
3. Per-scale knee tuning — alpha ≈ 0.05 is more coherence-matched on 1B than the
   0.10 knee used for the primary comparison.
4. Add a 2B (or larger) scale to extend the cross-scale ladder.
5. Calibrate the off-family judge against human labels.

**Model / data:** gemma-3-270m-it (layer 16) and gemma-3-1b-it (layer 18), concept
"ocean", n=20 seeds, off-family Gemini judge, exp#116/117.

**PROVISIONAL — E7 SUPPORTED (directional, scale-replicated) at rung-3, NOT external-ready.**
This is the closest the program had come to an external-ready finding on synthetic data;
the external-ready count remains zero. S-16 below shows this synthetic +0.135 win does NOT
generalize to the real AxBench benchmark: across two scales the real-vs-shuffled advantage
is at best +0.004 (at 2B), and NEGATIVE at 270M.

---

### S-16: On the real AxBench benchmark, E7 DiffMean steering's advantage over a shuffled control is SCALE-DEPENDENT and WEAK — NEGATIVE/at-the-floor at 270M, significant-but-tiny (+0.004) at 2B — and the synthetic-ocean +0.135 win did not generalize

**This is the program's first evaluation on a REAL, external, published benchmark,
run at TWO model scales, and it delivers a WEAK/negative result that corrects S-15.**
Every prior observation — including the strongest one (S-15) — used data this project
authored itself: a single synthetic concept ("ocean") with hand-built contrast
sentences. S-16 replaces ALL of that with AxBench, an external benchmark built by
independent authors, and asks the exact same question S-15 asked — does the real
DiffMean concept direction beat a matched-displacement label-shuffled control? The same
experiment was run on two model scales: gemma-3-270m-it (layer 16, exp#118) and
google/gemma-2-2b-it (layer 20, the scale and layer AxBench's data targets, exp#119).
On the real benchmark the answer is informative and unflattering: at 270M the real
direction is significantly WORSE and both conditions are at the floor (the model cannot
express the concepts at all); at 2B the concepts become expressible and the real
direction significantly beats shuffled, but only by a SUBSTANTIVELY TINY margin (+0.004
on a 0–1 scale) that fails the strict ordinal gate and is dwarfed by the generic
(direction-agnostic) displacement effect. This is exactly the kind of correction a real
benchmark exists to deliver, and it is a SUCCESS of the process: moving off self-authored
data caught a claim that did not generalize.

---

**What hypothesis this tests:** "E7 — Norm-relative alpha," and specifically the sharp
directional question carried over from S-15: at a fixed displacement magnitude, does the
real concept direction steer behavior better than a content-free (label-shuffled) direction
of the same length? S-15 answered YES on the synthetic "ocean" concept (provisionally,
+0.135). S-16 re-asks it on 500 real, abstract AxBench concepts at two model scales (270M
and 2B).

**Every term defined inline (assume the reader cross-references nothing):**

- **AxBench (the real benchmark).** AxBench (Wu, Zhong, et al., ICML 2025,
  arXiv:2501.17148; dataset `pyvene/axbench-concept500`) is a published, external
  benchmark for evaluating representation-steering and feature-control methods. It
  supplies, for 500 distinct concepts: (1) the concepts themselves; (2) the contrast
  text (positive vs negative outputs) from which a DiffMean steering vector is built;
  and (3) the held-out evaluation instructions used to test whether a steered model
  actually expresses the concept. Critically, in this experiment AxBench replaced ALL
  of the prior synthetic/hand-authored data — the concepts, the contrast text that
  builds the vector, AND the eval instructions are all AxBench's, not ours.

- **The off-family LOCAL judge (the behavior instrument).** Behavior was scored by
  Qwen2.5-7B-Instruct (4-bit, run locally on the 4090) at BOTH scales. The generator is
  Gemma; the judge is Qwen — a DIFFERENT model family, so there is no same-family
  circularity (the audit-disclosed risk that a judge rewards family-typical text rather
  than genuine on-concept behavior). The judge applies AxBench's own rubric — a concept
  score (0–2) and a fluency score (0–2) — fluency-gated and rescaled to [0,1].

- **The judge AUC-0.68 disclosure (a weak but unbiased instrument).** The judge was
  validated against AxBench's own ground-truth labels: does its concept score separate
  AxBench's labeled-positive from labeled-negative outputs? It scored ROC-AUC = 0.68 —
  BELOW the 0.80 "trustworthy" bar this project uses. This is disclosed openly so
  readers can weigh it. A diagnostic showed the sub-0.80 AUC largely reflects AxBench's
  subtle positives and noisy labels (e.g. Python outputs labeled positive for a "C/C++"
  concept, which the judge correctly scores 0) rather than judge incompetence. The key
  point: a weak-but-UNBIASED judge still yields a VALID paired real-vs-shuffled comparison
  — label noise WIDENS the confidence interval, it does not BIAS the sign of the delta.
  With n=500 concepts the paired test remains valid; the AUC is reported so the strength
  of the instrument is transparent.

- **The shuffled-label control (the matched-displacement directional control).** For
  each concept, v_real = DiffMean(AxBench positive outputs vs negative outputs) at the
  injection layer (layer 16 on 270M, layer 20 on 2B); v_shuf = DiffMean of a
  matched-displacement, shuffled-label control (the same activations re-partitioned by
  random labels, then displacement-matched). Steering uses relative_add at the knee
  (alpha=0.1), so v_real and v_shuf push the activation by the identical magnitude — only
  the DIRECTION differs. If the real direction beats the shuffled one, the effect is
  attributable to the concept's meaning, not the mechanics of pushing the activation
  around. The fraction of the steering effect that v_shuf already captures is the measure
  of how much of steering is GENERIC (direction-agnostic) rather than concept-specific.

- **The concept as the replication unit (n=500).** The independent replication unit here
  is the CONCEPT, not a bootstrap resample and not a generation seed. There are 500
  independent concepts. Per concept: build v_real and v_shuf, steer the model with
  relative_add at alpha=0.1 on 10 AxBench eval instructions, and judge each steered output.
  The real-vs-shuffled comparison is then paired across the 500 concepts (paired Wilcoxon
  + bootstrap CI + ordinal gate via `stats.rigor_report`).

- **The floor effect (270M only).** "At the floor" means both the real and the shuffled
  conditions score near the bottom of the 0–1 behavior scale (~0.05). The 270M model barely
  expresses AxBench's abstract concepts under steering AT ALL — neither the real direction
  nor the control produces meaningful concept expression. At 2B this floor is escaped: both
  conditions score ~0.135 (3× higher), so the concepts ARE expressible and the real-vs-
  shuffled question becomes meaningful.

**Design specifics.** Same experiment at two scales. (270M, exp#118) model gemma-3-270m-it,
injection layer 16. (2B, exp#119) model google/gemma-2-2b-it, injection layer 20 — matching
the 2b/layer-20 model AxBench's data was built from. Both: operation relative_add, alpha=0.10
(the knee); all 500 concepts from `pyvene/axbench-concept500`; 10 AxBench held-out eval
instructions per concept; judge Qwen2.5-7B-Instruct (4-bit, local, off-family), AxBench
concept(0–2)+fluency(0–2) rubric, fluency-gated to [0,1]; primary comparison = real DiffMean
direction vs matched-displacement shuffled-label control, paired across the 500 concepts.

**Results — 270M (exp#118) vs 2B (exp#119), side by side:**

| Quantity | 270M (exp#118, layer 16) | 2B (exp#119, layer 20) |
|---|---|---|
| Mean behavior, REAL direction | 0.0459 | 0.1382 |
| Mean behavior, SHUFFLED control | 0.0562 | 0.1342 |
| Paired delta (real − shuffled) | **−0.0103** | **+0.0040** |
| Bootstrap 95% CI on the delta | **[−0.0142, −0.0066]** (excludes 0, NEGATIVE) | **[+0.0003, +0.0077]** (barely excludes 0, POSITIVE) |
| Paired Wilcoxon p | **1.41 × 10⁻⁶** | **0.0106** |
| Replication unit | concept (n=500) | concept (n=500) |
| Ordinal gate | FALSE | FALSE |
| external_ready | FALSE | FALSE |
| Absolute concept-expression | at the FLOOR (~0.05/1.0) | expressible (~0.135/1.0, 3× higher) |
| Shuffled control's share of the steering effect | — (both at floor) | ~97% (0.1342 / 0.1382) |
| Judge validation (AxBench ground truth) | ROC-AUC = 0.68 (below the 0.80 bar — disclosed) | same judge, AUC 0.68 (disclosed) |

**The sign-bug correction (270M).** On the 270M run the driver's auto-label initially read
"DIRECTIONAL" — that was a SIGN-BLIND bug: it checked statistical significance and that the
CI excluded zero, but it did NOT check the SIGN of the delta. The CORRECTED 270M verdict is
NEGATIVE / NULL. The real DiffMean direction does NOT beat the shuffled control; it is in
fact slightly — but, given n=500, significantly — WORSE, with both conditions at the floor.
The significant negative delta is statistically real but substantively tiny (0.01 on a 0–1
scale), and both conditions essentially fail to express the concept.

**The 2B reading — DIRECTIONAL but substantively TINY.** At 2B the concepts become
expressible (both conditions ~0.135) and the real direction significantly BEATS shuffled.
But every quantitative qualifier says the effect is weak: the delta is +0.004 on a 0–1 scale
(~3% relative edge); the 95% CI [+0.0003, +0.0077] BARELY clears zero; the ordinal gate FAILS
(the worst real concept does not beat the best shuffled concept); and a label-shuffled vector
already captures ~97% of the steering effect (0.1342 of 0.1382). The honest reading is
therefore: at 2B the real concept direction carries a real but WEAK concept-specific signal,
and the overwhelming majority of the steering effect is GENERIC — produced by the
displacement itself, not by its direction.

**Honest headline — the cross-scale synthesis.** On the real AxBench benchmark, DiffMean
relative-steering's advantage over a matched-displacement shuffled-label control is
SCALE-DEPENDENT and WEAK. At 270M it is NEGATIVE / at the floor (the model can't express the
concepts at all). At 2B concepts become expressible and a SMALL, statistically-significant
real-direction advantage appears (+0.004, p=0.011) — but it is tiny, fragile (CI barely
excludes 0), fails the strict ordinal gate, and is dwarfed by the generic-displacement effect
(shuffled gets ~97% of it). This is a FAR cry from the +0.135 directional effect the EASY
synthetic "ocean" concept suggested (S-15). The real benchmark shows that on 500 abstract
concepts, the real concept direction carries only a WEAK concept-specific signal; most of the
steering effect is generic, not direction-specific. This aligns with AxBench's own published
finding that steering is hard.

**Methodology lesson (stated plainly).** The synthetic single-concept "ocean" evaluation
massively OVERSTATED the effect (+0.135). Only a real benchmark + a matched control + a
population of 500 concepts, run at two scales, revealed how weak and scale-dependent the true
effect is. A controlled directional win on a single, self-authored, concept-dominated contrast
set is NOT evidence that the method works on real, abstract, diverse concepts — internal
statistics can be clean and still mislead. This is the most rigorous evaluation in the program,
and it is, honestly, a WEAK/negative result for DiffMean steering. There are STILL zero
external-ready results.

**Caveats / next steps (stated prominently):**

1. **The judge is weak (AUC 0.68).** Disclosed and unbiased: n=500 keeps the paired
   comparison valid (label noise WIDENS the CI, does not bias the SIGN), but a stronger or
   human-calibrated judge would tighten the estimate and could shift the small 2B effect.
2. **Alpha=0.1 was not per-concept-tuned.** A single global knee was used for all 500
   concepts at both scales; per-concept alpha tuning could shift the small 2B effect (in
   either direction). The qualitative picture — weak, scale-dependent, mostly-generic — is
   robust to the n=500 paired test, but the precise size of the 2B edge is not pinned down.
3. **The synthetic "ocean" win (S-15) is now contextualized as an EASY-concept artifact
   that did not generalize.** S-16 across two scales is the real-benchmark verdict on E7 and
   SUPERSEDES the synthetic S-15.

**Cross-reference:** S-16 corrects and contextualizes **S-15** (the synthetic "ocean"
relative-steering directional win, +0.135). S-15's effect was real within its synthetic
setting but did not survive the move to a real benchmark at either scale.

**Model / data:** gemma-3-270m-it (layer 16, exp#118) and google/gemma-2-2b-it (layer 20,
exp#119); both relative_add, alpha=0.10; AxBench `pyvene/axbench-concept500`, all 500 concepts,
10 eval instructions each; off-family local judge Qwen2.5-7B-Instruct (4-bit), AxBench
concept+fluency rubric (judge AUC 0.68).

**WEAK / SCALE-DEPENDENT — E7's directional effect on real AxBench is NEGATIVE/at-the-floor at
270M and significant-but-tiny (+0.004, ordinal-fail, ~97%-generic) at 2B.** The synthetic-ocean
+0.135 win did not generalize. This supersedes the synthetic S-15 as the real-benchmark verdict
on E7. The external-ready count remains zero.

---

### S-17: E3 alpha-coherence cliff SUPPORTED on real AxBench (2B, layer 20) — behavior and coherence both peak at alpha~0.10 and collapse super-linearly past alpha~0.20

**This is the program's first confirmation of the E3 alpha-coherence cliff on a REAL external
benchmark.** S-2, S-4, S-8, S-9, and S-11 established the cliff on synthetic single-concept
data and across multiple model scales, using an activation-projection behavior proxy. S-17 retests
the same shape question — does output quality collapse super-linearly once steering strength passes
a threshold, and does a safe operating window exist around alpha~0.10? — on real AxBench concepts,
measured by an off-family local judge that scores both behavior and coherence from generated text,
not activations. The cliff is SUPPORTED, and crucially this is a POSITIVE generalization result in
contrast to S-16 (E7's directional claim, which did not generalize): the program's
geometry/coherence findings carry over to a real benchmark even when the direction-specificity
finding does not.

---

**What hypothesis this tests:** "E3 — Alpha coherence cliff." The pre-registered prediction is
that the steering coefficient alpha has a behavior-specific coherence cliff: below a threshold,
output quality remains acceptable; above it, perplexity rises super-linearly. The practical
corollary is that a safe operating window exists around the peak-behavior alpha, and that steering
past it costs coherence faster than it gains behavior.

**Secondary cross-reference — N17 (off-shell displacement predicts incoherence):** The cliff's
mechanism in the program's geometry framework is that large alpha pushes the activation vector off
the sphere of its natural activation distribution (large off-shell displacement Δ‖h‖), which
predicts incoherence per N17 (confirmed at rung-3 on WikiText-2, see above). S-17 is a
downstream behavioral confirmation: the incoherence N17 predicts from geometry is now also
observed via a real off-family fluency judge on real AxBench outputs.

**Every term defined inline:**

- **relative_add (the steering operation).** The steering vector is normalized to unit length and
  the push size is set to `alpha × ‖h‖` — the activation's own length — so alpha is a
  fractional displacement. Alpha=0.10 means "push the activation 10% of its current length in the
  steering direction." This normalization makes alpha comparable across prompts and layers, as
  established in the earlier E7/E36 work (S-9).

- **AxBench (the real benchmark).** AxBench (Wu, Zhong et al., ICML 2025, arXiv:2501.17148;
  dataset `pyvene/axbench-concept500`) is a published, external benchmark for evaluating
  representation-steering and feature-control methods. It supplies, for 500 distinct concepts:
  the concepts themselves; contrast text (positive vs negative outputs) from which a DiffMean
  steering vector is built; and held-out evaluation instructions used to test whether a steered
  model actually expresses the concept. In this experiment, 30 of those 500 concepts were used
  (a curve-tracing sample, not a population-size test), with 8 AxBench eval instructions per
  concept.

- **v_real = DiffMean(AxBench positive vs negative).** For each of the 30 concepts, the steering
  vector is built from the difference of mean hidden-layer activations over the AxBench contrast
  text at injection layer 20. This uses AxBench's own contrast data, not hand-authored synthetic
  sentences. The method is the same relative_add DiffMean used throughout the program; only the
  source data is now AxBench's.

- **The off-family local judge (behavior and coherence instrument).** Qwen2.5-7B-Instruct (4-bit,
  run locally on the 4090) scored each steered output using AxBench's own rubric: a concept score
  (0–2) converted to behavior in [0,1], and a fluency score (0–2) converted to coherence in [0,1].
  The generator is Gemma; the judge is Qwen — different model families, so there is no same-family
  circularity. Both behavior AND coherence are scored from the same generated text by the same
  judge call, which is the key design feature: it allows direct observation of the joint
  (behavior, coherence) curve as alpha varies. **Judge validation:** the judge's concept score
  separates AxBench labeled-positive from labeled-negative outputs at ROC-AUC = 0.68 — below
  the 0.80 "trustworthy" bar. This is disclosed openly. The AUC is WEAK but UNBIASED (noise widens
  confidence in the curve estimate; it does not bias the direction of the cliff). The sub-0.80
  AUC largely reflects AxBench's subtle positives and noisy ground-truth labels rather than judge
  incompetence (see the parallel disclosure in S-16).

- **The alpha grid and the curve shape.** Six values of alpha were tested: 0.02, 0.05, 0.10,
  0.20, 0.40, 0.80. For each value, all 30 concepts × 8 eval instructions were steered and judged;
  the mean behavior score and mean coherence score across those 240 outputs constitute one point on
  the curve. The curve is the observable that E3 predicts: a behavior peak near a moderate alpha,
  then super-linear degradation of coherence past the cliff.

**Design specifics.** Model: google/gemma-2-2b-it (the program's standard evaluation model).
Injection layer: 20 (the same 2b/layer-20 setup used in exp#118/119 and matching AxBench's own
data targets). Operation: relative_add. Concepts: 30 of AxBench's 500, sampled for the alpha-curve
tracing. Eval instructions: 8 per concept (AxBench held-out set). Judge: Qwen2.5-7B-Instruct
4-bit, AxBench concept(0–2)+fluency(0–2) rubric, each rescaled to [0,1]. Experiment tag:
E3-axbench-gemma-2-2b-it. Experiment number: exp#120.

**The alpha-behavior-coherence curve (30 concepts × 8 eval instructions, mean across all):**

| Alpha | Behavior (judge, [0,1]) | Coherence (judge fluency, [0,1]) | Note |
|-------|------------------------|----------------------------------|------|
| 0.02 | 0.150 | 0.619 | Below peak; some steering, good coherence |
| 0.05 | 0.160 | 0.660 | Rising toward peak |
| **0.10** | **0.163 (PEAK)** | **0.677 (PEAK)** | **Safe window: behavior maximal, coherence still high** |
| 0.20 | 0.148 | 0.665 | Behavior declined; coherence roughly flat — still safe |
| 0.40 | 0.133 | 0.458 | **CLIFF: coherence drops super-linearly** |
| 0.80 | 0.052 | 0.106 | Collapse: both behavior and coherence near floor |

**Reading the curve.** Behavior peaks at alpha=0.10 (0.163) and declines past it. Coherence also
peaks at alpha=0.10 (0.677), holds roughly flat through alpha=0.20 (0.665) — a small plateau in
the safe zone — and then COLLAPSES super-linearly: 0.46 at alpha=0.4, 0.11 at alpha=0.8. The knee
of the cliff is approximately alpha=0.10–0.20: the zone where behavior is at or near its maximum
while coherence is still high. Past alpha~0.20, both metrics fall off a cliff together. The
collapse between alpha=0.20 and alpha=0.80 is not gradual — coherence drops by ~84% (0.665 to
0.106) while alpha quadruples.

**Interpretation — what the cliff means in practice.** The AxBench experiment confirms the
practical finding from the synthetic campaigns: on a real benchmark with real AxBench concepts,
the safe operating window for relative_add steering at 2B/layer-20 is approximately
alpha~0.10–0.20. Steering within that window gives close-to-peak behavior with coherence above
0.65. Steering above alpha~0.20 buys nothing in behavior (it was already declining) and costs
sharply in coherence. The cliff is not a quirk of any single synthetic concept — it appears as
a clean, consistent shape averaged across 30 real AxBench concepts.

**The E7-contrast (geometry generalizes; direction-specificity does not).** S-17 establishes a
clear asymmetry in what AxBench confirms and what it does not:

- **E3 (coherence cliff) — SUPPORTED on AxBench** (this section). The program's
  geometry/coherence finding — that steering has a safe window around alpha~0.10 and collapses
  super-linearly past alpha~0.20 — survives the move to a real benchmark. So does N17
  (off-shell displacement predicts incoherence): the incoherence the geometry probe predicts is
  now observed from real generated text judged by an off-family fluency scorer.

- **E7 (real DiffMean direction beats a matched-displacement shuffled-label control) — WEAK /
  SCALE-DEPENDENT on AxBench** (see S-16). At 2B the real direction beats the shuffled
  control by only +0.004 (with ~97% of the steering effect captured by the shuffled vector),
  and at 270M the effect is NEGATIVE/at the floor. The specific direction of the steering
  vector carries very little of the behavioral effect — most of the effect is generic
  (direction-agnostic displacement).

In plain language: **the cliff is real and useful** (it tells you where to operate: alpha~0.10 is
safe; alpha>0.20 is risky). **The claim that the specific concept direction matters much is
weak** on this benchmark. A program that only tested E7 would conclude "steering is mostly
generic, barely directional" — which is somewhat discouraging. But the E3/N17 results show
that the geometry and coherence structure of steering is robust and predictable, which is itself
useful: you can screen hundreds of steering configurations for coherence impact cheaply, and you
know where the safe operating window is.

**Caveats (stated prominently):**

1. **30 concepts only.** This is a curve-tracing sample, not a population test. The curve shape
   is consistent and visually clean, but no paired statistical test was run across the 30 concepts
   (the experiment measures the curve shape, not a per-concept delta). The 500-concept
   population-test design (used in S-16 for E7) would be needed to turn this into a rung-3
   statistical result for E3.
2. **Single model (2B) and single layer (20).** The cliff alpha may shift at other model sizes
   or other injection layers. The synthetic-data campaigns (S-2, S-4, S-8, S-9) showed the cliff
   knee also sits around alpha~0.05–0.10 in relative terms on 270M and 1B, suggesting the
   qualitative shape is robust across scale, but the exact knee has not been reproduced at
   rung-3 on the non-2B models with AxBench data.
3. **Judge AUC 0.68 (weak but unbiased, disclosed).** The fluency score is an imperfect proxy
   for true coherence, and the concept score is a weak instrument. Both are directionally
   unbiased given n=30 concepts × 8 instructions = 240 data points per alpha. A stronger or
   human-calibrated judge would produce more precise estimates of the cliff position.
4. **SCREENING — not external-ready.** The six-part statistical contract (Section "What counts
   as an external-ready finding") has not been applied. There is no per-concept paired Wilcoxon,
   no bootstrap CI on the cliff alpha, and no Holm-Bonferroni correction across the alpha family.
   This is a clear, rung-3-style observation — stronger than the synthetic screens because it
   uses real AxBench data and a real off-family judge — but it requires the full population-test
   design to be promoted to an external-ready finding.

**Cross-references:** S-16 (E7 WEAK on AxBench — contrasting result for direction-specificity);
S-2, S-4, S-8, S-9, S-11 (the synthetic cliff confirmations this result extends to real data);
the N17 rung-3 section above (off-shell displacement predicts incoherence — the geometric
mechanism underlying the cliff).

**Model / data:** google/gemma-2-2b-it (layer 20, relative_add, DiffMean), 30 AxBench concepts
× 8 eval instructions, off-family local judge Qwen2.5-7B-Instruct (4-bit), AxBench
concept+fluency rubric (judge AUC 0.68, disclosed), exp#120 (tag: E3-axbench-gemma-2-2b-it).

**SUPPORTED on real AxBench — E3 coherence cliff confirmed at 2B.** The cliff shape (peak ~alpha
0.10, super-linear collapse past alpha~0.20) is visible on 30 real AxBench concepts measured by
a real off-family judge. This generalizes the program's synthetic-data cliff result and the N17
geometry result to a real benchmark. Contrast with S-16: geometry/coherence findings generalize;
direction-specificity (E7) does not. SCREENING ONLY — not an external claim. Zero
external-ready findings.

---

### S-18: E2 layer curve is NEARLY FLAT on real AxBench (2B, 20 concepts) — weak mid-late preference (best ~L18–20), no sharp optimum

**What hypothesis this tests:** "E2 — Fisher layer selection." The pre-registered prediction was
that the layer of maximum linear separability (Fisher ratio) is the best injection layer. E2 was
already FALSIFIED on synthetic Gemma-270m screening (S-5, Spearman +0.14, p=0.74). S-18 asks the
more general layer-selection question on the REAL AxBench benchmark at 2B scale: does ANY layer
stand out as clearly best for steering?

**Design.** Model: google/gemma-2-2b-it (26 layers). Benchmark: AxBench
(`pyvene/axbench-concept500`), 20 concepts × 8 eval instructions. Operation: relative_add at the
knee alpha=0.10. Layers swept: 6, 10, 14, 18, 20, 22. Both behavior (concept score 0–2 → [0,1])
and coherence (fluency score 0–2 → [0,1]) scored by the off-family local judge
(Qwen2.5-7B-Instruct, 4-bit; judge AUC 0.68 vs AxBench ground truth — WEAK but UNBIASED,
disclosed). Experiment: exp#121, tag E2-axbench-gemma-2-2b-it.

**The layer-behavior-coherence curve:**

| Layer | Behavior ([0,1]) | Coherence ([0,1]) |
|-------|-----------------|-------------------|
| 6 | 0.166 | 0.609 |
| 10 | 0.163 | 0.641 |
| 14 | 0.166 | 0.606 |
| **18** | **0.184 (PEAK)** | 0.631 |
| 20 | 0.178 | **0.669 (PEAK)** |
| 22 | 0.175 | 0.619 |

**Interpretation — honest.** The layer curve is NEARLY FLAT. Behavior ranges only 0.163–0.184 —
a ~13% relative spread — across layers 6–22. The peak behavior layer (18) beats the weakest
(layer 10) by just 0.021 absolute. The peak coherence layer (20) is one step later. This is a
WEAK mid-to-late layer preference, not a sharp single optimum. There is no layer that dominates;
steering works roughly equally across a wide range of layers in this mid-to-late regime. The mild
best (layers 18–20) is consistent with AxBench's own use of layer 20 for the 2B model.

**Verdict: WEAK / FLAT.** No strong layer dependence. This fits the emerging program theme from
E7 (direction weak, S-16) and E2 (layer nearly flat, this section): the DIRECTION of the steering
vector and the LAYER of injection both matter little in the range tested. The dominant control
variable is alpha — the E3 coherence cliff (S-17) — not the specific direction or layer. Taken
together: practitioners should spend optimization budget on alpha tuning, not on layer search.

**Caveats:** 20 concepts only (single sweep, not a population paired test); single model (2B)
and single knee alpha; layers sampled at intervals of 4, not exhaustively; judge AUC 0.68
(disclosed, unbiased). Single seed / screening tier — not external-ready.

**Cross-references:** S-5 (E2 FALSIFIED on synthetic Gemma-270m — Fisher ratio does not predict
best layer); S-16 (E7 WEAK — direction does not matter much); S-17 (E3 SUPPORTED — alpha is the
dominant control).

**Model / data:** google/gemma-2-2b-it, 20 AxBench concepts × 8 eval instructions, relative_add
alpha=0.10, layers 6/10/14/18/20/22, off-family local judge Qwen2.5-7B-Instruct (4-bit), exp#121
(tag: E2-axbench-gemma-2-2b-it).

**WEAK / FLAT — E2 layer curve is nearly flat on real AxBench (2B). Mild mid-late preference
(best ~L18–20), no sharp optimum.** Reinforces the emerging theme: alpha (E3) is the dominant
control; direction (E7) and layer (E2) matter little. SCREENING ONLY — not an external claim.
Zero external-ready findings.

---

### S-19: DiffMean and PCA-top1 are only MODERATELY aligned on real AxBench concepts (mean |cos| 0.65) — the synthetic cos~0.99 did not generalize

**What hypothesis this tests:** "E4 — DiffMean vs PCA cosine alignment." The pre-registered
prediction is that DiffMean and PCA-top1 produce the same steering direction (cosine > 0.95)
so the cheaper DiffMean suffices for all practical purposes. S-1 (Qwen), S-3 (Gemma-270m),
and S-8 (Gemma-1B) established cosine alignment of 0.994–0.999 across 3 models and 4 behaviors
on synthetic single-concept data, apparently confirming E4. S-19 re-tests E4 as a pure geometry
check on 100 REAL AxBench concepts (no generation, no judge) and finds the near-equivalence
does NOT hold: on real concepts the two extraction methods are only moderately aligned.

**Design.** Model: google/gemma-2-2b-it, injection layer 20. Benchmark: AxBench
(`pyvene/axbench-concept500`), 100 concepts. For each concept: build v_DiffMean = mean(positive
activations) − mean(negative activations) from the AxBench contrast text; build v_PCA = top-1
principal component of the set of (positive minus negative) activation difference vectors from
the same contrast text; compute |cos(v_DiffMean, v_PCA)|. This is a pure geometry check — no
steering, no generation, no judge. Experiment tag: E4-axbench-gemma-2-2b-it. Experiment
number: exp#122.

**Results (100 AxBench concepts, Gemma-2-2B-it, layer 20):**

| Metric | Value |
|--------|-------|
| Mean |cos(DiffMean, PCA-top1)| | **0.6529** |
| Median |cos(DiffMean, PCA-top1)| | **0.7292** |
| 5th percentile (p5) | **0.1889** |
| Fraction of concepts with |cos| >= 0.90 | **0.16 (16%)** |

**Honest interpretation.** The synthetic screens (S-1, S-3, S-8) found cos 0.994–0.999 across
three models and four behaviors and labelled E4 SUPPORTED. On 100 real AxBench concepts the
picture is completely different: the mean alignment is only 0.65 and the median 0.73. Only 16%
of concepts reach the pre-registered >=0.95 threshold; the bottom 5% of concepts are nearly
ORTHOGONAL (p5=0.19). DiffMean and PCA-top1 are NOT interchangeable on real, diverse concepts
— they point in meaningfully different directions for the majority of the AxBench population.

**Why the synthetic result was misleading — the unpaired-data nuance.** The synthetic screens
used a small set of TRUE matched pairs: each positive sentence was paired with its corresponding
negative, so the (positive − negative) difference vectors all pointed in essentially the same
direction. In that setting, the first PCA component captures almost all the variance and aligns
tightly with the DiffMean. AxBench's contrast data does not supply true matched pairs: the
positive outputs and negative outputs for a concept are NOT individually matched sentence-to-
sentence. PCA-top1 of arbitrarily-paired differences picks up sentence-to-sentence text
variation as well as the concept direction, and that extra variance pulls the PCA direction
away from the DiffMean. This is the mechanism by which the cos~0.99 paired-data result failed
to generalize: it was a consequence of clean pairing, not of an inherent geometric equivalence.

**Connection to the synthetic-overstates theme.** This result continues the pattern established
by S-16 (E7: synthetic +0.135 direction win did not generalize — weak at 2B, negative at 270M),
S-18 (E2: layer curve nearly flat on real AxBench, no sharp optimum), and this section (E4:
cos~0.99 equivalence did not generalize). The one real-benchmark positive result — the E3
coherence cliff (S-17) — concerns the alpha-sweep geometry, which does not depend on the
quality of a single contrast pair. The pattern is: claims that depend on the quality or pairing
structure of the contrast set (E4, E7) are fragile when moved to real diverse data; claims about
gross steering geometry (E3, N17) are robust.

**Caveats.** 100 concepts only (a sample of AxBench's 500); single model (2B) and single layer
(20); the unpaired-data nuance above means the result is not a clean test of the original E4
hypothesis (which assumed paired data). A version of E4 re-stated for unpaired data would need
a dedicated test design. SCREENING — not external-ready.

**Cross-references:** S-1, S-3, S-8 (the synthetic E4 confirmations this section revises);
S-16 (E7 WEAK on real AxBench); S-17 (E3 SUPPORTED on real AxBench — contrast: geometry
findings generalize, direction-specificity and extraction-equivalence do not).

**Model / data:** google/gemma-2-2b-it (layer 20), 100 AxBench concepts from
`pyvene/axbench-concept500`, pure geometry check (no generation, no judge), exp#122
(tag: E4-axbench-gemma-2-2b-it).

**PARTIAL (revised from SUPPORTED) — E4 cosine equivalence does NOT generalize to real AxBench
concepts.** Mean |cos| 0.65, median 0.73, only 16% >=0.90, p5 0.19 on 100 concepts (2B,
layer 20, exp#122). The synthetic cos~0.99 relied on clean paired data and was a pairing
artifact. On real unpaired AxBench contrast data the two methods diverge substantially.
SCREENING ONLY — not an external claim. Zero external-ready findings.

---

### S-20: E27 rotation-vs-add on real AxBench — NOT SUPPORTED: rotation gives no benefit, operations roughly equal

**What hypothesis this tests:** "E27 — Rotation beats addition on small models." The
pre-registered prediction was that norm-preserving rotation outperforms additive
steering, specifically on small (<3B) models where additive edits more easily exit the
manifold. In synthetic screening (S-7), full-vector rotation was FALSIFIED: it costs
+42% PPL at matched behavior. S-20 re-asks the E27 rotation-vs-addition question on
real AxBench concepts with the relative_add and relative_rotate operations (not
full-vector rotation), as part of the (source × operation) 4-cell grid.

**Design.** Model: google/gemma-2-2b-it, layer 20. Benchmark: AxBench
(`pyvene/axbench-concept500`), 20 concepts × 8 eval instructions. Operations:
relative_add and relative_rotate at alpha=0.10. Sources: DiffMean and PCA-top1. Both
behavior (concept score 0–2 → [0,1]) and coherence (fluency score 0–2 → [0,1]) scored
by off-family local judge (Qwen2.5-7B-Instruct, 4-bit; judge AUC 0.68 — WEAK but
UNBIASED, disclosed). Experiment: exp#123, tag E27-axbench.

**The 4-cell (source × operation) grid:**

| Source | Operation | Behavior ([0,1]) | Coherence ([0,1]) |
|--------|-----------|-----------------|-------------------|
| DiffMean | add | 0.169 | 0.681 |
| DiffMean | rotate | 0.166 | 0.681 |
| PCA | add | 0.178 | 0.644 |
| PCA | rotate | 0.172 | 0.653 |

**Rotate-vs-add delta (DiffMean):** −0.003 behavior, 0.000 coherence. Rotate-vs-add
delta (PCA): −0.006 behavior, +0.009 coherence.

**Interpretation — honest.** Rotation does NOT beat additive steering. Across both
sources the behavior difference between add and rotate is at most −0.006, and
coherence is essentially identical for DiffMean and slightly higher for PCA-rotate.
The rotate-vs-add effect is marginally WORSE in behavior at identical or near-identical
coherence. This is consistent with the synthetic E27 FALSIFICATION (S-7), but the
real-benchmark result is more nuanced: the difference is tiny (rotate ~ add), not the
dramatic +42% PPL penalty seen with full-vector rotation on synthetic data. The
correct verdict is NOT SUPPORTED rather than FALSIFIED: rotation gives no benefit,
but it is not dramatically harmful on real abstract concepts at this scale and alpha.

**Caveat on the earlier synthetic result.** S-7 tested FULL-VECTOR rotation (rotating
the entire hidden state toward the steering direction), which is a much more aggressive
operation than the relative_rotate operation used here (a selective norm-preserving
rotation in the steering plane). The difference in severity is expected: the earlier
+42% PPL penalty was from a crude full-vector rotation; the current result uses a more
surgical plane-rotation at a small alpha. Both confirm that rotation offers no advantage
over addition, but by different margins.

**Connection to the real-benchmark theme.** S-20 extends the pattern established by
S-16 through S-19: yet another design choice (operation type) that synthetic results
suggested might matter turns out to make no meaningful difference on real diverse
concepts. The only knob that matters is alpha (S-17, E3).

**Caveats.** 20 concepts only; single model (2B) and single layer (20); single alpha
(0.10); judge AUC 0.68 (disclosed, unbiased). Single screening run — not external-ready.
Four-cell grid with 20 concepts cannot support a per-cell paired statistical test with
adequate power.

**Cross-references:** S-7 (E27 FALSIFIED on synthetic full-vector rotation, +42% PPL
penalty); S-16 through S-19 (real-benchmark theme — direction, layer, source also weak);
S-21 (E36 source wash — same grid, source axis).

**Model / data:** google/gemma-2-2b-it (layer 20), 20 AxBench concepts × 8 eval
instructions, relative_add / relative_rotate, alpha=0.10, DiffMean and PCA-top1 sources,
off-family local judge Qwen2.5-7B-Instruct (4-bit), exp#123 (tag: E27-axbench).

**NOT SUPPORTED on real AxBench — E27 rotation gives no benefit; add and rotate are
roughly equal (rotate-vs-add delta −0.003 behavior, 0.000 coherence for DiffMean).
Consistent with synthetic FALSIFICATION but difference is tiny, not dramatic.
SCREENING ONLY — not an external claim. Zero external-ready findings.**

---

### S-21: E36 source (DiffMean vs PCA-top1) on real AxBench — WEAK/WASH: source barely matters

**What hypothesis this tests:** "E36 — DiffMean equals PCA at matched displacement."
The pre-registered prediction was that the two vector construction methods steer
identically when compared fairly (equal fractional displacement). In synthetic screening
(S-9), this was SUPPORTED: at matched relative alpha, DiffMean and PCA-top1 produce
behavior within 0.02 and PPL within 8%. S-21 re-asks the E36 source question on real
AxBench concepts as part of the (source × operation) 4-cell grid, at a fixed layer, alpha,
and operation.

**Design.** Same 4-cell grid as S-20. See S-20 for the full design specification. The
source comparison focuses on the add operation (the primary comparison); the rotate
rows provide a secondary cross-check. Experiment: exp#124, tag E36-axbench.

**The source comparison (additive operation):**

| Source | Behavior ([0,1]) | Coherence ([0,1]) | Delta vs DiffMean |
|--------|-----------------|-------------------|-------------------|
| DiffMean + add | 0.169 | 0.681 | — (reference) |
| PCA + add | 0.178 | 0.644 | +0.009 behavior, −0.037 coherence |

**The source comparison (rotate operation, secondary):**

| Source | Behavior ([0,1]) | Coherence ([0,1]) | Delta vs DiffMean |
|--------|-----------------|-------------------|-------------------|
| DiffMean + rotate | 0.166 | 0.681 | — (reference) |
| PCA + rotate | 0.172 | 0.653 | +0.006 behavior, −0.028 coherence |

**PCA-vs-DiffMean delta (add):** +0.009 behavior, −0.037 coherence.

**Interpretation — honest.** Source barely matters. PCA-top1 steers marginally HIGHER
behavior (+0.009) than DiffMean but at LOWER coherence (−0.037). This is a
behavior-coherence tradeoff, with both differences small enough to be considered a wash.
Note the asymmetry with the direction-alignment result (S-19): although DiffMean and
PCA-top1 point in meaningfully DIFFERENT directions on real AxBench concepts (mean
|cos| 0.65, S-19), they produce SIMILAR steering outcomes in behavior and coherence when
used at the same relative alpha. This suggests the steering effect is largely insensitive
to the specific direction, as long as the magnitude is matched — consistent with S-16
(E7: most of the steering effect is generic, not direction-specific, at 2B).

Note: AxBench's own paper found diff-in-means best for concept DETECTION (classifying
whether text expresses a concept). For STEERING the two sources are roughly equal —
this is consistent with the S-16 finding that the specific direction matters little;
the detection-vs-steering asymmetry may reflect that detection relies on direction
specificity while steering mainly relies on displacement magnitude.

**Connection to the real-benchmark theme.** S-21 closes the (source × operation) grid
and confirms: neither axis provides a meaningful advantage on real abstract concepts.
Combined with S-16 (direction weak), S-18 (layer flat), S-20 (operation ~equal), the
picture is complete: alpha is the only knob that matters in this regime.

**Caveats.** 20 concepts only; single model (2B) and single layer (20); single alpha
(0.10); judge AUC 0.68 (disclosed, unbiased). The +0.009 behavior difference and
−0.037 coherence difference are small relative to concept-to-concept variation; no
paired statistical test run on the 20 concepts. SCREENING — not external-ready.

**Cross-references:** S-9 (E36 SUPPORTED on synthetic data — PCA ≈ DiffMean at matched
alpha, confirmed); S-19 (E4 — DiffMean and PCA-top1 directions are only moderately
aligned on real concepts, mean |cos| 0.65; yet S-21 shows steering outcomes are similar
despite directional divergence); S-16 (E7 — most steering effect is generic, not
direction-specific, consistent with source wash); S-20 (E27 operation — same grid,
operation axis).

**Model / data:** google/gemma-2-2b-it (layer 20), 20 AxBench concepts × 8 eval
instructions, relative_add / relative_rotate, alpha=0.10, DiffMean and PCA-top1 sources,
off-family local judge Qwen2.5-7B-Instruct (4-bit), exp#124 (tag: E36-axbench).

**WEAK/WASH on real AxBench — E36 source barely matters. PCA marginally higher behavior
(+0.009) but lower coherence (−0.037) vs DiffMean — a tradeoff, not a win. Despite
DiffMean and PCA-top1 pointing in meaningfully different directions on real concepts
(S-19), they produce similar steering outcomes. SCREENING ONLY — not an external claim.
Zero external-ready findings.**

---

## Required experiments before any external claim

The project's internal ICML-style review rates the current methodology as Borderline
6/10 on methodology and Reject 3/10 as a claims source in its present state.
Three categories of work are required before any screening observation can be
promoted to an external finding:

**1. Real behavior measurement.** Every behavior efficacy number in the screening
observations above uses an activation-projection proxy — the cosine similarity
between the steered output activation and the steering vector that was used to
steer it. This is circular: the same representations used to construct the steering
direction are used to assess whether it worked. The required fix is an independent
LLM judge (or an AxBench scorer) that evaluates real generated text, not
activations, against human-verified criteria. This work is in progress.

**2. Real safety measurement.** The safety axis (JailbreakBench Compliance Rate —
what fraction of harmful prompts does the steered model comply with?) is not yet
wired to real generation and real judgement. Baseline compliance rate numbers in
some experiments come from a stub, not actual harmful-prompt evaluation. The required
fix: generate responses to real JailbreakBench prompts, judge them SAFE or UNSAFE,
verify that the baseline compliance rate is approximately 0%, and confirm that the
safety auto-discard mechanism fires correctly. This work is in progress.

**3. Full statistical rigor on a larger model.** The six-part statistical contract
must be satisfied on Gemma-2-2B-it (the project's standard evaluation model, larger
than any model tested so far in full sweeps) with:
- Real AxBench (behavior), MMLU (capability), and WikiText-2 (coherence) evaluation
- A prompting baseline: can the same behavior be achieved by prompting the model
  differently, without any activation steering at all? Steering must beat this baseline.
- A held-out-concept split: the behavior concept used in evaluation was not seen
  during steering vector construction
- A shuffle-test negative control: vectors built from randomly shuffled positive/
  negative labels should produce near-zero steering effect (validates that the signal
  is in the semantic contrast, not the mechanics of the procedure)

---

## Template — structure for finding entries when they exist

The following shows the required format for each entry once a result clears the
Rigor Contract. No entry exists yet in this section.

### F-[N]: [One-line claim in plain English]

**Source experiments:** exp-NNN, exp-MMM
**Hypothesis tested:** [Full text of the pre-registered hypothesis from the hypothesis registry]
**Model:** Gemma-2-2B-it (4-bit), with smoke-confirmation on Gemma-3-1B-it
**Evaluation rung:** STANDARD (rung 3) or FULL (rung 4)

**Result table:**

| Metric | Value | 95% CI | Seeds (n) |
|---|---|---|---|
| Behavior efficacy (independent judge) | X.XX | [A, B] | 7 |
| MMLU delta (percentage points) | X.XX | [A, B] | 7 |
| Perplexity on WikiText-2 | X.XX | [A, B] | 7 |
| JailbreakBench Compliance Rate | X.X% | [A, B] | 7 |
| Over-refusal rate on benign inputs (XSTest) | X.X% | [A, B] | 7 |
| Composite score (all five axes) | X.XXXX | [A, B] | 7 |
| Delta vs baseline composite | +X.XXXX | [A, B] | 7 |

**Statistics:**
- Paired Wilcoxon signed-rank: W=X, p=Y (Holm-Bonferroni corrected across N family members)
- Bootstrap CI (10,000 resamples): [A, B]
- Ordinal gate: worst evaluation seed composite (X.XXXX) > best baseline seed composite (X.XXXX) — PASS

**Pre-registered falsifier:** [Quoted verbatim from the hypothesis registry.] Result: NOT FIRED.

**Qualifier:** Internal QA pass — independent external review pending.

---

> This file is updated each time an experiment clears rung 3 (STANDARD) or
> rung 4 (FULL) with the full Rigor Contract satisfied. Until then, all
> quantitative observations live in EXPERIMENT_LEDGER.md as screening entries.
> Composite formula fingerprint: `a9001e87087e`
