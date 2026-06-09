# FINDINGS — Steering Research Program

**One-line summary: the steering effects measured so far are weak and largely
generic; the safety method is newly built but unvalidated. Zero external-ready
findings across 124 experiments.**

---

## How to read this document

**E1-E50** are pre-registered hypotheses (see IDEA_TABLE.md). **N1-N20** are novel
first-principles hypotheses. **S-1 through S-21** are screening/confirmation observations.
S-1-S-14 are n=1 directional screens. S-15 is the first controlled multi-seed (n=20)
result on a synthetic concept. S-16-S-21 are the real-AxBench evaluations (see sections below).

**rung-3**: real held-out benchmark data + proper statistical tests + cross-scale validation.
**Verdict tiers**: SUPPORTED / FALSIFIED / PARTIAL / INCONCLUSIVE / PENDING --
all S-observations are SCREENING ONLY and cannot be cited as research findings.

**Key terms**: DiffMean = mean(positive activations) minus mean(negative activations);
PCA-top1 = top PC of positive-minus-negative differences; alpha = fractional push size
(relative_add: push = alpha x activation_norm); off-shell displacement = |activation norm change|;
PPL = perplexity (baseline 74-90 on WikiText-2; >1000 = garbage). Composite fingerprint: a9001e87087e.

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

**No external-ready findings yet -- 124 experiments run.**

Real-benchmark synthesis (S-16 through S-21): across all six testable hypotheses
re-evaluated on AxBench (pyvene/axbench-concept500, off-family judge Qwen AUC 0.68
disclosed), the ONLY steering knob that strongly matters is alpha -- the E3 coherence
cliff (S-17, supported). Everything else is weak or a wash: direction barely beats
shuffled (E7, S-16 -- weak, scale-dependent, ~97% captured by shuffled at 2B), layer
is nearly flat (E2, S-18), source is a wash (E36, S-21), operation gives no benefit
(E27, S-20), and DiffMean/PCA alignment is only moderate on real concepts (E4, S-19).
Synthetic single-concept evaluation systematically overstated these effects. Geometry
findings generalize (E3, N17); direction-specificity does not. All S-1 through S-21
are SCREENING ONLY -- not citable claims.

---

## Summary table
 — every hypothesis tested so far

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

### S-15: E7 directional win on synthetic "ocean" — PROVISIONAL, superseded by S-16

**Context:** n=20 seeds, off-family Gemini judge, matched-displacement shuffled-label
control, two model scales. The program's first controlled multi-seed, off-family-judged
result. Superseded by S-16 which shows the effect does not generalize.

**Operation:** relative_add — push size is alpha × ‖h‖ (fractional, not absolute),
so each condition receives an identical displacement magnitude; only DIRECTION differs.
Primary control: shuffled-label (same activations, random label partition), which
destroys semantic contrast while preserving displacement. Behavior scored by off-family
Gemini judge (real ocean text 8/10; keyword-stuffing 2/10 — not fooled by soup).

**Results at alpha=0.10 (real vs shuffled-label control):**

| Quantity | 270M (exp#116) | 1B (exp#117) |
|---|---|---|
| Real behavior ([0,1]) | 0.730 | 0.549 |
| Shuffled-label behavior | 0.595 | 0.453 |
| Paired delta | **+0.135** | **+0.096** |
| Bootstrap 95% CI | [+0.084, +0.184] | [+0.025, +0.163] |
| Wilcoxon p (Holm-corrected) | 0.0004 | 0.014 |
| Ordinal gate | **FAIL** | **FAIL** |

Ordinal gate fails because the shuffled control is unexpectedly strong on a
small concept-dominated contrast set (random re-partition recovers ~50% of the
direction). On 1B the real direction also trades more coherence at the knee.
Key measurement-validity finding: the old activation-projection proxy produced
OPPOSITE conclusions (+0.022 / −0.019) on the same data; the validated off-family
judge amplified the real signal ~6×.

**PROVISIONAL — not external-ready. Superseded by S-16.**

---

### S-16: E7 on real AxBench — WEAK/SCALE-DEPENDENT; +0.004 at 2B, NEGATIVE at 270M

Corrects S-15. AxBench benchmark (pyvene/axbench-concept500, 500 concepts). Off-family
local judge: Qwen2.5-7B-Instruct 4-bit; AUC 0.68 vs AxBench ground truth — weak but
unbiased, disclosed. Replication unit: concept (n=500). Steering: relative_add alpha=0.10.

| Quantity | 270M (exp#118, L16) | 2B (exp#119, L20) |
|---|---|---|
| Mean behavior, REAL direction | 0.046 | 0.138 |
| Mean behavior, SHUFFLED control | 0.056 | 0.134 |
| Paired delta (real − shuffled) | −0.010 | **+0.004** |
| Bootstrap 95% CI | [−0.014, −0.007] (NEGATIVE) | [+0.0003, +0.008] (barely >0) |
| Wilcoxon p | 1.4×10⁻⁶ | 0.011 |
| Ordinal gate | FAIL | FAIL |
| Shuffled share of steering effect | — (both at floor) | ~97% |

270M: real direction WORSE than shuffled; both at the floor (model cannot express
abstract concepts). 2B: real beats shuffled by +0.004 — tiny, fragile, ordinal gate
fails, and ~97% of the steering effect is captured by a direction-free displacement.
The synthetic +0.135 win (S-15) was an easy-concept artifact that did not generalize.
SCREENING ONLY — not an external claim.

---

### S-17: E3 coherence cliff SUPPORTED on real AxBench (2B, L20, 30 concepts, exp#120)

Judge: Qwen2.5-7B-Instruct 4-bit (AUC 0.68, disclosed). Steering: relative_add DiffMean.

| Alpha | Behavior ([0,1]) | Coherence ([0,1]) |
|-------|-----------------|-------------------|
| 0.02 | 0.150 | 0.619 |
| 0.05 | 0.160 | 0.660 |
| **0.10** | **0.163** | **0.677** |
| 0.20 | 0.148 | 0.665 |
| 0.40 | 0.133 | 0.458 |
| 0.80 | 0.052 | 0.106 |

Behavior and coherence both peak at alpha~0.10; coherence collapses super-linearly
past alpha~0.20. Cliff shape is robust and consistent with N17 geometry prediction.
Generalizes synthetic results to a real benchmark. Key asymmetry: E3/N17 geometry
generalizes; E7 direction-specificity does not. Caveats: 30 concepts only (curve
trace, not a population paired test); single model/layer; judge AUC 0.68. SCREENING.

---

### S-18: E2 layer curve NEARLY FLAT on real AxBench (2B, 20 concepts, exp#121)

Layers 6/10/14/18/20/22; relative_add alpha=0.10; judge Qwen AUC 0.68.

| Layer | Behavior | Coherence |
|-------|---------|-----------|
| 6 | 0.166 | 0.609 |
| 10 | 0.163 | 0.641 |
| 14 | 0.166 | 0.606 |
| 18 | **0.184** | 0.631 |
| 20 | 0.178 | **0.669** |
| 22 | 0.175 | 0.619 |

Behavior range 0.163–0.184 (~13% relative), no sharp optimum. Mild mid-to-late
preference (L18–20). Layer barely matters. Reinforces the emerging theme: alpha is
the dominant control variable. SCREENING.

---

### S-19: E4 DiffMean vs PCA-top1 MODERATELY aligned on real AxBench (2B, 100 concepts, exp#122)

Pure geometry check (no generation). Mean |cos|=0.65, median=0.73, p5=0.19, only 16%
of concepts ≥0.90. The synthetic cos~0.99 was a paired-data artifact: AxBench contrast
data is unpaired, so PCA picks up text-variation variance that pulls the PCA direction
away from DiffMean. Claims that depend on pairing structure are fragile on real data;
gross geometry findings (E3, N17) are robust. SCREENING.

---

### S-20: E27 rotation vs addition NOT SUPPORTED on real AxBench (2B, 20 concepts, exp#123)

(source × operation) 4-cell grid. DiffMean: add 0.169/0.681 vs rotate 0.166/0.681.
Rotate-vs-add delta: −0.003 behavior, 0.000 coherence. Operations are roughly equal
on real abstract concepts at this scale and alpha. Consistent with synthetic FALSIFICATION
but difference is tiny, not dramatic. Neither source nor operation provides a meaningful
advantage. SCREENING.

---

### S-21: E36 DiffMean vs PCA source WEAK/WASH on real AxBench (2B, 20 concepts, exp#124)

Same 4-cell grid as S-20. PCA+add: behavior 0.178 / coherence 0.644 vs DiffMean+add:
behavior 0.169 / coherence 0.681. Delta: +0.009 behavior, −0.037 coherence. A
behavior-coherence tradeoff, not a win — source barely matters despite the two
directions being only moderately aligned (S-19). Source barely matters; alpha does.
SCREENING.

---

## Required experiments before any external claim

1. **Real safety benchmark.** Wire JailbreakBench end-to-end. The current "CR"
   dashboard column is a synthetic 22-string regex on 10 prompts — not a real benchmark.
2. **Calibrated judge.** Replace the AUC-0.68 Qwen judge with a calibrated classifier
   (Llama-Guard-3 / ShieldGemma) achieving ≥0.90 human agreement on ≥100 items.
3. **Real capability tax.** Wire real MMLU (≥500 items) in the same run.
4. **n≥7 seed evaluation.** All evaluations so far are n=1, n=20 synthetic, or
   n=500 concepts under a weak judge. The six-part contract requires n≥7 iid seeds.
5. **Prompting baseline.** AxBench results show prompting competes with steering;
   this baseline must be confronted before any external claim.
6. **Full statistical contract.** Paired Wilcoxon + bootstrap CI (10k resamples) +
   Holm-Bonferroni + ordinal gate, all applied in the same evaluation.

See `audits/reviews/IMPROVEMENTS_100.md` for the full 100-item roadmap.

---

## S-22 — First REAL end-to-end safety-method run (SCREENING, n=12, --no-log)

2026-06-07. gemma-2-2b-it (4bit) generator + Qwen-7B few-shot judge on REAL
JailbreakBench (12 harmful) + XSTest (12). Full pipeline ran end-to-end (22 min,
exit 0): extract refusal direction -> conditional CAST gate -> generate (method +
7 baselines) -> Qwen judge ASR + over-refusal -> Pareto + rigor. First time the
built method touched real models + real benchmarks.

| method | ASR | over-refusal |
|---|---|---|
| cast_method (conditional) | 0.000 | 0.250 |
| no_steer (baseline) | 0.000 | 0.333 |
| unconditional_steer | 0.000 | 0.667 |
| few_shot_prompting | 0.000 | 0.583 |
| system_prompt_refusal | 0.000 | 0.750 |

Findings (all SCREENING — n=12, judge has a conservative bias):
1. **gemma-2-2b-it ASR=0 on raw JBB** — it refuses every harmful prompt unsteered
   ("I cannot fulfill your request..."; verified by eyeballing real generations,
   NOT a judge artifact). There is NO ASR headroom on an already-aligned model
   without adversarial attacks. This refutes the naive "reduce ASR vs baseline"
   framing for aligned models.
2. **The baseline OVER-REFUSES 33% of benign XSTest prompts** even unsteered.
3. **Conditioning helps the over-refusal axis**: the conditional method (0.25)
   over-refuses LESS than unconditional steering (0.67) and prompting (0.58/0.75)
   — directional support for the conditional-gate hypothesis (M2/M4), n=12.
4. **Driver caveat**: the composite charges the conditional method the
   UNCONDITIONAL worst-case capability/coherence tax (an upper bound); the true
   conditional tax (gate rarely fires on benign MMLU) is ~0 and must be measured
   under the actual method. Fix pending.

Implication: the headline must target a regime with ASR headroom — safety UNDER
ATTACK (apply the adversarial harness to break the aligned baseline) and/or the
over-refusal/selectivity axis. See PREREGISTRATION.md.

---

## S-23 — gemma-3-270m-it is UNSTEERABLE on AxBench (SCREENING, --no-log)

2026-06-08. New standard dev config (user-mandated): target gemma-3-270m-it,
judge Qwen2.5-3B-Instruct, dataset AxBench, layer 12 (of 18). Alpha sweep
(10 concepts): behavior PEAKS at the gentlest push (alpha=0.1 -> 0.20/2.0) and
collapses monotonically to 0 (gibberish) by alpha=0.6 — NO steering window.
E7 directional (8 concepts, knee 0.1): real 0.141 vs shuffled 0.172, delta
-0.031, VERDICT NULL. Conclusion: 270m is too small to steer; any method result
on it is near-floor noise. Target switched to gemma-3-1b-it.

## S-24 — gemma-3-1b-it steers (modestly); E7 direction NULL replicates (SCREENING)

2026-06-08. Config: gemma-3-1b-it (26 layers), layer 16, judge Qwen2.5-3B,
AxBench. Alpha sweep (12 concepts): behavior peaks at alpha~0.05 (0.46/2.0,
coherence 0.51), ~2.3x better than 270m, then declines (knee ~0.05-0.08; no
"rise" past the gentlest push — coherence-limited). E7 directional (30 concepts,
knee 0.06): real 0.4306 vs shuffled 0.4583, delta -0.0278, CI [-0.056, +0.003],
VERDICT NULL. The real DiffMean direction does NOT beat a same-norm shuffled-label
control — REPLICATES the 2b AxBench result (S-19, ~97% captured by shuffled) on
the small-model dev config. Plain directional steering is generic at this scale:
the displacement norm + coherence carry the effect, not the specific direction.
Implication: the contribution must be the CONDITIONAL gate (WHEN to steer /
collateral avoidance), not the steering direction itself.

## S-25 — The conditional GATE on 1b: moderate detector + collateral avoidance (SCREENING)

2026-06-08. Config: gemma-3-1b-it layer 16, AxBench, judge Qwen2.5-3B.

(a) **Gate-as-detector (N6, judge-free).** Does the concept DiffMean direction
detect concept-relevant INPUTS (so it can gate)? Score cos(pool(h@16), v_C) on C's
held-out pos_texts (on-target) vs other concepts' texts (off-target), AUC per
concept over 20 concepts: **mean AUC 0.747, median 0.732, range 0.52-0.99; 35%
of concepts >0.8, 55% >0.7.** The raw direction is a MODERATE input-detector —
conditioning is partially viable; a trained probe (intent_gate.py) should lift this.

(b) **Conditional vs unconditional steering (collateral).** no_steer / unconditional
/ conditional(gated) over ON-target (C's eval instructions) + OFF-target (other
concepts' texts), 4-concept smoke: unconditional steering DAMAGES off-target
fluency (0.52->0.40); conditional PRESERVES it (0.52) — **fluency saved +0.125**,
the selectivity value, in the right direction. BUT the AxBench setup is confounded
for the conditional GENERATION test: eval instructions already elicit the concept
(clean induction 0.44 >= steered), and the trigger is in the OUTPUT not the INPUT,
so the gate rarely fires on-target (0.25). Conditional gating's clean testbed is
INPUT-triggered steering (safety: harmful prompt -> refuse), not concept induction.

Synthesis of the dev-config screening (S-23..S-25): plain steering is generic
(direction NULL) and weak at small scale; the conditional gate has a moderate
basis (AUC 0.75) and shows directional collateral-avoidance. Next: a TRAINED gate
(probe) to lift detection, and an input-triggered (safety) testbed on 1b.

---

> Composite formula fingerprint: `a9001e87087e`
> All S-1..S-25 are SCREENING ONLY — not external claims.
> No external-ready finding exists as of the latest experiment (#124).
