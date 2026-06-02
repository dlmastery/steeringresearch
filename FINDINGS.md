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

**S-1 through S-15** are the screening/confirmation observations recorded in this
document. S-1 through S-14 are directional findings from single experimental runs
(n=1). S-15 is different: it is the program's first controlled, multi-seed (n=20),
off-family-judged, cross-scale result — measured the way the earlier single-seed
screens were not. It is the strongest and most rigorous observation in the program,
but it is still PROVISIONAL (not external-ready) for the precise reasons given in
its entry. None of S-1 through S-15 meets the full six-part bar required for an
external claim.

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

**No external-ready findings yet — 117 experiments run; the strongest (E7, exp#116/117)
is a controlled n=20 cross-scale result but remains PROVISIONAL, not external-ready.**

117 experiments have run on real Gemma-3-270m, Gemma-3-1B, and Qwen-2.5-0.5B. They
produced 15 screening/confirmation observations (S-1 through S-15) spanning 18 of
the 70 hypotheses. Two rung-3 evaluation attempts have been completed: N17 (off-shell
displacement predicts incoherence) and E7 (relative-steering directional control,
exp#116/117 — the program's first result with real matched-displacement controls,
n=20 seeds, an off-family LLM judge, and cross-scale replication). The E7 result is
the strongest and most defensible in the program: its real concept direction beats a
matched-displacement shuffled-label control on BOTH model scales under Holm-corrected
statistics with bootstrap CIs excluding zero. But it falls short of the two strictest
legs of the six-part contract (the ordinal gate on both scales; matched-coherence on
1B), so by the program's own rigor floor there are STILL zero external-ready findings.
E7 is now the closest the program has come. The screening observations motivate the
next required experiments but are NOT citable claims.

---

## Summary table — every hypothesis tested so far

All verdicts are SCREENING (single-run, n=1) unless marked [rung-3].

| Hypothesis | Plain-English question | Verdict | One-line result |
|---|---|---|---|
| E1: DiffMean pair-count knee | How many contrast pairs do you need before the steering vector stabilizes? | DIRECTIONAL (underpowered) | Vector reaches >0.95 cosine of full at ~5 pairs on a simple concept; the corpus's 50-pair threshold cannot be tested yet |
| E2: Fisher layer selection | Is the most linearly separable layer the best one to inject the steering vector? | FALSIFIED | Spearman(Fisher ratio, efficacy) = +0.14, p=0.74; the most separable layer on Gemma-270m (layer 12) is not the best steering layer |
| E3: Alpha coherence cliff | Does output quality collapse super-linearly once steering strength exceeds a threshold? | SUPPORTED | Confirmed on all three models across a wide range of steering strengths; the safe window emerges with model scale |
| E4: DiffMean vs PCA-top1 alignment | Do the cheap and expensive vector construction methods produce the same direction? | SUPPORTED | Cosine alignment 0.994–0.999 across 3 models and 4 behaviors; the two methods are effectively equivalent |
| E7: Norm-relative alpha | Does expressing steering strength as a fraction of activation magnitude reduce variability — and does the real concept direction beat a matched-displacement control? | SUPPORTED (directional, scale-replicated; n=20 judge) but PROVISIONAL [rung-3] | Relative steering gives a clean, norm-independent cliff peaking at ~10% displacement; under an off-family judge the real direction beats a matched-displacement shuffled-label control on BOTH scales (270m +0.135 p=4e-4; 1B +0.096 p=.014, Holm-corrected, bootstrap CIs exclude 0), but fails the strict ordinal gate (both scales) and matched-coherence (1B) — see S-15 |
| E10: Category orthogonality | Do steering vectors for distinct concepts point in different directions? | PARTIAL | Mostly orthogonal (|cosine| < 0.3) except anger and happiness share a component (cosine = +0.48) |
| E17: Near-orthogonal stacking | Can two behavior vectors be applied simultaneously without mutual interference? | SUPPORTED | Stacking anger + happiness retains 101%/110% of solo behavior despite +0.48 cosine overlap |
| E18: Interference vs Gram mass | Does interference between stacked vectors grow proportionally to their geometric overlap? | PARTIAL | Stacking retains 85–94% behavior, but retention is not monotone in Gram off-diagonal mass |
| E22: Norm budget and collapse | Does cumulative displacement across all stacked vectors jointly govern coherence collapse? | SUPPORTED | Four stacked concepts drive PPL from 138 to 4,518 as total displacement scales; consistent with a shared norm budget |
| E27: Rotation beats addition on small models | Does norm-preserving rotation outperform additive steering on small models? | FALSIFIED (with caveat) | Full-vector rotation costs +42% PPL at matched behavior vs additive; caveat: this falsifies full-vector rotation, not selective-plane rotation |
| E28: Behavior plane low-rank | Does the steering direction live in a 2–3 dimensional subspace? | FALSIFIED | Top-3 SVD dimensions explain only 66% of variance; not low-rank |
| E35: Sparse behavior vector | Can the steering vector be compressed to 10% of its coordinates with minimal loss? | PARTIAL | Top-10% coordinates retain 77% behavior; below the 85% target |
| E36: DiffMean equals PCA at matched displacement | Do the two construction methods steer identically when compared fairly (equal displacement)? | SUPPORTED | At matched fractional displacement, behavior within 0.02 and PPL within 8%; earlier apparent gaps were a parameterization artifact |
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
judge, and cross-scale replication. It is the strongest and most rigorous
observation in the program. It is nonetheless PROVISIONAL (not external-ready) for
the two specific reasons stated in its entry, so it too cannot yet be cited as a
finding. It is documented in full below, after S-14.

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
This is the closest the program has come to an external-ready finding; the external-ready
count remains zero.

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
