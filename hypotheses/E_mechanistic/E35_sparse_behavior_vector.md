# E35 — Sparse Behavior Vector: Behaviors Live in <10% of Dimensions

> **One-line claim:** A DiffMean behavior vector sparsified to its top-
> magnitude 10% of coordinates retains >= 85% of steering efficacy;
> the behavior signal is concentrated in a sparse coordinate set, not
> distributed uniformly across the full residual-stream dimension.
>
> **Block:** E — Mechanistic and interpretability-guided (E34-E40).
> **Primary axes:** A2 (WHAT — direction) + A12 (BASIS/SUPERPOSITION).
> **Implementation status:** `o planned / UNTESTED`.

---

## 1. Motivation (>= 100 words)

The Linear Representation Hypothesis (Park et al. 2023, arXiv:2311.03658)
posits that high-level concepts are encoded as approximately linear
directions in activation space. If true, it raises an immediate structural
question: does a concept direction uniformly involve all d_model coordinates,
or is it sparse? For Gemma-2-2B, d_model = 2304. If the effective signal
were uniform, each coordinate would contribute roughly 1/2304 of the total,
and any coordinate-sparse approximation would degrade proportionally.
Conversely, if concepts are encoded in a sparse coordinate basis — as
suggested by the superposition hypothesis (Elhage et al. 2022, arXiv:
2209.11895), which posits that a network of rank r can represent far more
than r concepts by exploiting near-orthogonal sparse codes — then retaining
the top 10% of coordinates by magnitude should preserve almost all of the
concept signal, because the small-magnitude coordinates are noise. This
distinction has direct practical relevance for the steering program. A
sparse behavior direction could be (a) stored and applied more efficiently
(sparse vector-matrix multiply), (b) made more interpretable (the non-zero
coordinates are inspectable), (c) composed more robustly (sparse vectors
are less likely to have residual overlap via the long tail of small
components), and (d) compared with SAE feature representations (which
are by design sparse in feature space). The competing hypothesis is that
high-magnitude-coordinate sparsification is simply performing soft-
thresholding on noise and the actual behaviour requires the full vector,
meaning that the "top-magnitude" criterion is a proxy for signal-to-noise,
not for conceptual locality. E35 tests which account is correct and
establishes the sparsity regime of Gemma-2-2B behavior directions.

---

## 2. Formal Hypothesis (>= 50 words)

**H:** For at least three distinct behavior directions (e.g., refusal,
sycophancy, formality) extracted via DiffMean on Gemma-2-2B-it, sparsifying
the vector to its top-k coordinates by absolute magnitude, where k = 10%
of d_model (k = 230 out of 2304), will retain >= 85% of the behavior-success
rate measured on real generated text relative to the full-dimensional vector
at matched fractional alpha, with perplexity change no greater than 15%
relative to the full-vector condition. A random-coordinate-k control (keep
230 random coordinates) will retain significantly less efficacy, confirming
the signal is localized to high-magnitude coordinates rather than being
uniformly distributed.

---

## 3. Falsifier (>= 30 words)

If the top-10% sparsified vector retains LESS THAN 85% of behavior-success
rate at matched fractional alpha on two or more of the three tested
behaviors, OR if the random-k control retains >= 75% efficacy (indicating
no concentration in high-magnitude coordinates), the sparse-coordinate
hypothesis is DISCARDED and Status moves to `x disproved`.

---

## 4. Citations (Citation-Rigor format, >= 80 words)

```
Elhage, Nelson, et al. 2022 'Toy Models of Superposition' arXiv:2209.11895
Transformer Circuits Thread — the foundational superposition hypothesis
that motivates why behavior directions may be sparse in coordinate space:
features are encoded in near-orthogonal directions with the sparse
activation pattern that is the basis of this experiment.

Park, Kiho, et al. 2023 'The Linear Representation Hypothesis and the
Geometry of Large Language Models' arXiv:2311.03658 — establishes
empirically that high-level concepts are linear directions; our experiment
asks the further question of whether those directions are sparse.

Templeton, Adly, et al. 2024 'Scaling Monosemanticity' (Anthropic) and
Gao, Leo, et al. 2024 'Scaling and evaluating sparse autoencoders'
arXiv:2406.04093 — SAE work that recovers sparse feature directions from
dense activations; the claim that SAE features are both interpretable and
sparse motivates the hypothesis that the raw DiffMean direction is itself
sparse, since it is an approximation of the dominant SAE-feature direction.

Yin, et al. 2024 'Selective Activation Steering' arXiv:2601.19375
(Selective) — uses sparse per-head steering that implicitly assumes
behavior information is concentrated in specific attention heads; E35
tests the analogous claim at the coordinate level.
```

---

## 5. Mechanism

The DiffMean vector v = mean(h_harmful) - mean(h_harmless) has components
in all d_model = 2304 dimensions. The superposition hypothesis predicts
that the statistically salient components will cluster around a small set
of "feature" directions that the network consistently activates for the
relevant concept, while the remaining components reflect noise across
many weakly activated unrelated features.

Sparsification by top-magnitude coordinate retention is equivalent to
projecting v onto the subspace spanned by its largest-amplitude basis
vectors. If the behavior is localized in that subspace, the projection
preserves the causal signal. The random-k control destroys the projection
onto the high-amplitude subspace and replaces it with a projection onto
a random subspace, which by isotropy should recover only ~k/d_model of
the behavior if it is uniformly distributed. A large gap between top-k and
random-k efficacy is the definitive signature of coordinate-sparse
behavior encoding.

The experiment also tests a gradient of sparsity levels (k = 1%, 5%, 10%,
25%, 50%, 100%) to locate the knee, which reveals the effective
dimensionality of the behavior direction and informs stacking design (how
many near-orthogonal behaviors can be represented in d_model dimensions
before coordinate overlap causes interference — connecting to N3/E18).

---

## 6. Predicted Delta

| Condition | Predicted behavior success (relative to full-vector) |
|---|---|
| Top-1% coords (23 of 2304) | 50-70% |
| Top-5% coords (115) | 75-85% |
| Top-10% coords (230) | >= 85% (hypothesis threshold) |
| Top-25% coords (576) | >= 92% |
| Top-50% coords (1152) | >= 97% |
| Random-10% coords (230) | 40-60% (much lower than top-10%) |

PPL: top-10% condition within 15% of full-vector PPL. The random-control
gap (top-10% >> random-10%) is the key falsifiable signature.

---

## 7. Protocol

### 7.1 Primary experiment

- Model: Gemma-2-2B-it (4-bit), RTX 4090.
- Behaviors: (i) refusal direction (harmful vs harmless, Sorry-Bench/CAA);
  (ii) sycophancy direction (CAA sycophancy pairs); (iii) formality
  direction (formal vs informal text pairs, Sentiment/Style corpus).
- Vector extraction: DiffMean from >= 50 contrast pairs per behavior.
- Sparsification: implement coordinate-threshold mask on v; retain top-k
  by |v_i|; separately implement random-k mask with matched seed.
- Alpha: relative_add (alpha = fraction of ||h||); sweep alpha in
  {0.05, 0.10, 0.20}; report at alpha=0.10 as primary.
- Sparsity levels: k/d = {0.01, 0.05, 0.10, 0.25, 0.50, 1.00}.
- Eval: behavior-success rate on CAA/AxBench (LLM-judge on real text),
  WikiText-103 PPL, MMLU-500.
- Seeds: 3 (screening); 7 for rung-3.

### 7.2 Where it shines

This is the minimum-cost test of the superposition hypothesis applied to
steering vectors. If behaviors are sparse at the coordinate level, it
opens the door to SAE-guided steering (E37), efficient multi-vector
composition (E18), and interpretable feature identification.

---

## 8. Cross-references

- IDEA_TABLE.md Block E row E35.
- E37 (interpretability != controllability): the high-magnitude coordinates
  are the interpretable-and-controllable intersection.
- E18 (interference vs Gram mass): sparse vectors have smaller Gram
  off-diagonal mass when their non-zero coordinates are in different
  positions — this is a structural argument for why sparsification helps
  stacking.
- N3 (orthogonal capacity theorem): effective dimensionality of the behavior
  subspace sets the stacking limit; sparse behaviors reduce this effective
  use of dimensions.
- N8 (controllability != interpretability decomposition): if top-magnitude
  coordinates are also interpretable SAE features, N8's predicted gap may
  not exist.

---

## 9. Committee Q&A

**Q: Is coordinate-level sparsity the same as SAE-feature sparsity?**

> No. Coordinate sparsity measures which of the d_model ambient dimensions
> carry most signal. SAE-feature sparsity measures which learned feature
> directions are active. The two are related (SAE decoder columns are linear
> combinations of ambient coordinates) but not identical; this experiment
> tests the simpler, coordinate-level claim first.

**Q: Why 10% as the primary threshold?**

> 10% (k=230) is a round number in the superposition-hypothesis literature
> for "sparse" and is consistent with the observation that SAE feature
> activation rates are typically 1-5% (most features are off most of the
> time), but the individual feature directions are dense in ambient space.
> The 10% coordinate threshold is a more lenient version of SAE-level
> sparsity. The full sparsity sweep will reveal the knee empirically.

**Q: What if the result is behavior-specific?**

> Then we report per-behavior sparsity levels and the hypothesis is
> downgraded to `~ partial` (behaviors differ in their coordinate
> concentration). This is still informative for steering design.

---

## 10. Verification checklist

- [ ] DiffMean vector extraction confirmed correct (unit-normalized before
      sparsification to avoid confounding magnitude with direction, per C7
      lesson).
- [ ] Random-k control uses a fixed seed, repeated per behavior.
- [ ] Sparsification implemented as a binary mask (not soft-threshold) to
      ensure exact coordinate counts.
- [ ] Behavior eval uses LLM-judge on real generated text (not projection
      proxy).
- [ ] PPL computed on WikiText-103 (not training data).
- [ ] Rung-3 gate: n >= 7, Wilcoxon + bootstrap CI (Holm corrected).
- [ ] IDEA_TABLE.md row updated post-experiment.

---

## 11. Status Journal

- 2026-05-31 — Created from corpus block E, hypothesis E35.
  Status: `o UNTESTED`. Theoretically motivated by superposition
  hypothesis (Elhage 2022) and SAE literature. Dependency: requires
  generation-based behavior eval and relative_add parameterisation
  (both available after C9b; see FINDINGS.md).

---

## Addendum: Research-Scientist Critique

*Reviewer: SciCritic-E (mechanistic interpretability specialist).*

### Prior plausibility
**MEDIUM-HIGH.** The superposition hypothesis has strong empirical support
in toy models; the extension to real model behavior directions is plausible
but not established. The key uncertainty is whether the "behavior" signal
in a DiffMean vector is itself sparse or is a dense projection of many
weak concept activations.

### Mechanism scrutiny
The mechanism is logically sound. The gap between top-k and random-k is
the right test. However, "top-magnitude coordinate" sparsity is sensitive
to the basis (it is measuring sparsity in the standard basis of the
residual stream, which is not the natural feature basis of the model).
Results may differ significantly in the feature basis uncovered by SAEs.

### Confounds
1. The DiffMean vector norm varies across behaviors; without unit
   normalization before sparsification, top-magnitude coordinates at
   high alpha may still exceed the manifold budget, inflating PPL and
   confounding the efficacy comparison with the full vector.
2. If multiple behaviors have high-magnitude coordinates in the same
   dimensions (which is plausible if those dimensions are "loud" in
   general), the random-k vs top-k comparison may not cleanly isolate
   concept-specific sparsity.

### Numerology check
The 85% threshold for a 10% coordinate fraction is generous. If the
behavior direction were uniform, top-10% would retain roughly sqrt(0.1) ~
31.6% of the signal energy (by Parseval's theorem). Getting 85% suggests
the signal is about (0.85)^2 / 0.10 ~ 7x more concentrated than uniform.
This is a testable and non-trivial claim.

### Expected effect size
My prior: top-10% coordinates retain 60-80% of efficacy (not >= 85%),
because residual-stream directions in practice have significant off-peak
mass. The falsifier threshold of 85% may fire, but the sparsity curve
will still be informative.

### Minimum-distinguishing experiment
Run the sparsity sweep and report efficacy vs k/d_model for one behavior
(refusal). Compute the "effective dimensionality" as the k that crosses
85% efficacy. Compare to SAE feature activation rates for the same
behavior.

### Verdict
**TESTABLE + INFORMATIVE EVEN IF FALSIFIED** — The sparsity curve provides
mechanistic information regardless of whether the 85% threshold is met.
Recommend reporting the full curve and computing effective dimensionality
as the primary output.

---

## Provenance & Tracing

Full per-hypothesis provenance (exact experiments, reproduce commands, artifact links, reasoning trace): [`PROVENANCE/E35.md`](../PROVENANCE/E35.md).

- **Experiments:** analysis campaign (computed quantities in the campaign JSON; see the provenance file).
- **Reproduce:**

```bash
PYTHONPATH=src python scripts/campaign_sweep.py --model models/google/gemma-3-270m-it --quant none --hyp E35 --tag-prefix E35-sparse --layers 16 --alphas 0.1 --ops relative_add --behaviors anger  # sparsify the behavior vector to top-magnitude coords; efficacy vs sparsity
```
