# E33 — Curved Flow Beats Linear Steering on Non-Convex Traits (FLAS-Style)

> **One-line claim:** Curved flow-based transport (FLAS-style multi-step steering)
> beats single-step linear addition primarily when the steered activation manifold
> is non-convex for the target trait; on convex traits the gain vanishes.
>
> **Source design space:** Block D — Geometry and Rotational Methods (E27–E33).
>
> **Implementation status:** UNTESTED. No screening data as of 2026-05-31.

---

## In Plain English

**What we're testing, simply:** The usual nudge moves the model's "thought" in one
straight shove. This experiment tries moving it in several small curved steps
instead. The claim is that the curved path only helps when the behavior's region
has an awkward, dented shape; for simple, smoothly-shaped behaviors the straight
shove is just as good.

**Key terms (defined here):**
- **Steering / steering vector:** nudging the model by adding a direction to its
  internal "thoughts"; the direction is the steering vector.
- **Residual stream:** the model's running internal "thoughts" that we edit.
- **Layer:** the processing step where we make the edit.
- **Alpha / strength:** how hard we push.
- **DiffMean:** the simple recipe for building a nudge.
- **Coherence:** whether the text stays fluent and sensible.
- **Curvature / curved flow:** instead of one straight jump, we move the thought in
  several small steps that bend along the way (a "flow"), following the natural
  shape of the thought space. The straight one-shot version is "linear addition."
- **Convex vs non-convex:** "convex" means a nice, simply-shaped region where a
  straight line between two good points stays inside good territory. "Non-convex"
  means a dented or curved region where a straight line can cut through bad
  territory — and that's where curving around should help.

**Why we're doing this (the point):** If some behaviors have awkward shapes, a
straight nudge can pass through "bad text" on the way. Curving around it might
reach the target more cleanly.

**What the result would mean:** A win means: for awkwardly-shaped behaviors, use
the curved multi-step path; for simple ones, don't bother. A loss means the curved
path never helps and the straight shove is always fine.

See [`../GLOSSARY.md`](../GLOSSARY.md) for any other term.

---

## 1. Motivation (>= 100 words)

Additive steering moves hidden states from their current position h along a straight
chord to h + alpha*v. This chord approximation is valid when the activation manifold
is locally flat — i.e., when the region between the unsteered and target distributions
is convex or near-convex. For a simple behavioral trait like "respond in French,"
the steered distribution may be close to the unsteered distribution along most
activation dimensions, making the manifold approximately flat between the two
populations. A single straight-line step covers the distance efficiently without
leaving the manifold. For a complex, multi-modal, or safety-relevant behavioral
trait, the manifold may be non-convex: the straight chord between the unsteered and
target distributions passes through a region of activation space that is out-of-
distribution (off-manifold), even if both endpoints are in-distribution. In this
case, a flow-based or multi-step transport that follows a curved path along the
manifold should outperform the straight-line chord. The FLAS paper (arXiv:2605.05892)
proposes exactly this: Flow-based Latent Activation Steering uses normalizing flows
or transport maps to find curved paths through activation space from the current
state to the target behavioral state. The Curveball paper (arXiv:2603.09313) reaches
the same conclusion from a different angle (optimal control). The Manifold Steering
paper (arXiv:2605.05115) provides the formal framework. E33 tests the key
conditional prediction of these papers: that curved flow outperforms linear addition
specifically when the target trait's activation manifold is non-convex. This
conditionality is important because it provides a falsifiable and actionable
criterion: measure convexity, predict which traits benefit from flow, confirm.
Without this conditionality, the claim is unfalsifiable ("flow is always better"
would not explain why the simpler additive methods work well in practice for many
traits). Our screening data (S-6, N5) show that off-shell displacement predicts
incoherence at R2=0.81 for additive steering — this is exactly the off-manifold
penalty that FLAS-style curved paths should avoid.

---

## 2. Formal hypothesis (>= 50 words)

For behavioral traits whose activation distribution in the contrast set is non-convex
(measured by a convexity proxy such as the ratio of mean-of-pairwise-distances to
the diameter of the convex hull, or by the deviation of the chord from the data
manifold as measured by log-likelihood under a fitted density estimator), FLAS-style
multi-step curved transport achieves >= 20% better composite score than single-step
additive steering at matched behavior success. For behavioral traits that are convex
by the same proxy, the gain is < 5% (the two methods are equivalent on convex traits).
This stratified prediction must hold across at least 2 traits per convexity class on
at least one model (Gemma-3-1B).

---

## 3. Falsifier (>= 30 words)

If FLAS-style curved transport does not achieve >= 15% better composite than
additive steering on ANY tested non-convex trait (i.e., the improvement is
uniformly below 15% regardless of convexity proxy), the conditionality claim is
REJECTED and curved flow provides no meaningful advantage over linear methods.
Status moves to x disproved. Secondary falsifier: if the convexity proxy fails to
predict which traits benefit (Spearman < 0.5 between convexity and flow gain), the
conditional framework itself is rejected even if the average gain is positive.

---

## 4. Citations (Citation Rigor format, >= 80 words)

```
FLAS: Flow-based Latent Activation Steering (arXiv:2605.05892) — the primary
method E33 tests; claims that curved transport paths through activation space
outperform single-step addition for complex behavioral traits. E33 tests the
conditionality: does the advantage hold specifically for non-convex traits?

Curveball Steering: Raval, Song, Wu et al. 2026 'Curveball Steering: The Right
Direction To Steer Isn't Always Linear' (arXiv:2603.09313) — demonstrates that
optimal steering paths are curved; provides the theoretical motivation for why
non-convex manifolds require curved paths.

Manifold Steering: Wurgaft et al. 2026 'Manifold Steering: Geometry-Aware
Activation Interventions' (arXiv:2605.05115) — formal framework for manifold-
following steering; distinguishes convex traits (where chord approximation is good)
from non-convex ones (where geodesic diverges from chord).

CAA / DiffMean baseline: Turner et al. 2024 (arXiv:2308.10248) — the linear
comparison method; single-step additive steering that E33 compares against.

Our C2 screening (N5/N17): log PPL = 5.40 + 2.87*off-shell at R2=0.81 — the off-
shell displacement law quantifies the coherence penalty of chord-following (linear
addition); curved paths that minimize off-shell displacement should reduce this
penalty specifically for non-convex traits where the chord is furthest from the
manifold.

GeoSteer: Kazama et al. 2026 (arXiv:2601.10229) — trajectory-aware CoT steering
via manifold gradients; a related multi-step approach that reports +0.9 accuracy /
+4.5 reasoning quality [NEEDS REPLICATION]; provides evidence that multi-step
manifold-following helps on reasoning tasks (presumably non-convex trait manifolds).

HyCon: Briglia et al. 2026 (arXiv:2603.14093) — hyperbolic concept control for
hierarchical traits; hyperbolic geometry is appropriate for tree-structured (non-
Euclidean, non-convex) concept hierarchies; E33's convexity-conditional claim
generalizes across geometry types.
```

---

## 5. Mechanism

### 5.1 Convexity vs non-convexity of behavioral manifolds

A behavioral trait B defines a contrast distribution: {h_pos_i} (activations from
positive examples) and {h_neg_i} (activations from negative examples). The trait
manifold is convex if the linear interpolation between any two points in the positive
distribution stays within a region of high manifold density — i.e., the chord between
h_pos_1 and h_pos_2 passes through in-distribution space. Non-convex traits have
positive-example activations that form clusters separated by low-density regions
of activation space.

Example convex traits: "respond in French" (a simple register shift; the French-
response activations form a roughly convex cluster in the representation space),
"be polite" (a style direction, approximately linear in activation space).

Example non-convex traits: "refuse harmful requests" (the refusal activations may
form a multimodal cluster, since refusals can be brief/polite vs long/detailed vs
redirecting; or since the harmful category itself is non-convex: hate speech and
medical advice and illegal activities all activate differently); "be factually
accurate" (accuracy may correlate with topic, creating multiple clusters).

### 5.2 How FLAS finds the curved path

FLAS uses a normalizing flow (or continuous normalizing flow) to learn a transport
map F: h_neg -> h_pos that follows the data manifold. During steering, instead of
computing h' = h + alpha*v (the chord), it applies F(h) = h' where the path of F
follows the manifold. The multi-step version integrates a flow ODE from h to h_pos
along the vector field of the learned transport. This path stays in-manifold by
construction if the flow was trained on in-distribution data.

### 5.3 The convexity-conditional prediction

For convex traits: the chord from h to h + v stays within the in-distribution region.
Off-shell displacement is minimal even for additive steering. FLAS finds a curved
path, but the straight chord is already near-optimal, so FLAS provides no benefit.

For non-convex traits: the chord passes through a low-density region (off-manifold).
Off-shell displacement spikes in the middle of the interpolation. FLAS curves around
this low-density region, staying in-manifold. Off-shell displacement is minimized.
From our C2 law (log PPL = 5.40 + 2.87 * off-shell, R2=0.81), reducing off-shell
displacement should directly reduce PPL degradation.

---

## 6. Predicted Delta (pre-registered)

| Metric | FLAS (non-convex traits) | FLAS (convex traits) |
|---|---|---|
| Composite delta vs linear | >= +0.20 (>= 20% better) | < +0.05 (< 5% better) |
| PPL at matched behavior | >= 20% lower | < 5% lower |
| Off-shell displacement (chord midpoint) | high for non-convex traits | low for convex traits |
| Spearman(convexity proxy, flow gain) | >= 0.50 (flow gain predicted by convexity) | — |
| Training cost for FLAS flow | 1-2 hours on 4090 per trait | same |

---

## 7. Protocol

### 7.1 Primary experiment

Step 1 — Select traits by convexity:
  - Convex proxy: for each trait, compute the mean distance between all positive-
    example activations vs. the diameter of their convex hull. If mean_dist /
    diameter < 0.5, call it convex; if > 0.7, call it non-convex.
  - Alternatively: fit a Gaussian mixture model (2 components) to the positive
    activations; if the cluster separation > 1.5 sigma, call it non-convex.
  - Select 2 convex traits and 2 non-convex traits for the comparison.

Step 2 — Implement FLAS-style multi-step transport:
  - Minimal version: train a 2-layer MLP normalizing flow on positive vs negative
    activations at L16 (270M) and L18 (1B) for each trait.
  - Apply 10-step integration from h to the flow-predicted h' using Euler steps.
  - Measure off-shell displacement at each step (compare to chord midpoint).

Step 3 — Compare FLAS vs linear addition at matched behavior success on each trait.
Record behavior, PPL, off-shell displacement, composite.

Wall-clock: ~4 hours per trait (flow training) + ~1 hour per model. Total: ~20 hours
for 4 traits * 2 models * 3 seeds.

### 7.2 Where it shines

E33 shines on safety-critical multi-modal traits (e.g. "refuse harmful requests
across multiple harm categories"). These are the traits most likely to have non-
convex activation distributions (different harm types cluster differently). If FLAS
provides a 20%+ composite improvement here, it directly motivates deployment.

---

## 8. Cross-references

- E29 (SLERP geodesic): SLERP is a 2-point approximation to curved-path transport;
  FLAS is the multi-step generalisation. E33 > E29 in generality.
- E27-A (rotation): rotation is another curved-path approximation (arc of a sphere);
  E33 tests whether a more general curvilinear path further improves on rotation
- N13 (geodesic > chord): E33 is the empirical test of N13's formal claim
- N14 (metric-matched operation): E33's convexity-conditional framework is a
  special case of matching the operation to the manifold geometry
- Corpus: FLAS 2605.05892, Curveball 2603.09313, Manifold Steering 2605.05115

---

## 9. Committee Q&A

**Q: How do you operationalize "convexity" without fitting a full manifold model?**

> The protocol uses two proxies: (1) the ratio of mean pairwise distance to convex
> hull diameter (cheap to compute from the contrast set activations) and (2) a
> Gaussian mixture model with 2 components (fit using sklearn, ~1 second per trait).
> If these two proxies agree on the convexity classification, the trait classification
> is reliable. If they disagree, the trait is marked ambiguous and excluded from
> the stratified analysis.

**Q: Isn't FLAS expensive — does it fit within the 4090 budget?**

> Training a 2-layer MLP flow on ~50-100 contrast activations at d=4096 takes
> approximately 100-200 gradient steps on a 4090, which is < 60 seconds. The
> inference cost (10 Euler steps vs 1 additive step) adds ~10x per forward pass,
> which is negligible for generation latency. The main cost is the flow training,
> which is well within the budget.

**Q: If FLAS requires fitting a new flow per trait per model, is it practical?**

> For a research comparison, yes. For deployment, no — the flow must be retrained
> for each new trait, model, and checkpoint. This is a known limitation of flow-
> based methods. E33 tests the POTENTIAL advantage of curved paths; practical
> deployment would require amortized or zero-shot flow estimation, which is out of
> scope for this experiment.

**Q: What if the convexity proxy is uncorrelated with the actual trait manifold
geometry?**

> Then E33's stratification strategy fails and the conditional prediction cannot
> be tested. The falsifier (Spearman < 0.5 between convexity and flow gain) covers
> this case: if the proxy is uninformative, the claim is rejected on the conditional
> framework itself, not just on the per-trait results.

---

## 10. Verification checklist

- [ ] Convexity proxy computation implemented and validated on synthetic data
  (sphere = convex; two-cluster = non-convex; proxy correctly classifies)
- [ ] FLAS minimal implementation: 2-layer MLP flow trained on contrast activations
- [ ] 10-step Euler integration implemented and unit-tested (endpoint close to h_pos)
- [ ] Off-shell displacement at chord midpoint vs flow midpoint measured and compared
- [ ] 2 convex traits and 2 non-convex traits selected and classified by proxy
- [ ] FLAS vs linear comparison on each trait at n>=3 seeds
- [ ] Spearman(convexity, flow gain) computed across traits
- [ ] Rows added to EXPERIMENT_LEDGER.md

---

## 11. Status journal

- 2026-05-31 — Created. UNTESTED. The FLAS paper (2605.05892) is a direct reference;
  the convexity-conditional prediction is our novel contribution (the paper does not
  stratify by trait convexity). The C2 off-shell displacement law provides the
  mechanistic bridge: curved paths minimize off-shell displacement, which the law
  predicts reduces log-PPL. Priority: lower than E27-A, E29, E31, E32; queued after
  the simpler geometry experiments are complete.

---

## Addendum: Research-Scientist Critique

*Reviewer: SciCritic-D. Critiquing the idea, not the implementation.*

### Prior plausibility

**MEDIUM.** The geometric intuition is sound: non-convex traits require curved paths.
The concern is that the convexity proxy (contrast-set geometry) may not reflect the
true geometry of the behavioral manifold — the contrast set is a finite sample from
a high-dimensional space, and the empirical convex hull is a poor estimator of the
true manifold's convexity in high dimensions. In d=4096, a Gaussian mixture with
2 components always looks "non-convex" (the components are well-separated by
concentration of measure), so the proxy may classify ALL traits as non-convex.

### Mechanism scrutiny

The off-shell displacement law (C2: log PPL = 5.40 + 2.87 * off-shell, R2=0.81)
is the mechanistic bridge from "curved path" to "lower PPL." The law was estimated
for additive steering only; its applicability to flow-based paths (where off-shell
is low by construction) has not been tested. If the law holds for flow paths, then
reducing off-shell displacement by 50% should reduce log PPL by approximately 1.44
(from the 2.87 coefficient). This is a large, detectable effect. However: the law
has R2=0.81 (not 0.99), and was estimated on n=23 rows at n=1 seed — noise is high.

### Confounds

1. **Flow-fitting confound:** the MLP flow is trained on the contrast set and
   evaluated on the same prompts — in-sample fit may overestimate generalization.
   Must use held-out prompts for evaluation.
2. **Convexity proxy confound:** as noted above, high-d concentration of measure
   may make all traits appear non-convex by the proposed proxy.
3. **Expressive flow confound:** a 2-layer MLP flow may not be expressive enough
   to fit non-convex manifolds; the FLAS paper likely uses deeper architectures.
4. **Semantic non-convexity vs geometric non-convexity:** the trait may be
   semantically heterogeneous (multiple sub-behaviors) without being geometrically
   non-convex in the activation sense.

### Does it specifically matter?

**MEDIUM-HIGH.** If the convexity-conditional prediction holds, it provides a
practical decision rule: measure trait convexity, then select linear vs curved
transport accordingly. This would unify the additive and flow-based method families
under a single theoretical framework. The result would be publishable as a
"geometry of behavioral traits" finding.

### Literature precedent

FLAS (2605.05892) reports advantages over linear steering but does not stratify by
trait convexity. GeoSteer (2601.10229) shows multi-step manifold-following helps on
reasoning (presumably non-convex) but not on simple tasks. The conditionality test
is novel and fills a genuine gap in the flow-vs-linear comparison literature.

### Skeptical effect-size re-prediction

For non-convex traits: FLAS composite gain ~ 5-20% (not 20%+ as claimed). The
off-shell reduction may be large but the PPL improvement per unit of off-shell
reduction is estimated from R2=0.81 data at n=1, making the prediction uncertain.
For convex traits: FLAS gain ~ -2% to +5% (possibly slightly negative due to flow
overfitting). The < 5% claim for convex traits is plausible.

### Minimum-distinguishing experiment

Pick one known-convex trait (language register) and one known-non-convex trait
(safety refusal across 3 harm categories). Train minimal FLAS flow on existing
contrast sets. Compare flow vs linear at matched behavior on each trait. If the
flow gain is > 10% for safety and < 5% for language register, the conditionality
claim is strongly supported. Cost: ~6 hours on a 4090.

### Verdict

**THEORETICALLY MOTIVATED AND FALSIFIABLE, but the convexity proxy and the flow
expressivity are the key risks. The experiment is feasible but lower priority than
E27-A, E29, E31, and E32. The conditionality prediction is the novel contribution;
validate the proxy first before building the full FLAS infrastructure.**

---

## Pseudocode & Methodology

This section specializes [`../METHODOLOGY.md`](../METHODOLOGY.md) to curved flow-based
transport (FLAS) vs the linear add chord, stratified by trait convexity. Status:
`UNTESTED` — needs a normalizing-flow module + a convexity proxy. GEOMETRY/OPERATION
hypothesis: replace the straight chord with a multi-step curved path along the manifold.

### 1. Steering recipe (DiffMean chord baseline vs a learned curved transport)

```python
# Linear baseline = METHODOLOGY §1.3 DiffMean injected as the add chord (§2):
v = extract.build_vector_bank(model, tok, load_concept(trait), L)[L]["diffmean"]
# h' = h + alpha*v   (the chord; off-shell spikes at the chord midpoint for non-convex traits)

# FLAS arm — a learned transport F: h_neg -> h_pos that FOLLOWS the manifold (§5.2):
flow = train_mlp_flow(h_pos, h_neg, L)              # 2-layer MLP, ~100-200 steps (the ONLY trained part)
def flas_steer(h, flow, steps=10):                  # 10-step Euler integration of the flow ODE
    for _ in range(steps): h = h + (1/steps) * flow.vector_field(h)
    return h                                         # stays in-manifold by construction
```

### 2. Experiment procedure

```text
Step 1 convexity proxy: per trait, mean_pairwise_dist(h_pos)/convex_hull_diameter,
        AND a 2-component GMM cluster separation; convex if <0.5, non-convex if >0.7.
Step 2 select 2 convex + 2 non-convex traits.
Step 3 for each trait, model in {270M @L16, 1B @L18}:
         linear: hooks.apply_operation(h, v, "add"/"relative_add", alpha)
         FLAS:   h' = flas_steer(h, flow)
         for seed in 1..3:
           measure behavior, PPL, composite,
                   geometry.offshell_displacement   # at the chord midpoint vs flow midpoint (N5 bridge)
Step 4 spearman(convexity_proxy, flow_gain) across traits.
```

### 3. Measurement & decision rule

- PRIMARY metric: composite delta (FLAS − linear) stratified by trait convexity,
  plus Spearman(convexity, flow gain).
- Pre-registered FALSIFIER (§3): FLAS does NOT achieve `>= 15%` better composite on
  ANY non-convex trait ⇒ conditionality REJECTED; OR Spearman(convexity, flow gain)
  `< 0.5` ⇒ conditional framework itself rejected.
- Predicted (§6): non-convex traits `>= +0.20` composite / `>= 20%` lower PPL; convex
  traits `< +0.05` (chord already near-optimal).

### 4. Where the code is / status

The linear `add`/`relative_add` baseline and `geometry.offshell_displacement` exist.
MISSING: the **normalizing-flow module** (train + 10-step Euler integration), the
**convexity proxy** (hull-ratio + GMM), and held-out-prompt evaluation (avoid the
in-sample flow-fitting confound). This is the only Block-D hypothesis needing a trained
component beyond E20's SAE; `UNTESTED` and lowest priority in the block.

See [`../METHODOLOGY.md`](../METHODOLOGY.md) for the shared recipe.

---

## Provenance & Tracing

No experiments run yet — see this design doc's protocol (§7) for what would be run. Once a campaign logs rows for this hypothesis, re-run `scripts/build_provenance.py` to generate `hypotheses/PROVENANCE/E33.md`.
