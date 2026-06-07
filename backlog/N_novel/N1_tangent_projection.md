# N1 — The Steering Manifold Tangent Hypothesis

> **One-line claim:** Projecting an additive steering vector onto the local tangent
> plane of the activation manifold (estimated from a kNN patch of natural activations)
> restores additive steering's coherence to match rotational-method quality, unifying
> additive and rotational steering as flat vs curved approximations of a single
> tangent-space translation.
>
> **Primary axes:** A8 (geometry/path), A9 (metric/space)
> **Status:** UNTESTED

---

## In Plain English

**What we're testing, simply:** As the model "thinks," its internal state is a
point floating in a huge space. Healthy thoughts don't float just anywhere — they
sit on a gently curved surface, like cities sit on the curved surface of the Earth.
Steering nudges the point. If we shove it in a straight line, we push it *off* that
curved surface (the thought goes "off-world" and the text breaks). This idea says:
before nudging, bend the nudge so it slides *along* the surface instead of off it —
and that should keep the text readable while still changing the behavior.

**Key terms (defined here):**
- **Steering / steering vector** — instead of retraining the model, we add a chosen
  direction to its internal state mid-sentence to change how it behaves. That
  direction is the steering vector.
- **Residual stream** — the model's running internal "thought" as it flows through
  its processing steps; what we edit.
- **Layer** — one processing step (this doc edits at step 16). Which step we pick is a knob.
- **alpha / strength** — how hard we push. Small = gentle, large = a shove that breaks the text.
- **Coherence** — whether the text stays fluent and sensible rather than turning to gibberish.
- **Manifold / surface** — the natural curved region where healthy thoughts live.
- **Tangent** — a direction that slides *along* that surface rather than off it.
- **Norm** — the length (size) of the thought-point.
- **Off-shell displacement** — how far a nudge knocks the thought off its healthy size/surface.
- **Geodesic** — the shortest path that stays *on* the curved surface (a rotation-style move).
- **kNN / PCA** — a cheap recipe for guessing which way the surface points locally, by
  looking at the nearest natural thoughts.

**Why we're doing this (the point):** If additive steering only breaks text because
it leaves the surface, then trimming away the "off-surface" part of the nudge should
let cheap addition behave as gently as the fancier rotation methods.

**What the result would mean:** A win means we get rotation-quality smoothness almost
for free. A loss (the falsifier) means trimming the nudge doesn't help — or only helps
because it made the nudge smaller, not because of the geometry.

See [`../GLOSSARY.md`](../GLOSSARY.md) for any other term.

---

## 1. Motivation (>= 100 words)

Activation steering methods divide into two families: additive methods (CAA, DiffMean,
function vectors) that simply add alpha*v to the residual stream, and rotational methods
(Angular Steering, Spherical Steering, CRH angle-only) that move the activation along
the surface of a sphere. Practitioners treat these as categorically different algorithms.
But the geometric picture from Manifold Steering (arXiv:2605.05115) suggests they are
the same operation viewed at different levels of approximation. Any smooth manifold is
locally flat: in the infinitesimal limit, tangent-space addition and geodesic motion on
the manifold are identical. The divergence emerges when alpha is large enough that the
curvature of the manifold becomes perceptible — i.e., when the chord (additive vector)
deviates appreciably from the geodesic. On high-dimensional manifolds with low intrinsic
dimension, curvature effects can appear at surprisingly small absolute distances because
the ambient directions off the manifold are numerous. This is precisely the regime small
LLMs (270M–1B parameters) occupy: the activation manifold has lower intrinsic dimension
relative to the ambient space, so the curvature is felt earlier. If that geometric
account is correct, the remedy is not to switch algorithms but to project — project the
steering vector onto the estimated tangent plane before applying the additive update.
This tangent-projected additive update should be indistinguishable from a geodesic step
to first order in curvature, recovering the coherence advantage of rotational steering
without its computational overhead.

## 2. Formal Hypothesis (>= 50 words)

Let T(h) be the local tangent plane of the activation manifold M_h at the current
activation h, estimated by PCA of the k nearest natural activations to h. Define
v_T = Proj_{T(h)}(v). The claim is:

  PPL(h + alpha * v_T) ≈ PPL(R_theta(h))

at matched behavior displacement, where R_theta is a selective 2D-plane rotation
achieving the same behavior-cosine shift. Specifically, the PPL gap between
tangent-projected additive and angular rotation is less than 5% of the gap between
naive additive and angular rotation, on Gemma-3-1B @L16, for alpha/||h|| in [0.05, 0.20].

## 3. Falsifier (>= 30 words)

If tangent-projected additive steering's log-PPL exceeds naive additive log-PPL at
matched behavior cosine for any alpha/||h|| in [0.05, 0.15] on Gemma-3-1B @L16 across
3 behaviors and 3 seeds, the projection hypothesis is DISCARDED. Status moves to
`FALSIFIED`. If the PPL gap closure is less than 20% (vs the 95%+ gap closure claimed),
the hypothesis is downgraded to `INCONCLUSIVE`.

## 4. Citations (Citation Rigor >= 80 words)

```
Wurgaft, Rager, Kowal, et al. (Goodfire/Stanford/Harvard) 2026.
'Manifold Steering' arXiv:2605.05115 (ICML 2026). Directly establishes
the bidirectional activation manifold M_h <-> behavior manifold M_y link;
coins the framing "finding the right geometry" over "finding the right direction."
This paper is the primary theoretical anchor for N1: its Theorem 1 shows that
steering along M_h tracks M_y trajectories, while Euclidean chords diverge.

Gao, Zhang, Liu, et al. (MBZUAI) 2026.
'The Cylindrical Representation Hypothesis' arXiv:2605.01844 (ICML 2026).
CRH decomposes activation differences into radial (magnitude) and angular (direction)
components; the screening result S-7 (R2=0.997 for angular predicting rotation logPPL)
supports the geometric separation claim that underlies N1's tangent-projection rationale.

Venkatesh & Kurapath (Manipal, ICLR 2026 workshop).
'On the Non-Identifiability of Steering Vectors' arXiv:2602.06801. Shows that
large cosets of behaviorally-equivalent vectors exist; tangent projection selects
the coset representative closest to M_h, making N1 a special case of N15's
coset-min-collateral principle.

Raval, Song, Wu, et al. 2026.
'Curveball Steering' arXiv:2603.09313. Empirically demonstrates that curved paths
outperform linear; N1 is the zero-order version (tangent plane vs full geodesic flow).
```

## 5. Mechanism

The residual stream h at layer L lives on a data manifold M_h of intrinsic dimension
d_int << d_model. A DiffMean vector v is a chord from h to the steered region; adding
alpha*v walks along this chord. When the manifold is curved, after step alpha the new
point h + alpha*v lies off M_h by a distance proportional to the curvature kappa times
alpha^2. The tangent-projection step replaces v with v_T = v - (v . n) * n for each
normal direction n at h, keeping only the manifold-parallel component. After this
projection:

  ||h + alpha*v_T - exp_h(alpha*v)||  <=  (1/2) * kappa_max * alpha^2 * ||v_T||^2

which is second-order in alpha, vs first-order for naive additive. The kNN tangent
estimation: collect the k=50 nearest activations {h_1,...,h_k} from a corpus cache,
center, and PCA. The top d_int eigenvectors (d_int chosen by 99% variance threshold)
span T(h). Projection cost: one matrix-vector multiply of size d_int x d_model -- O(10^6)
operations vs the O(d_model^2) forward pass -- negligible latency.

Connection to rotational steering: a 2D-plane rotation in the plane spanned by {h, v}
is equivalent to tangent projection when the plane IS the tangent plane (which holds
when the behavior direction lies in the manifold). N1 predicts that when the manifold
dimension is much smaller than d_model, tangent projection recovers most of the angular
steering advantage, since the out-of-manifold component of v is projected away exactly
as the spherical constraint discards the radial component.

## 6. Predicted Delta

| Metric | Predicted Delta (vs naive additive) | Rationale |
|---|---|---|
| log-PPL gap closure vs angular | >= 80% | tangent projection removes the leading curvature error term |
| Behavior cosine at matched alpha | +/- 2% of naive additive | direction preserved post-projection |
| Computation overhead | < 1% forward-pass time | O(d_int * d_model) with d_int ~ 50-200 |
| Gap closure on 270m model | >= 70% | more curved manifold -> larger absolute gain |
| Gap closure on 1B model | >= 60% | less curved but still significant |

## 7. Protocol

### 7.1 Primary experiment

- Model: Gemma-3-1B-it @L16 (identified as optimal layer in C9b)
- Behaviors: 3 contrastive pairs from the synthetic suite (polite/rude, factual/evasive,
  concise/verbose)
- Vectors: DiffMean extracted from 50 contrast pairs each
- Conditions: (a) naive additive alpha*v, (b) tangent-projected alpha*v_T,
  (c) 2D-plane angular rotation at matched behavior cosine
- kNN tangent: k=50 natural activations from WikiText-103 validation set; PCA with 99%
  variance cutoff to get T(h)
- Alpha range: [0.01, 0.02, 0.05, 0.10, 0.15, 0.20] * ||h||
- Metrics: log-PPL (WikiText continuation), behavior cosine, off-shell delta-||h||
- Seeds: 3 extraction seeds x 3 prompt seeds = 9 cells per condition
- Wall-clock estimate: ~2 hours on RTX 4090

### 7.2 Where it shines

This experiment shines on small models (270M, 1B) where the additive-vs-rotational gap
(observed in S-7/S-4) is largest. Run on Gemma-3-270m-it first as the positive-control
model; the gap should be most recoverable here.

## 8. Cross-References

- N5 (norm-budget): tangent projection does NOT directly conserve norm; combine with
  norm-cap for full coherence
- N16 (CRH radius/angle): tangent projection approximates angle-only steering (the radial
  component of v is partly in the normal-to-manifold subspace for curved M_h)
- N17 (concentration penalty): off-shell displacement is the metric; tangent projection
  should reduce it
- N15 (coset min-collateral): tangent projection selects a specific coset rep; coset
  optimization (N15) is the richer generalization
- N12 (capstone): tangent projection is the Proj_T(.) term in the unified operator
- S-7: additive vs rotation gap quantified; N1 predicts the gap closes with projection
- IDEA_TABLE.md: N1 row, axes A8+A9

## 9. Committee Q&A

**Q: Is kNN-based tangent estimation computationally feasible at inference time?**

> Yes. The kNN cache is pre-computed offline from a corpus pass; at inference only the
> matrix multiplication Proj_T(v) is needed — O(d_int * d_model) with d_int typically
> 50-200. This is < 1% of the MHA forward pass cost. Alternatively, the PCA basis can
> be pre-cached at each layer and updated infrequently.

**Q: Doesn't this just reduce to a weaker version of full geodesic steering?**

> Yes, intentionally. N1 claims the first-order correction (tangent projection) already
> captures most of the PPL improvement, making it practically useful without the full
> computational overhead of flow-based geodesic methods (N13, CurveballSteering).
> If the 80% gap closure prediction holds, tangent projection is Pareto-superior on
> cost/benefit.

**Q: The tangent plane is estimated from corpus activations at the same layer — but
the steered h might be far from any corpus point. Then the tangent is wrong.**

> This is the key vulnerability. The falsifier includes checks at alpha/||h|| > 0.15
> where this is most likely. If the tangent estimate degrades at large alpha, the method
> should be coupled with small-step iterative application (connecting to N19 trajectory).

**Q: Isn't S-7 evidence that full-vector rotation HURTS coherence vs addition?**

> S-7 found that full-vector rotation (rotating the entire d_model-dimensional vector)
> hurts; Angular/Selective steering rotates only in a 2D behavior plane. N1 predicts
> that tangent projection achieves the selective effect automatically: out-of-manifold
> components are trimmed, leaving approximately the 2D-plane component intact.

## 10. Verification Checklist

- [ ] kNN tangent cache implemented and passing shape tests (d_int recovered correctly)
- [ ] Projection correctness: Proj_T(v) is orthogonal to all normal vectors in test
- [ ] Behavior cosine preservation test: |cos(v, v_T)| >= 0.85 for all behaviors
- [ ] PPL gap closure metric defined and logged as `(PPL_naive - PPL_proj) / (PPL_naive - PPL_rot)`
- [ ] 3 behaviors x 6 alpha values x 9 seeds = 162 cells recorded in EXPERIMENT_LEDGER.md
- [ ] Off-shell displacement measured and compared to N17 master curve
- [ ] Cross-model run on Gemma-3-270m-it confirming larger gap closure
- [ ] Result reflected in IDEA_TABLE.md N1 row status

## 11. Status Journal

- 2026-05-31 — Design doc created. Status: UNTESTED. No data exists for this hypothesis.
  The S-7 screening result (additive gentler than full-vector rotation) motivates the
  test but does not address tangent projection specifically.

---

## Addendum: Research-Scientist Critique

*Reviewer: SciCritic-N (elite steering-geometry critic). Critiquing the IDEA.*

### Prior plausibility

MEDIUM. The geometric reasoning is sound in principle — tangent-space approximations
are the standard first step in Riemannian optimization (see Riemannian SGD, RSGD).
The question is whether the activation manifold has enough curvature at behavioral
scales (alpha/||h|| ~ 0.1) for the correction to matter. Screening result S-6
showed the off-shell law log PPL = 5.40 + 2.87*offshell (R2=0.81), which is
consistent with curvature effects; but it does not isolate whether curvature or
simply leaving the data distribution is the mechanism.

### Mechanism scrutiny

The mechanism requires: (a) the activation manifold is smooth enough that a linear
tangent approximation is useful at steering scales, (b) the kNN-estimated tangent
plane is an accurate proxy for the true manifold tangent, (c) the in-plane component
of the behavior vector is sufficient for the behavior change (not largely out-of-plane).
Point (c) is the critical unknowable — if behavior directions are primarily
off-manifold (which would be surprising given they were extracted from natural
activations), tangent projection would gut their efficacy.

### Confounds

1. Norm confound: tangent projection reduces ||v_T|| < ||v||, which itself reduces
   off-shell displacement (N17). The mechanism comparison must hold alpha * ||v_T||
   fixed, not alpha * ||v||, to isolate projection vs norm reduction.
2. Dimensionality estimation: d_int chosen by 99% variance may include many noise
   dimensions if the manifold is not cleanly low-rank at the injection layer.
3. Distribution of natural activations: the kNN cache is from WikiText; if prompt
   distribution is different, the tangent is estimated at the wrong point.

### Does the specific geometry claim matter?

PARTIALLY. The claim that tangent projection specifically (rather than, say, norm
normalization or any dimensionality-reducing projection) is the right operation
needs the ablation: norm-normalized v (not projected) vs tangent-projected v.
If they match, the effect is just "reduce magnitude of out-of-distribution component,"
not the specific geometric claim.

### Literature precedent

Riemannian optimization (Absil et al. 2008, "Optimization Algorithms on Matrix
Manifolds") uses exactly this tangent projection; applying it to inference-time
steering is novel but the tool is classical. GeoSteer (arXiv:2601.10229) follows
manifold gradients across CoT steps but does not isolate the tangent-projection
correction for single-layer additive steering.

### Skeptical effect-size estimate

My prior: gap closure 20-50% (vs claimed 80%). Rationale: (a) the kNN tangent
will be noisy, (b) behavior vectors extracted from natural activations likely
have substantial in-manifold component already, so projection removes only a
small fraction, (c) the norm reduction confound is real. Still worth testing —
even 20% gap closure at negligible cost is practically valuable.

### Minimum distinguishing experiment

Gemma-3-1B @L16, one behavior, 6 alpha values, 3 seeds: naive additive vs
tangent-projected additive vs norm-normalized additive (same ||v||_T as v_T
but direction of original v). Cost ~30 min. If tangent-projected beats
norm-normalized at same alpha, the geometry claim is supported beyond the
trivial norm-reduction explanation.

### Verdict

TESTABLE-MEDIUM-CONFIDENCE. The geometric claim is falsifiable with a single
ablation adding ~30 min to an existing sweep. The 80% gap closure prediction
is overconfident; 30-60% is a more realistic prior. The norm-reduction confound
is the primary threat to interpretation and must be controlled explicitly.

---

## Pseudocode & Methodology

This section specializes [`../METHODOLOGY.md`](../METHODOLOGY.md). N1 modifies the **direction** of an additive edit: it projects v onto the local manifold tangent before adding. **UNTESTED** — it needs a local-tangent estimator that does not yet exist.

### 1. Steering-vector recipe (tangent-projected additive)

Start from the ordinary DiffMean vector (METHODOLOGY §1.3), then trim its off-manifold component:

```python
v = bank[L]["diffmean"]                                  # extract.diffmean_vector

# local tangent T(h): PCA of the k=50 nearest natural activations to h (NEW machinery)
neighbors = knn_natural_activations(h, L, k=50)          # MISSING: kNN cache
T = pca_basis(neighbors, var=0.99)                       # d_int top components span T(h)
v_T = T @ (T.T @ v)                                      # Proj_T(v): keep manifold-parallel part

# inject v_T additively (METHODOLOGY §2, add):  h' = h + alpha * v_T
# geometry logged: offshell_displacement(h, h') should DROP vs naive add; angular_displacement preserved
```

### 2. Experiment procedure

```text
1. Extract DiffMean v at L16 for 3 behaviors; build the kNN tangent cache from WikiText activations.
2. For each alpha/||h|| in {0.01,0.02,0.05,0.10,0.15,0.20}, run THREE matched conditions:
     (a) naive additive  h+alpha*v
     (b) tangent-proj     h+alpha*v_T
     (c) 2D-plane angular rotation (METHODOLOGY §2 rotate) at matched behavior cosine
   plus a CONFOUND control: norm-normalized v at ||v_T|| (isolates projection vs magnitude).
3. Measure log-PPL, behavior cosine, offshell_displacement for each.
4. Gap-closure = (PPL_naive - PPL_proj) / (PPL_naive - PPL_rot).
```

### 3. Measurement & decision rule

- **Primary metric:** PPL gap-closure of tangent-projection toward angular rotation.
- **Pre-registered falsifier (§3):** PPL(proj) > PPL(naive) at matched behavior across 3 behaviors/3 seeds ⇒ DISCARDED; gap-closure < 20% ⇒ INCONCLUSIVE (claim is ≥80%).
- **Verdict logic:** KEEP only if proj beats the norm-normalized control (geometry claim ≠ magnitude reduction).

### 4. Where the code is / status

UNTESTED. Reuses `geometry.offshell_displacement` / `angular_displacement`, but the **local-manifold tangent estimator** (kNN cache + per-point PCA + `Proj_T`) is not implemented — that missing machinery is why N1 cannot run.

See [`../METHODOLOGY.md`](../METHODOLOGY.md) for the shared recipe.

---

## Provenance & Tracing

No experiments run yet — see this design doc's protocol (§7) for what would be run. Once a campaign logs rows for this hypothesis, re-run `scripts/build_provenance.py` to generate `hypotheses/PROVENANCE/N1.md`.
