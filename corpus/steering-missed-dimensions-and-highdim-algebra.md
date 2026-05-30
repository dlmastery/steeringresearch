# Missed Dimensions + High-Dimensional / Abstract-Algebra Findings
### Supplement to the steering treasure trove (literature search, May 2026)

> Purpose: A targeted literature sweep for intervention DIMENSIONS not yet captured by the
> seven-axis framework, plus first-principles findings from high-dimensional geometry and
> abstract algebra that should reshape how the autoresearch harness reasons about steering.
> Every arXiv ID below was verified directly on arxiv.org (not via Grok). Numbers inherited
> from abstracts are tagged where they need replication.

---

## PART 0 - TL;DR: what the old 7 axes MISSED

The previous framework (axes 1-7: WHERE, WHAT, HOW MUCH, HOW, WHEN, WHICH TOKENS, HOW DERIVED)
is implicitly EUCLIDEAN and STATIC. It assumes activation space is flat R^d, that a steering
direction is a fixed vector, and that 'add alpha*v' is the natural operation. A wave of 2026
geometry papers shows that assumption is the single biggest source of incoherence and rogue
damage. The missed dimensions are:

  A8. GEOMETRY / CURVATURE of the path  (straight chord vs geodesic on a curved manifold)
  A9. METRIC of the space               (Euclidean vs spherical vs hyperbolic vs cylindrical)
  A10. IDENTIFIABILITY / GAUGE          (which of many equivalent vectors you picked, and the null space you ignored)
  A11. DYNAMICS / TRAJECTORY            (one-shot shift vs trajectory-aware, multi-step control over the forward pass)
  A12. SUPERPOSITION / BASIS            (dense entangled basis vs sparse feature basis; interference budget)

So the framework grows from 7 to TWELVE axes. Axes 8-12 are not 'more knobs of the same kind' -
they change the SPACE the other 7 axes live in. They are meta-axes.

---

## PART 1 - VERIFIED NEW PAPERS (the geometry wave)

### 1.1 Manifold Steering  [arXiv:2605.05115, Wurgaft, Rager, Kowal, ... Goodman, Fel, Geiger, Lubana; Goodfire+Stanford+Harvard, May 2026]
Core claim: fit an activation manifold M_h to representations and a behavior manifold M_y to
output distributions; steering ALONG M_h yields behavioral trajectories that follow M_y, while
linear (Euclidean) steering cuts THROUGH off-manifold regions and produces unnatural outputs.
The bidirectional M_h <-> M_y link holds across reasoning tasks (cyclic/sequential/graph
geometries) and a video world model.
Money quote: it 'recasts the core problem of steering from finding the right direction to
finding the right geometry.' This is the thesis statement for axes 8-9.
DIRECTLY explains the Rogue Scalpel: off-manifold = exactly the nonlinear off-manifold knock-off
of the refusal ridge we diagnosed. Manifold steering is the PRINCIPLED version of our guard (A)+(B).

### 1.2 Curveball Steering: The Right Direction To Steer Isn't Always Linear  [arXiv:2603.09313, Raval, Song, Wu, Harrasse, Phillips, Barez, Abdullah; 2026]
Core claim: the optimal steering path is curved; respecting manifold curvature beats global
linear directions. Motivates geometry-aware nonlinear steering. (Axis 8: path curvature.)

### 1.3 Spherical Steering + 'Minimizing Collateral Damage in Activation Steering'  [ICML 2026]
Core idea: NORMALIZE the activation (project to a hypersphere of fixed radius), ROTATE on the
sphere, then rescale - preserving activation norm/scale. Effective-rank analysis shows spherical
rotation is more 'collapse-efficient' than additive steering (less representational collapse per
unit of behavior change). This is the same family as Angular Steering (2510.26243). (Axis 9: spherical metric; Axis 4: operation = rotate not add.)

### 1.4 Not All Latent Spaces Are Flat: Hyperbolic Concept Control (HyCon)  [arXiv:2603.14093, Briglia, Facchiano, Cursi, Sampieri, Rodola, ...; Mar 2026]
Core claim: many concept hierarchies are better modeled in HYPERBOLIC space (negative curvature),
where tree-like / hierarchical concepts embed with low distortion. HyCon does concept control in
hyperbolic geometry and reports increased stability ('geometric pacing under strong steering')
vs Euclidean, where directions are only locally valid. (Axis 9: hyperbolic metric.)

### 1.5 The Cylindrical Representation Hypothesis (CRH)  [arXiv:2605.01844, Gao, Zhang, Liu, ... Chen; MBZUAI, ICML 2026]
Core claim: representation differences arise from a linear combination of a RADIAL component and
an ANGULAR component - i.e., activation differences live on a CYLINDER, not a flat subspace.
Decomposing steering into (radius = magnitude/strength) x (angle = semantic direction) makes
steering more stable and predictable. Unifies why both 'how much' (radius) and 'rotate' (angle)
matter, and why coupling them (plain additive) is unstable. (Axis 9 metric + clean split of axes 2 and 3.)

### 1.6 GeoSteer: Faithful Chain-of-Thought Steering via Latent Manifold Gradients  [arXiv:2601.10229, Kazama et al.; 2026]
Core idea: steer reasoning by following gradients ALONG the latent manifold across CoT steps -
a trajectory-aware, manifold-gradient method. Reported +0.9 acc / +4.5 reasoning-quality points
avg vs baseline [NEEDS REPLICATION]. (Axis 11: dynamics/trajectory + Axis 8 geometry.)

### 1.7 Manifold-Guided Attention Steering (MAGS)  [arXiv, May 2026]
Trajectory-aware inference-time intervention grounded in a geometric observation about attention.
Intervenes on the attention pathway rather than residual-stream add. (Axis 1 site = attention; Axis 11 trajectory.)

### 1.8 On the Non-Identifiability of Steering Vectors  [arXiv:2602.06801, Venkatesh & Kurapath; Manipal, ICLR 2026 workshop]
Core claim (a THEORY result, not a method): under white-box single-layer access, steering vectors
are FUNDAMENTALLY NON-IDENTIFIABLE - there exist large equivalence classes of behaviorally
indistinguishable interventions. Orthogonal perturbations achieve near-equivalent efficacy; the
null-space dimensionality (estimated via SVD of the activation covariance) is large and the
equivalence is robust across prompt distributions. Conclusion: behavioral testing alone cannot
recover 'the' direction; you need STRUCTURAL constraints. (Axis 10: identifiability/gauge.)

---

## PART 2 - THE ABSTRACT-ALGEBRA / HIGH-DIMENSIONAL FINDINGS (first principles)

These are the underlying mathematical facts that the papers above are circling. The harness
should treat them as priors, not hypotheses.

### F-A. Concentration of measure: high-d Gaussians live on a thin shell.
In R^d with large d, the norm of a sample concentrates at radius ~ sqrt(d) with O(1) spread.
  ASCII:  density of ||h||
     |        ____
     |       /    \        <- almost ALL mass in a thin spherical shell
     |  ____/      \____      (NOT near the origin, NOT spread out)
     +----------------------> ||h||
            ~ sqrt(d)
Consequence: adding alpha*v moves you OFF the shell (changes the radius), which is itself
out-of-distribution. This is the algebraic reason norm-preserving (spherical) steering is gentler:
it slides along the shell instead of leaving it. Connects CRH (radius vs angle) and Spherical Steering.

### F-B. Near-orthogonality / Johnson-Lindenstrauss: room for exp(d) almost-orthogonal directions.
In high d you can pack exponentially many unit vectors that are pairwise near-orthogonal
(|cos| < eps). This is WHY superposition works (Anthropic toy-models) and WHY 817/1000 random-ish
SAE features can each jailbreak (Rogue Scalpel): the harmful-effect subspace is generic, not special.
  ASCII:  d=2 fits ~2 orthogonal dirs; d=4096 fits ~exp(c*4096) near-orthogonal dirs.
Consequence for stacking: two 'independent' safety vectors are near-orthogonal by DEFAULT, so
first-order they DON'T interfere - but their SECOND-order cross terms (off-manifold) can. Budget those.

### F-C. The manifold is curved; the tangent space is only a LOCAL linearization.
A steering vector v derived at point h0 is a tangent vector at h0. Walking alpha*v assumes the
manifold is flat for distance alpha. For small alpha that's fine (first-order); for the large
alpha needed to actually change behavior, curvature bites and you leave the data manifold.
  ASCII:    manifold M_h (curved)
      h0 *----v---->   (chord = linear steering, leaves M)
          \........    (geodesic = manifold steering, stays on M)
           '--._ M_h
This is Manifold/Curveball/GeoSteer in one picture. Rogue damage = the gap between chord and geodesic.

### F-D. The metric is not always Euclidean.
  - SPHERICAL (S^{d-1}): when only DIRECTION matters and norm is a nuisance scale -> Angular/Spherical Steering.
  - HYPERBOLIC (H^d): when concepts are HIERARCHICAL/tree-like; volume grows exponentially with
    radius, so trees embed with low distortion -> HyCon.
  - CYLINDRICAL (S^{d-1} x R): factor 'meaning' (angle) from 'intensity' (radius) -> CRH.
Choosing the metric is Axis 9. The right operation (Axis 4) is DERIVED from the metric:
Euclidean -> add; spherical -> rotate (geodesic = great circle); hyperbolic -> Mobius add / exp-map;
cylindrical -> (rotate angle) x (scale radius) independently.

### F-E. Gauge freedom / non-identifiability (the deepest one).
Behavior depends on h only through downstream maps. If part of the perturbation lies in the
null space of every downstream readout AT THAT LAYER, it is behaviorally invisible. So the set of
vectors producing a given behavior change is an AFFINE SUBSPACE (a coset of the null space), not a point.
  ASCII:  effective directions  =  v_min  +  Null(downstream readouts)
          (what you measured)      (the part that matters)   (a huge invisible subspace)
Implication: 'the refusal direction' is a GAUGE CHOICE. Interpretability claims built on one
recovered vector are under-determined (Non-Identifiability 2602.06801). For control this is
GOOD news: pick the representative of the coset that is most on-manifold / lowest collateral.
That is a free optimization the harness should always run (min collateral over the equivalence class).

### F-F. Superposition => steering has an interference budget.
Features are stored in superposition (more features than dimensions, near-orthogonal, F-B).
Pushing one feature leaks onto features whose vectors have nonzero overlap. The damage is the
sum of |alpha| * |cos(v, w_j)| * (sensitivity_j) over all other features j. This is the algebraic
form of the Rogue Scalpel's 'nonlinear interference.' Sparse (SAE) bases reduce per-step leakage
but do NOT eliminate it (817/1000 features still jailbreak). (Axis 12: basis choice + budget.)

---

## PART 3 - THE UPGRADED AXIS TABLE (7 -> 12)

  Axis  Name              Question                         Old default      New options unlocked
  ----  ----------------  -------------------------------  ---------------  ----------------------------------
  1     WHERE (site)      which layer/stream/head          1 mid layer      attention path (MAGS), multi-site
  2     WHAT (direction)  which direction                  1 fixed vector   coset representative (F-E)
  3     HOW MUCH (coeff)  magnitude                         flat alpha       radius axis of cylinder (CRH); PSR token-wise
  4     HOW (operation)   add/rotate/clamp/project          add              rotate/geodesic/Mobius (metric-derived)
  5     WHEN (condition)  gate on/off                       always           conditional (CAST)
  6     WHICH TOKENS      span                              all tokens       PSR token-specific
  7     HOW DERIVED       source of v                       diff-of-means    SAE / DIM / fit-to-manifold
  8     GEOMETRY (path)   chord vs geodesic                 chord (linear)   geodesic/curved (Manifold,Curveball,GeoSteer)
  9     METRIC (space)    which geometry                    Euclidean        spherical / hyperbolic / cylindrical
  10    IDENTIFIABILITY   which gauge rep + null space      ignore it        optimize over equivalence class (Non-ID)
  11    DYNAMICS (time)   one-shot vs trajectory            one-shot         trajectory-aware (GeoSteer, MAGS)
  12    BASIS / SUPERPOS  dense vs sparse + budget          dense, no budget sparse basis + interference budget

Axes 8-12 are META-axes: they define the arena in which axes 1-7 operate. Fixing axes 8-12 to
the Euclidean/static/single-vector defaults is exactly the 'naive additive steering' that the
Rogue Scalpel punishes.

---

## PART 4 - NEW FIRST-PRINCIPLES HYPOTHESES (extend N1-N12 with N13-N20)

N13 (Geodesic > chord). For matched behavior change, manifold/geodesic steering yields strictly
     lower off-manifold displacement and lower rogue-compliance than linear add. (Manifold,Curveball)
N14 (Metric-matched operation). The best operation is determined by the concept's geometry:
     hierarchical concepts -> hyperbolic; polar/intensity concepts -> cylindrical; directional
     traits -> spherical. A metric-mismatched operation costs coherence. Test by swapping metrics on the same trait.
N15 (Coset-min-collateral). Among all behaviorally-equivalent vectors (Non-ID coset), the one
     with minimal projection onto fragile mid-layer subspaces has the least alignment damage at
     equal efficacy. -> a cheap convex post-processing step on ANY derived vector.
N16 (Radius/angle decoupling, CRH). Steering that moves ONLY the angle (fixed radius) is more
     coherent than steering that moves both; the rogue damage scales with the radius excursion, not the angle.
N17 (Concentration penalty). Off-shell displacement |delta(||h||)| predicts incoherence better
     than raw ||alpha*v||. Norm-preserving steers should beat norm-changing steers at equal angle change.
N18 (Interference-budget additivity). Stacking k near-orthogonal safety vectors is safe iff the
     summed interference budget (F-F) stays below a threshold; degradation is predicted by sum of
     |cos| overlaps, not by k. -> a quantitative stacking law.
N19 (Trajectory beats endpoint). Distributing a fixed total intervention budget across the forward
     trajectory (small nudges per layer/step) produces less collateral than one large endpoint shift. (GeoSteer/MAGS)
N20 (Curvature as a fragility sensor). Local manifold curvature (or local effective-rank collapse)
     at a layer predicts that layer's rogue-fragility - giving a cheap, behavior-free way to pick safe layers.

---

## PART 5 - WHAT THE HARNESS SHOULD CHANGE TODAY

1. Add a NORM-PRESERVING (spherical) variant of every additive steer as a default comparison arm.
2. Add a COSET-MIN-COLLATERAL post-processing step (project the derived vector off fragile subspaces;
   F-E) - nearly free, applies to diff-of-means, SAE, DIM vectors alike.
3. Add an OFF-MANIFOLD / OFF-SHELL displacement metric (delta-radius, effective-rank drop) to the
   eval suite at EVERY benchmark rung - it is the leading indicator of rogue damage (N17, N20).
4. Add a CURVATURE / effective-rank probe per layer to auto-select safe injection sites (N20).
5. Treat metric (axis 9) as a swept hyperparameter for any concept that is hierarchical or polar (N14).
6. Replace 'find the direction' framing with 'find the geometry, then the coset rep, then the path'
   (Manifold Steering thesis).

---

## APPENDIX - VERIFIED arXiv IDs (this sweep)
  2605.05115  Manifold Steering (Wurgaft et al., Goodfire/Stanford/Harvard)
  2603.09313  Curveball Steering (Raval et al.)
  2603.14093  Hyperbolic Concept Control / HyCon (Briglia et al.)
  2605.01844  Cylindrical Representation Hypothesis / CRH (Gao et al., MBZUAI, ICML 2026)
  2601.10229  GeoSteer (Kazama et al.)
  2602.06801  On the Non-Identifiability of Steering Vectors (Venkatesh & Kurapath, ICLR 2026 ws)
  2510.26243  Angular Steering (prior sweep; spherical family anchor)
  Spherical Steering / 'Minimizing Collateral Damage in Activation Steering' (ICML 2026; ID pending direct verify)
  MAGS - Manifold-Guided Attention Steering (May 2026; ID pending direct verify)

All numbers from abstracts are pending independent replication on the 4090-laptop ladder.