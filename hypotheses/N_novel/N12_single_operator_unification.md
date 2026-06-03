# N12 — The Single-Operator Unification (Capstone)

> **One-line claim:** CAST, Angular, CAA, SAE-TS, FLAS, and KV-cache steering are
> special cases of one operator h <- h + g(h) * Proj_T(Phi_t(v)) capped at budget B;
> implementing this operator and ablating its terms recovers each named method, and
> the fully-on configuration Pareto-dominates all of them on Gemma-3-1B.
>
> **Primary axes:** A1-A12 (all axes — capstone)
> **Status:** UNTESTED

---

## 1. Motivation (>= 100 words)

The LLM steering literature has converged on a set of method families that each
solve one piece of the problem: CAST handles conditional gating; Angular Steering
handles norm preservation; CAA handles simple additive injection; SAE-TS handles
sparse feature selection; FLAS handles curved transport; KV-cache steering handles
attention-pathway injection. Each paper evaluates against simple baselines,
not against the other specialist methods. The result is a fragmented toolkit where
practitioners must choose a method without a principled basis. This capstone hypothesis
argues that all these methods are special cases of one parameterized operator:

  h <- h + g(h) * Proj_T(Phi_t(v)) capped at B

where:
  g(h) is the gating scalar field (N2 — recovers CAST, SCS, discriminative-layer)
  Proj_T is the tangent-space projection (N1 — recovers additive when T=R^d, rotational when T=manifold)
  Phi_t is the flow transport at time t (recovers additive at t=0, geodesic at t=1)
  B is the norm budget cap (N5 — recovers uncapped when B=inf)

Setting each component to its "off" state recovers the specific named methods.
The "all-on" configuration should Pareto-dominate because each component addresses
an independent failure mode: g removes false positives (over-refusal), Proj_T keeps
the update on-manifold (coherence), Phi_t follows the geodesic (accuracy), B prevents
budget overflow (safety). This is the scientific endpoint of the N-block program:
the capstone experiment that validates or refutes the unified geometric framework.

## 2. Formal Hypothesis (>= 50 words)

Let U(g, T, Phi_t, B) denote the unified operator. The claim is:

(A) Ablation recovery: setting g=1 (constant), T=R^d (identity projection),
    Phi_t=I (identity flow), B=inf recovers CAA within 2% behavior efficacy and PPL;
    setting g=CAST, T=R^d, Phi_t=I, B=inf recovers CAST; and analogously for
    Angular (Proj_T = 2D-plane rotation), SAE-TS (Phi_t = feature flow), FLAS (Phi_t = curved flow).

(B) Pareto dominance: U(g=learned, T=local-tangent, Phi_t=angular, B=95th-percentile)
    achieves behavior efficacy >= max(CAA, CAST, Angular) AND PPL <= min(PPL_CAA, PPL_CAST, PPL_Angular)
    on Gemma-3-1B-it across 3 behaviors, 3 seeds.

## 3. Falsifier (>= 30 words)

If ANY named method that the unified operator claims to subsume is NOT recoverable
within 5% on both efficacy and PPL by ablating the appropriate components, claim (A)
is FALSIFIED. If the all-on configuration does not Pareto-dominate all component-ablated
versions, claim (B) is FALSIFIED. The capstone requires both claims to hold.

## 4. Citations (Citation Rigor >= 80 words)

```
Wurgaft et al. 2026. 'Manifold Steering' arXiv:2605.05115. Provides the M_h manifold
framework; Proj_T is the tangent projection onto M_h; Phi_t is the manifold geodesic
flow. N12 implements the discrete (one-step) version of their continuous flow.

Gao et al. 2026. 'CRH' arXiv:2605.01844 (ICML 2026). The radial/angular decomposition
is exactly the g(h) * Proj_T(v) split: g(h) controls the radial component (how far)
while Proj_T(v) controls the angular component (which direction).

Venkatesh & Kurapath 2026. 'Non-Identifiability' arXiv:2602.06801. The coset structure
means there exists a Proj_T that selects the min-collateral coset rep (N15); this is
the mathematically principled version of N12's tangent projection.

Turner et al. 2023. 'Activation Addition' arXiv:2312.06681. CAA = U(g=alpha, T=R^d,
Phi_t=I, B=inf): the simplest special case, recovered by setting all advanced
components to their "off" state.

Raval et al. 2026. 'Curveball Steering' arXiv:2603.09313. Curveball ≈ U(g=alpha,
T=local-manifold, Phi_t=curveball-flow, B=inf): sets the geometric components on
while keeping g constant (no conditional gating).
```

## 5. Mechanism

The operator is constructed as a chain of four independent transformations:

Step 1 — Gating: compute g(h) via the learned scalar field (N2's logistic gate).
  Range: [0, 1]. Off state: g(h) = alpha (constant).

Step 2 — Tangent projection: compute v_T = Proj_T(v) where T is the local tangent
  plane estimated from kNN-PCA (N1). Off state: Proj_T = I (identity).

Step 3 — Flow transport: compute Phi_t(v_T) = v_T * t + geodesic_correction * t^2 / 2
  for small t. At t=0: Phi_t = identity (CAA additive). At t=1: Phi_t = full
  geodesic step (Curveball). At t=0.5: half-step geometric correction.
  Off state: Phi_t = I (t=0).

Step 4 — Budget cap: apply delta = g(h) * Phi_t(v_T); if ||delta|| > B, scale to B.
  Off state: B = inf.

Update: h_new = h + delta.

The operator has 4 independent components; each can be independently "off" by
setting to its trivial state. This gives 2^4 = 16 ablation configurations, of which
the following recover named methods:
  (1,0,0,0) -> CAST (only gating)
  (0,0,0,0) -> CAA (all off = constant alpha)
  (0,1,0,0) -> Angular rotation (tangent projection to 2D plane = rotation)
  (0,0,1,0) -> Curveball steering
  (0,0,0,1) -> Norm-capped CAA (E22 recipe)
  (1,1,0,1) -> CAST + Angular + Budget (the E47 recommended recipe)
  (1,1,1,1) -> Unified (all on, Pareto-target)

KV-cache steering: operates at a different site (A1 axis); it is not a special case
of the single-layer operator but can be composed with it (the N12 operator handles
the residual stream; KV steering handles the attention cache independently).

## 6. Predicted Delta

| Metric | Predicted Delta (all-on vs best component alone) | Rationale |
|---|---|---|
| Behavior efficacy | +5% to +15% relative | All four components contribute independently |
| PPL at matched behavior | -10% to -20% relative improvement | Manifold-preserving components eliminate waste |
| Over-refusal (XSTest) | -20% to -40% relative | Gating component eliminates false positives |
| True-positive refusal | <= -3% loss | Gating preserves true positives |
| Off-shell displacement | -30% to -50% relative vs CAA | Budget cap + tangent projection |
| Ablation recovery accuracy | >= 95% for all named methods | Claimed exact subsumption |

## 7. Protocol

### 7.1 Primary experiment

This is the most expensive single experiment in the N-block (est. ~8-12 hours).
It is run LAST, after all component N-hypotheses have at least screening results.

- Model: Gemma-3-1B-it (primary), Gemma-3-270m (cross-scale)
- Behaviors: 3 behaviors (refusal/safety, politeness, factuality)
- Ablation grid: 16 binary configurations of (g, T, Phi_t, B)
- Named method recovery: compare each ablated config to the named method's
  reported performance in the corpus (within 5% tolerance)
- All-on Pareto test: compare (1,1,1,1) to all 16 configs on behavior x PPL space
- Evaluation: behavior-cosine shift, log-PPL, off-shell delta, over-refusal (XSTest),
  true-positive refusal (JailbreakBench)
- Seeds: 3 extraction x 3 evaluation = 9 cells per configuration = 144 total cells
- Wall-clock: ~10 hours on RTX 4090

### 7.2 Where it shines

The capstone shines most as a unifying UNDERSTANDING: even if the all-on configuration
does not Pareto-dominate on every metric, the ablation table provides the definitive
comparison of all existing methods in a controlled within-experiment setting, which is
more valuable than a collection of separate method papers with different evaluation setups.

## 8. Cross-References

- N1 (tangent projection): provides Proj_T component
- N2 (gating factorization): provides g(h) component
- N5 (norm-budget): provides budget cap B
- N7 (parallel transport): the Phi_t component in multi-layer settings uses transport
- N9 (closed-loop control): N9's P-controller is a time-adaptive version of N12's g(h)
- N13 (geodesic > chord): N12 tests this with the Phi_t component on/off
- N15 (coset min-collateral): Proj_T is the mechanism for selecting the min-collateral
  coset representative
- N16 (CRH radius/angle): the (g=radius, Proj_T=angle) decomposition IS the CRH decomposition
- N17 (concentration penalty): B enforces the norm budget that N17 discovered is critical
- IDEA_TABLE.md: N12 row, axes A1-A12 (all)

## 9. Committee Q&A

**Q: 16 ablation configurations x 9 cells = 144 experimental cells. This is
a large experiment. What is the minimal version that tests the core claim?**

> Minimal version: 4 configs (all-off = CAA, CAST-only, Angular-only, all-on)
> x 9 cells = 36 cells, ~3 hours. This tests whether all-on Pareto-dominates
> the two most studied baselines. The full 16-config ablation table is for
> completeness (which components contribute how much); it is run after the minimal
> version confirms the Pareto claim.

**Q: The "named method recovery" claim (>= 95% of named method efficacy) requires
that the operator implements each method correctly. How do you verify the
implementation without access to the original codebases?**

> For CAA: trivial (alpha*v). For CAST: implement the cosine-threshold gate.
> For Angular: implement the 2D-plane rotation. Implementation correctness is
> verified by comparing to the named method's expected outcome on a synthetic
> linear test case (tractable analytically), then on one real prompt. If the
> synthetic test passes, the implementation is correct.

**Q: KV-cache steering is listed as not a special case of the residual-stream
operator. Doesn't this limit the unification claim?**

> Yes. The unification covers axes A3-A12 fully and A1 (site) partially:
> the residual stream site is unified; the KV-cache site is a separate
> operator that CAN BE COMPOSED with N12 (as in E24's residual + KV-cache
> combination). The claim is "all residual-stream methods are special cases";
> KV-cache is a different architectural site.

## 10. Verification Checklist

- [ ] All four components implemented and individually tested:
      g(h) logistic gate, Proj_T kNN-PCA, Phi_t angular/geodesic, B norm-cap
- [ ] Synthetic test: verify CAA recovery (within 1% on linear test), CAST recovery, Angular
- [ ] 16 ablation configurations run; behavior and PPL logged for all 144 cells
- [ ] Named method recovery table: each method with recovery accuracy reported
- [ ] Pareto frontier plot: all 16 configs on behavior x PPL; all-on position labeled
- [ ] Cross-scale run on Gemma-3-270m (6 configs subset)
- [ ] Wall-clock and GPU memory logged
- [ ] Status promoted to "TESTED" in IDEA_TABLE.md N12 row with appropriate verdict
- [ ] Results cross-referenced in N1, N2, N5, N13, N15, N16, N17 status journals

## 11. Status Journal

- 2026-05-31 — Design doc created. Status: UNTESTED. This is the capstone experiment
  that the entire N-block program builds toward. Prerequisites: N1 (tangent projection
  implementation), N2 (gating implementation), N5 (norm-budget implementation) should
  all have at least screening results before N12 is run. Currently all three are UNTESTED
  or SUPPORTED(screening). N12 is the last experiment to run in the execution order.

---

## Addendum: Research-Scientist Critique

### Prior plausibility

MEDIUM. The operator decomposition is mathematically elegant and the ablation framework
is a clean experimental design. The practical risk is that the components interact:
optimizing each independently (N1 through N17) does not guarantee the all-on
configuration works, because the components may interfere (e.g., aggressive gating
may prevent the tangent projection from receiving meaningful activations, or the budget
cap may cut the geodesic correction before it completes).

### Mechanism scrutiny

The "Pareto dominance" claim requires that each component provides ADDITIVE value
when combined with the others. This is only guaranteed if the failure modes addressed
are INDEPENDENT. They may not be: over-refusal (gating) and off-manifold displacement
(tangent projection) are related, since over-refusal itself causes off-manifold
displacement. If the failure modes are correlated, fixing one may automatically fix
the other, and the all-on configuration provides diminishing returns over the best
single-component configuration.

### Confounds

1. Hyperparameter interaction: each component has its own hyperparameters (K_p for
   gating, k for kNN, t for flow, B quantile). These are tuned independently but
   may require joint tuning for all-on. A proper all-on configuration requires a
   small joint grid search.
2. Implementation bugs: a complex multi-component operator is harder to debug;
   a bug in one component may penalize the all-on configuration unfairly vs the
   simpler ablated versions.
3. The 16-config ablation with 9 cells each uses n=1 (screening); Pareto dominance
   claims at n=1 are not statistically reliable. The full rigor contract (n=7,
   Wilcoxon, bootstrap CI) requires 16 * 7 = 112 cells per behavior = 336 total
   cells, which is ~30 hours. Plan for this from the start.

### Does the unified formulation specifically matter?

HIGH. Beyond the scientific claim, the unified operator has engineering value:
one codebase, one API, one set of hyperparameters that subsumes all existing methods.
If the ablation table shows that the all-on configuration achieves the best Pareto
position, it motivates replacing the fragmented toolkit with the unified implementation
in any deployment. This is a rare case where scientific and engineering goals align.

### Literature precedent

Unified frameworks for sequence modeling (The Annotated Transformer, Vaswani 2017;
The Illustrated Transformer, Alammar 2018) enabled rapid adoption by unifying
scattered methods. N12 aims to do the same for steering methods. The closest
scientific precedent is conceptor theory (Jaeger 2014), which unifies AND/OR/NOT
operations for pattern storage.

### Skeptical effect-size estimate

Pareto dominance: 60-70% probability that the all-on configuration is on the Pareto
frontier (vs claimed "Pareto-dominates all"). It is unlikely that ALL four components
provide independent value; 2-3 of the four are likely to contribute meaningfully.
The most likely non-dominant configurations: all-on may match (1,1,0,1) (gating +
tangent + budget) with the flow component Phi_t providing marginal value, since
the tangent projection and budget cap together already handle the main off-manifold issue.

### Minimum distinguishing experiment

4 configs x 9 cells = 36 cells, ~3 hours: CAA (all-off), CAST-only, Angular-only,
and all-on (g=CAST, T=kNN, Phi_t=angular, B=90th). If all-on Pareto-dominates both
CAST-only and Angular-only, the capstone claim is supported at screening level;
proceed to full 16-config ablation table.

### Verdict

TESTABLE-HIGH-VALUE-HIGH-COST. The capstone is the correct final experiment;
it provides the unified picture that the program lacks. The 3-hour minimum (4 configs)
should be the first run; commit to the full 10-hour ablation only after the minimum
confirms the Pareto claim. The statistical power issue (n=1 screening) is real and
must be planned for: commit to n=7 runs for the final claim.

---

## Pseudocode & Methodology

This section specializes [`../METHODOLOGY.md`](../METHODOLOGY.md). N12 is the **capstone**: one operator `h ← h + g(h)·Proj_T(Φ_t(v))` capped at budget B subsumes CAA/CAST/Angular/etc.; ablating components recovers each. **UNTESTED** — it composes N1/N2/N5 machinery, which does not exist yet.

### 1. Steering-vector recipe (the four-component unified operator)

```python
# v = bank[L]["diffmean"]; four independently-switchable components:
g   = g_star(h)              if gating   else alpha        # N2 gate  (off = constant alpha)
v_T = Proj_T(v)              if tangent  else v             # N1 tangent (off = identity)
v_F = Phi_t(v_T)             if flow     else v_T           # N13 geodesic flow (off = t=0 = additive)
delta = g * v_F
delta = delta * (B / delta.norm())   if delta.norm() > B   # N5 budget cap (off = B=inf)
#   h_new = h + delta        (METHODOLOGY §2 add as the base operation)
```

Named-method recovery: `(0,0,0,0)=CAA`, `(1,0,0,0)=CAST`, `(0,1,0,0)=Angular`, `(0,0,1,0)=Curveball`, `(0,0,0,1)=norm-capped CAA`, `(1,1,1,1)=unified`.

### 2. Experiment procedure

```text
1. Implement & unit-test each component vs a synthetic linear case (CAA within 1%, CAST, Angular).
2. Run the 2^4 = 16 binary ablation grid; for each, log behavior, log-PPL, off-shell, XSTest over-refusal,
   JailbreakBench true-positive (eval.evaluate_bundle, METHODOLOGY §3).
3. Named-method recovery: compare each ablated config to the corpus number within 5%.
4. Pareto test: is (1,1,1,1) on the behavior x PPL frontier vs all 16? Cross-scale subset on 270m.
   Minimal version first: 4 configs (CAA / CAST / Angular / all-on) x 9 cells.
```

### 3. Measurement & decision rule

- **Primary metrics:** named-method recovery accuracy; Pareto position of the all-on config.
- **Pre-registered falsifier (§3):** any named method NOT recovered within 5% on both efficacy and PPL, OR all-on fails to Pareto-dominate component-ablated versions ⇒ FALSIFIED.
- **Verdict logic:** both claims (A recovery, B dominance) must hold; final claim needs n≥7 (CLAUDE.md §7).

### 4. Where the code is / status

UNTESTED — and **last in execution order**. The five-axis bundle exists, but every advanced component (`g*(h)` N2, `Proj_T` N1, `Φ_t` geodesic N13, budget cap N5) is missing or unconfirmed; the operator cannot be assembled until those prerequisites land. That is why N12 is UNTESTED.

See [`../METHODOLOGY.md`](../METHODOLOGY.md) for the shared recipe.

---

## Provenance & Tracing

No experiments run yet — see this design doc's protocol (§7) for what would be run. Once a campaign logs rows for this hypothesis, re-run `scripts/build_provenance.py` to generate `hypotheses/PROVENANCE/N12.md`.
