# Backlog — Parked Hypothesis Design Docs

These hypothesis design documents have UNTESTED or PENDING status in IDEA_TABLE.md.
They are parked here because the infrastructure they require does not yet exist.
They are NOT deliverables of the current sprint; they are NOT counted in the
hypothesis totals on the dashboard or in FINDINGS.md.

Return to the main hypotheses/ directory once the required infra is built and
the corresponding P0/P1 items in audits/reviews/IMPROVEMENTS_100.md are done.

---

## What each group needs before it can run

### A_foundations (E5, E6, E8)

- **E5** (4-bit vs fp16 invariance): needs a reproduced real-model run at both
  quantizations on a fixed behavior; infra exists but no run allocated.
- **E6** (over-steering linear probe): needs a calibrated probe on layer
  activations predicting pre-generation incoherence; no probe code yet.
- **E8** (IT->base transfer): needs real base-model weights and a reproduction
  run; base Gemma license and download needed.

### C_stacking (E19, E21, E23-E26)

- **E19** (Gram-Schmidt orthogonalization): requires multi-vector orchestration
  in cast.py (missing). This is also a P0 item in IMPROVEMENTS_100.md item 6.
- **E21** (Conceptor AND vs sum): requires conceptor implementation; no code.
- **E23** (same-plane vs orthogonal-plane composition): needs selective-plane
  rotation (as distinct from full-vector rotation tested in E27).
- **E24** (residual + KV-cache composition): needs KV-cache steering hooks;
  not in harness.
- **E25** (DoLa stacking): needs DoLa decoding-time integration; not in runner.
- **E26** (gate-before-behavior injection order): needs multi-vector ordering
  control in cast.py (missing).

### D_geometry (E29-E33)

- **E29** (geodesic interpolation): needs geodesic/spherical interpolation
  operation in hooks.py; not implemented.
- **E30** (adaptive rotation on partial-aligned tokens): needs per-token
  alignment check; not in harness.
- **E31** (rotation preserves activation norm): dependent on E27 selective-plane
  rotation (which E27 real-AxBench eval showed is a wash); low priority.
- **E32** (refusal vs detection direction separability): needs real refusal
  direction extracted (Rung 0 of METHOD_LADDER.md); gate must exist first.
- **E33** (curved flow vs linear on convex/non-convex): needs FLAS-style flow
  implementation; not in harness.

### E_mechanistic (E34, E37-E39)

- **E34** (refusal via OV circuit): needs attention-score freezing harness;
  not implemented.
- **E37** (interpretable != controllable): needs GemmaScope SAE features and
  causal objective dictionary; neither is wired.
- **E38** (task + style in one edit): needs ICL-derived function vectors;
  not extracted.
- **E39** (persona-vector drift monitoring): needs per-position persona-vector
  projection during generation; not in runner.

### N_novel (N1-N4, N6, N8-N15, N18, N19)

All require infrastructure or data that does not yet exist:

- N1 (tangent projection): needs geodesic/tangent-space operation (see E29).
- N2 (conditioning = curvature factorization): needs unified gate operator;
  cast.py not wired end-to-end.
- N3 (orthogonal capacity theorem): needs per-layer participation ratio vs
  stacking degradation data; requires N-vector stacking harness.
- N4 (steering as inverse ICL): needs ICL activation deltas vs DiffMean;
  no ICL extraction in harness.
- N6 (gate in read not write): needs cos(condition, behavior) = 0 enforcement;
  requires cast.py end-to-end.
- N8 (controllability != interpretability): needs GemmaScope SAE features.
- N9 (closed-loop proportional feedback): needs per-token control loop in runner.
- N10 (concept algebra closure): needs primitive-vector library and combination
  function; not implemented.
- N11 (curvature-aware alpha per prompt): needs local PCA spectrum per prompt;
  expensive and not in harness.
- N12 (single-operator unification, capstone): requires all prior methods as
  ablation components; the last hypothesis to run.
- N13 (geodesic > chord): needs geodesic operation (see E29).
- N14 (metric-matched operation): needs hyperbolic/cylindrical/spherical
  geometry variants; not in harness.
- N15 (coset min-collateral): needs Non-ID coset enumeration; not implemented.
- N18 (interference-budget additivity): needs N-vector stacking data beyond
  the 2-4 vectors tested so far; requires more stacking experiments.
- N19 (trajectory beats endpoint): needs per-layer distributed steering;
  hooks.py supports one layer at a time but not trajectory distribution.

---

## How to promote a doc back to hypotheses/

1. The required infra item(s) listed above are implemented and passing Rung 0.
2. The corresponding IMPROVEMENTS_100.md item is marked done.
3. The IDEA_TABLE.md row status is updated from PENDING to an active tier.
4. The doc is moved back with `git mv backlog/<group>/<file> hypotheses/<group>/`.
