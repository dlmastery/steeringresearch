---
name: steering-hillclimb
description: >
  Use when hill-climbing a surfaced steering candidate via coordinate descent
  over the steering cube. Enforces strict-greater-than champion rule, logs every
  trial to EXPERIMENT_LEDGER.md, and stops at 20-25 trials or when the cube is
  exhausted. This is the steering instantiation of the meta
  autoresearch-per-hypothesis-hillclimb process.
---

# Skill — steering-hillclimb

This is the steering instantiation of [meta-skills/autoresearch-per-hypothesis-hillclimb/SKILL.md](../../meta-skills/autoresearch-per-hypothesis-hillclimb/SKILL.md).

Read that file first for the general coordinate-descent loop. This file adds
the steering-specific cube definition and stopping rules.

---

## Steering-specific notes

### The steering cube

The hill-climb sweeps coordinate descent over this 6-dimensional cube (analogous
to the lr x wd x batch x opt x seed cube in supervised learning):

```
Axis 1: layer         — which residual-stream layer receives the intervention
Axis 2: alpha         — steering coefficient magnitude
Axis 3: source        — {diffmean, pca_top1, sae_selected}
Axis 4: operation     — {add, rotate, geodesic, project}
Axis 5: span          — {all_tokens, suffix_only, instruction_only, adaptive}
Axis 6: seed          — random seed for pair selection and eval sampling
```

This is derived from CLAUDE.md Section 8: "(layer x alpha x source[diffmean/PCA]
x operation[add/rotate] x span) x seed."

Extended cube for geometry-wave experiments (N13-N20, Blocks D+):
```
Axis 7: metric        — {euclidean, spherical, hyperbolic, cylindrical}
Axis 8: gauge_rep     — {naive_diffmean, coset_min_collateral, on_manifold}
Axis 9: trajectory    — {one_shot, distributed_N_layers, p_controller}
Axis 10: basis        — {dense, sae_reconstruction, causal_objective}
```

### Coordinate descent rules

1. Start from the current champion config in `autoresearch_results/best_config.json`.
2. Perturb one axis at a time. Try the three most promising axis values first
   (rank by prior experiments + corpus prior; never random).
3. Accept the new config iff composite is STRICTLY GREATER than the current
   champion (strict-greater-than; ties go to the champion — no drift).
4. Update `autoresearch_results/best_config.json` immediately on acceptance.
5. Continue until 20-25 trials are exhausted OR all candidate axis values have
   been tried for all axes (cube is exhausted).
6. Log every trial to EXPERIMENT_LEDGER.md, including rejected ones.

### L3 — No selection decisions at n=1 (adaptive seed allocation)

**Never make a champion-update decision on a single seed.** A single-seed
result is pure noise: you condition on the noise, and the apparent gain
regresses to the mean. The winner's curse is acute in steering because
behavior scores have high seed variance.

**Adaptive seed allocation within the hill-climb (successive-halving rule):**

| Stage | Seeds | Action |
|---|---|---|
| Initial comparison | n=1 per candidate | Narrow field: keep top 50% of candidates |
| Promoted candidates | n=3 | Retain only those that still beat the champion |
| Finalist (potential champion) | n=7 | Confirm before updating `best_config.json` |

- A trial at n=1 is EXPLORATORY. It can add to the log, but it cannot
  replace the champion. Only a trial confirmed at n ≥ 3 may replace the
  champion in the inner hill-climb loop.
- A trial confirmed at n ≥ 7 with the full four-part statistical contract
  (per [[steering-paper-rigor]]) is EVALUATION tier.
- The hard budget of 25 trials is over EXPLORATORY trials. Promotion to
  n=3 and n=7 is additional and is drawn from the rung's compute budget,
  not the trial budget.

**New harness support:** `src/steering/stats.py:seed_noise_band`,
`src/steering/stats.py:paired_wilcoxon`.

### L4 — Use a joint response surface when axes interact

Coordinate descent is correct when the objective surface is approximately
axis-aligned convex (each axis has a single optimum independent of the others).
In activation steering, **layer and alpha interact non-convexly**: the
coherence cliff location is layer-dependent, so the optimal alpha at layer L
is not the same as the optimal alpha at layer L+2.

**When to fit a joint surface:**

- If coordinate descent oscillates (the same axis reverses direction on the
  second pass), the surface is non-convex — switch to joint modeling.
- If geometry leading indicators (off-shell displacement Δ‖h‖, effective-rank)
  show a cliff that moves as the layer changes, the layer×alpha interaction is
  confirmed — fit the joint surface before any champion update.

**Joint-surface protocol:**

1. Allocate up to 15 of the 25-trial budget to a structured grid over
   (layer, alpha) — the two strongest interacting axes in steering.
2. Fit a quadratic or Gaussian-process (GP) response surface to the n=1
   composite values over this grid.
3. Use the geometric surrogate (off-shell displacement Δ‖h‖, which is a
   confirmed incoherence predictor) as a cheap inner-loop coherence proxy
   — constrain the fitted surface to the sub-region where Δ‖h‖ < 0.10.
4. Pick the predicted knee of the constrained surface analytically as the
   starting champion before resuming coordinate descent on the remaining axes.

The surrogate is cheap (one forward pass per trial for the geometry probe),
so it does not add materially to the compute budget. Use
`src/steering/controls.py` for the surrogate fitting utilities.

**Why this matters:** a coordinate-descent sweep that first optimizes layer
(holding alpha fixed at the default) will select a layer where the default
alpha is in the safe region. It then optimizes alpha at that layer — and may
find a higher composite, but it has never seen the joint landscape. The joint
surface reveals the cliff shape and prevents you from discovering the coherence
breakdown only after committing to a champion layer.

### Stopping conditions

Stop hill-climbing when:
- 20-25 trials are complete (hard budget), OR
- The composite has not improved for 8 consecutive trials (plateau), OR
- The cube is exhausted for the currently active axes.

Do NOT stop just because one trial improves dramatically — always exhaust the
neighboring coordinates before declaring a local optimum.

### Never hill-climb a broken implementation

If VERIFY.md is not signed or AUDIT.md has unresolved HIGH items, STOP.
Fix the implementation first; hill-climbing a broken implementation wastes the
trial budget and produces misleading results.

### Never hill-climb a NUMEROLOGY hypothesis

Per CLAUDE.md Section 7: a NUMEROLOGY verdict means "any nearby axis value would
do" (the steering analogue of the phi-numerology check). If the hypothesis is
NUMEROLOGY, discard it; do not hill-climb.

### Geometry leading indicators during hill-climb

Log delta_norm, eff_rank_drop, norm_budget, and part_ratio for EVERY trial.
These are cheap and provide the off-manifold signal needed by N5, N11, N20.
A trial that improves composite but increases delta_norm past the N5 budget
threshold is a FALSE POSITIVE — flag it and check.

### L5 — Pareto-climb, not scalar-climb

The fixed-weight composite is a **tiebreak summary**, not the sole champion
criterion. A fixed scalarization bakes arbitrary trade-off priors into every
ranking and can be Goodharted: a method that over-optimizes one cheap axis
(e.g., inflating behavior score at the cost of coherence) wins the composite
if that axis's weight is under-priced.

**The Pareto frontier is the primary scientific object:**

- Accept a move iff it **Pareto-dominates** the current champion (no axis
  regresses) OR **extends the frontier** at matched coherence (same coherence
  tier, behavior improves).
- Use the scalar composite as a tiebreak when two configs are not Pareto-
  comparable (one is better on behavior, the other on coherence).
- The per-hypothesis dashboard's **per-axis Pareto small-multiples** (behavior
  vs capability, behavior vs coherence, behavior vs safety) are the canonical
  view — never collapse to a single-axis ranking for a champion decision.
- At each rung, the promotion gate is Pareto-dominance over the prior champion
  on all five axes — not just a composite improvement.

Log the five-axis vector for every trial so the Pareto frontier can be
reconstructed from the EXPERIMENT_LEDGER at any point.

### The N5 norm-budget constraint

If cumulative ||delta h||/||h|| exceeds the empirical activation-norm quantile
(established by E22), the trial is DISCARDED even if composite improves. The
norm budget is a hard constraint, not a soft penalty.

---

## Quick checklist

- [ ] Starting from `best_config.json` (not from memory)
- [ ] Only one axis changes per trial
- [ ] Metric calibrated (ρ ≥ 0.70) against reference labels before first hill-climb
      (see [[steering-eval-bundle]] §10)
- [ ] Strict-greater-than champion rule — no champion update at n=1 (L3)
- [ ] Adaptive seed allocation: n=1 explore → n=3 promote → n=7 confirm
- [ ] Layer × alpha interaction checked; joint surface fitted if oscillation detected (L4)
- [ ] Geometry surrogate (Δ‖h‖) used as inner-loop coherence constraint (L4)
- [ ] Pareto-dominance check on ALL five axes before each champion update (L5)
- [ ] Geometry leading indicators logged every trial
- [ ] N5 norm-budget constraint applied
- [ ] VERIFY.md signed before hill-climbing
- [ ] Plateau detection: 8 consecutive non-improvements -> stop
- [ ] Trial budget: <= 25 exploratory trials total

---

## Cross-references

- Meta-process: `../../meta-skills/autoresearch-per-hypothesis-hillclimb/SKILL.md`
- Cube definition: CLAUDE.md Section 8
- Norm budget: hypothesis N5 in `../../IDEA_TABLE.md`
- Participation ratio: hypothesis N3 in `../../IDEA_TABLE.md`
- Champion config: `../../autoresearch_results/best_config.json`
- Ledger: `../../EXPERIMENT_LEDGER.md`
- Metric calibration + off-family judge: `../steering-eval-bundle/SKILL.md` §10–11
- Statistical rigor floor + power requirements: `../steering-paper-rigor/SKILL.md`
- Controls (matched-norm random, shuffled-label): `../steering-experiment/SKILL.md` §Controls
- Harness modules: `src/steering/stats.py` (seed_noise_band, paired_wilcoxon),
  `src/steering/controls.py` (surrogate fitting, matched_norm_random)
