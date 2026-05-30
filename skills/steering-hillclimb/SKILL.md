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

### The N5 norm-budget constraint

If cumulative ||delta h||/||h|| exceeds the empirical activation-norm quantile
(established by E22), the trial is DISCARDED even if composite improves. The
norm budget is a hard constraint, not a soft penalty.

---

## Quick checklist

- [ ] Starting from `best_config.json` (not from memory)
- [ ] Only one axis changes per trial
- [ ] Strict-greater-than champion rule (no ties accepted)
- [ ] Geometry leading indicators logged every trial
- [ ] N5 norm-budget constraint applied
- [ ] VERIFY.md signed before hill-climbing
- [ ] Plateau detection: 8 consecutive non-improvements -> stop
- [ ] Trial budget: <= 25 trials total

---

## Cross-references

- Meta-process: `../../meta-skills/autoresearch-per-hypothesis-hillclimb/SKILL.md`
- Cube definition: CLAUDE.md Section 8
- Norm budget: hypothesis N5 in `../../IDEA_TABLE.md`
- Participation ratio: hypothesis N3 in `../../IDEA_TABLE.md`
- Champion config: `../../autoresearch_results/best_config.json`
- Ledger: `../../EXPERIMENT_LEDGER.md`
