---
name: autoresearch-per-hypothesis-hillclimb
description: Use after a single-config screening sweep has produced a candidate hypothesis, before any external claim is made about it. Runs a 20-25-trial coordinate-descent hill-climb over the project's tunable hyperparameter cube, generates a per-hypothesis dashboard, and statistically confirms the best config with a 3-seed re-run.
---

# Skill — Per-hypothesis hill-climb (the proper evaluation tier)

## When to use

- **AFTER** a screening pass (e.g., `autoresearch-ablation-sweep` or
  the project's equivalent sweep script) has surfaced a candidate
  hypothesis with a notable single-config result on the development
  split — positive (KEEP) or even mildly negative (NEAR-MISS).
- **BEFORE** any external claim is made: a FINDINGS headline, paper
  abstract, README badge, or graduation to a harder evaluation rung.
  Screening alone is NOT sufficient for an external claim.
- After a fixer campaign patches a hypothesis: re-screen first, then
  hill-climb to re-establish the tuned ceiling.
- Whenever the user asks to "tune", "hill-climb", "properly evaluate",
  "confirm with seeds", or "promote a hypothesis to a winner".

## The lesson this skill encodes

Early-project evaluations are typically **screening-only**: each
candidate runs at the project default hyperparameter recipe with all
other axes frozen. That conflates "bad hypothesis" with "good hypothesis
at the wrong config". A negative screening result is a statement about
the default recipe, not about the hypothesis itself.

The corrective is a two-stage funnel:
  cheap screening at default config
  → coordinate-descent hill-climb over the tunable cube
  → statistical confirmation with multiple seeds
  → external claim.

Treat screening as directional evidence only. The hill-climb is the
minimum tier before any quantitative claim leaves the project.

## The project's tunable hyperparameter cube

The exact axes and their values are defined in the project's sweep
configuration. As a template, a typical cube covers:

| axis | example values | rationale |
|---|---|---|
| primary learning rate | {low, mid, high} | log-uniform window centred on the default |
| regularisation strength | {low, mid, high} | log-spaced, bracketing the default |
| batch / sequence size | {small, default} | coupling with the learning-rate scale |
| optimiser family | {adaptive, momentum-SGD} | the two dominant families with documented baselines |
| seed | {0, 1, 2} | reserved for the final confirmation; coordinate descent stays on seed 0 |

The full Cartesian product of a 3×3×2×2 cube is 36 cells. Coordinate
descent reaches a near-optimal result in **20–25 runs** with negligible
regret (Bergstra & Bengio 2012 JMLR — random search beats grid in fewer
trials for smooth HP landscapes; Hutter et al. 2014 — sequential
model-based optimisation).

Document the cube axes and values in
`ideas/<NN>/hillclimb_config.yaml` before launching.

## Coordinate-descent algorithm (default)

```python
# Pseudocode — adapt axis names to the project's config schema
state = base_config_from_screening(tag)        # project-default recipe
best_metric = run(state)                        # 1 run

for axis in [lr, regularisation, optimiser, batch_size]:
    candidates = cube[axis] - {state[axis]}     # skip the already-used value
    for v in candidates:
        cand = state.copy(); cand[axis] = v
        metric = run(cand)                      # +1 run per candidate
        if metric > best_metric:               # strict-greater (champion rule)
            state, best_metric = cand, metric
    # advance to the next axis at the current best

# Final 3-seed confirmation at the hill-climbed best config
for seed in {0, 1, 2}:
    if seed not in seeds_already_run_at(state):
        run({**state, seed: seed})
```

**Strict-`>` champion rule:** a tied later candidate does NOT replace
the champion. This protects against seed noise being mistaken for a
real improvement. The same discipline applies to the project's broader
hill-climb framework.

Total runs = 1 (start) + 2 (lr alternates) + 2 (regularisation
alternates) + 1 (optimiser alternate) + 1 (batch alternate) + 2 (seeds
1 and 2 for confirmation) = **~9 runs minimum**. With a mini second
pass on the lr and regularisation axes after an optimiser flip (the
two optimiser families often occupy different optimal lr scales), the
typical campaign reaches **20–25 runs**.

## Alternative algorithms

The hill-climb script should support at least three search flavours:

| algorithm | when to use | typical run count |
|---|---|---|
| `coordinate` | DEFAULT — smooth landscape, axis-decomposable | 20–25 |
| `random` | Suspected interactions between axes (lr × regularisation) | `--budget` (default 25) |
| `grid` | Validation / one-time sanity check on a pilot hypothesis | full cube size |

`random` is the recommended fallback when coordinate descent surfaces a
multi-modal landscape (e.g., the optimal lr changes when the optimiser
family flips). When in doubt, run coordinate first, inspect the
per-axis Pareto plot in the per-hypothesis dashboard, and re-launch
with `--algorithm random` only if the coordinate sweep shows
non-monotone responses.

## Output contract

### Per cell

Each hyperparameter cell writes to a deterministically named directory:

```
experiments/<split>/<tag>__hc_<config_hash>_seed<SEED>/
├── metrics.json     # RunMetrics + extra: config_id, hc_axis, hc_step
├── history.json
└── checkpoint.pt   (or project's checkpoint format)
```

`config_id` is a deterministic hash of the non-seed axes so the
per-experiment dashboard renderer can group seeds under the same config.

### Per hypothesis

Top-level summary at `ideas/<NN>/hillclimb_results.json`:

```jsonc
{
  "tag": "<hypothesis_short_name>",
  "algorithm": "coordinate",
  "budget": 25,
  "cube": {"lr": [...], "regularisation": [...], ...},
  "base_config": {"lr": ..., "reg": ..., "batch": ..., "opt": "..."},
  "best_config":  {"lr": ..., "reg": ..., "batch": ..., "opt": "..."},
  "best_composite_median": 0.xxxx,
  "best_composite_min":    0.xxxx,
  "best_composite_range":  0.xxxx,
  "seeds_confirmed":  [0, 1, 2],
  "cells": [
    {"config": {...}, "seed": 0, "composite": 0.xxxx, "wallclock_s": ...},
    ...
  ],
  "fingerprint": "<COMPOSITE_FORMULA sha256 first 12 hex chars>"
}
```

### Per-hypothesis dashboard

`ideas/<NN>/dashboard/index.html` — a standalone page, NOT a modal off
the master dashboard. It must contain:

1. **Best-config callout** — top of page, monospace box, showing the
   winning axis values and composite median ± half-range over seeds.
2. **Cells table** — sortable, one row per cell: config_id, each axis
   value, seed, composite metric, delta versus best, wall-clock,
   link to the per-experiment page.
3. **Per-axis Pareto plot** — small-multiples PNG, one panel per axis,
   composite on y, axis value on x, colour-coded by the other fixed
   axes. Reveals non-monotone responses. (PNG not SVG — per the
   typography hard rule in autoresearch-typography-and-rendering.)
4. **Seed-stability bar chart** — at the best config: composite for
   each confirmation seed, with the standard deviation annotated.
5. **Footer** — base config, hill-climb algorithm, total wall-clock,
   composite formula fingerprint, link back to the master dashboard.

## Statistical confirmation gate

A candidate hypothesis is `EXTERNAL-READY` only when:

```
best_composite_min(3-seed) > baseline_composite_max(3-seed)
```

i.e., the WORST seed at the tuned config beats the BEST seed of the
project-default baseline on the same split. The headline number is the
median composite; the gate uses the minimum.

A hill-climb result that fails this gate is recorded as `NEAR-MISS` in
the project's FINDINGS document and is NOT promoted to the next
evaluation rung.

See [`autoresearch-paper-rigor`](../autoresearch-paper-rigor/SKILL.md)
for the full statistical contract (Wilcoxon, bootstrap CI, Holm–
Bonferroni) required when n escalates beyond 3 seeds for a formal
external claim.

## Cost discipline

| step | runs | indicative wall-clock | cumulative |
|---|---|---|---|
| Screening (single config) | 1 | baseline | baseline |
| HC: primary lr axis | +2 | 2× baseline | 3× |
| HC: regularisation axis | +2 | 2× | 5× |
| HC: optimiser axis | +1 | 1× | 6× |
| HC: batch axis | +1 | 1× | 7× |
| HC: second-pass revisit (optional) | +4 | 4× | 11× |
| 3-seed confirmation at best | +2 | 2× | **13×** |

The default budget of **25 runs** absorbs the above plus ~12 cells
of slack. Hill-climbing multiple hypotheses should be scheduled in
waves to stay within daily compute budgets.

## Auto-checkpoint integration

The project's runner should call a checkpoint commit after every cell.
The commit scope is limited to:

```
experiments/<split>/<tag>__hc_*/
ideas/<NN>/hillclimb_results.json
ideas/<NN>/dashboard/
```

so concurrent edits in `src/` are not swept up. A pull-rebase fallback
(5 attempts) guards against concurrent agent commits. Running a
background auto-checkpoint loop alongside the per-cell commits is
recommended for crash safety — per-cell commits cover the happy path;
the loop covers crashes mid-cell.

## L3 — No selection decisions at n=1 (adaptive seed allocation)

**Never update the champion based on a single-seed result.** A single seed
is a noisy measurement: you condition on noise, and the apparent gain
regresses to the mean upon replication. This is the winner's curse, and it
is especially acute when the per-trial metric variance is high.

**Adaptive allocation protocol (successive halving):**

| Stage | Seeds per candidate | Rule |
|---|---|---|
| Exploration | n=1 | Compute a ranking; retain top 50% as nominees |
| Nominee confirmation | n=3 | Retain only those that still strictly beat the champion |
| Champion update | n=7 | Confirm at n=7 before writing to best_config; apply the statistical rigor floor (see [`autoresearch-paper-rigor`](../autoresearch-paper-rigor/SKILL.md)) |

- **n=1 → champion update is FORBIDDEN.** Any config that appears best at
  n=1 is a NOMINEE, not a champion.
- The 25-trial budget counts EXPLORATION trials. Promotion to n=3 and n=7
  is drawn from the rung's compute budget, not the exploration budget.
- Log every exploration AND confirmation trial in the per-hypothesis results
  file with the seed count used.

## L4 — Fit a joint surface when axes interact

Coordinate descent finds the optimum correctly only when the objective
surface is approximately axis-aligned convex. When two or more axes interact
(the optimum of one axis changes depending on the value of another), coordinate
descent can converge to a suboptimal local minimum and never discover the
global landscape.

**Detection:** if a second pass of coordinate descent reverses a decision from
the first pass (the same axis flips direction), the surface is non-convex and
coordinate descent is unreliable. Switch to joint modeling.

**Joint-surface protocol (domain-agnostic version):**

1. Identify the pair of axes most likely to interact (e.g., two axes where
   the optimum of one depends on the value of the other).
2. Run a structured grid over those two axes only (e.g., 3×4 = 12 trials).
3. Fit a quadratic or GP response surface to the n=1 results.
4. Use the project's cheapest validated surrogate (a leading indicator that
   predicts the expensive metric — like off-shell displacement for coherence)
   as an inner-loop constraint to exclude unstable regions.
5. Pick the predicted knee of the constrained surface analytically as the
   starting point for resuming coordinate descent on the remaining axes.

**Residual-budget rule:** the structured grid for joint modeling consumes up
to 15 of the 25-trial budget. After fitting the surface, resume coordinate
descent on the remaining axes with the remaining budget.

## L5 — Pareto-climb, not scalar-climb

The composite scalar is a convenience summary and tiebreak, not the
scientific champion criterion. A fixed-weight scalarization encodes arbitrary
trade-off preferences and can be Goodharted when one axis is under-priced.

**The Pareto frontier is the primary scientific object:**

- Accept a move iff it **Pareto-dominates** the current champion (at least as
  good on every axis, strictly better on at least one) OR extends the Pareto
  frontier at matched cost (same cost tier, benefit improves).
- Use the scalar composite as a tiebreak only when two configs are
  Pareto-incomparable.
- The per-hypothesis dashboard must display the **per-axis Pareto
  small-multiples** (benefit vs each cost axis). Ranking by composite alone
  is not sufficient for a champion decision.
- Log the full multi-axis vector for every trial so the Pareto frontier can
  be reconstructed at any time.

## Anti-patterns

1. **Full Cartesian grid without justification.** Wastes compute for an
   expected composite gain of < 0.5 pp over coordinate descent on smooth
   landscapes. Use only as a one-time validation on a single pilot
   hypothesis to confirm that coordinate descent did not miss a corner.
2. **Single-seed champion update (violation of L3).** Any config that looks
   best at n=1 is a NOMINEE. Champion update requires n ≥ 3 confirmation,
   and external claims require n ≥ 7 with the rigor floor.
3. **Cherry-picking the highest seed.** Reporting `max(composite)` over
   the confirmation seeds is a Goodhart violation. The headline is the
   `median`; the gate uses the `min`.
4. **Reporting the hill-climb best without the seed-noise gate.** A
   tuned config that beats the screening baseline by a small margin on
   seed 0 alone may be noise. Do not claim it.
5. **Ignoring axis interactions (violation of L4).** If coordinate descent
   oscillates (an axis reverses on a second pass), the surface is non-convex.
   Fit a joint surface; do not run another round of coordinate descent.
6. **Scalar-only ranking (violation of L5).** A champion decision based only
   on the composite value without inspecting all cost axes can crown a method
   that regresses on a hidden axis. Always check Pareto-dominance.
7. **Hill-climbing a broken hypothesis.** If the critic audit gave a
   BROKEN verdict for this hypothesis's implementation, fix the
   implementation first, then hill-climb. Tuning a buggy prior produces
   a spurious tuned ceiling.
8. **Tuning the screening config in place.** The screening baseline must
   stay frozen so downstream comparisons are valid. The hill-climb writes
   NEW directories; the original screening row remains untouched.
9. **Skipping the per-hypothesis dashboard.** The dashboard is the
   deliverable. A hill-climb without an `index.html` is incomplete.
10. **Hill-climbing on the most expensive evaluation rung directly.**
    Expensive rung runs (e.g., the FULL benchmark suite) cost orders of
    magnitude more than the development rung. Hill-climb on the cheapest
    rung that is informative about the composite, then graduate the tuned
    config to the expensive rung at 3-seed. See
    [`autoresearch-tiered-ladder`](../autoresearch-tiered-ladder/SKILL.md).

## Cross-references

- [`autoresearch-ablation-sweep`](../autoresearch-ablation-sweep/SKILL.md)
  — the screening counterpart. The ablation gives the candidate; this
  skill gives the verdict.
- [`autoresearch-fixer-campaign`](../autoresearch-fixer-campaign/SKILL.md)
  — when hill-climb reveals an implementation bug (tuned ceiling is
  suspiciously below the baseline at the same config), defer to the
  fixer; do NOT keep re-tuning a broken module.
- [`autoresearch-per-experiment-page`](../autoresearch-per-experiment-page/SKILL.md)
  — the per-cell renderer this skill consumes. Every cell gets its own
  page.
- [`autoresearch-auto-checkpoint-loop`](../autoresearch-auto-checkpoint-loop/SKILL.md)
  — companion crash-safety loop; pair it with the per-cell commits.
- [`autoresearch-paper-rigor`](../autoresearch-paper-rigor/SKILL.md)
  — a hill-climbed winner must satisfy the full statistical-rigor floor
  before any external claim.
- [`autoresearch-tiered-ladder`](../autoresearch-tiered-ladder/SKILL.md)
  — the rung promotion gate. Hill-climb confirms a hypothesis at the
  current rung before it is allowed to consume the next rung's compute.
- [`autoresearch-data-split-audit`](../autoresearch-data-split-audit/SKILL.md)
  — the split audit must be green before any hill-climb cell runs.
- [`autoresearch-shuffle-test`](../autoresearch-shuffle-test/SKILL.md)
  — semantic leakage must be cleared before a hill-climbed result is
  promoted to an external claim.
