---
name: autoresearch-ablation-sweep
description: Use when running a structured screening sweep across many configurations (baselines + single-prior-on + leave-one-out + curated subset). Produces the ranked candidate list that feeds the hill-climb. This is the screening counterpart to the hill-climb evaluation tier; confusing the two tiers is a common source of over-claiming.
---

# Skill — Structured ablation / screening sweep

## When to use

- Before a hill-climb: to identify which design choices are worth
  hill-climbing at all (positive directional signal at the default
  config, n ≤ 3 seeds).
- When attributing a headline effect to specific design choices in a
  multi-prior block — you need at least single-prior-on and
  leave-one-out rows to attribute effects cleanly.
- Whenever the user asks to "screen candidates", "run a sweep matrix",
  "identify which priors are worth tuning", or "attribute effects".
- NOT as a substitute for the hill-climb evaluation tier. A positive
  screening result is directional evidence about the default config,
  not a confirmed result about the prior. See
  [`../autoresearch-per-hypothesis-hillclimb/SKILL.md`](../autoresearch-per-hypothesis-hillclimb/SKILL.md).

## Why this is the screening counterpart to hill-climb

The funnel has two tiers:

```
SCREENING (this skill)           EVALUATION (hill-climb skill)
─────────────────────────────    ────────────────────────────────────────
n ≤ 3 seeds at default config    20–25-trial coordinate descent over
                                 the tuning cube + n ≥ 3-seed confirm

Many configs, cheap              One config, expensive + statistically
                                 confirmed

Produces: ranked candidate list  Produces: tuned ceiling + external-
                                 ready verdict (with paper-rigor floor)
```

An n ≤ 3 screening result is NOT a "winner". It is a statement about
the default recipe at that config. A negative screening result may
simply mean the prior is mis-tuned at the default config, not that
the hypothesis is wrong.

Pre-register the screening-vs-evaluation classification BEFORE the
sweep. Reclassifying a loser as "screening" after the fact is HARKing
— a BLOCKER per the rigor floor (see
[`../autoresearch-paper-rigor/SKILL.md`](../autoresearch-paper-rigor/SKILL.md)).

## The minimum useful ablation matrix

A sweep that can attribute effects must include at least:

| row type | count | purpose |
|---|---|---|
| Literature / reference baseline | 1 | the anchored comparison point |
| Vanilla (all design choices OFF) | 1 | your pipeline at its zeroed state |
| Single-prior-on rows | N (one per design choice) | attribute each choice in isolation |
| Full hybrid (all choices ON) | 1 | total effect; compare with single-prior sum |
| Leave-one-out rows (optional) | N | attribute from the all-on end; useful when full > sum |

A curated subset (10–15 rows × 1 seed × short training) fits in
approximately 60–90 minutes on a single compute unit. A full sweep
adds leave-one-out rows and multiple seeds at 3–5× that cost.

## Sweep driver pattern

```python
def build_matrix(design_choices: list[str],
                 curated: bool = True) -> list[dict]:
    """
    design_choices: the orthogonal design choices being screened.
    Each element maps to a Boolean flag in the runner config.
    curated: if False, also adds leave-one-out rows.
    """
    base = {flag: False for flag in design_choices}
    full = {flag: True  for flag in design_choices}
    rows = []

    # Baselines (always included)
    rows.append(dict(tag="baseline_reference",
                     overrides=dict(use_reference_architecture=True)))
    rows.append(dict(tag="baseline_vanilla",
                     overrides=dict(**base)))

    # Single-prior-on rows (one per design choice)
    for flag in design_choices:
        cfg = base.copy()
        cfg[flag] = True
        rows.append(dict(tag=f"only_{flag}", overrides=cfg))

    # Full hybrid
    rows.append(dict(tag="full_hybrid",
                     overrides=dict(**full)))

    if not curated:
        # Leave-one-out from full
        for flag in design_choices:
            cfg = full.copy()
            cfg[flag] = False
            rows.append(dict(tag=f"loo_no_{flag}", overrides=cfg))

    return rows
```

The flag names must match the project's runner config schema exactly.
The `baseline_reference` row uses the project's literature-anchored
architecture; the `baseline_vanilla` row uses the project's own
pipeline with every design choice disabled.

## Hard rules

1. **Skip-existing is the default.** Each row writes to a deterministic
   directory. Restarting a sweep skips rows whose directory already
   exists; to re-run a row, remove its directory first. This makes
   sweeps safe to restart after a crash.
2. **Per-row wall-clock is logged.** A row that takes more than 1.5×
   the median wall-clock for its rung triggers a warning — likely a
   misconfiguration.
3. **A failed row does not stop the sweep.** A crashed row writes an
   `error.txt` to its directory and the sweep continues. Investigate
   failures after the sweep; do not block the other rows.
4. **Each row is one experiment per the 7-step ritual.** Reasoning
   entries can be batch-authored for a screening sweep, but each row
   still requires its own gated entry (see
   [`../autoresearch-experiment/SKILL.md`](../autoresearch-experiment/SKILL.md)).
   Batch-authoring is acceptable; skipping the entry is not.
5. **No new design choices added mid-sweep.** Adding a flag after rows
   have run breaks the comparability of the single-prior-on rows.
   Plan the matrix completely before launching.

## Output contract

Per row: a deterministically named directory containing at minimum
`metrics.json` (run results dataclass), `history.json` (per-step
training log), and a model checkpoint if applicable.

Global: an `experiment_log.jsonl` in the sweep root, one row per run,
append-only.

Per row verdict: `KEEP` / `NEAR-MISS` / `DISCARD` is filled into
`reasoning.json` (or the project's equivalent reasoning file) within
one working day of run completion. This is the ranked candidate list
consumed by the hill-climb.

## Recommended cadence

| stage | rows | seeds | compute budget |
|---|---|---|---|
| Smoke validation | 1–2 (reference + vanilla) | 1 | < 5 min |
| Curated ablation | 10–15 | 1 | 60–90 min |
| 3-seed re-sweep of top candidates | 3–5 best rows | 3 | 3–4× curated |
| Leave-one-out (if warranted) | N additional | 1–3 | 2× extra |
| Scale-up (best row only) | 1 | 3+ | hours–days |

Do not run the 3-seed re-sweep before validating the curated matrix.
Do not run leave-one-out before the curated sweep shows a meaningful
positive result in the full-hybrid row.

## What a good sweep delivers

- A ranked list of design choices by single-prior-on lift, sortable
  by the project's composite metric.
- A Pareto plot and ablation bar chart as standard dashboard outputs
  (see [`../autoresearch-dashboard/SKILL.md`](../autoresearch-dashboard/SKILL.md)).
- A `FINDINGS.md` headline (or equivalent ledger row) rephrased with
  concrete Δ-vs-reference numbers — not "improves significantly."
- At least one clearly positive single-prior-on row to hand to the
  hill-climb as the first candidate.

## Anti-patterns

| Anti-pattern | Why it fails |
|---|---|
| Reporting the curated sweep as a "winner" claim | n ≤ 3 is SCREENING; claims require the rigor floor |
| Running leave-one-out before the curated matrix | Expensive without knowing whether the full hybrid is positive |
| Adding new flags mid-sweep | Breaks comparability of earlier single-prior rows |
| Sorting rows by any single axis (not composite) | Honours Goodhart; sort by the composite fingerprint |
| Hill-climbing before screening identifies candidates | Wastes compute on priors that fail even at default config |

## Cross-references

- [`../autoresearch-per-hypothesis-hillclimb/SKILL.md`](../autoresearch-per-hypothesis-hillclimb/SKILL.md)
  — the evaluation counterpart; consumes this sweep's candidate list.
- [`../autoresearch-experiment/SKILL.md`](../autoresearch-experiment/SKILL.md)
  — the 7-step ritual that governs each individual row.
- [`../autoresearch-combo-ladder/SKILL.md`](../autoresearch-combo-ladder/SKILL.md)
  — the next step after hill-climbing: identifies which confirmed
  priors are safe to stack.
- [`../autoresearch-paper-rigor/SKILL.md`](../autoresearch-paper-rigor/SKILL.md)
  — the rigor floor that governs what the sweep's results may claim.
- [`../autoresearch-dashboard/SKILL.md`](../autoresearch-dashboard/SKILL.md)
  — the dashboard that renders the sweep's ranked table, Pareto plot,
  and ablation bar chart.
- [`../autoresearch-auto-checkpoint-loop/SKILL.md`](../autoresearch-auto-checkpoint-loop/SKILL.md)
  — pair with a background auto-checkpoint loop for sweeps expected
  to run longer than 15 minutes.
- [`../autoresearch-modular-block/SKILL.md`](../autoresearch-modular-block/SKILL.md)
  — if the design choices are Boolean flags on a composable block, the
  modular-block skill defines the flag discipline that makes this sweep
  matrix well-formed.
