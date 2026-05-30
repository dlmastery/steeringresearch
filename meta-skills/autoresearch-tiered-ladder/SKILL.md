---
name: autoresearch-tiered-ladder
description: Use when deciding how much compute to spend on a candidate method, when deciding whether a result is ready for an external claim, or when building the project's evaluation harness. Defines five evaluation rungs (UNIT, SMOKE, DEV, STANDARD, FULL) with explicit cost brackets, promotion gates, and a structured promotion/demotion log. The same axes are measured at every rung; only dataset size and realism grow. No method skips a rung.
---

# Skill — Tiered benchmark ladder

## When to use

- When deciding which evaluation rung to run a new candidate method on.
- When deciding whether a result justifies the compute cost of the
  next rung.
- When building or auditing the project's evaluation harness to
  confirm that the rung structure is implemented correctly.
- When a method that passed a lower rung now fails a higher rung —
  to determine the correct demotion action and log entry.
- When writing the methods or experimental-setup section of a paper —
  to report which rungs were completed and which were the basis of
  the headline claim.

## Philosophy

Never run an expensive benchmark to discover a bug a cheap one would
have caught. The ladder exists because:

1. Most candidate methods have a fatal flaw (wrong sign, off-target
   effect, catastrophic side-effect) that can be detected in seconds
   or minutes at rung 0 or rung 1.
2. Only directionally correct methods should consume the compute
   budget of higher rungs.
3. The same measurement axes at every rung mean that a result at a
   lower rung is a true (imperfect but useful) predictor of the
   result at the next rung — not a different measurement.

This mirrors the established computer-vision practice of using a small
dataset (e.g., CIFAR-10) as a fast directional test before a medium
dataset (e.g., CIFAR-100) before a large dataset (e.g., ImageNet),
where the same accuracy metric is used at every scale.

## The 5 rungs

```
RUNG  NICKNAME   ANALOGY        COST/RUN        PURPOSE                          GATE TO PASS
0     UNIT       sanity check   seconds          mechanics correct?               all invariants hold; no state leak
1     SMOKE      CIFAR-10       1–3 min          right direction?                 monotonic effect + no collapse
2     DEV        CIFAR-100      10–30 min        generalises to harder split?     beats baseline on held-out split
3     STANDARD   ImageNet       1–4 hours        real comparative result?         Pareto-dominates prior method
4     FULL       full suite     half-day+        defendable publication result?   multi-model, ablated, red-teamed
```

### Rung 0 — UNIT (seconds) — "does the plumbing work?"

This is not a benchmark; it is a mechanics check. Run it before any
other experiment.

What to check (adapt to the project's domain):

- The method's intervention actually changes the intended quantity
  (e.g., an activation vector changes a residual stream, a mask
  changes the loss, a weight delta changes the output logits). Assert
  that `||delta_quantity|| > 0` at the target location.
- Removing or zeroing the intervention restores the baseline exactly
  (no state leak, no accumulated side-effect).
- Any invariants the method claims to preserve are verifiable on a
  handful of hand-crafted inputs (e.g., a method claimed to be
  input-agnostic at a certain layer should produce the same effect on
  five diverse inputs).
- The method's data contract is satisfied: expected input shape,
  dtype, and range.

**Data:** 5–20 hand-crafted inputs sufficient to exercise the
invariants. Results are deterministic.

**Gate:** all assertions pass; run time under 60 seconds.

### Rung 1 — SMOKE (1–3 min) — "are we going in the right direction?"

The fast directional sanity test. SMALL, FIXED, CHEAP. Run on every
iteration of the autoresearch loop, including after every config change.

What to measure (adapt axis labels to the project's domain):

- **Primary axis (the method's intended effect):** does the effect
  increase monotonically over a small sweep of the method's strength
  parameter {0, 0.5×, 1.0×}? The effect must exist and be ordered.
- **Coherence tripwire:** does the model or pipeline's general output
  quality stay bounded? A large side-effect on coherence or task
  performance at a mild intervention strength is a red flag.
- **Safety tripwire (if applicable):** does the method accidentally
  enable unsafe or out-of-spec behaviour at the working strength?
- **No collapse:** the output distribution must not degenerate (e.g.,
  constant output, repetition loop, NaN loss).

**Data:** a small, fixed, pinned subset of the project's development
data — small enough to run in a few minutes, but large and diverse
enough to be informative. Pin the subset so SMOKE results are
comparable across iterations.

**Gate to Rung 2:** monotonic primary effect + bounded coherence/quality
tripwire + no safety leak (if applicable). If monotonicity fails, the
method has the wrong target, wrong sign, or wrong layer/location — fix
before spending more compute.

**Cost target:** keep Rung 1 under ~3 minutes so it can be run dozens
of times per day inside the autoresearch loop.

### Rung 2 — DEV (10–30 min) — "does it generalise a little?"

Harder evaluation, larger dataset, held-out split. Run when a method
passes SMOKE and you want to believe it before committing to the
standard evaluation budget.

What to measure:

- **Primary axis on the held-out split:** does the method still show
  a positive effect on data that was not seen during development?
  The held-out split should include concept categories, input types,
  or distribution conditions that were not present in the Rung 1
  pinned subset.
- **Full coherence panel** (not just the tripwire): standard metric on
  the project's coherence benchmark at the matched intervention
  strength.
- **Safety panel** (if applicable): full safety evaluation suite, not
  just the tripwire subset.
- **Secondary axis (a second behaviour family):** verify the effect is
  not specific to the primary family. A method that only works on
  one narrow class of inputs is a fragile result.

**Gate to Rung 3:** beats the chosen baseline on the held-out split at
matched coherence, with any safety metric no worse than baseline.

### Rung 3 — STANDARD (1–4 hours) — "is this a real comparative result?"

The comparative benchmark you would put in a results table in a paper.

What to measure:

- **Primary axis on the full benchmark** (the project's standard
  evaluation protocol, including the held-out test set).
- **Full capability / task-performance panel:** the standard set of
  benchmarks the project uses to characterise side-effects.
- **Full coherence panel:** standard metrics at matched intervention
  strength.
- **Full safety panel** (if applicable): all safety and refusal
  benchmarks at the same intervention strength.
- **Selectivity / precision panel** (if applicable): does the method
  affect only the intended target, or does it have broad, unintended
  effects?

**Protocol:** use a calibrated evaluation procedure (e.g., a
judge that has been validated against a human-labelled slice);
report mean ± CI; compare at MATCHED intervention strength, not
matched primary-axis score.

**Gate to Rung 4:** Pareto-dominates the prior state-of-the-art method
on the composite of (primary axis × coherence × capability × safety) —
no axis regresses. A method that improves the primary axis while
degrading capability is NOT a Pareto improvement.

### Rung 4 — FULL (half-day+) — "can you defend it?"

Everything in Rung 3 plus:

- **Cross-variant generalisation:** run on at least two variants of
  the base system (e.g., two model sizes, two architectural families)
  to show scale-robustness.
- **Red-team stress test:** attempt to break or circumvent the
  method's guarantees using adversarial inputs, jailbreak suffixes,
  or out-of-distribution prompts. Report success/failure rate.
- **Ablations:** each component of the method enabled/disabled
  independently, so the contribution of each component is
  attributable.
- **Long-run stability:** does the method's effect persist over
  extended generation or inference? Does it drift or accumulate?

**Gate:** full multi-axis win with ablation-justified attribution of
every gain. All limitations documented. All red-team findings reported
in the paper.

## Same axes at every rung — the invariant

The most important structural property of the ladder:

> **The axes measured at Rung 1 are the same axes measured at Rung 4.
> Only the dataset size and the realism of the evaluation grow.**

This invariant ensures that a Rung 1 result is a genuine predictor of
the Rung 4 result. If a method passes Rung 1 on the primary axis
but fails Rung 3 on the same axis, the failure is informative — the
effect does not scale. If the axes changed between rungs, the
promotion gate would carry no information.

Operationally: define the composite metric formula once, fingerprint
it (SHA-256 of the formula string), and use the same formula at every
rung. See [`autoresearch-experiment`](../autoresearch-experiment/SKILL.md)
for the composite-fingerprint discipline.

## Promotion-gate rule

A method MUST clear the gate at rung K before it is allowed to consume
rung K+1 compute. A regression at any rung sends the method back with
a logged failure reason.

The gate is:

```
gate_passed(method, rung) = True
  if ALL of the following hold at rung:
    1. primary_axis_score > baseline_primary_axis_score (strict)
    2. all side-effect axes within pre-registered tolerance
    3. no safety violation (if applicable)
    4. split audit green (autoresearch-data-split-audit)
    5. shuffle test green at STANDARD and above (autoresearch-shuffle-test)
```

"Strict greater" means a tie does NOT count as a promotion. This
mirrors the champion rule in the hill-climb (see
[`autoresearch-per-hypothesis-hillclimb`](../autoresearch-per-hypothesis-hillclimb/SKILL.md)).

## Promotion/demotion log schema

Every gate evaluation — pass or fail — is logged. The log is
append-only. The schema:

```
method_id | rung | <axis_1> | <axis_2> | ... | <axis_N> | verdict | failure_reason
```

Where:
- `method_id` — the canonical short name or tag of the method
  (matches the tag in `experiment_log.jsonl`).
- `rung` — integer 0–4.
- `<axis_1>` through `<axis_N>` — the numeric value of each measured
  axis at this rung. The axis names are fixed for the project and
  match the composite metric formula. All axis columns are present
  at every rung; columns that are not evaluated at the current rung
  (e.g., ablation axes at Rung 1) are recorded as `null`.
- `verdict` — one of: `PROMOTE` | `DEMOTE` | `HOLD`.
  - `PROMOTE`: method cleared the gate; approved for rung K+1.
  - `DEMOTE`: method failed the gate; sent back to rung K-1 or
    to a fix campaign.
  - `HOLD`: method is at the maximum rung and confirmed as a winner,
    OR is awaiting a re-run before verdict.
- `failure_reason` — free text, required when `verdict = DEMOTE`.
  Concise, actionable (e.g., "primary axis collapsed at held-out
  split", "coherence tripwire violated at 1.0× strength",
  "shuffle test FAIL — off-by-one alignment suspected").

Example log rows (JSONL format):

```jsonl
{"method_id": "method_A", "rung": 0, "primary": null, "coherence": null, "safety": null, "verdict": "PROMOTE", "failure_reason": null}
{"method_id": "method_A", "rung": 1, "primary": 0.72, "coherence": 0.91, "safety": 0.99, "verdict": "PROMOTE", "failure_reason": null}
{"method_id": "method_A", "rung": 2, "primary": 0.68, "coherence": 0.89, "safety": 0.99, "verdict": "PROMOTE", "failure_reason": null}
{"method_id": "method_B", "rung": 1, "primary": 0.45, "coherence": 0.82, "safety": 0.99, "verdict": "DEMOTE", "failure_reason": "primary axis not monotone over strength sweep; wrong target layer suspected"}
{"method_id": "method_C", "rung": 2, "primary": 0.71, "coherence": 0.61, "safety": 0.99, "verdict": "DEMOTE", "failure_reason": "coherence tripwire violated at 1.0x strength; PPL doubled"}
```

The log file lives at `autoresearch_results/ladder_log.jsonl`. It is
append-only; no row is ever edited. Amend a verdict by appending a
new row with the same `method_id` and rung, marked with the new
verdict and a `replaces_row` field citing the prior log timestamp.

## Cost discipline and scheduling

The ladder's value is that most compute is spent only on methods that
have cleared the cheap gates.

| rung | typical run count | indicative cumulative cost |
|---|---|---|
| 0 UNIT | 1 per method | seconds |
| 1 SMOKE | many (every iteration) | minutes per method |
| 2 DEV | 1–3 | 30–90 min per method |
| 3 STANDARD | 1 (+ 3-seed confirm) | 4–12 h per method |
| 4 FULL | 1 per winner | half-day to days |

Heuristic: 90% of candidate methods should fail at Rung 0 or Rung 1.
If the failure rate at Rung 1 is below 50%, the screening hypotheses
are not diverse enough, or the Rung 1 evaluation is not discriminating.

For scheduling on constrained hardware:
- Keep Rung 1 cheap enough to run inside the autoresearch inner loop.
- Run Rung 2 in the background while developing the next batch of
  candidates.
- Schedule Rung 3 as an overnight or batch job.
- Reserve Rung 4 for methods with at least one confirmed Rung 3 win.

## Integration with other skills

- **Before any rung:** run the split audit
  ([`autoresearch-data-split-audit`](../autoresearch-data-split-audit/SKILL.md)).
  A stale or failed audit blocks all rungs.
- **Before STANDARD or FULL:** run the shuffle test
  ([`autoresearch-shuffle-test`](../autoresearch-shuffle-test/SKILL.md)).
  A semantic leakage failure blocks promotion to STANDARD.
- **Before any external claim:** the result must be at STANDARD or
  FULL rung, and must satisfy the full statistical-rigor floor
  ([`autoresearch-paper-rigor`](../autoresearch-paper-rigor/SKILL.md)).
- **Before FULL:** the method's tuned config must have been confirmed
  by the hill-climb at the DEV rung
  ([`autoresearch-per-hypothesis-hillclimb`](../autoresearch-per-hypothesis-hillclimb/SKILL.md)).
- **Every rung:** the 7-step experiment ritual
  ([`autoresearch-experiment`](../autoresearch-experiment/SKILL.md))
  governs each individual run.

## Anti-patterns

- **Skipping Rung 1 because "the method is obviously correct".**
  Rung 1 catches mechanical bugs (wrong sign, wrong target, wrong
  layer) in minutes. Skipping it is a compute gamble that frequently
  fails expensively.
- **Running Rung 3 on a method that has not passed Rung 2.** The
  standard evaluation budget is wasted if the method does not
  generalise to the held-out split.
- **Reporting a Rung 1 or Rung 2 result as a headline claim.** A
  SMOKE or DEV pass is a directional signal, not an external claim.
  External claims require STANDARD rung at minimum, plus the full
  statistical-rigor floor.
- **Changing the composite metric formula between rungs.** That
  breaks the invariant that lower rungs predict higher rungs. The
  formula is fingerprinted; changing it is a project integrity
  violation.
- **Not logging failed gates.** The demotion log is as important as
  the promotion log. Knowing which methods failed and why is the
  project's institutional memory.
- **Promoting a method with a degraded side-effect axis.** A method
  that improves the primary axis while worsening capability or safety
  is NOT a Pareto improvement and does NOT clear the gate.
- **Using different data subsets for the same rung across iterations.**
  The SMOKE data must be pinned. If the SMOKE subset changes, the
  results are not comparable across iterations.
- **Treating the Full rung as optional for a paper claim.** The
  STANDARD rung is the minimum for a paper result table. The FULL
  rung is required when the paper claims cross-model generality,
  adversarial robustness, or component attribution.

## Cross-references

- [`autoresearch-experiment`](../autoresearch-experiment/SKILL.md)
  — the 7-step ritual for each individual run at any rung.
- [`autoresearch-per-hypothesis-hillclimb`](../autoresearch-per-hypothesis-hillclimb/SKILL.md)
  — the hill-climb closes the gap between DEV screening and DEV
  evaluation before promotion to STANDARD.
- [`autoresearch-paper-rigor`](../autoresearch-paper-rigor/SKILL.md)
  — the statistical-rigor floor that governs which rung justifies
  an external claim, and how many seeds are required.
- [`autoresearch-data-split-audit`](../autoresearch-data-split-audit/SKILL.md)
  — the structural split gate; required at every rung.
- [`autoresearch-shuffle-test`](../autoresearch-shuffle-test/SKILL.md)
  — the semantic leakage gate; required at STANDARD and FULL rungs.
- [`autoresearch-meta`](../autoresearch-meta/SKILL.md)
  — the top-level orchestration skill that places the ladder in the
  broader campaign workflow.
- [`autoresearch-dashboard`](../autoresearch-dashboard/SKILL.md)
  — the dashboard should visualise the ladder log, showing each
  method's current rung, gate status, and axis values.
