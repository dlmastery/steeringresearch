# Stacking hill-climb — the ladder measures coherence destruction, not steering

**Tier: SCREENING** (n=40 harmful / 20 benign, off-family Qwen-3B judge). Not a win, and
not offered as one. Pre-registration written before running and left unrevised:
`PREREGISTRATION_hillclimb.md`. Artifact: `artifacts/hillclimb_results.json`.

## The ladder

| rung | + prior | refusal | gibberish | **marginal** | standalone |
|---|---|---|---|---|---|
| R0 | — (unsteered) | **0.275** | 0.500 | — | — |
| R1 | A — refusal @ block 12 | 0.100 | 0.800 | −0.175 | −0.175 |
| R2 | + B — same vector @ block 8 | 0.025 | 0.950 | −0.075 | −0.100 |
| R3 | + C — orthogonal PC @ block 12 | 0.000 | 0.975 | −0.025 | −0.175 |
| R4 | + CLAMP (meta) | 0.075 | 0.925 | +0.075 | n/a |
| R5 | + GATE (meta) | 0.125 | 0.775 | +0.050 | benign gibb **0.900 → 0.000** |

## The finding that voids the rest

**There is no direction specificity.** Prior **A** is the refusal diff-of-means. Prior **C**
is a direction constructed to be *exactly orthogonal to it* — measured
`cos(C, refusal) = 3.0e-08` — applied at the **same site** and the **same α**.

Both score **0.100**.

A meaningless direction does exactly what the refusal direction does. So **no rung in this
ladder is attributable to refusal**, and the stack/compete question cannot be asked of
these priors at this budget: there is no signal to compose. This is M-a's random-control
logic arriving at the opposite verdict — there, a norm-matched random direction cost far
less than the real one, which is what made the effect direction-specific. Here it costs the
same.

**No prior helps at all.** Every prior scores *below* the unsteered baseline (0.275):
A 0.100, B 0.175, C 0.100, B′ 0.175. Steering **lowers** refusal because **gibberish takes
the mass** — 0.500 → 0.975 across the ladder. The ladder is measuring coherence
destruction, and refusal is falling out of the denominator.

## Eight recorded contradictions, including against my own design

- **P4 failed.** The norm clamp moved gibberish −0.050 against a required −0.10. Cause:
  it captures `h_base` *in-pass*, so it is per-site local, not a global budget.
- **P5 failed.** The CAST gate is not inert — the probe fires on **53%** of harmful and 0%
  of benign, so R5's change mixes gating with reduced coverage.
- **The pre-registered STACK/COMPETE test is degenerate** with negative standalones. It
  read "not competing" at R2/R3 purely because the arithmetic inverts on negatives. By the
  README's own *best-single* criterion, **every rung competes** (R2 −0.150, R3 −0.175,
  R4 −0.100, R5 −0.050).
- **The ladder anchor is wrong.** [A] (0.100) is not the best single prior — B alone is
  0.175 — so every marginal was measured from the wrong baseline.
- **A × B′ COMPETE: confirmed.** 0.000, below both constituents. The one unconditional
  COMPETE prediction held.

## `B_refusal_rate` = 0.2533 — born orphaned

Present in commit `1bc1192` in identical shape and **never read** by `classify_ladder`, the
plot, or the README — 150 generations spent per run to compute a number nothing consumed.
It is **above every ladder rung** and above base A (0.200) on both axes, and it understates
`stack_marginal` by **38%** (−0.14 vs −0.193, z = −4.78). Now surfaced with its z = 1.11
noise caveat, and replicated at n=40 (B 0.175 > A 0.100).

`AUDIT.md` is also **stale** — it validates numbers absent from the current `results.json`.

## What this does and does not say

It does **not** show that stacking fails in general. It shows that **this lesson's priors,
at this budget, on this abliterated model, produce no direction-specific effect**, so the
composition question is unanswerable here until a prior is found that beats its own
orthogonal control. That is the prerequisite, and it is now the pre-registered next step.

> Internal QA pass — independent external review pending.
