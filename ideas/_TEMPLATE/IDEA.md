# Formal claim for H<NN>

> This document is the binding pre-registration contract for idea H<NN>.
> Once the first experiment row referencing this idea is committed to
> EXPERIMENT_LEDGER.md, this file is FROZEN. Any change requires a new
> idea number and a fresh pre-registration.

## Claim

<One sentence, present tense, falsifiable. Must be specific enough that
a third party could check whether it fired.>

Example form: "On Gemma-2-2B-it (4-bit), <intervention> achieves <metric>
>= <threshold> on <evaluation set> compared to <baseline>."

## Falsifier

<One observation that DISCARDS this claim. Quote the metric, threshold,
evaluation setup. This is binding — no goalpost-shifting after first run.>

"If <primary metric> does not reach <threshold> on <held-out evaluation set>
at rung <N> with n >= <seeds>, this claim is DISCARDED."

## Pre-registered prediction

Primary metric: <metric name>
Expected value: [low, high] (95% plausible range from the cited paper)
Baseline value (pre-registered): <value from documented baseline>

Sub-metrics pre-registered:
- MMLU delta: [low, high] pp
- PPL delta: [low, high]
- JailbreakBench CR: <= X%
- Over-refusal: <= X%

Geometry leading indicators:
- Off-shell displacement delta_norm: [low, high]
- Norm budget fraction (N5): <= X

All values above are [NEEDS VERIFICATION] pre-registered predictions,
not established facts.

## Composite fingerprint at time of registration

SHA-256 of `src/steering/eval.py:COMPOSITE_FORMULA` at registration time:
`a9001e87087e` (placeholder — fill with actual hash before first experiment)

## Signed off by

Author: <identifier>
Date: <YYYY-MM-DD>
Git commit at registration: <sha>
