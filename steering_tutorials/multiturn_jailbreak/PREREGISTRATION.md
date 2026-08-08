# PRE-REGISTRATION — multiturn_jailbreak

**Registered 2026-08-08, before the re-run.** The authoritative copy lives in
`config.PREREGISTRATION` (a dict, not prose) and `run_multiturn.print_preregistration()`
prints it **before any number exists**, so it cannot be dropped, softened, or
retrofitted to whatever came back. This file is the human-readable mirror; if the two
ever disagree, `config.py` wins.

## Why this file exists

The pre-2026-08 falsifier — *"AUC(sequence model) > AUC(per_turn_max) on HARD"* —
appeared **only in the README that also reported the result**. It was never in git
before the run, so in principle it could not have failed: `per_turn_max` is a max-pool
over turns, and any model that merely learned "the last window turn is the ask" would
clear it while reading no trajectory at all. CLAUDE.md §17 rule 8 requires a
pre-registered falsifier per claim; that is what this replaces.

## The claim under test

> A multi-turn jailbreak (Crescendo / ActorAttack) hides the attack in the escalation
> **trajectory** across turns, so a model that reads the ordered turn sequence detects
> it while a model that reads only a single turn does not.

Condition: **`hard`** (positives = an attack window containing the payload; negatives =
the benign lead-up prefix of a **different** attack from a disjoint group half, same
style, same turn count). The `easy` condition is a cautionary contrast and carries no
claim.

Headline embedder: **`embgemma`** (`google/embeddinggemma-300m`). `minilm` numbers are
a legacy reference arm and may not be headlined.

## The three falsifiers

Each is evaluated automatically in `run_multiturn._evaluate_falsifiers` and written to
`results.json` under `conditions.hard.falsifiers`. A missing input yields
`not_evaluable` — never a silent pass.

### F1 — against the BINDING CONFOUND BAR, not against `per_turn_max`

> **FALSIFIED if** `AUC(best sequence model) <= worst_auc` from
> `common.confound.confound_report` — the largest **directionless** bar among
> {length, unit count, TF-IDF content}.

Beating `per_turn_max` is **not sufficient**. The negatives here are keyword-topic
matched, so the content/TF-IDF bar is the honest test of whether a method reads
*escalation* or merely *vocabulary*, and CLAUDE.md §17 rule 7 permits claiming only the
margin above the larger of {baseline, confound}.

### F2 — against `last_turn_only`

> **FALSIFIED if** `AUC(best sequence model) <= AUC(last_turn_only) + 0.02`.

`last_turn_only` is a logistic regression on the **final turn embedding alone**. In the
HARD condition the positive window ends on the payload turn and the negative window
does not, so "the last turn is the ask" is a complete alternative explanation for every
sequence model's score. `per_turn_max` is a *max over turns* and is **not** this
control. Without F2, "trajectory detection" is indistinguishable from "the final turn
is just harmful".

### F3 — the shuffled-turn control (scopes what "trajectory" may mean)

> The **trajectory** reading specifically is **FALSIFIED if**
> `AUC(seq_gru, turn order permuted) >= AUC(seq_gru, true order) - 0.02`.

The permutation destroys order while leaving the turn vectors untouched. Note that
`trajectory_mlp` and `hier_attn` are largely permutation-**invariant** by construction
(mean/last/max/std features; softmax pooling), so F3 is informative for `seq_gru` and
near-vacuous for them. If F3 fires, the licensed claim shrinks to *"aggregating across
turns beats reading a single turn"* — never *"only a stateful model can see it"*.

## Screening vs evaluation, declared in advance

This run is **SCREENING** unless it reaches ≥500 per class **and** ≥7 seeds
(CLAUDE.md §7). The lesson is currently single-seed (`MJ_SEED=0`), so no result from it
may use the words *winner*, *beats baseline*, or *significant*. Reclassifying after the
fact is HARKing.

`n` and `n_distinct_groups` are reported **separately** in `results.json`. `SafeMTData_1K`
has several attack paths per harmful goal, so raising `n` without raising the group
count inflates rows, not information.

## What would make me abandon the claim rather than weaken it

If F1 and F2 both fire on the headline embedder, the lesson's thesis is not merely
unproven — the correct action is to rewrite §8 of the README around "the payload turn is
recognisable and the confound bar is high", and to say so in the results table rather
than reporting the AUCs and appending a caveat.
