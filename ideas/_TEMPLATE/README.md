# H<NN> — <one-line idea title>

> Back to master registry: [IDEA_TABLE.md](../../IDEA_TABLE.md)

## TL;DR

<2-3 sentences: what the idea does, why it matters, what is the expected gain.>

## Citation

<Full citation: Author1, Author2, ... YEAR VENUE 'Title' (arXiv:XXXX.XXXXX)>

If no peer-reviewed citation exists, this idea must stay in BACKLOG.md until
one is found. No citation = no idea (per meta-skills/autoresearch-idea-scaffold/SKILL.md).

## Hypothesis

<!-- Word count >= 50. Must include mechanism (which of the 12 axes moves, what
     it does in the residual stream) and expected delta. -->

<Mechanism + expected delta. State which axis from the 12-axis framework is
being perturbed. Explain WHY the intervention should work (not just what it does)
and what the cited paper predicts about the outcome.>

## Falsifier

<What single observation would DISCARD this idea? Be specific:
"If <primary metric> delta <= <threshold> on <evaluation setup (model, rung, dataset)>,
this idea is DISCARDED."
The falsifier threshold must be a number, not a vague phrase.>

## Predicted delta range

| metric | delta vs baseline | rationale |
|--------|------------------|-----------|
| composite | [+X.XX, +Y.YY] | <one sentence from the cited paper> |
| behavior efficacy | [+X.X pp, +Y.Y pp] | <one sentence> |
| MMLU drop | [0, -Z.Z pp] | <expected capability tax> |
| PPL delta | [+A.A, +B.B] | <expected coherence cost> |

All thresholds are [NEEDS VERIFICATION] until reproduced on the 4090 ladder.

## Status

| stage | done? |
|-------|-------|
| implementation written | [ ] |
| tests green | [ ] |
| AUDIT.md filed | [ ] |
| IMPROVEMENTS.md addressed | [ ] |
| VERIFY.md sealed | [ ] |
| First experiment archived | [ ] |
| Verdict authored | [ ] |

## How to test this idea (idea-specific experiment strategy)

<What is the SINGLE BEST evaluation setup that would prove or falsify this idea?
Be concrete: model, dataset split, rung, n_seeds, primary metric.>

## Cross-idea interactions

<Which other ideas from IDEA_TABLE.md does this compose well with (orthogonal axes)?
Which conflict (same site / same direction / same operation)? Where in the
combination suite does it slot?>

Composable with: (list hypothesis IDs)
Competes with: (list hypothesis IDs)
Combination suite entry: (if applicable)
