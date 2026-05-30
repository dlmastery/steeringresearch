---
name: autoresearch-idea-scaffold
description: Use when scaffolding a new idea sub-project from a template. Each idea equals one hypothesis with its own implementation, tests, audit, and experiments. Modular, independently ablatable, domain-agnostic.
---

# Skill — Scaffold a new idea sub-project

## When to use

When introducing a new hypothesis into the project — a new inductive bias,
optimisation trick, regulariser, evaluation strategy, or structural primitive
that deserves its own ablation.

Do NOT use for trivial cleanups or refactors — those go in existing modules.

## What is an "idea" exactly?

An idea is a falsifiable claim that maps to:

- a peer-reviewed citation (Author YEAR VENUE 'Title' full-ID)
- a single mechanism (one parameter or structural choice that changes between
  the baseline and the variant)
- a measurable outcome (primary metric delta, sub-metric delta)

If you cannot write the citation, you do not have an idea — you have a hunch.
Write it up as a `BACKLOG.md` line instead, and return when you can cite it.

## Directory layout

```
ideas/NN_<short_name>/
├── README.md        — idea statement + lit review + hypothesis
├── IDEA.md          — formal claim + falsifier + predicted delta range
├── implementation.py — standalone module
├── tests.py         — correctness tests
├── AUDIT.md         — self-audit (weaknesses found before first run)
├── IMPROVEMENTS.md  — fix log (addressed weaknesses from AUDIT)
├── VERIFY.md        — verification log: tests green, sanity checks, sign-off
├── experiment.py    — idea-specific experiment driver
├── configs/         — configuration files (YAML or equivalent)
├── experiments/     — per-experiment archives
├── results.md       — auto-generated or manually-maintained summary
└── dashboard/       — idea-level dashboard
```

**Numbering:** use 10-spacing (`10`, `20`, `30`, …) to leave room for
closely-related ideas to be inserted between existing ones without renaming.

## Scaffold recipe

```bash
NN=10; SHORT="my_idea"
cp -r ideas/_TEMPLATE ideas/${NN}_${SHORT}
# Edit ideas/${NN}_${SHORT}/README.md and IDEA.md:
#   - citation (full ID, not just author name)
#   - hypothesis (≥ 50 words, mechanism + expected delta)
#   - falsifier (what single observation would discard this idea?)
#   - predicted delta range on the primary composite metric
```

## README.md template

```markdown
# H<NN> — <one-line idea title>

## TL;DR

<2–3 sentences>

## Citation

<Full citation: Author YEAR VENUE 'Title' (full-ID)>

## Hypothesis

<Mechanism + expected delta. Word count ≥ 50.>

## Falsifier

<What single observation would discard this idea? Be specific:
"If primary metric delta ≤ threshold on <evaluation setup>,
this idea is DISCARDED.">

## Predicted delta range

| metric    | delta vs baseline | rationale      |
|-----------|-------------------|----------------|
| composite | [+0.01, +0.03]    | <one sentence> |
| primary   | [+0.5, +2.0 pp]   | <one sentence> |
| cost      | [-20%, -5%]       | <one sentence> |

## Status

| stage                              | done? |
|------------------------------------|-------|
| implementation written             | [ ]   |
| tests green                        | [ ]   |
| AUDIT.md filed                     | [ ]   |
| IMPROVEMENTS.md addressed          | [ ]   |
| VERIFY.md sealed                   | [ ]   |
| First experiment archived          | [ ]   |
| Verdict authored                   | [ ]   |

## How to test this idea (idea-specific experiment strategy)

<What is the SINGLE BEST evaluation setup that would prove this idea?
Don't reuse the common runner unless it really is the best test.
Examples:
- For an equivariance idea: a transformed evaluation set.
- For a convergence-speed idea: small data subset, plot steps-to-target.
- For a robustness idea: corrupted inputs.
- For a sparsity idea: long-context or high-cardinality evaluation.>

## Cross-idea interactions

<Which other ideas does this compose well with? Which conflict?
Where in the combination suite does it slot?>
```

## IDEA.md (formal contract)

```markdown
# Formal claim for H<NN>

## Claim

<One sentence, present tense, falsifiable.>

## Falsifier

<One observation that DISCARDS this claim. Quote the threshold.>

## Pre-registered prediction

<Exact numeric range on the primary metric + sub-metrics.>

## Composite fingerprint at time of registration

<Hash of the primary evaluation formula in the project's runner.>

## Signed-off by

<Author identifier + date>
```

## AUDIT.md template (after first implementation, before first run)

```markdown
# AUDIT — H<NN>

> Adversarial self-critique. Treat the implementation as if you are
> reviewing it for a top venue.

## Weaknesses found

1. **<one-line summary>** — <why it matters, where in the code>
2. ...

## Bugs caught by tests

- ...

## Bugs NOT caught by tests but suspected

- ...

## Mitigations queued for IMPROVEMENTS.md

- ...
```

## VERIFY.md template (before first archive)

```markdown
# VERIFY — H<NN>

## Tests

- [ ] tests pass all assertions
- [ ] No new linter warnings on implementation file
- [ ] No new type-check errors on implementation file

## Sanity

- [ ] Vanilla / disabled flag combination produces identical output to the
      literature baseline
- [ ] All flag combinations run forward without shape or type errors (smoke)
- [ ] Resource cost (parameter count, memory, FLOPs) within ±10% of predicted

## Signed off by

<Author identifier + date>
```

## The falsifier-is-a-contract rule

The IDEA.md falsifier is binding. If the pre-registered observation actually
fires, the idea is **DISCARDED** — no goalpost-shifting, no redefining the
metric, no adding new confounds to explain the negative result. Close the
idea, update its STATUS to DISCARDED, and move on.

## Hard rules

1. **No new idea without a citation.** Hunches go in BACKLOG.md.
2. **No experiment until VERIFY.md is signed.** This catches bugs before
   expensive runs.
3. **The IDEA.md falsifier is the contract.** See above.
4. **Cross-idea interactions are documented.** When two ideas compose, both
   READMEs link to each other and to the combination suite entry.
5. **Use 10-spacing numbering.** Never sequential integers.

## What "good" looks like

- A new contributor can read just `ideas/<NN>/README.md` and understand the
  idea, the citation, the falsifier, and the predicted delta range in 5 minutes.
- The AUDIT.md has at least 3 weaknesses listed; if it is empty, the audit
  was lazy.
- VERIFY.md is signed on a real date, not "TBD".
- The falsifier threshold is a number, not a vague phrase.

## Anti-patterns

- Numbering ideas with sequential integers — leaves no insertion room. Use
  10-spacing.
- Mixing two ideas in one sub-directory because they "feel related." Split
  them; the modular contract is that each idea must be independently ablatable.
- A README that does not link to the project's master idea index — creates
  orphans.
- Writing the falsifier as "if results are bad" — that is not falsifiable.
  A falsifier must name a metric, a threshold, and an evaluation setup.
- Signing VERIFY.md before all tests actually pass.

## Cross-references

- `../autoresearch-critic-team/SKILL.md` — audits the implementation against
  the IDEA.md claim.
- `../autoresearch-scicritic-team/SKILL.md` — audits the IDEA.md claim itself
  for scientific merit; checks the falsifier for pre-registration completeness.
- `../autoresearch-fixer-campaign/SKILL.md` — resolves AUDIT.md findings
  before the idea is cleared for experimentation.
- `../autoresearch-winner-archive/SKILL.md` — when an idea's experiment
  becomes the champion config, archive it there.
- `../autoresearch-checkpoint/SKILL.md` — commit the scaffold immediately
  after creation, and after each doc is filled in.
