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
├── PROVENANCE.md    — tracing artifact: experiment tags, commands, result paths
├── implementation.py — standalone module (or note pointing to shared harness)
├── tests.py         — correctness tests
├── AUDIT.md         — self-audit (weaknesses found before first run)
├── IMPROVEMENTS.md  — fix log (addressed weaknesses from AUDIT)
├── VERIFY.md        — verification log: tests green, sanity checks, sign-off
├── experiment.py    — idea-specific experiment driver (or shared harness note)
├── configs/         — configuration files (YAML or equivalent)
├── experiments/     — per-experiment archives
├── results.md       — auto-generated or manually-maintained summary
└── dashboard/       — idea-level dashboard
```

`PROVENANCE.md` is new and mandatory once the first experiment for this idea
has run. See the Provenance contract section below.

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

## PROVENANCE.md template (created after the first experiment runs)

```markdown
# Provenance — H<NN> — <idea short name>

> Generated from experiment_log.jsonl. Update after every new experiment
> that tests this hypothesis.

## Experiments that tested this hypothesis

| tag | rung | verdict | composite | result path | reasoning entry |
|-----|------|---------|-----------|-------------|-----------------|
| expNNN_<short> | SMOKE | KEEP | 0.XXXX | ideas/NN/experiments/expNNN_<short>/ | autoresearch_results/reasoning_annotations.json#expNNN |

## Exact commands

For each experiment tag above, the exact command used:

```bash
# expNNN_<short>
python -m <project>.runner \
  --config ideas/NN_<short>/configs/expNNN_<short>.yaml \
  --tag expNNN_<short> --seed 0 \
  --root ideas/NN_<short>/experiments/expNNN_<short>/run
```

## Result interpretation (2–3 sentences per experiment)

**expNNN_<short>:** [What the numbers mean for this specific hypothesis —
not just the raw scores. Did the mechanism work? What is the implication
for the next experiment on this hypothesis?]

## What new harness code this hypothesis needs (if untested)

[If this hypothesis has NOT yet been tested because the shared harness
cannot do it yet, describe exactly what new capability the harness needs.
If all needed code exists, write "No additional harness code required."]
```

Generate or update `PROVENANCE.md` programmatically from `experiment_log.jsonl`
filtered to the hypothesis's tag(s). The auto-generation script lives in
`scripts/generate_provenance.py` (or the project-equivalent). After each new
experiment verdict, re-run the generator and commit the updated file.

## Shared-harness architecture note (required in README when applicable)

If this idea is tested via a shared harness rather than per-hypothesis code,
the idea's `README.md` MUST include a section:

```markdown
## Harness architecture

This hypothesis is tested via the project's shared experimental harness
(`src/`). The hypothesis is differentiated by:
- Configuration: `configs/expNNN_<short>.yaml`
- Reasoning annotation: `autoresearch_results/reasoning_annotations.json`
  (filtered to tags: [list of tags])
- Result artifacts: `ideas/NN/experiments/expNNN_<short>/`

No separate per-hypothesis implementation file is needed because
[one sentence explaining why the shared harness covers this hypothesis].
```

If the shared harness CANNOT yet test this hypothesis, instead write:

```markdown
## What new harness code is needed

The shared harness does not yet support [specific capability]. To test
this hypothesis, the following must be added to `src/`:
- [bullet list of required harness additions]
```

This prevents a reader from inferring incompleteness from the absence of a
per-hypothesis implementation file.

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
6. **PROVENANCE.md is created after the first experiment.** It must be
   updated (or regenerated) after every subsequent experiment on this idea.
   A tested hypothesis without a PROVENANCE.md is untraceable.
7. **Untested hypotheses declare their harness requirements.** Any idea
   that has never been run must state in its README what new harness code
   is needed. "No experiments yet" is acceptable; "no explanation why" is not.
8. **Shared-harness architecture disclosed.** If the idea relies on the
   shared harness, the README says so explicitly and points to the
   differentiating config/annotation. Absence of a per-hypothesis
   implementation file is explained, not left implicit.

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
- `../autoresearch-findings-ledger/SKILL.md` — the PROVENANCE.md contract
  here and Mandate C in the findings-ledger skill are complementary; the
  provenance artifact feeds the ledger's tracing requirements.
- `../autoresearch-experiment-archive/SKILL.md` — each per-experiment archive
  under `ideas/<NN>/experiments/` is the low-level provenance unit; PROVENANCE.md
  is the human-readable summary layer on top.
