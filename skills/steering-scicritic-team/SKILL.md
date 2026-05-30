---
name: steering-scicritic-team
description: >
  Use when auditing the scientific quality of a steering hypothesis: falsifier
  completeness, pre-registration adequacy, numerology check, corpus traceability,
  and arXiv ID validity. Distinct from steering-critic-team (which audits code).
  Carries the same-model-family circularity disclosure. This is the steering
  instantiation of the meta autoresearch-scicritic-team process.
---

# Skill — steering-scicritic-team

This is the steering instantiation of [meta-skills/autoresearch-scicritic-team/SKILL.md](../../meta-skills/autoresearch-scicritic-team/SKILL.md).

Read that file first for the general scientific critique protocol. This file adds
steering-specific checks: the 12-axis taxonomy, corpus traceability, and the
numerology check relevant to the steering domain.

---

## Steering-specific notes

### Mandatory circularity disclosure

Every scicritic verdict must open with:

> **Circularity disclosure:** This scientific audit was produced by a model in
> the same family as the one that formulated the hypothesis. All verdicts carry:
> "Internal QA pass — independent external review pending."

### What the scicritic team audits (steering-specific)

**1. Falsifier completeness (the most important check)**

A valid steering falsifier must specify ALL of:
- A METRIC (one of the five axes or geometry leading indicators)
- A THRESHOLD (a number, not "low" or "worse")
- A BASELINE (which model config, which dataset split, which rung)
- A CONSEQUENCE ("this claim is DISCARDED")

Failing: "If results are bad" — NOT a falsifier.
Failing: "If behavior doesn't improve" — NOT a falsifier (no threshold).
Passing: "If Spearman(Fisher, efficacy) < 0.7 on Gemma-2-2B refusal at Rung-1
SMOKE, this claim is DISCARDED" — complete falsifier.

**2. Pre-registration adequacy**

- Is the success criterion written BEFORE any run? (check git timestamps)
- Is the predicted value range informed by the cited paper, or is it fabricated?
- Is the range so wide that almost any result qualifies? (mark NUMEROLOGY)
- Is the composite fingerprint recorded? (a9001e87087e placeholder is acceptable
  at registration; must be updated before the first run)

**3. Numerology check (steering-specific)**

A hypothesis is NUMEROLOGY if:
- The predicted optimal value is a POINT ESTIMATE and nearby values would
  trivially also pass (e.g., "layer 18 is optimal" when layers 16-20 all
  perform similarly)
- The threshold is not motivated by a mechanistic prediction from the cited
  paper (it is just a round number)
- The metric is not the natural primary output of the cited method

For steering: ">=90% of asymptote at N=50 pairs" is mechanistic (it follows
from the 1/sqrt(N) variance reduction of DiffMean). ">=90% at N=42 pairs" would
be NUMEROLOGY.

**4. 12-axis taxonomy check**

- Is the primary axis correctly identified?
- If the intervention touches multiple axes, are all of them listed?
- Is the hypothesis falsifiable WITH RESPECT TO the stated axis? (e.g., a
  hypothesis about axis A3 coefficient should be falsified by a metric that
  changes with alpha, not a metric that is alpha-invariant)

**5. Corpus traceability**

- Is every quantitative threshold traceable to a specific corpus document?
- Are inherited numbers tagged [NEEDS VERIFICATION]?
- Is each arXiv ID real? (run a spot-check on at least 2 IDs per hypothesis;
  flag [UNVERIFIED] if the paper cannot be found)

**6. Citation format compliance**

Format: `Author1, ..., YEAR VENUE 'Title' (arXiv:XXXX.XXXXX)`
- Year must be 4 digits
- Venue must include workshop if applicable
- arXiv ID must be in format YYYY.NNNNN (5 digits after dot)

### Scicritic verdict format

```
## Scicritic verdict for H<NN> — <idea title>

**Circularity disclosure:** [mandatory, see above]

### Findings

| check | result | notes |
|-------|--------|-------|
| Falsifier complete | PASS / FAIL | <if FAIL: what is missing> |
| Pre-registration adequate | PASS / FAIL | <if FAIL: what is missing> |
| Numerology check | PASS / NUMEROLOGY | <if NUMEROLOGY: explain> |
| 12-axis alignment | PASS / FAIL | <if FAIL: axis mismatch> |
| Corpus traceability | PASS / FAIL | <if FAIL: unverified numbers> |
| Citation format | PASS / FAIL | <if FAIL: format violations> |

### Verdict tier assigned

[ ] NOVEL+TESTABLE
[ ] DERIVATIVE+TESTABLE
[ ] NUMEROLOGY — DO NOT HILL-CLIMB
[ ] UNFALSIFIABLE — revise hypothesis before any experiment
[ ] FALSIFIED — (only after an experiment fires the falsifier)

### Overall

[ ] CLEARED — hypothesis is scientifically sound; may proceed to implementation
[ ] NEEDS REVISION — <specific revision required before proceeding>

**Internal QA pass — independent external review pending.**
```

---

## Cross-references

- Meta-process: `../../meta-skills/autoresearch-scicritic-team/SKILL.md`
- Implementation critic: `../steering-critic-team/SKILL.md`
- Paper rigor: `../steering-paper-rigor/SKILL.md`
- Hypothesis registry: `../../IDEA_TABLE.md`
- Verdict tiers: CLAUDE.md Section 7
