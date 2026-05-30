---
name: steering-critic-team
description: >
  Use when running an implementation critique on a steering idea's code, tests,
  and AUDIT.md findings. Checks that the implementation matches the IDEA.md claim,
  that tests are non-trivial, and that the AUDIT has at least 3 real weaknesses.
  Carries the same-model-family circularity disclosure. This is the steering
  instantiation of the meta autoresearch-critic-team process.
---

# Skill — steering-critic-team

This is the steering instantiation of [meta-skills/autoresearch-critic-team/SKILL.md](../../meta-skills/autoresearch-critic-team/SKILL.md).

Read that file first for the general critic-team protocol. This file adds
steering-specific critique targets and the mandatory circularity disclosure.

---

## Steering-specific notes

### Mandatory circularity disclosure

Every critic-team verdict must open with:

> **Circularity disclosure:** This audit was produced by a model in the same
> family as the one that wrote the code under review. All verdicts carry the
> qualification: "Internal QA pass — independent external review pending."
> Do not present any verdict here as independent validation.

This is non-negotiable (CLAUDE.md Section 14).

### What the critic team audits (steering-specific targets)

**1. Implementation vs IDEA.md claim alignment**
- Does `implementation.py:extract()` produce a unit-normed vector? (required)
- Does `implementation.py:apply()` modify ONLY the claimed axis? (no side effects)
- Does `implementation.py:evaluate()` use the fingerprinted composite formula?
- Does the implementation cite the arXiv ID from README.md / IDEA.md?

**2. Data-split hygiene**
- Is `audit_or_die()` from `src/steering/extract.py` called before any eval?
- Are extraction and eval indices disjoint and pinned with a fixed seed?
- Is activation caching verified not to recompute inside a sweep loop?

**3. 12-axis compliance**
- Does the implementation change ONLY the claimed axis (no implicit axis coupling)?
- If two axes necessarily couple (e.g., spherical metric requires rotation),
  is the coupling documented in both README.md and IDEA.md?

**4. Geometry leading-indicator instrumentation**
- Is delta_norm logged? Is eff_rank_drop logged? Is norm_budget computed?
- Is the N5 budget constraint enforced (reject if norm_budget > threshold)?

**5. Test non-triviality**
- Are tests.py tests actually asserting something about steering behavior,
  or just importing without error? (trivial tests are a AUDIT finding)
- Is the zero-alpha identity test present? (Rung 0 gate)
- Is the data-split disjoint test present?

**6. AUDIT.md completeness**
- At least 3 weaknesses listed with severity labels
- At least one HIGH-severity item with a queued IMPROVEMENTS.md fix
- An empty AUDIT.md or one with "No weaknesses found" is flagged as lazy

### Critique format

```
## Critic-team verdict for H<NN> — <idea title>

**Circularity disclosure:** [mandatory, see above]

### Findings

| # | file | line | finding | severity | action |
|---|------|------|---------|----------|--------|
| 1 | implementation.py | 42 | <finding> | HIGH | fix before VERIFY |
| 2 | tests.py | 18 | <finding> | MEDIUM | fix or accept |
| 3 | AUDIT.md | — | <finding> | LOW | note |

### Overall verdict

[ ] CLEARED — no HIGH-severity issues; implementation may proceed to VERIFY
[ ] BLOCKED — HIGH-severity issues remain; implement fixes before VERIFY

**Internal QA pass — independent external review pending.**
```

---

## Cross-references

- Meta-process: `../../meta-skills/autoresearch-critic-team/SKILL.md`
- Sci-critic (hypothesis quality, not code): `../steering-scicritic-team/SKILL.md`
- Fixer campaign: `../../meta-skills/autoresearch-fixer-campaign/SKILL.md`
- AUDIT template: `../../ideas/_TEMPLATE/AUDIT.md`
- IMPROVEMENTS template: `../../ideas/_TEMPLATE/IMPROVEMENTS.md`
