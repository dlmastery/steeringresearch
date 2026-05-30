---
name: steering-checkpoint
description: >
  Use when committing a milestone in the steering autoresearch program. Enforces
  the CLAUDE.md Section 13 checkpoint discipline: commit + push on every
  milestone, specific file scopes, no -A, no --no-verify, never --amend. This is
  the steering instantiation of the meta autoresearch-checkpoint process.
---

# Skill — steering-checkpoint

This is the steering instantiation of [meta-skills/autoresearch-checkpoint/SKILL.md](../../meta-skills/autoresearch-checkpoint/SKILL.md).

Read that file first for the general checkpoint protocol. This file adds the
steering-specific commit scope and the dashboard regeneration requirement.

---

## Steering-specific notes

### Commit triggers (CLAUDE.md Section 13)

Commit + push at ALL of the following milestones:
- File edit + tests green (any implementation.py or tests.py edit)
- Run folder produced (any experiments/<timestamp>/ dir created)
- EXPERIMENT_LEDGER.md row added
- FINDINGS.md updated (finding graduated or screening observation added)
- Dashboard regenerated (master + any sub-dashboard)
- SKILL.md edited (skill or meta-skill update)
- CLAUDE.md edited (process change)
- IDEA_TABLE.md updated (status or rung change)
- Before AND after every background task (GPU run)
- Every ~15 minutes of active editing
- FIRST THING on every session start (commit any dangling changes from prior session)

### Pre-commit check

Before every commit:
1. `git status` — if dirty from experiment N, STOP; commit experiment N before launching N+1.
2. Regenerate dashboard if EXPERIMENT_LEDGER.md changed.
3. Validate composite fingerprint in IDEA.md files matches current eval.py.
4. Run `pytest` (or at minimum `pytest --co -q` to collect) — do not commit a broken test suite.

### Scope rules (steering-specific)

NEVER use `git add -A` or `git add .`. Always scope by file:
```
git add EXPERIMENT_LEDGER.md ideas/<NN>/results.md dashboard/index.html
```

Disjoint scopes when multi-agent:
- Agent A: EXPERIMENT_LEDGER.md, FINDINGS.md, IDEA_TABLE.md (ledger agent)
- Agent B: dashboard/ and docs/dashboard/ (dashboard agent)
- Agent C: ideas/<NN>/ (per-hypothesis agent, one idea at a time)
- Agent D: skills/ and meta-skills/ (process agent)
- Agent E: src/ and tests/ (implementation agent)

Two agents MUST NOT commit to the same file simultaneously.
Retry-wrapped commits: 5 attempts, pull-rebase fallback on conflict.

### Forbidden git flags

- NEVER `--no-verify` (hooks enforce fingerprint and lint)
- NEVER `--amend` (append-only log discipline; amend = lost history)
- NEVER `--force` to main/master
- NEVER `-A` or `.` for staging

### Commit message format

```
[steering/<block>/<id>] <one-line summary>

- <bullet 1: what changed>
- <bullet 2: why it changed>
- Composite fingerprint: <sha>
- Rung: <N> | Verdict: <KEEP/DISCARD/NEAR-MISS>
```

Examples:
```
[steering/A/E1] E1 SMOKE complete: knee at N=32

- Behavior success at N=32 = 91% of N=256 asymptote
- Updated EXPERIMENT_LEDGER.md row 1; rung 1 KEEP
- Composite fingerprint: a9001e87087e
- Rung: 1 | Verdict: KEEP
```

---

## Quick checklist

- [ ] `git status` clean from prior experiment before launching next
- [ ] Dashboard regenerated if ledger changed
- [ ] Composite fingerprint validated in IDEA.md
- [ ] Tests pass (at minimum `pytest --co -q`)
- [ ] File scope is specific (no -A, no .)
- [ ] Commit message includes fingerprint, rung, verdict
- [ ] Pushed to remote immediately after commit

---

## Cross-references

- Meta-process: `../../meta-skills/autoresearch-checkpoint/SKILL.md`
- Commit discipline: CLAUDE.md Section 13
- Dashboard regeneration: `../steering-dashboard/SKILL.md`
- Multi-agent scope: CLAUDE.md Section 14 and `../steering-multi-agent-dispatch/SKILL.md`
