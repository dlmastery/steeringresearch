---
name: autoresearch-winner-archive
description: Use when an experiment's result beats the current global best — the winning config, result, and reasoning are immediately archived as an immutable record. The archive is append-only: old winner records are never deleted or modified. The next session reads the archive to know the current champion config without re-running experiments.
---

# Skill — Winner archive (champion config tracker)

## When to use

- An experiment completes and its primary metric exceeds the current champion.
- The hillclimber selects a new KEEP verdict after a sweep.
- A fixer campaign resolves BROKEN/MAJOR findings and the re-run of a
  previously-losing config now beats the champion.
- Session start: read the archive to restore the current champion config
  without re-running anything.

## What to archive

When a new champion is established, immediately write (or append) a winner
record containing:

1. **Config snapshot** — the exact configuration that produced the result
   (parameter names, values, random seeds, framework versions). Enough detail
   to reproduce the run from scratch.
2. **Result snapshot** — the primary metric value, all sub-metrics, and the
   evaluation setup (dataset, split, number of seeds, aggregation method).
3. **Reasoning** — why this config beat the prior champion (which hypothesis
   it implements, what mechanism is credited, what the marginal delta was).
4. **Commit SHA** — the git commit at which this result was produced and
   archived.
5. **Provenance** — the experiment tag, the run directory or artifact path,
   and the ledger row.

## Archive location and naming convention

```
archive/winners/
├── WINNER_LOG.md             — append-only human-readable log
├── champion_config.yaml      — always overwritten with the CURRENT champion
└── <YYYY-MM-DD>_<tag>/
    ├── config.yaml           — frozen config snapshot
    ├── result.json           — frozen result snapshot
    └── REASONING.md          — frozen reasoning doc
```

The dated sub-directory is **immutable once written**. Never edit or delete
it. `champion_config.yaml` at the root is the only mutable file — it always
reflects the current champion for quick access by the session-resume ritual.

## WINNER_LOG.md append format

```markdown
---

## <YYYY-MM-DD HH:MM UTC> — New champion: <config tag>

**Primary metric:** <value> (prev champion: <value>, delta: <+/->)
**Config:** `archive/winners/<date>_<tag>/config.yaml`
**Result:** `archive/winners/<date>_<tag>/result.json`
**Reasoning:** `archive/winners/<date>_<tag>/REASONING.md`
**Commit:** <SHA>
**Hypothesis:** H<NN> — <one-line hypothesis title>
**Mechanism credited:** <one sentence>
**Evaluation:** <dataset, split, seeds, aggregation>
```

## The immutable winner record rule

Once a dated sub-directory is committed, it is **never modified**. If a
subsequent experiment reveals that the archived result was produced by buggy
code (e.g., a BROKEN critic finding that was missed), the correct action is:

1. Re-run the experiment on the fixed code.
2. If the fixed run still beats all prior champions: archive a NEW winner
   record with a note in REASONING.md that the prior record was produced
   under a bug.
3. If the fixed run no longer beats the prior champion: demote the prior
   winner by appending a `## DEMOTION` note to WINNER_LOG.md (never delete
   the record).

Do not silently overwrite or delete historical winner records.

## Demotion format

```markdown
## <YYYY-MM-DD> — DEMOTION: <config tag>

The winner recorded on <original date> was produced by code with
a BROKEN finding (H<NN>, `audits/G<N>_audit.md`). The re-run on
fixed code produced primary metric <value>, which is
<above/below> the threshold. This record is demoted.
Fixer commit: <SHA>
```

## Hillclimber integration

The project's hillclimber (selection / ranking pass) determines KEEP vs
DISCARD verdicts. When a KEEP verdict is issued:

1. Check whether the KEEP config beats the current `champion_config.yaml`.
2. If yes: immediately run the winner-archive procedure.
3. If no: still checkpoint the KEEP result (see
   `../autoresearch-checkpoint/SKILL.md`) but do not archive as winner.

The hillclimber verdict alone does not trigger archiving — only "beats the
global best" does.

## Champion config format (`champion_config.yaml`)

```yaml
# Current champion — overwrite on every new champion
# DO NOT edit manually; written by autoresearch-winner-archive
updated: <YYYY-MM-DD>
tag: <config tag>
primary_metric: <value>
config_snapshot: archive/winners/<date>_<tag>/config.yaml
result_snapshot: archive/winners/<date>_<tag>/result.json
commit: <SHA>
```

## Output discipline

- `git add archive/winners/` scoped — never `-A` unless no other agent is
  running concurrently.
- Commit immediately after archiving: the champion config should never be
  ahead of the remote.
- Message format: `"Archive winner: <tag> primary=<value> (prev=<value>)"`.
- Push: `git push`.

## Session-start use

At session start, read `archive/winners/champion_config.yaml` to restore the
current champion without re-running experiments. The
`../autoresearch-session-resume/SKILL.md` ritual includes this step.

## Anti-patterns

- Overwriting a dated winner sub-directory instead of creating a new one —
  immutability is the whole point.
- Archiving a config produced by code with an outstanding BROKEN finding —
  always resolve BROKEN findings and re-run before archiving.
- Archiving without a commit SHA — the record is unverifiable without a SHA.
- Updating `champion_config.yaml` without also appending to `WINNER_LOG.md`
  — the log must reflect every champion transition.
- Deleting demoted winner records — demotion is an annotation, not a deletion.

## Cross-references

- `../autoresearch-checkpoint/SKILL.md` — mandatory push immediately after
  archiving.
- `../autoresearch-session-resume/SKILL.md` — reads `champion_config.yaml`
  at session start.
- `../autoresearch-fixer-campaign/SKILL.md` — a fixer campaign that resolves
  BROKEN findings may trigger a re-run and a new winner archive event.
- `../autoresearch-critic-team/SKILL.md` — a BROKEN verdict on an archived
  winner triggers the demotion procedure.
- `../autoresearch-idea-scaffold/SKILL.md` — the winning hypothesis ID maps
  back to its `ideas/NN_<short>/` directory; cross-link the archive record
  to the idea scaffold.
