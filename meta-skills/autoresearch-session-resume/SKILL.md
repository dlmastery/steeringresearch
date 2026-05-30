---
name: autoresearch-session-resume
description: Use at the START of every new session to orient a cold agent — read the checkpoint document, verify git status is clean, reconstruct the current state of the project (what ran, what is queued, what the champion config is), and resume from the exact point the prior session left off. Also use to WRITE the checkpoint document at the END of a session so the next session can resume cold.
---

# Skill — Session resume from checkpoint

## The problem this solves

A long-running research project spans many sessions. Between sessions:

- The conversation context window is lost.
- Background compute tasks may have completed, failed, or still be running.
- The working tree may have uncommitted changes from the prior session.
- The champion config may have changed.

Without a structured checkpoint document, the next session wastes time
reconstructing state from git log and result files. This skill defines what
to record at END-of-session and how to read it at START-of-session.

## Session-start ritual (READ this first, every session)

```
1. Read the checkpoint document (see location convention below).
2. Run `git status --short`.
   - If CLEAN: the prior session committed everything. Proceed.
   - If DIRTY: the prior session crashed or was interrupted.
     DO NOT start new work. Commit the dirty state first
     (see autoresearch-checkpoint/SKILL.md), then re-read this ritual.
3. Run `git log --oneline -10` to confirm the last checkpoint commit.
4. Read the "Current state" section of the checkpoint doc to determine:
   - What experiment ran last.
   - What is queued next.
   - What the current champion config is.
   - What outstanding audit findings remain.
5. Confirm any background compute tasks that were in-flight:
   - Check the result directory for completed output.
   - If completed: archive the result, update the ledger, checkpoint commit.
   - If still running: monitor before starting new work.
6. Only after steps 1–5 are complete, begin new work.
```

## Checkpoint document location

Store the checkpoint document at a consistent path in the repository, e.g.:

```
CHECKPOINT.md         (project root, always overwritten in-place)
```

or

```
checkpoints/SESSION_<YYYY-MM-DD>.md   (dated, append-only)
```

Choose one convention and apply it consistently. The dated variant is
preferable when session-to-session diffs matter for audit purposes.

## Checkpoint document template (write at END of session)

```markdown
# Session checkpoint — <YYYY-MM-DD HH:MM UTC>

## Git state at close
- Branch: <branch name>
- Last commit: <SHA> — <title>
- `git status`: CLEAN / DIRTY (describe if dirty)

## Current champion config
- Config identifier: <name or tag>
- Primary metric: <value>
- Commit where result was archived: <SHA>
- Archive location: <path>

## Last experiment run
- Experiment tag: <tag>
- Status: COMPLETE / RUNNING / FAILED
- Result: <primary metric value or "pending">
- Output location: <path>

## Queue — next experiments to run (in order)
1. <experiment tag> — <one-line rationale>
2. <experiment tag> — <one-line rationale>
...

## Outstanding audit findings
- BROKEN: <list of hypothesis IDs>
- MAJOR:  <list of hypothesis IDs>
- (MINOR findings can be deferred)

## Outstanding fixer work
- <hypothesis ID>: <one-line description of what remains>

## Background tasks in-flight at close
- Task: <description>
  Status: RUNNING / COMPLETED
  Expected output: <path>

## Notes for next session
<Any context the next session needs that doesn't fit the above structure>

## Resume instruction
Start by completing the session-start ritual in
`skills/autoresearch-session-resume/SKILL.md`. Do NOT skip the
`git status` gate.
```

## The "git status clean" gate

This is a hard gate, not a soft recommendation. If `git status` is dirty at
session start:

- There are uncommitted changes from a prior session.
- Those changes may include experiment results, code edits, or doc updates
  that were never pushed to the remote.
- Starting new work on top of a dirty tree risks losing the prior work or
  creating a confusing commit history.

**Action:** commit the dirty state with a message like
`"Emergency checkpoint: uncommitted state from prior session <date>"`,
push it, then re-read the checkpoint document and the git log.

## What to record so the next session resumes cold

The checkpoint document must be self-contained enough that an agent with NO
memory of the prior session can:

1. Know what the champion config is without reading all result files.
2. Know what experiment to run next without reading all queue files.
3. Know what audit findings remain without reading all audit files.
4. Know whether any background task completed and whether its output was
   archived.

If any of these four items is missing from the checkpoint document, the
document is incomplete.

## Session-end discipline

At the end of every session:

1. Write or overwrite the checkpoint document using the template above.
2. Commit it: `git add CHECKPOINT.md && git commit -m "Session checkpoint <date>"`.
3. Push: `git push`.
4. Verify `git status` is CLEAN before closing the session.

Never close a session with a dirty working tree.

## Anti-patterns

- Starting new work before checking `git status`.
- Skipping the checkpoint document read at session start — the agent wastes
  time re-deriving state that was already recorded.
- Writing a checkpoint document but not committing and pushing it — a local
  checkpoint is worthless on a machine that powers off.
- Recording "experiment X ran" without recording whether the result was
  archived and the ledger was updated.
- Leaving background tasks in-flight without recording their expected output
  location — the next session cannot find the result.

## Cross-references

- `../autoresearch-checkpoint/SKILL.md` — the per-milestone commit discipline
  that keeps the working tree clean between sessions.
- `../autoresearch-winner-archive/SKILL.md` — the champion config recorded
  in this document comes from the winner archive.
- `../autoresearch-fixer-campaign/SKILL.md` — outstanding fixer work is
  recorded in this document.
- `../autoresearch-critic-team/SKILL.md` — outstanding audit findings are
  recorded in this document.
- `../autoresearch-multi-agent-dispatch/SKILL.md` — multi-agent dispatches
  that were in-flight at session close are recorded here.
