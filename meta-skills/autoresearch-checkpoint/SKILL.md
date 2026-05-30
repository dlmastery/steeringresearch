---
name: autoresearch-checkpoint
description: Use whenever progress has been made — code edit, test passing, experiment completed, dashboard refreshed, ledger updated. Commits and pushes a checkpoint to the remote so a power outage or session crash never loses progress. Default cadence ≤ 15 min during active work; mandatory before and after every background compute task.
---

# Skill — Periodic remote checkpoint

## When to use

Continuously. This is not a one-shot skill — it is the heartbeat of the
project. Specifically:

| trigger                                             | priority  | what to commit                                      |
|-----------------------------------------------------|-----------|-----------------------------------------------------|
| File edit complete and tests green                  | high      | the edited files + test output                      |
| Background compute task about to launch             | mandatory | code + configs as they will run                     |
| Background compute task completed                  | mandatory | result artifacts + log + dashboard refresh          |
| New or updated index, ledger, or findings doc       | high      | the document                                        |
| Skill or architecture doc updated                   | high      | the document                                        |
| Every ~15 min of active editing                     | medium    | whatever changed                                    |
| Session resume / wake-up                            | mandatory | check `git status` first thing; do not start work   |
|                                                     |           | if the tree is dirty — commit the prior state first |

## Why

- Power outage on a machine with no UPS.
- OS killing background tasks on sleep or lid-close.
- The conversation's own context window exhausting before work is committed.
- Recovery should always be from a remote, never from a local WIP state.

## How — the command (PowerShell)

```powershell
git status --short                       # quick sanity check first
git add -A
git -c user.name="<name>" -c user.email="<email>" commit -m "<msg>"
git push
```

For multi-line commit messages (HEREDOC):

```powershell
git commit -m @'
<title line ≤ 70 chars>

<body line 1>
<body line 2>
'@
```

The Bash equivalent:

```bash
git status --short
git add -A
git -c user.name="<name>" -c user.email="<email>" \
    commit -m "<title>

<body>"
git push
```

## Commit-message contract

A good checkpoint commit message:

- **Title (≤ 70 chars):** what changed at the highest level.
- **Body:** what is recoverable from this checkpoint. If it is mid-flight
  work (a compute task still running, the dashboard partially updated), say so.
- **No `--no-verify`, no `--amend`.** Always a fresh commit.

Example bad message: `wip`

Example good message:
```
Mid-sweep checkpoint 3/11: baselines + variant-A complete

Recoverable from this commit:
- baseline run:  primary metric 84.78
- variant-A run: primary metric 82.16
- variant-B run: primary metric 80.11
Sweep still running variant-C seed=0; remaining 8 runs queued.
```

## Hard rules

1. **Push every commit.** Local commits on a dead machine are worthless.
   `git push` is part of the checkpoint, not a follow-up.
2. **Many small commits beat one big commit.** Granularity matters for
   selective revert and bisect.
3. **Always glance at `git status` first.** Do not blindly `git add -A`
   if you suspect a secret or a large binary slipped in.
4. **Never `--no-verify`.** Hooks exist to catch bugs; bypassing them defeats
   the safety net.
5. **Never `--amend`.** Amending rewrites history; always create a new commit.
6. **Background-task launch is a checkpoint trigger** — commit BEFORE launch
   (so the launch state is recoverable) AND AFTER the task completes (so
   the artifacts are recoverable).

## Per-experiment commit cadence (sharpened rule)

The strictest cadence used in practice:

- **Per-experiment commit + push BEFORE moving to the next experiment.** No
  batching. Every experiment's full state — result record, reasoning
  annotation, dashboard sync, checkpoint document — lands on the remote
  before the next launch.
- **Allowed exception: cheap-burst commits.** For very-cheap (< 60 s)
  rapid-fire bursts of 3–5 sequential experiments at the same baseline, a
  single commit covering the burst is acceptable IF the dashboard is synced
  and pushed at the END of the burst BEFORE switching to a different config.
- **Pre-flight check on every experiment:** if `git status` shows uncommitted
  changes from the PRIOR experiment, STOP. Commit + push first, then re-read
  this rule, THEN launch the next experiment.

## Anti-patterns

- "I'll commit at the end of the turn." — No. Commit on every milestone.
- "I'll squash these into one nice commit later." — No. Granular commits
  preserve a useful history.
- Adding secrets, virtual-environment directories, dataset tarballs, or raw
  binary checkpoints larger than ~100 MB without a `.gitignore` entry or
  large-file store entry.
- Pushing without verifying `git push` succeeded — check the exit code or
  the remote's commit history.

## Cross-references

- `../autoresearch-session-resume/SKILL.md` — the session-resume document
  this skill persists; "git status clean" is the session-start gate.
- `../autoresearch-multi-agent-dispatch/SKILL.md` — in multi-agent contexts,
  each agent uses scoped `git add` rather than `-A`, but the push discipline
  is identical.
- `../autoresearch-fixer-campaign/SKILL.md` — fixer agents checkpoint after
  each fix-and-verify cycle.
- `../autoresearch-winner-archive/SKILL.md` — a winner archive event is also
  a mandatory checkpoint trigger.
