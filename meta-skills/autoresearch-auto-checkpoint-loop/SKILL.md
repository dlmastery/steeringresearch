---
name: autoresearch-auto-checkpoint-loop
description: Use when launching any background task expected to run longer than 15 minutes (a multi-run sweep, a parallel agent team, a long training job). Run a companion background loop that auto-commits and pushes new artifacts every ~10 minutes with retry-wrapped scoped commits. Distinct from the milestone checkpoint skill — this loop provides crash safety for the duration of the long task, not just at named milestones.
---

# Skill — Background auto-checkpoint loop for crash safety

## When to use

- **Before** launching any background task expected to run > 15 min:
  a multi-run sweep campaign, a parallel agent team, a long training
  or evaluation job, a fixer campaign.
- This skill is the **crash-safety companion** to
  [`../autoresearch-checkpoint/SKILL.md`](../autoresearch-checkpoint/SKILL.md),
  which is the one-shot milestone commit skill. The auto-loop runs
  **continuously** while the long task runs; the milestone checkpoint
  runs at named events (before/after task launch, on file edit, etc.).
  Both are needed for full crash safety.
- After a crash recovery: restart the loop as soon as the resumed
  long task is relaunched.

## The distinction: auto-loop vs milestone checkpoint

| property | milestone checkpoint | auto-checkpoint loop |
|---|---|---|
| Trigger | Named event (file edit, task launch/stop, ~15 min idle) | Continuous, fixed interval |
| Scope | Whatever changed at the milestone | Only the long task's output directory |
| Runs in | Foreground (blocking) | Background (non-blocking) |
| Purpose | Never lose a specific milestone | Never lose more than one interval of a long task |
| Skill | `autoresearch-checkpoint` | **this skill** |

Both run during a long campaign. Neither replaces the other.

## The loop

The loop is a bounded, background-safe, scoped commit loop. Adapt
platform and path placeholders to the project.

**Shell variant:**

```bash
# Bounded: 10 ticks × 10 min = up to 100 min.
# Size the tick count to slightly exceed your expected campaign duration.
# Scope: only the long task's output directory (NOT src/ or skills/).
for i in $(seq 1 10); do
  sleep 600
  if [ -n "$(git status -s path/to/campaign/outputs/)" ]; then
    git add path/to/campaign/outputs/
    git commit -m "Auto-checkpoint: campaign results (tick $i)" \
      >/dev/null 2>&1 \
      && git push >/dev/null 2>&1 \
      && echo "tick $i: committed + pushed"
  else
    echo "tick $i: no new results"
  fi
done
echo "Auto-checkpoint loop finished"
```

Launch with a background-task mechanism (the tool's `run_in_background`
flag or equivalent). Do NOT run in the foreground — that would block
the agent for the entire loop duration.

**PowerShell variant (Windows):**

```powershell
for ($i = 1; $i -le 10; $i++) {
  Start-Sleep -Seconds 600
  $status = git status -s path/to/campaign/outputs/
  if ($status) {
    git add path/to/campaign/outputs/
    git commit -m "Auto-checkpoint: campaign results (tick $i)"
    if ($?) {
      git push
      if ($?) { Write-Output "tick $i: committed + pushed" }
    }
  } else {
    Write-Output "tick $i: no new results"
  }
}
Write-Output "Auto-checkpoint loop finished"
```

## Stop discipline

When the foreground task finishes, **explicitly stop the auto-loop**.
The loop terminates on its own at the configured iteration count (bounded
loops are safe to forget), but stopping it early avoids redundant ticks
and keeps the commit log clean.

Stop mechanism: the tool's `TaskStop` facility, a PID-based `kill`,
or any equivalent that terminates the background process. After stopping,
run one final milestone checkpoint (the `autoresearch-checkpoint` skill)
to capture the task's last artifacts.

## Disjoint commit scope — the critical constraint

The auto-loop commits ONLY the long task's output directory (e.g.,
`experiments/<campaign>/`). If parallel agents are concurrently
committing other files (source code, dashboard files, skill edits),
their scopes must be disjoint from the loop's scope.

Why this matters:
- An agent mid-edit of a `src/` file must NOT be swept into the loop's
  commit. The loop's `git add <output-dir>/` scoping prevents this.
- Two writers with overlapping scopes will race on the git index lock.
  Disjoint scopes eliminate the race without retry complexity in the loop
  itself.

Each concurrent writer (the loop + each parallel agent) uses
`git add <its-own-paths>` (not `git add -A`) and its own retry wrapper.

## Crash recovery

After a power outage, OS sleep, or kernel panic during a long campaign:

1. Check the remote: confirm what was pushed before the crash. The
   loop guarantees at most one interval (default ~10 min) of work was
   lost — everything before the last pushed tick is on the remote.
2. Identify surviving artifacts: the missing work is only the in-flight
   run at the moment of the crash, plus anything produced after the
   last pushed tick.
3. Resume the campaign: the project's runner with a `--skip-existing`
   flag (or equivalent) re-runs only the missing rows.
4. Restart the auto-loop alongside the resumed campaign.

This recovery pattern is fast because the loop's crash window is bounded:
a 10-minute tick means at most one interval of results must be re-run.

## Anti-patterns

| Anti-pattern | Why it fails |
|---|---|
| Foreground sleep loop | Blocks the agent for the entire duration |
| Unbounded `while true` loop | Safe only if you remember to stop it; risks polluting the commit log for days if forgotten |
| `git add -A` inside the loop | Sweeps up concurrent agents' in-flight src/ edits into the loop's commit |
| One loop for multiple unrelated campaigns | Commit messages become meaningless; one loop per campaign |
| Skipping the stop-and-final-checkpoint | The last artifacts after the final tick may not be committed |
| Running the loop in the foreground "to watch it" | Blocks the session; use background monitoring tools instead |

## Cross-references

- [`../autoresearch-checkpoint/SKILL.md`](../autoresearch-checkpoint/SKILL.md)
  — the one-shot milestone checkpoint; the companion to this skill.
  Both are needed: the loop for continuous crash safety, the milestone
  for named events.
- [`../autoresearch-multi-agent-dispatch/SKILL.md`](../autoresearch-multi-agent-dispatch/SKILL.md)
  — the skill that fans out parallel agents whose work this loop
  preserves. Disjoint scopes are defined at dispatch time.
- [`../autoresearch-per-hypothesis-hillclimb/SKILL.md`](../autoresearch-per-hypothesis-hillclimb/SKILL.md)
  — running the auto-loop alongside a hill-climb is recommended; the
  hill-climb's per-cell commits cover the happy path, the loop covers
  crashes mid-cell.
- [`../autoresearch-ablation-sweep/SKILL.md`](../autoresearch-ablation-sweep/SKILL.md)
  — sweep campaigns expected to run > 15 min pair naturally with this
  loop.
- [`../autoresearch-meta/SKILL.md`](../autoresearch-meta/SKILL.md) §11
  — the instantiation checklist wraps all ten steps in crash safety
  via the checkpoint heartbeat and session-resume; this loop is the
  long-task crash-safety component of that wrap.
