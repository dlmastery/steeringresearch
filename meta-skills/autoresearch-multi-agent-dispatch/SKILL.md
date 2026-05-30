---
name: autoresearch-multi-agent-dispatch
description: Use when 8+ independent code/doc tasks can run in parallel — pattern for dispatching N agents with disjoint file scopes, retry-wrapped commits, and index.lock contention handling. Each agent edits only its scoped files, commits with a 5-attempt retry loop (add → commit → push; on push fail pull-rebase + retry).
---

# Skill — Parallel multi-agent dispatch with disjoint scopes

## When to use

- Any time the workload partitions into 2+ independent groups (audits by
  hypothesis group, fixers per module family, doc-sync per file family).
- Compute-bound work (e.g., sequential hardware constraints) cannot be
  parallelised; ONLY code, docs, research, audit, and critique work parallelises.

## The contract per dispatched agent

Each agent gets:

1. **Disjoint file scope** — an explicit list of the exact files this agent
   is allowed to edit. Other agents own other files; do NOT touch them.
2. **Scoped `git add <specific paths>`** — NEVER `git add -A` in a
   multi-agent dispatch (would sweep in another agent's in-flight changes).
3. **Retry-wrapped commit + push** — wrap every commit in a 5-attempt loop:
   `git add` → `git commit` → `git push`; on push failure `git pull --rebase`
   then retry; 3-second wait between attempts. This handles both `index.lock`
   races and non-fast-forward collisions.
4. **Identity flags on commit** — `git -c user.name=… -c user.email=…`
   per-commit (no `--global`).
5. **Bounded structured return** ≤ 200–250 words: commit SHA + tier counts
   + top findings.

## The PowerShell retry pattern

```powershell
$ok = $false
for ($i = 0; $i -lt 5 -and -not $ok; $i++) {
    git add <scoped paths>
    if (-not $?) { Start-Sleep 3; continue }
    git -c user.name="Agent-Name" -c user.email="agent@project" commit -m "..."
    if (-not $?) { Start-Sleep 3; continue }
    git push
    if ($?) { $ok = $true } else { git pull --rebase 2>$null }
}
```

## The Bash equivalent

```bash
for i in 1 2 3 4 5; do
  git add <scoped paths> && \
    git -c user.name="Agent-Name" -c user.email="agent@project" commit -m "..." && \
    git push && break
  git pull --rebase || true
  sleep 3
done
```

## Disjoint-scope design — how to partition

Plan the agent boundary BEFORE dispatch. The boundaries that work in practice:

| dispatch type       | partition axis          | example                                                   |
|---------------------|-------------------------|-----------------------------------------------------------|
| Implementation      | thematic group          | one agent per hypothesis group directory                  |
| Audit               | thematic group + type   | one `audits/G<N>_audit.md` per agent                     |
| Doc-sync            | doc family              | one agent for index + table; one for README + overview; one for ledger + architecture |
| Fixer               | primary source file     | one agent per primary module family                       |
| Dashboard           | layer                   | one agent for the build script + renderer; one for README links |

If two agents would touch the SAME file, MERGE their scopes into one agent.
(Example: if Fixer-A and Fixer-B both need `common_module.py`, merge them
into a single Fixer-Common agent.)

## Why retry-wrapped is mandatory

- Concurrent `git commit` on the same repo races on `.git/index.lock`.
  Without retry, one of the agents fails its commit silently.
- Concurrent `git push` races on remote refs. Without `pull --rebase +
  retry`, one of the agents permanently aborts and loses its work.
- An auto-checkpoint loop committing experiment artifacts also races with
  agents committing source files. Retry-wrapped is the only safe multi-writer
  pattern.

## Partition-by-thematic-group reference table

Use this table to assign group boundaries when the project's hypothesis set
spans multiple orthogonal concerns:

| group label | typical scope                                          | example output files                    |
|-------------|--------------------------------------------------------|-----------------------------------------|
| G1          | foundational / baseline hypotheses                     | `audits/G1_audit.md`                   |
| G2          | structural / architectural hypotheses                  | `audits/G2_audit.md`                   |
| G3          | training-dynamics hypotheses                           | `audits/G3_audit.md`                   |
| G4          | regularisation hypotheses                              | `audits/G4_audit.md`                   |
| G5          | optimiser / scheduling hypotheses                      | `audits/G5_audit.md`                   |
| G6          | data / evaluation hypotheses                           | `audits/G6_audit.md`                   |
| G7          | composition / multi-component hypotheses               | `audits/G7_audit.md`                   |
| G8          | infrastructure / tooling hypotheses                    | `audits/G8_audit.md`                   |

Adjust group labels and descriptions to the project's actual partitioning.

## Anti-patterns

- `git add -A` from any agent — sweeps another agent's mid-write files into
  the wrong commit. Always use scoped `git add <paths>`.
- Dispatching N agents and waiting for ALL before any commit — prompt-cache
  windows are short; commit incrementally within each agent's scope.
- Agents committing without retry — the first push collision permanently
  loses one agent's work.
- Merging scopes after dispatch — decide boundaries before launch; merging
  mid-flight creates undefined ownership.
- Skipping the `pull --rebase` step on push failure — creates diverged
  branches that require manual resolution.

## Cross-references

- `../autoresearch-critic-team/SKILL.md` — critic agents use this dispatch
  pattern.
- `../autoresearch-scicritic-team/SKILL.md` — sci-critic agents use this
  dispatch pattern.
- `../autoresearch-fixer-campaign/SKILL.md` — fixer agents use this dispatch
  pattern.
- `../autoresearch-checkpoint/SKILL.md` — checkpoint discipline that each
  dispatched agent relies on.
