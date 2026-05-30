---
name: steering-multi-agent-dispatch
description: >
  Use when parallelizing steering research work across multiple agents. Assigns
  disjoint file scopes, enforces retry-wrapped commits, and caps agent returns at
  250 words structured. GPU/sweep work is always sequential (one 4090); docs,
  code, research, audit, and critique parallelize. This is the steering
  instantiation of the meta autoresearch-multi-agent-dispatch process.
---

# Skill — steering-multi-agent-dispatch

This is the steering instantiation of [meta-skills/autoresearch-multi-agent-dispatch/SKILL.md](../../meta-skills/autoresearch-multi-agent-dispatch/SKILL.md).

Read that file first for the general dispatch protocol. This file adds the
steering-specific file-scope assignments and the GPU sequencing constraint.

---

## Steering-specific notes

### The hard GPU constraint

There is exactly ONE RTX 4090 (16 GB VRAM). ALL forward passes, activation
caching, sweep runs, and ladder evaluations are SEQUENTIAL. Never dispatch two
agents that both need the GPU. The GPU queue is:

```
E1 -> E2 -> E3 -> [hill-climb winner] -> E4 -> E5 -> E6 -> E7 -> E8 -> ...
```

Parallelizable work (dispatch freely):
- Writing or editing any SKILL.md, README.md, IDEA.md, AUDIT.md, VERIFY.md
- Writing or editing implementation.py stubs and tests.py
- Updating IDEA_TABLE.md, EXPERIMENT_LEDGER.md, FINDINGS.md
- Generating or updating dashboard HTML
- Running critic-team or scicritic-team audits (no GPU needed)
- Editing corpus documents

NOT parallelizable (sequential on GPU):
- Any experiment.py run
- Any activation caching pass
- Any hill-climb trial
- Any eval-bundle sweep

### Standard file-scope assignments

| Agent role | Files it may write | Files it must NOT touch |
|------------|-------------------|------------------------|
| Ledger agent | EXPERIMENT_LEDGER.md, FINDINGS.md, IDEA_TABLE.md | anything else |
| Dashboard agent | dashboard/, docs/dashboard/ | anything else |
| Per-hypothesis agent (idea N) | ideas/<NN>/*, only that idea dir | other idea dirs, src/ |
| Process agent | skills/<name>/SKILL.md, meta-skills/<name>/SKILL.md, CLAUDE.md | src/, ideas/, dashboard/ |
| Implementation agent | src/steering/*.py, conftest.py, pyproject.toml | dashboard/, ideas/, skills/ |
| GPU/experiment agent | autoresearch_results/*.json, autoresearch_results/*.jsonl | everything else |
| Critic agent | audits/<name>.md | the files being critiqued (read-only) |

Two agents with overlapping file scopes MUST NOT run simultaneously.

### Commit protocol for multi-agent

Each agent:
1. Stages ONLY its assigned files (`git add <specific files>`, never `-A`)
2. Attempts commit with retry loop (5 attempts max):
   - On failure: `git pull --rebase origin main`, then retry
3. Returns a structured summary <= 250 words:
   ```
   Agent: <role>
   Files written: <list>
   Summary: <what was done>
   Blockers: <any issues>
   Next: <handoff if applicable>
   ```

### Dispatch template

```
Dispatch batch: <batch ID>
Parallelizable tasks (launch simultaneously):
  - Agent A: [ledger agent] update EXPERIMENT_LEDGER.md row for exp-NNN
  - Agent B: [dashboard agent] regenerate dashboard/index.html
  - Agent C: [per-hypothesis agent, idea 10] write AUDIT.md for H10
  - Agent D: [critic agent] review H20 implementation.py

Sequential task (wait for batch to complete, then GPU agent):
  - Agent E: [GPU/experiment agent] run E1 Rung-1 SMOKE
```

### Circularity disclosure (same-model-family)

When a critic agent reviews code written by another dispatch of the same model
family, the review carries:
> "Internal QA pass — independent external review pending."

This must appear in every audit, critic, and scicritic output. It is not a
reflection of the quality of the review; it is an honest disclosure of the
circularity (CLAUDE.md Section 14).

### When to dispatch vs not

Dispatch when:
- >= 2 independent tasks exist that do NOT share file scope
- None of the tasks needs the GPU simultaneously

Do NOT dispatch when:
- Only one task exists
- Tasks share file scope (merge into one agent's work)
- The overhead of dispatch coordination exceeds the parallelism gain
  (rule of thumb: dispatch only when each task >= 10 minutes)

---

## Cross-references

- Meta-process: `../../meta-skills/autoresearch-multi-agent-dispatch/SKILL.md`
- Checkpoint discipline: `../steering-checkpoint/SKILL.md`
- Critic team: `../steering-critic-team/SKILL.md`
- Scicritic team: `../steering-scicritic-team/SKILL.md`
- Agent-team discipline: CLAUDE.md Section 14
- GPU constraint: CLAUDE.md Section 2 (hardware budget)
