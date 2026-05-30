---
name: autoresearch-fixer-campaign
description: Use when audits from the critic team have been filed and the project needs a parallel campaign of agents to fix the flagged findings. Each fixer agent consumes one audit file as its spec, applies fixes, verifies the specific complaint is resolved, then re-screens through the hillclimber. Never claims "fixed" without running the exact verification the critic specified.
---

# Skill — Parallel fixer campaign

## When to use

- After `autoresearch-critic-team` has produced `audits/G<N>_audit.md` files
  containing MAJOR or BROKEN verdicts.
- When the user says "fix the audit findings" or "apply the critic's patches."
- Before any new experiment sweep on code that has outstanding MAJOR or BROKEN
  findings — unresolved findings contaminate sweep results.

## The core rule: fix then verify the specific complaint

A fixer agent that applies a patch and then claims "fixed" without running the
exact verification step the critic specified is **not done**. The regression
class most likely to recur is "a fix was applied but the specific complaint
was not re-tested." The mandatory sequence is:

```
1. Read the audit finding (hypothesis ID + concrete fix instruction).
2. Apply the minimal patch that addresses ONLY that finding.
3. Run the verification step the critic specified
   (e.g., re-run the specific test case, re-run the module's smoke test,
   re-run the mechanism-verifying assertion).
4. Confirm the original complaint no longer reproduces.
5. Only then mark the finding as resolved.
```

If step 4 fails, iterate. Do NOT move to the next finding until the current
one is verified.

## Fixer agent setup

Each fixer agent receives:

1. **One audit file** as its spec — e.g., `audits/G3_audit.md`.
2. **A disjoint file scope** — the source files, test files, and docs that
   correspond to that audit's hypotheses. No other agent's files.
3. **Priority order** — fix BROKEN findings first, then MAJOR, then MINOR.
   Do not fix PASS findings (they are not findings).

Dispatch one fixer agent per audit group, in parallel, using the pattern in
`../autoresearch-multi-agent-dispatch/SKILL.md`.

## Fixer workflow per finding

```
for each finding in audit (BROKEN first, then MAJOR, then MINOR):
    1. Identify the module, line range, and complaint.
    2. Apply the minimal patch (prefer surgical edits over rewrites).
    3. Re-run the mechanism-verifying test or the critic's specified check.
    4. If the test now passes: append a resolution note to IMPROVEMENTS.md.
    5. If the test still fails: iterate; do not proceed to the next finding.
    6. Checkpoint commit (scoped git add) after each resolved finding.
```

## Audit consumption format

The fixer reads each finding block from the audit and extracts:

- **Hypothesis ID** — e.g., `H<NN>`.
- **Module** — the source file to patch.
- **Verdict** — BROKEN / MAJOR / MINOR (skip PASS).
- **Concrete fix instruction** — the patch or instruction the critic wrote.
- **Mechanism check** — the test or assertion that was missing or wrong; this
  becomes the verification target.

## Re-screen and re-hillclimb after fixes

After all BROKEN and MAJOR findings in a group are resolved:

1. **Re-run the full test suite** for the affected modules.
2. **Re-run any experiments** that produced results under the pre-fix code —
   results produced by BROKEN code are unreliable and must be marked stale.
3. **Re-run the hillclimber** (the project's selection / ranking pass) so
   that the champion config reflects fixed code.
4. If re-running reveals new findings, route them back into the audit ledger.

## IMPROVEMENTS.md discipline

Every resolved finding gets a dated entry in the idea's `IMPROVEMENTS.md`:

```markdown
## Fix log

### <YYYY-MM-DD> — H<NN>: <one-line summary of the fix>
**Finding:** <quote the critic's complaint>
**Patch:** <what was changed and why>
**Verified by:** <test or assertion run + outcome>
**Verdict before fix:** <BROKEN|MAJOR|MINOR>
**Verdict after fix:** <PASS|MINOR> (per re-audit)
```

## Output discipline

- Scoped `git add <source file> <test file> <IMPROVEMENTS.md>` per finding —
  never `-A`.
- Commit after each resolved finding, not in a batch at the end.
- Retry-wrapped commit + push (see
  `../autoresearch-multi-agent-dispatch/SKILL.md`).
- DO NOT touch audit files (`audits/G<N>_audit.md`) — those are the critic's
  output and are append-only. Fixer adds to IMPROVEMENTS.md, not to the audit.
- DO NOT touch another group's files.

## Return-to-coordinator format (≤ 200 words)

- Counts of findings resolved per verdict tier (BROKEN→PASS, MAJOR→PASS,
  MAJOR→MINOR, MINOR→PASS).
- Any finding that could NOT be resolved (with reason).
- Whether re-screen / re-hillclimb was run and what changed.
- Commit SHA of the final checkpoint.

## Anti-patterns

- Claiming "fixed" without running the specific verification step — this is
  the single most common regression source.
- Applying broad rewrites instead of surgical patches — broad rewrites
  introduce new bugs that the existing tests won't catch.
- Touching audit files to change a verdict — the audit is a historical
  record; update IMPROVEMENTS.md instead.
- Batching all fixes into one commit — if one fix introduces a regression,
  a granular commit history allows bisect.
- Moving to the next experiment sweep before all BROKEN and MAJOR findings
  are resolved and re-verified.

## Cross-references

- `../autoresearch-critic-team/SKILL.md` — produces the audit specs this
  skill consumes; also defines the post-fix verify-the-complaint discipline.
- `../autoresearch-multi-agent-dispatch/SKILL.md` — dispatch and retry-wrapped
  commit pattern for fixer agents.
- `../autoresearch-checkpoint/SKILL.md` — mandatory checkpoint after each
  resolved finding.
- `../autoresearch-scicritic-team/SKILL.md` — a NUMEROLOGY or UNFALSIFIABLE
  sci-verdict may redirect to redesigning the hypothesis rather than fixing
  the implementation.
- `../autoresearch-idea-scaffold/SKILL.md` — IMPROVEMENTS.md is the fix log
  within each idea's scaffold directory.
- `../autoresearch-winner-archive/SKILL.md` — after fixes, re-run the
  hillclimber; if a newly-fixed idea produces the new champion, archive it.
