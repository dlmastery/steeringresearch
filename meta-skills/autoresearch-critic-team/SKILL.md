---
name: autoresearch-critic-team
description: Use when the user does not trust existing test coverage — dispatch a parallel team of skeptical implementation-critic agents to audit hypothesis modules line-by-line. Each agent owns one thematic group, reads the design doc + source + tests, and emits a structured PASS/MINOR/MAJOR/BROKEN verdict to `audits/G<X>_audit.md`. The goal is to catch shape-only tests, math errors, citation mismatches, and code-vs-doc divergence that implementer agents missed.
---

# Skill — Parallel implementation-critic team

## When to use

- The user says something like "I don't believe enough validation happened."
- A large agent-written codebase is about to be used for external claims
  (paper, headline, dashboard).
- Before any Fixer campaign — the audit IS the fixer's spec.
- Roughly every 50+ hypothesis implementations or 1 000+ test lines.

## The doctrine — be skeptical, not confirmatory

Implementer agents trained to "make tests pass" often write **shape-only
tests** (e.g., assert `output.shape == expected_shape`) that do not verify
the mechanism. A passing test suite is **not** evidence the hypothesis is
implemented correctly. The critic's job is to find:

1. **Mechanism check** — does the code actually implement what the formal
   hypothesis claims? PASS only if a test asserts the mechanism, not just the
   shape or return type.
2. **Math correctness** — are constants right? off-by-one indexing? wrong
   axis or dimension? wrong normalisation? hardcoded approximation drift?
   RNG / seed state? log or sqrt of a possibly-negative value?
3. **Test rigor** — a shape-only test is MINOR at best. PASS requires at
   least one mechanism-verifying assertion.
4. **Citation alignment** — does the cited reference actually describe the
   technique, or is it a name-drop? Flag a citation without a full ID.
5. **Falsifier reachability** — can the relevant metric be read off a real
   run without manual reconstruction?
6. **Hidden bugs / cargo-cult** — invariant-constant biases? a learnable
   parameter registered as a buffer (or vice versa)? gradient-stop inside
   a forward pass? a mask buffer that is silently trainable?

## Verdict tiers (strict)

- **PASS** — mechanism implemented correctly AND at least one
  mechanism-verifying test exists.
- **MINOR** — mechanism correct, but tests are shape-only OR there is a
  cosmetic issue (wrong year in citation, typo in docstring).
- **MAJOR** — mechanism partially wrong OR a critical test is missing OR
  documented behaviour differs from the code.
- **BROKEN** — code contradicts the hypothesis (e.g., a zero-bias when the
  doc claims a non-trivial bias), OR the module does not run, OR a test
  asserts the wrong thing.

Be conservative with PASS. Most agent-written code lands MINOR or worse.

## Partition by thematic group

Dispatch N agents in parallel, one per hypothesis group (G1..GN, or whatever
the project's grouping convention is). Each agent:

- Reads all design docs, source modules, and test files for every hypothesis
  in its scope. NO skimming — line-by-line.
- Writes a single `audits/G<N>_audit.md` using the template below.

```markdown
# G<N> audit — <theme>
Reviewer: Critic-G<N> (expert critic)
Date: <YYYY-MM-DD>

## Summary
PASS: <list>   MINOR: <list>   MAJOR: <list>   BROKEN: <list>

## Per-hypothesis findings

### H<NN> — <name>
**Module:** `src/.../<module>`
**Verdict:** <PASS|MINOR|MAJOR|BROKEN>
**Mechanism check:** <1 paragraph, quote line numbers>
**Math correctness:** <findings or "verified">
**Test rigor:** <quote shape-only test assertions if any>
**Citation alignment:** <findings>
**Bugs / cargo-cult:** <specific line references>
**Concrete fix (if needed):** <patch or instruction for the Fixer>

## Group-level concerns
<patterns across the group>

## Recommended follow-ups (prioritised)
1. ...
```

## Output discipline

- Scoped `git add audits/G<N>_audit.md` — never `-A`.
- Retry-wrapped commit + push (see
  `../autoresearch-multi-agent-dispatch/SKILL.md`).
- DO NOT modify any source or test file — critics LOG findings; they do not
  patch. Patching is the Fixer's job.
- DO NOT touch experiment archives, dashboards, or another group's docs.

## Return-to-coordinator format (≤ 250 words)

- Counts per verdict tier.
- The 3 most damning findings (hypothesis ID + 1 sentence each).
- Commit SHA.

## Auditor-self-grading circularity disclosure

When the implementer, critic, and fixer agents share the same model family,
the critic's verdict — including any "PASS" — is an **internal QA pass, not
independent external review**. When the critic's output is referenced in an
externally-facing artefact (paper, README, dashboard banner), it MUST carry
the qualifier:

> Internal QA pass — critic verdict by same-family agent.
> Independent external review pending.
> See the project's external reviewer audit for the hostile-reviewer pass.

The critic SHOULD calibrate against a well-known third-party reference
implementation at least once per major release to measure the audit's
false-positive rate. Record the calibration result in
`audits/CALIBRATION_<date>.md` and reference it alongside audit-derived rates.

Until that calibration exists, audit-derived rates in externally-facing
artefacts MUST carry a descriptive-not-diagnostic disclosure.

## Post-fix verify-the-complaint discipline

After ANY fix landed in response to a critic finding, the next operator MUST
verify that the specific complaint no longer reproduces — NOT just that "a
fix shipped." For audit-finding fixes, re-run the audit's specific test case
on the patched code (per
`../autoresearch-fixer-campaign/SKILL.md`). Claiming "fixed" without
verification is the regression class most likely to re-ship.

## Anti-patterns

- Issuing a PASS because "the tests are green" without checking what the
  tests actually assert.
- Skimming hypothesis docs rather than reading line-by-line.
- Touching source or test files — critics are read-only.
- Using `git add -A` in a multi-agent dispatch context.
- Marking a complaint "fixed" without running the specific test case.

## Cross-references

- `../autoresearch-scicritic-team/SKILL.md` — the science-critic counterpart
  (does the hypothesis itself make scientific sense?); same circularity caveat
  applies.
- `../autoresearch-fixer-campaign/SKILL.md` — consumes audits as fixer specs;
  mandates verify-the-complaint after every fix.
- `../autoresearch-multi-agent-dispatch/SKILL.md` — retry-wrapped commit
  pattern used by each critic agent.
- `../autoresearch-checkpoint/SKILL.md` — checkpoint discipline that persists
  audit files to the remote.
