---
name: autoresearch-scicritic-team
description: Use to critique the SCIENTIFIC MERIT of each hypothesis (not the code). Different from the implementation critic — the sci-critic asks whether the hypothesis is novel, derivative, or numerology, independent of whether the implementation is correct. Output is an "Addendum: Research-Scientist Critique" section appended directly into each design doc.
---

# Skill — Parallel research-scientist critic team

## When to use

- After a large body of hypothesis docs has been written (≥ 30 docs).
- Before publication or external claims, to filter out numerology
  rediscoveries and unfalsifiable claims.
- Whenever the user asks "is this idea actually novel, or did someone already
  publish this?"

## How it differs from `autoresearch-critic-team`

| dimension         | impl-critic                                   | sci-critic                                         |
|-------------------|-----------------------------------------------|----------------------------------------------------|
| audits            | the CODE — does it implement the doc?         | the IDEA — is the doc itself defensible?           |
| output            | `audits/G<X>_audit.md`                        | addendum appended into each design doc             |
| verdict           | PASS / MINOR / MAJOR / BROKEN                 | NOVEL / DERIVATIVE / NUMEROLOGY / FALSIFIED / UNFALSIFIABLE |
| can both apply?   | yes — they're orthogonal                      | yes — code can be impl-PASS while idea is NUMEROLOGY |

## Sci-verdict tiers (strict)

- **NOVEL+TESTABLE** — never previously studied; a falsifier exists.
- **DERIVATIVE+TESTABLE** — rediscovery of a known technique under a new
  name; a falsifier exists. (Usually the strongest defensible rating.)
- **NUMEROLOGY** — the chosen constant or structural choice is decorative;
  any constant in a similar range would produce the same effect. Suggest the
  specific control ablation that would prove or disprove this.
- **FALSIFIED** — already refuted by an existing experiment in the project's
  own data.
- **UNFALSIFIABLE** — too many simultaneously-changing variables; no outcome
  can be attributed to the claimed mechanism.
- **INFRASTRUCTURE** — a methodology improvement (e.g., multi-seed error
  bars), not a testable hypothesis.
- **UNTESTED_ON_RIGHT_DATASET** — the pre-registered falsifier specifies a
  dataset or evaluation regime that was NOT in the sweep; verdict deferred
  until the pre-registered evaluation is available (see below).

## Per-hypothesis addendum template

The sci-critic appends THIS exact section to the END of each hypothesis
design doc:

```markdown

---

## Addendum: Research-Scientist Critique (<YYYY-MM-DD>)

*Reviewer: SciCritic-G<X> (elite research-scientist critic).
Critiquing the IDEA, not the implementation
(impl audit at `audits/G<X>_audit.md`).*

### Prior plausibility (LOW / MED / HIGH + why)
<3 sentences specific to this hypothesis>

### Mechanism scrutiny — is the "because" clause real or post-hoc?
<quote the doc's mechanism claim and critique it>

### Confounds — what else could explain a positive (or negative) result?
<≥ 2 alternative explanations>

### Control-ablation check — does the specific constant / choice matter?
<Would a simpler or randomly-chosen alternative produce the same effect?
Specify the cheapest ablation that answers this question.>

### Literature: precedent or rediscovery?
<Has this appeared under another name? Cite a real reference with full ID.>

### Expected effect size — sceptical a-priori re-prediction
<Your 90 % CI, not the doc's optimistic claim>

### Minimum-distinguishing experiment
<Cheapest experiment that distinguishes this from a numerology placebo>

### Verdict
<NOVEL+TESTABLE | DERIVATIVE+TESTABLE | NUMEROLOGY | UNFALSIFIABLE |
FALSIFIED | UNTESTED_ON_RIGHT_DATASET | INFRASTRUCTURE + 1 sentence>
```

## Doctrine — disagree with the author

- Quote claims and challenge them. Citations without full IDs violate good
  scholarly practice — flag them.
- No hedging on numerology — call it when any constant in a similar range
  would produce the same result.
- Cite real references for "this is already known" claims. Use the format:
  `Author YEAR VENUE 'Title' (full ID) — relevance`.
- When raising a NUMEROLOGY verdict, always name the specific control ablation
  (e.g., "replace the chosen constant with 1.5, 1.7, and 2.0; if all three
  give equivalent results, the constant is decorative").

## UNTESTED_ON_RIGHT_DATASET — the dataset-aware verdict

A hypothesis whose pre-registered falsifier specifies an evaluation regime
NOT in the current sweep cannot earn a NUMEROLOGY or FALSIFIED verdict from
that sweep. Use UNTESTED_ON_RIGHT_DATASET instead.

Example reasoning: if the falsifier pre-registers "wrap-aware synthetic
dataset" as the required evaluation, but the sweep ran on a standard
upright benchmark, the correct verdict is UNTESTED_ON_RIGHT_DATASET — NOT
NUMEROLOGY. Concluding failure from the wrong dataset is the error this tier
was created to prevent.

When in doubt, default to UNTESTED_ON_RIGHT_DATASET rather than NUMEROLOGY.
The latter is a strong claim; the former is the honest verdict.

## Output discipline

- Scoped `git add <path/to/hypothesis/docs>` — never `-A`.
- ONE commit per group (not per hypothesis); the addendum is text-only.
- Retry-wrapped commit + push (see
  `../autoresearch-multi-agent-dispatch/SKILL.md`).
- DO NOT modify source files, tests, audit files, dashboards, experiment
  archives, or other groups' docs.

## Return-to-coordinator format (≤ 200 words)

- Counts per verdict tier.
- The single most damning critique (with hypothesis ID).
- Commit SHA.

## Auditor-self-grading circularity disclosure

When the sci-critic agents share a model family with the implementer,
impl-critic, and fixer agents, the sci-verdict — NOVEL+TESTABLE in particular
— is an **internal sci-QA verdict, not independent external review**. When
referenced externally:

- The tier counts are a snapshot at a specific commit, not a permanent
  classification.
- An external reviewer might revise DERIVATIVE+TESTABLE upward to
  NOVEL+TESTABLE (upon deeper literature search) OR downward (upon discovery
  of overlooked prior art).
- External reviewer findings override the internal sci-verdict; that override
  must be applied within the same commit that processes the external audit.

## Anti-patterns

- Issuing NUMEROLOGY without specifying the control ablation that would
  distinguish it — that is an unfalsifiable critique of an unfalsifiable claim.
- Issuing FALSIFIED when the sweep ran on a dataset other than the one the
  falsifier pre-registered.
- Hedging on NUMEROLOGY with language like "may be decorative" — be direct or
  say DERIVATIVE+TESTABLE.
- Modifying source or test files — sci-critics are read-only on non-doc files.

## Cross-references

- `../autoresearch-critic-team/SKILL.md` — the code-side counterpart; same
  circularity caveat applies.
- `../autoresearch-fixer-campaign/SKILL.md` — a NUMEROLOGY or UNFALSIFIABLE
  verdict may feed a fixer campaign to redesign the hypothesis.
- `../autoresearch-multi-agent-dispatch/SKILL.md` — retry-wrapped commit
  pattern used by each sci-critic agent.
- `../autoresearch-idea-scaffold/SKILL.md` — the falsifier contract that
  sci-critics check for pre-registration completeness.
