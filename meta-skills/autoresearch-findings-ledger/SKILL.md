---
name: autoresearch-findings-ledger
description: >
  Use when writing or auditing FINDINGS.md, EXPERIMENT_LEDGER.md, or any
  equivalent external-facing summary of results. Enforces three mandates:
  (1) self-contained FINDINGS readable with zero cross-referencing — every
  identifier defined inline; (2) interpretable ledger with a "how to read
  this" section and a campaign-arc narrative; (3) a shared glossary preamble
  so no reader must open another file to understand a sentence.
metadata:
  companion_skills:
    - autoresearch-paper-rigor
    - autoresearch-dashboard
    - autoresearch-doc-organization
  added: 2026-06-01
---

# Skill — Self-contained Findings and Interpretable Ledger

## When to use

- Before publishing or committing any update to `FINDINGS.md` or
  `EXPERIMENT_LEDGER.md` (or their project-specific equivalents).
- After any new verdict is written — verify the updated document still
  satisfies all three mandates below.
- When a reviewer or collaborator reports confusion about an identifier,
  abbreviation, or result reference — the document has failed the
  self-contained mandate and must be corrected.
- When the ledger grows beyond ~10 rows — the campaign-arc narrative must
  be updated to group the raw rows into a readable story.

---

## Mandate A — Self-contained FINDINGS

### The preamble contract

`FINDINGS.md` (or the project's equivalent) must open with a **preamble
section** containing ALL of the following, in this order, before the first
finding is stated:

**A1. How to read this document (3–5 sentences)**
Explain the document's purpose, what a "finding" is in this project (a
rigor-gated empirical claim, not an observation), how findings are numbered,
and the difference between a finding that is EXTERNAL-READY vs one marked
INTERNAL.

Example preamble opener:

> "This document records rigor-gated empirical findings from the [project name]
> autoresearch program. A 'finding' here means a claim that has cleared the
> statistical-rigor floor (see Methodology), is tied to a specific experiment
> tag in EXPERIMENT_LEDGER.md, and carries a screening/evaluation tier label.
> Findings are numbered F-1, F-2, … in order of confirmation, not importance.
> EXTERNAL-READY means the four-part statistical contract and the ordinal gate
> have been satisfied. INTERNAL means the result is promising but has not yet
> cleared those gates."

**A2. Inline glossary — every identifier defined on first use**

A `FINDINGS.md` that forces the reader to open another file to understand a
sentence has failed the self-contained mandate. Required in the glossary:

| Category | Must define inline |
|---|---|
| Hypothesis IDs | Every Exx / Nxx / hypothesis tag used, e.g. "E4 (Curvature-aware injection, ideas/04/)" |
| Result IDs | Every Sxx / finding tag, e.g. "S-1 (first confirmed positive result)" |
| Metric abbreviations | composite, PPL, MMLU-delta, CR_jailbreak, and every axis abbreviation used |
| Verdict tiers | KEEP, DISCARD, NEAR-MISS, NOVEL+TESTABLE, DERIVATIVE+TESTABLE, NUMEROLOGY, FALSIFIED, UNTESTED_ON_RIGHT_DATASET |
| Screening vs evaluation | n≤3 = SCREENING; n≥7 = EVALUATION; what each means for external claims |
| EXTERNAL-READY | the four-part contract + ordinal gate (one sentence each) |

The glossary need not be a formal section — it can be woven into the preamble
prose — but every term used in the document must be defined before first use.
**Never** use an identifier like "E4 lifts behavior by +0.12" without first
stating "E4 is the Curvature-aware injection hypothesis (ideas/04/)."

**A3. Summary table**

Immediately after the preamble, a summary table:

| ID | Hypothesis | Verdict | Composite Δ | n | Tier | Experiment tags |
|---|---|---|---|---|---|---|
| F-1 | [plain English, ≤12 words] | KEEP | +0.08 | 7 | EVALUATION | exp042, exp043 |
| F-2 | … | … | … | … | … | … |

The summary table is the entry point. A reader who reads only the summary
table must be able to answer: what did the program find, and how confident
is it?

**A4. Plainly-stated "strongest result"**

One sentence, in plain English, naming the strongest result. No hedge
language from the preamble. Example:

> "The strongest confirmed result is F-1 (method X, +0.08 composite,
> n=7 EVALUATION, p=0.012 Wilcoxon): it Pareto-dominates all prior
> baselines on all five axes simultaneously."

If no EXTERNAL-READY result exists yet, state that plainly:

> "No EXTERNAL-READY result has been confirmed as of this writing
> (all results to date are INTERNAL or SCREENING tier)."

### The zero-cross-reference rule (binding)

Every sentence in FINDINGS.md must be independently interpretable. A reviewer
reading only FINDINGS.md must never need to open:
- IDEA_TABLE.md to understand a hypothesis reference
- EXPERIMENT_LEDGER.md to understand an experiment tag
- eval.py to understand the composite
- Any other file to understand any abbreviation

When you write a sentence like "E4 at layer 6 achieves behavior efficacy 0.62
with a MMLU-delta of −0.8 pp (composite 0.5414)", every term in that sentence
must have been defined in the preamble. If it wasn't — add it.

**How to verify:** read FINDINGS.md in isolation, pretending you have no other
context. Highlight every term that requires background knowledge. If you
highlight anything, the preamble is incomplete.

---

## Mandate B — Interpretable Experiment Ledger

### "How to read this ledger" section (required, at the top)

`EXPERIMENT_LEDGER.md` must begin with a **"How to read this ledger"** section
(before the first data row) containing:

**B1. What one experiment is (2–3 sentences)**
Define an experiment in plain English: a single-axis perturbation of the
champion config, run at a specific rung, producing a verdict. Explain that
experiments are append-only: a new row is added for each run, never edited.

**B2. Column definitions with "what good looks like"**

For every column in the ledger, define:
- The column name in plain English
- Units (if numeric)
- What a "good" value looks like vs what should be investigated
- Whether higher or lower is better

Required columns and their definitions (adapt column names to the project):

| Column | Plain English | Good value |
|---|---|---|
| tag | unique experiment identifier matching experiment_log.jsonl | — |
| method | the algorithm or config variant tested | — |
| rung | the benchmark ladder level (0=UNIT … 4=FULL) | higher = more evidence |
| composite | multi-axis score (formula fingerprinted) | higher = better; above baseline = candidate |
| behavior_efficacy | primary objective score (0–1 or 0–100) | higher = better |
| capability_delta | change in capability vs unsteered (pp) | near 0 = good; large drop = fail |
| coherence_delta | change in perplexity vs unsteered | near 0 = good; large increase = fail |
| safety_leak | safety constraint violation rate (0–1) | must be 0 = non-negotiable |
| selectivity | gap between intended and unintended behavior change | higher = better |
| verdict | KEEP / NEAR-MISS / DISCARD / BROKEN | — |
| failure_reason | why it failed (if DISCARD/BROKEN) | — |
| n | number of seeds run | ≥7 for EVALUATION claims |
| tier | SCREENING (n≤3) or EVALUATION (n≥7) | EVALUATION for any external claim |

**B3. Verdict-tier meanings (explicit, not assumed)**

The ledger must include a one-sentence definition of EACH verdict:

| Verdict | Meaning |
|---|---|
| KEEP | Method cleared the rung gate; composite improved at matched coherence |
| NEAR-MISS | Composite improved but ≤ 2 pp above baseline, or one axis narrowly missed its gate |
| DISCARD | Method failed at least one rung gate, or composite did not improve |
| BROKEN | Implementation error found; result is invalid regardless of metric value |

Additionally: **SCREENING and DISCARD are orthogonal.** A SCREENING result that
fails is a "SCREENING/DISCARD" row — it tells us the default config did not work,
not that the hypothesis is falsified. A SCREENING result that passes is
"SCREENING/KEEP" — promising but unconfirmed. Never conflate tier (how many seeds)
with verdict (what the result showed).

**B4. Campaign-arc narrative**

After the column glossary and verdict definitions, include a **campaign-arc
narrative** that groups the raw rows into a story. This is a brief prose
section (1–3 paragraphs) that explains:
- What the program has tried so far, grouped by phase or axis
- What the current champion is and how it was reached
- What the open questions are going into the next phase

The campaign-arc narrative must be updated every time a new phase begins.
A ledger that is a raw row dump with no arc narrative is interpretable only
by the original researcher — it fails the cold-reader test.

Example structure:

> **Phase 1 (experiments exp001–exp012): baseline and direction survey.**
> Established the plumbing baseline (exp001, KEEP), tested three direction
> extraction methods (exp002–exp004, all NEAR-MISS or DISCARD). Best result:
> exp004 with composite 0.41.
>
> **Phase 2 (exp013–exp028): layer and alpha sweep.**
> Moved injection from the default layer to layers 4–18 in coordinate
> descent. exp019 (layer 9, alpha 20) is the current champion at composite
> 0.58. Layer 6 and layer 12 were consistently weaker.
>
> **Open question:** the coherence penalty at layers > 14 suggests a
> manifold-fragility issue. Phase 3 will test norm-clamped injection.

---

## Mandate C — Provenance tracing for tested hypotheses

Even when the codebase uses ONE shared harness (not per-hypothesis code files),
every hypothesis in the project's idea registry that has been tested must have a
**provenance artifact** that lets a reader trace exactly what was done for it.
See `../autoresearch-idea-scaffold/SKILL.md` and
`../autoresearch-experiment-archive/SKILL.md` for the per-hypothesis and
per-experiment archive contracts. This skill's additional requirements:

**C1. Per-hypothesis provenance summary (in the idea's README or a dedicated
`PROVENANCE.md` within `ideas/<NN>/`):**

- The exact experiment numbers / tags that tested this hypothesis
- The precise command (or script + arguments) that produced each run
- The paths to the result JSON / artifact directories
- Links to the relevant rows in `experiment_log.jsonl` and
  `reasoning_annotations.json`
- A 2–3 sentence result interpretation: what the numbers mean for this
  hypothesis, not just the raw scores

**C2. Untested hypotheses must declare what they still need**

Any hypothesis listed in the idea registry that has NOT yet been tested must
include, in its README or IDEA.md, a section:

> "**What new harness code is needed to test this hypothesis:**
> [description of what the shared harness cannot yet do for this idea]"

This prevents a reader from mistaking "no experiments found for this idea" with
"this idea does not require new code."

**C3. Shared-harness architecture disclosure (in README and dashboard)**

When the project uses a single shared harness for most experiments (rather than
per-hypothesis code files), the project README and the master dashboard must
explicitly state:

> "This project uses a shared experimental harness (`src/`). The small number
> of source files does not indicate incomplete work — individual hypotheses are
> differentiated by configuration files and reasoning annotations, not by
> separate codebases. See `ideas/<NN>/` for the per-hypothesis differentiation."

A reader who sees "few files in src/" must not be left to infer this; it must
be stated.

---

## Hard rules

1. **FINDINGS.md preamble is required.** No finding may appear before the
   preamble (A1–A4). A findings document that begins with "F-1: …" without
   any preamble fails the mandate on the first line.
2. **Every identifier defined before use.** "E4", "S-1", "composite",
   "MMLU-delta" — all must be defined in the preamble or on first use.
3. **Summary table present.** Even a one-row table is required.
4. **Strongest result stated plainly.** Even if the answer is "none yet."
5. **Ledger "How to read" section required** before the first data row.
6. **Column glossary complete.** One entry per column, with good-value guide.
7. **SCREENING vs DISCARD orthogonality stated.** Not assumed.
8. **Campaign-arc narrative updated each phase.** Not a static section written
   once and left stale as new experiments accumulate.
9. **Provenance summary per tested hypothesis.** Not just in experiment_log.jsonl
   — in a human-readable location linked from the hypothesis's own README.
10. **Shared-harness disclosure in README and dashboard** when applicable.

---

## Anti-patterns

| Anti-pattern | Why it fails |
|---|---|
| Opening FINDINGS.md with "F-1: [result]" | A newcomer cannot interpret any finding without the preamble |
| Using "E4" without definition | Forces cross-referencing to IDEA_TABLE.md — breaks zero-cross-reference rule |
| "Composite 0.5414" without defining composite | The metric is opaque without the formula or at minimum a one-line description |
| EXPERIMENT_LEDGER.md as a raw row dump | Unreadable to anyone but the original researcher; fails the cold-reader test |
| Conflating SCREENING with DISCARD | "n=1, DISCARD" and "n=7, DISCARD" are very different conclusions; the distinction must be explicit |
| Campaign-arc narrative written once and never updated | After 20 experiments, a one-phase narrative is stale and misleading |
| "See EXPERIMENT_LEDGER.md for details" in FINDINGS.md | Forces cross-referencing; findings must stand alone |
| Hypothesis listed in IDEA_TABLE but no provenance artifact | A reader cannot investigate what was done for the hypothesis |
| README says nothing about shared harness | "Few files in src/" is mistaken for incomplete work |

---

## Cross-references

- `../autoresearch-paper-rigor/SKILL.md` — the statistical-rigor floor that
  governs which claims qualify as EXTERNAL-READY findings.
- `../autoresearch-idea-scaffold/SKILL.md` — the per-hypothesis sub-project
  layout that hosts the provenance artifacts (Mandate C).
- `../autoresearch-experiment-archive/SKILL.md` — the per-experiment archive
  that is the provenance anchor for every row in the ledger.
- `../autoresearch-dashboard/SKILL.md` — the master dashboard whose findings
  panels must comply with the same self-contained mandate.
- `../autoresearch-doc-organization/SKILL.md` — the repo-root discipline
  that determines where FINDINGS.md and EXPERIMENT_LEDGER.md live.
- `../autoresearch-meta/SKILL.md` — the process spine; the findings ledger
  is one of the canonical state files (§12) maintained by the program.
