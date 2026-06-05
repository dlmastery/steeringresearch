---
name: steering-dashboard
description: >
  Use when generating or updating the master dashboard, per-hypothesis
  sub-dashboards, or per-experiment pages. Enforces the CLAUDE.md Section 11
  hard rules: self-contained HTML, PNG plots, no CDN, markdown rendered,
  composite fingerprint in footer. This is the steering instantiation of the
  meta autoresearch-dashboard process.
---

# Skill — steering-dashboard

This is the steering instantiation of [meta-skills/autoresearch-dashboard/SKILL.md](../../meta-skills/autoresearch-dashboard/SKILL.md).

Read that file first for the general dashboard architecture. This file adds
the steering-specific panels required by CLAUDE.md Section 11.

---

## Steering-specific notes

### Panel 0 — Newcomer grounding (steering instantiation)

The master dashboard must open with the newcomer grounding block defined in
`meta-skills/autoresearch-dashboard/SKILL.md`. For the steering program,
the required content is:

**Research goal (steering-specific):**

> "Goal: discover the activation steering configuration (layer × alpha ×
> direction × operation × span) that maximizes steering behavior efficacy on
> Gemma models while keeping capability drop < 2 pp MMLU, perplexity increase
> bounded (< 20% PPL uplift), safety compliance rate at 0%, and over-refusal
> rate near baseline — the best simultaneous trade-off known for small-Gemma
> steering. A 'winner' Pareto-dominates all prior configurations on the
> fingerprinted composite metric."

**Domain primitives (steering-specific):**
One paragraph explaining what activation steering is, what the residual stream
is, what an injection layer is, and why the five-axis trade-off is non-trivial.
No assumed ML background — a cognitive scientist or linguist must understand
what is being varied and why it matters.

**Methodology tabs (steering-specific):**
The methodology accordion must include: the 12-axis taxonomy, the composite
formula with λ values, the five-rung ladder with gate thresholds, and the
screening/evaluation distinction. All tabs open by default.

### Required panels (master dashboard)

The master dashboard at `dashboard/index.html` must include ALL of the following:

**1. Runs table**
- Sortable/filterable; default sort = composite descending
- Global champion row highlighted (gold or green border)
- Every numeric cell carries `n=X` and a `SCREENING`/`EVALUATION` chip
- Sub-links from every row to its per-hypothesis and per-experiment pages

**2. Five-axis radar / parallel-coordinates panel**
- One radar per method showing all five axes: behavior efficacy, capability
  retention (MMLU delta), coherence (PPL), safety integrity (CR_jailbreak),
  selectivity (over_refusal gap)
- The multi-objective view makes it impossible to "win" by sacrificing one axis
- Tooltip shows the axis definition and the composite weight lambda for that axis

**3. Pareto panel (three scatter plots)**
- behavior_efficacy vs MMLU_drop
- behavior_efficacy vs PPL_delta
- behavior_efficacy vs CR_jailbreak
- Prior methods as star markers; dominated methods labeled
- At least one dominated row must exist per Pareto plot (proves the harness discriminates)

**4. Ladder board**
- Per method: rung reached, gate cleared/failed, failure_reason
- Visual: colored rung cells (green = passed, red = failed, grey = not attempted)

**5. Stack/compete matrix**
- Rendered from the CLAUDE.md Section 9 decision rules applied to current data
- Orthogonal axes -> STACK (green); same site + direction -> COMPETE (red)

**6. Geometry leading-indicators panel**
- Per method: delta_norm, eff_rank_drop, norm_budget, part_ratio
- The off-manifold view; expected to correlate with coherence failures

**7. Orientation block and footer**
- 4-bullet "how to read this" orientation
- COMPOSITE_FORMULA SHA-256 fingerprint
- Git commit SHA
- "Internal QA pass — independent external review pending" qualifier

### Sub-dashboard (per hypothesis, `ideas/<NN>/dashboard/index.html`)

Required:
- Best-config callout (layer, alpha, source, operation, span)
- Per-axis coordinate-descent Pareto small-multiples (one per axis in the cube)
- Seed-stability bar chart (composite across n seeds)
- Cells table linking to per-experiment pages
- Hypothesis statement, falsifier, predicted delta, current verdict
- Back-link to master dashboard

### Per-experiment page (`docs/dashboard/experiments/expNNN.html`)

Required:
- Full 7-step reasoning entry (pre-run: Diagnose, Cite, Hypothesize, Predict;
  post-run: Analyse, Checkpoint) rendered from markdown
- Alpha/layer sweep curves (PPL and behavior efficacy vs parameter)
- Generation samples: steered vs unsteered side-by-side (5 prompts minimum)
- Geometry probes: delta_norm, eff_rank_drop, norm_budget curves vs alpha
- All five axis metrics with 95% CIs (n chips on every number)

### Hard rules (verbatim from CLAUDE.md Section 11, extended)

- Self-contained HTML: no CDN, no external JS frameworks; one inline `<script>`
  for sort/filter
- PNG not SVG for plots (reproducible screenshots)
- Markdown rendered: Playwright asserts no literal `##` / `**` / `|---|` leaked
- Absolute GitHub-blob links HEAD-tested (no 404s)
- Small-multiples over dense charts (one concept per panel)
- No self-graded ACCEPT banner without the "Internal QA pass — external review
  pending" qualifier
- No emoji unless explicitly requested
- **Newcomer grounding block** (Panel 0) present before any table or chart;
  includes steering goal, domain primitives paragraph, and open methodology tabs
- **Per-tab orientation paragraph** on every named section (geometry, ladder,
  hypotheses, runs, Pareto, stack/compete)
- **Per-table "What is this" block** on every table: purpose, per-column glossary
  (incl. axis abbreviations expanded), what-to-pay-attention-to, expected values
- **Winner color-coding with explicit legend**: KEEP/champion = green,
  NEAR-MISS = yellow, DISCARD = muted, BROKEN = grey; legend present
- **"In Plain English" box** present at the top of EVERY per-hypothesis and
  per-experiment page, before any table or chart; defines all technical terms
  in-page; links to the project GLOSSARY (L17)
- **Project GLOSSARY** at `hypotheses/GLOSSARY.md` (or `docs/GLOSSARY.md`);
  linked from master dashboard "how to read this" block and every per-page
  plain-English box (L17)

---

## Plain-English accessibility mandate (L17)

**L17 — Every hypothesis/experiment page must open with a plain-language
box, and the project must maintain a GLOSSARY.**

Technical documentation that assumes ML or mechanistic-interpretability
background is incomplete. A cognitive scientist, a policy researcher, or a
domain expert should be able to read any hypothesis or experiment page and
understand: what is being tested, in normal words; what every term means;
why it matters; and what a result would mean in plain English.

### Per-page "In Plain English" box (required on every hypothesis and
experiment page)

Every `ideas/<NN>/dashboard/index.html` and every
`docs/dashboard/experiments/expNNN.html` must open with an "In Plain English"
box (visually distinct — light background, slightly larger font) containing:

1. **What this tests** (1–2 sentences in ordinary English, no jargon).
   Example: "We test whether nudging the model's internal representation
   toward an 'ocean' concept makes it write ocean-related text more often."

2. **Key terms defined in-page** (not by reference to another file).
   Every technical term used in the page — "residual stream", "steering
   vector", "alpha", "injection layer", "composite score", etc. — must be
   defined in this box or in a collapsible inline glossary panel immediately
   below it. Definition length: 1–2 sentences each.

3. **Why it matters** (1 sentence). What would change about the project's
   conclusions if this hypothesis is true vs false?

4. **What a result would mean** (1–2 sentences). Concretely: what does
   KEEP mean, and what does DISCARD mean, for this specific hypothesis?

The box must appear BEFORE any tables, charts, or technical metrics on the
page. It is the first thing a reader sees.

### Project GLOSSARY (`hypotheses/GLOSSARY.md`)

Maintain a single project-wide glossary at `hypotheses/GLOSSARY.md`
(or `docs/GLOSSARY.md`). It must define every term that appears in any
hypothesis or experiment page. Required entries for this project:

- activation steering, residual stream, injection layer, steering vector,
  alpha (steering coefficient), composite score (one-sentence description
  of the formula and what it measures), behavior efficacy, capability
  retention, coherence, safety integrity, selectivity, screening,
  evaluation, KEEP, DISCARD, NEAR-MISS, EXTERNAL-READY, AxBench,
  JailbreakBench, off-shell displacement, effective rank, norm budget.

The glossary is linked from:
- The master dashboard "how to read this" block.
- Every per-page "In Plain English" box (one link: "See full glossary →").
- The project README.

The glossary is updated whenever a new technical term is introduced in any
hypothesis, experiment page, or dashboard section.

### Verification

Playwright accessibility check on any dashboard page:

```javascript
// In the rendering audit: every experiment page must contain a
// "plain-english" block before the first table or chart.
const box = page.locator('[data-role="plain-english"]');
await expect(box).toBeVisible();
// The box must appear before the first <table> or <canvas>.
const boxY = (await box.boundingBox()).y;
const tableY = (await page.locator('table').first().boundingBox()).y;
expect(boxY).toBeLessThan(tableY);
```

---

## Regeneration trigger (when to update)

Regenerate master dashboard on every checkpoint (CLAUDE.md Section 13):
- After any EXPERIMENT_LEDGER.md row is added
- After any verdict changes
- After any finding graduates to FINDINGS.md
- Before every commit (dashboard is always in sync with ledger)

---

## Cross-references

- Meta-process: `../../meta-skills/autoresearch-dashboard/SKILL.md`
- Dashboard comprehension: `../../meta-skills/autoresearch-dashboard-comprehension/SKILL.md`
- Per-experiment page: `../../meta-skills/autoresearch-per-experiment-page/SKILL.md`
- Typography and rendering: `../../meta-skills/autoresearch-typography-and-rendering/SKILL.md`
- Link discipline: `../../meta-skills/autoresearch-link-discipline/SKILL.md`
- Findings and ledger discipline: `../../meta-skills/autoresearch-findings-ledger/SKILL.md`
- Composite formula: `../../src/steering/eval.py:COMPOSITE_FORMULA`
- Ledger: `../../EXPERIMENT_LEDGER.md`
- Master output: `../../dashboard/index.html`
