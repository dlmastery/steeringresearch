---
name: autoresearch-per-experiment-page
description: >
  Use to emit one independent, self-contained HTML page per experiment row.
  Each page renders the full 7-step reasoning entry from the project's
  reasoning_annotations.json; sweep curves (parameter vs. objective); side-
  by-side artifact comparisons (e.g. outputs before and after the
  intervention); all objective-axis metrics with confidence intervals; and
  back-links to its hypothesis sub-dashboard and to the master dashboard.
  Pages mirror to docs/dashboard/experiments/ for GitHub Pages. No row-click
  modals — every row navigates to a page.
metadata:
  companion_skills:
    - autoresearch-dashboard
    - autoresearch-dashboard-comprehension
    - autoresearch-typography-and-rendering
    - autoresearch-link-discipline
  added: 2026-05-30
---

# Skill — Per-experiment dashboard pages

## When to use

- After the runs table in the master dashboard exceeds ~20 rows — a flat
  table becomes unreadable; per-experiment pages take over detail.
- When a user or reviewer wants to drill into one experiment and read the
  hypothesis design, reasoning chain, sweep curves, and all metrics in one
  place without opening multiple tabs.
- When adding a new experiment to the campaign — generate its page as part
  of the Checkpoint step, not retrospectively.
- After any change to the reasoning_annotations.json or the per-run
  output artifacts — regenerate the affected page immediately.

## The two halves of this deliverable

### Half A — Master and sub-dashboards become navigation hubs

Every row in the master and sub-dashboard tables navigates to a per-
experiment page. This means:

- `<tr class="row-link" onclick="location.href='experiments/<id>.html'">` —
  cursor pointer, no modal.
- The tag / method cell may additionally wrap in `<a href="…">` with
  `event.stopPropagation()` so the cell link also navigates independently.
- Group-sectioned tables (one `<section>` per hypothesis group) replace
  any single flat 50+ row table in the master.

### Half B — The per-experiment page (10 sections, in order)

File: `docs/dashboard/experiments/<experiment_id>.html`

Also built to `dashboard/experiments/<experiment_id>.html` and mirrored.

Filename collision rule: if the same method tag is tested under multiple
conditions or datasets, disambiguate with `<condition>__<experiment_id>`.
Never let a later run silently overwrite an earlier one's page.

#### Section 1 — Header and navigation

```html
<header>
  <nav>
    <a href="../../index.html">← Master dashboard</a>
    &nbsp;/&nbsp;
    <a href="../../../ideas/<NN>/dashboard/index.html">← Hypothesis <NN> sub-dashboard</a>
  </nav>
  <h1><span class="method-tag">method_A</span> — Experiment <id></h1>
  <!-- KN-strip (see Pillar 3 of autoresearch-dashboard-comprehension) -->
  <div class="kn-strip">
    <span class="method-tag">method_A</span>
    <span class="delta">+X.XX</span>
    <span class="vs">delta vs. baseline</span>
    <span class="seed-badge n7">n=7</span>
    <span class="tier-chip evaluation">EVALUATION</span>
    <span class="commit-sha">abc1234</span>
  </div>
  <!-- Multi-tag pills for combination methods -->
  <div class="tag-pills">
    <span class="tag-pill">Objective-A</span>
    <span class="plus">+</span>
    <span class="tag-pill">Objective-B</span>
  </div>
</header>
```

#### Section 2 — Hypothesis card

Inline digest from the matching hypothesis document:
- Hypothesis ID and title
- One-line plain-English summary
- Mechanism: which objective axis moves and why
- Numeric falsifier threshold (pre-registered)
- Predicted composite delta (pre-registered before the run)
- First citation in full format
- Link to the full hypothesis document (absolute GitHub-blob URL)

#### Section 3 — Verdict

The matching verdict paragraph from the project's FINDINGS ledger,
rendered through the GFM markdown converter (see
`../autoresearch-typography-and-rendering/SKILL.md`). Includes:
- Verdict tag: `KEEP` / `DISCARD` / `NEAR-MISS` / `UNTESTED`
- If verdict is internally assessed: the qualifier
  "Internal QA pass — external review pending"
- Link to the full FINDINGS section (absolute GitHub-blob URL)

#### Section 4 — 7-step reasoning entry

Rendered from `autoresearch_results/reasoning_annotations.json` for this
experiment's ID. Each of the 7 fields is rendered as a labelled block:

| Step | field key | min word-count |
|---|---|---|
| 1. Diagnose | `diagnosis` | 60 |
| 2. Cite | `citations` | 40 (single) / 80 (multi) |
| 3. Hypothesize | `hypothesis` | 50 |
| 4. Predict | `prediction` | 25 |
| 5. Execute | `execution_note` | — |
| 6. Analyse | `analysis` | 30 |
| 7. Checkpoint | `checkpoint` | 40 |

Render each field through the GFM markdown converter. If a field is
absent or below minimum length, render a visible warning:
`(reasoning field missing or below minimum — gate violation)`.

If no reasoning annotation exists for this experiment ID, render a
discreet section: "(No reasoning annotation found — 7-step gate not
verified for this experiment.)"

#### Section 5 — Configuration

If a `config.yaml` or `config.json` exists in the per-run directory,
render it as a fenced code block. Otherwise, reconstruct the inferred
overrides from `experiment_log.jsonl` for this experiment ID and render
those. Label the source clearly (file or inferred).

#### Section 6 — Metrics table

Full `metrics.json` content rendered as an HTML table:

| metric | value | n | CI 95% low | CI 95% high |
|---|---:|---:|---:|---:|
| composite | 0.8612 | 7 | 0.8401 | 0.8823 |
| objective_axis_1 | … | … | … | … |

Every numeric cell carries the `n=X` badge inline. If CI is absent,
note `(CI not computed)` rather than leaving the cell blank.

#### Section 7 — Composite breakdown

Show the composite formula term-by-term:

| term | weight | raw value | weighted contribution |
|---|---:|---:|---:|
| objective_axis_1 | 1.0 | +1.34 | +1.34 |
| − cost_axis_1 penalty | −λ₁ | +0.02 | −0.08 |
| − cost_axis_2 penalty | −λ₂ | 0.00 | 0.00 |
| **composite (sum)** | | | **0.8612** |

The displayed composite must match the value in `metrics.json` to 4
decimal places. If it diverges, render a visible discrepancy warning
and do NOT silently round.

#### Section 8 — Sweep curves

If `sweep_curves.json` (or equivalent) exists for this experiment:

- One PNG per swept parameter axis (e.g., injection layer, scale factor,
  span): objective metric on y, swept parameter on x.
- Small-multiples layout — one panel per parameter, not all overlaid.
- PNG, not SVG. ≤ 200 KB each.
- Each chart carries a one-sentence caption naming the swept parameter
  and what the optimum shows.

If no sweep data exists, render: "(No parameter sweep data for this
experiment.)"

#### Section 9 — Side-by-side artifact comparison

If the per-run `artifacts/` directory contains before/after outputs:

- Render them in a two-column layout: left = baseline output, right =
  method output.
- Label each column with its condition name (not just "before"/"after" —
  use the method tag or config diff label).
- If the artifacts are text, render them in `<pre>` blocks. If images,
  use `<img>` tags with `alt` text describing the condition.
- Caption: "Comparison of [baseline condition] (left) vs. [method
  condition] (right) on [input description]."

If no artifact data exists, render: "(No artifact comparison data for
this experiment.)"

#### Section 10 — Cross-references and footer

Cross-references section:
- Links to other seeds of the same method tag (within this campaign)
- Links to other conditions or dataset variants of the same method tag
- Each link is an absolute GitHub-blob URL or a relative path to a
  sibling page under `docs/dashboard/experiments/`

Footer:
```html
<footer>
  <code>COMPOSITE_FORMULA SHA-256: <span>…</span></code> |
  <code>Commit: <span class="commit-sha">abc1234</span></code> |
  Epochs run: <span>…</span> |
  Training duration: <span>…</span> s |
  Run dir: <code>…</code>
</footer>
```

## Mirror discipline

After building under `dashboard/experiments/`, copy byte-identically to
`docs/dashboard/experiments/`:

```python
import shutil, pathlib
exp_src = pathlib.Path("dashboard/experiments")
exp_dst = pathlib.Path("docs/dashboard/experiments")
exp_dst.mkdir(parents=True, exist_ok=True)
for page in exp_src.glob("*.html"):
    shutil.copy2(page, exp_dst / page.name)
```

## Markdown-rendering verification (binding gate)

The reasoning entry blocks, verdict block, and hypothesis card all source
from `.md`-formatted strings and MUST pipe through the GFM-table +
blockquote converter (see `../autoresearch-typography-and-rendering/`).
After every template change, the Playwright probe at
`scripts/verify_markdown_rendering.py` MUST pass on at least 5 sampled
pages — including any headline experiment pages and any pages where block-
quote tables appear.

The fix is NOT done until the Playwright probe passes. This regression
has shipped multiple times in past sessions because the gate was skipped.

## Hard rules

1. **No row-click modals.** Every table row navigates to a page. Modals
   are too cramped for 10 sections of context.
2. **Reasoning entry rendered fully.** All 7 fields, every time. Missing
   fields get a visible warning, not a silent gap.
3. **Composite breakdown must reconcile** to 4 dp. Discrepancy = visible
   warning.
4. **KN-strip on every page header.** `n=X` + tier chip, always.
5. **Multi-tag pills show ALL constituent tags.** No truncation to a
   single lead tag.
6. **No self-graded ACCEPT banner** without the "Internal QA pass —
   external review pending" qualifier.
7. **All markdown-sourced content rendered through GFM converter.**
   Playwright gate must pass.
8. **Every link is an absolute GitHub-blob URL** for files outside
   `docs/`, or a relative path for sibling pages under
   `docs/dashboard/`. See `../autoresearch-link-discipline/SKILL.md`.
9. **Back-links required:** "← Master dashboard" and "← Hypothesis
   sub-dashboard" in the page header.
10. **Mirror to `docs/dashboard/experiments/`** before marking the
    checkpoint done.

## Anti-patterns

- A "reasoning modal" that pops on row click — too cramped; rows
  navigate to pages.
- A flat aggregate table with 50+ rows and no group sectioning.
- Per-experiment pages that embed the hypothesis title but not its
  mechanism, falsifier, and predicted delta — readers need all three.
- Updating `dashboard/experiments/` without re-mirroring
  `docs/dashboard/experiments/` — GitHub Pages goes stale.
- Composite breakdown that doesn't reconcile to the reported composite.
- Section 4 (reasoning) silently empty when the annotation is missing —
  render the warning so the gap is visible.

## Cross-references

- `../autoresearch-dashboard/SKILL.md` — the master dashboard that
  links every row to these pages; hierarchy contract lives there.
- `../autoresearch-dashboard-comprehension/SKILL.md` — KN-strip badge
  pattern, multi-tag pills, and no-self-grading-banner discipline.
- `../autoresearch-typography-and-rendering/SKILL.md` — GFM markdown
  converter and serif font stack that this page template depends on.
- `../autoresearch-link-discipline/SKILL.md` — absolute GitHub-blob
  URLs for every cross-reference link on the page.
- `../autoresearch-experiment/SKILL.md` — the 7-step ritual whose
  reasoning fields are rendered in Section 4.
- `../autoresearch-meta/SKILL.md` — process spine; per-experiment pages
  are produced at the Checkpoint step (Step 7) of every experiment.
