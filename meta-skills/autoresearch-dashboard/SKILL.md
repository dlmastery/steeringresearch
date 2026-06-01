---
name: autoresearch-dashboard
description: >
  Use when generating the master sortable/filterable HTML dashboard and
  its per-hypothesis sub-dashboards. Produces self-contained HTML (no CDN,
  single inline <script>), PNG plots, a Pareto panel, a ladder board, a
  radar/parallel-coordinates panel, a 4-bullet "how to read", a
  COMPOSITE_FORMULA fingerprint footer, and hierarchical sub-links from
  every row to per-hypothesis and per-experiment pages. Mirrors to
  docs/dashboard/ for GitHub Pages.
metadata:
  companion_skills:
    - autoresearch-dashboard-comprehension
    - autoresearch-per-experiment-page
    - autoresearch-typography-and-rendering
    - autoresearch-link-discipline
  added: 2026-05-30
---

# Skill — Master + sub-dashboard generator

## When to use

After any sweep, mid-campaign progress check, or before publishing a
checkpoint. Re-run whenever the project's `experiment_log.jsonl` or any
per-run `metrics.json` changes. The dashboard is the primary deliverable
— regenerate and push it on every milestone, not just at campaign end.

## Required inputs

| source | role |
|---|---|
| `autoresearch_results/experiment_log.jsonl` | append-only run history; one JSON object per line |
| `autoresearch_results/best_config.json` | global champion config + composite score |
| `autoresearch_results/reasoning_annotations.json` | 7-step pre/post reasoning entries per experiment |
| `ideas/<NN>/` | per-hypothesis sub-project directories |
| `src/.../eval.py:COMPOSITE_FORMULA` | the multi-axis composite formula + SHA-256 fingerprint |

Optional enrichers (skip gracefully when absent):

| source | role |
|---|---|
| per-run `sweep_curves.json` | parameter-sweep curves for the per-experiment page |
| per-run `artifacts/` | side-by-side output samples for comparison panels |
| `autoresearch_results/running.json` | transient signal; omit from static HTML |

## Outputs

| file | purpose |
|---|---|
| `dashboard/index.html` | master dashboard — self-contained sortable HTML |
| `dashboard/plot_pareto_<axis>.png` | one PNG per objective-vs-cost-axis Pareto panel |
| `dashboard/plot_radar.png` | multi-axis radar / parallel-coordinates per method |
| `dashboard/plot_ladder.png` | ladder-board rung diagram |
| `ideas/<NN>/dashboard/index.html` | per-hypothesis sub-dashboard |
| `docs/dashboard/index.html` | byte-identical mirror of master for GitHub Pages |
| `docs/dashboard/ideas/<NN>/dashboard/index.html` | byte-identical mirror of each sub-dashboard |
| `docs/index.html` | landing page linking to the master dashboard |

## Hierarchy contract — the core deliverable

The dashboard is HIERARCHICALLY LINKED across three tiers:

```
dashboard/index.html                   ← master dashboard
  └─ ideas/<NN>/dashboard/index.html   ← per-hypothesis sub-dashboard
        └─ docs/dashboard/experiments/expNNN.html  ← per-experiment page
```

Every row in the master table carries TWO link cells:

1. A sub-dashboard link → `ideas/<NN>/dashboard/index.html`
2. A per-experiment link → `docs/dashboard/experiments/<id>.html`

Every sub-dashboard row links to its per-experiment page AND carries a
"Back to master" link in its header.  Every per-experiment page carries
both "Back to sub-dashboard" and "Back to master" links.

No orphan pages. No row without both sub-links.

## Required panels — master dashboard

### Panel 0 — Newcomer grounding block (MANDATORY, rendered first)

Every master dashboard MUST open with a **newcomer grounding block** that appears
before any table, chart, or ribbon. A first-time reader who knows nothing about
the project must be able to answer three questions from this block alone:

**A. Research goal / outcome (≥ 2 sentences)**
State the concrete deliverable in plain English. Do NOT write "we study X" —
write what you are trying to invent or discover, what SOTA means concretely
in this domain (the trade-off being maximized), and what a "winner" means. Example:

> "Goal: discover the parameter configuration that maximizes \[primary_efficacy\]
> while keeping \[cost_axis_1\] and \[cost_axis_2\] within acceptable bounds —
> the best simultaneous trade-off known for this model class. A 'winner' is any
> method that Pareto-dominates the current champion on the composite metric
> (defined below) at matched \[coherence/quality\] cost."

**B. Domain primitives paragraph (≥ 3 sentences)**
Define the core concept the program manipulates in one paragraph. Use no acronyms
without expansion on first use. No assumed background — a researcher from a
neighboring field must understand what is being done and why it is non-trivial.

**C. Methodology section with tabs OPEN by default**
The methodology section must be:
- Present and non-empty (not a "Methodology…" label that links nowhere)
- Rendered with all tabs/accordions **open** by default (not collapsed)
- Contains at minimum: (1) the 5-rung benchmark ladder summary, (2) the composite
  formula (with all axis names spelled out), (3) the screening vs evaluation
  distinction, (4) a one-sentence description of what one "experiment" is

**Checklist — verify before any publish:**
- [ ] Research goal states the concrete outcome, not just the research topic
- [ ] Domain primitives paragraph uses no undefined acronyms
- [ ] Methodology section present, non-empty, and open by default
- [ ] All other major tabs (geometry, ladder, hypotheses, runs) each have a
      one-paragraph "what this tab is and how to read it" label

### Panel 0b — Per-tab orientation paragraphs

Every named tab or major section panel in the dashboard (not just the main
orientation block) MUST have its own one-paragraph introduction:
- What this tab/section shows
- How to read the numbers or charts it contains
- What "good" looks like vs what to investigate further

This applies to: geometry panels, ladder panels, hypothesis panels, runs/table
panels, Pareto panels, stack/compete panels. A tab whose content is visible but
unexplained fails this rule.

### Panel 1 — Runs table (sortable + filterable)

The runs table is the centrepiece. HTML contract:

```html
<input id="q" oninput="filterTable()" placeholder="Type to filter…">
<table id="runs" data-dir="desc">
  <thead>
    <tr>
      <th onclick="sortTable(0)">Method</th>
      <th onclick="sortTable(1)">Composite</th>
      <!-- one <th onclick="sortTable(N)"> per column -->
      <th>n</th>
      <th>Tier</th>
      <th>Sub-dashboard</th>
      <th>Experiment</th>
    </tr>
  </thead>
  <tbody>
    <tr class="champion">  <!-- champion row highlighted -->
      <td data-v="method_A">Method A</td>
      <td data-v="0.8612">0.8612</td>
      <!-- ... -->
      <td><span class="seed-badge">n=7</span></td>
      <td><span class="tier-chip evaluation">EVALUATION</span></td>
      <td><a href="../ideas/01/dashboard/index.html">Hypothesis 01</a></td>
      <td><a href="experiments/exp042.html">exp042</a></td>
    </tr>
  </tbody>
</table>
```

Rules:
- Default sort = **composite descending**. `data-dir` toggles `asc`/`desc`
  on header click.
- `data-v` carries the raw numeric so float sort is correct.
- The **global champion row** carries `class="champion"` (green
  background highlight). There is exactly one champion row.
- **KEEP / champion rows are green; DISCARD rows are muted/grey; NEAR-MISS rows
  are yellow.** A visible legend (color key) must appear directly above or below
  the table — never assume the reader infers the color scheme.
- Every numeric cell carries `n=X` + `SCREENING` / `EVALUATION` tier
  chip. No bare numbers.
- The `filterTable()` function does `textContent.toLowerCase()` substring
  match against `#q`. No external JS library.

**Per-table "What is this" block (required on every table)**

Every table in the dashboard must be preceded by an expandable (or always-visible)
"What is this" block containing:

```html
<details class="table-explainer" open>
  <summary>What is this table?</summary>
  <p><strong>Purpose:</strong> [one sentence describing what this table shows]</p>
  <dl class="col-glossary">
    <dt>Method</dt><dd>The algorithm or configuration variant tested.</dd>
    <dt>Composite</dt><dd>The multi-axis score (higher = better). Computed as
      [formula summary]. See footer for fingerprint.</dd>
    <dt>n</dt><dd>Number of independent seeds run. n≤3 = SCREENING (cannot
      support external claims). n≥7 = EVALUATION.</dd>
    <!-- one <dt>/<dd> pair per column -->
  </dl>
  <p><strong>What to pay attention to:</strong> [what distinguishes good vs bad rows]</p>
  <p><strong>Expected/good values:</strong> [concrete examples or ranges]</p>
</details>
```

This block is required for: the runs table, the Pareto panel caption, the ladder
board, the stack/compete matrix, and any geometry table. A table without a
"What is this" block fails the dashboard audit.

### Panel 2 — Multi-axis radar / parallel-coordinates

One radar chart (or parallel-coordinates plot) per method group, rendered
as a **PNG** (not SVG), showing all objective axes simultaneously. Where
the project logs more than two objective axes, parallel-coordinates is
preferred over a 2D Pareto scatter.

The panel title must name every axis shown.  Axis labels must match the
composite formula term names exactly (no abbreviations without a legend).

### Panel 3 — Pareto panels (one per objective-vs-cost pair)

For each combination of (primary objective axis) × (cost/constraint axis):

- Plot every run as a circle; prior methods / baselines as star markers.
- Mark dominated runs visibly (e.g. red outline, `dominated` CSS class).
- **At least one dominated run must exist** — proves the harness
  discriminates rather than plotting noise.
- Caption under each plot: "What to read: a circle below-and-right of all
  stars is dominated on both axes."
- PNG, not SVG. ≤ 200 KB per PNG.

### Panel 4 — Ladder board

A visual per-method ladder showing:

- Which rung each method reached (UNIT / SMOKE / DEV / STANDARD / FULL
  or the project's equivalent tier names).
- The gate each method cleared or failed at, with the logged
  `failure_reason` shown as a tooltip or truncated inline string.
- Methods sorted by highest rung reached, then composite within rung.

Render as a PNG table or a CSS grid — no SVG, no external JS.

### Panel 5 — Stack / compete matrix (if the project tracks stacking)

If the project's experiment log contains stacking/combination entries,
render a matrix: rows = method A, columns = method B, cells = STACK /
COMPETE / UNTESTED, colour-coded. Derived live from `experiment_log.jsonl`
via the project's orthogonality decision rule. Omit this panel when fewer
than 3 combination experiments exist.

### Panel 6 — "How to read this dashboard" orientation block

Exactly 4 bullets. Always at the top of the page, above the headline
ribbon. See `../autoresearch-dashboard-comprehension/SKILL.md` for the
required HTML template and the non-negotiable 4-bullet count.

### Panel 7 — Headline ribbon + COMPOSITE_FORMULA fingerprint footer

The ribbon announces the current champion and any phase headline. The
footer carries:

```html
<footer>
  <code class="fingerprint">COMPOSITE_FORMULA SHA-256: <span id="csha">…</span></code>
  &nbsp;|&nbsp;
  <code class="commit-sha">Commit: <span id="gitsha">…</span></code>
  &nbsp;|&nbsp;
  Generated: <time datetime="…">…</time>
</footer>
```

The formula fingerprint must match `src/.../eval.py:COMPOSITE_FORMULA`
SHA-256. If they diverge, the dashboard generator must fail loudly
(raise, don't warn).

## Per-hypothesis sub-dashboard contract

Each `ideas/<NN>/dashboard/index.html` is a focused page for one
hypothesis. It must contain:

1. **Best-config callout** — the single best run for this hypothesis,
   highlighted, with all objective-axis metrics and `n=X` + tier chip.
2. **Coordinate-descent Pareto small-multiples** — one mini-Pareto per
   swept parameter axis (e.g., injection layer, scale factor, span),
   showing how the metric evolves as each parameter is tuned.
3. **Seed-stability bars** — per-seed composite bars with mean ± 1σ
   overlay, so variance is visible.
4. **Runs table** — same HTML contract as master, but scoped to this
   hypothesis only.  Each row links to its per-experiment page.
5. **Hypothesis card** — the formal statement, predicted Δ, falsifier
   threshold, current verdict (`KEEP` / `DISCARD` / `NEAR-MISS` /
   `UNTESTED`), and the reasoning citations.
6. **Navigation** — "Back to master dashboard" link in the page header
   and a "← master" breadcrumb in the footer.

## Sortable-table JS contract (inline, no CDN)

```javascript
// Entire script block lives inside the HTML <script> tag — no src=.
function sortTable(col) {
  const tbl = document.getElementById('runs');
  const dir = tbl.dataset.dir === 'asc' ? 'desc' : 'asc';
  tbl.dataset.dir = dir;
  const rows = Array.from(tbl.tBodies[0].rows);
  rows.sort((a, b) => {
    const va = parseFloat(a.cells[col].dataset.v ?? a.cells[col].textContent);
    const vb = parseFloat(b.cells[col].dataset.v ?? b.cells[col].textContent);
    return dir === 'asc' ? va - vb : vb - va;
  });
  rows.forEach(r => tbl.tBodies[0].appendChild(r));
}
function filterTable() {
  const q = document.getElementById('q').value.toLowerCase();
  Array.from(document.getElementById('runs').tBodies[0].rows)
    .forEach(r => {
      r.style.display = r.textContent.toLowerCase().includes(q) ? '' : 'none';
    });
}
```

The script block is the **only** `<script>` tag in the HTML. No CDN
links. No `import`. No frameworks.

## Mirror discipline

After every build, copy byte-identically:

```python
import shutil, pathlib
for src in pathlib.Path("dashboard").rglob("*"):
    if src.is_file():
        dst = pathlib.Path("docs") / src.relative_to("dashboard")
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
```

The live GitHub Pages URL becomes
`https://<user>.github.io/<repo>/dashboard/index.html`.
Link it from `README.md` near the top.

## Hard rules

1. **Self-contained HTML.** No external `<script src=…>`, no CSS
   framework CDN. GitHub Pages must serve it correctly without network.
2. **PNG, not SVG, for all plots.** ≤ 200 KB each; full HTML ≤ 80 KB.
3. **Composite is the default sort.** All other columns require a click.
4. **Every numeric cell carries `n=X` + tier chip.** No bare numbers
   anywhere on any dashboard surface.
5. **At least one dominated row** in every Pareto panel.
6. **Champion row highlighted** with a distinct background (CSS class
   `champion`); exactly one champion row exists. KEEP rows = green,
   DISCARD = muted, NEAR-MISS = yellow. A color legend is required.
7. **Sub-links mandatory.** Every runs-table row has a sub-dashboard
   link and a per-experiment link. No row without both.
8. **COMPOSITE_FORMULA fingerprint in the footer.** Mismatch = hard
   failure in the generator.
9. **Mirror to `docs/dashboard/`** before marking the milestone done.
10. **No emoji** on any dashboard surface unless the user explicitly
    requested it.
11. **No self-graded ACCEPT banner** without the qualifier "Internal QA
    pass — external review pending". See
    `../autoresearch-dashboard-comprehension/SKILL.md` Pillar 4.
12. **Markdown rendered.** Any `.md`-sourced content must pass through
    the GFM converter (see `../autoresearch-typography-and-rendering/`).
    Playwright asserts no literal `##` / `**` / `|---|` leaks through.
13. **Newcomer grounding block present.** Panel 0 (goal, domain
    primitives, open methodology section) must appear before any table
    or chart. A dashboard without it fails the publish gate.
14. **Per-tab orientation paragraphs.** Every named section/tab has a
    one-paragraph "what this is and how to read it" label.
15. **Per-table "What is this" block.** Every table has an expandable
    glossary + purpose + expected-values block (see Panel 1 contract).

## Companion skills — a dashboard commit is INCOMPLETE without all of

1. `../autoresearch-dashboard-comprehension/SKILL.md` — small-multiples,
   4-bullet orientation, seed-tier badges, no self-grading banners.
2. `../autoresearch-per-experiment-page/SKILL.md` — per-experiment pages
   linked from every row; mirrored to `docs/dashboard/experiments/`.
3. `../autoresearch-typography-and-rendering/SKILL.md` — serif font
   stack, GFM markdown pipeline, Playwright rendering gate.
4. `../autoresearch-link-discipline/SKILL.md` — absolute GitHub-blob
   URLs, first-mention linkification, Playwright HEAD-test every href.

After every generator change, run the Playwright verification gate
(typography + markdown + links) BEFORE marking the change done.

## Anti-patterns

- Using Plotly / Bokeh / D3 or any CDN library — too heavy; forbidden.
- SVG plots — use PNG.
- A single dense overlaid chart when 3 side-by-side small-multiples
  convey the same information (Rule 33 / comprehension skill).
- Numeric cells without `n=X` + tier chip (Rule 34 violation).
- A multi-hypothesis method showing only the lead objective pill —
  show ALL participating objective axes or hypothesis tags.
- A self-graded ACCEPT banner without the external-review qualifier.
- Rows with no sub-links — every row must link to its sub-dashboard
  and its per-experiment page.
- Regenerating `dashboard/` without re-mirroring `docs/dashboard/`
  — GitHub Pages goes stale.
- Auto-refreshing via JS polling — keep the dashboard purely static.
- Pulling fonts from a CDN without a `Charter, Georgia, serif`
  fallback chain (breaks offline + edge-cache misses).

## Cross-references

- `../autoresearch-dashboard-comprehension/SKILL.md` — prerequisite
  for any chart on any dashboard surface.
- `../autoresearch-per-experiment-page/SKILL.md` — the drill-down
  target for every row link.
- `../autoresearch-typography-and-rendering/SKILL.md` — font palette
  + markdown pipeline this template depends on.
- `../autoresearch-link-discipline/SKILL.md` — absolute-URL and
  first-mention disciplines that all links on this page must follow.
- `../autoresearch-meta/SKILL.md` — the process spine; dashboard
  generation is Step 7 (Checkpoint) of the 7-step experiment ritual.
- `../autoresearch-tiered-ladder/SKILL.md` — ladder rung definitions
  used in Panel 4.
