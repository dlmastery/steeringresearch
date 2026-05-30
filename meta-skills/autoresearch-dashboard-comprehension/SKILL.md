---
name: autoresearch-dashboard-comprehension
description: >
  Use when designing or revising any chart on any dashboard surface or
  per-experiment page. Enforces four non-negotiable pillars: (1) small-
  multiples over dense overlaid charts, (2) mandatory 4-bullet "how to
  read" orientation block at the top of every dashboard surface, (3) n=X
  seed-count + SCREENING/EVALUATION tier chip on every numeric, (4) all
  participating objective-axis or hypothesis tags shown in multi-method
  pills — no truncation to a single lead tag — plus the no-self-grading-
  banner discipline.
metadata:
  pillars_enforced: [small-multiples, orientation-block, seed-tier-badge,
                     multi-tag-pills, no-self-grading]
  added: 2026-05-30
---

# Skill — Dashboard / figure comprehension discipline

## When to use

- Designing any new chart for the master dashboard, a sub-dashboard, a
  per-experiment page, a paper figure, or a README visual.
- Revising any existing chart that has been flagged as dense, illegible,
  or mis-framed.
- Before publishing any dashboard surface — run all 4 pillars as a
  pre-commit checklist.
- After any generator change that touches chart-rendering code — re-verify
  all 4 pillars, not just the changed panel.

## Pillar 1 — Small-multiples over dense overlaid charts

A chart with 3 or more overlaid axes / lines / method variants is
FORBIDDEN when the same information fits in side-by-side small-multiples.

### Conversion table

| dense (forbidden) | small-multiples (required) |
|---|---|
| One composite bar chart with 30+ method bars, all mixed | One mini-bar-chart per hypothesis group in a grid; ≤ 10 bars each |
| One training-curve plot with 20 lines | One panel per method group, ≤ 5 lines each, shared y-axis |
| One Pareto scatter with all variants overlaid | One Pareto panel per objective-vs-cost pair; baselines repeated in each |
| One leaderboard table with 50+ rows | Group-sectioned tables, ≤ 12 rows each; one section per hypothesis group |

The master dashboard `dashboard/index.html` is already group-sectioned
— extend the same discipline to all embedded figures.

### Caption contract

Every chart carries a 1-sentence "what to read" caption directly under
the image. If the chart requires a paragraph to explain, it is the wrong
chart.

Example caption for a Pareto panel:

> Composite objective vs. parameter count. Lower-right = dominated on
> both axes; stars = prior baselines; circles = current campaign methods.
> **What to read: any circle below-and-right of all stars is dominated.**

## Pillar 2 — "How to read this dashboard" orientation block

Every dashboard surface — master, sub-dashboard, or per-experiment page —
opens with EXACTLY 4 bullets at the top of the page, above the headline
ribbon. Do not bloat to 6; do not shrink to 2.

Required HTML template:

```html
<aside class="how-to-read">
  <h3>How to read this dashboard</h3>
  <ul>
    <li><strong>What this page shows:</strong>
        every experiment row in this campaign, with its hypothesis context
        and composite score across all objective axes.</li>
    <li><strong>Colour coding:</strong>
        green row = KEEP verdict; yellow = NEAR-MISS; orange = DISCARD;
        red = BROKEN / FALSIFIED / NUMEROLOGY.</li>
    <li><strong>Screening vs. evaluation:</strong>
        rows tagged <code>SCREENING</code> used ≤ 3 seeds and cannot
        support external claims. <code>EVALUATION</code> rows used ≥ 7
        seeds and have cleared the full statistical gate.</li>
    <li><strong>Drill-down:</strong>
        click any row to navigate to its per-experiment page (no
        modals). The sub-dashboard link navigates to the hypothesis-level
        view.</li>
  </ul>
</aside>
```

Customise the text of each bullet to the project's terminology, but
keep the four structural topics: what/colour/tier/drill-down.

## Pillar 3 — Seed-count + tier chip on every numeric

Every numeric value on every visual surface — table cells, KPI strips,
headline ribbons, radar axis labels, Pareto axis ticks — carries a
seed-count badge and a tier chip. No bare numbers.

### KN-strip pattern (per-experiment page header)

```html
<div class="kn-strip">
  <span class="method-tag">method_A</span>
  <span class="delta">+X.XX pp</span>
  <span class="vs">delta vs. baseline</span>
  <span class="seed-badge n7">n=7</span>
  <span class="tier-chip evaluation">EVALUATION</span>
  <span class="commit-sha">abc1234</span>
</div>
```

### Leaderboard table columns

Add `n` and `Tier` as the final two columns of every leaderboard:

| Method | Composite | Objective | Cost-axis | **n** | **Tier** |
|---|---:|---:|---:|---:|---|
| method_A | 0.8612 | +1.34 | 1.4 ms | 7 | EVALUATION |
| method_B | 0.5741 | +0.25 | 1.2 ms | 3 | EVALUATION |
| method_C | 0.5196 | −0.80 | 1.5 ms | 1 | SCREENING |

### Headline ribbon pattern

```
Phase-N winners (n=7 seeds, EVALUATION, Holm-Bonferroni α'=0.007):
  method_A +1.34 pp composite, method_B +0.25 pp.
Phase-N negatives (n=1 seed, SCREENING; not falsifications):
  axis_X baseline −1 pp.
```

Negative results must appear with equal prominence to positive results —
do not suppress them.

### CSS palette (consistent across all surfaces)

```css
.seed-badge     { background: #eef;    color: #224;    padding: 2px 6px; border-radius: 3px; font-size: .8em; }
.tier-chip.screening  { background: #fff4e5; color: #8a5800; padding: 2px 8px; border-radius: 3px; font-size: .8em; }
.tier-chip.evaluation { background: #e8f4e8; color: #1f5320; padding: 2px 8px; border-radius: 3px; font-size: .8em; }
tr.champion     { background: #f0faf0; font-weight: 600; }
tr.discard      { background: #fff5f5; }
tr.near-miss    { background: #fffbe5; }
```

Apply these CSS classes identically on the master dashboard, every sub-
dashboard, and every per-experiment page. A unified palette across all
three tiers is required — mismatched palettes signal unfinished work.

## Pillar 4 — Multi-tag pills + no-self-grading banners

### Multi-tag (multi-objective / multi-hypothesis) pill display

A method that stacks or combines multiple objectives or hypotheses MUST
display ALL of them in its pill row. Truncating to a single lead tag
is forbidden.

```html
<!-- WRONG: shows only the lead objective -->
<div class="tag-pill">Objective-A</div>

<!-- RIGHT: shows all participating objectives or hypotheses -->
<div class="tag-pills">
  <span class="tag-pill">Objective-A</span>
  <span class="plus">+</span>
  <span class="tag-pill">Objective-B</span>
  <span class="plus">+</span>
  <span class="tag-pill">Objective-C</span>
</div>
```

The mapping from method tags to their constituent objectives or
hypotheses lives in the project's `_tag_objective_map.py` (or
equivalent) and is the single source of truth. Update it every time a
new combination experiment is added.

### No self-grading banners

Any verdict banner on any dashboard surface — master, sub-dashboard,
per-experiment page — that uses "ACCEPT" / "FINAL" / "Confirmed winner"
MUST carry the external-review qualifier. Pattern:

```html
<!-- FORBIDDEN -->
<div class="banner ok">
  Internal ACCEPT verdict at commit abc1234.
</div>

<!-- REQUIRED -->
<div class="banner internal-qa">
  <strong>Internal QA pass</strong>
  (verdict: ACCEPT at commit abc1234, same-family critic agent).
  <em>Independent external review pending — see
  <a href="https://github.com/<user>/<repo>/blob/main/audits/REVIEWER_PASS.md"
     target="_blank" rel="noopener">audits/REVIEWER_PASS.md</a>
  for external verdict status.</em>
</div>
```

When an external reviewer's verdict downgrades an internal verdict, the
dashboard surface MUST reflect the downgrade in the same commit that
processes the audit. An outdated "ACCEPT" banner after a WEAK_REJECT is
on file is a blocking violation.

## Anti-patterns

- **3+ axes on one chart with a multi-paragraph caption** — split into
  small-multiples with one-sentence captions each.
- **"How to read" block of 6+ bullets** — reduce to the 4 load-bearing
  topics: what / colour / tier / drill-down.
- **A leaderboard cell showing a bare number without `n=X`** — every
  numeric, every time, on every surface.
- **A combined-method pill showing only the lead objective or hypothesis
  tag** — show ALL participants.
- **Headline ribbon that lists positive results but omits calibrated
  negatives** — equal prominence required.
- **"ACCEPT" / "FINAL" banner without the "Internal QA pass — external
  review pending" qualifier.**
- **Different CSS palette on the master dashboard vs. sub-dashboards
  vs. per-experiment pages** — one unified palette across all tiers.
- **Radar/parallel-coordinates chart axis labels that don't match the
  composite formula term names** — no silent abbreviations.

## Cross-references

- `../autoresearch-dashboard/SKILL.md` — the master dashboard generator
  that calls this skill as a prerequisite for every panel.
- `../autoresearch-per-experiment-page/SKILL.md` — the KN-strip badge
  pattern is embedded in the per-experiment page template.
- `../autoresearch-typography-and-rendering/SKILL.md` — the visual
  palette and chip CSS originate from the same shared CSS block.
- `../autoresearch-paper-rigor/SKILL.md` — the statistical arm of the
  same screening-vs-evaluation discipline (seed count → Wilcoxon →
  Holm-Bonferroni chain).
- `../autoresearch-meta/SKILL.md` — the process spine; this skill
  enforces presentation discipline at the Checkpoint step.
