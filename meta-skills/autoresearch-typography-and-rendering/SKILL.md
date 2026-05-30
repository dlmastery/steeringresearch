---
name: autoresearch-typography-and-rendering
description: >
  Use when generating ANY externally-facing HTML artefact — master dashboard,
  sub-dashboard, per-experiment page, GitHub Pages landing page — that embeds
  content sourced from .md files. Enforces a serif font stack with offline
  fallback, a GFM-table + blockquote + fenced-code markdown pipeline, and a
  Playwright assertion gate that no literal markdown syntax (##, **, |---|)
  leaks through any embedded block on EITHER the dashboard OR per-experiment
  pages. This gate must run after every generator change; marking a change
  "done" without it is a violation.
metadata:
  pillars_enforced: [typography-palette, markdown-pipeline, playwright-gate]
  added: 2026-05-30
---

# Skill — Academic typography + verified markdown rendering

## When to use

- Generating `dashboard/index.html`, any `ideas/<NN>/dashboard/index.html`
  sub-dashboard, or any `docs/dashboard/experiments/*.html` per-experiment
  page.
- Writing the GitHub Pages landing page (`docs/index.html`).
- Embedding any content sourced from `.md` files — reasoning entries,
  verdict blocks, hypothesis digests, citations, or any freeform
  annotation field — into HTML.
- After ANY change to a dashboard generator script, re-run the Playwright
  verification gate before committing. Do NOT mark the change done without
  the gate passing.

## Pillar 1 — Academic typography palette

### Required font stack

```css
body {
  font-family: 'Source Serif 4', 'Source Serif Pro', Charter, Georgia, serif;
  font-feature-settings: 'kern', 'liga', 'onum';
  font-size: 1rem;
  line-height: 1.6;
}
h1, h2, h3, h4, h5, h6 {
  font-family: 'Source Serif 4', 'Source Serif Pro', Charter, Georgia, serif;
  font-weight: 600;
  /* NOT font-style: italic — italic is for emphasis within text, not headings */
}
code, pre, .mono, .commit-sha, .fingerprint {
  font-family: 'IBM Plex Mono', ui-monospace, 'Cascadia Code', Consolas, monospace;
  font-size: 0.875em;
}
```

### Font loading with offline fallback

```html
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="preload" as="style"
  href="https://fonts.googleapis.com/css2?family=Source+Serif+4:opsz,wght@8..60,400;8..60,600&family=IBM+Plex+Mono:wght@400;600&display=swap">
<link rel="stylesheet"
  href="https://fonts.googleapis.com/css2?family=Source+Serif+4:opsz,wght@8..60,400;8..60,600&family=IBM+Plex+Mono:wght@400;600&display=swap">
```

The `Charter, Georgia, serif` tail of the fallback chain renders without
network on a local `dashboard.html` when GitHub Pages is unreachable or
when a corporate proxy blocks CDN requests.

### Forbidden font families on research artefacts

- `Newsreader` — magazine display face; italic-as-emphasis aesthetic.
- `Playfair Display`, `Lora`, `Crimson Pro` — editorial display; too
  decorative for data-dense tables.
- `Inter`, `Roboto`, `Open Sans` — generic SaaS sans-serif. Acceptable
  for utility UI chrome (buttons, nav labels), NOT for body text, headings,
  or KPI labels on any research artefact.
- `font-style: italic` for headings or KPI labels — italic is reserved
  for emphasis WITHIN running text, never as display styling.

### Unified palette across all surfaces

The master dashboard, every sub-dashboard, and every per-experiment page
MUST use the same CSS variables block. A shared `_shared.css` or an
identical `<style>` block in every generator is the only acceptable
approach. Visual mismatch between tiers signals unfinished work and is
treated as a presentation-layer DISCARD.

### Footer self-description must match the CSS

If the footer claims "Source Serif 4 / IBM Plex Mono" but the CSS
imports a different family, that is both a typography violation and a
self-description bug. Drive the footer font-name string from the same
constant that populates the `<link>` href — so the two cannot drift.

## Pillar 2 — Markdown rendering pipeline

Any content sourced from a `.md` file — reasoning entries, verdict
blocks, hypothesis digests, paper abstract, citations — MUST pass through
a converter that handles ALL of the following constructs before being
written into HTML:

| markdown construct | required output | example |
|---|---|---|
| `**bold**` | `<strong>` | bold |
| `*em*` / `_em_` | `<em>` | em |
| `` `code` `` | `<code>` | code |
| `# H1` … `###### H6` | `<h1>` … `<h6>` | headings |
| `- bullet` / `* bullet` | `<ul><li>` | unordered list |
| `1. item` | `<ol><li>` | ordered list |
| `> blockquote` | `<blockquote>` | blockquote |
| `\| col1 \| col2 \|` GFM table | `<table><thead><tbody>` | table |
| `> > nested blockquote` | nested `<blockquote>` | nested block |
| `> \| t1 \| t2 \|` table inside blockquote | `<blockquote><table>` | the historic bug class |
| triple-backtick fenced code | `<pre><code>` | fenced code |
| `[text](url)` inline link | `<a href="…">` | link |

A converter that handles `**` and `*` but omits GFM tables or
`>`-blockquotes is the bug class that has been shipped and "fixed"
multiple times. Use a well-tested library (e.g., `marked.parse(src,
{gfm:true})`, `markdown-it` with GFM table extension) or a custom
converter that has passing test fixtures for EVERY row in the table
above — especially the "table inside blockquote" case.

### Embedded-block CSS selectors to protect

The Playwright gate checks these CSS selectors for literal markdown leaks:

```python
EMBEDDED_MD_SELECTORS = [
    ".headline-ribbon",
    ".findings-verdict",
    ".reasoning-diagnosis",
    ".reasoning-citations",
    ".reasoning-hypothesis",
    ".reasoning-prediction",
    ".reasoning-analysis",
    ".reasoning-checkpoint",
    ".hypothesis-digest",
    ".sci-critic-block",
    ".impl-critic-block",
]
```

Any element matching these selectors that contains a literal markdown
token is a failing assertion. Apply the GFM converter to the content
before inserting it into any of these elements.

## Pillar 3 — Playwright verification gate (the binding contract)

After EVERY change to any dashboard or per-experiment page generator,
run the following verification BEFORE committing. The fix is not done
until this script produces no assertion errors.

```python
# scripts/verify_markdown_rendering.py
from playwright.sync_api import sync_playwright
from pathlib import Path

DASHBOARD_ROOT = Path("dashboard")

LITERALS_THAT_MUST_NOT_APPEAR = [
    "## ", "### ", "#### ",   # leaked headings
    "**", "__",               # leaked bold
    "|---|", "|---:|",        # leaked table separators
    "&gt; |",                 # escaped blockquote table
    "> |",                    # raw blockquote table
    "```",                    # leaked fenced code
]

EMBEDDED_MD_SELECTORS = [
    ".headline-ribbon",
    ".findings-verdict",
    ".reasoning-diagnosis",
    ".reasoning-citations",
    ".reasoning-hypothesis",
    ".reasoning-prediction",
    ".reasoning-analysis",
    ".reasoning-checkpoint",
    ".hypothesis-digest",
]

def verify_page(page_path: Path):
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()
        page.goto(f"file:///{page_path.absolute().as_posix()}")
        for sel in EMBEDDED_MD_SELECTORS:
            el = page.query_selector(sel)
            if not el:
                continue
            text = el.inner_text()
            for lit in LITERALS_THAT_MUST_NOT_APPEAR:
                assert lit not in text, (
                    f"FAIL {page_path.name}: literal {lit!r} leaked in {sel}"
                )
        browser.close()

html_pages = list(DASHBOARD_ROOT.rglob("*.html"))
for html in html_pages:
    verify_page(html)
print(f"Markdown rendering PASS across {len(html_pages)} pages.")
```

Run on BOTH `dashboard/` and `docs/dashboard/` to catch any divergence
between the built output and the Pages mirror. Minimum coverage: 1
aggregate dashboard page + 4 per-experiment pages, including any pages
that embed block-quote tables.

The same Playwright session that verifies markdown rendering should also
run the link sweep from `../autoresearch-link-discipline/SKILL.md` — run
both in sequence, not separately.

## Anti-patterns

- **Claiming "markdown rendering fixed" without Playwright proof.** This
  regression has shipped in multiple sessions because the gate was skipped.
  Every PR or commit that touches a markdown rendering path is suspect
  until the gate passes.
- **Different font stacks on the master dashboard vs. per-experiment
  pages.** One CSS variables block, consumed by all generators.
- **`font-style: italic` on headings or KPI labels.** Use `font-weight:
  600` for emphasis; never italic as display chrome.
- **CDN-only font loading without a serif fallback chain.** Offline
  render must work in `Charter, Georgia, serif`.
- **A converter that handles inline markdown but not GFM tables.** The
  table-in-blockquote case is the hardest and most frequently missed.
- **Not running the Playwright gate after a "minor" template tweak.**
  Minor tweaks are how the regression ships. The gate is always required.
- **Running the gate on only the master dashboard HTML.** Run it across
  ALL HTML pages including per-experiment pages — they use the same
  converter and are equally prone to the regression.

## Cross-references

- `../autoresearch-dashboard/SKILL.md` — lists this skill as a mandatory
  companion; no dashboard.html commit is complete without the Playwright
  gate passing.
- `../autoresearch-per-experiment-page/SKILL.md` — page template routes
  all reasoning and verdict content through this same converter.
- `../autoresearch-link-discipline/SKILL.md` — sibling skill; the link
  sweep and the markdown rendering gate run in sequence in the same
  Playwright session.
- `../autoresearch-dashboard-comprehension/SKILL.md` — the CSS chip
  palette (`.tier-chip`, `.seed-badge`) is defined in the shared CSS
  block managed by this skill.
- `../autoresearch-meta/SKILL.md` — the Playwright verification gate is
  part of the Checkpoint step (Step 7) of the 7-step experiment ritual.
