---
name: autoresearch-link-discipline
description: >
  Use to enforce link hygiene across all externally-facing artefacts —
  README, dashboard HTML, sub-dashboards, per-experiment pages, hypothesis
  docs, audit files. Enforces absolute GitHub-blob URLs for any file outside
  docs/ referenced from Pages-served HTML; first-mention linkification of
  methods, datasets, techniques, and arXiv IDs; Playwright HEAD-test of every
  href; scoped per-experiment commits (never git add -A); and the append-only
  audit ledger discipline.
metadata:
  pillars_enforced: [absolute-urls, first-mention-linkification,
                     playwright-link-sweep, scoped-commits, append-only-audits]
  added: 2026-05-30
---

# Skill — Link discipline + first-mention linkification

## When to use

- AFTER any file restructure that moves files — rewrite every cross-
  reference before committing.
- AFTER any change to the dashboard generator or per-experiment page
  template — re-run the Playwright link sweep.
- BEFORE marking any artefact (README, FINDINGS, dashboard) "done" —
  every method name, dataset name, technique, and arXiv ID must be
  hyperlinked on its first mention.
- WHEN adding a new audit file — append to the ledger, never overwrite.
- WHEN authoring a per-experiment page — every cross-reference link in
  Section 9 must pass the HEAD-test before the page is committed.

## Pillar 1 — Absolute GitHub-blob URLs from Pages-served HTML

GitHub Pages publishes ONLY the `docs/` directory. Any link in generated
HTML that points to a file outside `docs/` using a relative path (e.g.
`../../FINDINGS.md`) resolves on Pages to the repo root and returns 404.

The fix — all references to files outside `docs/` must use absolute
GitHub-blob URLs:

```python
# src/.../link_helpers.py
GITHUB_BLOB = "https://github.com/<user>/<repo>/blob/main"

def repo_link(path: str) -> str:
    """Convert a repo-rooted path to an absolute GitHub-blob URL.

    Use for any file OUTSIDE docs/ referenced from Pages-served HTML.
    Use a relative path ONLY for files inside docs/dashboard/ (siblings
    of the page being served).
    """
    return f"{GITHUB_BLOB}/{path.lstrip('/')}"

# Usage examples in the dashboard generator:
findings_url   = repo_link("FINDINGS.md")
ledger_url     = repo_link("EXPERIMENT_LEDGER.md")
hypothesis_url = repo_link("ideas/01/HYPOTHESIS.md")
skill_url      = repo_link("meta-skills/autoresearch-dashboard/SKILL.md")
audit_url      = repo_link("audits/REVIEWER_PASS.md")
```

### Forbidden HTML patterns (will 404 on Pages)

```html
<!-- BROKEN: relative path outside docs/ -->
<a href="../../FINDINGS.md">FINDINGS</a>
<a href="../../../ideas/01/HYPOTHESIS.md">Hypothesis 01</a>
<script>fetch('../../FINDINGS.md').then(…)</script>

<!-- CORRECT: absolute blob URL -->
<a href="https://github.com/<user>/<repo>/blob/main/FINDINGS.md"
   target="_blank" rel="noopener">FINDINGS</a>
<a href="https://github.com/<user>/<repo>/blob/main/ideas/01/HYPOTHESIS.md"
   target="_blank" rel="noopener">Hypothesis 01</a>
```

Relative paths are permitted ONLY for sibling pages under
`docs/dashboard/` — for example, a per-experiment page linking to a
neighbouring experiment page via `./expNNN.html` is acceptable.

### Link target conventions

| what is linked | target pattern |
|---|---|
| Files inside `docs/dashboard/` | relative path `./file.html` or `../index.html` |
| Files outside `docs/` in the same repo | `repo_link("path/to/file")` → absolute blob URL |
| arXiv abstracts | `https://arxiv.org/abs/<ID>` |
| External datasets, papers, tools | canonical homepage or arXiv abstract URL |
| Hypothesis sub-dashboards | `repo_link("ideas/<NN>/dashboard/index.html")` if outside docs/, else relative |

## Pillar 2 — First-mention linkification

The first time any of the following items appears in any artefact
(README, FINDINGS, EXPERIMENT_LEDGER, any dashboard page, any hypothesis
doc), it MUST be a hyperlink. Subsequent mentions in the same document
may be plain text.

| category | link target |
|---|---|
| Methods / architectures / algorithms | arXiv abstract URL of the originating paper |
| Datasets or benchmarks | dataset homepage or arXiv URL |
| Techniques (optimisers, regularisers, etc.) | original paper arXiv URL |
| arXiv IDs (`arXiv:XXXX.XXXXX`) | `https://arxiv.org/abs/XXXX.XXXXX` |
| Hypothesis IDs (`ideas/<NN>/`) | `repo_link("ideas/<NN>/HYPOTHESIS.md")` |
| Skill references | `repo_link("meta-skills/<name>/SKILL.md")` |
| Audit files | `repo_link("audits/<file>.md")` |
| CLAUDE.md section references | `repo_link("CLAUDE.md")` + `#section-N` anchor |

Common first-mention offences to check on every artefact:

- A method name mentioned in the abstract but not linked until a later
  section.
- "arXiv:XXXX.XXXXX" written as plain text rather than a live link.
- Hypothesis IDs (`ideas/01`) written as plain text without a link to
  the hypothesis document.
- Audit file names cited in the verdict without a link to the file.

First-mention linkification at the moment of writing is the discipline.
Retrofitting takes longer than doing it once correctly.

## Pillar 3 — Playwright link sweep (the binding contract)

After any generator change or file restructure, run the following sweep
BEFORE committing. A failing sweep is a blocking gate — do not mark the
task done until the sweep produces zero broken links.

```python
# scripts/verify_links.py
from playwright.sync_api import sync_playwright
from pathlib import Path
import requests

DASHBOARD_ROOT = Path("dashboard")
DOCS_ROOT      = Path("docs")
TIMEOUT_S      = 10

def collect_hrefs(page) -> list[str]:
    hrefs = page.eval_on_selector_all(
        "a[href]",
        "els => els.map(e => e.getAttribute('href'))"
    )
    return [h for h in hrefs if h and not h.startswith("#")]

def head_test(url: str) -> tuple[bool, int]:
    try:
        r = requests.head(url, allow_redirects=True, timeout=TIMEOUT_S)
        return r.status_code < 400, r.status_code
    except Exception:
        return False, -1

broken = []
with sync_playwright() as p:
    browser = p.chromium.launch()
    all_html = (
        list(DASHBOARD_ROOT.rglob("*.html")) +
        list(DOCS_ROOT.rglob("*.html"))
    )
    for html in all_html:
        page = browser.new_page()
        page.goto(f"file:///{html.absolute().as_posix()}")
        for href in collect_hrefs(page):
            if href.startswith("http"):
                ok, status = head_test(href)
                if not ok:
                    broken.append((str(html), href, status))
        page.close()
    browser.close()

if broken:
    for h, link, status in broken:
        print(f"BROKEN [{status}] {link}  in  {h}")
    raise SystemExit(1)
print(f"Link sweep PASS: 0 broken across {len(all_html)} pages.")
```

Run this in the same Playwright session as `scripts/verify_markdown_rendering.py`
(from `../autoresearch-typography-and-rendering/SKILL.md`) — they share a
browser context and together form the pre-commit verification gate.

## Pillar 4 — Scoped commits (never git add -A)

Every commit that touches dashboard files must scope its `git add` to
the exact files changed. Never use `git add -A` or `git add .` — these
accidentally include:
- Sensitive files (`.env`, credential configs).
- Unrelated experiment artifacts from a concurrent run.
- Large binary files (`.png`, `.pt` model weights) that inflate repo size.

Correct pattern for a dashboard update commit:

```bash
git add dashboard/index.html docs/dashboard/index.html
git add dashboard/plot_pareto_objective_vs_cost.png docs/dashboard/plot_pareto_objective_vs_cost.png
git add docs/dashboard/experiments/exp042.html
# NOT: git add -A
git commit -m "dashboard: regenerate after exp042 (method_A, KEEP, composite 0.8612)"
```

Each per-experiment commit message must name the experiment ID and verdict.

## Pillar 5 — Append-only audit ledger

Audit files in `audits/` are append-only. Rules:

- NEVER overwrite an existing audit file. Create a new file with a date
  suffix (`audits/REVIEWER_PASS_2026-05-30.md`) or append a new dated
  section to the existing file.
- The prior audit's verdict is preserved. A new pass references the
  prior verdict; it never silently replaces it.
- The audit ledger index lives at `audits/AUDIT_SUMMARY.md` — list
  every audit file with its date, verdict, and rebuttal status.
- When an external verdict downgrades an internal one, update the
  `audits/AUDIT_SUMMARY.md` status column in the same commit that
  processes the external audit. Do not leave a stale "PASS" status after
  a "WEAK_REJECT" is on file.

## Anti-patterns

- **Inline `fetch('../some.md')` to render markdown client-side.**
  Forbidden. Files outside `docs/` are not on Pages. Use
  `window.open(repo_link("…"), "_blank")` instead, or pre-render the
  content into the HTML at build time.
- **"I'll add the link the next time I'm in the file."** Do it at the
  moment of first mention, not retrospectively.
- **Linking to the GitHub `/blob/` view for files inside `docs/`.** Those
  files are live on Pages — use the Pages URL for stylistic consistency.
- **Overwriting an audit file with a "v2" pass.** Append a dated section;
  the prior verdict is part of the historical record.
- **Running the link sweep only on `dashboard/` and not `docs/`.** The
  Pages mirror must also pass — it is the surface users actually visit.
- **Skipping scoped `git add` for "small" changes.** The scope discipline
  is not about size; it is about auditability and preventing accidental
  inclusion of sensitive or binary files.
- **Relative links from Pages HTML to files outside `docs/`.** These 404
  silently on Pages while appearing to work on a local file:// open.

## Cross-references

- `../autoresearch-dashboard/SKILL.md` — lists this skill as a mandatory
  companion; no dashboard commit is complete without the link sweep.
- `../autoresearch-typography-and-rendering/SKILL.md` — sibling skill;
  the markdown rendering gate and the link sweep run together in one
  Playwright session.
- `../autoresearch-per-experiment-page/SKILL.md` — every cross-reference
  link in Section 9 of each per-experiment page must pass the HEAD-test.
- `../autoresearch-meta/SKILL.md` — the link sweep is part of the
  Checkpoint step (Step 7) of the 7-step experiment ritual; scoped
  commits are required at every checkpoint.
- `../autoresearch-experiment/SKILL.md` — the per-experiment commit
  message format originates in the experiment ritual and is enforced here.
