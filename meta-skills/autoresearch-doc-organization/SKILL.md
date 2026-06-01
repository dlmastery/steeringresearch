---
name: autoresearch-doc-organization
description: Use when planning a project's top-level directory layout, when the repo root has accumulated too many markdown files, or when the project README needs to serve as a clear entry point for an external reader. Enforces a maximum of 4 canonical root files, defines where every other file type lives, and specifies the structure of a front-door README that surfaces both positive and negative findings with equal prominence.
---

# Skill — Repo-root discipline and front-door README

## When to use

- Before creating any new documentation file at the project root —
  confirm it belongs in a subdirectory instead.
- When the repo root has accumulated many markdown files that obscure
  the primary entry points from an external reader.
- When the project README leads with motivation, history, or marketing
  language instead of the primary finding and a working quick-start.
- After any restructure that moves root-level files into subdirectories
  — verify the cross-references are updated and nothing new is added
  to root afterwards.
- Before any paper submission, external demo, or code release: the
  README is the first surface a reviewer sees.

## The 4-canonical-root-files rule

The project root contains AT MOST four canonical files:

| file | purpose |
|---|---|
| `README.md` | The front door: one-paragraph pitch, quick-start, headline findings (positive AND negative), repo map |
| `CLAUDE.md` (or equivalent operator spec) | Normative process rules for any future operator of the project |
| The paper / primary report (e.g., `PAPER.md` or `paper.pdf`) | The primary research output |
| `LICENSE` | The license file |

Every OTHER markdown, text, or documentation file lives in a
subdirectory. The subdirectory contract:

| subdirectory | what lives here |
|---|---|
| `paper/` or `docs/paper/` | Supporting sections: findings ledger, statistical tests, reviewer checklist, limitations, ethics statement, SOTA comparison |
| `docs/` | Dashboard mirror, GitHub Pages content |
| `experiments/` | Per-run archives (see [`../autoresearch-experiment-archive/SKILL.md`](../autoresearch-experiment-archive/SKILL.md)) |
| `ideas/<NN>/` | Per-hypothesis sub-projects |
| `audits/` | Implementation critic, science critic, data-split, shuffle-test, meta-process audits |
| `meta-skills/` or `skills/` | Content-agnostic process skills |
| `memory/` | Per-session crash-recovery checkpoint documents |
| `scripts/` | Runner, sweep, dashboard, build scripts |
| `src/<pkg>/` | Project source code |
| `tests/` | Unit and integration tests |
| `configs/` | Shared configuration files |

**Transitional files** (e.g., a `RESTRUCTURE_PLAN.md` mid-restructure)
are allowed at root temporarily; delete them after the restructure
completes.

## Why this matters

A project root with many markdown files presents two problems:

1. **External reader confusion.** A reviewer or collaborator arriving
   at the root cannot tell which file is the entry point, which is
   the primary finding, and which is an internal working document.
   The signal-to-noise ratio of the root is the first impression.
2. **Internal cross-reference drift.** Each file moved or created at
   root without discipline acquires links from other files. After
   several rounds of restructuring, many of those links are stale.
   The 4-file rule prevents the proliferation that makes link audits
   expensive.

## The front-door README structure

The README must serve a reader who arrives cold. Required sections, in
this order:

**1. One-paragraph elevator pitch (≤ 80 words) — with concrete outcome.**
What the project studies, what the methodology contribution is, and
one calibrated positive finding alongside one calibrated negative
finding — never the positive alone. The pitch must be factual, not
marketing: avoid "revolutionary", "state-of-the-art", "novel framework".

The pitch MUST state the **concrete outcome** being pursued: what is being
invented or discovered, what SOTA means in this domain (the trade-off being
maximized), and what a "winner" is. "This project studies X" is insufficient —
state what you are trying to find or build, specifically. See the goal/outcome
clarity mandate in `../autoresearch-meta/SKILL.md §0`.

**2. Badges row (optional but recommended).**
Links to the paper, the live dashboard, and the license. Keep it to
one line.

**3. Quick start (≤ 4 commands).**
Commands that work from a clean clone, with no hidden prerequisites.
If setup needs extra steps, document them. A quick-start that silently
requires an unlisted dependency fails the cold-reader test.

**4. Headline findings.**
Report positive and negative findings with equal visual weight. Format:
> **Positive:** [finding] (n=X, EVALUATION — [statistical test result];
> [effect size]).
> **Negative / null:** [finding] (n=Y, SCREENING; falsifier
> pre-registered before sweep; not rebuttable post-hoc).

A README that lists only positive findings and buries negatives in an
appendix fails the audit.

**5. Methodological notes.**
Seed protocol, hardware contract, screening-vs-evaluation framing,
and any protocol invariant a reader would need to reproduce the headline.

**6. Repo map (one screen).**
A directory-tree snapshot of the canonical subdirectories, showing
where to find papers, experiments, dashboards, and skills.

**7. Citation block.**
Bibtex entry or equivalent, including the arXiv identifier when
available.

**8. License and acknowledgements.**
One short paragraph.

### What the README must NOT contain

- Self-grading banners ("ACCEPT", "FINAL") without the "Internal QA
  pass — independent external review pending" qualifier.
- Negative results buried below the fold or in a limitations section
  only. Positive and negative findings receive equal visual weight in
  the headline section.
- A quick-start block that doesn't run from a clean clone.
- Marketing language that contradicts the calibrated findings in the
  paper or statistical tests.
- A vague goal statement ("this project studies X") with no concrete
  outcome: what is being maximized, what SOTA means, what a winner is.
- Implicit shared-harness architecture: if the project uses a single
  shared runner for multiple hypotheses, state it explicitly so "few
  files in src/" is not mistaken for incomplete work.

## Cross-reference rewriting after a restructure

When a file moves from `<root>/FINDINGS.md` to `paper/FINDINGS.md`:

1. Grep the entire project for every link to the old path.
2. Rewrite relative links to the new relative path from each file's
   location.
3. For HTML dashboard pages: use absolute repository blob links (HEAD-
   tested), not relative paths — relative paths break when the page
   is served from a different directory than the source.
4. After all rewrites, run the link audit (see
   [`../autoresearch-link-discipline/SKILL.md`](../autoresearch-link-discipline/SKILL.md))
   to confirm zero broken links.

## Automation: root-file check script

A pre-push or CI hook that enforces the 4-file rule:

```python
# scripts/check_root_files.py — adapt allowed list to the project.
import pathlib, sys

ALLOWED_ROOT_FILES = {
    "README.md", "CLAUDE.md", "PAPER.md", "LICENSE",
    "RESTRUCTURE_PLAN.md",   # transitional only
}

extras = [
    f.name for f in pathlib.Path(".").iterdir()
    if f.is_file() and f.suffix in {".md", ".txt", ".rst"}
    and f.name not in ALLOWED_ROOT_FILES
]

if extras:
    print(f"Rule violation: extra documentation at repo root: {extras}")
    sys.exit(1)

print("Root-file check PASS.")
```

## Anti-patterns

| Anti-pattern | Why it fails |
|---|---|
| "I'll put it at root just for now" | "Just for now" becomes permanent; the cold-reader audit finds it months later |
| README starts with project history or motivation | First sentence must name the contribution and a finding; history belongs in a `paper/` section |
| Positive finding in the headline, negative buried in limitations | Both receive equal prominence; a hostile reviewer reads the front page first |
| Quick-start that assumes a pre-configured environment | A clean-clone test is cheap and catches missing steps before release |
| Self-grading ACCEPT banner without the circularity qualifier | Every internal audit verdict carries the "independent external review pending" disclosure |

## Cross-references

- [`../autoresearch-link-discipline/SKILL.md`](../autoresearch-link-discipline/SKILL.md)
  — the link-audit protocol that pairs with any restructure; confirms
  no links broke when files moved.
- [`../autoresearch-typography-and-rendering/SKILL.md`](../autoresearch-typography-and-rendering/SKILL.md)
  — the visual discipline for the front-door README (rendered markdown,
  PNG not SVG, no-emoji default).
- [`../autoresearch-experiment-archive/SKILL.md`](../autoresearch-experiment-archive/SKILL.md)
  — defines what lives in `experiments/<NN>/`; the subdirectory
  contract here and the archive taxonomy there are complementary.
- [`../autoresearch-winner-archive/SKILL.md`](../autoresearch-winner-archive/SKILL.md)
  — the champion archive that the README's headline section should
  reference, with its frozen config and provenance.
- [`../autoresearch-paper-rigor/SKILL.md`](../autoresearch-paper-rigor/SKILL.md)
  — the statistical claims in the README headline section must satisfy
  the rigor floor; the README is an external-facing surface.
- [`../autoresearch-findings-ledger/SKILL.md`](../autoresearch-findings-ledger/SKILL.md)
  — the self-contained FINDINGS and interpretable ledger discipline;
  FINDINGS.md is one of the canonical root-adjacent documents governed
  by both this skill and the findings-ledger skill.
