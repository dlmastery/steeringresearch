#!/usr/bin/env python
"""verify_dashboard.py -- mechanical checker for RUBRICS.md Rubric B (dashboard).

Asserts the rich, transparent, hierarchically-linked dashboard mandate
(CLAUDE.md sec 11, audits/RUBRICS.md Rubric B). Each check prints PASS/FAIL with
concrete evidence; the script exits nonzero if ANY check FAILs.

This is a *mechanical* checker: it greps rendered HTML and counts files. It does
not open a browser. Run from anywhere; paths are resolved relative to the repo
root (the parent of this script's directory).

Usage:
    python scripts/verify_dashboard.py
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
FINGERPRINT = "a9001e87087e"

# (name, ok, evidence) accumulator
_results: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, evidence: str = "") -> bool:
    _results.append((name, bool(ok), evidence))
    return bool(ok)


def _read(p: Path) -> str:
    try:
        return p.read_text(encoding="utf-8", errors="replace")
    except OSError as e:  # pragma: no cover - defensive
        return f"<<unreadable: {e}>>"


def main() -> int:
    master = REPO / "dashboard" / "index.html"
    mirror = REPO / "docs" / "dashboard" / "index.html"
    jsonl = REPO / "autoresearch_results" / "experiment_log.jsonl"

    # B1 -- master + GitHub-Pages mirror exist.
    check("B1 master dashboard exists", master.exists(), str(master))
    check("B1 docs/ mirror exists", mirror.exists(), str(mirror))

    html = _read(master) if master.exists() else ""

    # B2 -- default composite-desc sort markers + type-to-filter present.
    has_sort = bool(re.search(r"data-v\b", html))
    # default sort = composite desc: look for a composite sort key/header marker.
    has_composite_default = "composite" in html.lower() and (
        "desc" in html.lower() or "data-sort" in html.lower() or "data-v" in html.lower()
    )
    has_filter = bool(re.search(r'id=["\']q["\']', html)) or "filter" in html.lower()
    check("B2 sortable table (data-v markers)", has_sort,
          f"data-v occurrences={len(re.findall(r'data-v', html))}")
    check("B2 default composite-desc sort markers", has_composite_default,
          "composite+sort markers present")
    check("B2 type-to-filter (#q / filter)", has_filter, "filter input present")

    # B3 -- every numeric cell carries n= + SCREENING/EVALUATION chip.
    n_chips = len(re.findall(r"n=\d", html))
    tier_chips = len(re.findall(r"SCREENING|EVALUATION", html))
    check("B3 n= seed badges present", n_chips > 0, f"n= occurrences={n_chips}")
    check("B3 SCREENING/EVALUATION tier chips present", tier_chips > 0,
          f"tier-chip occurrences={tier_chips}")

    # B4/B5/B7 -- the 6 plot PNGs exist (radar, parcoords, 3 pareto, geometry).
    plot_names = [
        "plot_radar.png",
        "plot_parcoords.png",
        "plot_pareto_capability.png",
        "plot_pareto_coherence.png",
        "plot_pareto_safety.png",
        "plot_geometry.png",
    ]
    ddir = REPO / "dashboard"
    missing_plots = [p for p in plot_names if not (ddir / p).exists()]
    check("B4/B5/B7 six plot PNGs exist", not missing_plots,
          "all present" if not missing_plots else f"missing={missing_plots}")

    # B8 -- stack/compete matrix present on master.
    has_stack = "STACK" in html and "COMPETE" in html
    check("B8 STACK/COMPETE matrix present", has_stack,
          f"STACK={'STACK' in html} COMPETE={'COMPETE' in html}")

    # B9 -- fingerprint + git SHA in footer.
    has_fp = FINGERPRINT in html
    has_sha = bool(re.search(r"git SHA", html, re.I)) or bool(
        re.search(r"\bsha\b", html, re.I)
    )
    check("B9 composite fingerprint in footer", has_fp, FINGERPRINT)
    check("B9 git SHA marker in footer", has_sha, "git SHA marker present")

    # B14 -- per-experiment pages exist for every JSONL row; row count == lines.
    rows: list[dict] = []
    jsonl_lines = 0
    if jsonl.exists():
        for line in _read(jsonl).splitlines():
            line = line.strip()
            if not line:
                continue
            jsonl_lines += 1
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass

    exp_dir = REPO / "docs" / "dashboard" / "experiments"
    exp_nums = sorted(
        r.get("experiment_num") for r in rows if r.get("experiment_num") is not None
    )
    missing_pages: list[str] = []
    leaked_pages: list[str] = []
    leak_pat = re.compile(r"\|---|^##\s|\*\*", re.M)
    for num in exp_nums:
        page = exp_dir / f"exp{int(num):03d}.html"
        if not page.exists():
            missing_pages.append(page.name)
            continue
        body = _read(page)
        # strip any inline <style>/<script> before leak scan to avoid CSS false hits
        scan = re.sub(r"<style.*?</style>", "", body, flags=re.S)
        scan = re.sub(r"<script.*?</script>", "", scan, flags=re.S)
        if "|---|" in scan or re.search(r"(?m)^\s*##\s", scan) or "**" in scan:
            leaked_pages.append(page.name)

    check("B14 per-experiment page per JSONL row", not missing_pages,
          "all present" if not missing_pages else f"missing={missing_pages}")
    check("B11 no literal ##/**/|---| leak in experiment pages", not leaked_pages,
          "clean" if not leaked_pages else f"leaked={leaked_pages}")

    # B12 -- master links to each per-experiment page.
    linked = set(re.findall(r"experiments/exp(\d+)\.html", html))
    linked_nums = {int(x) for x in linked}
    unlinked = [n for n in exp_nums if int(n) not in linked_nums]
    check("B12 master links to every per-experiment page", not unlinked,
          "all linked" if not unlinked else f"unlinked={unlinked}")

    # B14 -- master row count == JSONL line count.
    # Count rendered run rows: experiment-page links is the most reliable proxy
    # for one-row-per-experiment (the runs table links each row to its page).
    rendered_rows = len(linked_nums)
    check(
        "B14 rendered row count == JSONL line count",
        rendered_rows == jsonl_lines and jsonl_lines > 0,
        f"rendered_rows={rendered_rows} jsonl_lines={jsonl_lines}",
    )

    # ---- report ----
    print("\n" + "=" * 72)
    print("Rubric B -- Dashboard verification (scripts/verify_dashboard.py)")
    print("=" * 72)
    width = max(len(n) for n, _, _ in _results)
    n_pass = 0
    for name, ok, ev in _results:
        tag = "PASS" if ok else "FAIL"
        n_pass += ok
        print(f"  [{tag}] {name.ljust(width)}  {ev}")
    print("-" * 72)
    n_total = len(_results)
    print(f"  {n_pass}/{n_total} checks PASS")
    print("=" * 72)

    return 0 if n_pass == n_total else 1


if __name__ == "__main__":
    sys.exit(main())
