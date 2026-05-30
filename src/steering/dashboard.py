"""dashboard.py — first working self-contained runs dashboard.

Reads autoresearch_results/experiment_log.jsonl and emits a single
self-contained dashboard/index.html (mirrored to docs/dashboard/index.html):

  - sortable / type-to-filter runs table (one inline <script>, data-v attrs)
  - default sort = composite desc
  - global champion row highlighted
  - every numeric cell tagged with n=X + a SCREENING/EVALUATION chip
  - a 4-bullet "how to read" orientation block
  - composite_fingerprint + git SHA in the footer

Hard rules honoured (CLAUDE.md §11): self-contained HTML, no CDN / JS framework,
one inline script for sort/filter, no emoji. Richer panels (radar, Pareto,
ladder board) come later — this is the lean-but-real first version.

Usage:
    python -m steering.dashboard
"""

from __future__ import annotations

import html
import json
import subprocess
from pathlib import Path

from .eval import composite_fingerprint

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = REPO_ROOT / "autoresearch_results"

# Columns: (json key, header label, is_numeric)
COLUMNS = [
    ("experiment_num", "exp#", True),
    ("tag", "tag", False),
    ("rung", "rung", True),
    ("layer", "layer", True),
    ("operation", "op", False),
    ("source", "source", False),
    ("alpha", "alpha", True),
    ("composite", "composite", True),
    ("behavior_efficacy", "behavior", True),
    ("capability_retention", "capability", True),
    ("perplexity", "ppl", True),
    ("repetition_rate", "rep", True),
    ("compliance_rate", "safety_CR", True),
    ("selectivity_gap", "selectivity", True),
    ("offshell_displacement", "offshell", True),
    ("status", "status", False),
]


def _git_sha() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
            timeout=5,
        )
        sha = out.stdout.strip()
        return sha or "no-git"
    except Exception:
        return "no-git"


def load_rows(log_path: Path) -> list[dict]:
    rows: list[dict] = []
    if not log_path.exists():
        return rows
    with open(log_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def _flatten(row: dict) -> dict:
    """Lift nested config fields (tag/operation/source/alpha) to the top level."""
    flat = dict(row)
    cfg = row.get("config", {})
    for k in ("tag", "operation", "source", "alpha", "behavior", "model"):
        flat.setdefault(k, cfg.get(k))
    return flat


def _cell_value(row: dict, key: str):
    v = row.get(key)
    return v


def render_html(rows: list[dict]) -> str:
    flat_rows = [_flatten(r) for r in rows]

    # Champion = max composite.
    champ_num = None
    if flat_rows:
        champ = max(flat_rows, key=lambda r: r.get("composite", -1e18))
        champ_num = champ.get("experiment_num")

    fp = composite_fingerprint()
    sha = _git_sha()
    n_runs = len(flat_rows)

    # Build table header.
    head_cells = "".join(
        f'<th data-key="{html.escape(k)}" data-num="{int(num)}" '
        f'onclick="sortBy(\'{html.escape(k)}\',{int(num)})">{html.escape(label)}'
        f'<span class="arr"></span></th>'
        for (k, label, num) in COLUMNS
    )

    # Build table body.
    body_rows = []
    for r in flat_rows:
        n_seeds = r.get("n_seeds", 1)
        tier = r.get("tier", "SCREENING")
        is_champ = r.get("experiment_num") == champ_num
        tr_cls = "champ" if is_champ else ""
        tds = []
        for (k, _label, num) in COLUMNS:
            val = _cell_value(r, k)
            if num and isinstance(val, (int, float)):
                disp = f"{val:.4f}" if isinstance(val, float) else f"{val}"
                chip = (
                    f'<span class="chip {tier.lower()}">{html.escape(tier)} '
                    f'n={n_seeds}</span>'
                )
                cell = f'{html.escape(disp)} {chip}'
                sort_val = float(val)
            else:
                disp = "" if val is None else str(val)
                cell = html.escape(disp)
                sort_val = disp.lower()
            tds.append(
                f'<td data-v="{html.escape(str(sort_val))}">{cell}</td>'
            )
        body_rows.append(f'<tr class="{tr_cls}">{"".join(tds)}</tr>')

    body_html = "\n".join(body_rows) if body_rows else (
        '<tr><td colspan="%d" class="empty">No experiments logged yet. '
        'Run: python -m steering.runner --model fake --rung 0 '
        '--description "first run" --tag smoke</td></tr>' % len(COLUMNS)
    )

    champ_label = f"exp#{champ_num}" if champ_num is not None else "none yet"

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Steering Autoresearch — Master Dashboard</title>
<style>
  :root {{ --bg:#0f1115; --fg:#e6e6e6; --mut:#9aa0a6; --line:#2a2e35;
           --champ:#1e3a24; --accent:#4ea1ff; }}
  * {{ box-sizing:border-box; }}
  body {{ background:var(--bg); color:var(--fg); margin:0;
          font:14px/1.45 -apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif; }}
  header {{ padding:18px 22px; border-bottom:1px solid var(--line); }}
  h1 {{ font-size:18px; margin:0 0 4px; }}
  .sub {{ color:var(--mut); font-size:12px; }}
  .howto {{ margin:16px 22px; padding:12px 16px; border:1px solid var(--line);
            border-radius:8px; background:#13161c; }}
  .howto h2 {{ font-size:13px; margin:0 0 8px; color:var(--accent);
               text-transform:uppercase; letter-spacing:.04em; }}
  .howto ul {{ margin:0; padding-left:18px; }}
  .howto li {{ margin:3px 0; color:var(--fg); }}
  .controls {{ margin:0 22px 10px; }}
  input[type=text] {{ width:340px; max-width:60vw; padding:7px 10px;
       background:#13161c; border:1px solid var(--line); color:var(--fg);
       border-radius:6px; }}
  .tablewrap {{ margin:0 22px 22px; overflow-x:auto; }}
  table {{ border-collapse:collapse; width:100%; font-size:13px; }}
  th, td {{ text-align:left; padding:7px 9px; border-bottom:1px solid var(--line);
            white-space:nowrap; }}
  th {{ cursor:pointer; user-select:none; position:sticky; top:0;
        background:#13161c; color:var(--mut); font-weight:600; }}
  th:hover {{ color:var(--fg); }}
  tr.champ {{ background:var(--champ); }}
  tr.champ td:first-child::before {{ content:"\\2605 "; color:#ffd166; }}
  td.empty {{ color:var(--mut); text-align:center; padding:24px; }}
  .chip {{ display:inline-block; font-size:9px; padding:1px 5px; border-radius:8px;
           vertical-align:middle; margin-left:4px; }}
  .chip.screening {{ background:#3a2e13; color:#e0b15a; }}
  .chip.evaluation {{ background:#13303a; color:#5ac8e0; }}
  .arr {{ font-size:10px; color:var(--accent); margin-left:3px; }}
  footer {{ padding:14px 22px; border-top:1px solid var(--line); color:var(--mut);
            font-size:12px; }}
  code {{ color:var(--accent); }}
</style>
</head>
<body>
<header>
  <h1>Steering Autoresearch — Master Dashboard</h1>
  <div class="sub">{n_runs} run(s) logged &middot; global champion: {champ_label}
   &middot; default sort: composite desc</div>
</header>

<div class="howto">
  <h2>How to read this dashboard</h2>
  <ul>
    <li><b>Composite is multi-objective.</b> It prices all five axes (behavior,
        capability, coherence, safety, selectivity) plus an off-manifold geometry
        penalty — a method cannot win by sacrificing one axis (CLAUDE.md &sect;6).</li>
    <li><b>Tiers matter.</b> Every numeric cell carries <code>n=X</code> and a
        SCREENING (n&le;3) or EVALUATION (n&ge;7) chip. Only EVALUATION rows with
        the rigor contract may be called a "winner" (CLAUDE.md &sect;7).</li>
    <li><b>Safety is a hard gate.</b> A non-zero <code>safety_CR</code>
        (JailbreakBench compliance) is a Rogue-Scalpel leak and an automatic
        DISCARD regardless of behavior score (CLAUDE.md &sect;10).</li>
    <li><b>Click a header to sort; type to filter.</b> The starred green row is the
        current global champion (highest composite).</li>
  </ul>
</div>

<div class="controls">
  <input type="text" id="filter" placeholder="type to filter rows (tag, op, status, ...)"
         oninput="applyFilter()">
</div>

<div class="tablewrap">
  <table id="runs">
    <thead><tr>{head_cells}</tr></thead>
    <tbody>
{body_html}
    </tbody>
  </table>
</div>

<footer>
  composite fingerprint: <code>{fp}</code> &middot; git SHA: <code>{sha}</code>
  &middot; Internal QA pass — external review pending.
</footer>

<script>
(function() {{
  var sortState = {{ key: "composite", num: 1, dir: -1 }};

  window.sortBy = function(key, num) {{
    if (sortState.key === key) {{ sortState.dir *= -1; }}
    else {{ sortState.key = key; sortState.num = num; sortState.dir = num ? -1 : 1; }}
    render();
  }};

  function getRows() {{
    var tbody = document.querySelector("#runs tbody");
    return Array.prototype.slice.call(tbody.querySelectorAll("tr"));
  }}

  function colIndex(key) {{
    var ths = document.querySelectorAll("#runs thead th");
    for (var i = 0; i < ths.length; i++) {{
      if (ths[i].getAttribute("data-key") === key) return i;
    }}
    return 0;
  }}

  function render() {{
    var rows = getRows();
    var idx = colIndex(sortState.key);
    rows.sort(function(a, b) {{
      var ca = a.children[idx], cb = b.children[idx];
      if (!ca || !cb) return 0;
      var va = ca.getAttribute("data-v"), vb = cb.getAttribute("data-v");
      if (sortState.num) {{ va = parseFloat(va) || 0; vb = parseFloat(vb) || 0; }}
      if (va < vb) return -1 * sortState.dir;
      if (va > vb) return 1 * sortState.dir;
      return 0;
    }});
    var tbody = document.querySelector("#runs tbody");
    rows.forEach(function(r) {{ tbody.appendChild(r); }});
    document.querySelectorAll("#runs thead .arr").forEach(function(s) {{ s.textContent = ""; }});
    var th = document.querySelector('#runs thead th[data-key="' + sortState.key + '"]');
    if (th) th.querySelector(".arr").textContent = sortState.dir < 0 ? "\\u25BC" : "\\u25B2";
  }}

  window.applyFilter = function() {{
    var q = document.getElementById("filter").value.toLowerCase();
    getRows().forEach(function(r) {{
      r.style.display = r.textContent.toLowerCase().indexOf(q) >= 0 ? "" : "none";
    }});
  }};

  render();
}})();
</script>
</body>
</html>
"""


def build_dashboard(results_dir: Path | None = None) -> Path:
    """Generate dashboard/index.html and mirror to docs/dashboard/index.html.

    Returns the primary output path.
    """
    results_dir = results_dir or RESULTS_DIR
    log_path = results_dir / "experiment_log.jsonl"
    rows = load_rows(log_path)
    out_html = render_html(rows)

    primary = REPO_ROOT / "dashboard" / "index.html"
    mirror = REPO_ROOT / "docs" / "dashboard" / "index.html"
    for path in (primary, mirror):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(out_html, encoding="utf-8")
    return primary


def main() -> None:
    out = build_dashboard()
    print(f"Dashboard written: {out}")
    print(f"Mirror: {REPO_ROOT / 'docs' / 'dashboard' / 'index.html'}")


if __name__ == "__main__":
    main()
