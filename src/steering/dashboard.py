"""dashboard.py — the RICH, hierarchically-linked autoresearch dashboard.

Three linked tiers (CLAUDE.md §11 "The Dashboard Mandate"):

  A. MASTER     dashboard/index.html  (mirror docs/dashboard/index.html)
  B. PER-HYPO   ideas/<dir>/dashboard/index.html
                (mirror docs/dashboard/hyp/<id>.html)
  C. PER-EXP    docs/dashboard/experiments/expNNN.html
                (also dashboard/experiments/expNNN.html)

Master panels:
  - "How to read this" 4-bullet orientation block.
  - Sortable / type-to-filter runs table (default sort composite desc;
    champion row highlighted; every numeric cell carries n=X + a
    SCREENING/EVALUATION tier chip; per-row links to the per-experiment page
    and, when resolvable, the per-hypothesis sub-dashboard).
  - 5-axis radar / parallel-coordinates PNG per method.
  - Pareto PNGs: behavior-vs-capability, behavior-vs-coherence,
    behavior-vs-safety (baseline/prior rows drawn as star markers; dominated
    rows outlined).
  - Ladder board: per method highest rung + gate cleared/failed + reason.
  - Stack/compete matrix: the pairwise method-composability prior transcribed
    from corpus/steering-stackable-vs-competing-analysis.md §4 (STACK/CARE/
    COMPETE cells), with any measured pair-verdict overlaid.
  - Geometry small-multiples: dNorm, effective-rank-drop, norm-budget.
  - Footer: COMPOSITE_FORMULA + composite_fingerprint() + git SHA + timestamp.

Per-hypothesis sub-dashboard:
  - Best-config callout; hypothesis statement + falsifier + predicted delta
    (parsed from ideas/<dir>/IDEA.md or README.md); scoped cells table linking
    to per-experiment pages; coordinate-descent small-multiples (when sweep data
    exists); seed-stability bars; back-link to master.

Per-experiment page:
  - The full 7-step reasoning entry rendered from reasoning_annotations.json
    markdown -> HTML via a small GFM-ish converter (no literal ## / ** / |---|
    leaks); the config; all five axis metrics + geometry probes with any CIs;
    a sweep curve (behavior + PPL vs the swept alpha/layer axis for the
    hypothesis group; single-point fallback when only one row exists); a
    side-by-side steered-vs-unsteered sample section (consumed from whatever
    sample_steered / sample_unsteered / samples keys the runner logged, with a
    ready placeholder when absent); back-links to hypothesis + master.

Hard rules honoured: self-contained HTML, no CDN / JS framework, one inline
sort/filter script using data-v attributes, PNG (not SVG) plots, no emoji.

Usage:
    python -m steering.dashboard
"""

from __future__ import annotations

import html
import json
import math
import re
import subprocess
import time
from pathlib import Path
from typing import Optional

from .eval import COMPOSITE_FORMULA, COMPOSITE_WEIGHTS, composite_fingerprint

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = REPO_ROOT / "autoresearch_results"

# ---------------------------------------------------------------------------
# Shared CSS palette (identical across master / sub / per-experiment tiers).
# ---------------------------------------------------------------------------
SHARED_CSS = """
  :root { --bg:#0f1115; --fg:#e6e6e6; --mut:#9aa0a6; --line:#2a2e35;
          --champ:#1e3a24; --accent:#4ea1ff; --card:#13161c; }
  * { box-sizing:border-box; }
  body { background:var(--bg); color:var(--fg); margin:0;
         font:14px/1.5 -apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif; }
  a { color:var(--accent); }
  header { padding:18px 22px; border-bottom:1px solid var(--line); }
  h1 { font-size:18px; margin:0 0 4px; }
  h2 { font-size:14px; margin:0 0 8px; color:var(--accent);
       text-transform:uppercase; letter-spacing:.04em; }
  h3 { font-size:13px; margin:18px 0 6px; }
  .sub { color:var(--mut); font-size:12px; }
  .nav { font-size:12px; margin-bottom:6px; }
  section { margin:18px 22px; padding:14px 16px; border:1px solid var(--line);
            border-radius:8px; background:var(--card); }
  .howto ul { margin:0; padding-left:18px; }
  .howto li { margin:3px 0; }
  input[type=text] { width:340px; max-width:60vw; padding:7px 10px;
       background:var(--bg); border:1px solid var(--line); color:var(--fg);
       border-radius:6px; }
  .tablewrap { overflow-x:auto; }
  table { border-collapse:collapse; width:100%; font-size:13px; }
  th, td { text-align:left; padding:7px 9px; border-bottom:1px solid var(--line);
           white-space:nowrap; vertical-align:top; }
  th { cursor:pointer; user-select:none; position:sticky; top:0;
       background:var(--card); color:var(--mut); font-weight:600; }
  th:hover { color:var(--fg); }
  tr.champion { background:var(--champ); }
  tr.champion td:first-child::before { content:"\\2605 "; color:#ffd166; }
  tr.discard td { opacity:.85; }
  td.empty { color:var(--mut); text-align:center; padding:24px; }
  .chip { display:inline-block; font-size:9px; padding:1px 6px; border-radius:8px;
          vertical-align:middle; margin-left:4px; }
  .chip.screening { background:#3a2e13; color:#e0b15a; }
  .chip.evaluation { background:#13303a; color:#5ac8e0; }
  .seed-badge { display:inline-block; font-size:9px; padding:1px 5px;
       border-radius:8px; background:#1b2330; color:#9fc0ff; margin-left:3px; }
  .arr { font-size:10px; color:var(--accent); margin-left:3px; }
  .verdict { font-weight:600; }
  .verdict.keep { color:#6fd08c; }
  .verdict.discard { color:#e0795a; }
  .verdict.near-miss { color:#e0c15a; }
  .callout { border-left:3px solid var(--accent); padding:8px 12px;
             background:#11202c; border-radius:4px; margin:8px 0; }
  .tag-pill { display:inline-block; font-size:11px; padding:2px 8px;
       border-radius:10px; background:#1b2330; color:#9fc0ff; margin:2px; }
  img.plot { max-width:100%; border:1px solid var(--line); border-radius:6px;
             background:#fff; margin:6px 0; }
  .grid { display:flex; flex-wrap:wrap; gap:14px; }
  .grid .cell { flex:1 1 280px; min-width:260px; }
  .cap { color:var(--mut); font-size:11px; margin:2px 0 10px; }
  .md h1,.md h2,.md h3 { color:var(--fg); text-transform:none; letter-spacing:0; }
  .md table { width:auto; margin:8px 0; }
  .md code { color:#9fe0a0; background:#0b1a12; padding:1px 4px; border-radius:3px; }
  .md pre { background:#0b1015; border:1px solid var(--line); padding:8px;
            border-radius:6px; overflow-x:auto; }
  .md blockquote { border-left:3px solid var(--line); margin:6px 0;
            padding:2px 10px; color:var(--mut); }
  .warn { color:#e0795a; font-style:italic; }
  table.stackmx td.sc { text-align:center; font-weight:600; font-size:11px;
       white-space:nowrap; }
  table.stackmx th { white-space:nowrap; }
  .sc.stack { background:#16331f; color:#7fe0a0; }
  .sc.care { background:#3a3010; color:#e6c25a; }
  .sc.compete { background:#3a1616; color:#f08080; }
  .sc.self { background:#1a1d24; color:var(--mut); }
  .sc.measured { outline:2px solid #fff; outline-offset:-2px; }
  .sc .meas { font-size:9px; color:#fff; font-weight:700; }
  .sc.scl { display:inline-block; padding:1px 6px; border-radius:4px;
       font-weight:600; }
  .grid.samples .cell { flex:1 1 340px; }
  pre.sample { background:#0b1015; border:1px solid var(--line); padding:8px;
       border-radius:6px; white-space:pre-wrap; word-break:break-word;
       max-height:340px; overflow:auto; font-size:12px; }
  footer { padding:14px 22px; border-top:1px solid var(--line); color:var(--mut);
           font-size:12px; }
  code { color:var(--accent); }
"""

# Master runs-table columns the mandate names explicitly:
# exp# | tag | hypothesis | rung | behavior | dMMLU | PPL | CR_jail |
#       over_refusal | composite | verdict | n + chip | link
MASTER_COLUMNS = [
    ("experiment_num", "exp#", True),
    ("tag", "tag", False),
    ("hypothesis", "hypothesis", False),
    ("rung", "rung", True),
    ("behavior", "behavior", False),
    ("mmlu_drop_pp", "dMMLU", True),
    ("perplexity", "PPL", True),
    ("compliance_rate", "CR_jail", True),
    ("harmless_refusal_rate", "over_refusal", True),
    ("composite", "composite", True),
    ("status", "verdict", False),
]


# ===========================================================================
# I/O helpers
# ===========================================================================
def _git_sha() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(REPO_ROOT), capture_output=True, text=True, timeout=5,
        )
        return out.stdout.strip() or "no-git"
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


def load_reasoning(results_dir: Path) -> dict:
    path = results_dir / "reasoning_annotations.json"
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _flatten(row: dict) -> dict:
    """Lift nested config fields to the top level for table rendering."""
    flat = dict(row)
    cfg = row.get("config", {}) or {}
    for k in ("tag", "operation", "source", "alpha", "behavior", "model",
              "hypothesis_id"):
        if flat.get(k) is None:
            flat[k] = cfg.get(k)
    return flat


def _tier_of(row: dict) -> str:
    """n<=3 => SCREENING ; n>=7 => EVALUATION (4..6 stays SCREENING per §7)."""
    explicit = row.get("tier")
    n = int(row.get("n_seeds", 1) or 1)
    if explicit in ("SCREENING", "EVALUATION"):
        return explicit
    return "EVALUATION" if n >= 7 else "SCREENING"


def _verdict_class(status: Optional[str]) -> str:
    s = (status or "").upper()
    if s == "KEEP":
        return "keep"
    if s == "DISCARD":
        return "discard"
    if "MISS" in s:
        return "near-miss"
    return ""


# ===========================================================================
# Hypothesis directory resolution (row <-> ideas/<dir>)
# ===========================================================================
def list_idea_dirs(repo_root: Path) -> list[Path]:
    ideas = repo_root / "ideas"
    if not ideas.exists():
        return []
    out = []
    for d in sorted(ideas.iterdir()):
        if d.is_dir() and not d.name.startswith("_"):
            out.append(d)
    return out


def _idea_id(idea_dir: Path) -> str:
    """The leading numeric id of an ideas dir, e.g. '10' from '10_foo_bar'."""
    m = re.match(r"(\d+)", idea_dir.name)
    return m.group(1) if m else idea_dir.name


def resolve_hypothesis_dir(row: dict, idea_dirs: list[Path]) -> Optional[Path]:
    """Map an experiment row to its ideas/<dir>, if one can be resolved.

    Priority:
      1. explicit row/config 'hypothesis_id' (matches the dir's numeric id or
         a token in the dir name).
      2. tag token overlap with the dir name slug.
    """
    if not idea_dirs:
        return None
    hyp = row.get("hypothesis_id") or (row.get("config", {}) or {}).get("hypothesis_id")
    tag = (row.get("tag") or (row.get("config", {}) or {}).get("tag") or "")
    tag = str(tag).lower()

    if hyp is not None:
        h = str(hyp).lower()
        for d in idea_dirs:
            if _idea_id(d) == h.lstrip("h").lstrip("e"):
                return d
            slug = d.name.lower()
            if h in slug or h.lstrip("h").lstrip("e") == _idea_id(d):
                return d

    # token overlap heuristic on the tag.
    tag_tokens = set(re.split(r"[^a-z0-9]+", tag)) - {""}
    best, best_score = None, 0
    for d in idea_dirs:
        slug_tokens = set(re.split(r"[^a-z0-9]+", d.name.lower())) - {""}
        # drop the leading numeric id token from the comparison
        slug_tokens = {t for t in slug_tokens if not t.isdigit()}
        score = len(tag_tokens & slug_tokens)
        if score > best_score:
            best, best_score = d, score
    return best if best_score > 0 else None


def hypothesis_label(row: dict, idea_dirs: list[Path]) -> str:
    d = resolve_hypothesis_dir(row, idea_dirs)
    if d is not None:
        return f"H{_idea_id(d)}"
    hyp = row.get("hypothesis_id") or (row.get("config", {}) or {}).get("hypothesis_id")
    return str(hyp) if hyp else "—"


# ===========================================================================
# Minimal GFM-ish markdown -> HTML converter (NO literal ## / ** / |---| leak)
# ===========================================================================
def _md_inline(text: str) -> str:
    """Inline markdown: escape, then bold / italic / code / links."""
    out = html.escape(text)
    # inline code first (protect its contents from further formatting)
    code_spans: list[str] = []

    def _stash_code(m):
        code_spans.append(m.group(1))
        return f"\x00CODE{len(code_spans) - 1}\x00"

    out = re.sub(r"`([^`]+)`", _stash_code, out)
    # links [text](url)
    out = re.sub(r"\[([^\]]+)\]\(([^)]+)\)",
                 lambda m: f'<a href="{html.escape(m.group(2), quote=True)}">{m.group(1)}</a>',
                 out)
    # bold **x** or __x__
    out = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", out)
    out = re.sub(r"__([^_]+)__", r"<strong>\1</strong>", out)
    # italic *x* or _x_ (avoid touching remaining asterisks in isolation)
    out = re.sub(r"(?<![\w*])\*([^*\n]+)\*(?![\w*])", r"<em>\1</em>", out)
    out = re.sub(r"(?<![\w_])_([^_\n]+)_(?![\w_])", r"<em>\1</em>", out)
    # restore code spans
    for i, c in enumerate(code_spans):
        out = out.replace(f"\x00CODE{i}\x00", f"<code>{c}</code>")
    return out


def md_to_html(md: str) -> str:
    """Convert a GFM-ish markdown fragment to HTML.

    Supports: ATX headings, bold/italic/inline-code, fenced code blocks,
    blockquotes, unordered/ordered lists, and pipe tables (incl. the
    |---|---| separator row, which is consumed — never leaked).
    """
    if not md:
        return ""
    md = md.replace("\r\n", "\n").replace("\r", "\n")
    lines = md.split("\n")
    out: list[str] = []
    i = 0
    n = len(lines)

    def close_list(stack):
        while stack:
            out.append(f"</{stack.pop()}>")

    list_stack: list[str] = []

    while i < n:
        line = lines[i]

        # fenced code block
        if line.strip().startswith("```"):
            close_list(list_stack)
            i += 1
            buf = []
            while i < n and not lines[i].strip().startswith("```"):
                buf.append(html.escape(lines[i]))
                i += 1
            i += 1  # skip closing fence
            out.append("<pre><code>" + "\n".join(buf) + "</code></pre>")
            continue

        # table: a line with pipes followed by a separator row of ---
        if "|" in line and i + 1 < n and re.match(r"^\s*\|?[\s:|-]*-[-\s:|]*\|?\s*$", lines[i + 1]) and "-" in lines[i + 1]:
            close_list(list_stack)
            header = [c.strip() for c in line.strip().strip("|").split("|")]
            i += 2  # skip header + separator (separator is consumed, not leaked)
            rows = []
            while i < n and "|" in lines[i] and lines[i].strip():
                cells = [c.strip() for c in lines[i].strip().strip("|").split("|")]
                rows.append(cells)
                i += 1
            thead = "".join(f"<th>{_md_inline(h)}</th>" for h in header)
            body = ""
            for r in rows:
                body += "<tr>" + "".join(f"<td>{_md_inline(c)}</td>" for c in r) + "</tr>"
            out.append(f"<table><thead><tr>{thead}</tr></thead><tbody>{body}</tbody></table>")
            continue

        # heading
        m = re.match(r"^(#{1,6})\s+(.*)$", line)
        if m:
            close_list(list_stack)
            lvl = min(len(m.group(1)) + 1, 6)  # bump so page h1 stays unique
            out.append(f"<h{lvl}>{_md_inline(m.group(2).strip())}</h{lvl}>")
            i += 1
            continue

        # blockquote
        if line.strip().startswith(">"):
            close_list(list_stack)
            buf = []
            while i < n and lines[i].strip().startswith(">"):
                buf.append(_md_inline(re.sub(r"^\s*>\s?", "", lines[i])))
                i += 1
            out.append("<blockquote>" + "<br>".join(buf) + "</blockquote>")
            continue

        # unordered list
        m = re.match(r"^\s*[-*+]\s+(.*)$", line)
        if m:
            if not list_stack or list_stack[-1] != "ul":
                close_list(list_stack)
                list_stack.append("ul")
                out.append("<ul>")
            out.append(f"<li>{_md_inline(m.group(1))}</li>")
            i += 1
            continue

        # ordered list
        m = re.match(r"^\s*\d+[.)]\s+(.*)$", line)
        if m:
            if not list_stack or list_stack[-1] != "ol":
                close_list(list_stack)
                list_stack.append("ol")
                out.append("<ol>")
            out.append(f"<li>{_md_inline(m.group(1))}</li>")
            i += 1
            continue

        # blank line
        if not line.strip():
            close_list(list_stack)
            i += 1
            continue

        # paragraph (gather consecutive plain lines)
        close_list(list_stack)
        buf = [line]
        i += 1
        while i < n and lines[i].strip() and not re.match(
            r"^\s*([-*+]\s|\d+[.)]\s|#{1,6}\s|>|```)", lines[i]
        ) and "|" not in lines[i]:
            buf.append(lines[i])
            i += 1
        out.append("<p>" + _md_inline(" ".join(buf)) + "</p>")

    close_list(list_stack)
    return "\n".join(out)


# ===========================================================================
# matplotlib PNG plots (lazy import; degrade gracefully if unavailable)
# ===========================================================================
def _mpl():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        return plt
    except Exception:
        return None


def _num(row: dict, key: str, default: float = 0.0) -> float:
    v = row.get(key)
    try:
        x = float(v)  # type: ignore[arg-type]  # guarded below + by except
    except (TypeError, ValueError):
        return default
    # NaN/inf would silently poison radar/Pareto/composite-breakdown; clamp them.
    if not math.isfinite(x):
        return default
    return x


def _axis_scores(row: dict) -> dict:
    """Five axes normalised to a 0..1 'higher=better' radar coordinate."""
    behavior = max(0.0, min(1.0, _num(row, "behavior_efficacy")))
    capability = max(0.0, min(1.0, 1.0 - _num(row, "mmlu_drop_pp")))
    coherence = max(0.0, min(1.0, 1.0 / (1.0 + _num(row, "dppl_norm")) - _num(row, "repetition_rate")))
    safety = max(0.0, min(1.0, 1.0 - _num(row, "compliance_rate")))
    selectivity = max(0.0, min(1.0, 0.5 + 0.5 * _num(row, "selectivity_gap")))
    return {
        "behavior": behavior, "capability": capability, "coherence": coherence,
        "safety": safety, "selectivity": selectivity,
    }


def plot_radar(rows: list[dict], out_path: Path) -> bool:
    plt = _mpl()
    if plt is None or not rows:
        return False
    import math
    axes = ["behavior", "capability", "coherence", "safety", "selectivity"]
    angles = [n / len(axes) * 2 * math.pi for n in range(len(axes))]
    angles += angles[:1]
    fig = plt.figure(figsize=(5.2, 4.2))
    ax = fig.add_subplot(111, polar=True)
    cmap = plt.get_cmap("tab10")
    for idx, r in enumerate(rows[:8]):
        sc = _axis_scores(r)
        vals = [sc[a] for a in axes]
        vals += vals[:1]
        label = str((r.get("config", {}) or {}).get("tag") or r.get("tag")
                    or f"exp{r.get('experiment_num')}")
        ax.plot(angles, vals, linewidth=1.3, label=label, color=cmap(idx % 10))
        ax.fill(angles, vals, alpha=0.07, color=cmap(idx % 10))
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(axes, fontsize=8)
    ax.set_ylim(0, 1)
    ax.set_title("5-axis profile per method (1=best on each axis)", fontsize=9)
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.1), fontsize=6)
    fig.tight_layout()
    fig.savefig(out_path, dpi=90, bbox_inches="tight")
    plt.close(fig)
    return True


def plot_parcoords(rows: list[dict], out_path: Path) -> bool:
    plt = _mpl()
    if plt is None or not rows:
        return False
    axes = ["behavior", "capability", "coherence", "safety", "selectivity"]
    fig, ax = plt.subplots(figsize=(6.2, 3.4))
    cmap = plt.get_cmap("tab10")
    xs = list(range(len(axes)))
    for idx, r in enumerate(rows[:8]):
        sc = _axis_scores(r)
        ys = [sc[a] for a in axes]
        label = str((r.get("config", {}) or {}).get("tag") or r.get("tag")
                    or f"exp{r.get('experiment_num')}")
        ax.plot(xs, ys, marker="o", markersize=3, linewidth=1.1,
                color=cmap(idx % 10), label=label)
    ax.set_xticks(xs)
    ax.set_xticklabels(axes, fontsize=8)
    ax.set_ylim(0, 1)
    ax.set_ylabel("normalised score (1=best)", fontsize=8)
    ax.set_title("Parallel coordinates — five axes", fontsize=9)
    ax.legend(fontsize=6, loc="lower left")
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=90)
    plt.close(fig)
    return True


def _is_baseline(row: dict) -> bool:
    tag = str((row.get("config", {}) or {}).get("tag") or row.get("tag") or "").lower()
    return ("baseline" in tag or "prior" in tag or bool(row.get("is_baseline"))
            or int(row.get("rung", 0) or 0) == 0 and "smoke" in tag)


def plot_pareto(rows: list[dict], x_key: str, x_label: str, out_path: Path,
                x_transform=None) -> bool:
    """behavior_efficacy (y) vs a cost/constraint axis (x). Stars = baselines.

    Dominated rows (some other row is >= on behavior AND better on the cost
    axis) are drawn with a red outline.
    """
    plt = _mpl()
    if plt is None or not rows:
        return False
    pts = []
    for r in rows:
        y = _num(r, "behavior_efficacy")
        x = _num(r, x_key)
        if x_transform:
            x = x_transform(x)
        pts.append((x, y, r))
    # dominance: a point is dominated if another has y>=y and x<=x (lower cost
    # better) with at least one strict.
    dominated = []
    for (xi, yi, ri) in pts:
        dom = False
        for (xj, yj, rj) in pts:
            if ri is rj:
                continue
            if yj >= yi and xj <= xi and (yj > yi or xj < xi):
                dom = True
                break
        dominated.append(dom)

    fig, ax = plt.subplots(figsize=(4.6, 3.4))
    for (xi, yi, ri), dom in zip(pts, dominated):
        if _is_baseline(ri):
            ax.scatter([xi], [yi], marker="*", s=160, color="#f0a000",
                       edgecolors="#333", zorder=3)
        else:
            ax.scatter([xi], [yi], s=55,
                       color=("#e05a5a" if dom else "#4ea1ff"),
                       edgecolors=("#ff0000" if dom else "#1b3a5a"),
                       linewidths=(1.6 if dom else 0.8), zorder=2)
        ax.annotate(str(ri.get("experiment_num", "")), (xi, yi),
                    fontsize=6, xytext=(3, 3), textcoords="offset points")
    ax.set_xlabel(x_label, fontsize=8)
    ax.set_ylabel("behavior_efficacy", fontsize=8)
    ax.set_title(f"Pareto: behavior vs {x_label}", fontsize=9)
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=90)
    plt.close(fig)
    return True


def plot_geometry(rows: list[dict], out_path: Path) -> bool:
    """Small-multiples: dNorm (offshell), effective-rank-drop, norm-budget."""
    plt = _mpl()
    if plt is None or not rows:
        return False
    xs = [str(r.get("experiment_num", i)) for i, r in enumerate(rows)]
    offshell = [_num(r, "offshell_displacement") for r in rows]
    erank_drop = [
        _num(r, "effective_rank_base") - _num(r, "effective_rank_steer")
        if ("effective_rank_base" in r or "effective_rank_steer" in r) else 0.0
        for r in rows
    ]
    nbudget = [_num(r, "norm_budget") for r in rows]
    series = [
        (offshell, "off-shell displacement Δ‖h‖"),
        (erank_drop, "effective-rank drop"),
        (nbudget, "norm budget ‖Δh‖/‖h‖"),
    ]
    fig, axs = plt.subplots(1, 3, figsize=(8.4, 2.8))
    for ax, (vals, title) in zip(axs, series):
        ax.bar(xs, vals, color="#4ea1ff")
        ax.set_title(title, fontsize=8)
        ax.tick_params(axis="x", labelsize=6, rotation=0)
        ax.tick_params(axis="y", labelsize=6)
        ax.grid(True, axis="y", alpha=0.2)
    fig.suptitle("Geometry probes per experiment", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=90)
    plt.close(fig)
    return True


def plot_seed_stability(rows: list[dict], out_path: Path) -> bool:
    """Per-experiment composite bars (seed-stability surrogate when n=1)."""
    plt = _mpl()
    if plt is None or not rows:
        return False
    xs = [str(r.get("experiment_num", i)) for i, r in enumerate(rows)]
    ys = [_num(r, "composite") for r in rows]
    fig, ax = plt.subplots(figsize=(5.0, 2.8))
    ax.bar(xs, ys, color="#6fd08c")
    if ys:
        mean = sum(ys) / len(ys)
        ax.axhline(mean, color="#e0c15a", linestyle="--", linewidth=1,
                   label=f"mean {mean:.3f}")
        ax.legend(fontsize=7)
    ax.set_title("Composite per run (seed-stability)", fontsize=9)
    ax.tick_params(labelsize=7)
    ax.grid(True, axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=90)
    plt.close(fig)
    return True


def plot_coord_descent(rows: list[dict], out_path: Path) -> bool:
    """Coordinate-descent small-multiples: composite vs layer / alpha."""
    plt = _mpl()
    if plt is None or len(rows) < 2:
        return False

    def _cfg(r, k):
        return (r.get("config", {}) or {}).get(k, r.get(k))

    fig, axs = plt.subplots(1, 2, figsize=(6.6, 2.8))
    for ax, key, label in ((axs[0], "layer", "injection layer"),
                           (axs[1], "alpha", "alpha")):
        pts = []
        for r in rows:
            v = _cfg(r, key)
            try:
                pts.append((float(v), _num(r, "composite")))
            except (TypeError, ValueError):
                continue
        pts.sort()
        if pts:
            ax.plot([p[0] for p in pts], [p[1] for p in pts],
                    marker="o", color="#4ea1ff")
        ax.set_xlabel(label, fontsize=8)
        ax.set_ylabel("composite", fontsize=8)
        ax.set_title(f"composite vs {label}", fontsize=8)
        ax.grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=90)
    plt.close(fig)
    return True


def _sweep_axis(rows: list[dict]) -> tuple[str, list[float]]:
    """Pick the swept axis ('alpha' or 'layer') for a group of rows.

    Returns (axis_name, distinct_values). The axis with more distinct logged
    values wins; ties prefer alpha. Empty values => single-point fallback.
    """
    def _cfg(r, k):
        return (r.get("config", {}) or {}).get(k, r.get(k))

    cand = {}
    for key in ("alpha", "layer"):
        vals = []
        for r in rows:
            try:
                vals.append(float(_cfg(r, key)))
            except (TypeError, ValueError):
                continue
        cand[key] = vals
    n_alpha = len(set(cand["alpha"]))
    n_layer = len(set(cand["layer"]))
    if n_alpha >= n_layer and n_alpha >= 1:
        return "alpha", cand["alpha"]
    if n_layer >= 1:
        return "layer", cand["layer"]
    return "alpha", cand["alpha"]


def plot_sweep(rows: list[dict], out_path: Path) -> tuple[bool, bool]:
    """Per-experiment sweep curve: behavior + PPL vs the swept axis (dual-y).

    Groups the supplied sibling rows (already filtered to one hypothesis /
    behavior) by the swept axis (alpha or layer). Returns
    (rendered_ok, is_single_point). When only one point exists, a single-point
    plot is drawn with a note that the sweep accumulates as more alpha/layer
    rows are logged.
    """
    plt = _mpl()
    if plt is None or not rows:
        return (False, False)

    def _cfg(r, k):
        return (r.get("config", {}) or {}).get(k, r.get(k))

    axis, _ = _sweep_axis(rows)
    pts = []
    for r in rows:
        try:
            x = float(_cfg(r, axis))
        except (TypeError, ValueError):
            continue
        pts.append((x, _num(r, "behavior_efficacy"), _num(r, "perplexity")))
    if not pts:
        return (False, False)
    pts.sort()
    xs = [p[0] for p in pts]
    beh = [p[1] for p in pts]
    ppl = [p[2] for p in pts]
    single = len(set(xs)) < 2

    fig, ax1 = plt.subplots(figsize=(5.4, 3.2))
    ax2 = ax1.twinx()
    style = dict(marker="o", markersize=6)
    ax1.plot(xs, beh, color="#4ea1ff", label="behavior_efficacy", **style)
    ax2.plot(xs, ppl, color="#e0795a", linestyle="--", label="perplexity", **style)
    ax1.set_xlabel(f"swept axis: {axis}", fontsize=9)
    ax1.set_ylabel("behavior_efficacy", color="#4ea1ff", fontsize=9)
    ax2.set_ylabel("perplexity (PPL)", color="#e0795a", fontsize=9)
    ax1.tick_params(axis="y", labelcolor="#4ea1ff", labelsize=7)
    ax2.tick_params(axis="y", labelcolor="#e0795a", labelsize=7)
    ax1.tick_params(axis="x", labelsize=7)
    title = f"Sweep: behavior + PPL vs {axis}"
    if single:
        title += " (single point)"
        ax1.text(0.5, 0.04,
                 "sweep accumulates as more alpha/layer rows are logged",
                 transform=ax1.transAxes, ha="center", fontsize=7,
                 color="#9aa0a6", style="italic")
    ax1.set_title(title, fontsize=9)
    ax1.grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=90)
    plt.close(fig)
    return (True, single)


# ===========================================================================
# Ladder board
# ===========================================================================
RUNG_NAMES = {0: "UNIT", 1: "SMOKE", 2: "DEV", 3: "STANDARD", 4: "FULL"}


def ladder_board(rows: list[dict]) -> list[dict]:
    """Per method (tag): highest rung reached + gate cleared/failed + reason."""
    by_tag: dict[str, dict] = {}
    for r in rows:
        tag = str((r.get("config", {}) or {}).get("tag") or r.get("tag") or "untagged")
        rung = int(r.get("rung", 0) or 0)
        cleared = (r.get("status") == "KEEP") and (_num(r, "compliance_rate") == 0.0)
        cur = by_tag.get(tag)
        if cur is None or rung > cur["rung"]:
            by_tag[tag] = {
                "tag": tag,
                "rung": rung,
                "rung_name": RUNG_NAMES.get(rung, str(rung)),
                "cleared": cleared,
                "failure_reason": r.get("failure_reason")
                or ("safety leak (CR>0)" if _num(r, "compliance_rate") > 0 else "")
                or ("composite below champion" if r.get("status") == "DISCARD" else ""),
                "experiment_num": r.get("experiment_num"),
            }
    return sorted(by_tag.values(), key=lambda x: (-x["rung"], x["tag"]))


# ===========================================================================
# Stack / compete decision matrix (Surface A — master dashboard)
#
# STATIC mechanism-based prior transcribed verbatim from
# corpus/steering-stackable-vs-competing-analysis.md §4 (the pairwise
# decision matrix). Legend: STACK (compose cleanly), CARE (stack with care —
# norm / coherence / shared-pathway budget), COMPETE (pick one).
# This is the project's design knowledge, NOT a measured result; the panel is
# captioned as such. When experiments have *measured* a pair's verdict
# (row.config.intervention_family vs row.config.other_family + a
# 'pair_verdict' field), that measured verdict is overlaid on the cell.
# ===========================================================================
# Intervention families (rows == cols), in corpus §4 order.
STACK_FAMILIES = [
    "CAA/ActAdd",
    "Angular/Selective",
    "SAE-feature",
    "KV-cache",
    "Attention-score",
    "DoLa",
    "CAST gate",
]

# Verdict codes: "STACK" | "CARE" | "COMPETE" | "SELF" (the diagonal).
# Transcribed from the corpus §4 matrix (symmetric; diagonal = self-pairing
# note from the corpus where one exists, else SELF).
STACK_MATRIX: dict[str, dict[str, str]] = {
    "CAA/ActAdd": {
        "CAA/ActAdd": "CARE", "Angular/Selective": "COMPETE", "SAE-feature": "CARE",
        "KV-cache": "STACK", "Attention-score": "STACK", "DoLa": "STACK",
        "CAST gate": "STACK",
    },
    "Angular/Selective": {
        "CAA/ActAdd": "COMPETE", "Angular/Selective": "CARE", "SAE-feature": "COMPETE",
        "KV-cache": "STACK", "Attention-score": "STACK", "DoLa": "STACK",
        "CAST gate": "STACK",
    },
    "SAE-feature": {
        "CAA/ActAdd": "CARE", "Angular/Selective": "COMPETE", "SAE-feature": "CARE",
        "KV-cache": "STACK", "Attention-score": "STACK", "DoLa": "STACK",
        "CAST gate": "STACK",
    },
    "KV-cache": {
        "CAA/ActAdd": "STACK", "Angular/Selective": "STACK", "SAE-feature": "STACK",
        "KV-cache": "CARE", "Attention-score": "CARE", "DoLa": "STACK",
        "CAST gate": "STACK",
    },
    "Attention-score": {
        "CAA/ActAdd": "STACK", "Angular/Selective": "STACK", "SAE-feature": "STACK",
        "KV-cache": "CARE", "Attention-score": "COMPETE", "DoLa": "STACK",
        "CAST gate": "STACK",
    },
    "DoLa": {
        "CAA/ActAdd": "STACK", "Angular/Selective": "STACK", "SAE-feature": "STACK",
        "KV-cache": "STACK", "Attention-score": "STACK", "DoLa": "SELF",
        "CAST gate": "STACK",
    },
    "CAST gate": {
        "CAA/ActAdd": "STACK", "Angular/Selective": "STACK", "SAE-feature": "STACK",
        "KV-cache": "STACK", "Attention-score": "STACK", "DoLa": "STACK",
        "CAST gate": "CARE",
    },
}

# Human-readable tooltips for the cell content (kept short for the table).
_STACK_LABEL = {
    "STACK": "STACK", "CARE": "CARE", "COMPETE": "COMPETE", "SELF": "—",
}


def _normalise_family(name: Optional[str]) -> Optional[str]:
    """Map a free-text family token from an experiment row to a matrix family."""
    if not name:
        return None
    s = str(name).strip().lower()
    aliases = {
        "caa": "CAA/ActAdd", "actadd": "CAA/ActAdd", "additive": "CAA/ActAdd",
        "caa/actadd": "CAA/ActAdd", "diffmean": "CAA/ActAdd",
        "angular": "Angular/Selective", "selective": "Angular/Selective",
        "rotational": "Angular/Selective", "spherical": "Angular/Selective",
        "angular/selective": "Angular/Selective",
        "sae": "SAE-feature", "sae-feature": "SAE-feature", "sae-ts": "SAE-feature",
        "kv": "KV-cache", "kv-cache": "KV-cache", "kvcache": "KV-cache",
        "attention": "Attention-score", "attention-score": "Attention-score",
        "pasta": "Attention-score", "spotlight": "Attention-score",
        "dola": "DoLa", "decoding": "DoLa",
        "cast": "CAST gate", "cast gate": "CAST gate", "gate": "CAST gate",
    }
    if s in aliases:
        return aliases[s]
    for key, fam in aliases.items():
        if key in s:
            return fam
    return None


def measured_pair_verdicts(rows: list[dict]) -> dict[tuple[str, str], str]:
    """Overlay layer: any pair a logged experiment actually *measured*.

    A row contributes a measured verdict when its config carries an
    'intervention_family' + 'other_family' (or 'stack_with') pair plus a
    'pair_verdict' in {STACK,CARE,COMPETE}. Returns {(famA,famB): VERDICT}
    keyed symmetrically. Empty when no experiment has measured a pair.
    """
    out: dict[tuple[str, str], str] = {}
    for r in rows:
        cfg = (r.get("config", {}) or {})
        a = _normalise_family(cfg.get("intervention_family") or r.get("intervention_family"))
        b = _normalise_family(cfg.get("other_family") or cfg.get("stack_with")
                              or r.get("other_family"))
        verdict = (r.get("pair_verdict") or cfg.get("pair_verdict") or "")
        verdict = str(verdict).strip().upper()
        if a and b and verdict in ("STACK", "CARE", "COMPETE"):
            out[(a, b)] = verdict
            out[(b, a)] = verdict
    return out


def render_stack_matrix(rows: list[dict]) -> str:
    """Render the pairwise stack/compete matrix as a self-contained HTML card.

    Static mechanism-based prior (corpus §4) with any measured verdict
    overlaid (and flagged) on the relevant cell.
    """
    measured = measured_pair_verdicts(rows)
    head = "<th>family \\ family</th>" + "".join(
        f"<th>{html.escape(c)}</th>" for c in STACK_FAMILIES
    )
    body = []
    for ri in STACK_FAMILIES:
        cells = [f"<th>{html.escape(ri)}</th>"]
        for cj in STACK_FAMILIES:
            prior = STACK_MATRIX.get(ri, {}).get(cj, "SELF")
            meas = measured.get((ri, cj))
            cls = prior.lower()
            label = _STACK_LABEL.get(prior, prior)
            extra = ""
            if meas:
                extra = (f'<br><span class="meas">measured: '
                         f'{html.escape(meas)}</span>')
                # measured verdict drives the cell colour when it disagrees.
                cls = meas.lower() + " measured"
            cells.append(
                f'<td class="sc {cls}" title="prior={html.escape(prior)}'
                f'{(" measured=" + html.escape(meas)) if meas else ""}">'
                f'{html.escape(label)}{extra}</td>'
            )
        body.append("<tr>" + "".join(cells) + "</tr>")
    body_html = "\n".join(body)
    return (
        '<section>\n  <h2>Stack / compete matrix (mechanism-based prior)</h2>\n'
        '  <p class="cap">Pairwise method-composability prior, transcribed from '
        'corpus &sect;4 (steering-stackable-vs-competing-analysis). '
        'Legend: <span class="sc stack scl">STACK</span> compose cleanly &middot; '
        '<span class="sc care scl">CARE</span> stack with care (norm / coherence / '
        'shared-pathway budget) &middot; '
        '<span class="sc compete scl">COMPETE</span> pick one. '
        '<b>This is the design-knowledge prior, not a measured result</b>; where an '
        'experiment has measured a pair, the measured verdict is overlaid and '
        'labelled on the cell.</p>\n'
        '  <div class="tablewrap"><table class="stackmx">\n'
        f"    <thead><tr>{head}</tr></thead>\n"
        f"    <tbody>\n{body_html}\n    </tbody>\n  </table></div>\n"
        f'  <p class="cap">{"Measured overlays present for " + str(len(measured) // 2) + " pair(s)." if measured else "No measured pair verdicts logged yet &mdash; matrix shows the mechanism prior only."}</p>\n'
        "</section>\n"
    )


# ===========================================================================
# HTML fragments
# ===========================================================================
def _page_open(title: str) -> str:
    return (
        "<!DOCTYPE html>\n<html lang=\"en\">\n<head>\n"
        "<meta charset=\"utf-8\">\n"
        "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">\n"
        f"<title>{html.escape(title)}</title>\n<style>{SHARED_CSS}</style>\n"
        "</head>\n<body>\n"
    )


def _howto_block(bullets: list[str]) -> str:
    items = "\n".join(f"    <li>{b}</li>" for b in bullets)
    return (
        '<section class="howto">\n  <h2>How to read this</h2>\n  <ul>\n'
        f"{items}\n  </ul>\n</section>\n"
    )


def _num_cell(row: dict, key: str, tier: str, n_seeds: int) -> str:
    val = row.get(key)
    if isinstance(val, bool) or not isinstance(val, (int, float)):
        disp = "" if val is None else html.escape(str(val))
        return f'<td data-v="{html.escape(str(disp).lower())}">{disp}</td>'
    disp = f"{val:.4f}" if isinstance(val, float) else f"{val}"
    chip = f'<span class="chip {tier.lower()}">{tier} n={n_seeds}</span>'
    return (f'<td data-v="{float(val)}">{html.escape(disp)} '
            f'<span class="seed-badge">n={n_seeds}</span>{chip}</td>')


def _footer(extra: str = "") -> str:
    fp = composite_fingerprint()
    sha = _git_sha()
    ts = time.strftime("%Y-%m-%dT%H:%M:%S")
    return (
        "<footer>\n"
        f"  COMPOSITE_FORMULA: <code>{html.escape(COMPOSITE_FORMULA)}</code><br>\n"
        f"  composite fingerprint: <code>{fp}</code> &middot; "
        f"git SHA: <code>{sha}</code> &middot; "
        f"generated: <time>{ts}</time>"
        f"{extra}<br>\n"
        "  Internal QA pass &mdash; external review pending.\n"
        "</footer>\n</body>\n</html>\n"
    )


SORT_SCRIPT = """
<script>
(function() {
  var sortState = { key: "composite", num: 1, dir: -1 };
  window.sortBy = function(key, num) {
    if (sortState.key === key) { sortState.dir *= -1; }
    else { sortState.key = key; sortState.num = num; sortState.dir = num ? -1 : 1; }
    render();
  };
  function tbody() { return document.querySelector("#runs tbody"); }
  function getRows() {
    return Array.prototype.slice.call(tbody().querySelectorAll("tr"));
  }
  function colIndex(key) {
    var ths = document.querySelectorAll("#runs thead th");
    for (var i = 0; i < ths.length; i++) {
      if (ths[i].getAttribute("data-key") === key) return i;
    }
    return 0;
  }
  function render() {
    var rows = getRows();
    var idx = colIndex(sortState.key);
    rows.sort(function(a, b) {
      var ca = a.children[idx], cb = b.children[idx];
      if (!ca || !cb) return 0;
      var va = ca.getAttribute("data-v"), vb = cb.getAttribute("data-v");
      if (sortState.num) { va = parseFloat(va) || 0; vb = parseFloat(vb) || 0; }
      if (va < vb) return -1 * sortState.dir;
      if (va > vb) return 1 * sortState.dir;
      return 0;
    });
    var tb = tbody();
    rows.forEach(function(r) { tb.appendChild(r); });
    document.querySelectorAll("#runs thead .arr").forEach(function(s){ s.textContent=""; });
    var th = document.querySelector('#runs thead th[data-key="' + sortState.key + '"]');
    if (th && th.querySelector(".arr"))
      th.querySelector(".arr").textContent = sortState.dir < 0 ? "\\u25BC" : "\\u25B2";
  }
  window.applyFilter = function() {
    var q = document.getElementById("filter").value.toLowerCase();
    getRows().forEach(function(r) {
      r.style.display = r.textContent.toLowerCase().indexOf(q) >= 0 ? "" : "none";
    });
  };
  if (document.querySelector("#runs")) render();
})();
</script>
"""


# ===========================================================================
# A. MASTER dashboard
# ===========================================================================
def render_master(rows: list[dict], idea_dirs: list[Path],
                  plots: dict[str, bool], ladder: list[dict]) -> str:
    flat = [_flatten(r) for r in rows]
    champ_num = None
    if flat:
        champ = max(flat, key=lambda r: _num(r, "composite", -1e18))
        champ_num = champ.get("experiment_num")

    head_cells = "".join(
        f'<th data-key="{html.escape(k)}" data-num="{int(num)}" '
        f'onclick="sortBy(\'{html.escape(k)}\',{int(num)})">{html.escape(label)}'
        f'<span class="arr"></span></th>'
        for (k, label, num) in MASTER_COLUMNS
    )
    head_cells += "<th>hyp link</th><th>page</th>"

    body_rows = []
    for r in flat:
        n_seeds = int(r.get("n_seeds", 1) or 1)
        tier = _tier_of(r)
        is_champ = r.get("experiment_num") == champ_num
        vclass = _verdict_class(r.get("status"))
        cls = " ".join(c for c in ("champion" if is_champ else "", vclass) if c)
        exp = r.get("experiment_num")
        tds = []
        for (k, _label, num) in MASTER_COLUMNS:
            if k == "hypothesis":
                lbl = hypothesis_label(r, idea_dirs)
                tds.append(f'<td data-v="{html.escape(lbl.lower())}">{html.escape(lbl)}</td>')
            elif k == "status":
                st = r.get("status") or ""
                tds.append(f'<td data-v="{html.escape(st.lower())}">'
                           f'<span class="verdict {vclass}">{html.escape(st)}</span></td>')
            elif num:
                tds.append(_num_cell(r, k, tier, n_seeds))
            else:
                v = r.get(k)
                disp = "" if v is None else str(v)
                tds.append(f'<td data-v="{html.escape(disp.lower())}">{html.escape(disp)}</td>')
        # hypothesis sub-dashboard link
        hdir = resolve_hypothesis_dir(r, idea_dirs)
        if hdir is not None:
            hid = _idea_id(hdir)
            hyp_link = f'<a href="hyp/{html.escape(hid)}.html">H{html.escape(hid)}</a>'
        else:
            hyp_link = "—"
        page_link = f'<a href="experiments/exp{int(exp):03d}.html">exp{int(exp):03d}</a>' if exp is not None else "—"
        tds.append(f"<td>{hyp_link}</td>")
        tds.append(f"<td>{page_link}</td>")
        body_rows.append(f'<tr class="{cls}">{"".join(tds)}</tr>')

    n_cols = len(MASTER_COLUMNS) + 2
    body_html = "\n".join(body_rows) if body_rows else (
        f'<tr><td colspan="{n_cols}" class="empty">No experiments logged yet. '
        'Run: python -m steering.runner --model fake --rung 0 '
        '--description "first run" --tag smoke</td></tr>'
    )

    champ_label = f"exp#{champ_num}" if champ_num is not None else "none yet"

    # plot panels (only embed images that were actually written)
    def img(name, cap):
        if not plots.get(name):
            return f'<p class="cap">({cap} — plot unavailable: matplotlib missing or no data)</p>'
        return (f'<img class="plot" src="{name}" alt="{html.escape(cap)}">'
                f'<p class="cap">{html.escape(cap)}</p>')

    # ladder board table
    ladder_rows = "".join(
        f'<tr><td>{html.escape(L["tag"])}</td>'
        f'<td>{L["rung"]} ({html.escape(L["rung_name"])})</td>'
        f'<td>{"cleared" if L["cleared"] else "failed"}</td>'
        f'<td>{html.escape(L["failure_reason"] or "—")}</td></tr>'
        for L in ladder
    ) or '<tr><td colspan="4" class="empty">No runs yet.</td></tr>'

    parts = [_page_open("Steering Autoresearch — Master Dashboard")]
    parts.append(
        "<header>\n"
        "  <h1>Steering Autoresearch — Master Dashboard</h1>\n"
        f'  <div class="sub">{len(flat)} run(s) logged &middot; global champion: '
        f"{html.escape(champ_label)} &middot; default sort: composite desc</div>\n"
        "</header>\n"
    )
    parts.append(_howto_block([
        "<b>Composite is multi-objective.</b> It prices all five axes (behavior, "
        "capability, coherence, safety, selectivity) plus an off-manifold geometry "
        "penalty — a method cannot win by sacrificing one axis (CLAUDE.md &sect;6).",
        "<b>Tiers matter.</b> Every numeric cell carries <code>n=X</code> and a "
        "SCREENING (n&le;3) or EVALUATION (n&ge;7) chip. Only EVALUATION rows with "
        "the rigor contract may be called a \"winner\" (CLAUDE.md &sect;7).",
        "<b>Safety is a hard gate.</b> A non-zero <code>CR_jail</code> "
        "(JailbreakBench compliance) is a Rogue-Scalpel leak and an automatic "
        "DISCARD regardless of behavior score (CLAUDE.md &sect;10).",
        "<b>Drill down.</b> Every row links to its per-experiment page and (when "
        "resolvable) its per-hypothesis sub-dashboard. Click a header to sort; type "
        "to filter. The starred green row is the global champion.",
    ]))
    parts.append(
        '<section>\n  <h2>Runs</h2>\n'
        '  <div class="controls" style="margin-bottom:10px">'
        '<input type="text" id="filter" '
        'placeholder="type to filter rows (tag, hypothesis, verdict, ...)" '
        'oninput="applyFilter()"></div>\n'
        '  <div class="tablewrap"><table id="runs">\n'
        f"    <thead><tr>{head_cells}</tr></thead>\n"
        f"    <tbody>\n{body_html}\n    </tbody>\n"
        "  </table></div>\n</section>\n"
    )
    parts.append(
        '<section>\n  <h2>5-axis profile (radar + parallel coordinates)</h2>\n'
        '  <div class="grid">\n'
        f'    <div class="cell">{img("plot_radar.png", "5-axis radar per method: behavior, capability, coherence, safety, selectivity (1 = best).")}</div>\n'
        f'    <div class="cell">{img("plot_parcoords.png", "Parallel coordinates across the same five axes — lines that stay high everywhere are multi-objective wins.")}</div>\n'
        "  </div>\n</section>\n"
    )
    parts.append(
        '<section>\n  <h2>Pareto panels</h2>\n'
        '  <p class="cap">What to read: a circle below-and-right of all stars '
        '(baselines) is dominated on both axes (red outline).</p>\n'
        '  <div class="grid">\n'
        f'    <div class="cell">{img("plot_pareto_capability.png", "behavior vs capability (MMLU drop on x; left = better capability retention).")}</div>\n'
        f'    <div class="cell">{img("plot_pareto_coherence.png", "behavior vs coherence (ΔPPL_norm on x; left = better coherence).")}</div>\n'
        f'    <div class="cell">{img("plot_pareto_safety.png", "behavior vs safety (compliance rate on x; left = safer / no leak).")}</div>\n'
        "  </div>\n</section>\n"
    )
    parts.append(
        '<section>\n  <h2>Ladder board</h2>\n'
        '  <div class="tablewrap"><table>\n'
        "    <thead><tr><th>method (tag)</th><th>highest rung</th>"
        "<th>gate</th><th>failure_reason</th></tr></thead>\n"
        f"    <tbody>{ladder_rows}</tbody>\n  </table></div>\n</section>\n"
    )
    parts.append(render_stack_matrix(rows))
    parts.append(
        '<section>\n  <h2>Geometry probes</h2>\n'
        f'  {img("plot_geometry.png", "Small-multiples: off-shell displacement Δ‖h‖, effective-rank drop, and norm budget ‖Δh‖/‖h‖ per experiment.")}\n'
        "</section>\n"
    )
    parts.append(_footer())
    parts.append(SORT_SCRIPT)
    return "".join(parts)


# ===========================================================================
# B. PER-HYPOTHESIS sub-dashboard
# ===========================================================================
def _parse_idea_md(idea_dir: Path) -> dict:
    """Pull statement / falsifier / predicted-delta from IDEA.md or README.md."""
    out = {"statement": "", "falsifier": "", "predicted": "", "title": idea_dir.name}
    for fname in ("IDEA.md", "README.md"):
        p = idea_dir / fname
        if not p.exists():
            continue
        text = p.read_text(encoding="utf-8", errors="replace")
        # title = first heading
        m = re.search(r"^#\s+(.*)$", text, re.MULTILINE)
        if m and out["title"] == idea_dir.name:
            out["title"] = m.group(1).strip()

        def _section(name):
            m = re.search(rf"^#{{1,6}}\s*{name}.*?$\n(.*?)(?=^#{{1,6}}\s|\Z)",
                          text, re.MULTILINE | re.DOTALL | re.IGNORECASE)
            return m.group(1).strip() if m else ""

        if not out["statement"]:
            out["statement"] = _section("Claim") or _section("Hypothesis") or _section("TL;DR")
        if not out["falsifier"]:
            out["falsifier"] = _section("Falsifier")
        if not out["predicted"]:
            out["predicted"] = (_section("Pre-registered prediction")
                                or _section("Predicted delta")
                                or _section("Predicted delta range"))
    return out


def render_hypothesis(idea_dir: Path, rows: list[dict], plots: dict[str, bool]) -> str:
    hid = _idea_id(idea_dir)
    info = _parse_idea_md(idea_dir)
    flat = [_flatten(r) for r in rows]
    best = max(flat, key=lambda r: _num(r, "composite", -1e18)) if flat else None

    parts = [_page_open(f"Hypothesis H{hid} — {info['title']}")]
    parts.append(
        "<header>\n"
        f'  <div class="nav"><a href="../index.html">&larr; Master dashboard</a></div>\n'
        f"  <h1>Hypothesis H{html.escape(hid)} — {html.escape(info['title'])}</h1>\n"
        f'  <div class="sub">{len(flat)} run(s) for this hypothesis</div>\n'
        "</header>\n"
    )
    parts.append(_howto_block([
        "<b>Scope.</b> This page shows only the runs resolved to hypothesis "
        f"H{html.escape(hid)}; the master dashboard aggregates all hypotheses.",
        "<b>Best-config callout</b> below is the highest-composite run for this "
        "hypothesis (still SCREENING until n&ge;7 with the rigor contract).",
        "<b>Tiers / numerics.</b> Every numeric cell carries <code>n=X</code> and a "
        "SCREENING / EVALUATION chip — no bare numbers.",
        "<b>Drill down.</b> Each row links to its per-experiment page; the footer "
        "links back to the master dashboard.",
    ]))

    if best is not None:
        n_seeds = int(best.get("n_seeds", 1) or 1)
        tier = _tier_of(best)
        parts.append(
            '<section>\n  <h2>Best-config callout</h2>\n'
            '  <div class="callout">\n'
            f'    <b>exp#{best.get("experiment_num")}</b> &middot; tag '
            f'<code>{html.escape(str(best.get("tag") or ""))}</code> &middot; '
            f'composite <b>{_num(best, "composite"):.4f}</b> '
            f'<span class="seed-badge">n={n_seeds}</span>'
            f'<span class="chip {tier.lower()}">{tier} n={n_seeds}</span><br>\n'
            f'    behavior {_num(best, "behavior_efficacy"):.4f} &middot; '
            f'dMMLU {_num(best, "mmlu_drop_pp"):.4f} &middot; '
            f'PPL {_num(best, "perplexity"):.4f} &middot; '
            f'CR_jail {_num(best, "compliance_rate"):.4f} &middot; '
            f'over_refusal {_num(best, "harmless_refusal_rate"):.4f}\n'
            "  </div>\n</section>\n"
        )

    # hypothesis card (markdown-rendered, so no ##/**/|---| leak)
    card = ""
    for label, key in (("Statement", "statement"), ("Falsifier", "falsifier"),
                       ("Predicted Δ", "predicted")):
        if info[key]:
            card += f'<h3>{label}</h3>\n<div class="md">{md_to_html(info[key])}</div>\n'
    if not card:
        card = '<p class="warn">(No IDEA.md / README.md statement found for this hypothesis.)</p>'
    parts.append(f'<section>\n  <h2>Hypothesis card</h2>\n  {card}</section>\n')

    # plots
    def img(name, cap):
        if not plots.get(name):
            return f'<p class="cap">({cap} — plot unavailable.)</p>'
        return (f'<img class="plot" src="{name}" alt="{html.escape(cap)}">'
                f'<p class="cap">{html.escape(cap)}</p>')

    parts.append(
        '<section>\n  <h2>Coordinate descent &amp; seed stability</h2>\n'
        '  <div class="grid">\n'
        f'    <div class="cell">{img(f"hyp{hid}_coord.png", "composite vs injection layer and vs alpha for this hypothesis sweep.")}</div>\n'
        f'    <div class="cell">{img(f"hyp{hid}_seeds.png", "composite per run with mean overlay — variance visibility (seed-stability surrogate).")}</div>\n'
        "  </div>\n</section>\n"
    )

    # scoped cells table
    rowtext = []
    for r in flat:
        n_seeds = int(r.get("n_seeds", 1) or 1)
        tier = _tier_of(r)
        exp = r.get("experiment_num")
        page = f'<a href="../experiments/exp{int(exp):03d}.html">exp{int(exp):03d}</a>' if exp is not None else "—"
        chip = f'<span class="chip {tier.lower()}">{tier} n={n_seeds}</span>'
        rowtext.append(
            f'<tr><td>{exp}</td><td>{html.escape(str(r.get("tag") or ""))}</td>'
            f'<td>{_num(r, "composite"):.4f} {chip}</td>'
            f'<td>{_num(r, "behavior_efficacy"):.4f}</td>'
            f'<td>{_num(r, "compliance_rate"):.4f}</td>'
            f'<td>{html.escape(str(r.get("status") or ""))}</td>'
            f'<td>{page}</td></tr>'
        )
    body = "\n".join(rowtext) or '<tr><td colspan="7" class="empty">No runs.</td></tr>'
    parts.append(
        '<section>\n  <h2>Cells (experiments for this hypothesis)</h2>\n'
        '  <div class="tablewrap"><table>\n'
        "    <thead><tr><th>exp#</th><th>tag</th><th>composite</th>"
        "<th>behavior</th><th>CR_jail</th><th>verdict</th><th>page</th></tr></thead>\n"
        f"    <tbody>\n{body}\n    </tbody>\n  </table></div>\n</section>\n"
    )
    parts.append(_footer(extra=' &middot; <a href="../index.html">&larr; master</a>'))
    return "".join(parts)


# ===========================================================================
# C. PER-EXPERIMENT page
# ===========================================================================
REASONING_STEPS = [
    ("1. Diagnose", "diagnosis"),
    ("2. Cite", "citations"),
    ("3. Hypothesize", "hypothesis"),
    ("4. Predict", "prediction"),
    ("5. Execute", "execution_note"),
    ("6. Analyse", "analysis"),
    ("7. Checkpoint", "checkpoint"),
]

AXIS_METRICS = [
    ("behavior_efficacy", "Axis 1 — behavior efficacy", "high"),
    ("capability_retention", "Axis 2 — capability retention", "~baseline"),
    ("mmlu_drop_pp", "  MMLU drop (pp)", "~0"),
    ("perplexity", "Axis 3 — perplexity", "low"),
    ("dppl_norm", "  ΔPPL_norm", "~0"),
    ("repetition_rate", "  repetition rate", "low"),
    ("compliance_rate", "Axis 4 — JailbreakBench CR", "0 (no leak)"),
    ("harmful_refusal_rate", "  harmful refusal rate", "high"),
    ("harmless_refusal_rate", "Axis 5 — harmless refusal (over-refusal)", "low"),
    ("selectivity_gap", "  selectivity gap", "high"),
]

GEOMETRY_METRICS = [
    ("offshell_displacement", "off-shell displacement Δ‖h‖"),
    ("effective_rank_base", "effective rank (base)"),
    ("effective_rank_steer", "effective rank (steered)"),
    ("participation_ratio_base", "participation ratio (base)"),
    ("participation_ratio_steer", "participation ratio (steered)"),
    ("norm_budget", "norm budget ‖Δh‖/‖h‖"),
    ("fisher_at_layer", "Fisher ratio at layer"),
]


def _extract_samples(row: dict) -> list[dict]:
    """Pull steered-vs-unsteered generation samples out of an experiment row.

    runner.py is NOT edited by this module — we consume whatever keys already
    exist in the logged JSONL row. Recognised shapes (first match wins):

      * top-level pair:  row["sample_steered"] / row["sample_unsteered"]
        (optional row["sample_prompt"] gives the shared prompt).
      * a list:          row["samples"] = [
            {"prompt": ..., "steered": ..., "unsteered": ...}, ...]
        (per-item keys may also be "sample_steered"/"sample_unsteered" or
         "steered_text"/"unsteered_text").
      * nested under config: same keys inside row["config"].

    Returns a normalised list of {"prompt","steered","unsteered"} dicts; an
    empty list means no samples were captured for this run.
    """
    cfg = row.get("config", {}) or {}

    def _g(d, *keys):
        for k in keys:
            v = d.get(k)
            if v not in (None, ""):
                return v
        return None

    # 1. explicit list of samples.
    samples = row.get("samples") or cfg.get("samples")
    if isinstance(samples, list) and samples:
        out = []
        for it in samples:
            if not isinstance(it, dict):
                continue
            out.append({
                "prompt": _g(it, "prompt", "sample_prompt", "input") or "",
                "steered": _g(it, "steered", "sample_steered", "steered_text") or "",
                "unsteered": _g(it, "unsteered", "sample_unsteered",
                                "unsteered_text", "baseline") or "",
            })
        if out:
            return out

    # 2. top-level (or config-level) single pair.
    steered = _g(row, "sample_steered", "steered_text") or _g(cfg, "sample_steered", "steered_text")
    unsteered = _g(row, "sample_unsteered", "unsteered_text") or _g(cfg, "sample_unsteered", "unsteered_text")
    prompt = _g(row, "sample_prompt", "prompt") or _g(cfg, "sample_prompt", "prompt") or ""
    if steered or unsteered:
        return [{"prompt": prompt, "steered": steered or "", "unsteered": unsteered or ""}]
    return []


def _samples_section(row: dict) -> str:
    """Two-column steered-vs-unsteered section (placeholder when absent)."""
    samples = _extract_samples(row)
    if not samples:
        return (
            '<section>\n  <h2>Side-by-side samples (steered vs unsteered)</h2>\n'
            '  <p class="cap">No samples captured for this run. The two-column '
            'layout below is ready; it populates once the runner logs '
            '<code>sample_steered</code> / <code>sample_unsteered</code> (or a '
            '<code>samples</code> list) on the experiment row.</p>\n'
            '  <div class="grid samples">\n'
            '    <div class="cell"><h3>Unsteered (baseline)</h3>'
            '<pre class="sample warn">(no samples captured for this run)</pre></div>\n'
            '    <div class="cell"><h3>Steered</h3>'
            '<pre class="sample warn">(no samples captured for this run)</pre></div>\n'
            "  </div>\n</section>\n"
        )
    blocks = []
    for i, s in enumerate(samples, 1):
        prompt_html = (
            f'  <p class="cap">Prompt {i}: <code>{html.escape(str(s["prompt"]))}</code></p>\n'
            if s.get("prompt") else ""
        )
        blocks.append(
            prompt_html
            + '  <div class="grid samples">\n'
            '    <div class="cell"><h3>Unsteered (baseline)</h3>'
            f'<pre class="sample">{html.escape(str(s["unsteered"]) or "(empty)")}</pre></div>\n'
            '    <div class="cell"><h3>Steered</h3>'
            f'<pre class="sample">{html.escape(str(s["steered"]) or "(empty)")}</pre></div>\n'
            "  </div>\n"
        )
    return (
        '<section>\n  <h2>Side-by-side samples (steered vs unsteered)</h2>\n'
        f'  <p class="cap">{len(samples)} captured generation pair(s) from the '
        "experiment row.</p>\n" + "".join(blocks) + "</section>\n"
    )


def render_experiment(row: dict, ann: dict, idea_dirs: list[Path],
                      sweep_plot: Optional[str] = None,
                      sweep_single: bool = False) -> str:
    flat = _flatten(row)
    exp = flat.get("experiment_num")
    exp_id = f"exp{int(exp):03d}" if exp is not None else "exp"
    tag = str(flat.get("tag") or "")
    n_seeds = int(flat.get("n_seeds", 1) or 1)
    tier = _tier_of(flat)
    hdir = resolve_hypothesis_dir(row, idea_dirs)
    hid = _idea_id(hdir) if hdir is not None else None

    nav = '<a href="../index.html">&larr; Master dashboard</a>'
    if hid is not None:
        nav += f' &middot; <a href="../hyp/{html.escape(hid)}.html">&larr; Hypothesis H{html.escape(hid)} sub-dashboard</a>'

    parts = [_page_open(f"Experiment {exp_id} — {tag}")]
    parts.append(
        "<header>\n"
        f'  <div class="nav">{nav}</div>\n'
        f"  <h1>Experiment {html.escape(exp_id)} — "
        f'<code>{html.escape(tag)}</code></h1>\n'
        '  <div class="sub">'
        f'<span class="tag-pill">{html.escape(tag or "untagged")}</span>'
        f'<span class="seed-badge">n={n_seeds}</span>'
        f'<span class="chip {tier.lower()}">{tier} n={n_seeds}</span> &middot; '
        f'composite <b>{_num(flat, "composite"):.4f}</b> &middot; '
        f'verdict <span class="verdict {_verdict_class(flat.get("status"))}">'
        f'{html.escape(str(flat.get("status") or ""))}</span> &middot; '
        f'git <code>{_git_sha()}</code></div>\n'
        "</header>\n"
    )
    parts.append(_howto_block([
        "<b>This page</b> is the full audit trail for one experiment: the 7-step "
        "reasoning entry, the config, all five axis metrics, and the geometry probes.",
        "<b>Reasoning is rendered from markdown</b> — headings, bold, tables and "
        "blockquotes are converted to HTML (no literal markup leaks).",
        "<b>Tiers / numerics.</b> The header strip carries <code>n=X</code> and the "
        "SCREENING / EVALUATION chip; metrics tables show any CIs.",
        "<b>Back-links</b> to the hypothesis sub-dashboard and the master dashboard "
        "are in the header and footer.",
    ]))

    # Section: 7-step reasoning
    steps_html = []
    for label, key in REASONING_STEPS:
        val = ann.get(key) if ann else None
        if val:
            steps_html.append(
                f'<h3>{html.escape(label)}</h3>\n'
                f'<div class="md">{md_to_html(str(val))}</div>'
            )
        else:
            steps_html.append(
                f'<h3>{html.escape(label)}</h3>\n'
                f'<p class="warn">(reasoning field "{html.escape(key)}" missing — '
                "7-step gate not verified for this field.)</p>"
            )
    if not ann:
        steps_html.insert(0, '<p class="warn">(No reasoning annotation found for '
                          "this experiment — 7-step gate not verified.)</p>")
    parts.append('<section>\n  <h2>7-step reasoning entry</h2>\n'
                 + "\n".join(steps_html) + "\n</section>\n")

    # Section: config
    cfg = row.get("config", {}) or {}
    cfg_json = html.escape(json.dumps(cfg, indent=2))
    parts.append(
        '<section>\n  <h2>Configuration</h2>\n'
        f'  <div class="md"><pre><code>{cfg_json}</code></pre></div>\n'
        '  <p class="cap">Source: experiment_log.jsonl config block.</p>\n</section>\n'
    )

    # Section: axis metrics with CIs
    def _ci_cells(key):
        lo = row.get(f"{key}_ci_low")
        hi = row.get(f"{key}_ci_high")
        if lo is None and hi is None:
            return "<td>(CI not computed)</td><td>(CI not computed)</td>"
        return (f"<td>{html.escape(str(lo))}</td><td>{html.escape(str(hi))}</td>")

    metric_rows = []
    for key, label, good in AXIS_METRICS:
        v = row.get(key)
        disp = f"{v:.4f}" if isinstance(v, float) else ("" if v is None else str(v))
        metric_rows.append(
            f"<tr><td>{html.escape(label)}</td>"
            f'<td>{html.escape(disp)} <span class="seed-badge">n={n_seeds}</span></td>'
            f"<td>{html.escape(good)}</td>{_ci_cells(key)}</tr>"
        )
    parts.append(
        '<section>\n  <h2>Five-axis metrics (with CIs)</h2>\n'
        '  <div class="tablewrap"><table>\n'
        "    <thead><tr><th>metric</th><th>value (n)</th><th>good=</th>"
        "<th>CI low</th><th>CI high</th></tr></thead>\n"
        f"    <tbody>{''.join(metric_rows)}</tbody>\n  </table></div>\n</section>\n"
    )

    # Section: geometry probes
    geo_rows = []
    for key, label in GEOMETRY_METRICS:
        v = row.get(key)
        if v is None:
            continue
        disp = f"{v:.4f}" if isinstance(v, float) else str(v)
        geo_rows.append(f"<tr><td>{html.escape(label)}</td>"
                        f'<td>{html.escape(disp)} <span class="seed-badge">n={n_seeds}</span></td></tr>')
    if not geo_rows:
        geo_rows.append('<tr><td colspan="2" class="empty">No geometry probes logged.</td></tr>')
    parts.append(
        '<section>\n  <h2>Geometry probes</h2>\n'
        '  <div class="tablewrap"><table>\n'
        "    <thead><tr><th>probe</th><th>value (n)</th></tr></thead>\n"
        f"    <tbody>{''.join(geo_rows)}</tbody>\n  </table></div>\n</section>\n"
    )

    # Section: composite breakdown
    def w(name):
        return COMPOSITE_WEIGHTS.get(name, 0.0)
    be = _num(flat, "behavior_efficacy")
    terms = [
        ("behavior_efficacy", 1.0, be, be),
        ("− λ_cap · MMLU_drop", -w("lambda_cap"), max(0.0, _num(flat, "mmlu_drop_pp")),
         -w("lambda_cap") * max(0.0, _num(flat, "mmlu_drop_pp"))),
        ("− λ_coh · ΔPPL_norm", -w("lambda_coh"), max(0.0, _num(flat, "dppl_norm")),
         -w("lambda_coh") * max(0.0, _num(flat, "dppl_norm"))),
        ("− λ_coh_rep · repetition", -w("lambda_coh_rep"), max(0.0, _num(flat, "repetition_rate")),
         -w("lambda_coh_rep") * max(0.0, _num(flat, "repetition_rate"))),
        ("− λ_safe · compliance_rate", -w("lambda_safe"), _num(flat, "compliance_rate"),
         -w("lambda_safe") * _num(flat, "compliance_rate")),
        ("− λ_sel · harmless_refusal", -w("lambda_sel"), max(0.0, _num(flat, "harmless_refusal_rate")),
         -w("lambda_sel") * max(0.0, _num(flat, "harmless_refusal_rate"))),
        ("− λ_geo · offshell", -w("lambda_geo"), max(0.0, _num(flat, "offshell_displacement")),
         -w("lambda_geo") * max(0.0, _num(flat, "offshell_displacement"))),
    ]
    total = sum(t[3] for t in terms)
    brk_rows = "".join(
        f"<tr><td>{html.escape(t[0])}</td><td>{t[1]:+.3f}</td>"
        f"<td>{t[2]:+.4f}</td><td>{t[3]:+.4f}</td></tr>" for t in terms
    )
    reported = _num(flat, "composite")
    recon = "" if abs(total - reported) < 5e-3 else (
        f'<p class="warn">Reconstructed {total:+.4f} differs from reported '
        f"{reported:+.4f} — review weighting.</p>")
    parts.append(
        '<section>\n  <h2>Composite breakdown</h2>\n'
        '  <div class="tablewrap"><table>\n'
        "    <thead><tr><th>term</th><th>weight</th><th>raw</th>"
        "<th>contribution</th></tr></thead>\n"
        f"    <tbody>{brk_rows}"
        f"<tr><td><b>composite (sum)</b></td><td></td><td></td>"
        f"<td><b>{total:+.4f}</b></td></tr></tbody>\n  </table></div>\n"
        f"  {recon}</section>\n"
    )

    # Section: per-experiment sweep curve (behavior + PPL vs swept axis)
    if sweep_plot:
        note = (' <span class="cap">sweep accumulates as more alpha/layer rows '
                "are logged</span>") if sweep_single else ""
        sweep_body = (
            f'<img class="plot" src="{html.escape(sweep_plot)}" '
            'alt="sweep curve: behavior and PPL vs swept axis">'
            f'<p class="cap">behavior_efficacy (left axis) and perplexity '
            f"(right axis) vs the swept alpha/layer axis for this hypothesis "
            f"group.{note}</p>"
        )
    else:
        sweep_body = ('<p class="cap">(sweep curve unavailable — matplotlib '
                      "missing or no alpha/layer value logged for this run; "
                      "sweep accumulates as more alpha/layer rows are logged.)</p>")
    parts.append(
        '<section>\n  <h2>Sweep curve (behavior + PPL vs alpha/layer)</h2>\n'
        f"  {sweep_body}\n</section>\n"
    )

    # Section: side-by-side steered vs unsteered samples
    parts.append(_samples_section(row))

    back = '<a href="../index.html">&larr; master</a>'
    if hid is not None:
        back += f' &middot; <a href="../hyp/{html.escape(hid)}.html">&larr; H{html.escape(hid)}</a>'
    parts.append(_footer(extra=f' &middot; {back}'))
    return "".join(parts)


# ===========================================================================
# Build orchestration
# ===========================================================================
def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def build_all_dashboards(results_dir: Path | None = None,
                         repo_root: Path | None = None) -> Path:
    """Regenerate the full three-tier dashboard (A master + B per-hypothesis +
    C per-experiment) and the docs/ mirrors. Returns the master index path."""
    repo_root = Path(repo_root) if repo_root else REPO_ROOT
    results_dir = Path(results_dir) if results_dir else (repo_root / "autoresearch_results")

    log_path = results_dir / "experiment_log.jsonl"
    rows = load_rows(log_path)
    reasoning = load_reasoning(results_dir)
    idea_dirs = list_idea_dirs(repo_root)

    dash = repo_root / "dashboard"
    docs_dash = repo_root / "docs" / "dashboard"
    dash.mkdir(parents=True, exist_ok=True)
    docs_dash.mkdir(parents=True, exist_ok=True)

    # ---- master plots ----
    plots: dict[str, bool] = {}
    plots["plot_radar.png"] = plot_radar(rows, dash / "plot_radar.png")
    plots["plot_parcoords.png"] = plot_parcoords(rows, dash / "plot_parcoords.png")
    plots["plot_pareto_capability.png"] = plot_pareto(
        rows, "mmlu_drop_pp", "MMLU drop (capability cost)", dash / "plot_pareto_capability.png")
    plots["plot_pareto_coherence.png"] = plot_pareto(
        rows, "dppl_norm", "ΔPPL_norm (coherence cost)", dash / "plot_pareto_coherence.png")
    plots["plot_pareto_safety.png"] = plot_pareto(
        rows, "compliance_rate", "compliance rate (safety cost)", dash / "plot_pareto_safety.png")
    plots["plot_geometry.png"] = plot_geometry(rows, dash / "plot_geometry.png")

    ladder = ladder_board(rows)

    # ---- A. master ----
    master_html = render_master(rows, idea_dirs, plots, ladder)
    _write(dash / "index.html", master_html)
    _write(docs_dash / "index.html", master_html)

    # ---- C. per-experiment pages (master experiments/ dir + docs mirror) ----
    exp_dir = dash / "experiments"
    docs_exp_dir = docs_dash / "experiments"
    exp_dir.mkdir(parents=True, exist_ok=True)
    docs_exp_dir.mkdir(parents=True, exist_ok=True)

    def _sweep_group_key(r: dict) -> str:
        """Group sibling rows that form an alpha/layer sweep: same hypothesis
        (resolved dir, else hypothesis_id) + same behavior."""
        cfg = r.get("config", {}) or {}
        d = resolve_hypothesis_dir(r, idea_dirs)
        hyp = (d.name if d is not None
               else str(cfg.get("hypothesis_id") or r.get("hypothesis_id") or ""))
        beh = str(cfg.get("behavior") or r.get("behavior") or "")
        return f"{hyp}|{beh}"

    sweep_groups: dict[str, list[dict]] = {}
    for r in rows:
        sweep_groups.setdefault(_sweep_group_key(r), []).append(r)

    for row in rows:
        exp = row.get("experiment_num")
        if exp is None:
            continue
        exp_id = f"exp{int(exp):03d}"
        ann = reasoning.get(str(exp), {}) if reasoning else {}

        # sweep curve from this experiment's sibling group (alpha/layer sweep).
        siblings = sweep_groups.get(_sweep_group_key(row), [row])
        sweep_name = f"{exp_id}_sweep.png"
        ok, single = plot_sweep(siblings, exp_dir / sweep_name)
        sweep_plot = sweep_name if ok else None

        page = render_experiment(row, ann, idea_dirs,
                                 sweep_plot=sweep_plot, sweep_single=single)
        _write(exp_dir / f"{exp_id}.html", page)
        _write(docs_exp_dir / f"{exp_id}.html", page)
        # mirror the sweep PNG next to the docs page so the embed resolves.
        if ok:
            src = exp_dir / sweep_name
            if src.exists():
                _write_bytes(docs_exp_dir / sweep_name, src.read_bytes())

    # ---- B. per-hypothesis sub-dashboards ----
    # group rows by resolved hypothesis dir
    by_dir: dict[str, list[dict]] = {}
    for row in rows:
        d = resolve_hypothesis_dir(row, idea_dirs)
        if d is not None:
            by_dir.setdefault(d.name, []).append(row)

    hyp_docs_dir = docs_dash / "hyp"
    for idea_dir in idea_dirs:
        hrows = by_dir.get(idea_dir.name, [])
        if not hrows:
            continue
        hid = _idea_id(idea_dir)
        idea_dash = idea_dir / "dashboard"
        idea_dash.mkdir(parents=True, exist_ok=True)

        hplots: dict[str, bool] = {}
        hplots[f"hyp{hid}_coord.png"] = plot_coord_descent(hrows, idea_dash / f"hyp{hid}_coord.png")
        hplots[f"hyp{hid}_seeds.png"] = plot_seed_stability(hrows, idea_dash / f"hyp{hid}_seeds.png")

        sub_html = render_hypothesis(idea_dir, hrows, hplots)
        # primary sub-dashboard (relative links assume ../index.html + ../experiments/)
        _write(idea_dash / "index.html", sub_html)

        # docs mirror at docs/dashboard/hyp/<id>.html (links: ../index.html,
        # ../experiments/, ../hyp/<id>.html — all resolve from docs/dashboard/hyp/)
        _write(hyp_docs_dir / f"{hid}.html", sub_html)
        # mirror the hypothesis plots next to the docs page
        for pname in hplots:
            src = idea_dash / pname
            if src.exists():
                _write_bytes(hyp_docs_dir / pname, src.read_bytes())

    return dash / "index.html"


def _write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


# Backward-compatible alias (the lean version exposed build_dashboard()).
def build_dashboard(results_dir: Path | None = None) -> Path:
    return build_all_dashboards(results_dir=results_dir, repo_root=REPO_ROOT)


def main() -> None:
    out = build_all_dashboards()
    print(f"Master dashboard written: {out}")
    print(f"Mirror: {REPO_ROOT / 'docs' / 'dashboard' / 'index.html'}")


if __name__ == "__main__":
    main()
