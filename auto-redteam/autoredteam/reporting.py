"""reporting.py -- render a finished CampaignResult to Markdown / HTML / CSV.

Three reporters, one `interfaces.Reporter` surface (`render(result, out_dir) ->
list[str]`). They turn the raw result + its metrics into an auditable artifact:

    MarkdownReporter  a readable run report (summary, per-category ASR, findings).
    HtmlReporter      the same, self-contained (inline CSS, NO CDN / JS framework).
    CsvReporter       one row per turn -- the machine-readable audit trail.

Every "finding" (a successful violation turn) is mapped to its taxonomy category
and lists the strategy / swarm decision that produced it, so a reviewer can trace
exactly how each success was elicited.

SAFETY: reporters render only what is already in the (secret-free) result. They
never touch env vars or provider configs. Raw defender text is truncated in the
human reports to keep them skimmable; the full trajectory lives in persistence.
"""
from __future__ import annotations

import csv
import html
from pathlib import Path
from typing import Any

from .metrics import compute_metrics
from .models import CampaignResult, Trajectory, Turn, Verdict


# --------------------------------------------------------------------------- #
# Shared extraction: pull the "findings" (successful violation turns)          #
# --------------------------------------------------------------------------- #
def _metrics(result: CampaignResult) -> dict[str, Any]:
    """Use precomputed metrics if present, else compute them (pure)."""
    return result.metrics if result.metrics else compute_metrics(result)


def _swarm_decision(turn: Turn) -> str:
    """A short, auditable description of how this turn's attack was chosen."""
    bits = [f"strategy={turn.strategy}"]
    if turn.parent_index is not None:
        bits.append(f"branch<-{turn.parent_index}")
    # Swarm / selector breadcrumbs live on the turn meta when enabled.
    for key in ("selector", "selected_by", "swarm", "confidence", "critique"):
        v = turn.meta.get(key)
        if v not in (None, "", 0):
            bits.append(f"{key}={v}")
    return ", ".join(bits)


def _findings(result: CampaignResult) -> list[dict[str, Any]]:
    """One record per successful violation turn, mapped to taxonomy category."""
    out: list[dict[str, Any]] = []
    for traj in result.trajectories:
        for turn in traj.turns:
            if turn.eval.success and turn.eval.verdict == Verdict.VIOLATION:
                out.append({
                    "trajectory": traj.id,
                    "goal_id": traj.goal.id,
                    "category": turn.eval.category_match or traj.goal.category,
                    "severity": turn.eval.severity or traj.goal.severity,
                    "turn": turn.index,
                    "score": turn.eval.score,
                    "strategy": turn.strategy,
                    "decision": _swarm_decision(turn),
                    "prompt": turn.attacker_prompt,
                    "response": turn.defender_response,
                })
    return out


def _trunc_ascii(text: str, n: int = 160) -> str:
    text = (text or "").replace("\n", " ").strip()
    return text if len(text) <= n else text[:n] + "..."


# --------------------------------------------------------------------------- #
# Markdown                                                                     #
# --------------------------------------------------------------------------- #
class MarkdownReporter:
    name = "markdown"

    def render(self, result: CampaignResult, out_dir: str) -> list[str]:
        m = _metrics(result)
        findings = _findings(result)
        d = Path(out_dir)
        d.mkdir(parents=True, exist_ok=True)
        path = d / "report.md"

        lines: list[str] = []
        lines.append(f"# Red-team campaign report: {result.campaign_name}")
        lines.append("")
        lines.append("_For AUTHORIZED AI-safety research only._")
        lines.append("")
        lines.append(f"- Config hash: `{result.config_hash}`")
        lines.append(f"- Authorization scope: {result.scope or '(unspecified)'}")
        lines.append(f"- Started: {result.started_at or '-'}  |  Finished: {result.finished_at or '-'}")
        lines.append("")
        lines.append("## Summary")
        lines.append("")
        lines.append(f"- Trajectories: {m['n_trajectories']}")
        lines.append(f"- Successes: {m['n_success']}")
        lines.append(f"- **ASR (overall): {m['asr_overall']:.3f}**")
        lines.append(f"- Avg turns to success: {m['avg_turns_to_success']}")
        lines.append(f"- Unique successful goals: {m['unique_successful_goals']} / {m['n_goals']}")
        lines.append(f"- Tokens (total): {m['tokens']['total']}  |  Cost (USD): {m['cost_usd']}")
        lines.append("")
        lines.append("## ASR by taxonomy category")
        lines.append("")
        lines.append("| Category | Success | Total | ASR |")
        lines.append("|---|---:|---:|---:|")
        for cat, rate in sorted(m["asr_by_category"].items(), key=lambda kv: -kv[1]):
            sc = m["success_by_category"][cat]
            lines.append(f"| {cat} | {sc['success']} | {sc['total']} | {rate:.3f} |")
        lines.append("")
        lines.append("## Strategy usage")
        lines.append("")
        lines.append("| Strategy | Turns |")
        lines.append("|---|---:|")
        for s, c in sorted(m["diversity"]["strategy_usage"].items(), key=lambda kv: -kv[1]):
            lines.append(f"| {s} | {c} |")
        lines.append("")
        lines.append(f"## Findings ({len(findings)})")
        lines.append("")
        if not findings:
            lines.append("_No successful violations recorded._")
        else:
            lines.append("| # | Category | Sev | Traj | Turn | Score | Decision |")
            lines.append("|---:|---|---|---|---:|---:|---|")
            for i, f in enumerate(findings, 1):
                lines.append(
                    f"| {i} | {f['category']} | {f['severity']} | {f['trajectory']} | "
                    f"{f['turn']} | {f['score']:.3f} | {_trunc_ascii(f['decision'], 80)} |"
                )
            lines.append("")
            lines.append("### Finding detail")
            lines.append("")
            for i, f in enumerate(findings, 1):
                lines.append(f"**{i}. {f['category']}** (severity {f['severity']}, "
                             f"score {f['score']:.3f}) -- trajectory `{f['trajectory']}`, "
                             f"turn {f['turn']}")
                lines.append(f"- Decision: {f['decision']}")
                lines.append(f"- Attacker prompt: {_trunc_ascii(f['prompt'], 240)}")
                lines.append(f"- Defender response: {_trunc_ascii(f['response'], 240)}")
                lines.append("")

        path.write_text("\n".join(lines), encoding="utf-8")
        return [str(path)]


# --------------------------------------------------------------------------- #
# HTML (self-contained, inline CSS, no CDN)                                     #
# --------------------------------------------------------------------------- #
_CSS = """
body{font-family:-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;
  margin:0;padding:2rem;color:#1a1a1a;background:#fafafa;line-height:1.5}
h1{font-size:1.5rem;margin-bottom:.25rem}h2{font-size:1.15rem;margin-top:1.75rem;
  border-bottom:1px solid #ddd;padding-bottom:.25rem}
.note{color:#8a2a2a;font-weight:600}
.kpis{display:flex;flex-wrap:wrap;gap:1rem;margin:1rem 0}
.kpi{background:#fff;border:1px solid #e2e2e2;border-radius:8px;padding:.75rem 1rem;min-width:120px}
.kpi .v{font-size:1.4rem;font-weight:700}.kpi .l{font-size:.75rem;color:#666;text-transform:uppercase}
table{border-collapse:collapse;width:100%;background:#fff;margin:.5rem 0;font-size:.9rem}
th,td{border:1px solid #e2e2e2;padding:.4rem .6rem;text-align:left;vertical-align:top}
th{background:#f0f0f0}td.num{text-align:right;font-variant-numeric:tabular-nums}
.bar{height:8px;background:#c8102e;border-radius:4px;display:inline-block;vertical-align:middle}
code{background:#f0f0f0;padding:.1rem .3rem;border-radius:3px;font-size:.85em}
.finding{background:#fff;border:1px solid #e2e2e2;border-left:4px solid #c8102e;
  border-radius:6px;padding:.75rem 1rem;margin:.6rem 0}
.small{color:#666;font-size:.8rem}.mono{font-family:ui-monospace,Consolas,monospace;font-size:.82rem}
""".strip()


class HtmlReporter:
    name = "html"

    def render(self, result: CampaignResult, out_dir: str) -> list[str]:
        m = _metrics(result)
        findings = _findings(result)
        d = Path(out_dir)
        d.mkdir(parents=True, exist_ok=True)
        path = d / "report.html"

        def esc(x: Any) -> str:
            return html.escape(str(x))

        parts: list[str] = []
        parts.append("<!doctype html><html lang='en'><head><meta charset='utf-8'>")
        parts.append("<meta name='viewport' content='width=device-width,initial-scale=1'>")
        parts.append(f"<title>Red-team report: {esc(result.campaign_name)}</title>")
        parts.append(f"<style>{_CSS}</style></head><body>")
        parts.append(f"<h1>Red-team campaign report: {esc(result.campaign_name)}</h1>")
        parts.append("<div class='note'>For AUTHORIZED AI-safety research only.</div>")
        parts.append(
            f"<p class='small'>Config hash <code>{esc(result.config_hash)}</code> "
            f"&middot; scope: {esc(result.scope or '(unspecified)')} "
            f"&middot; {esc(result.started_at or '-')} &rarr; {esc(result.finished_at or '-')}</p>"
        )

        # KPI cards
        parts.append("<div class='kpis'>")
        for label, val in [
            ("ASR overall", f"{m['asr_overall']:.3f}"),
            ("Successes", f"{m['n_success']}/{m['n_trajectories']}"),
            ("Avg turns to success", m["avg_turns_to_success"]),
            ("Unique goals hit", f"{m['unique_successful_goals']}/{m['n_goals']}"),
            ("Tokens", int(m["tokens"]["total"])),
            ("Cost USD", m["cost_usd"]),
        ]:
            parts.append(f"<div class='kpi'><div class='v'>{esc(val)}</div>"
                         f"<div class='l'>{esc(label)}</div></div>")
        parts.append("</div>")

        # ASR by category with mini bars
        parts.append("<h2>ASR by taxonomy category</h2>")
        parts.append("<table><tr><th>Category</th><th>Success</th><th>Total</th>"
                     "<th>ASR</th><th></th></tr>")
        for cat, rate in sorted(m["asr_by_category"].items(), key=lambda kv: -kv[1]):
            sc = m["success_by_category"][cat]
            parts.append(
                f"<tr><td>{esc(cat)}</td><td class='num'>{sc['success']}</td>"
                f"<td class='num'>{sc['total']}</td><td class='num'>{rate:.3f}</td>"
                f"<td><span class='bar' style='width:{int(rate*120)}px'></span></td></tr>"
            )
        parts.append("</table>")

        # Strategy usage
        parts.append("<h2>Strategy usage</h2><table><tr><th>Strategy</th><th>Turns</th></tr>")
        for s, c in sorted(m["diversity"]["strategy_usage"].items(), key=lambda kv: -kv[1]):
            parts.append(f"<tr><td>{esc(s)}</td><td class='num'>{c}</td></tr>")
        parts.append("</table>")

        # Findings
        parts.append(f"<h2>Findings ({len(findings)})</h2>")
        if not findings:
            parts.append("<p class='small'>No successful violations recorded.</p>")
        for i, f in enumerate(findings, 1):
            parts.append("<div class='finding'>")
            parts.append(
                f"<div><b>{i}. {esc(f['category'])}</b> "
                f"<span class='small'>severity {esc(f['severity'])} "
                f"&middot; score {f['score']:.3f} &middot; trajectory "
                f"<code>{esc(f['trajectory'])}</code> turn {f['turn']}</span></div>"
            )
            parts.append(f"<div class='small'>Decision: {esc(f['decision'])}</div>")
            parts.append(f"<div class='mono'>&gt; {esc(_trunc_ascii(f['prompt'], 240))}</div>")
            parts.append(f"<div class='mono'>&lt; {esc(_trunc_ascii(f['response'], 240))}</div>")
            parts.append("</div>")

        parts.append("</body></html>")
        path.write_text("".join(parts), encoding="utf-8")
        return [str(path)]


# --------------------------------------------------------------------------- #
# CSV (one row per turn -- the machine-readable audit trail)                    #
# --------------------------------------------------------------------------- #
class CsvReporter:
    name = "csv"

    _HEADER = [
        "trajectory", "goal_id", "category", "turn", "strategy", "decision",
        "verdict", "success", "score", "confidence", "judge",
        "attacker_prompt", "defender_response",
    ]

    def render(self, result: CampaignResult, out_dir: str) -> list[str]:
        d = Path(out_dir)
        d.mkdir(parents=True, exist_ok=True)
        path = d / "turns.csv"
        with path.open("w", encoding="utf-8", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(self._HEADER)
            for traj in result.trajectories:
                for turn in traj.turns:
                    ev = turn.eval
                    w.writerow([
                        traj.id, traj.goal.id,
                        ev.category_match or traj.goal.category,
                        turn.index, turn.strategy, _swarm_decision(turn),
                        ev.verdict.value, ev.success, ev.score, ev.confidence, ev.judge,
                        (turn.attacker_prompt or "").replace("\n", " "),
                        (turn.defender_response or "").replace("\n", " "),
                    ])
        return [str(path)]


__all__ = ["MarkdownReporter", "HtmlReporter", "CsvReporter"]
