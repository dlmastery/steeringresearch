"""Adapt the judge-free v2 experiment log into the legacy dashboard schema.

The v2 log records (budget_f, radius_frac, ppl, ppl_ratio) with NO judge and no
5-axis composite -- deliberately, because two judge calibrations failed (JUDGE_CARD.md).
The dashboard renderer, written for the old schema, drops any row lacking
experiment_num/composite/config, so v2 rows rendered as ZERO until adapted here.

composite := 1 - ppl_ratio  (judge-free: lower perplexity ratio is better).
Fingerprint is JUDGEFREE-ppl-ratio so it can never be confused with the retired
5-axis composite, whose own fingerprint no longer matches its code.
"""
from __future__ import annotations
import json, pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / ".dash_v2"; OUT.mkdir(exist_ok=True)

rows = [json.loads(l) for l in (ROOT / "autoresearch_results" / "experiment_log_v2.jsonl").open(encoding="utf-8")]
adapted = []
for i, r in enumerate(rows, 1):
    ratio = float(r.get("ppl_ratio") or 1.0)
    comp = round(1.0 - ratio, 6)
    adapted.append({
        "experiment_num": 200 + i,
        "tag": f"{r['exp']}-f{r['budget_f']}-r{r['radius_frac']}",
        "description": f"{r['exp']}: angle/radius at fixed chord f={r['budget_f']} r={r['radius_frac']}",
        "composite": comp, "composite_fingerprint": "JUDGEFREE-ppl-ratio",
        "perplexity": r.get("ppl"), "dppl_norm": round(ratio - 1.0, 6),
        "layer": 12, "n_seeds": 1, "rung": 2, "tier": "SCREENING",
        "status": "KEEP" if comp > 0 else "DISCARD",
        "model": "DavidAU/gemma-3-1b-it-heretic-extreme-uncensored-abliterated",
        "judge": None, "safety_real": False,
        "behavior_scorer": "none (judge-free: WikiText PPL)",
        "config": {"budget_f": r["budget_f"], "radius_frac": r["radius_frac"],
                   "operation": "angle_radius_fixed_chord", "layer": 12,
                   "citation": r.get("citation")},
        "elapsed_sec": None,
    })
(OUT / "experiment_log.jsonl").write_text("\n".join(json.dumps(a) for a in adapted) + "\n", encoding="utf-8")
for src, dst in (("best_config_v2.json", "best_config.json"),
                 ("reasoning_annotations.json", "reasoning_annotations.json")):
    p = ROOT / "autoresearch_results" / src
    if p.exists():
        (OUT / dst).write_text(p.read_text(encoding="utf-8"), encoding="utf-8")
print(f"adapted {len(adapted)} rows -> {OUT/'experiment_log.jsonl'}")
