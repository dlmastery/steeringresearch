"""run_matched_budget.py — is rotation more coherent than addition AT MATCHED BUDGET?

PRE-REGISTERED (written before the run; see hypotheses/PROGRAM_v2_certified_gate.md §3).

THE GAP. `2601.19375` (Selective Steering, Dang & Ngo) ships norm-preserving rotation and
dismisses additive steering as causing "catastrophic degradation on smaller models" — but
it compares three ROTATION methods to each other and never holds displacement fixed. The
field therefore assumes norm-preservation buys coherence, without a controlled equivalence
test. This is that test.

THE CONTROL. Both operations are parameterised by the same chord displacement f = ‖Δh‖/‖h‖:
  * relative_add(alpha=f)      -> h + f·‖h‖·v̂            chord = f·‖h‖   (norm CHANGES)
  * rotate(theta=2·asin(f/2))  -> norm-preserving turn     chord = f·‖h‖   (norm FIXED)
Equal chord, different path. Any coherence difference is attributable to the PATH, not to
how far we moved — which is exactly the comparison the literature skips.

ENDPOINT: WikiText perplexity. **No LLM judge is involved** — this experiment is immune to
the H0 judge failure (see autoresearch_results/JUDGE_CARD.md).

FALSIFIER. The field's assumption predicts PPL(rotate) < PPL(add) at matched f. If rotation
is EQUAL or WORSE across the sweep, "norm preservation buys coherence" is false at this
scale, and the norm-vs-angle decomposition needs re-attribution.

Usage:
  PYTHONPATH=src python scripts/run_matched_budget.py --layer 12 --n-ppl 40
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from steering.datasets import load_jailbreak_mini, load_xstest_mini  # noqa: E402
from steering.hooks import SteeringContext  # noqa: E402
from steering.model import load_model  # noqa: E402
from steering.real_metrics import wikitext_perplexity  # noqa: E402
from steering.safety_target import extract_refusal_direction  # noqa: E402

DEFAULT_MODEL = "DavidAU/gemma-3-1b-it-heretic-extreme-uncensored-abliterated"


def chord_for_rotation(f: float) -> float:
    """Rotation angle whose chord equals f·‖h‖ (matched budget)."""
    f = min(float(f), 2.0)                       # chord can't exceed the diameter
    return 2.0 * math.asin(f / 2.0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--layer", type=int, default=12)
    ap.add_argument("--budgets", default="0.0,0.02,0.05,0.10,0.15,0.20")
    ap.add_argument("--n-ppl", type=int, default=40)
    ap.add_argument("--quant", default="4bit")
    ap.add_argument("--out", default=str(ROOT / "autoresearch_results" / "matched_budget.json"))
    args = ap.parse_args()

    budgets = [float(x) for x in args.budgets.split(",")]

    print(f"[load] {args.model} ({args.quant})")
    model, tok = load_model(args.model, quant=args.quant)

    harmful = load_jailbreak_mini()
    harmless = load_xstest_mini()
    print(f"[dir ] diff-of-means at layer {args.layer} "
          f"({len(harmful)} harmful / {len(harmless)} harmless)")
    v = extract_refusal_direction(model, tok, harmful, harmless, layer=args.layer)
    v_t = torch.tensor(v, dtype=torch.float32)

    rows = []
    t0 = time.time()
    for f in budgets:
        row = {"budget_f": f}
        for op in ("relative_add", "rotate"):
            alpha = f if op == "relative_add" else chord_for_rotation(f)
            if f == 0.0:
                alpha = 0.0
            try:
                with SteeringContext(model, v_t, [args.layer], operation=op, alpha=alpha):
                    ppl = wikitext_perplexity(model, tok, n=args.n_ppl)
                row[f"{op}_alpha"] = alpha
                row[f"{op}_ppl"] = float(ppl)
            except Exception as exc:
                row[f"{op}_error"] = f"{type(exc).__name__}: {exc}"
                print(f"  [f={f} {op}] FAILED {exc}")
        # the headline contrast
        if "relative_add_ppl" in row and "rotate_ppl" in row:
            row["delta_rotate_minus_add"] = row["rotate_ppl"] - row["relative_add_ppl"]
            print(f"  f={f:<5} add PPL={row['relative_add_ppl']:>10.3f} | "
                  f"rot PPL={row['rotate_ppl']:>10.3f} | "
                  f"rot-add={row['delta_rotate_minus_add']:>+10.3f}")
        rows.append(row)

    # Verdict against the pre-registered falsifier (exclude the f=0 control).
    contrasts = [r["delta_rotate_minus_add"] for r in rows
                 if r.get("budget_f", 0) > 0 and "delta_rotate_minus_add" in r]
    rotate_better = sum(1 for d in contrasts if d < 0)
    verdict = (
        "ROTATION BETTER (field assumption supported)" if rotate_better == len(contrasts) and contrasts
        else "ROTATION NOT BETTER — field assumption NOT supported at matched budget"
        if contrasts and rotate_better == 0
        else "MIXED / inconclusive"
    )

    res = {
        "model": args.model, "layer": args.layer, "quant": args.quant,
        "n_ppl_passages": args.n_ppl, "judge": None,
        "endpoint": "WikiText perplexity (judge-free)",
        "n_harmful": len(harmful), "n_harmless": len(harmless),
        "budgets": budgets, "rows": rows,
        "n_budgets_where_rotate_better": rotate_better,
        "n_budgets_compared": len(contrasts),
        "verdict": verdict,
        "tier": "SCREENING (single extraction set, single layer, n=1 per cell)",
        "elapsed_s": round(time.time() - t0, 1),
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(res, indent=2), encoding="utf-8")

    print(f"\n=== MATCHED-BUDGET ATTRIBUTION (judge-free) ===")
    print(f"  rotation better at {rotate_better}/{len(contrasts)} budgets")
    print(f"  VERDICT: {verdict}")
    print(f"  tier: {res['tier']}")
    print(f"  wrote {args.out}  ({res['elapsed_s']}s)")


if __name__ == "__main__":
    main()
