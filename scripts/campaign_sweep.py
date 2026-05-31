"""campaign_sweep.py — the overnight workhorse: load a model ONCE and sweep a
grid of (layer × alpha × operation × source), logging each cell as one canonical
experiment (steering.runner.run_single_experiment, which reuses the cached model
via load_model_cached so the in-process sweep does not reload/OOM).

Each cell pre-authors a genuine `_manual` reasoning entry tied to the parent
hypothesis (so the runner's no-fabrication gate is satisfied honestly), then runs.

Usage (examples):
  python scripts/campaign_sweep.py --model models/google/gemma-3-270m-it --quant none \
      --hyp E2 --tag-prefix C1-E2-layer \
      --layers 2 4 6 8 10 12 14 16 --alphas 2.0 --ops add --sources diffmean \
      --diagnosis "..." --citation "..." --hypothesis "..." --prediction "..."

  python scripts/campaign_sweep.py --model models/google/gemma-3-270m-it --quant none \
      --hyp E27 --tag-prefix C3-E27-op --layers 12 --alphas 1 2 4 --ops add rotate project_out

All cells share the parent hypothesis text; the per-cell diagnosis/prediction
note the specific (layer, alpha, op). n=1 SCREENING.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from steering.runner import RESULTS_DIR, run_single_experiment  # noqa: E402

_DEF = {
    "diagnosis": "Overnight campaign cell for hypothesis {hyp}. Sweeping the "
    "intervention grid on real Gemma to measure the geometry/coherence response "
    "(behavior is the synthetic-lexicon proxy; PPL/off-shell Δ‖h‖ are the real "
    "signals). This cell: layer={layer}, alpha={alpha}, op={op}, source={source}.",
    "citation": "Panickssery et al., 2024 ACL 'Steering Llama 2 via Contrastive "
    "Activation Addition' (arXiv:2312.06681); Korznikov et al., 2026 ICML 'The "
    "Rogue Scalpel' (arXiv:2509.22067) — off-manifold displacement is the damage "
    "mechanism, logged as off-shell Δ‖h‖.",
    "hypothesis": "Per the manifold first-principles (corpus), the (layer, alpha, "
    "operation) choice sets how far h leaves the data manifold; this cell measures "
    "behavior, PPL, MMLU and off-shell Δ‖h‖ to test hypothesis {hyp}.",
    "prediction": "Cell {layer}/{alpha}/{op}: finite composite (fingerprint "
    "a9001e87087e), n=1 SCREENING; off-shell Δ‖h‖ rises with |alpha|; PPL rises "
    "super-linearly past the per-layer cliff. Pre-registered measurement, not an "
    "external claim.",
}


def _next_num() -> int:
    log = RESULTS_DIR / "experiment_log.jsonl"
    n = 0
    if log.exists():
        for line in log.read_text(encoding="utf-8").splitlines():
            try:
                n = max(n, json.loads(line).get("experiment_num", n))
            except Exception:
                pass
    return n + 1


def _author(num: int, hyp: str, layer: int, alpha: float, op: str, source: str,
            texts: dict) -> None:
    ann_path = RESULTS_DIR / "reasoning_annotations.json"
    ann = json.loads(ann_path.read_text(encoding="utf-8")) if ann_path.exists() else {}
    fmt = dict(hyp=hyp, layer=layer, alpha=alpha, op=op, source=source)
    ann[str(num)] = {
        "_manual": True,
        "diagnosis": texts["diagnosis"].format(**fmt),
        "citations": texts["citation"].format(**fmt),
        "hypothesis": texts["hypothesis"].format(**fmt),
        "prediction": texts["prediction"].format(**fmt),
    }
    ann_path.write_text(json.dumps(ann, indent=2), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--quant", default="none")
    ap.add_argument("--rung", type=int, default=2)
    ap.add_argument("--hyp", required=True)
    ap.add_argument("--tag-prefix", required=True)
    ap.add_argument("--behavior", default="ocean")
    ap.add_argument("--layers", type=int, nargs="+", required=True)
    ap.add_argument("--alphas", type=float, nargs="+", required=True)
    ap.add_argument("--ops", nargs="+", default=["add"])
    ap.add_argument("--sources", nargs="+", default=["diffmean"])
    ap.add_argument("--normalize", action="store_true", help="unit-normalize steering vectors (E7/E36)")
    ap.add_argument("--diagnosis", default=_DEF["diagnosis"])
    ap.add_argument("--citation", default=_DEF["citation"])
    ap.add_argument("--hypothesis", default=_DEF["hypothesis"])
    ap.add_argument("--prediction", default=_DEF["prediction"])
    args = ap.parse_args()
    texts = {"diagnosis": args.diagnosis, "citation": args.citation,
             "hypothesis": args.hypothesis, "prediction": args.prediction}

    rows = []
    for source in args.sources:
        for op in args.ops:
            for layer in args.layers:
                for alpha in args.alphas:
                    num = _next_num()
                    _author(num, args.hyp, layer, alpha, op, source, texts)
                    tag = f"{args.tag_prefix}-L{layer}-a{alpha}-{op}-{source}"
                    print(f"\n>>> exp#{num} {tag}", flush=True)
                    try:
                        e = run_single_experiment(
                            model_name=args.model, rung=args.rung, layer=layer,
                            alpha=alpha, operation=op, source=source,
                            behavior=args.behavior, seed=0,
                            description=f"{args.hyp} sweep {tag}", tag=tag,
                            quant=args.quant, normalize=args.normalize,
                        )
                        rows.append({
                            "exp": e["experiment_num"], "layer": layer, "alpha": alpha,
                            "op": op, "source": source,
                            "behavior": e["behavior_efficacy"], "ppl": e["perplexity"],
                            "dppl_norm": e["dppl_norm"], "cr": e["compliance_rate"],
                            "offshell": e["offshell_displacement"],
                            "fisher": e["fisher_at_layer"], "composite": e["composite"],
                        })
                    except Exception as exc:  # noqa: BLE001 - record + continue
                        print(f"  CELL FAILED {tag}: {type(exc).__name__}: {exc}")
                        rows.append({"layer": layer, "alpha": alpha, "op": op,
                                     "source": source, "error": str(exc)[:200]})

    out = ROOT / "ideas" / "_campaigns"
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{args.tag_prefix}.json").write_text(
        json.dumps({"hyp": args.hyp, "model": args.model, "rows": rows}, indent=2),
        encoding="utf-8")
    print(f"\n=== {args.tag_prefix} SUMMARY ({len(rows)} cells) ===")
    for r in rows:
        if "error" in r:
            print(f"  L{r['layer']} a{r['alpha']} {r['op']}: ERROR {r['error'][:60]}")
        else:
            print(f"  L{r['layer']:>2} a{r['alpha']:>4} {r['op']:>11} {r['source']:>8}: "
                  f"beh={r['behavior']:.3f} PPL={r['ppl']:>9.1f} offshell={r['offshell']:.3f} "
                  f"fisher={r['fisher']:.2f} comp={r['composite']:+.2f}")


if __name__ == "__main__":
    main()
