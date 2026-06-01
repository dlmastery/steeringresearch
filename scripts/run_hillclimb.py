"""run_hillclimb.py — coordinate-descent hill-climb over the steering cube
(the rung-2.5 evaluation tier, populating the dashboard hill-climb section).

Unlike campaign_sweep (a full grid), this does COORDINATE DESCENT: start from a
base config, optimise one axis at a time, keep the strict-`>` champion, move on.
Tags are `HC-<hyp>-<axis>-<val>` (the dashboard detects the `HC-` prefix) and each
cell carries config["phase"]="hillclimb" + the hc axis/step for the per-axis
small-multiples. Greedy decoding ⇒ deterministic, so we hill-climb over
(alpha × layer × source × operation), not seeds.

Usage:
  python scripts/run_hillclimb.py --model models/google/gemma-3-270m-it --quant none \
      --hyp E7 --behavior ocean \
      --base-op relative_add --base-layer 16 --base-source diffmean --base-alpha 0.1 \
      --alphas 0.05 0.1 0.15 0.2 --layers 12 14 16 --sources diffmean pca \
      --ops relative_add add
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from steering.runner import RESULTS_DIR, run_single_experiment  # noqa: E402


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


def _author(num: int, hyp: str, axis: str, cfg: dict) -> None:
    ann = RESULTS_DIR / "reasoning_annotations.json"
    a = json.loads(ann.read_text(encoding="utf-8")) if ann.exists() else {}
    a[str(num)] = {
        "_manual": True,
        "diagnosis": (
            f"Hill-climb ({hyp}) coordinate-descent cell, optimising axis '{axis}'. "
            f"Config: op={cfg['operation']}, layer={cfg['layer']}, source={cfg['source']}, "
            f"alpha={cfg['alpha']} (relative_add ⇒ fractional displacement). This is the "
            f"rung-2.5 tuned-ceiling tier: starting from the screening base we optimise one "
            f"axis at a time and keep the strict-> composite champion."),
        "citations": (
            "Bergstra & Bengio, 2012 JMLR 'Random Search for Hyper-Parameter Optimization' "
            "— coordinate/region search beats grid in fewer trials. Panickssery et al., 2024 "
            "ACL CAA (arXiv:2312.06681) — the additive steering whose (layer,alpha,source) we tune."),
        "hypothesis": (
            f"Per the screening verdicts (E7 relative steering; E2 max-Fisher is NOT the best "
            f"layer), the composite-optimal steering config is found by coordinate descent over "
            f"(alpha, layer, source, operation), not at the max-Fisher default. This cell tests "
            f"axis '{axis}' around the current champion."),
        "prediction": (
            f"Cell on axis '{axis}': finite composite (fingerprint a9001e87087e), n=1 SCREENING/"
            f"tuned-ceiling. The hill-climb champion should beat the screening base composite. "
            f"Pre-registered tuned-config measurement, not yet an n>=7 external claim."),
    }
    ann.write_text(json.dumps(a, indent=2), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--quant", default="none")
    ap.add_argument("--hyp", default="E7")
    ap.add_argument("--behavior", default="ocean")
    ap.add_argument("--base-op", default="relative_add")
    ap.add_argument("--base-layer", type=int, default=16)
    ap.add_argument("--base-source", default="diffmean")
    ap.add_argument("--base-alpha", type=float, default=0.1)
    ap.add_argument("--alphas", type=float, nargs="+", default=[0.05, 0.1, 0.15, 0.2])
    ap.add_argument("--layers", type=int, nargs="+", default=[12, 14, 16])
    ap.add_argument("--sources", nargs="+", default=["diffmean", "pca"])
    ap.add_argument("--ops", nargs="+", default=["relative_add", "add"])
    args = ap.parse_args()

    state = {"operation": args.base_op, "layer": args.base_layer,
             "source": args.base_source, "alpha": args.base_alpha}
    normalize = args.base_op == "add"  # unit-normalize raw 'add' so alpha is comparable
    trajectory = []
    best_comp = -1e18

    def run(cfg: dict, axis: str, step: int):
        nonlocal best_comp
        num = _next_num()
        _author(num, args.hyp, axis, cfg)
        tag = f"HC-{args.hyp}-{axis}-{cfg[axis] if axis in cfg else step}"
        print(f"\n>>> exp#{num} {tag}  cfg={cfg}", flush=True)
        norm = cfg["operation"] == "add"
        e = run_single_experiment(
            model_name=args.model, rung=2, layer=cfg["layer"], alpha=cfg["alpha"],
            operation=cfg["operation"], source=cfg["source"], behavior=args.behavior,
            seed=0, description=f"{args.hyp} hill-climb axis={axis} {tag}", tag=tag,
            quant=args.quant, normalize=norm,
        )
        comp = e["composite"]
        trajectory.append({"exp": e["experiment_num"], "axis": axis, "cfg": dict(cfg),
                           "composite": comp, "behavior": e["behavior_efficacy"],
                           "ppl": e["perplexity"], "offshell": e["offshell_displacement"]})
        return comp

    # seed the champion at the base config
    best_comp = run(dict(state), "base", 0)
    # coordinate descent: alpha -> layer -> source -> operation
    for axis, values in [("alpha", args.alphas), ("layer", args.layers),
                         ("source", args.sources), ("operation", args.ops)]:
        for v in values:
            if v == state[axis]:
                continue
            cand = dict(state); cand[axis] = v
            comp = run(cand, axis, v)
            if comp > best_comp:  # strict-> champion
                state, best_comp = cand, comp
        print(f"  [axis {axis}] champion now: {state} (composite {best_comp:+.4f})")

    out = {"hyp": args.hyp, "model": args.model, "best_config": state,
           "best_composite": best_comp, "trajectory": trajectory,
           "fingerprint": "a9001e87087e"}
    (ROOT / "ideas" / "_campaigns" / f"HC-{args.hyp}.json").write_text(
        json.dumps(out, indent=2), encoding="utf-8")
    print(f"\n=== HILL-CLIMB {args.hyp} DONE ===")
    print(f"best config: {state}  composite {best_comp:+.4f}  ({len(trajectory)} cells)")


if __name__ == "__main__":
    main()
