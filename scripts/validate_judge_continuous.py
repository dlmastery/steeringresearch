"""validate_judge_continuous.py — H0-2: does a CONTINUOUS score rescue the judge?

H0-1 measured ROC-AUC 0.665 for Qwen3-4B on AxBench concept10 using the rubric's
INTEGER {0,1,2} concept score, with a large mean separation (+0.58). That pattern is
the signature of a coarsely-quantized score, so this run tests the mechanism directly:
score the SAME items with `LocalJudge.score_axbench_expected`, which reads the expected
value under the model's own distribution over {"0","1","2"} instead of the argmax.

Reports both AUCs side by side so the delta is attributable to quantization alone --
same model, same prompt, same items, same labels; only the readout changes.

Usage:
  PYTHONPATH=src python scripts/validate_judge_continuous.py \
      --judge-model Qwen/Qwen3-4B-Instruct-2507 --dataset concept10 --concepts 10 --per 15
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from steering.axbench import load_axbench_labeled  # noqa: E402
from steering.gate import roc_auc  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--judge-model", default="Qwen/Qwen3-4B-Instruct-2507")
    ap.add_argument("--dataset", default="concept10")
    ap.add_argument("--concepts", type=int, default=10)
    ap.add_argument("--per", type=int, default=15)
    ap.add_argument("--out", default=str(ROOT / "autoresearch_results" / "H0-2_continuous_judge.json"))
    args = ap.parse_args()

    items = load_axbench_labeled(args.dataset, n_concepts=args.concepts, per_class=args.per)
    print(f"[H0-2] {len(items)} labelled examples ({sum(i['label'] for i in items)} positive)")

    from steering.local_judge import LocalJudge
    judge = LocalJudge(model_id=args.judge_model)

    y, cont = [], []
    t0 = time.time()
    for k, it in enumerate(items, 1):
        try:
            s = judge.score_axbench_expected(it["output"], it["concept"], it["instruction"])
        except Exception as exc:  # never silently score 0 -- that biases the result
            print(f"  [skip {k}] {type(exc).__name__}: {exc}")
            continue
        cont.append(s)
        y.append(int(it["label"]))
        if k % 50 == 0:
            print(f"  {k}/{len(items)} scored ({time.time()-t0:.0f}s)")

    y = np.asarray(y)
    cont = np.asarray(cont, dtype=float)
    auc = float(roc_auc(y, cont))
    pos, neg = cont[y == 1], cont[y == 0]
    n_unique = int(len(np.unique(np.round(cont, 6))))

    res = {
        "judge_model": args.judge_model,
        "dataset": args.dataset,
        "n_scored": int(len(y)),
        "n_positive": int(y.sum()),
        "readout": "expected value over p({0,1,2}) — continuous",
        "roc_auc_continuous": auc,
        "positives_mean": float(pos.mean()),
        "negatives_mean": float(neg.mean()),
        "separation": float(pos.mean() - neg.mean()),
        "distinct_score_values": n_unique,
        "baseline_h0_1_integer_auc": 0.665,
        "delta_vs_integer": auc - 0.665,
        "elapsed_s": round(time.time() - t0, 1),
        "gate": 0.85,
        "passes_gate": auc >= 0.85,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(res, indent=2), encoding="utf-8")

    print("\n=== H0-2: continuous (expected-value) readout ===")
    print(f"  ROC-AUC continuous = {auc:.4f}   (H0-1 integer readout = 0.665)")
    print(f"  delta              = {auc - 0.665:+.4f}")
    print(f"  positives {pos.mean():.3f} vs negatives {neg.mean():.3f}  "
          f"(separation {pos.mean()-neg.mean():+.3f})")
    print(f"  distinct score values = {n_unique} (integer readout had 3)")
    print(f"  elapsed {res['elapsed_s']}s")
    print(f"  GATE >= 0.85 -> {'PASS' if res['passes_gate'] else 'FAIL'}")
    print(f"  wrote {args.out}")


if __name__ == "__main__":
    main()
