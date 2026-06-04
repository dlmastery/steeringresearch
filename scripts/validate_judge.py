"""validate_judge.py — validate a behavior judge against AxBench GROUND TRUTH.

Before any result rests on a judge, prove the judge actually works: score
AxBench's own LABELED positive vs negative outputs and check it gives positives a
higher concept score than negatives. Reported as ROC-AUC (1.0 = perfect, 0.5 =
chance). No external judge is involved in the validation — the labels are the
reference. Use this to vet the local Qwen judge (or any judge) before trusting it.

Usage:
  PYTHONPATH=src python scripts/validate_judge.py --judge local --dataset concept10 --concepts 10 --per 10
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from steering.axbench import load_axbench_labeled  # noqa: E402
from steering.gate import roc_auc  # noqa: E402
from steering.judge import JudgeUnavailable  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--judge", default="local", choices=["local", "gemini"])
    ap.add_argument("--judge-model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--dataset", default="concept10")
    ap.add_argument("--concepts", type=int, default=10)
    ap.add_argument("--per", type=int, default=10, help="positives and negatives per concept")
    args = ap.parse_args()

    t0 = time.time()
    items = load_axbench_labeled(args.dataset, n_concepts=args.concepts, per_class=args.per)
    print(f"[validate] {len(items)} labeled examples "
          f"({sum(i['label'] for i in items)} positive) from {args.dataset}")

    if args.judge == "local":
        from steering.local_judge import LocalJudge
        judge = LocalJudge(model_id=args.judge_model)
        name = args.judge_model.split("/")[-1]
    else:
        from steering.judge import make_judge_or_none
        judge = make_judge_or_none()
        name = "gemini"
        if judge is None:
            raise SystemExit("no Gemini judge (key/credits)")

    scores: list[float] = []
    labels: list[int] = []
    fluency: list[float] = []
    for k, it in enumerate(items):
        try:
            r = judge.score_axbench(it["output"], it["concept"], it["instruction"])
        except JudgeUnavailable as exc:
            raise SystemExit(f"judge failed on item {k}: {exc}")
        scores.append(r["concept"])
        fluency.append(r["fluency"])
        labels.append(it["label"])
        if (k + 1) % max(1, len(items) // 8) == 0:
            print(f"  {k+1}/{len(items)} scored", flush=True)

    s = np.array(scores)
    y = np.array(labels)
    auc = roc_auc(s, y)
    pos_mean = float(s[y == 1].mean())
    neg_mean = float(s[y == 0].mean())
    print(f"\n=== judge validation: {name} on {args.dataset} ===")
    print(f"  concept score  positives mean={pos_mean:.3f}  negatives mean={neg_mean:.3f}  "
          f"separation={pos_mean - neg_mean:+.3f}")
    print(f"  ROC-AUC (concept score discriminates pos vs neg) = {auc:.3f}")
    print(f"  mean fluency = {np.mean(fluency):.3f} / 2")
    verdict = ("TRUSTWORTHY" if auc >= 0.80 else "WEAK" if auc >= 0.65 else "UNTRUSTWORTHY")
    print(f"  VERDICT: {verdict}  (>=0.80 trustworthy, >=0.65 weak, else do not use)")
    print(f"  elapsed {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
