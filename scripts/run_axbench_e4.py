"""run_axbench_e4.py — E4 (is DiffMean ~ PCA-top1?) on the REAL AxBench benchmark.

E4 claim: the DiffMean concept direction and the top-1 PCA direction of the
per-pair differences are nearly identical (cosine ~ 1). This is a pure geometry
check — NO generation, NO judge — so it runs over MANY real AxBench concepts
quickly. We report the distribution of cosine(DiffMean, PCA-top1) across the
concept population.

Usage:
  PYTHONPATH=src python scripts/run_axbench_e4.py --model google/gemma-2-2b-it \
      --quant none --layer 20 --dataset concept500 --concepts 100
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from method_exp_common import log_method_experiment, write_campaign  # noqa: E402
from run_axbench_e7 import _pool_acts  # noqa: E402

from steering.axbench import load_axbench_concepts  # noqa: E402
from steering.extract import cosine, diffmean_vector, pca_top1_vector  # noqa: E402
from steering.model import get_residual_layers, load_model_cached  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="google/gemma-2-2b-it")
    ap.add_argument("--quant", default="none")
    ap.add_argument("--layer", type=int, default=20)
    ap.add_argument("--dataset", default="concept500")
    ap.add_argument("--concepts", type=int, default=100)
    ap.add_argument("--max-pos", type=int, default=24)
    ap.add_argument("--no-log", action="store_true")
    args = ap.parse_args()

    import method_exp_common
    method_exp_common.LOGGING_ENABLED = not args.no_log
    t0 = time.time()

    model, tok = load_model_cached(args.model, quant=args.quant)
    n_layers = len(get_residual_layers(model))
    layer = int(min(max(args.layer, 0), n_layers - 1))
    concepts = load_axbench_concepts(args.dataset, n_concepts=args.concepts)
    print(f"[axbench-E4] {args.model} layer {layer} | {len(concepts)} concepts "
          f"(DiffMean vs PCA-top1 cosine; no judge/generation)")

    neg = _pool_acts(model, tok, concepts[0]["neg_texts"][: args.max_pos * 2], layer)
    cosines = []
    for ci, c in enumerate(concepts):
        pos = _pool_acts(model, tok, c["pos_texts"][: args.max_pos], layer)
        dm = diffmean_vector(pos, neg)
        pca = pca_top1_vector(pos, neg)
        cosines.append(abs(cosine(dm, pca)))
        if (ci + 1) % max(1, len(concepts) // 8) == 0:
            print(f"  {ci+1}/{len(concepts)}  running mean |cos| = {np.mean(cosines):.4f}", flush=True)

    cs = np.array(cosines)
    mean_c, med_c, p5 = float(cs.mean()), float(np.median(cs)), float(np.percentile(cs, 5))
    frac_high = float((cs >= 0.9).mean())
    print(f"\n=== AxBench E4 (DiffMean vs PCA-top1) ({args.model}) ===")
    print(f"  |cos(DiffMean, PCA-top1)| over {len(cs)} concepts: mean={mean_c:.4f} "
          f"median={med_c:.4f} p5={p5:.4f} | fraction >=0.90: {frac_high:.2f}")

    if not args.no_log:
        tag = f"E4-axbench-{Path(args.model).name}"
        write_campaign(tag, {"hyp": "E4", "model": args.model, "layer": layer,
                             "dataset": args.dataset, "n_concepts": len(concepts),
                             "mean_cosine": mean_c, "median_cosine": med_c,
                             "p5_cosine": p5, "fraction_ge_0.9": frac_high,
                             "cosines": [round(x, 4) for x in cosines],
                             "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")})
        reasoning = {
            "diagnosis": (
                f"E4 (DiffMean ~ PCA-top1) on the REAL AxBench benchmark over "
                f"{len(concepts)} concepts at layer {layer}: a pure geometry check "
                "(no judge/generation) of whether the two extraction sources give the "
                "same direction on real concepts, not just one synthetic concept."),
            "citations": (
                "Wu/Zhong et al., 2025 ICML 'AxBench' (arXiv:2501.17148), which finds "
                "difference-in-means is the strongest concept method; E4 asks whether "
                "PCA-top1 of the difference set is the same direction."),
            "hypothesis": (
                "Because the per-pair difference set is dominated by the shared concept "
                "axis, its top PCA component aligns with DiffMean; cosine ~ 1 across the "
                "concept population (a near-tautology the magnitude of which E4 measures)."),
            "prediction": (
                "mean |cos(DiffMean, PCA-top1)| is high (>0.9) across AxBench concepts; "
                "report mean/median/p5 + fraction>=0.9. fingerprint a9001e87087e."),
        }
        log_method_experiment(
            config={"model": args.model, "rung": 3, "layer": layer, "seed": 0,
                    "operation": "none", "source": "diffmean_vs_pca", "behavior": args.dataset,
                    "n_seeds": len(concepts), "quant": args.quant, "tag": tag},
            description=f"E4 DiffMean-vs-PCA cosine on AxBench {args.dataset} ({len(concepts)} concepts)",
            reasoning=reasoning, method="e4_axbench", method_metric="mean_cosine_dm_pca",
            method_value=mean_c,
            method_extra={"mean_cosine": mean_c, "median_cosine": med_c, "p5_cosine": p5,
                          "fraction_ge_0.9": frac_high, "n_concepts": len(concepts)},
            composite=round(mean_c, 4), behavior_efficacy=mean_c,
            behavior_scorer="geometry_cosine", started=t0)
    print(f"  elapsed {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
