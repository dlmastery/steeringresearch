"""run_axbench_e2.py — E2 (which layer is best for steering) on the REAL AxBench
benchmark with the off-family local judge.

E2 question: where in the residual stream does steering work best? We sweep the
injection layer (extract DiffMean at L and inject at L) at the knee alpha, and
measure AxBench-judged behavior + coherence per layer over a concept subset.

Usage:
  PYTHONPATH=src python scripts/run_axbench_e2.py --model google/gemma-2-2b-it \
      --quant none --layers 6 10 14 18 20 22 --dataset concept500 --concepts 20 --prompts 8

NOTE: the composite field now holds only the fingerprinted 5-axis composite; the
raw behavior metric (best-layer behavior) is in method_value/extra.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from method_exp_common import log_method_experiment, write_campaign  # noqa: E402
from run_axbench_e7 import _greedy_gen_batch, _pool_acts, _unit  # noqa: E402

from steering.axbench import load_axbench_concepts, load_axbench_eval_instructions  # noqa: E402
from steering.extract import diffmean_vector  # noqa: E402
from steering.judge import JudgeUnavailable  # noqa: E402
from steering.model import get_residual_layers, load_model_cached  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="google/gemma-2-2b-it")
    ap.add_argument("--quant", default="none")
    ap.add_argument("--layers", type=int, nargs="+", default=[6, 10, 14, 18, 20, 22])
    ap.add_argument("--dataset", default="concept500")
    ap.add_argument("--concepts", type=int, default=20)
    ap.add_argument("--prompts", type=int, default=8)
    ap.add_argument("--knee", type=float, default=0.1)
    ap.add_argument("--max-new-tokens", type=int, default=32)
    ap.add_argument("--max-pos", type=int, default=24)
    ap.add_argument("--judge-model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--no-log", action="store_true")
    args = ap.parse_args()

    import method_exp_common
    method_exp_common.LOGGING_ENABLED = not args.no_log
    t0 = time.time()

    model, tok = load_model_cached(args.model, quant=args.quant)
    n_layers = len(get_residual_layers(model))
    layers = sorted({int(min(max(li, 0), n_layers - 1)) for li in args.layers})
    concepts = load_axbench_concepts(args.dataset, n_concepts=args.concepts)
    instructions = load_axbench_eval_instructions(args.dataset, k=args.prompts)
    print(f"[axbench-E2] {args.model} ({n_layers} layers) | layers {layers} | "
          f"{len(concepts)} concepts x {len(instructions)} prompts | knee {args.knee}")

    from steering.local_judge import LocalJudge
    try:
        judge = LocalJudge(model_id=args.judge_model)
        judge.score_axbench("The deep blue ocean rolled with waves.", "the ocean", instructions[0])
    except JudgeUnavailable as exc:
        raise SystemExit(f"ABORT: judge unavailable ({exc}); E2 on AxBench requires the judge.")
    print(f"[instrument] local_judge:{args.judge_model.split('/')[-1]}")

    curve = []
    for layer in layers:
        neg = _pool_acts(model, tok, concepts[0]["neg_texts"][: args.max_pos * 2], layer)
        beh_vals, flu_vals = [], []
        for c in concepts:
            pos = _pool_acts(model, tok, c["pos_texts"][: args.max_pos], layer)
            v = torch.tensor(_unit(diffmean_vector(pos, neg)), dtype=torch.float32)
            gens = _greedy_gen_batch(model, tok, instructions, layer=layer, vector=v,
                                     alpha=args.knee, max_new_tokens=args.max_new_tokens)
            rs = judge.score_axbench_batch([(g, c["description"], instr)
                                            for g, instr in zip(gens, instructions)])
            beh_vals.append(np.mean([r["concept"] / 2.0 for r in rs]))
            flu_vals.append(np.mean([r["fluency"] / 2.0 for r in rs]))
        b, f = float(np.mean(beh_vals)), float(np.mean(flu_vals))
        curve.append({"layer": layer, "behavior": round(b, 4), "fluency": round(f, 4)})
        print(f"  layer={layer:<3} behavior={b:.3f}  coherence={f:.3f}", flush=True)

    beh = [c["behavior"] for c in curve]
    best_layer = layers[int(np.argmax(beh))]
    print(f"\n=== AxBench E2 layer sweep ({args.model}) ===")
    print(f"  best behavior layer = {best_layer}")
    print(f"  behavior(layer): {[(c['layer'], c['behavior']) for c in curve]}")
    print(f"  coherence(layer): {[(c['layer'], c['fluency']) for c in curve]}")

    if not args.no_log:
        tag = f"E2-axbench-{Path(args.model).name}"
        write_campaign(tag, {"hyp": "E2", "model": args.model, "dataset": args.dataset,
                             "n_concepts": len(concepts),
                             "instrument": f"local_judge:{args.judge_model.split('/')[-1]}",
                             "curve": curve, "best_layer": best_layer,
                             "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")})
        reasoning = {
            "diagnosis": (
                "E2 (which residual-stream layer is best for steering) on the REAL "
                f"AxBench benchmark; sweep the injection layer over {len(concepts)} "
                "concepts at the knee alpha, judged behavior+coherence by the off-"
                "family local judge. Replaces the synthetic single-concept layer sweep."),
            "citations": (
                "Wu/Zhong et al., 2025 ICML 'AxBench' (arXiv:2501.17148); the layer "
                "question is E2/E13 — mid-to-late residual layers usually steer best."),
            "hypothesis": (
                "Because concept directions are most linearly separable in mid-to-late "
                "layers, AxBench-judged behavior should peak at a mid/late layer and "
                "fall at very early/late layers — a single best injection layer."),
            "prediction": (
                "behavior(layer) is unimodal with a peak at a mid-to-late layer; the "
                "best-behavior layer is reported. fingerprint a9001e87087e."),
        }
        log_method_experiment(
            config={"model": args.model, "rung": 3, "layer": best_layer, "seed": 0,
                    "operation": "relative_add", "source": "diffmean", "behavior": args.dataset,
                    "n_seeds": len(concepts), "quant": args.quant, "tag": tag},
            description=f"E2 layer sweep on AxBench {args.dataset} ({len(concepts)} concepts, local judge)",
            reasoning=reasoning, method="e2_axbench", method_metric="best_layer",
            method_value=float(best_layer),
            method_extra={"instrument": f"local_judge:{args.judge_model.split('/')[-1]}",
                          "curve": curve, "best_layer": best_layer, "n_concepts": len(concepts)},
            # No 5-axis composite (a layer sweep, not a priced run); best_layer is
            # in method_value and the best-layer behavior in extra.
            composite=None, behavior_efficacy=float(max(beh)),
            behavior_scorer=f"local_judge:{args.judge_model.split('/')[-1]}", started=t0)
    print(f"  elapsed {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
