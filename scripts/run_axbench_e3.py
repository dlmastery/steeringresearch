"""run_axbench_e3.py — E3 (the alpha-coherence cliff) on the REAL AxBench
benchmark with the off-family local judge.

E3 claim: as steering strength alpha grows, behavior rises and then coherence
collapses super-linearly past a knee. Here we test it on AxBench concepts with
the AxBench judge measuring BOTH axes the judge returns: concept expression
(behavior) AND fluency (coherence). The cliff is the alpha where fluency falls
off while behavior saturates/declines.

Per concept we build v_real = DiffMean(AxBench positive vs negative) and sweep
relative_add over an alpha grid, judging the steered outputs on AxBench's eval
instructions. We aggregate behavior(alpha) and fluency(alpha) across a concept
subset (the cliff is a curve, not a population statistic, so a subset suffices).

Usage:
  PYTHONPATH=src python scripts/run_axbench_e3.py --model google/gemma-2-2b-it \
      --quant none --layer 20 --dataset concept500 --concepts 30 --prompts 8 \
      --alphas 0.02 0.05 0.1 0.2 0.4 0.8 --judge local
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
    ap.add_argument("--layer", type=int, default=20)
    ap.add_argument("--dataset", default="concept500")
    ap.add_argument("--concepts", type=int, default=30)
    ap.add_argument("--prompts", type=int, default=8)
    ap.add_argument("--alphas", type=float, nargs="+", default=[0.02, 0.05, 0.1, 0.2, 0.4, 0.8])
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
    layer = int(min(max(args.layer, 0), n_layers - 1))
    concepts = load_axbench_concepts(args.dataset, n_concepts=args.concepts)
    instructions = load_axbench_eval_instructions(args.dataset, k=args.prompts)
    print(f"[axbench-E3] {args.model} layer {layer} | {len(concepts)} concepts x "
          f"{len(instructions)} prompts | alphas {args.alphas}")

    from steering.local_judge import LocalJudge
    try:
        judge = LocalJudge(model_id=args.judge_model)
        judge.score_axbench("The deep blue ocean rolled with waves.", "the ocean", instructions[0])
    except JudgeUnavailable as exc:
        raise SystemExit(f"ABORT: judge unavailable ({exc}); E3 on AxBench requires the judge.")
    print(f"[instrument] local_judge:{args.judge_model.split('/')[-1]}")

    # DiffMean vector per concept (shared neg pooled once).
    neg = _pool_acts(model, tok, concepts[0]["neg_texts"][: args.max_pos * 2], layer)
    vecs = []
    for c in concepts:
        pos = _pool_acts(model, tok, c["pos_texts"][: args.max_pos], layer)
        vecs.append((c["description"],
                     torch.tensor(_unit(diffmean_vector(pos, neg)), dtype=torch.float32)))

    curve = []
    for alpha in args.alphas:
        beh_vals, flu_vals = [], []
        for desc, v in vecs:
            gens = _greedy_gen_batch(model, tok, instructions, layer=layer, vector=v,
                                     alpha=alpha, max_new_tokens=args.max_new_tokens)
            rs = judge.score_axbench_batch([(g, desc, instr) for g, instr in zip(gens, instructions)])
            beh_vals.append(np.mean([r["concept"] / 2.0 for r in rs]))   # behavior in [0,1]
            flu_vals.append(np.mean([r["fluency"] / 2.0 for r in rs]))   # coherence in [0,1]
        b, f = float(np.mean(beh_vals)), float(np.mean(flu_vals))
        curve.append({"alpha": alpha, "behavior": round(b, 4), "fluency": round(f, 4)})
        print(f"  alpha={alpha:<5} behavior={b:.3f}  fluency(coherence)={f:.3f}", flush=True)

    # The cliff: the alpha at which fluency drops most steeply (and behavior peak).
    flu = [c["fluency"] for c in curve]
    beh = [c["behavior"] for c in curve]
    drops = [flu[i] - flu[i + 1] for i in range(len(flu) - 1)]
    cliff_idx = int(np.argmax(drops)) + 1 if drops else 0
    cliff_alpha = args.alphas[cliff_idx]
    peak_alpha = args.alphas[int(np.argmax(beh))]
    print(f"\n=== AxBench E3 cliff ({args.model}) ===")
    print(f"  behavior peaks at alpha={peak_alpha}; fluency cliff (steepest drop) at alpha={cliff_alpha}")
    print(f"  behavior(alpha): {[c['behavior'] for c in curve]}")
    print(f"  fluency(alpha):  {[c['fluency'] for c in curve]}")

    if not args.no_log:
        tag = f"E3-axbench-{Path(args.model).name}"
        write_campaign(tag, {"hyp": "E3", "model": args.model, "layer": layer,
                             "dataset": args.dataset, "n_concepts": len(concepts),
                             "instrument": f"local_judge:{args.judge_model.split('/')[-1]}",
                             "curve": curve, "peak_alpha": peak_alpha, "cliff_alpha": cliff_alpha,
                             "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")})
        reasoning = {
            "diagnosis": (
                "E3 (the alpha-coherence cliff) on the REAL AxBench benchmark with the "
                "off-family local judge scoring BOTH behavior (concept) and fluency "
                f"(coherence). Sweep relative_add alpha over {len(concepts)} AxBench "
                "concepts; does coherence collapse past a knee while "
                "behavior saturates? Replaces the synthetic single-concept cliff."),
            "citations": (
                "Wu/Zhong et al., 2025 ICML 'AxBench' (arXiv:2501.17148); the coherence "
                "cliff is the program's N17/N5 geometry story (off-shell displacement "
                "predicts incoherence) tested on real concepts + a real fluency judge."),
            "hypothesis": (
                "Per the manifold first-principles, larger alpha pushes h further off "
                "the data shell; fluency (coherence) should fall super-linearly past a "
                "knee while behavior saturates — a Pareto cliff in (behavior, fluency)."),
            "prediction": (
                "behavior(alpha) rises then plateaus/declines; fluency(alpha) collapses "
                "past a knee alpha; the steepest-fluency-drop alpha marks the cliff. "
                "fingerprint a9001e87087e."),
        }
        log_method_experiment(
            config={"model": args.model, "rung": 3, "layer": layer, "seed": 0,
                    "operation": "relative_add", "source": "diffmean", "behavior": args.dataset,
                    "n_seeds": len(concepts), "quant": args.quant, "tag": tag},
            description=f"E3 alpha-coherence cliff on AxBench {args.dataset} ({len(concepts)} concepts, local judge)",
            reasoning=reasoning, method="e3_axbench", method_metric="cliff_alpha",
            method_value=float(cliff_alpha),
            method_extra={"instrument": f"local_judge:{args.judge_model.split('/')[-1]}",
                          "curve": curve, "peak_alpha": peak_alpha, "cliff_alpha": cliff_alpha,
                          "n_concepts": len(concepts)},
            composite=round(max(beh), 4), behavior_efficacy=float(max(beh)),
            behavior_scorer=f"local_judge:{args.judge_model.split('/')[-1]}", started=t0)
    print(f"  elapsed {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
