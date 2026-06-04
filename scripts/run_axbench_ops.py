"""run_axbench_ops.py — E27 (operation: add vs rotate) AND E36 (source: DiffMean
vs PCA) on the REAL AxBench benchmark with the off-family local judge.

Sweeps the (source x operation) grid and reports AxBench-judged behavior +
coherence for each combo, over a concept subset. Tests two hypotheses at once:
  E27 — does norm-preserving rotation beat additive steering? (synthetic: FALSIFIED)
  E36 — does the DiffMean or PCA source steer better? (and E4 showed they differ)

Usage:
  PYTHONPATH=src python scripts/run_axbench_ops.py --model google/gemma-2-2b-it \
      --quant none --layer 20 --dataset concept500 --concepts 20 --prompts 8
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any, Callable, cast

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from method_exp_common import log_method_experiment, write_campaign  # noqa: E402
from run_axbench_e7 import _pool_acts, _unit  # noqa: E402

from steering.axbench import load_axbench_concepts, load_axbench_eval_instructions  # noqa: E402
from steering.eval import _ids  # noqa: E402
from steering.extract import diffmean_vector, pca_top1_vector  # noqa: E402
from steering.hooks import SteeringContext  # noqa: E402
from steering.judge import JudgeUnavailable  # noqa: E402
from steering.model import get_residual_layers, load_model_cached  # noqa: E402


def _gen_batch(model, tok, prompts, *, layer, vector, operation, alpha, max_new_tokens=32):
    if getattr(tok, "pad_token_id", None) is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    enc = tok(prompts, return_tensors="pt", padding=True).to(next(model.parameters()).device)
    gen = cast(Callable[..., torch.Tensor], model.generate)
    with SteeringContext(model, vector, [layer], operation=operation, alpha=alpha):
        with torch.no_grad():
            out = gen(**enc, max_new_tokens=max_new_tokens, do_sample=False, num_beams=1,
                      pad_token_id=tok.eos_token_id)
    new = out[:, enc["input_ids"].shape[1]:]
    return [tok.decode(new[i], skip_special_tokens=True) for i in range(len(prompts))]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="google/gemma-2-2b-it")
    ap.add_argument("--quant", default="none")
    ap.add_argument("--layer", type=int, default=20)
    ap.add_argument("--dataset", default="concept500")
    ap.add_argument("--concepts", type=int, default=20)
    ap.add_argument("--prompts", type=int, default=8)
    ap.add_argument("--alpha", type=float, default=0.1)
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
    print(f"[axbench-OPS] {args.model} layer {layer} | {len(concepts)} concepts x "
          f"{len(instructions)} prompts | alpha {args.alpha} | sources diffmean,pca | ops add,rotate")

    from steering.local_judge import LocalJudge
    try:
        judge = LocalJudge(model_id=args.judge_model)
        judge.score_axbench("The deep blue ocean rolled with waves.", "the ocean", instructions[0])
    except JudgeUnavailable as exc:
        raise SystemExit(f"ABORT: judge unavailable ({exc}); requires the judge.")
    print(f"[instrument] local_judge:{args.judge_model.split('/')[-1]}")

    neg = _pool_acts(model, tok, concepts[0]["neg_texts"][: args.max_pos * 2], layer)
    # Pre-extract both source vectors per concept.
    vecs: dict[str, list[Any]] = {"diffmean": [], "pca": []}
    descs = []
    for c in concepts:
        pos = _pool_acts(model, tok, c["pos_texts"][: args.max_pos], layer)
        nsub = neg[: pos.shape[0]]
        vecs["diffmean"].append(torch.tensor(_unit(diffmean_vector(pos, nsub)), dtype=torch.float32))
        vecs["pca"].append(torch.tensor(_unit(pca_top1_vector(pos, nsub)), dtype=torch.float32))
        descs.append(c["description"])

    grid = []
    for source in ("diffmean", "pca"):
        for op in ("add", "rotate"):
            bvals, fvals = [], []
            for desc, v in zip(descs, vecs[source]):
                gens = _gen_batch(model, tok, instructions, layer=layer, vector=v,
                                  operation=("relative_add" if op == "add" else "rotate"),
                                  alpha=args.alpha, max_new_tokens=args.max_new_tokens)
                rs = judge.score_axbench_batch([(g, desc, instr) for g, instr in zip(gens, instructions)])
                bvals.append(np.mean([r["concept"] / 2.0 for r in rs]))
                fvals.append(np.mean([r["fluency"] / 2.0 for r in rs]))
            b, f = float(np.mean(bvals)), float(np.mean(fvals))
            grid.append({"source": source, "operation": op, "behavior": round(b, 4), "coherence": round(f, 4)})
            print(f"  source={source:<8} op={op:<6} behavior={b:.3f}  coherence={f:.3f}", flush=True)

    best = max(grid, key=lambda g: g["behavior"])
    add_rows = {(g["source"]): g for g in grid if g["operation"] == "add"}
    rot_rows = {(g["source"]): g for g in grid if g["operation"] == "rotate"}
    print(f"\n=== AxBench E27(op)/E36(source) ({args.model}) ===")
    print(f"  best combo: source={best['source']} op={best['operation']} "
          f"(behavior {best['behavior']}, coherence {best['coherence']})")
    print(f"  E27 add-vs-rotate (diffmean): add beh {add_rows['diffmean']['behavior']} coh "
          f"{add_rows['diffmean']['coherence']}  |  rotate beh {rot_rows['diffmean']['behavior']} coh "
          f"{rot_rows['diffmean']['coherence']}")
    print(f"  E36 diffmean-vs-pca (add): dm beh {add_rows['diffmean']['behavior']}  |  pca beh "
          f"{add_rows['pca']['behavior']}")

    if not args.no_log:
        for hyp, metric in (("E27", "rotate_vs_add"), ("E36", "pca_vs_diffmean")):
            tag = f"{hyp}-axbench-{Path(args.model).name}"
            write_campaign(tag, {"hyp": hyp, "model": args.model, "layer": layer,
                                 "dataset": args.dataset, "n_concepts": len(concepts),
                                 "instrument": f"local_judge:{args.judge_model.split('/')[-1]}",
                                 "grid": grid, "best": best, "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")})
            if hyp == "E27":
                val = rot_rows["diffmean"]["behavior"] - add_rows["diffmean"]["behavior"]
                diag = ("E27 (norm-preserving rotation vs additive steering) on real AxBench: "
                        "does rotate beat add at the same source/layer/alpha on judged "
                        "behavior+coherence? Synthetic E27 was FALSIFIED (rotate worse).")
                hyp_text = ("Rotation hugs the activation sphere (preserves ||h||) so it may "
                            "keep coherence higher; but if it under-moves the concept it loses "
                            "behavior. Tests whether rotate's coherence edge beats add on a real benchmark.")
            else:
                val = add_rows["pca"]["behavior"] - add_rows["diffmean"]["behavior"]
                diag = ("E36 (which extraction source steers better: DiffMean vs PCA-top1) on "
                        "real AxBench. E4 (exp#122) showed they are only moderately aligned on "
                        "real concepts, so they may steer differently.")
                hyp_text = ("If DiffMean is the better concept axis (AxBench's own finding that "
                            "diff-in-means is the strongest concept method), DiffMean should steer "
                            "at least as well as PCA-top1 on judged behavior.")
            reasoning = {"diagnosis": diag, "citations": (
                "Wu/Zhong et al., 2025 ICML 'AxBench' (arXiv:2501.17148); rotation/operation "
                "geometry from the project's first-principles corpus."),
                "hypothesis": hyp_text,
                "prediction": (f"report the (source x operation) grid; {metric} delta sign + "
                               "magnitude. fingerprint a9001e87087e.")}
            log_method_experiment(
                config={"model": args.model, "rung": 3, "layer": layer, "seed": 0,
                        "operation": "add", "source": "diffmean", "behavior": args.dataset,
                        "n_seeds": len(concepts), "quant": args.quant, "tag": tag},
                description=f"{hyp} (source x operation grid) on AxBench {args.dataset} ({len(concepts)} concepts, local judge)",
                reasoning=reasoning, method=f"{hyp.lower()}_axbench", method_metric=metric,
                method_value=float(val),
                method_extra={"instrument": f"local_judge:{args.judge_model.split('/')[-1]}",
                              "grid": grid, "best": best, "n_concepts": len(concepts)},
                composite=round(best["behavior"], 4), behavior_efficacy=float(best["behavior"]),
                behavior_scorer=f"local_judge:{args.judge_model.split('/')[-1]}", started=t0)
    print(f"  elapsed {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
