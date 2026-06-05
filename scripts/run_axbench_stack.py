"""run_axbench_stack.py — E17/E18/E22 (stacking multiple steering vectors) on the
REAL AxBench benchmark with the off-family local judge.

Can we steer TWO behaviors at once by adding two steering vectors? We pair AxBench
concepts, measure each behavior SOLO and then STACKED (both vectors injected, each
at its own knee displacement), and judge each behavior's presence + coherence.

  E17 — do near-orthogonal behaviors RETAIN their solo effect when stacked?
  E18 — does the vectors' OVERLAP (|cos|) predict the interference (retention loss)?
  E22 — does the doubled push break COHERENCE more (the norm budget)?

Stacking semantics: the combined edit is h + alpha*||h||*(v_hatA + v_hatB), i.e.
EACH vector gets its own knee displacement (we pass v_hatA+v_hatB to relative_add
with alpha scaled by ||v_hatA+v_hatB|| so the per-vector push is preserved, not
averaged away).

Usage:
  PYTHONPATH=src python scripts/run_axbench_stack.py --model google/gemma-2-2b-it \
      --quant none --layer 20 --dataset concept500 --pairs 20 --prompts 6
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
from steering.extract import cosine, diffmean_vector  # noqa: E402
from steering.judge import JudgeUnavailable  # noqa: E402
from steering.model import get_residual_layers, load_model_cached  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="google/gemma-2-2b-it")
    ap.add_argument("--quant", default="none")
    ap.add_argument("--layer", type=int, default=20)
    ap.add_argument("--dataset", default="concept500")
    ap.add_argument("--pairs", type=int, default=20)
    ap.add_argument("--prompts", type=int, default=6)
    ap.add_argument("--knee", type=float, default=0.1)
    ap.add_argument("--max-new-tokens", type=int, default=32)
    ap.add_argument("--max-pos", type=int, default=24)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--judge-model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--no-log", action="store_true")
    args = ap.parse_args()

    import method_exp_common
    method_exp_common.LOGGING_ENABLED = not args.no_log
    t0 = time.time()
    rng = np.random.default_rng(args.seed)

    model, tok = load_model_cached(args.model, quant=args.quant)
    n_layers = len(get_residual_layers(model))
    layer = int(min(max(args.layer, 0), n_layers - 1))
    concepts = load_axbench_concepts(args.dataset, n_concepts=2 * args.pairs + 5)
    instructions = load_axbench_eval_instructions(args.dataset, k=args.prompts)
    print(f"[axbench-STACK] {args.model} layer {layer} | {args.pairs} concept pairs x "
          f"{len(instructions)} prompts | knee {args.knee}")

    from steering.local_judge import LocalJudge
    try:
        judge = LocalJudge(model_id=args.judge_model)
        judge.score_axbench("The deep blue ocean rolled with waves.", "the ocean", instructions[0])
    except JudgeUnavailable as exc:
        raise SystemExit(f"ABORT: judge unavailable ({exc}); requires the judge.")
    print(f"[instrument] local_judge:{args.judge_model.split('/')[-1]}")

    neg = _pool_acts(model, tok, concepts[0]["neg_texts"][: args.max_pos * 2], layer)

    def vec_of(c) -> torch.Tensor:
        pos = _pool_acts(model, tok, c["pos_texts"][: args.max_pos], layer)
        return torch.tensor(_unit(diffmean_vector(pos, neg[: pos.shape[0]])), dtype=torch.float32)

    def behavior(vec: torch.Tensor, alpha: float, desc: str) -> tuple[float, float]:
        gens = _greedy_gen_batch(model, tok, instructions, layer=layer, vector=vec,
                                 alpha=alpha, max_new_tokens=args.max_new_tokens)
        rs = judge.score_axbench_batch([(g, desc, instr) for g, instr in zip(gens, instructions)])
        return (float(np.mean([r["concept"] / 2.0 for r in rs])),
                float(np.mean([r["fluency"] / 2.0 for r in rs])))

    idx = list(range(len(concepts)))
    rng.shuffle(idx)
    rows = []
    for k in range(args.pairs):
        ca, cb = concepts[idx[2 * k]], concepts[idx[2 * k + 1]]
        va, vb = vec_of(ca), vec_of(cb)
        gram = abs(cosine(va.numpy(), vb.numpy()))
        combined = va + vb
        alpha_stack = args.knee * float(combined.norm())   # preserve per-vector push

        solo_a, coh_a = behavior(va, args.knee, ca["description"])
        solo_b, coh_b = behavior(vb, args.knee, cb["description"])
        stack_a, coh_s = behavior(combined, alpha_stack, ca["description"])
        stack_b, _ = behavior(combined, alpha_stack, cb["description"])
        ret_a = stack_a / solo_a if solo_a > 1e-6 else 0.0
        ret_b = stack_b / solo_b if solo_b > 1e-6 else 0.0
        rows.append({"gram": round(gram, 3), "solo_a": round(solo_a, 3), "solo_b": round(solo_b, 3),
                     "stack_a": round(stack_a, 3), "stack_b": round(stack_b, 3),
                     "retention": round((ret_a + ret_b) / 2, 3),
                     "coh_solo": round((coh_a + coh_b) / 2, 3), "coh_stack": round(coh_s, 3)})
        if (k + 1) % max(1, args.pairs // 5) == 0:
            print(f"  pair {k+1}/{args.pairs}  gram={gram:.2f} retention={(ret_a+ret_b)/2:.2f} "
                  f"coh_solo={(coh_a+coh_b)/2:.2f} coh_stack={coh_s:.2f}", flush=True)

    grams = np.array([r["gram"] for r in rows])
    rets = np.array([r["retention"] for r in rows])
    coh_drop = float(np.mean([r["coh_solo"] - r["coh_stack"] for r in rows]))
    mean_ret = float(rets.mean())
    # E18: does overlap predict interference (retention loss)?
    corr = float(np.corrcoef(grams, rets)[0, 1]) if len(rows) > 2 and grams.std() > 0 else float("nan")
    print(f"\n=== AxBench stacking (E17/E18/E22) ({args.model}) ===")
    print(f"  E17 mean retention when stacked = {mean_ret:.3f}  (1.0 = no interference)")
    print(f"  E18 corr(overlap |cos|, retention) = {corr:+.3f}  (negative = more overlap -> more interference)")
    print(f"  E22 mean coherence drop (solo->stack) = {coh_drop:+.3f}  (>0 = stacking breaks text more)")

    if not args.no_log:
        results = {"model": args.model, "layer": layer, "dataset": args.dataset,
                   "instrument": f"local_judge:{args.judge_model.split('/')[-1]}",
                   "n_pairs": args.pairs, "rows": rows, "mean_retention": mean_ret,
                   "corr_overlap_retention": corr, "mean_coherence_drop": coh_drop,
                   "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")}
        for hyp, metric, val in (("E17", "mean_retention", mean_ret),
                                 ("E18", "corr_overlap_retention", corr if corr == corr else 0.0),
                                 ("E22", "coherence_drop", coh_drop)):
            tag = f"{hyp}-axbench-{Path(args.model).name}"
            write_campaign(tag, {"hyp": hyp, **results})
            reasoning = {
                "diagnosis": (f"{hyp} (multi-vector stacking) on the REAL AxBench benchmark over "
                              f"{args.pairs} random concept pairs, judged by the off-family local "
                              "judge. Replaces the synthetic stacking analysis."),
                "citations": ("Wu/Zhong et al., 2025 ICML 'AxBench' (arXiv:2501.17148); the project's "
                              "stacking corpus (interference vs Gram off-diagonal mass; norm budget)."),
                "hypothesis": ("Stacking two vectors steers both behaviors; near-orthogonal pairs "
                               "should retain solo effect (E17); overlap should predict interference "
                               "(E18); the doubled push should cost coherence (E22)."),
                "prediction": (f"report mean retention, corr(overlap,retention), coherence drop. "
                               "fingerprint a9001e87087e.")}
            log_method_experiment(
                config={"model": args.model, "rung": 3, "layer": layer, "seed": args.seed,
                        "operation": "relative_add", "source": "diffmean", "behavior": "stack:" + args.dataset,
                        "n_seeds": args.pairs, "quant": args.quant, "tag": tag},
                description=f"{hyp} stacking on AxBench {args.dataset} ({args.pairs} pairs, local judge)",
                reasoning=reasoning, method=f"{hyp.lower()}_axbench", method_metric=metric,
                method_value=float(val),
                method_extra={"instrument": results["instrument"], "n_pairs": args.pairs,
                              "mean_retention": mean_ret, "corr_overlap_retention": corr,
                              "mean_coherence_drop": coh_drop},
                composite=round(mean_ret, 4), behavior_efficacy=mean_ret,
                behavior_scorer=results["instrument"], started=t0)
    print(f"  elapsed {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
