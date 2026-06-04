"""run_axbench_e7.py — E7 (DiffMean relative steering) evaluated on the REAL
AxBench benchmark, with the CONCEPT as the replication unit.

Everything here comes from AxBench (arXiv:2501.17148), nothing is authored by us:
  * the CONCEPTS and their contrast text (-> the DiffMean vector),
  * the held-out evaluation INSTRUCTIONS,
  * the JUDGE RUBRIC (concept 0-2 + fluency 0-2, via the off-family Gemini judge).

Per concept c: extract v_real = DiffMean(positive, negative) at layer L; build a
matched shuffled-label control v_shuf; steer the model with relative_add at the
knee on each AxBench instruction; judge the steered output. The behavior score is
the fluency-gated concept score in [0,1]. The PAIRED test real-vs-shuffled then
runs ACROSS ALL CONCEPTS (n = number of concepts) — genuinely independent
replicates, not bootstrap redraws or generation seeds.

Usage:
  PYTHONPATH=src python scripts/run_axbench_e7.py --quick          # 4 concepts, proxy, no-log
  GEMINI_API_KEY=... PYTHONPATH=src python scripts/run_axbench_e7.py \
      --model models/google/gemma-3-270m-it --quant none --layer 16 \
      --dataset concept500 --concepts 0 --prompts 10 --knee 0.1     # --concepts 0 = ALL
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Callable, cast

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from method_exp_common import log_method_experiment, write_campaign  # noqa: E402

from steering.axbench import load_axbench_concepts, load_axbench_eval_instructions  # noqa: E402
from steering.controls import shuffled_label_vector  # noqa: E402
from steering.eval import _ids, concept_rate  # noqa: E402
from steering.extract import diffmean_vector  # noqa: E402
from steering.hooks import SteeringContext, probe_activations  # noqa: E402
from steering.judge import JudgeUnavailable, make_judge_or_none  # noqa: E402
from steering.model import get_residual_layers, load_model_cached  # noqa: E402
from steering.stats import rigor_report  # noqa: E402


def _unit(v: np.ndarray) -> np.ndarray:
    return (v / (np.linalg.norm(v) + 1e-8)).astype(np.float32)


def _pool_acts(model, tok, texts: list[str], layer: int) -> np.ndarray:
    """Mean-pooled residual activation at `layer` for each text -> [n, dim]."""
    out = []
    for t in texts:
        ids = _ids(tok, t, model)
        h = probe_activations(model, ids, [layer])[layer]   # [1, seq, dim]
        out.append(h[0].float().mean(0).cpu().numpy())
    return np.stack(out).astype(np.float32)


def _greedy_gen_batch(model, tok, prompts: list[str], *, layer: int, vector: torch.Tensor,
                      alpha: float, max_new_tokens: int = 32) -> list[str]:
    """Batched steered greedy generation — all prompts in ONE generate() call.

    Gemma-3 eager generation on Windows is ~2s/call; batching the eval prompts
    amortizes that. Left-padded (decoder generation); the SteeringContext hook
    steers every real position (pad positions are attention-masked, so steering
    them is harmless to the real outputs)."""
    if getattr(tok, "pad_token_id", None) is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    enc = tok(prompts, return_tensors="pt", padding=True).to(next(model.parameters()).device)
    gen = cast(Callable[..., torch.Tensor], model.generate)
    with SteeringContext(model, vector, [layer], operation="relative_add", alpha=alpha):
        with torch.no_grad():
            out = gen(**enc, max_new_tokens=max_new_tokens, do_sample=False, num_beams=1,
                      pad_token_id=tok.eos_token_id)
    new = out[:, enc["input_ids"].shape[1]:]
    return [tok.decode(new[i], skip_special_tokens=True) for i in range(len(prompts))]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="fake")
    ap.add_argument("--quant", default="none")
    ap.add_argument("--layer", type=int, default=16)
    ap.add_argument("--dataset", default="concept500")
    ap.add_argument("--concepts", type=int, default=0, help="number of concepts; 0 = ALL")
    ap.add_argument("--prompts", type=int, default=10, help="held-out AxBench eval instructions")
    ap.add_argument("--knee", type=float, default=0.1)
    ap.add_argument("--max-new-tokens", type=int, default=32)
    ap.add_argument("--max-pos", type=int, default=24, help="positive examples per concept for extraction")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--judge", default="local", choices=["local", "gemini"],
                    help="behavior judge: local off-family LLM (Qwen, free) or the Gemini API")
    ap.add_argument("--judge-model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--quick", action="store_true", help="tiny plumbing (4 concepts, proxy, no-log)")
    ap.add_argument("--allow-proxy", action="store_true",
                    help="permit the lexicon proxy on a real run (default: ABORT if the judge is unavailable)")
    ap.add_argument("--no-log", action="store_true")
    args = ap.parse_args()

    if args.quick:
        args.dataset, args.concepts, args.prompts, args.no_log = "concept10", 4, 3, True

    import method_exp_common
    method_exp_common.LOGGING_ENABLED = not args.no_log
    t0 = time.time()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    model, tok = load_model_cached(args.model, quant=args.quant)
    n_layers = len(get_residual_layers(model))
    layer = int(min(max(args.layer, 0), n_layers - 1))

    n_con = None if args.concepts <= 0 else args.concepts
    concepts = load_axbench_concepts(args.dataset, n_concepts=n_con)
    instructions = load_axbench_eval_instructions(args.dataset, k=args.prompts)
    print(f"[axbench] dataset={args.dataset} concepts={len(concepts)} "
          f"eval_instructions={len(instructions)} | model={args.model} layer={layer}")

    judge = None
    instrument = "lexicon_proxy"
    try:
        if args.judge == "local":
            from steering.local_judge import LocalJudge
            judge = LocalJudge(model_id=args.judge_model)
            instrument = f"local_judge:{args.judge_model.split('/')[-1]}"
        else:
            judge = make_judge_or_none()
            instrument = "gemini_judge_axbench"
        if judge is not None:
            judge.score_axbench("The deep blue ocean rolled with salty waves.", "the ocean", instructions[0])
    except JudgeUnavailable as exc:
        print(f"[judge] unavailable: {str(exc)[:120]}")
        judge, instrument = None, "lexicon_proxy"
    print(f"[instrument] {instrument}")
    if judge is None and not args.quick and not args.allow_proxy:
        raise SystemExit(
            "ABORT: the off-family judge is unavailable (no key, or Gemini credits "
            "depleted). The AxBench evaluation REQUIRES the judge — the lexicon proxy "
            "is invalid here and would produce a meaningless (all-zero) result. Top up "
            "the Gemini project credits, or pass --allow-proxy to override for plumbing."
        )

    # Shared negatives: pool ONCE (concept_id == -1).
    neg_acts = _pool_acts(model, tok, concepts[0]["neg_texts"][: args.max_pos * 2], layer)

    real_scores: list[float] = []
    shuf_scores: list[float] = []
    rows: list[dict] = []
    for ci, c in enumerate(concepts):
        pos_acts = _pool_acts(model, tok, c["pos_texts"][: args.max_pos], layer)
        v_real = torch.tensor(_unit(diffmean_vector(pos_acts, neg_acts)), dtype=torch.float32)
        v_shuf = torch.tensor(_unit(shuffled_label_vector(pos_acts, neg_acts, seed=1000 + ci)),
                              dtype=torch.float32)

        def beh(vec: torch.Tensor, desc: str = c["description"]) -> float:
            gens = _greedy_gen_batch(model, tok, instructions, layer=layer, vector=vec,
                                     alpha=args.knee, max_new_tokens=args.max_new_tokens)
            if judge is None:
                lex = desc.replace("//", " ").split()
                return float(np.mean([concept_rate(g, lex) for g in gens]))
            triples = [(g, desc, instr) for g, instr in zip(gens, instructions)]
            if hasattr(judge, "score_axbench_batch"):
                rs = judge.score_axbench_batch(triples)          # one batched GPU call
            else:
                rs = [judge.score_axbench(*t) for t in triples]   # Gemini: sequential API
            return float(np.mean([r["axbench"] for r in rs]))

        rb, sb = beh(v_real), beh(v_shuf)
        real_scores.append(rb)
        shuf_scores.append(sb)
        rows.append({"concept_id": c["concept_id"], "description": c["description"][:80],
                     "real": round(rb, 4), "shuffled": round(sb, 4), "delta": round(rb - sb, 4)})
        if (ci + 1) % max(1, len(concepts) // 25) == 0:
            print(f"  {ci+1}/{len(concepts)}  mean real {np.mean(real_scores):.3f} "
                  f"shuf {np.mean(shuf_scores):.3f}  (this: {c['description'][:45]!r} "
                  f"r={rb:.2f} s={sb:.2f})", flush=True)
            # incremental checkpoint so a mid-run crash on a multi-hour run is recoverable
            try:
                (ROOT / "ideas" / "_campaigns" / f"E7-axbench-{Path(args.model).name}.partial.json"
                 ).write_text(__import__("json").dumps(
                    {"done": ci + 1, "of": len(concepts), "instrument": instrument,
                     "mean_real": float(np.mean(real_scores)),
                     "mean_shuffled": float(np.mean(shuf_scores)), "rows": rows}, indent=1),
                    encoding="utf-8")
            except Exception:  # pragma: no cover - checkpoint must never crash the run
                pass

    rr = rigor_report(real_scores, shuf_scores)
    mean_delta = float(np.mean(real_scores) - np.mean(shuf_scores))
    sig = rr["legs"]["wilcoxon_significant"] and rr["legs"]["ci_excludes_zero"]
    if rr["external_ready"]:
        verdict = "EXTERNAL-READY(this-scale)"          # real significantly > shuffled, all gates
    elif sig and mean_delta > 0:
        verdict = "DIRECTIONAL"                          # real significantly > shuffled (not ordinal)
    elif sig and mean_delta < 0:
        verdict = "NEGATIVE (shuffled control beats real)"   # sign matters — NOT a win
    else:
        verdict = "NULL (real == shuffled)"
    print(f"\n=== AxBench E7 ({args.model}, {args.dataset}, n_concepts={len(concepts)}, "
          f"instrument={instrument}) ===")
    print(f"  mean real {np.mean(real_scores):.4f} vs shuffled {np.mean(shuf_scores):.4f} "
          f"(delta {mean_delta:+.4f}, CI [{rr['bootstrap_ci']['lo']:+.4f},{rr['bootstrap_ci']['hi']:+.4f}])")
    print(f"  paired Wilcoxon p={rr['wilcoxon']['p_value']:.4g}  "
          f"ordinal(worst_real>best_shuffled)={rr['ordinal_gate']['passes']}  "
          f"external_ready={rr['external_ready']}")
    print(f"  VERDICT: {verdict}")

    if not args.no_log:
        tag = f"E7-axbench-{Path(args.model).name}"
        write_campaign(tag, {
            "hyp": "E7", "model": args.model, "dataset": args.dataset, "layer": layer,
            "instrument": instrument, "n_concepts": len(concepts),
            "eval_instructions": instructions, "knee": args.knee,
            "mean_real": float(np.mean(real_scores)), "mean_shuffled": float(np.mean(shuf_scores)),
            "delta": mean_delta, "rigor": rr, "verdict": verdict,
            "per_concept": rows, "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        })
        reasoning = {
            "diagnosis": (
                "E7 evaluated on the REAL AxBench benchmark (arXiv:2501.17148): its "
                "concepts, its contrast text (-> DiffMean), its held-out eval "
                "instructions, and its concept+fluency judge rubric — none authored "
                f"by us. Replication unit = the CONCEPT ({len(concepts)} of them), so "
                "the paired real-vs-shuffled test is over independent concepts, not "
                "generation seeds or correlated bootstraps."
            ),
            "citations": (
                "Wu, Zhong et al., 2025 ICML 'AxBench: Steering LLMs? Even Simple "
                "Baselines Outperform Sparse Autoencoders' (arXiv:2501.17148) — the "
                "benchmark, which itself finds difference-in-means is the strongest "
                "concept method; Panickssery et al., 2024 (arXiv:2312.06681) CAA."
            ),
            "hypothesis": (
                "If the DiffMean concept direction carries the concept (not norm), "
                "the real direction beats a matched-displacement shuffled-label "
                "direction on AxBench's concept+fluency score, across the concept "
                "population — paired Wilcoxon + bootstrap CI + ordinal gate."
            ),
            "prediction": (
                "mean real-minus-shuffled delta > 0 with bootstrap CI excluding 0; "
                "EXTERNAL-READY only if the strict ordinal gate also passes. "
                "fingerprint a9001e87087e."
            ),
        }
        log_method_experiment(
            config={"model": args.model, "rung": 3, "layer": layer, "seed": args.seed,
                    "operation": "relative_add", "source": "diffmean", "behavior": args.dataset,
                    "n_seeds": len(concepts), "quant": args.quant, "tag": tag},
            description=f"E7 on AxBench {args.dataset}: real vs shuffled across {len(concepts)} concepts (judge rubric)",
            reasoning=reasoning, method="e7_axbench", method_metric="mean_real_minus_shuffled",
            method_value=mean_delta,
            method_extra={"instrument": instrument, "dataset": args.dataset,
                          "n_concepts": len(concepts), "mean_real": float(np.mean(real_scores)),
                          "mean_shuffled": float(np.mean(shuf_scores)),
                          "wilcoxon_p": rr["wilcoxon"]["p_value"],
                          "bootstrap_ci": [rr["bootstrap_ci"]["lo"], rr["bootstrap_ci"]["hi"]],
                          "ordinal_pass": rr["ordinal_gate"]["passes"],
                          "external_ready": rr["external_ready"], "verdict": verdict},
            composite=round(mean_delta, 4),
            behavior_efficacy=float(np.mean(real_scores)), behavior_scorer=instrument, started=t0,
        )
    print(f"  elapsed {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
