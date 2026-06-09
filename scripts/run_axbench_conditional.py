"""run_axbench_conditional.py — the CONDITIONAL-GATE selectivity experiment.

The project's contribution is NOT the steering direction (S-19/S-24: the DiffMean
direction does not beat a shuffled control — E7 is NULL). It is the CONDITIONAL
GATE: steer ONLY when the target concept is contextually present, so the edit does
not damage unrelated ("off-target") prompts the way unconditional steering does.

For each AxBench concept C we build v_C (DiffMean, unit) and run three arms over
two prompt pools — ON-target (C's own eval instructions) and OFF-target (other
concepts' eval instructions):

  no_steer        : alpha=0 everywhere (the clean reference).
  unconditional   : steer EVERY prompt with v_C at the knee (damages off-target).
  conditional     : gate per prompt on cos(pool(h@L), v_C) > tau; steer only when
                    it fires (tau calibrated to a target FPR on off-target).

A judge (off-family Qwen) scores, with concept=C's description:
  on_concept   : did C appear ON-target?      (induction — want HIGH, ~= uncond)
  off_concept  : did C appear OFF-target?      (spurious leak — want LOW, ~= clean)
  off_fluency  : coherence OFF-target          (collateral — want HIGH, ~= clean)

The conditional WIN: match unconditional ON-target induction while preserving
OFF-target (low leak + high fluency) like no_steer. Pre-registered direction in
PREREGISTRATION.md / IDEA_TABLE Block G (M2). BUILT; this is a SCREENING run.

Usage:
  PYTHONPATH=src python scripts/run_axbench_conditional.py \
    --model models/google/gemma-3-1b-it --quant none --layer 16 \
    --concepts 20 --on-prompts 5 --off-concepts 4 --knee 0.06 \
    --judge-model Qwen/Qwen2.5-3B-Instruct --no-log
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

from run_axbench_e7 import _greedy_gen_batch, _pool_acts, _unit  # noqa: E402

from steering.axbench import load_axbench_concepts, load_axbench_eval_instructions  # noqa: E402
from steering.extract import cosine, diffmean_vector  # noqa: E402
from steering.judge import JudgeUnavailable  # noqa: E402
from steering.model import get_residual_layers, load_model_cached  # noqa: E402


def _cos_pool(model, tok, prompts: list[str], layer: int, vhat: np.ndarray) -> np.ndarray:
    """cos(mean-pooled residual @layer of each prompt, vhat) -> [n]."""
    acts = _pool_acts(model, tok, prompts, layer)
    return np.array([cosine(a, vhat) for a in acts], dtype=np.float32)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/google/gemma-3-1b-it")
    ap.add_argument("--quant", default="none")
    ap.add_argument("--layer", type=int, default=16)
    ap.add_argument("--dataset", default="concept500")
    ap.add_argument("--concepts", type=int, default=20)
    ap.add_argument("--on-prompts", type=int, default=5, help="ON-target eval instructions per concept")
    ap.add_argument("--off-concepts", type=int, default=4, help="other concepts contributing OFF-target prompts")
    ap.add_argument("--knee", type=float, default=0.06)
    ap.add_argument("--target-fpr", type=float, default=0.10, help="gate FPR on OFF-target prompts")
    ap.add_argument("--max-new-tokens", type=int, default=32)
    ap.add_argument("--max-pos", type=int, default=24)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--judge-model", default="Qwen/Qwen2.5-3B-Instruct")
    ap.add_argument("--no-log", action="store_true")
    args = ap.parse_args()

    t0 = time.time()
    rng = np.random.default_rng(args.seed)
    model, tok = load_model_cached(args.model, quant=args.quant)
    n_layers = len(get_residual_layers(model))
    layer = int(min(max(args.layer, 0), n_layers - 1))
    n_need = args.concepts + args.off_concepts + 2
    concepts = load_axbench_concepts(args.dataset, n_concepts=n_need)
    instructions = load_axbench_eval_instructions(args.dataset, k=max(args.on_prompts, 6))
    print(f"[axbench-COND] {args.model} layer {layer} | {args.concepts} concepts | "
          f"on={args.on_prompts} off={args.off_concepts}x | knee {args.knee} fpr {args.target_fpr}",
          flush=True)

    from steering.local_judge import LocalJudge
    try:
        judge = LocalJudge(model_id=args.judge_model)
        judge.score_axbench("The deep blue ocean rolled with waves.", "the ocean", instructions[0])
    except JudgeUnavailable as exc:
        raise SystemExit(f"ABORT: judge unavailable ({exc}).")
    print(f"[instrument] local_judge:{args.judge_model.split('/')[-1]}", flush=True)

    neg = _pool_acts(model, tok, concepts[0]["neg_texts"][: args.max_pos * 2], layer)

    def vec_of(c) -> np.ndarray:
        pos = _pool_acts(model, tok, c["pos_texts"][: args.max_pos], layer)
        return _unit(diffmean_vector(pos, neg[: pos.shape[0]]))

    def judge_batch(gens: list[str], desc: str, instrs: list[str]) -> tuple[float, float]:
        rs = judge.score_axbench_batch([(g, desc, instr) for g, instr in zip(gens, instrs)])
        return (float(np.mean([r["concept"] / 2.0 for r in rs])),
                float(np.mean([r["fluency"] / 2.0 for r in rs])))

    rows = []
    idx = list(range(len(concepts)))
    for k in range(args.concepts):
        c = concepts[idx[k]]
        vhat = vec_of(c)
        vt = torch.tensor(vhat, dtype=torch.float32)
        desc = c["description"]
        on_prompts = instructions[: args.on_prompts]
        # OFF-target prompts: eval instructions are generic, so make them concept-
        # specific by pairing OTHER concepts' descriptions with the instructions is
        # not available; instead use other concepts' pos_texts as off-target prompts.
        off_pool: list[str] = []
        others = [j for j in idx if j != idx[k]]
        rng.shuffle(others)
        for oj in others[: args.off_concepts]:
            off_pool.extend(concepts[oj]["pos_texts"][:2])
        off_prompts = off_pool[: max(4, args.off_concepts * 2)]

        # gate threshold: tau at the (1 - fpr) quantile of OFF-target cos.
        cos_off = _cos_pool(model, tok, off_prompts, layer, vhat)
        cos_on = _cos_pool(model, tok, on_prompts, layer, vhat)
        tau = float(np.quantile(cos_off, 1.0 - args.target_fpr))
        on_fire = cos_on > tau
        off_fire = cos_off > tau

        # --- generate the three arms ---
        # no_steer (alpha=0) and unconditional (alpha=knee) over BOTH pools.
        on_clean = _greedy_gen_batch(model, tok, on_prompts, layer=layer, vector=vt, alpha=0.0,
                                     max_new_tokens=args.max_new_tokens)
        on_uncond = _greedy_gen_batch(model, tok, on_prompts, layer=layer, vector=vt, alpha=args.knee,
                                      max_new_tokens=args.max_new_tokens)
        off_clean = _greedy_gen_batch(model, tok, off_prompts, layer=layer, vector=vt, alpha=0.0,
                                      max_new_tokens=args.max_new_tokens)
        off_uncond = _greedy_gen_batch(model, tok, off_prompts, layer=layer, vector=vt, alpha=args.knee,
                                       max_new_tokens=args.max_new_tokens)
        # conditional = steered where the gate fires, clean otherwise.
        on_cond = [on_uncond[i] if on_fire[i] else on_clean[i] for i in range(len(on_prompts))]
        off_cond = [off_uncond[i] if off_fire[i] else off_clean[i] for i in range(len(off_prompts))]

        on_instr = on_prompts
        # judge concept presence (desc=C) on/off, and off fluency
        on_c_clean, _ = judge_batch(on_clean, desc, on_instr)
        on_c_uncond, _ = judge_batch(on_uncond, desc, on_instr)
        on_c_cond, _ = judge_batch(on_cond, desc, on_instr)
        off_c_clean, off_f_clean = judge_batch(off_clean, desc, off_prompts)
        off_c_uncond, off_f_uncond = judge_batch(off_uncond, desc, off_prompts)
        off_c_cond, off_f_cond = judge_batch(off_cond, desc, off_prompts)

        rows.append({
            "on_fire": float(on_fire.mean()), "off_fire": float(off_fire.mean()),
            "on_clean": on_c_clean, "on_uncond": on_c_uncond, "on_cond": on_c_cond,
            "off_c_clean": off_c_clean, "off_c_uncond": off_c_uncond, "off_c_cond": off_c_cond,
            "off_f_clean": off_f_clean, "off_f_uncond": off_f_uncond, "off_f_cond": off_f_cond,
        })
        if (k + 1) % max(1, args.concepts // 5) == 0:
            r = rows[-1]
            print(f"  {k+1}/{args.concepts} fire on={r['on_fire']:.2f} off={r['off_fire']:.2f} | "
                  f"ON induce clean/uncond/cond {r['on_clean']:.2f}/{r['on_uncond']:.2f}/{r['on_cond']:.2f} | "
                  f"OFF leak {r['off_c_uncond']:.2f}->{r['off_c_cond']:.2f} flu {r['off_f_uncond']:.2f}->{r['off_f_cond']:.2f}",
                  flush=True)

    def m(key: str) -> float:
        return float(np.mean([r[key] for r in rows]))

    print(f"\n=== AxBench CONDITIONAL selectivity ({args.model}, n={len(rows)} concepts, "
          f"judge {args.judge_model.split('/')[-1]}) ===")
    print(f"  gate fire rate   ON-target={m('on_fire'):.3f}  OFF-target={m('off_fire'):.3f}  "
          f"(want high ON, low OFF)")
    print(f"  ON-target induction   clean={m('on_clean'):.3f}  uncond={m('on_uncond'):.3f}  "
          f"cond={m('on_cond'):.3f}   (cond should ~= uncond)")
    print(f"  OFF-target LEAK (spurious concept)  clean={m('off_c_clean'):.3f}  "
          f"uncond={m('off_c_uncond'):.3f}  cond={m('off_c_cond'):.3f}   (cond should ~= clean << uncond)")
    print(f"  OFF-target FLUENCY (collateral)     clean={m('off_f_clean'):.3f}  "
          f"uncond={m('off_f_uncond'):.3f}  cond={m('off_f_cond'):.3f}   (cond should ~= clean >> uncond)")
    # the selectivity scores
    induce_keep = m("on_cond") - m("on_clean")
    leak_avoided = m("off_c_uncond") - m("off_c_cond")
    fluency_saved = m("off_f_cond") - m("off_f_uncond")
    print(f"\n  ON induction kept by cond (cond-clean)   = {induce_keep:+.3f}")
    print(f"  OFF leak avoided vs uncond (uncond-cond)  = {leak_avoided:+.3f}")
    print(f"  OFF fluency saved vs uncond (cond-uncond) = {fluency_saved:+.3f}")
    print(f"  elapsed {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
