"""run_ma_direction_seeds.py — M-a: the matched-budget result at n>=8, real benchmarks.

======================================================================== 7-STEP
DIAGNOSE
  HC-1/HC-2 established that at a FIXED chord displacement, spending the budget on
  RADIUS beats spending it on ANGLE, monotonically (pure rotation is the worst
  allocation; at f=0.10 it costs 4x the perplexity of pure addition). But the
  steering direction in those runs came from `load_jailbreak_mini()` /
  `load_xstest_mini()` -- **10 harmful / 8 harmless prompts**, far below this
  project's own data floor, and n=1 per cell. That is the single biggest threat to
  the finding: a direction estimated from 18 prompts could be mostly noise, and a
  noise direction might trivially favour the radial arm.

CITE
  Feng et al., 2026, "A Geometric Account of Activation Steering through Angle-Norm
  Decomposition" (arXiv:2606.06735) -- the angle/radius parameterisation under test.
  Dang & Ngo, 2026, "Selective Steering: Norm-Preserving Control" (arXiv:2601.19375)
  -- ships norm-preserving rotation but never holds displacement fixed, which is the
  control this experiment supplies. Real prompts come from JailbreakBench, HarmBench
  and AdvBench (harmful) and XSTest (harmless) via `steering.safety_bench`.

HYPOTHESIZE
  If the radial-beats-angular ordering is a property of the GEOMETRY rather than an
  artifact of a noisy 18-prompt direction, it must survive (a) a direction estimated
  from hundreds of real benchmark prompts and (b) resampling of the extraction set:
  every resample should reproduce the ordering.

PREDICT (pre-registered, before the run)
  1. PPL(r=0) > PPL(r=1) in >= 8/8 resamples at f=0.10  (the ordering is stable).
  2. The paired mean delta (rotation - addition) is POSITIVE with a bootstrap 95% CI
     excluding zero.
  3. A NORM-MATCHED RANDOM direction shows a SMALLER rotation-vs-addition gap than
     the real direction -- i.e. the effect is not purely geometric noise.

EXECUTE  — this file.  ANALYSE/CHECKPOINT -> autoresearch_results/ma_direction_seeds.json

OBJECTIVE: WikiText-2 perplexity. JUDGE-FREE (see autoresearch_results/JUDGE_CARD.md --
two judge calibrations failed at AUC 0.665 / 0.751 against a 0.85 gate).
========================================================================

Usage:
  PYTHONPATH=src python scripts/run_ma_direction_seeds.py --seeds 8 --layer 12
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from steering.hooks import _unit, get_residual_layers  # noqa: E402
from steering.model import load_model  # noqa: E402
from steering.real_metrics import wikitext_perplexity  # noqa: E402
from steering.safety_bench import load_safety_benchmark  # noqa: E402
from steering.safety_target import extract_refusal_direction  # noqa: E402
from steering.stats import power_note  # noqa: E402

DEFAULT_MODEL = "DavidAU/gemma-3-1b-it-heretic-extreme-uncensored-abliterated"


def load_pools(n_harm: int, n_benign: int) -> tuple[list[str], list[str]]:
    """Real benchmark prompts, pooled across three harmful sets + XSTest benign."""
    harm: list[str] = []
    for nm in ("jailbreakbench", "harmbench", "advbench"):
        try:
            harm += [it["prompt"] for it in load_safety_benchmark(nm, n=n_harm)]
        except Exception as exc:  # noqa: BLE001
            print(f"[data] {nm} unavailable: {type(exc).__name__}")
    benign = [it["prompt"] for it in load_safety_benchmark("xstest", n=n_benign)]
    return harm, benign


class AngleRadiusHook:
    """Displacement of magnitude f*||h|| split between angle (r=0) and radius (r=1)."""

    def __init__(self, v: torch.Tensor, f: float, r: float):
        self.v, self.f, self.r = v, float(f), float(r)

    def __call__(self, module, args, output):
        h = output[0] if isinstance(output, tuple) else output
        if self.f == 0.0:
            return output
        v_hat = _unit(self.v.to(dtype=h.dtype, device=h.device))
        hn = h.norm(dim=-1, keepdim=True)
        radial = self.f * self.r * hn * v_hat
        ang = self.f * (1.0 - self.r)
        if ang > 0:
            theta = 2.0 * math.asin(min(ang, 2.0) / 2.0)
            e1 = h / (hn + 1e-8)
            e2r = v_hat - torch.tensordot(e1, v_hat, dims=([-1], [0])).unsqueeze(-1) * e1
            e2 = e2r / (e2r.norm(dim=-1, keepdim=True) + 1e-8)
            h = hn * (math.cos(theta) * e1 + math.sin(theta) * e2)
        out = h + radial
        return (out,) + output[1:] if isinstance(output, tuple) else out


def ppl_with(model, tok, layer_mod, v, f, r, n_ppl) -> float:
    hk = layer_mod.register_forward_hook(AngleRadiusHook(v, f, r))
    try:
        return float(wikitext_perplexity(model, tok, n=n_ppl))
    finally:
        hk.remove()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--layer", type=int, default=12)
    ap.add_argument("--seeds", type=int, default=8, help="extraction resamples (>=8, see R6)")
    ap.add_argument("--budget", type=float, default=0.10)
    ap.add_argument("--n-harm", type=int, default=120)
    ap.add_argument("--n-benign", type=int, default=120)
    ap.add_argument("--n-ppl", type=int, default=25)
    ap.add_argument("--out", default=str(ROOT / "autoresearch_results" / "ma_direction_seeds.json"))
    args = ap.parse_args()

    harm, benign = load_pools(args.n_harm, args.n_benign)
    print(f"[data] harmful={len(harm)} harmless={len(benign)} (real benchmarks)")
    if len(harm) < 30 or len(benign) < 20:
        print("[data] pools too small; aborting rather than reporting a weak direction")
        return

    model, tok = load_model(args.model, quant="4bit")
    layer_mod = get_residual_layers(model)[args.layer]
    base = float(wikitext_perplexity(model, tok, n=args.n_ppl))
    print(f"[base] unsteered PPL = {base:.3f}")

    rng = np.random.default_rng(0)
    rows, t0 = [], time.time()
    for s in range(args.seeds):
        # resample the EXTRACTION set (not decode seeds -- greedy decoding has ~0 seed variance)
        hi = rng.choice(len(harm), size=min(len(harm), 60), replace=False)
        bi = rng.choice(len(benign), size=min(len(benign), 60), replace=False)
        v = torch.tensor(
            extract_refusal_direction(model, tok, [harm[i] for i in hi], [benign[i] for i in bi],
                                      layer=args.layer), dtype=torch.float32)
        # control: norm-matched RANDOM direction (2606.20852 protocol)
        vr = torch.randn_like(v)
        vr = vr / vr.norm() * v.norm()

        r_add = ppl_with(model, tok, layer_mod, v, args.budget, 1.0, args.n_ppl)
        r_rot = ppl_with(model, tok, layer_mod, v, args.budget, 0.0, args.n_ppl)
        c_add = ppl_with(model, tok, layer_mod, vr, args.budget, 1.0, args.n_ppl)
        c_rot = ppl_with(model, tok, layer_mod, vr, args.budget, 0.0, args.n_ppl)
        rows.append({"seed": s, "real_add": r_add, "real_rot": r_rot,
                     "rand_add": c_add, "rand_rot": c_rot,
                     "real_delta": r_rot - r_add, "rand_delta": c_rot - c_add})
        print(f"  seed {s}: real add={r_add:8.2f} rot={r_rot:9.2f} d={r_rot-r_add:+9.2f} | "
              f"rand d={c_rot-c_add:+9.2f}")

    d = np.array([r["real_delta"] for r in rows])
    dr = np.array([r["rand_delta"] for r in rows])
    boot = np.array([np.mean(rng.choice(d, len(d), replace=True)) for _ in range(10000)])
    ci = (float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5)))
    pw = power_note(len(d), effect=float(d.mean()), sd=float(d.std(ddof=1) or 1e-9), family_size=1)

    preds = {
        "P1_ordering_stable_all_seeds": bool((d > 0).all()),
        "P1_seeds_rotation_worse": int((d > 0).sum()),
        "P2_ci_excludes_zero": bool(ci[0] > 0),
        "P3_real_gap_exceeds_random_gap": bool(d.mean() > dr.mean()),
    }
    res = {"hypothesis": "radial-beats-angular survives a properly estimated direction",
           "citation": "arXiv:2606.06735; control protocol arXiv:2606.20852",
           "objective": "WikiText-2 perplexity (JUDGE-FREE)",
           "model": args.model, "layer": args.layer, "budget_f": args.budget,
           "n_harmful_pool": len(harm), "n_harmless_pool": len(benign),
           "seeds": len(d), "unsteered_ppl": base, "rows": rows,
           "real_delta_mean": float(d.mean()), "real_delta_ci95": ci,
           "random_delta_mean": float(dr.mean()),
           "power": pw, "predictions_evaluated": preds,
           "tier": "EVALUATION-eligible" if pw.get("can_reach_holm_alpha") else "SCREENING",
           "elapsed_s": round(time.time() - t0, 1)}
    Path(args.out).write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(f"\n[M-a] real delta {d.mean():+.2f} CI95 {ci} | random delta {dr.mean():+.2f}")
    print(f"[M-a] predictions {json.dumps(preds)}")
    print(f"[M-a] power: {pw['note']}")
    print(f"[write] {args.out}")


if __name__ == "__main__":
    main()
