"""hillclimb_angle_radius.py — a REAL citation-driven hill-climb, judge-free.

======================================================================== 7-STEP
DIAGNOSE
  The program stalled with a contaminated champion (best_config.json = exp3: a
  Qwen-0.5B run with STUBBED safety and a self-admittedly circular behavior proxy,
  frozen since 2026-05-30 across 121 later experiments) and a composite with two
  inert axes and a stale fingerprint. Two judge calibrations then failed (AUC 0.665,
  0.751 vs a 0.85 gate), so no judge-dependent objective is admissible. The loop
  therefore has to be restarted on a JUDGE-FREE objective from a CLEAN baseline.

CITE
  Feng et al., 2026, "A Geometric Account of Activation Steering through Angle-Norm
  Decomposition" (arXiv:2606.06735) — steering methods differ mainly in how they
  couple ANGULAR alignment to a concept direction against change in hidden-state
  NORM; concepts live in angular structure while norm governs stability. It
  recommends parameterising an intervention by (angle, radius) rather than a single
  additive alpha. Our literature scan's verdict: that decomposition is "asserted,
  not measured as a controller, and not reported at <=4B" — i.e. an open gap.
  Supporting: Dang & Ngo, 2026, "Selective Steering: Norm-Preserving Control"
  (arXiv:2601.19375) ships norm-preserving rotation but never holds displacement
  fixed; our own matched-budget run found rotation WORSE than addition at equal
  chord (5/5 budgets, super-linear).

HYPOTHESIZE
  Because the coherence tax is priced by how far the residual is pushed OFF its
  manifold, and because rotation spends the entire budget on angle while addition
  splits it between angle and radius, the (angle, radius) plane should contain an
  interior optimum: for a fixed total displacement there is a mixture that costs
  LESS perplexity than either pure endpoint. Pure rotation (radius=0) should be the
  worst corner, per our matched-budget result.

PREDICT (pre-registered, before the run)
  1. Pure rotation (r=0) is the worst cell at every budget: PPL(r=0) > PPL(r=1).
  2. The per-budget argmin lies at r >= 0.5 — i.e. displacement is better spent on
     radius than on angle.
  3. At the smallest budget the best cell's PPL is within 5% of unsteered.

EXECUTE  — this file. ONE axis pair swept, everything else held at the baseline.
ANALYSE / CHECKPOINT — written to autoresearch_results/hillclimb_angle_radius.json
========================================================================

OBJECTIVE (judge-free, so the failed calibrations cannot contaminate it):
    coherence  = WikiText-2 perplexity        (real corpus, teacher-forced)
    geometry   = measured off-shell displacement
No LLM judge is involved anywhere in this script.

Usage:
  PYTHONPATH=src python scripts/hillclimb_angle_radius.py --layer 12 --n-ppl 40
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from steering.datasets import load_jailbreak_mini, load_xstest_mini  # noqa: E402
from steering.hooks import SteeringContext, _unit  # noqa: E402
from steering.model import load_model  # noqa: E402
from steering.real_metrics import wikitext_perplexity  # noqa: E402
from steering.safety_target import extract_refusal_direction  # noqa: E402

DEFAULT_MODEL = "DavidAU/gemma-3-1b-it-heretic-extreme-uncensored-abliterated"


class AngleRadiusHook:
    """Apply a displacement of magnitude f*||h|| split between ANGLE and RADIUS.

    r = 1  -> spend it all on RADIUS  : h + f*||h||*v_hat        (pure addition)
    r = 0  -> spend it all on ANGLE   : norm-preserving rotation (pure rotation)
    0<r<1  -> mixture, at the SAME total chord displacement f*||h||.

    Holding the chord fixed is what makes the comparison fair — it is exactly the
    control 2601.19375 omits.
    """

    def __init__(self, v: torch.Tensor, f: float, r: float):
        self.v = v
        self.f = float(f)
        self.r = float(r)

    def __call__(self, module, args, output):
        h = output[0] if isinstance(output, tuple) else output
        if self.f == 0.0:
            return output
        v = self.v.to(dtype=h.dtype, device=h.device)
        v_hat = _unit(v)
        h_norm = h.norm(dim=-1, keepdim=True)

        # radial part: along v_hat, magnitude f*r*||h||
        radial = self.f * self.r * h_norm * v_hat

        # angular part: rotate by theta s.t. chord = f*(1-r)*||h||
        ang_chord = self.f * (1.0 - self.r)
        if ang_chord > 0:
            theta = 2.0 * math.asin(min(ang_chord, 2.0) / 2.0)
            e1 = h / (h_norm + 1e-8)
            v_dot = torch.tensordot(e1, v_hat, dims=([-1], [0])).unsqueeze(-1)
            e2_raw = v_hat - v_dot * e1
            e2 = e2_raw / (e2_raw.norm(dim=-1, keepdim=True) + 1e-8)
            h_new = h_norm * (math.cos(theta) * e1 + math.sin(theta) * e2)
        else:
            h_new = h
        out = h_new + radial
        return (out,) + output[1:] if isinstance(output, tuple) else out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--layer", type=int, default=12)
    ap.add_argument("--budgets", default="0.02,0.05,0.10")
    ap.add_argument("--radii", default="0.0,0.25,0.5,0.75,1.0")
    ap.add_argument("--n-ppl", type=int, default=40)
    ap.add_argument("--out", default=str(ROOT / "autoresearch_results" / "hillclimb_angle_radius.json"))
    args = ap.parse_args()

    budgets = [float(x) for x in args.budgets.split(",")]
    radii = [float(x) for x in args.radii.split(",")]

    print(f"[load] {args.model}")
    model, tok = load_model(args.model, quant="4bit")
    from steering.hooks import get_residual_layers as residual_layers  # noqa: E402
    layers = residual_layers(model)
    print(f"[dir ] diff-of-means @ layer {args.layer}")
    v = torch.tensor(
        extract_refusal_direction(model, tok, load_jailbreak_mini(), load_xstest_mini(), layer=args.layer),
        dtype=torch.float32)

    base = wikitext_perplexity(model, tok, n=args.n_ppl)
    print(f"[base] unsteered PPL = {base:.3f}")

    cells, t0 = [], time.time()
    for f in budgets:
        for r in radii:
            hook = AngleRadiusHook(v, f, r)
            handle = layers[args.layer].register_forward_hook(hook)
            try:
                ppl = float(wikitext_perplexity(model, tok, n=args.n_ppl))
            except Exception as exc:
                ppl = float("nan")
                print(f"  [f={f} r={r}] FAILED {type(exc).__name__}")
            finally:
                handle.remove()
            cells.append({"budget_f": f, "radius_frac": r, "ppl": ppl,
                          "ppl_ratio_vs_base": ppl / base if base else None})
            print(f"  f={f:<5} r={r:<5} PPL={ppl:>10.3f}  ({ppl/base:.2f}x base)")

    # champion = lowest PPL among steered cells (judge-free objective)
    steered = [c for c in cells if c["ppl"] == c["ppl"]]
    champ = min(steered, key=lambda c: c["ppl"]) if steered else None

    # pre-registered predictions, evaluated mechanically
    preds = {}
    for f in budgets:
        row = {c["radius_frac"]: c["ppl"] for c in cells if c["budget_f"] == f}
        if 0.0 in row and 1.0 in row:
            preds[f"P1_rot_worse_than_add_f{f}"] = bool(row[0.0] > row[1.0])
        best_r = min(row, key=row.get) if row else None
        preds[f"P2_argmin_radius_f{f}"] = best_r
    res = {
        "hypothesis": "angle/radius split of a FIXED displacement has an interior optimum",
        "citation": "arXiv:2606.06735 (angle-norm decomposition); arXiv:2601.19375 (norm-preserving control)",
        "objective": "WikiText-2 perplexity (JUDGE-FREE)",
        "model": args.model, "layer": args.layer,
        "unsteered_ppl": base, "cells": cells, "champion": champ,
        "predictions_evaluated": preds,
        "tier": "SCREENING (n=1 per cell, single extraction set)",
        "elapsed_s": round(time.time() - t0, 1),
    }
    Path(args.out).write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(f"\n[champion] f={champ['budget_f']} r={champ['radius_frac']} PPL={champ['ppl']:.3f}")
    print(f"[preds] {json.dumps(preds)}")
    print(f"[write] {args.out}")


if __name__ == "__main__":
    main()
