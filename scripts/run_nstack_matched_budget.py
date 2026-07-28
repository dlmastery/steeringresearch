"""run_nstack_matched_budget.py -- M-b: does radial-beats-angular SURVIVE STACKING?

======================================================================== 7-STEP
DIAGNOSE
  M-a (`FINDING_Ma_matched_budget_n8.md`) established, at n=8 with directions from 340
  harmful / 120 harmless REAL benchmark prompts, that at a FIXED chord displacement
  spending the budget on RADIUS beats spending it on ANGLE: rotation costs +91.24 PPL
  more than addition, CI95 [87.98, 94.71], ordering stable 8/8, and a norm-matched
  RANDOM control shows only +16.80 -- so the effect is direction-specific.

  But M-a is a SINGLE-VECTOR result, and that is precisely where it is weakest as a
  challenge to the literature. GEMS (arXiv:2606.19946) and ORBIT (arXiv:2606.22357)
  motivate norm preservation for **stacking** -- the argument is that when several
  edits compose, uncontrolled norm growth compounds, so rotation should pay off as N
  grows even if it loses at N=1. M-a cannot speak to that. This experiment can.

CITE
  Feng et al., 2026, "A Geometric Account of Activation Steering through Angle-Norm
  Decomposition" (arXiv:2606.06735) -- the angle/radius parameterisation under test.
  GEMS (arXiv:2606.19946) and ORBIT (arXiv:2606.22357) -- both build multi-vector
  steering on norm preservation; this is the regime their argument is actually about.
  Random-control protocol from arXiv:2606.20852.

HYPOTHESIZE
  If norm preservation earns its keep by preventing compounding norm growth, the
  angular arm's DISADVANTAGE must SHRINK as N grows, and ideally invert. Mechanism:
  N additive edits at a matched total budget grow ||h|| in quadrature while N rotations
  hold it fixed, so per GEMS the additive arm should degrade faster with N. If instead
  the gap is flat or WIDENS with N, the norm-preservation premise fails in exactly the
  regime it was proposed for.

PREDICT (pre-registered, before the run)
  P1. The additive arm beats the angular arm at EVERY N in {1,2,4,8} (ordering holds).
  P2. The gap (PPL_angular - PPL_additive) does NOT shrink monotonically with N.
      Under the GEMS/ORBIT premise it should; the M-a geometry says it should not.
  P3. Unsteered-relative PPL of the additive arm stays below 1.5x base at N=8, i.e.
      stacking at a matched TOTAL budget does not itself destroy coherence.

EXECUTE -- this file.  ANALYSE/CHECKPOINT -> autoresearch_results/nstack_matched_budget.json

MATCHED-BUDGET CONSTRUCTION
  N directions are made ORTHONORMAL (Gram-Schmidt), and each carries magnitude
  f/sqrt(N) so the TOTAL displacement is f regardless of N. Without this the
  comparison would confound "more vectors" with "more displacement" -- the same defect
  the voice program's V2 hit with an unmatched variance control.

OBJECTIVE: WikiText-2 perplexity. JUDGE-FREE (autoresearch_results/JUDGE_CARD.md --
two judge calibrations failed at AUC 0.665 / 0.751 against a 0.85 gate).
========================================================================

Usage:
  PYTHONPATH=src python scripts/run_nstack_matched_budget.py --seeds 5 --layer 12
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
    harm: list[str] = []
    for nm in ("jailbreakbench", "harmbench", "advbench"):
        try:
            harm += [it["prompt"] for it in load_safety_benchmark(nm, n=n_harm)]
        except Exception as exc:  # noqa: BLE001
            print(f"[data] {nm} unavailable: {type(exc).__name__}")
    benign = [it["prompt"] for it in load_safety_benchmark("xstest", n=n_benign)]
    return harm, benign


class NStackHook:
    """Apply N orthonormal directions at a MATCHED TOTAL displacement of f*||h||.

    Each direction carries f/sqrt(N), so orthogonal contributions add in quadrature to
    exactly f. r=1 spends the whole budget on radius (addition); r=0 spends it on angle
    (norm-preserving rotation). Rotations are applied sequentially, each in the plane
    spanned by the current state and its direction.
    """

    def __init__(self, V: torch.Tensor, f: float, r: float):
        self.V, self.f, self.r = V, float(f), float(r)      # V: (N, d)

    def __call__(self, module, args, output):
        h = output[0] if isinstance(output, tuple) else output
        if self.f == 0.0:
            return output
        N = self.V.shape[0]
        per = self.f / math.sqrt(N)                          # quadrature-matched share
        Vh = self.V.to(dtype=h.dtype, device=h.device)
        radial = torch.zeros_like(h)
        for i in range(N):
            v_hat = _unit(Vh[i])
            hn = h.norm(dim=-1, keepdim=True)
            radial = radial + per * self.r * hn * v_hat
            ang = per * (1.0 - self.r)
            if ang > 0:
                theta = 2.0 * math.asin(min(ang, 2.0) / 2.0)
                e1 = h / (hn + 1e-8)
                e2r = v_hat - torch.tensordot(e1, v_hat, dims=([-1], [0])).unsqueeze(-1) * e1
                e2 = e2r / (e2r.norm(dim=-1, keepdim=True) + 1e-8)
                h = hn * (math.cos(theta) * e1 + math.sin(theta) * e2)
        out = h + radial
        return (out,) + output[1:] if isinstance(output, tuple) else out


def orthonormalize(V: np.ndarray) -> np.ndarray:
    """Gram-Schmidt. Orthogonality is what makes the quadrature budget exact."""
    out = []
    for v in V:
        w = v.astype(np.float64).copy()
        for u in out:
            w -= (w @ u) * u
        n = np.linalg.norm(w)
        if n > 1e-8:
            out.append(w / n)
    return np.stack(out) if out else V


def ppl_with(model, tok, layer_mod, V, f, r, n_ppl) -> float:
    hk = layer_mod.register_forward_hook(NStackHook(V, f, r))
    try:
        return float(wikitext_perplexity(model, tok, n=n_ppl))
    finally:
        hk.remove()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--layer", type=int, default=12)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--budget", type=float, default=0.10)
    ap.add_argument("--ns", default="1,2,4,8")
    ap.add_argument("--n-ppl", type=int, default=20)
    ap.add_argument("--out", default=str(ROOT / "autoresearch_results" / "nstack_matched_budget.json"))
    args = ap.parse_args()
    NS = [int(x) for x in args.ns.split(",")]

    harm, benign = load_pools(120, 120)
    print(f"[data] harmful={len(harm)} harmless={len(benign)} (real benchmarks)", flush=True)
    if len(harm) < 60 or len(benign) < 20:
        print("[data] pools too small; aborting rather than reporting a weak direction")
        return

    model, tok = load_model(args.model, quant="4bit")
    layer_mod = get_residual_layers(model)[args.layer]
    base = float(wikitext_perplexity(model, tok, n=args.n_ppl))
    print(f"[base] unsteered PPL = {base:.3f}", flush=True)

    rng = np.random.default_rng(0)
    rows, t0 = [], time.time()
    # resume: reload any cells a previous (reaped) attempt already measured
    part = Path(args.out).with_suffix(".partial.json")
    if part.exists():
        rows = json.loads(part.read_text(encoding="utf-8")).get("rows", [])
        print(f"[resume] {len(rows)} cells already measured; skipping those", flush=True)
    done = {(r["seed"], r["N"]) for r in rows}
    for s in range(args.seeds):
        # N directions from DISJOINT harmful slices -- genuinely different estimates,
        # not N copies of one direction with noise
        hi = rng.permutation(len(harm))
        bi = rng.choice(len(benign), size=min(len(benign), 60), replace=False)
        bset = [benign[i] for i in bi]
        maxN = max(NS)
        chunk = max(20, len(harm) // maxN)
        dirs = []
        for j in range(maxN):
            sl = [harm[i] for i in hi[j * chunk:(j + 1) * chunk]]
            if len(sl) < 10:
                sl = [harm[i] for i in hi[:chunk]]
            dirs.append(extract_refusal_direction(model, tok, sl, bset, layer=args.layer))
        D = orthonormalize(np.stack(dirs))
        print(f"  [seed {s}] {D.shape[0]} orthonormal directions", flush=True)

        for N in NS:
            if D.shape[0] < N or (s, N) in done:
                continue
            V = torch.tensor(D[:N], dtype=torch.float32)
            add = ppl_with(model, tok, layer_mod, V, args.budget, 1.0, args.n_ppl)
            rot = ppl_with(model, tok, layer_mod, V, args.budget, 0.0, args.n_ppl)
            rows.append({"seed": s, "N": N, "add": add, "rot": rot, "delta": rot - add,
                         "add_ratio": add / base})
            print(f"    N={N}: add={add:8.2f} rot={rot:9.2f} gap={rot - add:+9.2f} "
                  f"(add {add / base:.3f}x base)", flush=True)
            # CHECKPOINT after every cell. Background jobs get reaped on this host --
            # this one died during model load on its first attempt -- and a run that
            # writes only at the end can never finish. Resume reads this file and skips
            # (seed, N) pairs already measured.
            Path(args.out).with_suffix(".partial.json").write_text(
                json.dumps({"unsteered_ppl": base, "rows": rows}, indent=2),
                encoding="utf-8")

    by_n = {}
    for N in NS:
        d = np.array([r["delta"] for r in rows if r["N"] == N])
        a = np.array([r["add_ratio"] for r in rows if r["N"] == N])
        if len(d) == 0:
            continue
        boot = np.array([np.mean(rng.choice(d, len(d), replace=True)) for _ in range(10_000)])
        by_n[str(N)] = {"n_seeds": int(len(d)), "gap_mean": float(d.mean()),
                        "gap_ci95": [float(np.percentile(boot, 2.5)),
                                     float(np.percentile(boot, 97.5))],
                        "additive_wins_all_seeds": bool((d > 0).all()),
                        "add_ratio_mean": float(a.mean())}

    gaps = [by_n[str(N)]["gap_mean"] for N in NS if str(N) in by_n]
    preds = {
        "P1_additive_wins_at_every_N": all(by_n[k]["additive_wins_all_seeds"] for k in by_n),
        "P2_gap_does_NOT_shrink_monotonically": not all(
            gaps[i] > gaps[i + 1] for i in range(len(gaps) - 1)),
        "P3_additive_ratio_below_1.5x_at_maxN": bool(
            by_n[str(max(NS))]["add_ratio_mean"] < 1.5) if str(max(NS)) in by_n else None,
    }
    pw = power_note(args.seeds, effect=float(np.mean(gaps)) if gaps else 0.0,
                    sd=1.0, family_size=len(NS))
    res = {"hypothesis": "M-b -- does radial-beats-angular survive STACKING?",
           "citation": "arXiv:2606.06735; challenges arXiv:2606.19946 (GEMS) and arXiv:2606.22357 (ORBIT)",
           "objective": "WikiText-2 perplexity (JUDGE-FREE)", "model": args.model,
           "layer": args.layer, "total_budget_f": args.budget,
           "budget_construction": "N orthonormal directions, each f/sqrt(N) -> total displacement f for all N",
           "unsteered_ppl": base, "Ns": NS, "seeds": args.seeds,
           "rows": rows, "by_N": by_n, "predictions_evaluated": preds, "power": pw,
           "elapsed_s": round(time.time() - t0, 1)}
    Path(args.out).write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(f"\n[M-b] gaps by N: {[(N, round(by_n[str(N)]['gap_mean'], 2)) for N in NS if str(N) in by_n]}")
    print(f"[M-b] predictions {json.dumps(preds)}")
    print(f"[write] {args.out}")


if __name__ == "__main__":
    main()
