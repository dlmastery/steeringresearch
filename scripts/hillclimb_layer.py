"""hillclimb_layer.py -- HC-3: coordinate descent on the LAYER axis, judge-free.

======================================================================== 7-STEP
DIAGNOSE
  `autoresearch_results/best_config.json` names the open axes explicitly:
  "layer": "only layer 12 tested; the cube's layer axis is UNEXPLORED". HC-1
  (`hillclimb_angle_radius.json`) swept budget_f x radius_frac at a SINGLE layer
  (12) and landed on f=0.05, r=1.0, PPL 87.652 = 0.949x base. M-a
  (`FINDING_Ma_matched_budget_n8.md`) then confirmed the radius-beats-angle
  ordering at n=8 with directions from 340 harmful / 120 harmless real benchmark
  prompts -- but its own limitations section lists "one layer (12)" as threat #1.
  Layer 12 was never chosen; it was inherited. This experiment climbs that axis.

CITE
  Feng et al., 2026, "A Geometric Account of Activation Steering through
  Angle-Norm Decomposition" (arXiv:2606.06735) -- the (angle, radius)
  parameterisation held fixed here at r=1 (pure radius), so LAYER is the only
  variable that moves. Random norm-matched control protocol from arXiv:2606.20852,
  the same control that made M-a's P3 decisive.

HYPOTHESIZE
  The coherence tax of a fixed FRACTIONAL displacement (f=0.05 of ||h||) is not
  constant in depth, because the number of downstream blocks that can amplify the
  perturbation falls monotonically with the injection layer. Mechanism: an edit at
  layer l is re-processed by (L - l) attention+MLP blocks, each able to compound
  it; per the angle-norm account the residual is most load-bearing mid-stack, so
  the same relative displacement should cost most in the early/mid range and
  nearly nothing at the top. If that is right, PPL should FALL with depth over the
  upper half of the stack -- and the naive hill-climb would then "win" by
  injecting where steering does nothing, which is why the random control is
  mandatory rather than optional.

PREDICT (pre-registered, written BEFORE the run)
  P1 (the axis is LIVE, not numerology): max - min real-direction PPL across the
     grid EXCEEDS 0.10 * unsteered base PPL. If it does not, any nearby layer
     would do and the axis is NUMEROLOGY per CLAUDE.md section 7.
  P2 (depth mechanism): the argmin layer is >= 17 -- late injection is cheaper
     because fewer downstream blocks remain to amplify the displacement.
  P3 (the sceptical crux): at the argmin layer the real direction and the
     norm-matched RANDOM direction differ by LESS than 0.05 * base PPL -- i.e.
     the cheapest layer is a NULL layer where the perplexity win is not
     direction-specific and therefore is not a steering win at all.
  P4 (numeric range): the best layer's ppl_ratio_vs_base lies in [0.85, 0.99].

  Falsifier for the whole experiment: if no layer's real PPL beats layer 12's by
  more than the seed-noise band, the champion's layer stands and the axis is
  reported as EXHAUSTED-AT-12, not as a win.

EXECUTE -- this file, two stages, per CLAUDE.md section 8's funnel:
  stage `screen`  : n=3 extraction resamples over the full layer grid (SCREENING;
                    may NEVER be called a win) -> picks ONE candidate layer.
  stage `confirm` : n=8 extraction resamples on {candidate, 12} ONLY, family
                    size m=1, disjoint seed stream. Paired delta + 10k-resample
                    bootstrap CI. This is the only stage that may carry a claim.
ANALYSE / CHECKPOINT -> autoresearch_results/hillclimb_layer[_confirm].json

PRE-REGISTRATION OF THE CONFIRM STAGE (written after `screen`, before `confirm`)
  The n=3 screen (`hillclimb_layer.json`, seeds 0-2) gives real-direction mean PPL
  L11 73.283 < L17 73.940 < L20 74.025 < L12 75.044 < L2 76.113 < L8 76.869
  < L14 76.958 < L23 77.287 < L5 78.622 < L25 81.896, base 77.310. The argmin is
  LAYER 11, one block below the incumbent, and it beats layer 12 in 3/3 seeds.
  THE SINGLE COMPARISON CARRIED FORWARD IS THEREFORE **layer 11 vs layer 12**,
  family size m = 1, n = 8 fresh extraction resamples (seeds 1000-1007, disjoint
  from the screen). Success criterion, fixed now: the paired delta
  (PPL_L11 - PPL_L12) is NEGATIVE with a 10k-resample bootstrap 95% CI EXCLUDING
  ZERO, and layer 11's real-vs-random control gap stays large (not a null layer).
  Anything else -> the axis is reported as EXHAUSTED-AT-12 and the champion stands.
  L17 and L20 also screened below L12 but are NOT confirmed here; declaring three
  comparisons would force Holm m=3 (alpha 0.0167), which n=8 (min p 0.0078) can
  still reach, but the coordinate-descent rule is one step at a time -- they
  remain open cells, logged as such.

WHAT IS HELD FIXED (one config change, CLAUDE.md section 1)
  budget_f = 0.05, radius_frac = 1.0 (pure relative_add, NO rotation), diffmean
  source, WikiText perplexity objective, same model. ONLY the layer moves. The
  direction is re-extracted AT the injection layer, because a diffmean direction
  is defined in the basis of the layer it is read from -- carrying layer 12's
  vector to layer 20 would change the SOURCE as well as the layer and would no
  longer be a single-axis perturbation.

OBJECTIVE: WikiText perplexity, teacher-forced. JUDGE-FREE -- see
`autoresearch_results/JUDGE_CARD.md`: two judge calibrations failed at AUC 0.665
and 0.751 against a 0.85 gate, so no judge-dependent objective is admissible.

Usage:
  PYTHONPATH=src python scripts/hillclimb_layer.py --stage screen  --seeds 3
  PYTHONPATH=src python scripts/hillclimb_layer.py --stage confirm --seeds 8 \
      --layers <candidate>,12
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from steering.hooks import _unit, get_residual_layers, probe_activations  # noqa: E402
from steering.model import encode_to_device, load_model  # noqa: E402
from steering.real_metrics import wikitext_perplexity  # noqa: E402
from steering.safety_bench import load_safety_benchmark  # noqa: E402
from steering.stats import power_note  # noqa: E402

DEFAULT_MODEL = "DavidAU/gemma-3-1b-it-heretic-extreme-uncensored-abliterated"

# Grid justification (Gemma-3-1B has 26 residual blocks, indices 0..25):
#   every ~3 layers across the full depth so no region of the stack is unsampled,
#   PLUS the incumbent layer 12 and its immediate neighbours 11/14 so the local
#   curvature around the champion is resolved (a coordinate-descent step needs to
#   know whether 12 sits in a basin or on a slope), PLUS the top layer 25 as the
#   endpoint of the depth mechanism in P2.
DEFAULT_GRID = "2,5,8,11,12,14,17,20,23,25"


class RadialHook:
    """Pure radial displacement of magnitude f*||h|| along v_hat.

    This is radius_frac = 1.0 of the champion's (angle, radius) parameterisation,
    i.e. `hooks.apply_operation(..., "relative_add", f)` -- identical math, kept
    local so this script never mutates a file another experiment depends on.
    """

    def __init__(self, v: torch.Tensor, f: float):
        self.v, self.f = v, float(f)

    def __call__(self, module, args, output):
        h = output[0] if isinstance(output, tuple) else output
        if self.f == 0.0:
            return output
        v_hat = _unit(self.v.to(dtype=h.dtype, device=h.device))
        out = h + self.f * h.norm(dim=-1, keepdim=True) * v_hat
        return (out,) + output[1:] if isinstance(output, tuple) else out


def load_pools(n_harm: int, n_benign: int) -> tuple[list[str], list[str]]:
    """Real benchmark prompts -- the same pools M-a used (no mini/toy sets)."""
    harm: list[str] = []
    for nm in ("jailbreakbench", "harmbench", "advbench"):
        try:
            harm += [it["prompt"] for it in load_safety_benchmark(nm, n=n_harm)]
        except Exception as exc:  # noqa: BLE001
            print(f"[data] {nm} unavailable: {type(exc).__name__}", flush=True)
    benign = [it["prompt"] for it in load_safety_benchmark("xstest", n=n_benign)]
    return harm, benign


def _pooled_mean(model, tok, texts, layers: list[int]) -> dict[int, np.ndarray]:
    """Mean over texts of the mean-pooled residual at EACH layer, in ONE pass.

    `safety_target.extract_refusal_direction` reads a single layer, so sweeping L
    layers with it costs L full passes over the prompt pool. Every layer's
    activation is available in the SAME forward, so we capture them all at once:
    identical arithmetic (mean-pool over the sequence, then mean over texts),
    L-fold cheaper. That is what makes n=8 resamples affordable on one 4090.
    """
    acc: dict[int, np.ndarray] = {}
    for text in texts:
        ids = encode_to_device(tok, text, model)
        acts = probe_activations(model, ids, layers)
        for lay in layers:
            p = acts[lay][0].mean(dim=0).float().cpu().numpy().astype(np.float64)
            acc[lay] = p if lay not in acc else acc[lay] + p
    return {lay: acc[lay] / len(texts) for lay in layers}


def extract_directions(model, tok, harm_texts, benign_texts,
                       layers: list[int]) -> dict[int, np.ndarray]:
    """Unit diffmean (harmful - harmless) at every requested layer."""
    mh = _pooled_mean(model, tok, harm_texts, layers)
    mb = _pooled_mean(model, tok, benign_texts, layers)
    out = {}
    for lay in layers:
        d = (mh[lay] - mb[lay]).astype(np.float32)
        out[lay] = d / (float(np.linalg.norm(d)) + 1e-8)
    return out


def ppl_with(model, tok, layer_mod, v: torch.Tensor, f: float, n_ppl: int) -> float:
    hk = layer_mod.register_forward_hook(RadialHook(v, f))
    try:
        return float(wikitext_perplexity(model, tok, n=n_ppl))
    finally:
        hk.remove()


def bootstrap_ci(x: np.ndarray, rng, n_boot: int = 10_000) -> list[float]:
    boot = np.array([np.mean(rng.choice(x, len(x), replace=True)) for _ in range(n_boot)])
    return [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--stage", choices=("screen", "confirm"), default="screen")
    ap.add_argument("--layers", default=DEFAULT_GRID)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--budget", type=float, default=0.05, help="CHAMPION value; do not move")
    ap.add_argument("--radius", type=float, default=1.0, help="CHAMPION value; do not move")
    ap.add_argument("--n-harm", type=int, default=120)
    ap.add_argument("--n-benign", type=int, default=120)
    ap.add_argument("--n-extract", type=int, default=60, help="prompts per class per resample")
    ap.add_argument("--n-ppl", type=int, default=25, help="WikiText passages (M-a protocol)")
    ap.add_argument("--ref-layer", type=int, default=12, help="the incumbent champion layer")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    grid = [int(x) for x in args.layers.split(",")]
    out_path = Path(args.out) if args.out else (
        ROOT / "autoresearch_results" /
        ("hillclimb_layer.json" if args.stage == "screen" else "hillclimb_layer_confirm.json"))
    part_path = out_path.with_suffix(".partial.json")
    # Disjoint seed streams: the confirm stage must NOT reuse the resamples that
    # selected the candidate, or the comparison is contaminated by the selection.
    seed_base = 0 if args.stage == "screen" else 1000

    harm, benign = load_pools(args.n_harm, args.n_benign)
    print(f"[data] harmful={len(harm)} harmless={len(benign)} (real benchmarks)", flush=True)
    if len(harm) < 60 or len(benign) < 20:
        print("[data] pools too small; aborting rather than reporting a weak direction")
        return

    model, tok = load_model(args.model, quant="4bit")
    layer_mods = get_residual_layers(model)
    n_layers = len(layer_mods)
    grid = [lay for lay in grid if 0 <= lay < n_layers]
    print(f"[model] {n_layers} residual layers; grid = {grid}", flush=True)

    base = float(wikitext_perplexity(model, tok, n=args.n_ppl))
    print(f"[base] unsteered PPL = {base:.3f}  (n_ppl={args.n_ppl})", flush=True)

    rows: list[dict] = []
    if part_path.exists():
        rows = json.loads(part_path.read_text(encoding="utf-8")).get("rows", [])
        print(f"[resume] {len(rows)} cells already measured; skipping those", flush=True)
    done = {(r["seed"], r["layer"]) for r in rows}

    rng = np.random.default_rng(seed_base)
    t0 = time.time()
    for s in range(args.seeds):
        # DIRECTION CACHE. Extraction of every grid layer happens before the first
        # cell of a seed can checkpoint; a reap inside that window would otherwise
        # cost the whole extraction. Key includes the grid and the layer count so a
        # different grid can never silently reuse the wrong file (18.4: stamp your
        # inputs -- an artifact that cannot be regenerated from the code beside it
        # is not evidence).
        gkey = "-".join(map(str, grid))
        dcache = (ROOT / "autoresearch_results" /
                  f".dircache_layer_{args.stage}_s{seed_base + s}_{gkey}.npz")
        # advance the RNG identically whether or not the cache hits, so seeds are
        # reproducible across a resumed run
        hi = rng.choice(len(harm), size=min(len(harm), args.n_extract), replace=False)
        bi = rng.choice(len(benign), size=min(len(benign), args.n_extract), replace=False)
        if dcache.exists():
            z = np.load(dcache)
            dirs = {int(k): z[k] for k in z.files}
            print(f"  [seed {s}] directions from cache", flush=True)
        else:
            dirs = extract_directions(model, tok, [harm[i] for i in hi],
                                      [benign[i] for i in bi], grid)
            np.savez(dcache, **{str(k): v for k, v in dirs.items()})
            print(f"  [seed {s}] extracted {len(dirs)} directions "
                  f"({len(hi)} harmful / {len(bi)} harmless)", flush=True)

        for lay in grid:
            if (s, lay) in done:
                continue
            v = torch.tensor(dirs[lay], dtype=torch.float32)
            # norm-matched RANDOM control (arXiv:2606.20852 protocol, as in M-a):
            # same layer, same magnitude, meaningless direction. A layer where the
            # two arms agree is a layer where steering does NOTHING SPECIFIC.
            g = torch.Generator().manual_seed(10_000 * (seed_base + s) + lay)
            vr = torch.randn(v.shape, generator=g, dtype=torch.float32)
            vr = vr / vr.norm() * v.norm()

            real = ppl_with(model, tok, layer_mods[lay], v, args.budget, args.n_ppl)
            rand = ppl_with(model, tok, layer_mods[lay], vr, args.budget, args.n_ppl)
            rows.append({"seed": seed_base + s, "layer": lay, "real_ppl": real,
                         "rand_ppl": rand, "real_ratio": real / base,
                         "rand_ratio": rand / base, "real_minus_rand": real - rand})
            print(f"    L{lay:<3} real={real:9.3f} ({real / base:.3f}x)  "
                  f"rand={rand:9.3f} ({rand / base:.3f}x)  "
                  f"real-rand={real - rand:+8.3f}", flush=True)
            # CHECKPOINT after EVERY cell -- this host reaps long jobs (18.5).
            part_path.write_text(json.dumps(
                {"unsteered_ppl": base, "budget_f": args.budget,
                 "radius_frac": args.radius, "rows": rows}, indent=2), encoding="utf-8")

    # ---------------- analysis ----------------
    by_layer: dict[str, dict] = {}
    for lay in grid:
        r = np.array([x["real_ppl"] for x in rows if x["layer"] == lay])
        q = np.array([x["rand_ppl"] for x in rows if x["layer"] == lay])
        if len(r) == 0:
            continue
        gap = r - q
        by_layer[str(lay)] = {
            "n_seeds": int(len(r)),
            "real_ppl_mean": float(r.mean()),
            "real_ppl_sd": float(r.std(ddof=1)) if len(r) > 1 else None,
            "real_ratio_mean": float(r.mean() / base),
            "rand_ppl_mean": float(q.mean()),
            "rand_ratio_mean": float(q.mean() / base),
            "real_minus_rand_mean": float(gap.mean()),
            "real_minus_rand_ci95": bootstrap_ci(gap, rng) if len(r) > 1 else None,
            # NULL LAYER = real and random cost the same -> not direction-specific
            "null_layer_gap_below_2pct_base": bool(abs(gap.mean()) < 0.02 * base),
        }

    ref = by_layer.get(str(args.ref_layer))
    means = {lay: by_layer[str(lay)]["real_ppl_mean"] for lay in grid if str(lay) in by_layer}
    best_layer = min(means, key=means.get) if means else None

    # paired (within-seed) delta of every layer against the incumbent
    paired: dict[str, dict] = {}
    ref_by_seed = {x["seed"]: x["real_ppl"] for x in rows if x["layer"] == args.ref_layer}
    for lay in grid:
        if lay == args.ref_layer:
            continue
        d = np.array([x["real_ppl"] - ref_by_seed[x["seed"]] for x in rows
                      if x["layer"] == lay and x["seed"] in ref_by_seed])
        if len(d) == 0:
            continue
        paired[str(lay)] = {
            "n_pairs": int(len(d)),
            "delta_mean_vs_ref": float(d.mean()),      # negative = better than layer 12
            "delta_ci95": bootstrap_ci(d, rng) if len(d) > 1 else None,
            "beats_ref_all_seeds": bool((d < 0).all()),
        }

    spread = (max(means.values()) - min(means.values())) if means else 0.0
    best_gap = abs(by_layer[str(best_layer)]["real_minus_rand_mean"]) if best_layer is not None else None
    preds = {
        "P1_axis_live_spread_gt_10pct_base": bool(spread > 0.10 * base),
        "P1_spread_ppl": float(spread),
        "P2_argmin_layer_ge_17": bool(best_layer is not None and best_layer >= 17),
        "P2_argmin_layer": best_layer,
        "P3_best_layer_is_NULL_real_vs_rand_lt_5pct_base": (
            bool(best_gap < 0.05 * base) if best_gap is not None else None),
        "P3_best_layer_real_minus_rand": best_gap,
        "P4_best_ratio_in_0.85_0.99": (
            bool(0.85 <= by_layer[str(best_layer)]["real_ratio_mean"] <= 0.99)
            if best_layer is not None else None),
    }

    # family size: screening compares every non-incumbent layer (m = grid-1);
    # confirm pre-registers ONE comparison (m = 1). power_note prices both.
    fam = 1 if args.stage == "confirm" else max(1, len(grid) - 1)
    dref = (np.array([paired[k]["delta_mean_vs_ref"] for k in paired])
            if paired else np.array([0.0]))
    pw = power_note(args.seeds, effect=float(abs(dref).max()), sd=1.0, family_size=fam)

    res = {
        "hypothesis": "HC-3 -- coordinate descent on the LAYER axis at the champion "
                      "(budget_f=0.05, radius_frac=1.0)",
        "citation": "arXiv:2606.06735 (angle-norm decomposition); random control "
                    "protocol arXiv:2606.20852",
        "objective": "WikiText perplexity (JUDGE-FREE; JUDGE_CARD.md AUC 0.665/0.751 < 0.85 gate)",
        "stage": args.stage,
        "tier": ("SCREENING (n<=3, family m=%d; may NOT be called a win)" % fam
                 if args.stage == "screen" else
                 "EVALUATION-eligible" if pw.get("can_reach_holm_alpha") else "SCREENING"),
        "model": args.model, "n_residual_layers": n_layers,
        "held_fixed": {"budget_f": args.budget, "radius_frac": args.radius,
                       "source": "diffmean (harmful - harmless)",
                       "operation": "relative_add (pure radius)"},
        "layer_grid": grid, "ref_layer": args.ref_layer,
        "seeds": args.seeds, "seed_base": seed_base,
        "n_harmful_pool": len(harm), "n_harmless_pool": len(benign),
        "n_extract_per_class": args.n_extract, "n_ppl": args.n_ppl,
        "unsteered_ppl": base,
        "rows": rows, "by_layer": by_layer, "paired_vs_ref": paired,
        "best_layer": best_layer,
        "predictions_evaluated": preds, "power": pw,
        "elapsed_s": round(time.time() - t0, 1),
    }
    out_path.write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(f"\n[HC-3] best layer = {best_layer} "
          f"(PPL {means[best_layer]:.3f}, {means[best_layer] / base:.3f}x base)")
    print(f"[HC-3] predictions {json.dumps(preds)}")
    print(f"[HC-3] power: {pw['note']}")
    print(f"[write] {out_path}")


if __name__ == "__main__":
    main()
