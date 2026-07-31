"""hillclimb_source.py -- HC-S: the SOURCE axis of the steering cube.

======================================================================== 7-STEP
DIAGNOSE
  `autoresearch_results/best_config.json` names four open hill-climb axes and says of
  this one: "source: diffmean only; PCA source untested against it at matched budget."
  Every result the program has ever produced -- HC-1, HC-2, M-a (n=8), M-b (n=8) -- used
  ONE direction estimator, mean(harmful) - mean(benign) (`safety_target.
  extract_refusal_direction`). CLAUDE.md section 8 lists `source[diffmean/PCA]` as a
  coordinate of the hill-climb cube, so the champion has never been challenged on it.
  M-a's own limitation 3 says the CI covers direction-ESTIMATION variance; that is
  exactly the variance this experiment interrogates, by swapping the estimator while
  holding the extraction pool fixed.

CITE
  Zou et al., 2023 NeurIPS-w, "Representation Engineering: A Top-Down Approach to AI
  Transparency" (arXiv:2310.01405) -- introduces PCA over PAIRED-STIMULUS DIFFERENCE
  vectors (LAT) as the reading direction, the construction `pca_centered` implements.
  Rimsky et al., 2024 ACL, "Steering Llama 2 via Contrastive Activation Addition"
  (arXiv:2312.06681) -- the diff-of-means champion estimator.
  Marks & Tegmark, 2023, "The Geometry of Truth" (arXiv:2310.06824) -- compares
  diff-of-means against PCA/LDA-style probes and argues mass-mean (diffmean) recovers
  the causally relevant direction where PCA can lock onto a high-variance nuisance axis.
  Random-control protocol from arXiv:2606.20852 (as used by M-a). [UNVERIFIED arXiv ids
  are marked as such; the four above are long-standing and used elsewhere in this repo.]

HYPOTHESIZE
  Because the four estimators differ only in how they WEIGHT the same pooled
  activations, and because all four are unit-normalised and injected at the identical
  budget (layer 12, f=0.05, r=1.0), any perplexity difference is attributable to
  estimator quality and nothing else. Mechanism: diffmean is the first moment of the
  contrast; uncentered PCA of the difference matrix maximises mean-squared projection
  and is therefore DOMINATED by that same first moment, so it should land near-parallel
  to diffmean; centered PCA removes the first moment by construction and reads the
  dominant axis of VARIATION instead, which per Marks & Tegmark may be a nuisance axis;
  LDA rescales the contrast by the inverse within-class covariance, which at d >> n is
  ill-conditioned and must be shrunk.

PREDICT (pre-registered, written before the run)
  P1. cos(diffmean, pca_uncentered) > 0.90 -- the two are effectively the SAME direction,
      so their PPL difference is noise, and any paired CI on it should straddle zero.
  P2. cos(diffmean, pca_centered) < 0.50 -- centering removes the contrast mean, so this
      is a genuinely different direction.
  P3. No estimator beats diffmean by a paired margin whose bootstrap CI95 excludes zero
      (i.e. the champion's SOURCE coordinate survives; the axis is a null).
  P4. The norm-matched RANDOM floor is worse than diffmean (positive paired delta).

EXECUTE -- this file.  ANALYSE/CHECKPOINT -> autoresearch_results/hillclimb_source.json

WHAT MAKES THE ARMS COMPARABLE
  Three things are held IDENTICAL across arms within a seed:
    (1) the extraction pool -- the same 60 harmful / 60 benign prompts, and in fact the
        same cached [n, d] pooled-activation matrices H and B, feed every estimator;
    (2) the injection -- layer 12, budget_f 0.05, radius_frac 1.0 (pure relative_add),
        the champion coordinates, unchanged;
    (3) the NORM -- every direction is unit-normalised before injection, so the applied
        displacement is exactly f*||h|| for all arms.
  With pool, injection and norm fixed, the only free variable left is the ESTIMATOR, so
  the PPL contrast measures estimator quality and nothing else. A sign convention is
  also fixed for all arms (below), because a PC's sign is arbitrary and steering is not
  sign-invariant.

SIGN CONVENTION
  Every direction v is oriented so that mean(H @ v) > mean(B @ v) -- harmful prompts
  project higher than benign ones. This is automatic for diffmean and LDA; for the PCs
  it replaces an arbitrary SVD sign with a class-discriminative one. For the random
  control the same rule is applied, which is the conservative choice (it gives the floor
  whatever weak class signal a random draw happens to carry).

OBJECTIVE: WikiText-2 perplexity. JUDGE-FREE (autoresearch_results/JUDGE_CARD.md --
two judge calibrations failed at AUC 0.665 / 0.751 against a 0.85 gate).
========================================================================

Usage:
  PYTHONPATH=src python scripts/hillclimb_source.py --seeds 6 --layer 12
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
from steering.safety_target import _pooled_reps  # noqa: E402
from steering.stats import power_note  # noqa: E402

DEFAULT_MODEL = "DavidAU/gemma-3-1b-it-heretic-extreme-uncensored-abliterated"

# The champion coordinates (autoresearch_results/best_config.json). Held FIXED.
CHAMPION_LAYER = 12
CHAMPION_BUDGET_F = 0.05
CHAMPION_RADIUS_FRAC = 1.0

ARMS = ("diffmean", "pca_uncentered", "pca_centered", "lda_shrunk", "random")


def load_pools(n_harm: int, n_benign: int) -> tuple[list[str], list[str]]:
    """Real benchmark prompts: three harmful sets pooled + XSTest benign (M-a's loader)."""
    harm: list[str] = []
    for nm in ("jailbreakbench", "harmbench", "advbench"):
        try:
            harm += [it["prompt"] for it in load_safety_benchmark(nm, n=n_harm)]
        except Exception as exc:  # noqa: BLE001
            print(f"[data] {nm} unavailable: {type(exc).__name__}")
    benign = [it["prompt"] for it in load_safety_benchmark("xstest", n=n_benign)]
    return harm, benign


class AngleRadiusHook:
    """Displacement of magnitude f*||h|| split between angle (r=0) and radius (r=1).

    Copied verbatim from `scripts/run_ma_direction_seeds.py` so this experiment injects
    through the SAME operator the champion was measured with (file-scope rule forbids
    importing from that script). At r=1.0 -- the champion -- this reduces to
    h -> h + f*||h||*v_hat, i.e. `hooks.relative_add`.
    """

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


# --------------------------------------------------------------------------- estimators
def unit(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64).reshape(-1)
    return v / (np.linalg.norm(v) + 1e-12)


def orient_dir(v: np.ndarray, H: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Fix the sign so harmful projects higher than benign (see SIGN CONVENTION)."""
    sep = float((H @ v).mean() - (B @ v).mean())
    return -v if sep < 0 else v


def est_diffmean(H: np.ndarray, B: np.ndarray, rng) -> tuple[np.ndarray, dict]:
    """CHAMPION. First moment of the contrast: mean(harmful) - mean(benign), unit."""
    return unit(H.mean(0) - B.mean(0)), {}


def _paired_diffs(H: np.ndarray, B: np.ndarray, rng) -> np.ndarray:
    """Random one-to-one pairing of harmful with benign rows -> difference matrix D.

    Harmful/benign safety corpora are not paired one-to-one, so the pairing is random
    (seeded). It is drawn ONCE per seed and shared by both PCA arms, so the two PCA
    constructions differ only in the centering, which is the contrast under test.
    """
    k = min(H.shape[0], B.shape[0])
    hi = rng.permutation(H.shape[0])[:k]
    bi = rng.permutation(B.shape[0])[:k]
    return H[hi] - B[bi]


def est_pca_uncentered(H: np.ndarray, B: np.ndarray, rng) -> tuple[np.ndarray, dict]:
    """Top right singular vector of the RAW difference matrix D (no centering).

    PRECISELY: D[i] = H[h_i] - B[b_i] over a random one-to-one pairing; take the top
    right singular vector of D itself. Because D is not centered, this direction
    maximises the mean SQUARED projection, which contains the squared mean projection --
    so it is pulled toward the diffmean. Reported explained-variance ratio is relative
    to the total squared Frobenius norm of D (uncentered second moment), not variance.
    """
    D = _paired_diffs(H, B, rng)
    _, S, Vt = np.linalg.svd(D, full_matrices=False)
    return unit(Vt[0]), {"top_ratio_uncentered_2nd_moment": float(S[0] ** 2 / (S ** 2).sum())}


def est_pca_centered(H: np.ndarray, B: np.ndarray, rng) -> tuple[np.ndarray, dict]:
    """Top PC of the difference matrix AFTER subtracting its own mean (the LAT/RepE form).

    PRECISELY: D as above, then Dc = D - mean(D, axis=0), then the top right singular
    vector of Dc. Subtracting mean(D) removes exactly the diffmean-like first moment, so
    this reads the dominant axis of VARIATION within the contrast set. This is the
    construction most likely to be a nuisance axis (Marks & Tegmark 2310.06824), and it
    is reported separately from the uncentered form precisely because the two are
    different objects that the word "PCA" is used for interchangeably in the literature.
    """
    D = _paired_diffs(H, B, rng)
    Dc = D - D.mean(0, keepdims=True)
    _, S, Vt = np.linalg.svd(Dc, full_matrices=False)
    return unit(Vt[0]), {"explained_variance_ratio": float(S[0] ** 2 / (S ** 2).sum())}


def est_lda_shrunk(H: np.ndarray, B: np.ndarray, rng, gamma: float = 0.10
                   ) -> tuple[np.ndarray, dict]:
    """Diffmean WHITENED by the shrunk pooled within-class covariance (Fisher/LDA).

    v = Sg^{-1} (mean(H) - mean(B)), Sg = (1-g) S + g (tr(S)/d) I, S the pooled
    within-class covariance. Shrinkage is NOT optional here: d (1152 for Gemma-3-1B) far
    exceeds n (120), so S has rank <= n-2 and is singular. gamma is reported, as is the
    condition number of Sg, so the reader can see how much of the whitening is real and
    how much is the ridge.
    """
    d = H.shape[1]
    Hc, Bc = H - H.mean(0), B - B.mean(0)
    S = (Hc.T @ Hc + Bc.T @ Bc) / max(H.shape[0] + B.shape[0] - 2, 1)
    Sg = (1.0 - gamma) * S + gamma * (np.trace(S) / d) * np.eye(d)
    v = np.linalg.solve(Sg, H.mean(0) - B.mean(0))
    ev = np.linalg.eigvalsh(Sg)
    return unit(v), {"shrinkage_gamma": gamma,
                     "cond_number_shrunk": float(ev[-1] / max(ev[0], 1e-30)),
                     "rank_deficit_d_minus_n": int(d - (H.shape[0] + B.shape[0] - 2))}


def est_random(H: np.ndarray, B: np.ndarray, rng) -> tuple[np.ndarray, dict]:
    """M-a's control: a random direction, unit-normalised = norm-matched to every arm."""
    return unit(rng.standard_normal(H.shape[1])), {}


ESTIMATORS = {"diffmean": est_diffmean, "pca_uncentered": est_pca_uncentered,
              "pca_centered": est_pca_centered, "lda_shrunk": est_lda_shrunk,
              "random": est_random}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--layer", type=int, default=CHAMPION_LAYER)
    ap.add_argument("--seeds", type=int, default=6, help="EXTRACTION resamples (>=5)")
    ap.add_argument("--budget", type=float, default=CHAMPION_BUDGET_F)
    ap.add_argument("--radius-frac", type=float, default=CHAMPION_RADIUS_FRAC)
    ap.add_argument("--n-extract", type=int, default=60, help="per class, per seed")
    ap.add_argument("--n-ppl", type=int, default=25)
    ap.add_argument("--lda-gamma", type=float, default=0.10)
    ap.add_argument("--cache-dir", default=str(ROOT / "autoresearch_results"),
                    help="where the per-seed activation cache lives; the cache is a "
                         "recomputable intermediate, not evidence, so it may live "
                         "outside the repo")
    ap.add_argument("--out", default=str(ROOT / "autoresearch_results" / "hillclimb_source.json"))
    args = ap.parse_args()
    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    out = Path(args.out)
    part = out.with_suffix(".partial.json")

    harm, benign = load_pools(120, 120)
    print(f"[data] harmful={len(harm)} harmless={len(benign)} (real benchmarks)", flush=True)
    if len(harm) < 60 or len(benign) < 20:
        print("[data] pools too small; aborting rather than reporting a weak direction")
        return

    model, tok = load_model(args.model, quant="4bit")
    layer_mod = get_residual_layers(model)[args.layer]

    # resume: a reap must never cost measured cells (host reaps long jobs -- CLAUDE 18.5)
    state = json.loads(part.read_text(encoding="utf-8")) if part.exists() else {}
    rows: list[dict] = state.get("rows", [])
    cosines: list[dict] = state.get("cosines", [])
    diags: list[dict] = state.get("diagnostics", [])
    base = state.get("unsteered_ppl")
    if base is None:
        base = float(wikitext_perplexity(model, tok, n=args.n_ppl))
    print(f"[base] unsteered PPL = {base:.4f}", flush=True)
    if rows:
        print(f"[resume] {len(rows)} cells already measured; skipping those", flush=True)
    done = {(r["seed"], r["arm"]) for r in rows}

    def checkpoint() -> None:
        part.write_text(json.dumps({"unsteered_ppl": base, "rows": rows,
                                    "cosines": cosines, "diagnostics": diags}, indent=2),
                        encoding="utf-8")

    pool_rng = np.random.default_rng(0)
    t0 = time.time()
    dir_store: dict[int, dict[str, np.ndarray]] = {}
    for s in range(args.seeds):
        # ---- ACTIVATION CACHE: every estimator in this seed sees the SAME H and B.
        # Extraction is 2*n_extract forward passes and happens before the first cell can
        # checkpoint, so a reap inside that window used to cost the whole seed.
        acache = cache_dir / f".srccache_s{s}_n{args.n_extract}_L{args.layer}.npz"
        if acache.exists():
            z = np.load(acache)
            H, B = z["H"].astype(np.float64), z["B"].astype(np.float64)
            print(f"  [seed {s}] activations from cache H{H.shape} B{B.shape}", flush=True)
        else:
            hi = pool_rng.choice(len(harm), size=min(len(harm), args.n_extract), replace=False)
            bi = pool_rng.choice(len(benign), size=min(len(benign), args.n_extract), replace=False)
            H = _pooled_reps(model, tok, [harm[i] for i in hi], args.layer).astype(np.float64)
            B = _pooled_reps(model, tok, [benign[i] for i in bi], args.layer).astype(np.float64)
            np.savez(acache, H=H.astype(np.float32), B=B.astype(np.float32))
            print(f"  [seed {s}] extracted H{H.shape} B{B.shape}", flush=True)

        # ---- estimate all arms from the identical (H, B); unit-norm + sign-orient
        est_rng = np.random.default_rng(1000 + s)
        dirs: dict[str, np.ndarray] = {}
        seed_diag = {"seed": s}
        for arm in ARMS:
            fn = ESTIMATORS[arm]
            v, info = (fn(H, B, est_rng, args.lda_gamma) if arm == "lda_shrunk"
                       else fn(H, B, est_rng))
            v = orient_dir(v, H, B)
            dirs[arm] = v
            seed_diag[arm] = {**info,
                              "class_separation": float((H @ v).mean() - (B @ v).mean()),
                              "norm_after_unit": float(np.linalg.norm(v))}
        dir_store[s] = dirs
        cos = {f"{a}|{b}": float(dirs[a] @ dirs[b])
               for i, a in enumerate(ARMS) for b in ARMS[i + 1:]}
        if not any(c.get("seed") == s for c in cosines):
            cosines.append({"seed": s, **cos})
        if not any(dg.get("seed") == s for dg in diags):
            diags.append(seed_diag)
        print("    cos: " + "  ".join(f"{k}={v:+.3f}" for k, v in cos.items()
                                      if k.startswith("diffmean")), flush=True)
        checkpoint()

        # ---- measure: identical layer / budget / radius_frac for every arm
        for arm in ARMS:
            if (s, arm) in done:
                continue
            v = torch.tensor(dirs[arm], dtype=torch.float32)
            ppl = ppl_with(model, tok, layer_mod, v, args.budget, args.radius_frac, args.n_ppl)
            rows.append({"seed": s, "arm": arm, "ppl": ppl, "ratio_vs_base": ppl / base})
            print(f"    {arm:16s} PPL={ppl:9.4f}  ({ppl / base:.4f}x base)", flush=True)
            checkpoint()   # CHECKPOINT AFTER EVERY CELL

    # ------------------------------------------------------------------ analysis
    rng = np.random.default_rng(7)
    by_arm: dict[str, dict] = {}
    ref = {r["seed"]: r["ppl"] for r in rows if r["arm"] == "diffmean"}
    for arm in ARMS:
        vals = {r["seed"]: r["ppl"] for r in rows if r["arm"] == arm}
        seeds = sorted(set(vals) & set(ref))
        if not seeds:
            continue
        p = np.array([vals[s] for s in seeds])
        d = np.array([vals[s] - ref[s] for s in seeds])          # PAIRED, per seed
        boot = np.array([np.mean(rng.choice(d, len(d), replace=True)) for _ in range(10_000)])
        ci = [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))]
        by_arm[arm] = {
            "n_seeds": len(seeds), "ppl_mean": float(p.mean()), "ppl_sd": float(p.std(ddof=1)),
            "ratio_vs_base_mean": float(p.mean() / base),
            "paired_delta_vs_diffmean_mean": float(d.mean()),
            "paired_delta_ci95": ci,
            "ci_excludes_zero": bool(ci[0] > 0 or ci[1] < 0),
            "beats_diffmean_all_seeds": bool((d < 0).all()) if arm != "diffmean" else None,
        }

    cos_mean = {k: float(np.mean([c[k] for c in cosines]))
                for k in cosines[0] if k != "seed"} if cosines else {}
    # cross-seed stability of each estimator (how reproducible is the direction itself)
    stability = {}
    for arm in ARMS:
        vs = [dir_store[s][arm] for s in sorted(dir_store)]
        if len(vs) > 1:
            pw = [float(vs[i] @ vs[j]) for i in range(len(vs)) for j in range(i + 1, len(vs))]
            stability[arm] = {"mean_cross_seed_cos": float(np.mean(pw)),
                              "min": float(np.min(pw)), "max": float(np.max(pw))}

    winner = min((a for a in by_arm), key=lambda a: by_arm[a]["ppl_mean"]) if by_arm else None
    challenger_significant = bool(
        winner and winner != "diffmean" and by_arm[winner]["ci_excludes_zero"]
        and by_arm[winner]["paired_delta_vs_diffmean_mean"] < 0)
    n_eff = min((by_arm[a]["n_seeds"] for a in by_arm), default=0)
    pw_note = power_note(n_eff, effect=1.0, sd=1.0, family_size=len(ARMS) - 1)
    tier = ("EVALUATION-eligible" if n_eff >= 7 and pw_note.get("can_reach_holm_alpha")
            else "SCREENING")

    preds = {
        "P1_cos_diffmean_pca_uncentered_gt_0.90":
            bool(cos_mean.get("diffmean|pca_uncentered", 0) > 0.90),
        "P2_cos_diffmean_pca_centered_lt_0.50":
            bool(abs(cos_mean.get("diffmean|pca_centered", 1)) < 0.50),
        "P3_no_estimator_beats_diffmean_with_CI_excluding_zero":
            not challenger_significant,
        "P4_random_floor_worse_than_diffmean":
            bool(by_arm.get("random", {}).get("paired_delta_vs_diffmean_mean", 0) > 0),
    }

    res = {
        "hypothesis": "HC-S -- does the SOURCE (direction estimator) axis move the champion?",
        "citation": "arXiv:2310.01405 (RepE/LAT PCA); arXiv:2312.06681 (CAA diffmean); "
                    "arXiv:2310.06824 (mass-mean vs PCA); control arXiv:2606.20852",
        "objective": "WikiText-2 perplexity (JUDGE-FREE; see JUDGE_CARD.md)",
        "model": args.model,
        "held_fixed": {"layer": args.layer, "budget_f": args.budget,
                       "radius_frac": args.radius_frac,
                       "extraction_pool": "identical H,B per seed for every arm",
                       "normalisation": "every direction unit-normalised before injection",
                       "sign_convention": "oriented so mean(H@v) > mean(B@v)"},
        "why_comparable": ("Pool, injection site/budget/operation, and vector norm are "
                           "identical across arms within a seed; the ESTIMATOR is the "
                           "only free variable, so the PPL contrast is estimator quality "
                           "and nothing else."),
        "n_harmful_pool": len(harm), "n_harmless_pool": len(benign),
        "n_extract_per_class": args.n_extract, "n_ppl_docs": args.n_ppl,
        "seeds": args.seeds, "unsteered_ppl": base,
        "rows": rows, "by_arm": by_arm,
        "pairwise_cosine_mean": cos_mean, "pairwise_cosine_per_seed": cosines,
        "cross_seed_direction_stability": stability,
        "estimator_diagnostics": diags,
        "winner_by_mean_ppl": winner,
        "new_champion": challenger_significant,
        "predictions_evaluated": preds,
        "power": pw_note, "tier": tier,
        "tier_note": ("n<=3 is SCREENING and may never be called a win (CLAUDE.md S7). "
                      "A new champion requires a paired bootstrap CI excluding zero."),
        "elapsed_s": round(time.time() - t0, 1),
    }
    out.write_text(json.dumps(res, indent=2), encoding="utf-8")
    print("\n[HC-S] arm summary (PPL mean | ratio | paired delta vs diffmean [CI95])")
    for a in ARMS:
        if a in by_arm:
            b = by_arm[a]
            print(f"  {a:16s} {b['ppl_mean']:9.4f} | {b['ratio_vs_base_mean']:.4f}x | "
                  f"{b['paired_delta_vs_diffmean_mean']:+8.4f} "
                  f"[{b['paired_delta_ci95'][0]:+.4f}, {b['paired_delta_ci95'][1]:+.4f}]")
    print(f"[HC-S] mean cosines: {json.dumps({k: round(v, 4) for k, v in cos_mean.items()})}")
    print(f"[HC-S] predictions {json.dumps(preds)}")
    print(f"[HC-S] winner={winner}  new_champion={challenger_significant}  tier={tier}")
    print(f"[write] {out}")


if __name__ == "__main__":
    main()
