"""rung3_n17.py — the rung-3 escalation of N17/N5 (the strongest screening result).

Upgrades the screening N17/N5 finding to a rigorous test:
  - REAL WikiText-2 perplexity (not the synthetic mini-corpus);
  - many pooled (model x layer x alpha) points across TWO models;
  - Spearman correlation with a 95% bootstrap CI (>=10k resamples);
  - a HELD-OUT generalization test: fit the N5 law `log PPL = a + b*offshell` on
    gemma-3-270m points, predict gemma-3-1b points, report held-out R^2.

PPL is teacher-forced (no generation) so this is fast. offshell is one forward.
Writes ideas/_campaigns/RUNG3_N17.json + a verdict.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import steering.datasets as ds  # noqa: E402
from steering.extract import extract_bank  # noqa: E402
from steering.geometry import offshell_displacement  # noqa: E402
from steering.hooks import SteeringContext, probe_activations  # noqa: E402
from steering.model import encode_to_device, load_model_cached  # noqa: E402

MODELS = [
    ("models/google/gemma-3-270m-it", [6, 9, 12, 14, 16]),
    ("models/google/gemma-3-1b-it", [10, 14, 18, 20, 22]),
]
ALPHAS = [0.0, 0.02, 0.05, 0.1, 0.2, 0.3]  # relative (fraction of ||h||)
PASSAGES = None  # filled at runtime


def real_ppl(model, tok, passages, layer=None, vector=None, alpha=0.0):
    """Teacher-forced perplexity over real passages, optionally steered (relative_add)."""
    from steering.eval import perplexity
    if layer is None or vector is None or alpha == 0.0:
        return perplexity(model, tok, passages)
    with SteeringContext(model, vector, [layer], operation="relative_add", alpha=alpha):
        return perplexity(model, tok, passages)


def main() -> None:
    passages = ds.load_wikitext2_real()
    pairs = ds.load_axbench_mini()
    pts = []  # (model, layer, alpha, offshell, ppl)
    for name, layers in MODELS:
        print(f"\n=== {name} ===", flush=True)
        model, tok = load_model_cached(name, quant="none")
        bank = extract_bank(model, tok, pairs, cache_dir=str(ROOT / "autoresearch_results" / "act_cache"),
                            model_tag=name, quant="none")
        sample_ids = encode_to_device(tok, pairs[0][0], model)
        base_ppl = real_ppl(model, tok, passages)
        print(f"  base real-PPL = {base_ppl:.2f}")
        for layer in layers:
            v = torch.tensor(np.asarray(bank[layer]["diffmean"]), dtype=torch.float32)
            v = v / (v.norm() + 1e-8)  # unit; relative_add scales by ||h||
            h_base = probe_activations(model, sample_ids, [layer])[layer]
            for a in ALPHAS:
                if a == 0.0:
                    pts.append((name, layer, a, 0.0, base_ppl)); continue
                with SteeringContext(model, v, [layer], operation="relative_add", alpha=a):
                    h_st = probe_activations(model, sample_ids, [layer])[layer]
                off = offshell_displacement(h_base, h_st)
                ppl = real_ppl(model, tok, passages, layer, v, a)
                pts.append((name, layer, a, off, ppl))
                print(f"  L{layer:>2} a={a:<4} off={off:.3f} realPPL={ppl:.1f}", flush=True)

    # --- stats ---
    from scipy.stats import spearmanr
    steered = [p for p in pts if p[3] > 0 and p[4] > 0 and np.isfinite(p[4])]
    off = np.array([p[3] for p in steered]); lp = np.log(np.array([p[4] for p in steered]))
    rho, pval = spearmanr(off, lp)
    # bootstrap CI on Spearman
    rng = np.random.default_rng(0); boots = []
    for _ in range(10000):
        idx = rng.integers(0, len(off), len(off))
        boots.append(spearmanr(off[idx], lp[idx])[0])
    ci = (float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5)))
    # held-out: fit on 270m, predict 1b
    m270 = [(p[3], np.log(p[4])) for p in steered if "270m" in p[0]]
    m1b = [(p[3], np.log(p[4])) for p in steered if "1b" in p[0]]
    b, a0 = np.polyfit([x for x, _ in m270], [y for _, y in m270], 1)
    yp = np.array([a0 + b * x for x, _ in m1b]); yt = np.array([y for _, y in m1b])
    ss = 1 - np.sum((yt - yp) ** 2) / np.sum((yt - yt.mean()) ** 2)

    out = {
        "n_points": len(steered), "n_models": 2, "real_ppl": True,
        "spearman_offshell_logPPL": float(rho), "spearman_p": float(pval),
        "spearman_95ci": ci,
        "n5_law_fit_270m": {"slope": float(b), "intercept": float(a0)},
        "heldout_R2_on_1b": float(ss),
        "points": [{"model": p[0].split("/")[-1], "layer": p[1], "alpha": p[2],
                    "offshell": round(p[3], 4), "real_ppl": round(p[4], 2)} for p in pts],
    }
    (ROOT / "ideas" / "_campaigns" / "RUNG3_N17.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print("\n=== RUNG-3 N17/N5 (real WikiText) ===")
    print(f"n={len(steered)} pooled points, 2 models, REAL WikiText-2 PPL")
    print(f"Spearman(offshell, log realPPL) = {rho:+.3f}  95% CI [{ci[0]:+.3f}, {ci[1]:+.3f}]  p={pval:.1e}")
    print(f"N5 law fit on 270m: log PPL = {a0:.2f} + {b:.2f}*offshell")
    print(f"HELD-OUT R^2 (fit 270m -> predict 1b) = {ss:.3f}")
    print("Verdict: N17/N5 clears rung-3 IF CI excludes 0 AND held-out R^2 > 0.5")


if __name__ == "__main__":
    main()
