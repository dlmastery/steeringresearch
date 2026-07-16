"""run_nonident.py — the payoff: low cosine, similar effect => non-identifiable.

Pipeline
--------
1. BUILD.  Load the abliterated Gemma. Read activations from N_EXTRACT harmful +
   benign prompts and build the K candidate directions (``vectors.py``). Save
   them and their pairwise cosine matrix.

2. STEER + JUDGE.  On a DISJOINT held-out set of harmful prompts, steer with each
   candidate at the SAME matched relative alpha and have the judge label each
   output REFUSAL / COMPLIANCE / GIBBERISH. Because every candidate is
   unit-normalized and steering is norm-relative, all candidates get an
   equal-magnitude nudge — so any difference in refusal rate is due to
   DIRECTION, not strength. Also record the unsteered baseline.

3. REPORT.  Cross-tabulate (pairwise cosine) against (per-direction refusal
   rate). The headline statistic: the MINIMUM pairwise cosine among the
   directions that are "effective" (reach >= EFFECTIVE_FRACTION of the best
   refusal rate). If two directions with cosine well below 1.0 both steer, the
   refusal direction is non-identifiable. Save results.json + a two-panel PNG.

Everything model-touching lives under ``main()`` so ``import run_nonident`` is a
no-op (safe for tests / a webapp).

RESULTS SCHEMA (kept in sync with README)
-----------------------------------------
{
  "model_id": str, "layer": int, "matched_alpha": float,
  "n_extract": int, "n_eval": int,
  "names": [str, ...],
  "cosine_matrix": [[float, ...], ...],           # K x K, order = names
  "directions": [{"name": str, "recipe": str, "pooling": str}, ...],
  "baseline": {"refusal_rate": float, "compliance_rate": float,
               "gibberish_rate": float, "n": int},
  "per_direction": [
      {"name": str, "refusal_rate": float, "compliance_rate": float,
       "gibberish_rate": float, "n": int}, ...
  ],
  "nonident": {"best_refusal": float, "effective_threshold": float,
               "effective_names": [str, ...], "min_cosine_effective": float,
               "refusal_spread": float, "verdict": str},
  "examples": [{"prompt": str, "direction": str,
                "baseline_response": str, "steered_response": str,
                "baseline_verdict": str, "steered_verdict": str}, ...],
  "plots": {"nonident": "nonident.png"}
}
"""
from __future__ import annotations

import json
import sys

import numpy as np

from . import config as C


# --------------------------------------------------------------------------- #
# Pure helpers (no model) — unit-tested in __main__.
# --------------------------------------------------------------------------- #
def _rates(verdicts: list[str]) -> dict[str, float]:
    """Fraction of REFUSAL / COMPLIANCE / GIBBERISH among a verdict list."""
    n = max(1, len(verdicts))
    return {
        "refusal_rate": verdicts.count("REFUSAL") / n,
        "compliance_rate": verdicts.count("COMPLIANCE") / n,
        "gibberish_rate": verdicts.count("GIBBERISH") / n,
    }


def summarize_nonidentifiability(
    per_direction: list[dict],
    names: list[str],
    cosine: np.ndarray,
    effective_fraction: float,
    control_name: str = "random_in_pcspan",
) -> dict:
    """Compute the headline non-identifiability statistics.

    The claim is supported when several directions of LOW mutual cosine all reach
    a SIMILAR (high) refusal rate. We quantify that as:

      * ``best_refusal``          — the highest per-direction refusal rate.
      * ``effective_threshold``   — ``effective_fraction`` of ``best_refusal``.
      * ``effective_names``       — directions at/above that threshold, EXCLUDING
                                    the random control (whose job is to *fail*).
      * ``min_cosine_effective``  — the minimum pairwise cosine among those
                                    effective directions. LOW here + several
                                    effective directions == non-identifiable.
      * ``refusal_spread``        — max-min refusal rate among effective
                                    directions (small == "similar effect").

    ``verdict`` is a plain-language read, not a statistical claim (see caveats).
    """
    idx = {n: i for i, n in enumerate(names)}
    rate = {d["name"]: d["refusal_rate"] for d in per_direction}

    contrast = [d for d in per_direction if d["name"] != control_name]
    best_refusal = max((d["refusal_rate"] for d in contrast), default=0.0)
    threshold = effective_fraction * best_refusal

    effective = [d["name"] for d in contrast if d["refusal_rate"] >= threshold
                 and best_refusal > 0.0]

    # Minimum pairwise cosine among effective directions (off-diagonal only).
    min_cos = 1.0
    for a in range(len(effective)):
        for b in range(a + 1, len(effective)):
            c = float(cosine[idx[effective[a]], idx[effective[b]]])
            min_cos = min(min_cos, c)
    if len(effective) < 2:
        min_cos = float("nan")   # need >= 2 directions to talk about "same effect"

    eff_rates = [rate[n] for n in effective]
    spread = (max(eff_rates) - min(eff_rates)) if eff_rates else 0.0

    if len(effective) >= 2 and min_cos < 0.9:
        verdict = (f"SUPPORTED (screening): {len(effective)} directions with "
                   f"pairwise cosine down to {min_cos:.2f} reach within "
                   f"{effective_fraction:.0%} of the best refusal rate")
    elif len(effective) >= 2:
        verdict = ("MIXED (screening): several directions steer, but they are "
                   f"near-collinear (min cosine {min_cos:.2f}) — weak evidence "
                   "of a genuine family")
    else:
        verdict = ("NOT SUPPORTED (screening): fewer than two effective "
                   "directions — the effect did not reproduce across recipes")

    return {
        "best_refusal": float(best_refusal),
        "effective_threshold": float(threshold),
        "effective_names": effective,
        "min_cosine_effective": (None if np.isnan(min_cos) else float(min_cos)),
        "refusal_spread": float(spread),
        "control_refusal": float(rate.get(control_name, float("nan"))),
        "verdict": verdict,
    }


def _summary_table(results: dict) -> str:
    """Plain-text recap printed at the end of a run."""
    lines = ["", "=" * 68, "NON-IDENTIFIABILITY SUMMARY", "=" * 68,
             f"model : {results['model_id']}",
             f"layer : {results['layer']}   matched alpha: {results['matched_alpha']:.3f}",
             f"baseline refusal (no steering): "
             f"{results['baseline']['refusal_rate']:.2f}",
             "",
             f"  {'direction':<20} {'refusal':>8} {'comply':>8} {'gibber':>8}"]
    for d in results["per_direction"]:
        lines.append(f"  {d['name']:<20} {d['refusal_rate']:>8.2f} "
                     f"{d['compliance_rate']:>8.2f} {d['gibberish_rate']:>8.2f}")
    ni = results["nonident"]
    mc = ni["min_cosine_effective"]
    lines += ["",
              f"effective directions (>= {C.EFFECTIVE_FRACTION:.0%} of best "
              f"refusal {ni['best_refusal']:.2f}): {ni['effective_names']}",
              f"min pairwise cosine among them: "
              f"{'n/a' if mc is None else f'{mc:.3f}'}",
              f"refusal spread among them     : {ni['refusal_spread']:.2f}",
              f"random-control refusal        : {ni['control_refusal']:.2f}",
              "", f"verdict: {ni['verdict']}", "=" * 68, ""]
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Plotting — matplotlib Agg (headless). One figure, two panels, saved as PNG.
# --------------------------------------------------------------------------- #
def _plot(results: dict, path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = results["names"]
    cos = np.array(results["cosine_matrix"], dtype=float)
    rates = {d["name"]: d["refusal_rate"] for d in results["per_direction"]}
    short = [n.replace("diffmean_", "dm_").replace("random_in_pcspan", "random")
             for n in names]

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(12, 5))

    # Panel 1: cosine heatmap (how DIFFERENT the directions are).
    im = ax0.imshow(cos, vmin=-1, vmax=1, cmap="RdBu_r")
    ax0.set_xticks(range(len(short)), short, rotation=45, ha="right", fontsize=8)
    ax0.set_yticks(range(len(short)), short, fontsize=8)
    for i in range(len(short)):
        for j in range(len(short)):
            ax0.text(j, i, f"{cos[i, j]:.2f}", ha="center", va="center",
                     fontsize=7, color="black")
    ax0.set_title("Pairwise cosine similarity of the candidate directions\n"
                  "(low off-diagonal = they are DIFFERENT vectors)")
    fig.colorbar(im, ax=ax0, fraction=0.046, pad=0.04)

    # Panel 2: refusal rate per direction (how SIMILAR the effect is).
    order = names
    vals = [rates[n] for n in order]
    colors = ["#888" if n == "random_in_pcspan" else "#2a7" for n in order]
    bars = ax1.bar(range(len(order)),
                   vals, color=colors)
    for b, v in zip(bars, vals):
        ax1.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.2f}",
                 ha="center", va="bottom", fontsize=8)
    ax1.axhline(results["baseline"]["refusal_rate"], color="#c33", ls="--",
                lw=1, label=f"baseline (no steer) = "
                            f"{results['baseline']['refusal_rate']:.2f}")
    ax1.set_xticks(range(len(order)), short, rotation=45, ha="right", fontsize=8)
    ax1.set_ylim(0, 1.05)
    ax1.set_ylabel("refusal rate on held-out harmful prompts")
    ax1.set_title(f"Steering effect at matched alpha={results['matched_alpha']:.2f}\n"
                  "(similar heights = SAME effect despite different vectors)")
    ax1.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# The pipeline — everything below loads / runs the model.
# --------------------------------------------------------------------------- #
def main() -> dict:
    import random

    import torch

    # Peer / lesson-2 modules, imported inside main() so a bare import stays
    # model-free and torch-free.
    from steering_tutorials.hello_world_steering.model_utils import (
        load_model, generate, num_layers,
    )
    from steering_tutorials.hello_world_steering.judge import Judge
    from steering_tutorials.common.data import load_harmful_benign
    from .vectors import build_candidate_directions, save_directions

    # Reproducibility.
    random.seed(C.SEED)
    np.random.seed(C.SEED)
    torch.manual_seed(C.SEED)

    # --- Load the model + data ----------------------------------------------
    model, tok = load_model(C.MODEL_ID)
    layer = min(C.LAYER, num_layers(model) - 1)

    data = load_harmful_benign(n_per_class=C.N_PER_CLASS, seed=C.SEED)
    harmful, benign = data["harmful"], data["benign"]

    # Three disjoint roles: build / eval / (unused headroom).
    build_harmful = harmful[:C.N_EXTRACT]
    build_benign = benign[:C.N_EXTRACT]
    eval_harmful = harmful[C.N_EXTRACT:C.N_EXTRACT + C.N_EVAL]
    print(f"[split] build: {len(build_harmful)}h/{len(build_benign)}b   "
          f"eval: {len(eval_harmful)}h", file=sys.stderr)

    # --- 1. Build the K candidate directions --------------------------------
    built = build_candidate_directions(
        model, tok, build_harmful, build_benign, layer,
        n_pc=C.N_PC, seed=C.SEED,
    )
    save_directions(C.DIRECTIONS_PATH, built)
    names = built["names"]
    print(f"[save] {C.DIRECTIONS_PATH}", file=sys.stderr)

    judge = Judge(model, tok)

    # --- 2a. Baseline: no steering on the held-out harmful prompts ----------
    base_verdicts: list[str] = []
    for p in eval_harmful:
        resp = generate(model, tok, p, max_new_tokens=C.MAX_NEW_TOKENS,
                        vector=None, layer=layer, alpha=0.0)
        base_verdicts.append(judge.verdict(p, resp))
    baseline = {"n": len(eval_harmful), **_rates(base_verdicts)}
    print(f"[baseline] refusal={baseline['refusal_rate']:.2f}", file=sys.stderr)

    # --- 2b. Steer with each candidate at the SAME matched alpha ------------
    per_direction: list[dict] = []
    examples: list[dict] = []
    for name in names:
        v_unit = built["candidates"][name]["v_unit"]
        verdicts: list[str] = []
        for i, p in enumerate(eval_harmful):
            steered = generate(model, tok, p, max_new_tokens=C.MAX_NEW_TOKENS,
                               vector=v_unit, layer=layer,
                               alpha=C.MATCHED_ALPHA, operation="relative_add")
            v = judge.verdict(p, steered)
            verdicts.append(v)
            # Keep a couple of side-by-side examples per direction.
            if i < 2:
                base_resp = generate(model, tok, p,
                                     max_new_tokens=C.MAX_NEW_TOKENS,
                                     vector=None, layer=layer, alpha=0.0)
                examples.append({
                    "prompt": p, "direction": name,
                    "baseline_response": base_resp, "steered_response": steered,
                    "baseline_verdict": judge.verdict(p, base_resp),
                    "steered_verdict": v,
                })
        rec = {"name": name, "n": len(eval_harmful), **_rates(verdicts)}
        per_direction.append(rec)
        print(f"[steer {name:<20}] refusal={rec['refusal_rate']:.2f} "
              f"gibber={rec['gibberish_rate']:.2f}", file=sys.stderr)

    # --- 3. Non-identifiability statistics + report -------------------------
    cosine = built["cosine"]
    nonident = summarize_nonidentifiability(
        per_direction, names, cosine, C.EFFECTIVE_FRACTION)

    results = {
        "model_id": C.MODEL_ID,
        "layer": int(layer),
        "matched_alpha": float(C.MATCHED_ALPHA),
        "n_extract": int(built["n_extract"]),
        "n_eval": int(len(eval_harmful)),
        "names": names,
        "cosine_matrix": cosine.tolist(),
        "directions": [{"name": n,
                        "recipe": built["candidates"][n]["recipe"],
                        "pooling": built["candidates"][n]["pooling"]}
                       for n in names],
        "baseline": baseline,
        "per_direction": per_direction,
        "nonident": nonident,
        "examples": examples[:12],
        "plots": {"nonident": C.PLOT_PATH.name},
    }

    C.RESULTS_PATH.write_text(json.dumps(results, indent=2), encoding="utf-8")
    _plot(results, C.PLOT_PATH)
    print(f"[save] {C.RESULTS_PATH}", file=sys.stderr)
    print(f"[save] {C.PLOT_PATH}", file=sys.stderr)
    print(_summary_table(results))
    return results


# --------------------------------------------------------------------------- #
# CPU self-test — NO model. Exercises the pure reporting helpers on fake data.
# Run: python -m steering_tutorials.non_identifiability.run_nonident
# --------------------------------------------------------------------------- #
def _self_test() -> None:
    # Fake a family: three low-cosine directions all "effective", one control low.
    names = ["dA", "dB", "dC", "random_in_pcspan"]
    # cosine matrix: dA,dB,dC mutually ~0.5, control near-orthogonal to all.
    cos = np.array([
        [1.0, 0.55, 0.45, 0.02],
        [0.55, 1.0, 0.50, 0.05],
        [0.45, 0.50, 1.0, 0.01],
        [0.02, 0.05, 0.01, 1.0],
    ])
    per_direction = [
        {"name": "dA", "refusal_rate": 0.80, "compliance_rate": 0.15, "gibberish_rate": 0.05, "n": 40},
        {"name": "dB", "refusal_rate": 0.78, "compliance_rate": 0.17, "gibberish_rate": 0.05, "n": 40},
        {"name": "dC", "refusal_rate": 0.72, "compliance_rate": 0.20, "gibberish_rate": 0.08, "n": 40},
        {"name": "random_in_pcspan", "refusal_rate": 0.10, "compliance_rate": 0.80, "gibberish_rate": 0.10, "n": 40},
    ]
    ni = summarize_nonidentifiability(per_direction, names, cos, 0.80)
    # dA/dB/dC are all within 80% of best (0.80); control excluded.
    assert set(ni["effective_names"]) == {"dA", "dB", "dC"}, ni["effective_names"]
    # min pairwise cosine among them is 0.45 (dA-dC).
    assert abs(ni["min_cosine_effective"] - 0.45) < 1e-6, ni["min_cosine_effective"]
    assert ni["verdict"].startswith("SUPPORTED"), ni["verdict"]
    assert ni["refusal_spread"] > 0

    # _rates sanity.
    r = _rates(["REFUSAL", "REFUSAL", "COMPLIANCE", "GIBBERISH"])
    assert abs(r["refusal_rate"] - 0.5) < 1e-9

    # Degenerate case: only one effective direction => NOT SUPPORTED, cosine n/a.
    ni2 = summarize_nonidentifiability(
        [{"name": "dA", "refusal_rate": 0.8, "compliance_rate": 0.2, "gibberish_rate": 0.0, "n": 10},
         {"name": "dB", "refusal_rate": 0.1, "compliance_rate": 0.9, "gibberish_rate": 0.0, "n": 10}],
        ["dA", "dB"], np.eye(2), 0.80)
    assert ni2["min_cosine_effective"] is None
    assert ni2["verdict"].startswith("NOT SUPPORTED"), ni2["verdict"]

    print("[self-test] OK - rates + non-identifiability summary behave.")


if __name__ == "__main__":
    _self_test()
