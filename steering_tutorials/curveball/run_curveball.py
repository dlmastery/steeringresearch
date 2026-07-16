"""run_curveball.py — the orchestrator: STRAIGHT-chord vs CURVED-arc steering.

The experiment in one screen. Build lesson 2's refusal direction on an extract
split, then generate on a MIXED held-out eval (harmful + benign) under three arms
and judge every reply:

    unsteered : alpha = 0                        — the abliterated baseline.
    straight  : chord  h + ALPHA*||h||*v_unit    — lesson-2 relative-add.
    curved    : arc    rotate h toward v by ALPHA radians (this lesson).

Both steered arms spend the SAME budget ALPHA; the only difference is the path
geometry (chord vs arc). Three numbers decide the comparison:

    - harmful REFUSAL rate    (efficacy — steering should install refusal)  UP good
    - harmful GIBBERISH rate  (coherence cost — the arm's collateral)       DOWN good
    - benign  OVER-REFUSAL    (did we wreck harmless answers too?)          DOWN good

The claim to test: the curved arc reaches a harmful refusal rate close to the
straight chord while producing LESS gibberish, because it never leaves the local
norm shell (off-shell displacement ~0). We also log that off-shell displacement
directly, as a model-light geometry probe on the eval activations, so the
mechanism is visible next to the behaviour. HONEST by construction: if curved does
not cut gibberish, or gives up too much refusal, the table says so.

Everything that loads or runs the model lives under ``main()`` so ``import
run_curveball`` is a no-op (safe for import-checks / tests). The team lead runs
this on the GPU.

RESULTS SCHEMA (kept in sync with README + plots)
-------------------------------------------------
{
  "model_id": str, "layer": int, "alpha": float, "n_curve_steps": int,
  "judge": str, "n_extract_per_class": int, "n_eval_per_class": int,
  "direction": {"layer": int, "n": int, "norm": float},
  "geometry": {"offshell_straight": float, "offshell_curved": float,
               "rotation_curved_rad": float},
  "arms": {
    "<unsteered|straight|curved>": {
       "harmful_refusal_rate": float, "harmful_gibberish_rate": float,
       "harmful_compliance_rate": float,
       "benign_over_refusal_rate": float, "benign_gibberish_rate": float,
       "n": int },
    ...
  },
  "examples": [ {"prompt": str, "class": "harmful|benign",
                 "straight": str, "straight_verdict": str,
                 "curved": str, "curved_verdict": str}, ... ],
  "plots": {"comparison": "straight_vs_curved.png",
            "geometry": "offshell_displacement.png"}
}
"""
from __future__ import annotations

import json
import sys

import numpy as np

from . import config as C
from .curveball import (
    curveball_endpoint,
    curveball_generate,
    relative_offshell,
    straight_endpoint,
)


# --------------------------------------------------------------------------- #
# Pure helpers (no model) — safe to unit-test in isolation.
# --------------------------------------------------------------------------- #
def _rate(verdicts: list[str], key: str) -> float:
    """Fraction of ``verdicts`` equal to ``key`` (e.g. "REFUSAL")."""
    n = max(1, len(verdicts))
    return verdicts.count(key) / n


def _arm_stats(harm_verdicts: list[str], ben_verdicts: list[str]) -> dict:
    """Assemble one arm's row from its per-prompt verdicts."""
    return {
        "harmful_refusal_rate": _rate(harm_verdicts, "REFUSAL"),
        "harmful_gibberish_rate": _rate(harm_verdicts, "GIBBERISH"),
        "harmful_compliance_rate": _rate(harm_verdicts, "COMPLIANCE"),
        "benign_over_refusal_rate": _rate(ben_verdicts, "REFUSAL"),
        "benign_gibberish_rate": _rate(ben_verdicts, "GIBBERISH"),
        "n": len(harm_verdicts),
    }


def _summary_table(results: dict) -> str:
    g = results["geometry"]
    lines = ["", "=" * 72,
             "STRAIGHT chord vs CURVED arc  (harm refusal UP  gibberish DOWN "
             "at matched budget)",
             "=" * 72,
             f"model: {results['model_id']}   layer: {results['layer']}   "
             f"alpha: {results['alpha']}   curve_steps: {results['n_curve_steps']}",
             f"judge: {results['judge']}",
             f"geometry: off-shell displacement  straight={g['offshell_straight']:.3f}"
             f"  curved={g['offshell_curved']:.3f}  "
             f"(curved rotates {g['rotation_curved_rad']:.3f} rad)",
             "",
             f"  {'arm':>10} {'harm_ref':>9} {'harm_gib':>9} {'harm_cmp':>9} "
             f"{'ben_oref':>9} {'ben_gib':>8}"]
    for name in ("unsteered", "straight", "curved"):
        a = results["arms"][name]
        lines.append(
            f"  {name:>10} {a['harmful_refusal_rate']:>9.2f} "
            f"{a['harmful_gibberish_rate']:>9.2f} "
            f"{a['harmful_compliance_rate']:>9.2f} "
            f"{a['benign_over_refusal_rate']:>9.2f} "
            f"{a['benign_gibberish_rate']:>8.2f}")
    lines += ["=" * 72, ""]
    return "\n".join(lines)


def _plot_comparison(arms: dict, path) -> None:
    """Grouped bars: harmful refusal, harmful gibberish, benign over-refusal."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = ["unsteered", "straight", "curved"]
    ref = [arms[n]["harmful_refusal_rate"] for n in names]
    gib = [arms[n]["harmful_gibberish_rate"] for n in names]
    oref = [arms[n]["benign_over_refusal_rate"] for n in names]
    x = np.arange(len(names))
    w = 0.26
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    b1 = ax.bar(x - w, ref, w, label="harmful refusal (up good)", color="#2a7")
    b2 = ax.bar(x, gib, w, label="harmful gibberish (down good)", color="#c33")
    b3 = ax.bar(x + w, oref, w, label="benign over-refusal (down good)", color="#e8a")
    for bars in (b1, b2, b3):
        for b in bars:
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.02,
                    f"{b.get_height():.2f}", ha="center", va="bottom", fontsize=7)
    ax.set_xticks(x); ax.set_xticklabels(names)
    ax.set_ylabel("rate"); ax.set_ylim(0, 1.08)
    ax.set_title("Straight chord vs curved arc at matched budget\n"
                 "curved aims to keep refusal high while cutting gibberish")
    ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig)


def _plot_geometry(offshell_straight: float, offshell_curved: float, path) -> None:
    """Two bars: mean relative off-shell displacement, straight vs curved."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5, 4))
    bars = ax.bar(["straight\n(chord)", "curved\n(arc)"],
                  [offshell_straight, offshell_curved],
                  color=["#c33", "#2a7"], width=0.6)
    for b in bars:
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.002,
                f"{b.get_height():.3f}", ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("mean |d||h||| / ||h||   (off-shell displacement, N5)")
    ax.set_title("Why the arc is gentler: it never leaves the norm shell")
    fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig)


# --------------------------------------------------------------------------- #
# The pipeline — everything below loads / runs the model.
# --------------------------------------------------------------------------- #
def main() -> dict:
    import random

    import torch

    from steering_tutorials.common.data import load_harmful_benign
    from steering_tutorials.hello_world_steering.judge import Judge
    from steering_tutorials.hello_world_steering.model_utils import (
        load_model, num_layers, last_token_activations,
    )
    from steering_tutorials.hello_world_steering.steer_vector import (
        extract_caa_vector, save_vector,
    )

    random.seed(C.SEED); np.random.seed(C.SEED); torch.manual_seed(C.SEED)

    # --- Load the model we steer, and the judge (off-family if STEER_JUDGE_MODEL) -
    model, tok = load_model(C.MODEL_ID)
    layer = min(C.LAYER, num_layers(model) - 1)
    judge = Judge(model, tok)

    # --- Data: >=500/class, split into disjoint extract / eval ----------------
    data = load_harmful_benign(C.N_PER_CLASS, C.SEED)
    harmful, benign = data["harmful"], data["benign"]
    ex_harm = harmful[:C.N_EXTRACT_PER_CLASS]
    ex_ben = benign[:C.N_EXTRACT_PER_CLASS]
    ev_harm = harmful[C.N_EXTRACT_PER_CLASS:][:C.N_EVAL_PER_CLASS]
    ev_ben = benign[C.N_EXTRACT_PER_CLASS:][:C.N_EVAL_PER_CLASS]
    print(f"[split] extract {len(ex_harm)}h/{len(ex_ben)}b   "
          f"eval {len(ev_harm)}h/{len(ev_ben)}b", file=sys.stderr)

    # --- 1. Build the refusal direction (lesson 2's diff-of-means) ------------
    vec = extract_caa_vector(model, tok, ex_harm, ex_ben, layer)
    save_vector(C.VECTOR_PATH, vec)
    v_unit = vec["v_unit"]
    print(f"[direction] layer={vec['layer']} n={vec['n']} norm={vec['norm']:.3f} "
          f"-> {C.VECTOR_PATH}", file=sys.stderr)

    # --- 2. Geometry probe (model-light): off-shell displacement of each path --
    # Read the eval prompts' last-token activations once, then apply BOTH paths in
    # NumPy and measure how far each leaves the local norm shell. This shows the
    # mechanism (straight inflates ||h||, curved does not) without any extra
    # generation. The curved rotation angle equals ALPHA by construction.
    acts = last_token_activations(model, tok, ev_harm + ev_ben, layer, log_every=0)
    straight_ep = straight_endpoint(acts, v_unit, C.ALPHA)
    curved_ep = curveball_endpoint(acts, v_unit, C.ALPHA, C.N_CURVE_STEPS)
    geometry = {
        "offshell_straight": float(np.mean(relative_offshell(straight_ep, acts))),
        "offshell_curved": float(np.mean(relative_offshell(curved_ep, acts))),
        "rotation_curved_rad": float(C.ALPHA),
    }
    print(f"[geometry] off-shell straight={geometry['offshell_straight']:.4f}  "
          f"curved={geometry['offshell_curved']:.4f}", file=sys.stderr)

    # --- 3. Evaluate the three arms on the mixed held-out eval ----------------
    def run_arm(name: str):
        """Generate + judge every eval prompt under one arm; return verdicts+recs."""
        h_verd, b_verd, recs = [], [], []
        for cls, prompts, verd_list in (
            ("harmful", ev_harm, h_verd),
            ("benign", ev_ben, b_verd),
        ):
            for i, prompt in enumerate(prompts):
                if name == "unsteered":
                    resp = curveball_generate(model, tok, prompt, None, layer,
                                              0.0, max_new_tokens=C.MAX_NEW_TOKENS)
                elif name == "straight":
                    resp = curveball_generate(model, tok, prompt, v_unit, layer,
                                              C.ALPHA, curved=False,
                                              max_new_tokens=C.MAX_NEW_TOKENS)
                else:  # curved
                    resp = curveball_generate(model, tok, prompt, v_unit, layer,
                                              C.ALPHA, curved=True,
                                              n_steps=C.N_CURVE_STEPS,
                                              max_new_tokens=C.MAX_NEW_TOKENS)
                verd_list.append(judge.verdict(prompt, resp))
                recs.append({"class": cls, "prompt": prompt,
                             "response": resp, "verdict": verd_list[-1]})
                if (i + 1) % 10 == 0:
                    print(f"[{name}/{cls}] {i + 1}/{len(prompts)}", file=sys.stderr)
        stats = _arm_stats(h_verd, b_verd)
        print(f"[{name}] harm_ref={stats['harmful_refusal_rate']:.2f} "
              f"harm_gib={stats['harmful_gibberish_rate']:.2f} "
              f"ben_oref={stats['benign_over_refusal_rate']:.2f}", file=sys.stderr)
        return stats, recs

    arms, per_arm_recs = {}, {}
    for name in ("unsteered", "straight", "curved"):
        arms[name], per_arm_recs[name] = run_arm(name)

    # --- 4. Side-by-side straight vs curved examples (a few of each class) -----
    examples = []
    straight_by_prompt = {r["prompt"]: r for r in per_arm_recs["straight"]}
    for r in per_arm_recs["curved"]:
        if len([e for e in examples if e["class"] == r["class"]]) >= 4:
            continue
        s = straight_by_prompt.get(r["prompt"], {})
        examples.append({
            "prompt": r["prompt"], "class": r["class"],
            "straight": s.get("response"), "straight_verdict": s.get("verdict"),
            "curved": r["response"], "curved_verdict": r["verdict"],
        })

    results = {
        "model_id": C.MODEL_ID,
        "layer": int(layer),
        "alpha": C.ALPHA,
        "n_curve_steps": C.N_CURVE_STEPS,
        "judge": getattr(judge, "judge_id", "self"),
        "n_extract_per_class": len(ex_harm),
        "n_eval_per_class": len(ev_harm),
        "direction": {"layer": int(vec["layer"]), "n": int(vec["n"]),
                      "norm": float(vec["norm"])},
        "geometry": geometry,
        "arms": arms,
        "examples": examples,
        "plots": {"comparison": C.COMPARISON_PNG.name,
                  "geometry": C.GEOMETRY_PNG.name},
    }

    # Save results BEFORE printing the summary, so a crash mid-print keeps the data.
    C.RESULTS_PATH.write_text(json.dumps(results, indent=2), encoding="utf-8")
    _plot_comparison(arms, C.COMPARISON_PNG)
    _plot_geometry(geometry["offshell_straight"], geometry["offshell_curved"],
                   C.GEOMETRY_PNG)
    print(f"[save] {C.RESULTS_PATH}", file=sys.stderr)
    print(f"[save] {C.COMPARISON_PNG}", file=sys.stderr)
    print(f"[save] {C.GEOMETRY_PNG}", file=sys.stderr)
    print(_summary_table(results))
    return results


if __name__ == "__main__":
    main()
