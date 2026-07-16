"""run_duality.py — the orchestrator: measure the duality, then steer two ways.

Two experiments, one pipeline:

  A. PROMPT vs VECTOR (the duality measurement).
     Build the diff-of-means refusal vector ``v`` (lesson 2). Then read each
     harmful prompt's last-token activation WITH and WITHOUT a refusal
     instruction prepended, and take the mean shift. Report
     ``cos(prompt_shift, v)`` against a random-direction baseline (and, as a
     control, the shift a *benign* instruction induces). If the refusal-prompt
     shift aligns with the steering vector far above the random floor, prompting
     and steering are dual — two handles on one internal axis.

  B. ATTENTION vs RESIDUAL (the intervention site).
     On held-out harmful prompts, generate under three arms — unsteered,
     residual-injected ``v``, attention-injected ``v`` — and judge each reply
     REFUSAL / COMPLIANCE / GIBBERISH. If injecting at the attention output
     steers too (and cleanly), the attention site is a viable, related locus,
     as the paper argues.

Everything that touches the model lives under ``main()`` so ``import
run_duality`` is a no-op (safe for tests / a webapp). Results are SAVED before
the summary is printed.

Paper: Kang, Liu, Ma, Huang, Tan & Jiang, 2026, 'Prompt-Activation Duality:
Improving Activation Steering via Attention-Level Interventions'
(arXiv:2605.10664).

RESULTS SCHEMA (kept in sync with README):
{
  "model_id": str, "layer": int, "alpha": float,
  "steering_vector": {"norm": float, "n_extract": int},
  "duality": {
     "cos_refusal_shift_vs_vector": float,
     "cos_control_shift_vs_vector": float,
     "random_cosine_baseline": float,
     "refusal_shift_norm": float, "control_shift_norm": float, "n_shift": int
  },
  "effect": [ {"site": str, "refusal_rate": float, "compliance_rate": float,
               "gibberish_rate": float, "n": int}, ... ],
  "examples": [ {"prompt": str, "baseline": str, "residual": str,
                 "attention": str, "baseline_verdict": str,
                 "residual_verdict": str, "attention_verdict": str}, ... ],
  "plots": {"duality_cosine": "...png", "attention_vs_residual": "...png"}
}
"""
from __future__ import annotations

import json
import sys

from . import config as C


# --------------------------------------------------------------------------- #
# Pure helpers (no model) — safe to unit-test in isolation.
# --------------------------------------------------------------------------- #
def _rates(verdicts: list[str]) -> dict[str, float]:
    """Fraction of REFUSAL / COMPLIANCE / GIBBERISH among a list of verdicts."""
    n = max(1, len(verdicts))
    return {
        "refusal_rate": verdicts.count("REFUSAL") / n,
        "compliance_rate": verdicts.count("COMPLIANCE") / n,
        "gibberish_rate": verdicts.count("GIBBERISH") / n,
    }


def _summary_table(results: dict) -> str:
    """Plain-text (ASCII-only) recap printed at the end of a run."""
    d = results["duality"]
    lines = ["", "=" * 66, "PROMPT-ACTIVATION DUALITY SUMMARY", "=" * 66,
             f"model : {results['model_id']}",
             f"layer : {results['layer']}   alpha: {results['alpha']:.2f}   "
             f"vector norm: {results['steering_vector']['norm']:.3f}",
             "",
             "A. Prompt vs vector (direction agreement, cosine):",
             f"  refusal-instruction shift . v : {d['cos_refusal_shift_vs_vector']:+.3f}",
             f"  control-instruction shift . v : {d['cos_control_shift_vs_vector']:+.3f}",
             f"  random-direction baseline     :  {d['random_cosine_baseline']:.3f}",
             "",
             "B. Attention vs residual (judged on held-out harmful):",
             f"  {'site':>10} {'refusal':>9} {'comply':>9} {'gibber':>9} {'n':>5}"]
    for r in results["effect"]:
        lines.append(f"  {r['site']:>10} {r['refusal_rate']:>9.2f} "
                     f"{r['compliance_rate']:>9.2f} {r['gibberish_rate']:>9.2f} "
                     f"{r['n']:>5d}")
    lines += ["=" * 66, ""]
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Plotting — matplotlib Agg (headless).
# --------------------------------------------------------------------------- #
def _plot_cosine(duality: dict, path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = ["refusal\nshift . v", "control\nshift . v", "random\nbaseline"]
    vals = [abs(duality["cos_refusal_shift_vs_vector"]),
            abs(duality["cos_control_shift_vs_vector"]),
            duality["random_cosine_baseline"]]
    colors = ["#2a7", "#c93", "#999"]
    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(labels, vals, color=colors)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.005, f"{v:.3f}",
                ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("|cosine| with steering vector v")
    ax.set_title("Duality: does a refusal PROMPT shift activations\n"
                 "along the steering VECTOR? (higher = more dual)")
    ax.set_ylim(0, max(vals) * 1.25 + 1e-6)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def _plot_effect(effect: list[dict], path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    sites = [r["site"] for r in effect]
    x = np.arange(len(sites))
    w = 0.38
    refusal = [r["refusal_rate"] for r in effect]
    gibber = [r["gibberish_rate"] for r in effect]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(x - w / 2, refusal, w, label="refusal (want high)", color="#2a7")
    ax.bar(x + w / 2, gibber, w, label="gibberish (want low)", color="#c33")
    ax.set_xticks(x)
    ax.set_xticklabels(sites)
    ax.set_ylabel("rate on held-out harmful prompts")
    ax.set_ylim(0, 1.05)
    ax.set_title("Same vector, two injection sites:\nresidual stream vs attention output")
    ax.legend()
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# The pipeline — everything below here loads / runs the model.
# --------------------------------------------------------------------------- #
def main() -> dict:
    import random

    import numpy as np
    import torch

    from steering_tutorials.hello_world_steering.model_utils import (
        load_model, last_token_activations, num_layers,
    )
    from .duality import (
        prompt_shift_direction, cosine, random_cosine_baseline, steered_generate,
    )
    from steering_tutorials.hello_world_steering.steer_vector import (
        extract_caa_vector, save_vector,
    )
    from steering_tutorials.hello_world_steering.judge import Judge
    from steering_tutorials.common.data import load_harmful_benign

    # Reproducibility: pin every RNG before anything stochastic happens.
    random.seed(C.SEED)
    np.random.seed(C.SEED)
    torch.manual_seed(C.SEED)

    # --- Load the model we read (shift + vector) and write (steer) ------------
    model, tok = load_model(C.MODEL_ID)
    layer = min(C.LAYER, num_layers(model) - 1)

    # --- Data: disjoint extract / eval halves per class -----------------------
    data = load_harmful_benign(C.N_PER_CLASS, C.SEED)
    extract_harmful = data["harmful"][:C.N_EXTRACT]
    extract_benign = data["benign"][:C.N_EXTRACT]
    eval_harmful = data["harmful"][C.N_EXTRACT:C.N_EXTRACT + C.N_EVAL_PER_CLASS]
    print(f"[split] extract: {len(extract_harmful)}h/{len(extract_benign)}b   "
          f"eval(harmful): {len(eval_harmful)}", file=sys.stderr)

    # --- 1. Build the diff-of-means refusal vector v (lesson 2) ---------------
    vec = extract_caa_vector(model, tok, extract_harmful, extract_benign, layer)
    save_vector(C.VECTOR_PATH, vec)
    v_unit = vec["v_unit"]
    print(f"[vector] layer={vec['layer']} n={vec['n']} norm={vec['norm']:.3f} "
          f"-> {C.VECTOR_PATH}", file=sys.stderr)

    # ======================================================================= #
    # EXPERIMENT A — PROMPT vs VECTOR (the duality)
    # ======================================================================= #
    # Read each harmful extract prompt's last-token activation three ways:
    #   plain, with the REFUSAL instruction, with the CONTROL instruction.
    plain = last_token_activations(model, tok, extract_harmful, layer)
    refusal_prompts = [C.REFUSAL_INSTRUCTION + C.INSTRUCTION_SEP + p
                       for p in extract_harmful]
    control_prompts = [C.CONTROL_INSTRUCTION + C.INSTRUCTION_SEP + p
                       for p in extract_harmful]
    acts_refusal = last_token_activations(model, tok, refusal_prompts, layer)
    acts_control = last_token_activations(model, tok, control_prompts, layer)

    refusal_shift = prompt_shift_direction(acts_refusal, plain)
    control_shift = prompt_shift_direction(acts_control, plain)
    torch.save({"refusal_shift": refusal_shift, "control_shift": control_shift},
               C.SHIFT_PATH)

    cos_refusal = cosine(refusal_shift["v_raw"], vec["v_raw"])
    cos_control = cosine(control_shift["v_raw"], vec["v_raw"])
    rand_base = random_cosine_baseline(vec["v_raw"], n_samples=300, seed=C.SEED)

    duality = {
        "cos_refusal_shift_vs_vector": cos_refusal,
        "cos_control_shift_vs_vector": cos_control,
        "random_cosine_baseline": rand_base,
        "refusal_shift_norm": refusal_shift["norm"],
        "control_shift_norm": control_shift["norm"],
        "n_shift": refusal_shift["n"],
    }
    print(f"[duality] cos(refusal shift, v)={cos_refusal:+.3f}  "
          f"cos(control shift, v)={cos_control:+.3f}  "
          f"random~{rand_base:.3f}", file=sys.stderr)

    # ======================================================================= #
    # EXPERIMENT B — ATTENTION vs RESIDUAL (the site)
    # ======================================================================= #
    judge = Judge(model, tok)
    sites = ["none", "residual", "attention"]
    per_site_verdicts: dict[str, list[str]] = {s: [] for s in sites}
    # Keep the per-prompt generations so we can show side-by-side examples.
    rows: list[dict] = []

    for i, prompt in enumerate(eval_harmful):
        gens: dict[str, str] = {}
        for site in sites:
            resp = steered_generate(
                model, tok, prompt, vector=v_unit, layer=layer, alpha=C.ALPHA,
                site=site, max_new_tokens=C.MAX_NEW_TOKENS,
            )
            gens[site] = resp
            per_site_verdicts[site].append(judge.verdict(prompt, resp))
        rows.append({
            "prompt": prompt,
            "baseline": gens["none"],
            "residual": gens["residual"],
            "attention": gens["attention"],
        })
        if (i + 1) % 5 == 0:
            print(f"[effect] {i + 1}/{len(eval_harmful)}", file=sys.stderr)

    effect = []
    site_label = {"none": "unsteered", "residual": "residual", "attention": "attention"}
    for s in sites:
        rec = {"site": site_label[s], "n": len(eval_harmful),
               **_rates(per_site_verdicts[s])}
        effect.append(rec)

    # Judge each example's three generations for the side-by-side table.
    examples = []
    for r in rows[:10]:
        examples.append({
            "prompt": r["prompt"],
            "baseline": r["baseline"],
            "residual": r["residual"],
            "attention": r["attention"],
            "baseline_verdict": judge.verdict(r["prompt"], r["baseline"]),
            "residual_verdict": judge.verdict(r["prompt"], r["residual"]),
            "attention_verdict": judge.verdict(r["prompt"], r["attention"]),
        })

    results = {
        "model_id": C.MODEL_ID,
        "layer": int(layer),
        "alpha": float(C.ALPHA),
        "steering_vector": {"norm": float(vec["norm"]), "n_extract": int(vec["n"])},
        "duality": duality,
        "effect": effect,
        "examples": examples,
        "plots": {"duality_cosine": C.COSINE_PNG.name,
                  "attention_vs_residual": C.EFFECT_PNG.name},
    }

    # --- Persist FIRST, then plot, then print (results saved before summary) --
    C.RESULTS_PATH.write_text(json.dumps(results, indent=2), encoding="utf-8")
    _plot_cosine(duality, C.COSINE_PNG)
    _plot_effect(effect, C.EFFECT_PNG)
    print(f"[save] {C.RESULTS_PATH}", file=sys.stderr)
    print(f"[save] {C.COSINE_PNG}", file=sys.stderr)
    print(f"[save] {C.EFFECT_PNG}", file=sys.stderr)
    print(_summary_table(results))
    return results


if __name__ == "__main__":
    main()
