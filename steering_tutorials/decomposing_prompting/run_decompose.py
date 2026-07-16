"""run_decompose.py -- orchestrator: decompose prompting's activation footprint,
then test how much of it a plain steering vector reproduces.

Spine of the lesson. Everything that touches the model lives under ``main()`` so
``import run_decompose`` is a cheap no-op (safe for tests / import-checks).

Lifecycle
---------
1. EXTRACT u.   Load the abliterated Gemma. Build the diff-of-means refusal
   direction ``u`` from a disjoint extract split of harmful vs benign prompts
   (this is the axis we project prompting's effect onto).
2. DECOMPOSE.   On a held-out slice of harmful prompts, measure the per-prompt
   delta d(x) = act(WITH the refusal instruction) - act(WITHOUT), and decompose
   it: on-direction (steering-vector-like) energy, off-direction residual,
   cross-prompt consistency, shared-translation fraction. This is the READ half
   and the paper's core measurement.
3. WRITE check. On a small held-out slice, generate + judge four conditions:
      baseline        - no instruction, no steering
      prompting       - WITH the instruction (the strong AxBench baseline)
      steer(v_proj)   - steer the bare prompt with the on-direction translation
      steer(v_shared) - steer the bare prompt with the full mean translation
   and report how much of prompting's refusal gain each reconstruction recovers.
4. REPORT.  Save results.json, two PNGs, print a summary table.

RESULTS SCHEMA (kept in sync with README + infer.py)
----------------------------------------------------
{
  "model_id": str, "layer": int, "instruction": str,
  "refusal_direction": {"norm": float, "layer": int, "n_extract": int},
  "decomposition": {
      "n": int, "mean_delta_norm": float, "mean_abs_proj": float,
      "mean_signed_proj": float, "mean_residual_norm": float,
      "on_direction_frac": float, "consistency": float,
      "shared_translation_frac": float
  },
  "recovery": {
      "n": int,
      "baseline_refusal": float, "prompting_refusal": float,
      "steer_proj_refusal": float, "steer_shared_refusal": float,
      "steer_proj_gibberish": float, "steer_shared_gibberish": float,
      "recovery_proj": float, "recovery_shared": float
  },
  "examples": [
      {"prompt": str, "baseline": str, "prompting": str,
       "steer_proj": str, "baseline_verdict": str, "prompting_verdict": str,
       "steer_proj_verdict": str}, ...
  ],
  "plots": {"decomposition": "decomposition.png", "recovery": "recovery.png"}
}
"""
from __future__ import annotations

import json
import sys

from . import config as C


# --------------------------------------------------------------------------- #
# Pure helpers (no model) -- safe to unit-test in isolation.
# --------------------------------------------------------------------------- #
def _refusal_rate(verdicts: list[str]) -> float:
    """Fraction of REFUSAL verdicts (the WRITE-check success metric)."""
    n = max(1, len(verdicts))
    return sum(v == "REFUSAL" for v in verdicts) / n


def _gibberish_rate(verdicts: list[str]) -> float:
    n = max(1, len(verdicts))
    return sum(v == "GIBBERISH" for v in verdicts) / n


def recovery_fraction(baseline: float, prompting: float, reconstructed: float) -> float:
    """How much of prompting's refusal GAIN a reconstruction reproduces.

    (reconstructed - baseline) / (prompting - baseline), clamped to [0, 1.5].
    ~1.0 means "the steering vector recovers all of prompting's effect" (prompting
    was essentially a translation); ~0 means the on-direction shift explains little
    and the off-direction residual is doing the work. The 1.5 ceiling keeps a
    reconstruction that OVERSHOOTS (steers harder than prompting) readable rather
    than exploding when the denominator is tiny.
    """
    gain = prompting - baseline
    if gain <= 1e-6:
        return 0.0
    return float(max(0.0, min(1.5, (reconstructed - baseline) / gain)))


def _summary_table(results: dict) -> str:
    """Plain-text recap printed last (contains the measured numbers)."""
    d = results["decomposition"]
    r = results["recovery"]
    lines = [
        "", "=" * 66, "DECOMPOSING PROMPTING - SUMMARY", "=" * 66,
        f"model      : {results['model_id']}",
        f"layer      : {results['layer']}   "
        f"refusal-direction norm: {results['refusal_direction']['norm']:.3f}",
        "",
        f"Decomposition of prompting's per-prompt delta (n={d['n']} harmful):",
        f"  mean ||delta||                 : {d['mean_delta_norm']:.3f}",
        f"  on-direction energy fraction   : {d['on_direction_frac']:.3f}"
        "   (how steering-vector-like)",
        f"  mean off-direction residual    : {d['mean_residual_norm']:.3f}",
        f"  cross-prompt consistency       : {d['consistency']:.3f}"
        "   (1.0 = one shared translation)",
        f"  shared-translation fraction    : {d['shared_translation_frac']:.3f}",
        "",
        f"WRITE check - refusal rate by condition (n={r['n']} harmful):",
        f"  baseline (no instruction)      : {r['baseline_refusal']:.2f}",
        f"  prompting (WITH instruction)   : {r['prompting_refusal']:.2f}",
        f"  steer(v_proj)  on-direction    : {r['steer_proj_refusal']:.2f}",
        f"  steer(v_shared) full transl.   : {r['steer_shared_refusal']:.2f}",
        f"  -> recovery of prompting gain  : "
        f"proj={r['recovery_proj']:.2f}  shared={r['recovery_shared']:.2f}",
        "",
        "Read: a high on-direction fraction + high recovery means prompting is,",
        "mechanically, mostly a single steering vector the model applies itself.",
        "=" * 66, "",
    ]
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Plotting -- matplotlib Agg backend (headless).
# --------------------------------------------------------------------------- #
def _plot_decomposition(d: dict, path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = ["on-direction\nenergy frac", "cross-prompt\nconsistency",
              "shared-translation\nfraction"]
    vals = [d["on_direction_frac"], d["consistency"], d["shared_translation_frac"]]
    colors = ["#2a7", "#37a", "#7a3"]
    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(labels, vals, color=colors)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.02, f"{v:.2f}",
                ha="center", va="bottom")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("fraction / cosine (in [0, 1])")
    ax.set_title("How steering-vector-like is prompting's activation delta?\n"
                 "(higher = more of prompting is one shared translation)")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def _plot_recovery(r: dict, path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = ["baseline", "prompting", "steer\n(v_proj)", "steer\n(v_shared)"]
    vals = [r["baseline_refusal"], r["prompting_refusal"],
            r["steer_proj_refusal"], r["steer_shared_refusal"]]
    colors = ["#999", "#37a", "#2a7", "#7a3"]
    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(labels, vals, color=colors)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.02, f"{v:.2f}",
                ha="center", va="bottom")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("refusal rate on held-out harmful prompts")
    ax.set_title("Does a steering vector reproduce prompting's refusal effect?\n"
                 f"(recovery: proj={r['recovery_proj']:.2f}, "
                 f"shared={r['recovery_shared']:.2f})")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# The pipeline -- everything below loads / runs the model.
# --------------------------------------------------------------------------- #
def main() -> dict:
    import random

    import numpy as np
    import torch

    # Peer-owned modules (lesson-2 mechanics) imported inside main() so a bare
    # ``import run_decompose`` never drags in torch or triggers a model load.
    from steering_tutorials.hello_world_steering.model_utils import (
        load_model, generate, num_layers,
    )
    from steering_tutorials.hello_world_steering.steer_vector import (
        extract_caa_vector, save_vector,
    )
    from steering_tutorials.hello_world_steering.judge import Judge
    from steering_tutorials.common.data import load_harmful_benign
    from .decompose import prompt_deltas, decompose_prompt_deltas

    # Reproducibility: pin every RNG before anything stochastic happens.
    random.seed(C.SEED)
    np.random.seed(C.SEED)
    torch.manual_seed(C.SEED)

    # --- Load the model we both read and steer --------------------------------
    model, tok = load_model(C.MODEL_ID)
    layer = min(C.STEER_LAYER, num_layers(model) - 1)

    # --- Data: disjoint extract / decompose / write splits --------------------
    data = load_harmful_benign(C.N_PER_CLASS, C.SEED)
    extract_harmful = data["harmful"][:C.N_EXTRACT]
    extract_benign = data["benign"][:C.N_EXTRACT]
    rest_harmful = data["harmful"][C.N_EXTRACT:]
    decomp_harmful = rest_harmful[:C.N_DECOMP]
    write_harmful = rest_harmful[C.N_DECOMP:C.N_DECOMP + C.N_WRITE]
    print(f"[split] extract={len(extract_harmful)}h/{len(extract_benign)}b  "
          f"decompose={len(decomp_harmful)}h  write={len(write_harmful)}h",
          file=sys.stderr)

    # --- 1. Refusal direction u (diff-of-means over the extract split) --------
    vec = extract_caa_vector(model, tok, extract_harmful, extract_benign, layer)
    save_vector(C.DIRECTION_PATH, vec)
    u_unit = vec["v_unit"]
    print(f"[direction] layer={vec['layer']} n={vec['n']} norm={vec['norm']:.3f} "
          f"-> {C.DIRECTION_PATH}", file=sys.stderr)

    # --- 2. DECOMPOSE prompting's per-prompt delta ----------------------------
    print(f"[decompose] reading WITH/WITHOUT deltas for {len(decomp_harmful)} "
          f"harmful prompts...", file=sys.stderr)
    deltas = prompt_deltas(model, tok, decomp_harmful, layer, C.REFUSAL_INSTRUCTION)
    dec = decompose_prompt_deltas(deltas, u_unit)
    v_proj = dec.pop("v_proj")        # on-direction translation (raw units)
    v_shared = dec.pop("v_shared")    # full mean translation (raw units)
    dec.pop("u_unit", None)
    print(f"[decompose] on_direction_frac={dec['on_direction_frac']:.3f} "
          f"consistency={dec['consistency']:.3f} "
          f"shared_frac={dec['shared_translation_frac']:.3f}", file=sys.stderr)

    # --- 3. WRITE check: does a steering vector reproduce prompting? ----------
    judge = Judge(model, tok)
    base_v, prompt_v, proj_v, shared_v = [], [], [], []
    examples: list[dict] = []
    for i, p in enumerate(write_harmful):
        # baseline: no instruction, no steering.
        base = generate(model, tok, p, max_new_tokens=C.MAX_NEW_TOKENS, alpha=0.0)
        # prompting: prepend the instruction (the strong AxBench baseline).
        prompting = generate(model, tok, f"{C.REFUSAL_INSTRUCTION}\n\n{p}",
                             max_new_tokens=C.MAX_NEW_TOKENS, alpha=0.0)
        # steer the BARE prompt with the extracted translation components. The
        # "add" op injects the raw vector as-is (alpha=1.0), so we add exactly
        # the average shift prompting produced -- no strength tuning.
        steer_proj = generate(model, tok, p, max_new_tokens=C.MAX_NEW_TOKENS,
                             vector=v_proj, layer=layer, alpha=1.0, operation="add")
        steer_shared = generate(model, tok, p, max_new_tokens=C.MAX_NEW_TOKENS,
                               vector=v_shared, layer=layer, alpha=1.0,
                               operation="add")

        bv = judge.verdict(p, base)
        pv = judge.verdict(p, prompting)
        prv = judge.verdict(p, steer_proj)
        shv = judge.verdict(p, steer_shared)
        base_v.append(bv); prompt_v.append(pv); proj_v.append(prv); shared_v.append(shv)

        if len(examples) < 10:
            examples.append({
                "prompt": p,
                "baseline": base, "prompting": prompting, "steer_proj": steer_proj,
                "baseline_verdict": bv, "prompting_verdict": pv,
                "steer_proj_verdict": prv,
            })
        if (i + 1) % 5 == 0:
            print(f"[write] {i + 1}/{len(write_harmful)}", file=sys.stderr)

    base_r = _refusal_rate(base_v)
    prompt_r = _refusal_rate(prompt_v)
    proj_r = _refusal_rate(proj_v)
    shared_r = _refusal_rate(shared_v)
    recovery = {
        "n": len(write_harmful),
        "baseline_refusal": base_r,
        "prompting_refusal": prompt_r,
        "steer_proj_refusal": proj_r,
        "steer_shared_refusal": shared_r,
        "steer_proj_gibberish": _gibberish_rate(proj_v),
        "steer_shared_gibberish": _gibberish_rate(shared_v),
        "recovery_proj": recovery_fraction(base_r, prompt_r, proj_r),
        "recovery_shared": recovery_fraction(base_r, prompt_r, shared_r),
    }

    results = {
        "model_id": C.MODEL_ID,
        "layer": int(layer),
        "instruction": C.REFUSAL_INSTRUCTION,
        "refusal_direction": {
            "norm": float(vec["norm"]), "layer": int(vec["layer"]),
            "n_extract": int(vec["n"]),
        },
        "decomposition": dec,
        "recovery": recovery,
        "examples": examples,
        "plots": {"decomposition": C.DECOMP_PNG.name,
                  "recovery": C.RECOVERY_PNG.name},
    }

    # --- 4. Persist + plot + print (numbers first, takeaway last) -------------
    C.RESULTS_PATH.write_text(json.dumps(results, indent=2), encoding="utf-8")
    # Save the raw reconstruction vectors so infer.py can steer one prompt with
    # exactly the translation this run extracted.
    torch.save({"v_proj": v_proj, "v_shared": v_shared, "layer": int(layer)},
               C.RECON_PATH)
    _plot_decomposition(dec, C.DECOMP_PNG)
    _plot_recovery(recovery, C.RECOVERY_PNG)
    print(f"[save] {C.RESULTS_PATH}", file=sys.stderr)
    print(f"[save] {C.DECOMP_PNG}", file=sys.stderr)
    print(f"[save] {C.RECOVERY_PNG}", file=sys.stderr)
    print(_summary_table(results))
    return results


if __name__ == "__main__":
    main()
