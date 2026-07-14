"""run_steering.py — the orchestrator: build a refusal vector, then steer with it.

This is the spine of lesson 2. It wires together the four pieces the peers own
(``model_utils``, ``steer_vector``, ``gate``, ``judge``) into one pipeline and
writes ``artifacts/results.json`` + two plots that the webapp and README read.

The lifecycle, end to end
--------------------------
1. EXTRACT.  Load the abliterated Gemma. Split each JailbreakBench class into a
   disjoint *extract* half (first ``N_EXTRACT``) and *eval* half (the rest).
   Contrast the extract halves with diff-of-means to get one "refuse this"
   direction (the CAA vector). Save it.

2. UNCONDITIONAL arm.  On the held-out harmful prompts, steer at every alpha in
   ``ALPHAS`` and have the judge label each output REFUSAL / COMPLIANCE /
   GIBBERISH. This traces the dose-response curve: how hard can we push before
   coherent refusals collapse into gibberish?

3. CONDITIONAL arm.  On a MIXED eval set (held-out harmful + held-out benign),
   ask the lesson-1 probe (the gate) whether each prompt is harmful. Steer only
   when it fires, at a single alpha chosen from the arm-2 curve. Judge both the
   unsteered baseline and the gated output. This is the whole point: refuse the
   harmful ones, leave the benign ones untouched.

4. REPORT.  Save the schema below, render two PNGs, print a summary table.

Everything that touches the model lives under ``main()`` so ``import
run_steering`` is a no-op (safe to import for tests / the webapp).

RESULTS SCHEMA (kept in sync with app.py + README)
--------------------------------------------------
{
  "model_id": str, "steer_layer": int, "alphas": [float, ...],
  "steering_vector": {"norm": float, "layer": int, "n_extract": int},
  "unconditional": [
      {"alpha": float, "refusal_rate": float, "compliance_rate": float,
       "gibberish_rate": float, "n": int}, ...
  ],
  "conditional": {
      "alpha": float, "harmful_refusal_rate": float,
      "benign_over_refusal_rate": float, "gibberish_rate": float,
      "gate_accuracy": float, "n_harmful": int, "n_benign": int
  },
  "examples": [
      {"prompt": str, "harmful": bool, "gated": bool,
       "baseline_response": str, "steered_response": str,
       "baseline_verdict": str, "steered_verdict": str}, ...
  ],
  "plots": {"rates_vs_alpha": "rates_vs_alpha.png",
            "conditional": "conditional.png"}
}
"""
from __future__ import annotations

import json
import sys

from . import config as C


# --------------------------------------------------------------------------- #
# Small pure helpers (no model needed) — safe to unit-test in isolation.
# --------------------------------------------------------------------------- #
def _rates(verdicts: list[str]) -> dict[str, float]:
    """Fraction of REFUSAL / COMPLIANCE / GIBBERISH among a list of verdicts."""
    n = max(1, len(verdicts))
    return {
        "refusal_rate": verdicts.count("REFUSAL") / n,
        "compliance_rate": verdicts.count("COMPLIANCE") / n,
        "gibberish_rate": verdicts.count("GIBBERISH") / n,
    }


def choose_conditional_alpha(unconditional: list[dict],
                             gibberish_tolerance: float) -> float:
    """Pick the alpha for the conditional arm from the dose-response curve.

    Rule: among the STEERING alphas (alpha > 0) whose gibberish rate stays at or
    below ``gibberish_tolerance``, take the highest refusal rate; break ties
    toward the SMALLEST alpha (least collateral damage to coherence). If every
    alpha is too gibberishy, fall back to the smallest steering alpha. This is
    the "highest refusal before gibberish rises, smallest that gets us there"
    heuristic documented in the task spec.
    """
    steering = [r for r in unconditional if r["alpha"] > 0.0]
    if not steering:
        return 0.0
    clean = [r for r in steering if r["gibberish_rate"] <= gibberish_tolerance]
    pool = clean if clean else steering
    # max refusal_rate, then min alpha as the tie-breaker.
    best = max(pool, key=lambda r: (r["refusal_rate"], -r["alpha"]))
    return float(best["alpha"])


def _summary_table(results: dict) -> str:
    """A plain-text recap printed at the end of a run."""
    lines = ["", "=" * 64, "STEERING SUMMARY", "=" * 64,
             f"model      : {results['model_id']}",
             f"steer layer: {results['steer_layer']}   "
             f"vector norm: {results['steering_vector']['norm']:.3f}",
             "", "Unconditional (held-out harmful prompts):",
             f"  {'alpha':>6} {'refusal':>9} {'comply':>9} {'gibber':>9}"]
    for r in results["unconditional"]:
        lines.append(f"  {r['alpha']:>6.2f} {r['refusal_rate']:>9.2f} "
                     f"{r['compliance_rate']:>9.2f} {r['gibberish_rate']:>9.2f}")
    c = results["conditional"]
    lines += ["", f"Conditional (gate + steer @ alpha={c['alpha']:.2f}):",
              f"  harmful refusal rate  : {c['harmful_refusal_rate']:.2f} "
              f"(n={c['n_harmful']})",
              f"  benign over-refusal   : {c['benign_over_refusal_rate']:.2f} "
              f"(n={c['n_benign']})",
              f"  gibberish rate        : {c['gibberish_rate']:.2f}",
              f"  gate accuracy         : {c['gate_accuracy']:.2f}", "=" * 64, ""]
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Plotting — matplotlib with the Agg backend (headless, no display needed).
# --------------------------------------------------------------------------- #
def _plot_rates_vs_alpha(unconditional: list[dict], path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    alphas = [r["alpha"] for r in unconditional]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(alphas, [r["refusal_rate"] for r in unconditional],
            "o-", label="refusal", color="#2a7")
    ax.plot(alphas, [r["compliance_rate"] for r in unconditional],
            "s-", label="compliance", color="#37a")
    ax.plot(alphas, [r["gibberish_rate"] for r in unconditional],
            "^-", label="gibberish", color="#c33")
    ax.set_xlabel("steering strength  α  (fraction of residual norm)")
    ax.set_ylabel("rate on held-out harmful prompts")
    ax.set_title("Dose-response: what steering does as α grows")
    ax.set_ylim(-0.02, 1.02)
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def _plot_conditional(conditional: dict, path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = ["harmful\nrefusal", "benign\nover-refusal", "gibberish"]
    vals = [conditional["harmful_refusal_rate"],
            conditional["benign_over_refusal_rate"],
            conditional["gibberish_rate"]]
    colors = ["#2a7", "#c93", "#c33"]
    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(labels, vals, color=colors)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.02, f"{v:.2f}",
                ha="center", va="bottom")
    ax.set_ylabel("rate")
    ax.set_ylim(0, 1.05)
    ax.set_title(f"Conditional steering @ α={conditional['alpha']:.2f}\n"
                 "(want: high harmful-refusal, low over-refusal + gibberish)")
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

    # Peer-owned modules. Imported inside main() so a bare ``import
    # run_steering`` never drags in torch or triggers a model load.
    from .model_utils import load_model, generate, num_layers
    from .steer_vector import extract_caa_vector, save_vector
    from .gate import HarmGate
    from .judge import Judge
    from .data import load_harmful_benign

    # Reproducibility: pin every RNG before anything stochastic happens.
    random.seed(C.SEED)
    np.random.seed(C.SEED)
    torch.manual_seed(C.SEED)

    # --- Load the model we will both read (gate) and write (steer) ------------
    model, tok = load_model(C.MODEL_ID)
    layer = min(C.STEER_LAYER, num_layers(model) - 1)

    # --- Data: split each class into disjoint extract / eval halves -----------
    data = load_harmful_benign(C.N_PER_CLASS, C.SEED)
    extract_harmful = data["harmful"][:C.N_EXTRACT]
    extract_benign = data["benign"][:C.N_EXTRACT]
    eval_harmful = data["harmful"][C.N_EXTRACT:]
    eval_benign = data["benign"][C.N_EXTRACT:]
    print(f"[split] extract: {len(extract_harmful)}h/{len(extract_benign)}b   "
          f"eval: {len(eval_harmful)}h/{len(eval_benign)}b", file=sys.stderr)

    # --- 1. Build the refusal steering vector (diff-of-means) -----------------
    vec = extract_caa_vector(model, tok, extract_harmful, extract_benign, layer)
    save_vector(C.VECTOR_PATH, vec)
    v_unit = vec["v_unit"]  # unit direction; alpha supplies the magnitude
    print(f"[vector] layer={vec['layer']} n={vec['n']} norm={vec['norm']:.3f} "
          f"-> {C.VECTOR_PATH}", file=sys.stderr)

    judge = Judge(model, tok)

    # --- 2. UNCONDITIONAL arm: sweep alpha on held-out harmful prompts --------
    unconditional: list[dict] = []
    for alpha in C.ALPHAS:
        verdicts: list[str] = []
        for i, prompt in enumerate(eval_harmful):
            # alpha == 0 is the true baseline: no vector, no injection.
            resp = generate(
                model, tok, prompt,
                max_new_tokens=C.MAX_NEW_TOKENS,
                vector=(None if alpha == 0.0 else v_unit),
                layer=layer, alpha=alpha, operation="relative_add",
            )
            verdicts.append(judge.verdict(prompt, resp))
            if (i + 1) % 5 == 0:
                print(f"[uncond α={alpha:.2f}] {i + 1}/{len(eval_harmful)}",
                      file=sys.stderr)
        rec = {"alpha": float(alpha), "n": len(eval_harmful), **_rates(verdicts)}
        unconditional.append(rec)
        print(f"[uncond α={alpha:.2f}] refusal={rec['refusal_rate']:.2f} "
              f"comply={rec['compliance_rate']:.2f} "
              f"gibber={rec['gibberish_rate']:.2f}", file=sys.stderr)

    # Pick the single alpha for the conditional arm from that curve.
    steer_alpha = choose_conditional_alpha(unconditional, C.GIBBERISH_TOLERANCE)
    print(f"[choose] conditional alpha = {steer_alpha:.2f}", file=sys.stderr)

    # --- 3. CONDITIONAL arm: gate-then-steer on the MIXED eval set ------------
    gate = HarmGate(model, tok)
    mixed = ([(p, True) for p in eval_harmful]
             + [(p, False) for p in eval_benign])

    records: list[dict] = []  # per-prompt, for metrics + example sampling
    for i, (prompt, is_harmful_true) in enumerate(mixed):
        fired, prob = gate.is_harmful(prompt)
        used_alpha = steer_alpha if fired else 0.0

        baseline_resp = generate(
            model, tok, prompt, max_new_tokens=C.MAX_NEW_TOKENS,
            vector=None, layer=layer, alpha=0.0, operation="relative_add",
        )
        gated_resp = generate(
            model, tok, prompt, max_new_tokens=C.MAX_NEW_TOKENS,
            vector=(v_unit if fired else None),
            layer=layer, alpha=used_alpha, operation="relative_add",
        )
        records.append({
            "prompt": prompt,
            "harmful": bool(is_harmful_true),
            "gated": bool(fired),
            "gate_prob": float(prob),
            "baseline_response": baseline_resp,
            "steered_response": gated_resp,
            "baseline_verdict": judge.verdict(prompt, baseline_resp),
            "steered_verdict": judge.verdict(prompt, gated_resp),
        })
        if (i + 1) % 5 == 0:
            print(f"[cond] {i + 1}/{len(mixed)}", file=sys.stderr)

    harmful_recs = [r for r in records if r["harmful"]]
    benign_recs = [r for r in records if not r["harmful"]]
    harmful_refusal_rate = (
        sum(r["steered_verdict"] == "REFUSAL" for r in harmful_recs)
        / max(1, len(harmful_recs)))
    benign_over_refusal_rate = (
        sum(r["steered_verdict"] == "REFUSAL" for r in benign_recs)
        / max(1, len(benign_recs)))
    cond_gibberish_rate = (
        sum(r["steered_verdict"] == "GIBBERISH" for r in records)
        / max(1, len(records)))
    gate_accuracy = (
        sum(r["gated"] == r["harmful"] for r in records) / max(1, len(records)))

    conditional = {
        "alpha": float(steer_alpha),
        "harmful_refusal_rate": harmful_refusal_rate,
        "benign_over_refusal_rate": benign_over_refusal_rate,
        "gibberish_rate": cond_gibberish_rate,
        "gate_accuracy": gate_accuracy,
        "n_harmful": len(harmful_recs),
        "n_benign": len(benign_recs),
    }

    # --- Pick 8–12 side-by-side examples: mostly gated-harmful (the money
    # shot: baseline complies, steered refuses), plus a couple of benign ones
    # to show the gate leaving harmless prompts alone. ------------------------
    example_keys = ("prompt", "harmful", "gated", "baseline_response",
                    "steered_response", "baseline_verdict", "steered_verdict")
    gated_harmful = [r for r in harmful_recs if r["gated"]]
    other = [r for r in records if r not in gated_harmful]
    chosen = gated_harmful[:8] + benign_recs[:2]
    if len(chosen) < 8:  # backfill if the gate fired on few prompts
        chosen += [r for r in other if r not in chosen][:8 - len(chosen)]
    examples = [{k: r[k] for k in example_keys} for r in chosen[:12]]

    results = {
        "model_id": C.MODEL_ID,
        "steer_layer": int(layer),
        "alphas": [float(a) for a in C.ALPHAS],
        "steering_vector": {
            "norm": float(vec["norm"]),
            "layer": int(vec["layer"]),
            "n_extract": int(vec["n"]),
        },
        "unconditional": unconditional,
        "conditional": conditional,
        "examples": examples,
        "plots": {"rates_vs_alpha": C.RATES_PNG.name,
                  "conditional": C.CONDITIONAL_PNG.name},
    }

    # --- 4. Persist + plot + print -------------------------------------------
    C.RESULTS_PATH.write_text(json.dumps(results, indent=2), encoding="utf-8")
    _plot_rates_vs_alpha(unconditional, C.RATES_PNG)
    _plot_conditional(conditional, C.CONDITIONAL_PNG)
    print(f"[save] {C.RESULTS_PATH}", file=sys.stderr)
    print(f"[save] {C.RATES_PNG}", file=sys.stderr)
    print(f"[save] {C.CONDITIONAL_PNG}", file=sys.stderr)
    print(_summary_table(results))
    return results


if __name__ == "__main__":
    main()
