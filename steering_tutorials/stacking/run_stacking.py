"""run_stacking.py — the orchestrator: build the additive 2->N ladder and read it.

This is the spine of lesson 12. It builds ONE refusal direction (diff-of-means,
reused from lesson 2), assembles the three archetypal priors, then walks the
additive ladder — rung 1 = A alone; rung 2a = A + B (disjoint site); rung 2b =
A + B' (same site, incompatible op); rung 3 = all-on hybrid — measuring at each
rung the target behavior (judge refusal rate), coherence (gibberish rate), and
the cumulative norm budget (N5: cumulative ||Deltah||/||h||). The MARGINAL effect of
each added prior is the whole point: it distinguishes stacking from competing.

Everything that touches the model lives under ``main()`` so ``import
run_stacking`` is a no-op (safe for tests / the lead's GPU launcher).

RESULTS SCHEMA (kept in sync with README)
-----------------------------------------
{
  "model_id": str, "primary_layer": int, "orthogonal_layer": int,
  "stack_alpha": float, "compete_add_alpha": float,
  "refusal_vector": {"norm": float, "layer": int, "n_extract": int},
  "single": {"B_refusal_rate": float, "B_gibberish_rate": float},  # B alone, ref
  "rungs": [
    {"key": str, "label": str, "category": str, "expect": str,
     "priors": [str, ...], "n": int,
     "refusal_rate": float, "compliance_rate": float, "gibberish_rate": float,
     "norm_budget": float, "marginal_refusal": float}, ...
  ],
  "decision": {"stack_marginal": float, "compete_marginal": float,
               "overstack_gibberish_delta": float, "verdict": str},
  "examples": [{"prompt": str, "rung1": str, "rung2a": str,
                "rung2b": str, "rung3": str}, ...],
  "plots": {"ladder": "ladder.png"}
}
"""
from __future__ import annotations

import json
import sys

from . import config as C
from .stacking import apply_stack, build_priors, ladder_rungs


# --------------------------------------------------------------------------- #
# Pure helpers (no model) — unit-testable in isolation.
# --------------------------------------------------------------------------- #
def _rates(verdicts: list[str]) -> dict[str, float]:
    """Fraction of REFUSAL / COMPLIANCE / GIBBERISH among a verdict list."""
    n = max(1, len(verdicts))
    return {
        "refusal_rate": verdicts.count("REFUSAL") / n,
        "compliance_rate": verdicts.count("COMPLIANCE") / n,
        "gibberish_rate": verdicts.count("GIBBERISH") / n,
    }


def classify_ladder(rungs: list[dict]) -> dict:
    """Read the stack-vs-compete decision rule off the measured ladder.

    * stack_marginal   = refusal(2a) - refusal(1): gain from the disjoint-site
      prior. Positive ⇒ the sites STACKED.
    * compete_marginal = refusal(2b) - refusal(1): gain from the same-site prior.
      <= 0 (or gibberish-driven) ⇒ they COMPETED.
    * overstack_gibberish_delta = gibberish(3) - gibberish(1): the coherence cost
      of the forbidden all-on hybrid. Positive ⇒ the norm budget was over-spent.
    """
    by = {r["key"]: r for r in rungs}
    r1 = by["rung1"]
    stack_marginal = by["rung2a"]["refusal_rate"] - r1["refusal_rate"]
    compete_marginal = by["rung2b"]["refusal_rate"] - r1["refusal_rate"]
    overstack_gib = by["rung3"]["gibberish_rate"] - r1["gibberish_rate"]

    stacked = stack_marginal > compete_marginal and stack_marginal > 0
    verdict = ("CONFIRMED: disjoint-site prior stacked; same-site prior competed"
               if stacked else
               "INCONCLUSIVE at this scale (1B toy; effects noisy — see caveats)")
    return {
        "stack_marginal": float(stack_marginal),
        "compete_marginal": float(compete_marginal),
        "overstack_gibberish_delta": float(overstack_gib),
        "verdict": verdict,
    }


def _summary_table(results: dict) -> str:
    lines = ["", "=" * 68, "STACKING LADDER SUMMARY", "=" * 68,
             f"model        : {results['model_id']}",
             f"sites        : A,B'@L{results['primary_layer']}  "
             f"B@L{results['orthogonal_layer']}   "
             f"alpha={results['stack_alpha']:.2f}",
             f"refusal vec  : layer={results['refusal_vector']['layer']} "
             f"norm={results['refusal_vector']['norm']:.3f}", "",
             f"  {'rung':<28} {'refusal':>8} {'gibber':>8} {'budget':>8} {'Deltarefus':>8}"]
    for r in results["rungs"]:
        lines.append(f"  {r['label']:<28} {r['refusal_rate']:>8.2f} "
                     f"{r['gibberish_rate']:>8.2f} {r['norm_budget']:>8.3f} "
                     f"{r['marginal_refusal']:>+8.2f}")
    d = results["decision"]
    lines += ["", f"stack marginal (2a-1)   : {d['stack_marginal']:+.2f}",
              f"compete marginal (2b-1) : {d['compete_marginal']:+.2f}",
              f"over-stack gibberish Delta  : {d['overstack_gibberish_delta']:+.2f}",
              f"verdict : {d['verdict']}", "=" * 68, ""]
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Plotting — matplotlib Agg (headless). One figure, two panels.
# --------------------------------------------------------------------------- #
def _plot_ladder(rungs: list[dict], path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = [r["label"].split(":")[0] for r in rungs]  # "1", "2a", "2b", "3"
    cat_color = {"base": "#666", "stack": "#2a7",
                 "compete": "#c93", "overstack": "#c33"}
    colors = [cat_color[r["category"]] for r in rungs]

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11, 4.2))

    # Left: refusal (bars, colored by category) + gibberish (red line).
    refus = [r["refusal_rate"] for r in rungs]
    gib = [r["gibberish_rate"] for r in rungs]
    axL.bar(labels, refus, color=colors, edgecolor="black", linewidth=0.6)
    for i, (rf, r) in enumerate(zip(refus, rungs)):
        axL.text(i, rf + 0.02, f"{rf:.2f}", ha="center", va="bottom", fontsize=9)
    axL.plot(labels, gib, "^--", color="#c33", label="gibberish rate")
    axL.set_ylabel("rate on held-out harmful prompts")
    axL.set_ylim(-0.02, 1.05)
    axL.set_title("Refusal per rung (bars) + gibberish (line)\n"
                  "green=stack  amber=compete  red=over-stack")
    axL.legend(loc="upper left", fontsize=8)
    axL.grid(axis="y", alpha=0.3)

    # Right: cumulative norm budget (N5) per rung — the ceiling that decides it.
    budget = [r["norm_budget"] for r in rungs]
    axR.bar(labels, budget, color=colors, edgecolor="black", linewidth=0.6)
    for i, b in enumerate(budget):
        axR.text(i, b + 0.005, f"{b:.3f}", ha="center", va="bottom", fontsize=9)
    axR.set_ylabel("cumulative  ||Deltah|| / ||h||   (norm budget, N5)")
    axR.set_title("Norm budget spent per rung\n(over-stack spends the most)")
    axR.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Norm budget (N5) — measured directly with two forward passes.
# --------------------------------------------------------------------------- #
def _measure_norm_budget(model, tok, prompt, priors) -> float:
    """Cumulative ||Deltah||/||h|| across the priors' layers for one prompt.

    Baseline forward captures the residual at each prior's layer; a second
    forward with the whole stack active captures the steered residual there. For
    each prior layer we take the mean over positions of ||Deltah||/||h|| and sum
    across the (unique) layers — the leading indicator N5 in CLAUDE.md sec. 3.
    """
    import torch

    from steering_tutorials.hello_world_steering.model_utils import residual_layers
    from .stacking import stack_contexts

    device = next(model.parameters()).device
    layers = residual_layers(model)
    target_idx = sorted({p.layer for p in priors})

    def _capture() -> dict[int, "torch.Tensor"]:
        grabbed: dict[int, torch.Tensor] = {}
        handles = []
        for idx in target_idx:
            def mk(i):
                def hook(_m, _in, out):
                    h = out[0] if isinstance(out, tuple) else out
                    grabbed[i] = h.detach()[0]  # [seq, hidden]
                return hook
            handles.append(layers[idx].register_forward_hook(mk(idx)))
        try:
            ids = tok.apply_chat_template(
                [{"role": "user", "content": prompt}],
                add_generation_prompt=True, return_tensors="pt",
            ).to(device)
            with torch.no_grad():
                model(ids)
        finally:
            for h in handles:
                h.remove()
        return grabbed

    base = _capture()
    special = set(getattr(tok, "all_special_ids", []) or [])
    with stack_contexts(model, priors, special):
        steered = _capture()

    total = 0.0
    for idx in target_idx:
        b, s = base[idx].float(), steered[idx].float()
        ratio = (s - b).norm(dim=-1) / b.norm(dim=-1).clamp_min(1e-6)  # [seq]
        total += float(ratio.mean().item())
    return total


# --------------------------------------------------------------------------- #
# The pipeline — everything below loads / runs the model.
# --------------------------------------------------------------------------- #
def main() -> dict:
    import random

    import numpy as np
    import torch

    from steering_tutorials.hello_world_steering.model_utils import (
        load_model, num_layers,
    )
    from steering_tutorials.hello_world_steering.steer_vector import (
        extract_caa_vector, save_vector,
    )
    from steering_tutorials.hello_world_steering.judge import Judge
    from steering_tutorials.hello_world_steering.data import load_harmful_benign

    random.seed(C.SEED)
    np.random.seed(C.SEED)
    torch.manual_seed(C.SEED)

    # --- Load the model we read (extract) and write (stack) -------------------
    model, tok = load_model(C.MODEL_ID)
    primary = min(C.PRIMARY_LAYER, num_layers(model) - 1)
    orthogonal = min(C.ORTHOGONAL_LAYER, num_layers(model) - 1)

    # --- Data: disjoint extract / eval halves of the harmful class ------------
    data = load_harmful_benign(C.N_PER_CLASS, C.SEED)
    extract_harmful = data["harmful"][:C.N_EXTRACT]
    extract_benign = data["benign"][:C.N_EXTRACT]
    eval_harmful = data["harmful"][C.N_EXTRACT:C.N_EXTRACT + C.N_EVAL]
    print(f"[split] extract {len(extract_harmful)}h/{len(extract_benign)}b   "
          f"eval {len(eval_harmful)}h", file=sys.stderr)

    # --- 1. One refusal direction (diff-of-means) at the PRIMARY layer --------
    vec = extract_caa_vector(model, tok, extract_harmful, extract_benign, primary)
    save_vector(C.VECTOR_PATH, vec)
    refusal_unit = vec["v_unit"]
    print(f"[vector] layer={vec['layer']} n={vec['n']} norm={vec['norm']:.3f}",
          file=sys.stderr)

    # Scale the competing prior B' (literal "add", raw) so that ALONE it is ~the
    # same magnitude as A: raw step = COMPETE_ADD_FRACTION * reference ||h|| at
    # the primary layer (measured on the extract activations). This makes rung 2b
    # a controlled comparison — B' is not weaker than A, it *competes*.
    from steering_tutorials.hello_world_steering.model_utils import last_token_activations
    ref_acts = last_token_activations(model, tok, eval_harmful, primary)
    ref_norm = float(np.linalg.norm(ref_acts, axis=1).mean())
    compete_add_alpha = C.COMPETE_ADD_FRACTION * ref_norm
    print(f"[scale] ref ||h||@L{primary}={ref_norm:.2f} -> B' add alpha="
          f"{compete_add_alpha:.3f}", file=sys.stderr)

    priors = build_priors(refusal_unit, primary, orthogonal,
                          C.STACK_ALPHA, compete_add_alpha)
    rungs_spec = ladder_rungs(priors)
    judge = Judge(model, tok)

    # --- Reference: prior B ALONE (disjoint site) so we can see it carries gain
    b_verdicts = [judge.verdict(p, apply_stack(model, tok, p, [priors["B"]],
                                               C.MAX_NEW_TOKENS))
                  for p in eval_harmful]
    b_rates = _rates(b_verdicts)
    print(f"[single] B@L{orthogonal} alone refusal={b_rates['refusal_rate']:.2f}",
          file=sys.stderr)

    # --- 2. Walk the additive ladder -----------------------------------------
    rung_results: list[dict] = []
    per_prompt_gen: dict[str, list[str]] = {}  # rung_key -> generations (for examples)
    baseline_refusal = None
    for spec in rungs_spec:
        verdicts, gens = [], []
        for i, prompt in enumerate(eval_harmful):
            resp = apply_stack(model, tok, prompt, spec["priors"], C.MAX_NEW_TOKENS)
            gens.append(resp)
            verdicts.append(judge.verdict(prompt, resp))
            if (i + 1) % 4 == 0:
                print(f"[{spec['key']}] {i + 1}/{len(eval_harmful)}", file=sys.stderr)
        rates = _rates(verdicts)
        if baseline_refusal is None:
            baseline_refusal = rates["refusal_rate"]

        # Norm budget: average cumulative ||Deltah||/||h|| over a few prompts.
        budget = float(np.mean([
            _measure_norm_budget(model, tok, p, spec["priors"])
            for p in eval_harmful[:C.N_NORM_BUDGET_PROMPTS]
        ]))

        rung_results.append({
            "key": spec["key"], "label": spec["label"],
            "category": spec["category"], "expect": spec["expect"],
            "priors": [p.name for p in spec["priors"]],
            "n": len(eval_harmful), **rates,
            "norm_budget": budget,
            "marginal_refusal": float(rates["refusal_rate"] - baseline_refusal),
        })
        per_prompt_gen[spec["key"]] = gens
        print(f"[{spec['key']}] refusal={rates['refusal_rate']:.2f} "
              f"gibber={rates['gibberish_rate']:.2f} budget={budget:.3f}",
              file=sys.stderr)

    decision = classify_ladder(rung_results)

    # --- Side-by-side examples: same prompt across all four rungs -------------
    examples = []
    for i, prompt in enumerate(eval_harmful[:6]):
        examples.append({
            "prompt": prompt,
            "rung1": per_prompt_gen["rung1"][i],
            "rung2a": per_prompt_gen["rung2a"][i],
            "rung2b": per_prompt_gen["rung2b"][i],
            "rung3": per_prompt_gen["rung3"][i],
        })

    results = {
        "model_id": C.MODEL_ID,
        "primary_layer": int(primary),
        "orthogonal_layer": int(orthogonal),
        "stack_alpha": float(C.STACK_ALPHA),
        "compete_add_alpha": float(compete_add_alpha),
        "refusal_vector": {"norm": float(vec["norm"]), "layer": int(vec["layer"]),
                           "n_extract": int(vec["n"])},
        "single": {"B_refusal_rate": b_rates["refusal_rate"],
                   "B_gibberish_rate": b_rates["gibberish_rate"]},
        "rungs": rung_results,
        "decision": decision,
        "examples": examples,
        "plots": {"ladder": C.LADDER_PNG.name},
    }

    # --- 3. Persist + plot + print -------------------------------------------
    C.RESULTS_PATH.write_text(json.dumps(results, indent=2), encoding="utf-8")
    _plot_ladder(rung_results, C.LADDER_PNG)
    print(f"[save] {C.RESULTS_PATH}", file=sys.stderr)
    print(f"[save] {C.LADDER_PNG}", file=sys.stderr)
    print(_summary_table(results))
    return results


if __name__ == "__main__":
    main()
