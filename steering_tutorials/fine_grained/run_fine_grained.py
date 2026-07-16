"""run_fine_grained.py — the frontier experiment: sparse vs dense steering.

We build ONE dense refusal vector (lesson-2 CAA diff-of-means) and then, at a
single matched strength ``STEER_ALPHA``, sweep how MANY of its coordinates we
keep. At each sparsity level we measure, on disjoint held-out prompts:

    refusal_rate  = fraction of HARMFUL prompts the model REFUSES   (want HIGH:
                    this is the target behavior the vector installs)
    over_refusal  = fraction of BENIGN  prompts the model REFUSES   (want LOW:
                    collateral damage — steering leaking onto safe requests)
    gibberish     = fraction of ALL     prompts judged GIBBERISH    (want LOW:
                    the coherence cost of the edit)

The claim we test (inspired by AUSteer, arXiv:2602.04428): a SPARSE edit (keep ~5-10% of
coordinates) matches the DENSE refusal rate while lowering over-refusal and
gibberish — "steering less, achieving more." The dense point is ``keep_frac=1.0``;
if the claim holds, some sparse row matches dense refusal with strictly lower
collateral.

The judge is lesson-2's ``Judge``. Set ``STEER_JUDGE_MODEL=Qwen/Qwen2.5-3B-Instruct``
to grade with an OFF-FAMILY model (recommended — a 1B model grading its own
steered output misreads hedged compliance as refusal). Everything below loads /
runs the model; it is all inside ``main()`` so importing this file is model-free.

Run (from the repo root):

    STEER_JUDGE_MODEL=Qwen/Qwen2.5-3B-Instruct \
        python -m steering_tutorials.fine_grained.run_fine_grained
"""
from __future__ import annotations

import json
import sys

from . import config as C
from .sparse import support_size


# --------------------------------------------------------------------------- #
# Pure helpers (no model) — safe to reason about in isolation.
# --------------------------------------------------------------------------- #
def _rate(verdicts: list[str], target: str) -> float:
    """Fraction of ``verdicts`` equal to ``target`` (e.g. REFUSAL / COMPLIANCE)."""
    return sum(v == target for v in verdicts) / max(1, len(verdicts))


def choose_best_sparse(rows: list[dict], dense: dict,
                       refusal_tol: float, collateral_slack: float) -> dict | None:
    """The sparsest row that matches dense refusal without worse collateral.

    A sparse row (keep_frac < 1) qualifies iff:
      * its refusal_rate >= dense.refusal_rate - refusal_tol   (matched behavior)
      * its over_refusal <= dense.over_refusal + collateral_slack
      * its gibberish   <= dense.gibberish   + collateral_slack
    Among qualifiers we return the one with the SMALLEST keep_frac (the strongest
    "steering less" win). ``None`` means no sparse level matched dense behavior
    within tolerance while holding collateral flat — the honest null result.
    """
    qualifiers = [
        r for r in rows
        if r["keep_frac"] < 1.0
        and r["refusal_rate"] >= dense["refusal_rate"] - refusal_tol
        and r["over_refusal"] <= dense["over_refusal"] + collateral_slack
        and r["gibberish"] <= dense["gibberish"] + collateral_slack
    ]
    if not qualifiers:
        return None
    return min(qualifiers, key=lambda r: r["keep_frac"])


# --------------------------------------------------------------------------- #
# Plotting — matplotlib, Agg backend (headless).
# --------------------------------------------------------------------------- #
def _plot_frontier(rows: list[dict], path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # x-axis = fraction of coordinates kept (log scale: 2% .. 100%).
    fracs = [r["keep_frac"] for r in rows]
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.plot(fracs, [r["refusal_rate"] for r in rows], "o-",
            label="refusal (harmful) — want HIGH", color="#2a7")
    ax.plot(fracs, [r["over_refusal"] for r in rows], "s-",
            label="over-refusal (benign) — want LOW", color="#c93")
    ax.plot(fracs, [r["gibberish"] for r in rows], "^-",
            label="gibberish (all) — want LOW", color="#c33")
    ax.set_xscale("log")
    ax.set_xlabel("fraction of steering-vector coordinates kept  (log; 1.0 = dense)")
    ax.set_ylabel("rate")
    ax.set_title("Fine-grained steering: steering less, achieving more?\n"
                 f"(matched strength alpha={C.STEER_ALPHA}, layer {C.LAYER})")
    ax.set_ylim(-0.02, 1.02)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def _summary_table(results: dict) -> str:
    lines = ["", "=" * 72, "FINE-GRAINED (SPARSE) STEERING SUMMARY", "=" * 72,
             f"model     : {results['model_id']}",
             f"judge     : {results['judge_id']}",
             f"layer     : {results['layer']}   alpha : {results['alpha']}   "
             f"hidden : {results['hidden']}",
             f"baseline (unsteered) refusal={results['baseline']['refusal_rate']:.2f}  "
             f"over_refusal={results['baseline']['over_refusal']:.2f}",
             "",
             f"  {'keep':>6} {'coords':>7} {'refusal':>8} {'over_ref':>9} "
             f"{'gibber':>7}"]
    for r in results["sweep"]:
        lines.append(f"  {r['keep_frac']:>6.2f} {r['coords']:>7d} "
                     f"{r['refusal_rate']:>8.2f} {r['over_refusal']:>9.2f} "
                     f"{r['gibberish']:>7.2f}")
    dense = results["dense"]
    best = results["best_sparse"]
    lines += ["", f"dense (keep=1.00): refusal={dense['refusal_rate']:.2f}  "
                  f"over_refusal={dense['over_refusal']:.2f}  "
                  f"gibberish={dense['gibberish']:.2f}"]
    if best is not None:
        lines.append(
            f"best sparse WIN  : keep={best['keep_frac']:.2f} "
            f"({best['coords']} coords)  refusal={best['refusal_rate']:.2f} "
            f"(dense {dense['refusal_rate']:.2f})  "
            f"over_refusal {dense['over_refusal']:.2f} -> {best['over_refusal']:.2f}  "
            f"gibberish {dense['gibberish']:.2f} -> {best['gibberish']:.2f}")
        lines.append("  => sparse matched dense refusal with <= collateral "
                     "(supports 'steering less, achieving more')")
    else:
        lines.append("best sparse WIN  : NONE — no sparse level matched dense refusal "
                     "within tolerance while holding over-refusal/gibberish flat")
        lines.append("  => dense was not beaten on this screening slice (honest null)")
    lines += ["=" * 72, ""]
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# The pipeline — everything below loads / runs the model.
# --------------------------------------------------------------------------- #
def _measure(model, tok, judge, vector, keep_frac, alpha,
             eval_harmful, eval_benign) -> dict:
    """Generate + judge one (keep_frac, alpha) point; return its metric row.

    ``keep_frac`` selects the sparsity of the injected vector via
    ``SparseSteeringContext`` (1.0 == dense). ``alpha == 0`` means no steering at
    all (the unsteered baseline), in which case the vector/sparsity are ignored.
    """
    from steering_tutorials.hello_world_steering.model_utils import generate
    from steering_tutorials.hello_world_steering.judge import is_gibberish
    from .sparse import sparsify

    steering = alpha != 0.0 and vector is not None
    # Pre-sparsify once so every generation in this row uses the same vector; the
    # generate() helper applies it densely via its own SteeringContext.
    v_used = sparsify(vector, keep_frac) if steering else None

    h_verdicts: list[str] = []
    gibber_flags: list[bool] = []
    for i, prompt in enumerate(eval_harmful):
        resp = generate(
            model, tok, prompt, max_new_tokens=C.MAX_NEW_TOKENS,
            vector=v_used, layer=C.LAYER, alpha=(alpha if steering else 0.0),
            operation="relative_add",
        )
        h_verdicts.append(judge.verdict(prompt, resp))
        gibber_flags.append(is_gibberish(resp))
        if (i + 1) % 10 == 0:
            print(f"[keep={keep_frac:.2f} harmful] {i + 1}/{len(eval_harmful)}",
                  file=sys.stderr)

    b_verdicts: list[str] = []
    for i, prompt in enumerate(eval_benign):
        resp = generate(
            model, tok, prompt, max_new_tokens=C.MAX_NEW_TOKENS,
            vector=v_used, layer=C.LAYER, alpha=(alpha if steering else 0.0),
            operation="relative_add",
        )
        b_verdicts.append(judge.verdict(prompt, resp))
        gibber_flags.append(is_gibberish(resp))
        if (i + 1) % 10 == 0:
            print(f"[keep={keep_frac:.2f} benign] {i + 1}/{len(eval_benign)}",
                  file=sys.stderr)

    return {
        "keep_frac": float(keep_frac),
        "coords": support_size(keep_frac, len(vector)) if vector is not None else 0,
        "alpha": float(alpha if steering else 0.0),
        "refusal_rate": _rate(h_verdicts, "REFUSAL"),
        "over_refusal": _rate(b_verdicts, "REFUSAL"),
        "gibberish": sum(gibber_flags) / max(1, len(gibber_flags)),
        "n_harmful": len(eval_harmful),
        "n_benign": len(eval_benign),
    }


def main() -> dict:
    import os
    import random

    import numpy as np
    import torch

    # Lesson-2 plumbing + judge + the shared dataset foundation. Imported inside
    # main() so a bare ``import run_fine_grained`` never loads torch or a model.
    from steering_tutorials.hello_world_steering.model_utils import (
        load_model, hidden_size,
    )
    from steering_tutorials.hello_world_steering.judge import Judge
    from steering_tutorials.hello_world_steering.steer_vector import (
        extract_caa_vector, save_vector,
    )
    from steering_tutorials.common.data import load_harmful_benign

    random.seed(C.SEED)
    np.random.seed(C.SEED)
    torch.manual_seed(C.SEED)

    # --- Load model + judge ---------------------------------------------------
    model, tok = load_model(C.MODEL_ID)
    judge = Judge(model, tok)
    hidden = hidden_size(model)

    # --- Data: disjoint extract / eval halves --------------------------------
    data = load_harmful_benign(C.N_PER_CLASS, C.SEED)
    ext_harmful = data["harmful"][:C.N_EXTRACT]
    ext_benign = data["benign"][:C.N_EXTRACT]

    # Optional caps for a RAM/time-constrained host (defaults = full config):
    n_eval = int(os.environ.get("FG_N_EVAL", "0") or C.N_EVAL)
    sparsity = ([float(x) for x in os.environ["FG_SPARSITY"].split(",")]
                if os.environ.get("FG_SPARSITY") else C.SPARSITY_LEVELS)
    alpha = float(os.environ.get("FG_ALPHA", "") or C.STEER_ALPHA)
    eval_harmful = data["harmful"][C.N_EXTRACT:C.N_EXTRACT + n_eval]
    eval_benign = data["benign"][C.N_EXTRACT:C.N_EXTRACT + n_eval]
    print(f"[eval] {len(eval_harmful)} harmful / {len(eval_benign)} benign held-out; "
          f"alpha={alpha}; sparsity={sparsity}", file=sys.stderr)

    # --- Build the ONE dense refusal vector (lesson-2 CAA diff-of-means) ------
    vec = extract_caa_vector(model, tok, ext_harmful, ext_benign, C.LAYER)
    dense_v = vec["v_raw"]                       # [hidden] float32
    save_vector(C.VECTOR_PATH, vec)
    print(f"[vector] dense CAA at layer {C.LAYER}: ||v||={vec['norm']:.3f} "
          f"(n={vec['n']}/side) -> {C.VECTOR_PATH}", file=sys.stderr)

    # --- Unsteered baseline (alpha=0) ----------------------------------------
    baseline = _measure(model, tok, judge, None, 1.0, 0.0, eval_harmful, eval_benign)
    print(f"[baseline] refusal={baseline['refusal_rate']:.2f} "
          f"over_refusal={baseline['over_refusal']:.2f} "
          f"gibberish={baseline['gibberish']:.2f}", file=sys.stderr)

    # --- Sweep sparsity at matched strength ----------------------------------
    rows: list[dict] = []
    for keep_frac in sparsity:
        row = _measure(model, tok, judge, dense_v, keep_frac, alpha,
                       eval_harmful, eval_benign)
        rows.append(row)
        print(f"[keep={keep_frac:.2f}] coords={row['coords']} "
              f"refusal={row['refusal_rate']:.2f} "
              f"over_refusal={row['over_refusal']:.2f} "
              f"gibberish={row['gibberish']:.2f}", file=sys.stderr)

    # Dense reference row (keep_frac closest to 1.0 that we actually ran).
    dense_row = next((r for r in rows if r["keep_frac"] >= 1.0), rows[0])
    best = choose_best_sparse(rows, dense_row, C.REFUSAL_MATCH_TOL, C.COLLATERAL_SLACK)

    results = {
        "model_id": C.MODEL_ID,
        "judge_id": judge.judge_id,
        "layer": C.LAYER,
        "alpha": alpha,
        "hidden": hidden,
        "vector_norm": vec["norm"],
        "sparsity_levels": sparsity,
        "baseline": baseline,
        "sweep": rows,
        "dense": dense_row,
        "best_sparse": best,
        "thresholds": {
            "refusal_match_tol": C.REFUSAL_MATCH_TOL,
            "collateral_slack": C.COLLATERAL_SLACK,
        },
        "plots": {"frontier": C.FRONTIER_PNG.name},
    }

    C.RESULTS_PATH.write_text(json.dumps(results, indent=2), encoding="utf-8")
    _plot_frontier(rows, C.FRONTIER_PNG)
    print(f"[save] {C.RESULTS_PATH}", file=sys.stderr)
    print(f"[save] {C.FRONTIER_PNG}", file=sys.stderr)
    print(_summary_table(results))
    return results


if __name__ == "__main__":
    main()
