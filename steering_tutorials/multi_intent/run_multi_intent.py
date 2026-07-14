"""run_multi_intent.py — the K=1..N compositional-steering experiment.

Spine of lesson 9. It builds K concept vectors once, then walks a ladder
K = 1, 2, ..., N adding ONE concept at a time, and at each rung measures — for
BOTH the naive raw-sum arm and the Gram-Schmidt orthogonalized arm — three things:

  1. STEERING SUCCESS: on each active concept's held-out prompts, does the
     steered model now REFUSE (vs. the abliterated model's baseline COMPLIANCE)?
     Averaged over the K active concepts = "did stacking still steer each one?".

  2. CROSS-TALK: on an INACTIVE concept's prompts (one we did NOT add to the
     mix), does the mixture change its outcome anyway? Steering concept A should
     not move concept B. We report the inactive-concept refusal rate; lower is
     cleaner (less accidental spillover).

  3. NORM BUDGET vs COHERENCE: the injected displacement sqrt(Σαᵢ²) climbs with
     K, and we track the GIBBERISH rate alongside it. The hypothesis to TEST (not
     assume): orthogonalization spends the budget more efficiently, so success
     stays higher and gibberish rises later than the raw-sum arm.

The sub-linear-interference hypothesis is stated honestly and MEASURED: we do not
assume orthogonalization wins — we plot both arms and let the numbers speak.

Everything that touches the model lives under ``main()`` so ``import
run_multi_intent`` is a no-op (safe for tests / import-checks). The team lead
runs ``main()`` on the GPU.

RESULTS SCHEMA (kept in sync with README)
-----------------------------------------
{
  "model_id": str, "steer_layer": int, "concepts": [str, ...],
  "per_concept_alpha": float,
  "cosine_matrix": {"labels": [...], "matrix": [[...]]},   # raw-dir overlap
  "ladder": [
      {"k": int, "active": [str,...], "inactive": [str,...],
       "raw":  {"success": float, "crosstalk": float, "gibberish": float,
                "budget": float},
       "ortho":{"success": float, "crosstalk": float, "gibberish": float,
                "budget": float}}, ...
  ],
  "examples": [ {"concept": str, "k": int, "arm": "raw"|"ortho",
                 "prompt": str, "response": str, "verdict": str}, ... ],
  "plots": {"success_vs_k": "success_vs_k.png"}
}
"""
from __future__ import annotations

import json
import sys

import numpy as np

from . import config as C
from .multi_intent import (
    apply_multi,
    cosine_matrix,
    extract_concept_vectors,
    gram_schmidt,
    norm_budget,
)


# --------------------------------------------------------------------------- #
# Pure helpers (no model) — safe to unit-test in isolation.
# --------------------------------------------------------------------------- #
def _refusal_rate(verdicts: list[str]) -> float:
    """Fraction of REFUSAL verdicts (the steering-success signal)."""
    n = max(1, len(verdicts))
    return verdicts.count("REFUSAL") / n


def _gibberish_rate(verdicts: list[str]) -> float:
    n = max(1, len(verdicts))
    return verdicts.count("GIBBERISH") / n


def summarize_ladder(ladder: list[dict]) -> str:
    """One-line-per-rung text table of the K-ladder (used by the console report)."""
    lines = [f"{'K':>2}  {'arm':<6} {'success':>8} {'crosstalk':>10} "
             f"{'gibber':>7} {'budget':>7}"]
    for rung in ladder:
        for arm in ("raw", "ortho"):
            a = rung[arm]
            lines.append(f"{rung['k']:>2}  {arm:<6} {a['success']:>8.2f} "
                         f"{a['crosstalk']:>10.2f} {a['gibberish']:>7.2f} "
                         f"{a['budget']:>7.3f}")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Plot (model-free; takes the finished ladder). Guarded import so a headless
# box without matplotlib still completes the numeric run.
# --------------------------------------------------------------------------- #
def _plot_ladder(ladder: list[dict], path) -> bool:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - optional dep
        print(f"[plot] skipped ({exc})", file=sys.stderr)
        return False

    ks = [r["k"] for r in ladder]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))

    for arm, color in (("raw", "#c0392b"), ("ortho", "#2471a3")):
        ax1.plot(ks, [r[arm]["success"] for r in ladder], "o-", color=color,
                 label=f"{arm} success")
        ax1.plot(ks, [r[arm]["gibberish"] for r in ladder], "s--", color=color,
                 alpha=0.6, label=f"{arm} gibberish")
    ax1.set_xlabel("K (concepts stacked)")
    ax1.set_ylabel("rate")
    ax1.set_title("Steering success & gibberish vs K")
    ax1.set_ylim(-0.03, 1.03)
    ax1.legend(fontsize=8)
    ax1.grid(alpha=0.3)

    for arm, color in (("raw", "#c0392b"), ("ortho", "#2471a3")):
        ax2.plot(ks, [r[arm]["budget"] for r in ladder], "o-", color=color,
                 label=f"{arm} budget")
        ax2.plot(ks, [r[arm]["crosstalk"] for r in ladder], "^--", color=color,
                 alpha=0.6, label=f"{arm} cross-talk")
    ax2.set_xlabel("K (concepts stacked)")
    ax2.set_ylabel("norm budget  /  cross-talk rate")
    ax2.set_title("Norm budget & cross-talk vs K")
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"[plot] wrote {path}", file=sys.stderr)
    return True


# --------------------------------------------------------------------------- #
# The experiment (model-touching). Under main() so importing this file is safe.
# --------------------------------------------------------------------------- #
def main() -> None:
    from steering_tutorials.hello_world_steering.model_utils import (
        generate,
        load_model,
    )
    from steering_tutorials.hello_world_steering.judge import Judge

    from .data import load_multi_intent

    print("[run] loading model + data ...", file=sys.stderr)
    model, tok = load_model(C.MODEL_ID)
    judge = Judge(model, tok)

    data = load_multi_intent(
        C.CONCEPTS,
        n_per_concept=C.N_PER_CONCEPT,
        n_eval_per_concept=C.N_EVAL_PER_CONCEPT,
        n_benign_baseline=C.N_BENIGN_BASELINE,
        seed=C.SEED,
    )
    concepts = list(data["concepts"])            # preserves config order
    baseline = data["baseline"]

    # 1) EXTRACT one raw diff-of-means direction per concept (shared baseline).
    concept_prompts = {name: data["concepts"][name]["extract"] for name in concepts}
    raw_vectors = extract_concept_vectors(
        model, tok, concept_prompts, C.STEER_LAYER, baseline_prompts=baseline
    )
    raw_list = [raw_vectors[name] for name in concepts]

    # Diagnostic: how much do the raw directions overlap? (drives interference)
    cmat = cosine_matrix(raw_list)
    print("[run] raw-direction cosine matrix:\n" + np.array2string(
        cmat, precision=2, suppress_small=True), file=sys.stderr)

    examples: list[dict] = []

    def eval_concept(vecs: list[np.ndarray], alphas: list[float],
                     concept_name: str, k: int, arm: str) -> list[str]:
        """Steer along ``vecs`` and judge one concept's held-out eval prompts."""
        verdicts = []
        for p in data["concepts"][concept_name]["eval"]:
            resp = apply_multi(model, tok, p, vecs, alphas, C.STEER_LAYER,
                               max_new_tokens=C.MAX_NEW_TOKENS)
            v = judge.verdict(p, resp)
            verdicts.append(v)
            if len(examples) < 40:
                examples.append({"concept": concept_name, "k": k, "arm": arm,
                                 "prompt": p, "response": resp, "verdict": v})
        return verdicts

    # 2) LADDER K=1..N. At rung k the ACTIVE concepts are the first k; the rest
    #    are INACTIVE (used to probe cross-talk).
    ladder: list[dict] = []
    for k in range(1, len(concepts) + 1):
        active = concepts[:k]
        inactive = concepts[k:]
        alphas = [C.PER_CONCEPT_ALPHA] * k
        raw_k = raw_list[:k]
        ortho_k = gram_schmidt(raw_k)            # orthonormal axes for the k active

        rung = {"k": k, "active": active, "inactive": inactive}
        for arm, vecs in (("raw", raw_k), ("ortho", ortho_k)):
            # success: mean refusal over the active concepts' eval prompts.
            succ, gib = [], []
            for name in active:
                vs = eval_concept(vecs, alphas, name, k, arm)
                succ.append(_refusal_rate(vs))
                gib.append(_gibberish_rate(vs))
            # cross-talk: refusal on ONE inactive concept (if any remain).
            if inactive:
                xt = eval_concept(vecs, alphas, inactive[0], k, arm)
                crosstalk = _refusal_rate(xt)
            else:
                crosstalk = float("nan")
            rung[arm] = {
                "success": float(np.mean(succ)),
                "gibberish": float(np.mean(gib)),
                "crosstalk": crosstalk,
                "budget": norm_budget(vecs, alphas),
            }
        ladder.append(rung)
        print(f"[run] K={k}  raw.success={rung['raw']['success']:.2f} "
              f"ortho.success={rung['ortho']['success']:.2f} "
              f"raw.gib={rung['raw']['gibberish']:.2f} "
              f"ortho.gib={rung['ortho']['gibberish']:.2f}", file=sys.stderr)

    # 3) REPORT.
    _plot_ladder(ladder, C.LADDER_PNG)
    results = {
        "model_id": C.MODEL_ID,
        "steer_layer": C.STEER_LAYER,
        "concepts": concepts,
        "per_concept_alpha": C.PER_CONCEPT_ALPHA,
        "cosine_matrix": {"labels": concepts, "matrix": cmat.tolist()},
        "ladder": ladder,
        "examples": examples,
        "plots": {"success_vs_k": C.LADDER_PNG.name},
    }
    C.RESULTS_PATH.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"[run] wrote {C.RESULTS_PATH}", file=sys.stderr)
    print("\n" + summarize_ladder(ladder))


# --------------------------------------------------------------------------- #
# CPU unit for the pure helpers — NO model. Run: python -m ...run_multi_intent
# --------------------------------------------------------------------------- #
def _self_test() -> None:
    assert _refusal_rate(["REFUSAL", "COMPLIANCE", "REFUSAL", "GIBBERISH"]) == 0.5
    assert _gibberish_rate(["GIBBERISH", "GIBBERISH", "REFUSAL", "COMPLIANCE"]) == 0.5
    fake = [{"k": 1, "raw": {"success": 1.0, "crosstalk": 0.0, "gibberish": 0.0,
                             "budget": 0.06},
             "ortho": {"success": 1.0, "crosstalk": 0.0, "gibberish": 0.0,
                       "budget": 0.06}}]
    table = summarize_ladder(fake)
    assert "success" in table and "budget" in table
    print("[self-test] OK - rate helpers + ladder summary behave as expected.")


if __name__ == "__main__":
    _self_test()
