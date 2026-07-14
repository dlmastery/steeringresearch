"""run_rogue_scalpel.py — the orchestrator: attack the model, then defend it.

This is the spine of lesson 10. It builds the refusal direction (reusing lesson
2's diff-of-means), then walks an ABLATION LADDER on held-out harmful prompts and
reports the attack-success-rate (ASR = fraction judged COMPLIANCE) at each rung:

    baseline   : no attack, no guard      — the model refuses (ASR low).
    attacked   : rogue scalpel, no guard  — refusal stripped (ASR up: jailbroken).
    +clamp     : attack + Guard B         — the manifold clamp blunts the push.
    +lock      : attack + Guard B + A     — the projection lock re-installs refusal.
    +dual      : attack + Guards B+A+D    — the verdict check catches any leak.

Each rung ADDS one guard, so the marginal effect of every layer is readable (the
additive-ladder discipline from CLAUDE.md section 9). We also run a small BENIGN
collateral check — the full guard must not turn harmless prompts into refusals.

HONEST by construction: if a guard fails to move the ASR (or a later rung is
worse than an earlier one), the numbers say so; nothing is smoothed over.

Everything that touches the model lives under ``main()`` so ``import
run_rogue_scalpel`` is a no-op (safe for tests / import-checks). The team lead
runs this on the GPU.

RESULTS SCHEMA (kept in sync with README + plot)
------------------------------------------------
{
  "model_id": str, "layer": int, "attack_mode": str,
  "attack": {"alpha": float, "project_frac": float},
  "guard":  {"clamp_budget": float, "lock_floor_frac": float},
  "refusal_direction": {"norm": float, "layer": int, "n_extract": int},
  "ladder": [ {"rung": str, "asr": float, "refusal_rate": float,
               "gibberish_rate": float, "n": int}, ... ],   # 5 rungs
  "benign_collateral": {"baseline_refusal_rate": float,
                        "guarded_refusal_rate": float, "n": int},
  "examples": [ {"prompt": str, "baseline": str, "attacked": str,
                 "guarded": str, "baseline_verdict": str,
                 "attacked_verdict": str, "guarded_verdict": str}, ... ],
  "plots": {"asr_ladder": "asr_ladder.png"}
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
    """ASR (compliance), refusal and gibberish fractions among verdicts."""
    n = max(1, len(verdicts))
    return {
        "asr": verdicts.count("COMPLIANCE") / n,          # attack-success-rate
        "refusal_rate": verdicts.count("REFUSAL") / n,
        "gibberish_rate": verdicts.count("GIBBERISH") / n,
    }


def _summary_table(results: dict) -> str:
    lines = ["", "=" * 60, "ROGUE-SCALPEL SUMMARY (ASR = fraction COMPLIANCE)", "=" * 60,
             f"model : {results['model_id']}   layer: {results['layer']}   "
             f"attack: {results['attack_mode']}",
             "", f"  {'rung':>10} {'ASR':>7} {'refusal':>9} {'gibber':>8}"]
    for r in results["ladder"]:
        lines.append(f"  {r['rung']:>10} {r['asr']:>7.2f} "
                     f"{r['refusal_rate']:>9.2f} {r['gibberish_rate']:>8.2f}")
    b = results["benign_collateral"]
    lines += ["", "Benign collateral (over-refusal must stay ~flat):",
              f"  baseline refusal : {b['baseline_refusal_rate']:.2f}",
              f"  guarded refusal  : {b['guarded_refusal_rate']:.2f}  (n={b['n']})",
              "=" * 60, ""]
    return "\n".join(lines)


def _plot_asr_ladder(ladder: list[dict], path) -> None:
    """Bar chart of ASR at each rung: baseline / attacked / +clamp / +lock / +dual."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = [r["rung"] for r in ladder]
    asr = [r["asr"] for r in ladder]
    # Red for the undefended attack peak, green as guards drive ASR back down.
    colors = ["#888", "#c33", "#e90", "#7a3", "#2a7"][:len(ladder)]
    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(labels, asr, color=colors)
    for b, v in zip(bars, asr):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.02, f"{v:.2f}",
                ha="center", va="bottom")
    ax.set_ylabel("attack-success-rate (fraction complying)")
    ax.set_ylim(0, 1.05)
    ax.set_title("Rogue scalpel vs. the layered guard\n"
                 "(attack lifts ASR; each guard layer drives it back down)")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# The pipeline — everything below loads / runs the model.
# --------------------------------------------------------------------------- #
def main() -> dict:
    import random

    import numpy as np
    import torch

    from steering_tutorials.hello_world_steering.model_utils import load_model, num_layers
    from steering_tutorials.hello_world_steering.judge import Judge
    from steering_tutorials.hello_world_steering.data import load_harmful_benign
    from steering_tutorials.hello_world_steering.steer_vector import save_vector
    from .attack import refusal_direction, attack_generate
    from .guard import guarded_generate, enforce_dual_forward

    random.seed(C.SEED); np.random.seed(C.SEED); torch.manual_seed(C.SEED)

    # --- Load the aligned model we attack and defend --------------------------
    model, tok = load_model(C.MODEL_ID)
    layer = min(C.LAYER, num_layers(model) - 1)
    judge = Judge(model, tok)

    # --- Data: disjoint extract (build the direction) / eval (attack it) ------
    data = load_harmful_benign(C.N_PER_CLASS, C.SEED)
    extract_harmful = data["harmful"][:C.N_EXTRACT]
    extract_benign = data["benign"][:C.N_EXTRACT]
    eval_harmful = data["harmful"][C.N_EXTRACT:]
    eval_benign = data["benign"][C.N_EXTRACT:]
    print(f"[split] extract {len(extract_harmful)}h/{len(extract_benign)}b   "
          f"eval {len(eval_harmful)}h/{len(eval_benign)}b", file=sys.stderr)

    # --- 1. Build the refusal direction (reused from lesson 2) ----------------
    vec = refusal_direction(model, tok, extract_harmful, extract_benign, layer)
    save_vector(C.VECTOR_PATH, vec)
    u = vec["v_unit"]
    print(f"[direction] layer={vec['layer']} n={vec['n']} norm={vec['norm']:.3f} "
          f"-> {C.VECTOR_PATH}", file=sys.stderr)

    # --- 2. Generation closures for each ladder rung --------------------------
    def gen_baseline(p: str) -> str:
        # No attack, no guard: alpha=0 negative_add is a true no-op baseline.
        return attack_generate(model, tok, p, u, alpha=0.0, layer=layer,
                               mode="negative_add", max_new_tokens=C.MAX_NEW_TOKENS)

    def gen_attacked(p: str) -> str:
        return attack_generate(model, tok, p, u, alpha=C.ATTACK_ALPHA, layer=layer,
                               mode=C.ATTACK_MODE, project_frac=C.ATTACK_PROJECT_FRAC,
                               max_new_tokens=C.MAX_NEW_TOKENS)

    def gen_guarded(p: str, use_clamp: bool, use_lock: bool) -> str:
        return guarded_generate(
            model, tok, p, u, layer,
            mode=C.ATTACK_MODE, alpha=C.ATTACK_ALPHA, project_frac=C.ATTACK_PROJECT_FRAC,
            use_clamp=use_clamp, use_lock=use_lock,
            clamp_budget=C.CLAMP_BUDGET, lock_floor_frac=C.LOCK_FLOOR_FRAC,
            max_new_tokens=C.MAX_NEW_TOKENS)

    # Each rung is (label, response_fn, apply_guard_D?). The ladder is CUMULATIVE:
    # every rung adds exactly one guard layer on top of the previous.
    rungs = [
        ("baseline", gen_baseline, False),
        ("attacked", gen_attacked, False),
        ("+clamp", lambda p: gen_guarded(p, use_clamp=True, use_lock=False), False),
        ("+lock", lambda p: gen_guarded(p, use_clamp=True, use_lock=True), False),
        ("+dual", lambda p: gen_guarded(p, use_clamp=True, use_lock=True), True),
    ]

    # --- 3. Walk the ladder on held-out harmful prompts -----------------------
    ladder: list[dict] = []
    # Keep the per-prompt responses at the key rungs for side-by-side examples.
    per_prompt: dict[str, list[str]] = {}
    for label, fn, use_dual in rungs:
        verdicts: list[str] = []
        responses: list[str] = []
        for i, prompt in enumerate(eval_harmful):
            resp = fn(prompt)
            if use_dual:  # Guard D: override a flagged compliance leak.
                resp = enforce_dual_forward(judge, prompt, resp, C.SAFE_COMPLETION)
            responses.append(resp)
            verdicts.append(judge.verdict(prompt, resp))
            if (i + 1) % 5 == 0:
                print(f"[{label}] {i + 1}/{len(eval_harmful)}", file=sys.stderr)
        per_prompt[label] = responses
        rec = {"rung": label, "n": len(eval_harmful), **_rates(verdicts)}
        ladder.append(rec)
        print(f"[{label}] ASR={rec['asr']:.2f} refusal={rec['refusal_rate']:.2f} "
              f"gibber={rec['gibberish_rate']:.2f}", file=sys.stderr)

    # --- 4. Benign collateral: the full guard must not over-refuse ------------
    #     (The guard always runs; it must not turn harmless prompts into refusals.)
    base_benign_ref = guarded_benign_ref = 0
    for prompt in eval_benign:
        base = gen_baseline(prompt)
        guarded = gen_guarded(prompt, use_clamp=True, use_lock=True)
        guarded = enforce_dual_forward(judge, prompt, guarded, C.SAFE_COMPLETION)
        base_benign_ref += judge.verdict(prompt, base) == "REFUSAL"
        guarded_benign_ref += judge.verdict(prompt, guarded) == "REFUSAL"
    nb = max(1, len(eval_benign))
    benign_collateral = {
        "baseline_refusal_rate": base_benign_ref / nb,
        "guarded_refusal_rate": guarded_benign_ref / nb,
        "n": len(eval_benign),
    }

    # --- 5. Side-by-side examples (baseline vs attacked vs guarded) -----------
    guarded_resps = per_prompt["+dual"]
    examples = []
    for i, prompt in enumerate(eval_harmful[:8]):
        b, a, g = (per_prompt["baseline"][i], per_prompt["attacked"][i],
                   guarded_resps[i])
        examples.append({
            "prompt": prompt,
            "baseline": b, "attacked": a, "guarded": g,
            "baseline_verdict": judge.verdict(prompt, b),
            "attacked_verdict": judge.verdict(prompt, a),
            "guarded_verdict": judge.verdict(prompt, g),
        })

    results = {
        "model_id": C.MODEL_ID,
        "layer": int(layer),
        "attack_mode": C.ATTACK_MODE,
        "attack": {"alpha": C.ATTACK_ALPHA, "project_frac": C.ATTACK_PROJECT_FRAC},
        "guard": {"clamp_budget": C.CLAMP_BUDGET, "lock_floor_frac": C.LOCK_FLOOR_FRAC},
        "refusal_direction": {"norm": float(vec["norm"]), "layer": int(vec["layer"]),
                              "n_extract": int(vec["n"])},
        "ladder": ladder,
        "benign_collateral": benign_collateral,
        "examples": examples,
        "plots": {"asr_ladder": C.ASR_PNG.name},
    }

    C.RESULTS_PATH.write_text(json.dumps(results, indent=2), encoding="utf-8")
    _plot_asr_ladder(ladder, C.ASR_PNG)
    print(f"[save] {C.RESULTS_PATH}", file=sys.stderr)
    print(f"[save] {C.ASR_PNG}", file=sys.stderr)
    print(_summary_table(results))
    return results


if __name__ == "__main__":
    main()
