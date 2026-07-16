"""run_contextual.py — the orchestrator: FIXED-alpha vs CONTEXTUAL-alpha steering.

The experiment in one screen. Build lesson 2's refusal direction on an extract
split, calibrate the contextual schedule on that same split, then generate on a
MIXED held-out eval (harmful + benign) under three arms and judge every reply:

    unsteered   : alpha = 0 everywhere            — the abliterated baseline.
    fixed       : alpha = ALPHA_BASE everywhere   — lesson-2-style (no gate).
    contextual  : alpha = ALPHA_BASE * schedule(proj_i) — this lesson.

Two axes decide the winner:
    - harmful REFUSAL rate  (efficacy — steering should install refusal)  ↑ good
    - benign  OVER-REFUSAL  (collateral — benign must stay answered)      ↓ good

The claim to demonstrate: contextual keeps the harmful refusal rate close to
fixed while cutting the benign over-refusal rate — the direction gates itself, so
benign prompts (low projection) are barely steered. HONEST by construction: if
contextual does NOT cut benign over-refusal, or gives up too much harmful
refusal, the table says so.

Everything that loads or runs the model lives under ``main()`` so ``import
run_contextual`` is a no-op (safe for import-checks / tests). The team lead runs
this on the GPU.

RESULTS SCHEMA (kept in sync with README + plots)
-------------------------------------------------
{
  "model_id": str, "layer": int, "alpha_base": float,
  "n_extract_per_class": int, "n_eval_per_class": int,
  "schedule": {"tau": float, "ref": float, "cap": float,
               "benign_percentile": float, "harmful_proj_mean": float,
               "benign_proj_mean": float, ...},
  "arms": {
    "<unsteered|fixed|contextual>": {
       "harmful_refusal_rate": float, "harmful_gibberish_rate": float,
       "benign_over_refusal_rate": float, "benign_gibberish_rate": float,
       "mean_alpha_harmful": float, "mean_alpha_benign": float, "n": int },
    ...
  },
  "eval_projections": {"harmful": [float], "benign": [float]},
  "examples": [ {"prompt": str, "class": "harmful|benign",
                 "proj": float, "fixed_alpha": float, "fixed": str,
                 "fixed_verdict": str, "ctx_alpha": float, "contextual": str,
                 "ctx_verdict": str}, ... ],
  "plots": {"comparison": "fixed_vs_contextual.png",
            "schedule": "alpha_schedule.png"}
}
"""
from __future__ import annotations

import json
import sys

import numpy as np

from . import config as C
from .contextual import ContextualSteerer, calibrate_schedule, contextual_alpha


# --------------------------------------------------------------------------- #
# Pure helpers (no model) — safe to unit-test in isolation.
# --------------------------------------------------------------------------- #
def _rate(verdicts: list[str], key: str) -> float:
    """Fraction of ``verdicts`` equal to ``key`` (e.g. "REFUSAL")."""
    n = max(1, len(verdicts))
    return verdicts.count(key) / n


def _summary_table(results: dict) -> str:
    lines = ["", "=" * 66,
             "CONTEXTUAL vs FIXED STEERING  (harmful refusal UP  benign over-refusal DOWN)",
             "=" * 66,
             f"model: {results['model_id']}   layer: {results['layer']}   "
             f"alpha_base: {results['alpha_base']}",
             f"schedule: tau={results['schedule']['tau']:.3f}  "
             f"ref={results['schedule']['ref']:.3f}  cap={results['schedule']['cap']}",
             "",
             f"  {'arm':>11} {'harm_ref':>9} {'harm_gib':>9} "
             f"{'ben_oref':>9} {'ben_gib':>8} {'a_harm':>7} {'a_ben':>7}"]
    for name in ("unsteered", "fixed", "contextual"):
        a = results["arms"][name]
        lines.append(
            f"  {name:>11} {a['harmful_refusal_rate']:>9.2f} "
            f"{a['harmful_gibberish_rate']:>9.2f} "
            f"{a['benign_over_refusal_rate']:>9.2f} "
            f"{a['benign_gibberish_rate']:>8.2f} "
            f"{a['mean_alpha_harmful']:>7.3f} {a['mean_alpha_benign']:>7.3f}")
    lines += ["=" * 66, ""]
    return "\n".join(lines)


def _plot_comparison(arms: dict, path) -> None:
    """Grouped bars: harmful refusal & benign over-refusal for the three arms."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = ["unsteered", "fixed", "contextual"]
    harm = [arms[n]["harmful_refusal_rate"] for n in names]
    ben = [arms[n]["benign_over_refusal_rate"] for n in names]
    x = np.arange(len(names))
    w = 0.38
    fig, ax = plt.subplots(figsize=(7, 4))
    b1 = ax.bar(x - w / 2, harm, w, label="harmful refusal (↑ good)", color="#2a7")
    b2 = ax.bar(x + w / 2, ben, w, label="benign over-refusal (↓ good)", color="#c33")
    for bars in (b1, b2):
        for b in bars:
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.02,
                    f"{b.get_height():.2f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(names)
    ax.set_ylabel("rate"); ax.set_ylim(0, 1.05)
    ax.set_title("Fixed vs contextual steering\n"
                 "contextual aims to keep green high while cutting red")
    ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig)


def _plot_schedule(proj_harmful, proj_benign, sched, alpha_base, cap, path) -> None:
    """The ramp α(proj) with the eval projections of both classes underneath it."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    tau, ref = sched["tau"], sched["ref"]
    lo = min([tau, ref, *proj_harmful, *proj_benign, 0.0]) - 0.05
    hi = max([tau, ref, *proj_harmful, *proj_benign]) + 0.05
    xs = np.linspace(lo, hi, 200)
    ys = [contextual_alpha(x, alpha_base, tau, ref, cap) for x in xs]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(xs, ys, color="#333", lw=2, label="contextual α(proj)")
    ax.axvline(tau, ls="--", color="#c33", lw=1, label=f"tau={tau:.3f} (benign floor)")
    ax.axvline(ref, ls="--", color="#2a7", lw=1, label=f"ref={ref:.3f} (harmful anchor)")
    ax.axhline(alpha_base, ls=":", color="#888", lw=1, label=f"alpha_base={alpha_base}")
    # Rug of eval projections just above the x-axis.
    y0 = -0.01 * alpha_base
    ax.scatter(proj_benign, np.full(len(proj_benign), y0), marker="|",
               color="#c33", s=80, label="benign eval proj")
    ax.scatter(proj_harmful, np.full(len(proj_harmful), 2 * y0), marker="|",
               color="#2a7", s=80, label="harmful eval proj")
    ax.set_xlabel("projection  cos(mean_pool(h), v_unit)")
    ax.set_ylabel("steering strength α")
    ax.set_title("The contextual schedule: benign prompts land below tau (α≈0)")
    ax.legend(fontsize=7, loc="upper left")
    fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig)


def _arm_stats(harm_verdicts, ben_verdicts, harm_alphas, ben_alphas) -> dict:
    """Assemble one arm's row from its per-prompt verdicts + chosen alphas."""
    return {
        "harmful_refusal_rate": _rate(harm_verdicts, "REFUSAL"),
        "harmful_gibberish_rate": _rate(harm_verdicts, "GIBBERISH"),
        "benign_over_refusal_rate": _rate(ben_verdicts, "REFUSAL"),
        "benign_gibberish_rate": _rate(ben_verdicts, "GIBBERISH"),
        "mean_alpha_harmful": float(np.mean(harm_alphas)) if harm_alphas else 0.0,
        "mean_alpha_benign": float(np.mean(ben_alphas)) if ben_alphas else 0.0,
        "n": len(harm_verdicts),
    }


# --------------------------------------------------------------------------- #
# The pipeline — everything below loads / runs the model.
# --------------------------------------------------------------------------- #
def main() -> dict:
    import random

    import torch

    from steering_tutorials.common.data import load_harmful_benign
    from steering_tutorials.hello_world_steering.model_utils import (
        load_model, num_layers, mean_pool_activation,
    )
    from steering_tutorials.hello_world_steering.judge import Judge
    from steering_tutorials.hello_world_steering.steer_vector import (
        extract_caa_vector, save_vector,
    )

    random.seed(C.SEED); np.random.seed(C.SEED); torch.manual_seed(C.SEED)

    # --- Load the model we steer, and the judge -------------------------------
    model, tok = load_model(C.MODEL_ID)
    layer = min(C.LAYER, num_layers(model) - 1)
    judge = Judge(model, tok)   # off-family judge if STEER_JUDGE_MODEL is set

    # --- Data: ≥500/class, split into disjoint extract / eval -----------------
    data = load_harmful_benign(C.N_PER_CLASS, C.SEED)
    harmful, benign = data["harmful"], data["benign"]
    ex_harm = harmful[:C.N_EXTRACT_PER_CLASS]
    ex_ben = benign[:C.N_EXTRACT_PER_CLASS]
    # Held-out eval, CAPPED (generation is the expensive part).
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

    # --- 2. Calibrate the contextual schedule on the EXTRACT split ------------
    #     (never the eval split — that would leak). Read each extract prompt's
    #     projection onto v_unit, then set tau above benign and ref at harmful.
    if C.CALIBRATE:
        ex_harm_proj = [
            float(np.dot(mean_pool_activation(model, tok, p, layer), v_unit)
                  / (np.linalg.norm(mean_pool_activation(model, tok, p, layer)) + 1e-9))
            for p in ex_harm
        ]
        ex_ben_proj = [
            float(np.dot(mean_pool_activation(model, tok, p, layer), v_unit)
                  / (np.linalg.norm(mean_pool_activation(model, tok, p, layer)) + 1e-9))
            for p in ex_ben
        ]
        sched = calibrate_schedule(ex_harm_proj, ex_ben_proj, C.TAU_BENIGN_PERCENTILE)
        tau, ref = sched["tau"], sched["ref"]
    else:
        tau, ref = C.TAU_FALLBACK, C.REF_FALLBACK
        sched = {"tau": tau, "ref": ref, "benign_percentile": None,
                 "harmful_proj_mean": None, "benign_proj_mean": None}
    sched["cap"] = C.CAP_MULT
    print(f"[schedule] tau={tau:.3f} ref={ref:.3f} cap={C.CAP_MULT}", file=sys.stderr)

    steerer = ContextualSteerer(
        model, tok, v_unit, layer,
        alpha_base=C.ALPHA_BASE, tau=tau, ref=ref, cap=C.CAP_MULT,
    )

    # --- 3. Evaluate the three arms on the mixed held-out eval ----------------
    # Cache each eval prompt's projection once (reused for contextual alpha + the
    # schedule plot + the recorded projections).
    ev_harm_proj = [steerer.projection(p) for p in ev_harm]
    ev_ben_proj = [steerer.projection(p) for p in ev_ben]

    def run_arm(name: str):
        """Generate + judge every eval prompt under one arm; return verdicts+alphas."""
        h_verd, b_verd, h_alpha, b_alpha, recs = [], [], [], [], []
        for cls, prompts, projs, verd_list, alpha_list in (
            ("harmful", ev_harm, ev_harm_proj, h_verd, h_alpha),
            ("benign", ev_ben, ev_ben_proj, b_verd, b_alpha),
        ):
            for i, prompt in enumerate(prompts):
                if name == "unsteered":
                    alpha, resp = 0.0, steerer._generate(prompt, 0.0, C.MAX_NEW_TOKENS)
                elif name == "fixed":
                    r = steerer.fixed_generate(prompt, C.MAX_NEW_TOKENS)
                    alpha, resp = r["alpha"], r["response"]
                else:  # contextual — reuse the cached projection, don't recompute.
                    alpha = steerer.alpha_for(projs[i])
                    resp = steerer._generate(prompt, alpha, C.MAX_NEW_TOKENS)
                verd_list.append(judge.verdict(prompt, resp))
                alpha_list.append(alpha)
                recs.append({"class": cls, "prompt": prompt, "proj": float(projs[i]),
                             "alpha": float(alpha), "response": resp,
                             "verdict": verd_list[-1]})
                if (i + 1) % 10 == 0:
                    print(f"[{name}/{cls}] {i + 1}/{len(prompts)}", file=sys.stderr)
        stats = _arm_stats(h_verd, b_verd, h_alpha, b_alpha)
        print(f"[{name}] harm_ref={stats['harmful_refusal_rate']:.2f} "
              f"ben_oref={stats['benign_over_refusal_rate']:.2f}", file=sys.stderr)
        return stats, recs

    arms, per_arm_recs = {}, {}
    for name in ("unsteered", "fixed", "contextual"):
        arms[name], per_arm_recs[name] = run_arm(name)

    # --- 4. Side-by-side fixed vs contextual examples (a few of each class) ----
    examples = []
    fixed_by_prompt = {r["prompt"]: r for r in per_arm_recs["fixed"]}
    for r in per_arm_recs["contextual"]:
        if len([e for e in examples if e["class"] == r["class"]]) >= 4:
            continue
        f = fixed_by_prompt.get(r["prompt"], {})
        examples.append({
            "prompt": r["prompt"], "class": r["class"], "proj": r["proj"],
            "fixed_alpha": f.get("alpha"), "fixed": f.get("response"),
            "fixed_verdict": f.get("verdict"),
            "ctx_alpha": r["alpha"], "contextual": r["response"],
            "ctx_verdict": r["verdict"],
        })

    results = {
        "model_id": C.MODEL_ID,
        "layer": int(layer),
        "alpha_base": C.ALPHA_BASE,
        "n_extract_per_class": len(ex_harm),
        "n_eval_per_class": len(ev_harm),
        "judge": getattr(judge, "judge_id", "self"),
        "schedule": sched,
        "arms": arms,
        "eval_projections": {"harmful": ev_harm_proj, "benign": ev_ben_proj},
        "examples": examples,
        "plots": {"comparison": C.COMPARISON_PNG.name,
                  "schedule": C.SCHEDULE_PNG.name},
    }

    C.RESULTS_PATH.write_text(json.dumps(results, indent=2), encoding="utf-8")
    _plot_comparison(arms, C.COMPARISON_PNG)
    _plot_schedule(ev_harm_proj, ev_ben_proj, sched, C.ALPHA_BASE, C.CAP_MULT,
                   C.SCHEDULE_PNG)
    print(f"[save] {C.RESULTS_PATH}", file=sys.stderr)
    print(f"[save] {C.COMPARISON_PNG}", file=sys.stderr)
    print(f"[save] {C.SCHEDULE_PNG}", file=sys.stderr)
    print(_summary_table(results))
    return results


if __name__ == "__main__":
    main()
