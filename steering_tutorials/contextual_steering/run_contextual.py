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

TWO GATES, REPORTED SIDE BY SIDE
--------------------------------
The per-prompt strength signal is selectable (config ``GATE`` / ``GATES_TO_RUN``):
  * "diffmean_cos" — cos(mean_pool(h) - benign_mean, v_unit). The HONEST NEGATIVE:
    ~zero per-instance class separation here, so ref <= tau, the ramp degenerates,
    and benign is NOT spared.
  * "probe" — the trained lesson-1 probe's logit (the LEARNED sensing vector CLAS
    prescribes; AUC ~0.87). ref > tau, the ramp operates, benign gets alpha ~ 0.
    THE FIX. Reused verbatim, mirroring how the sibling `flas` lesson gates.

RESULTS SCHEMA (kept in sync with README + plots)
-------------------------------------------------
{
  "model_id": str, "layer": int, "alpha_base": float,
  "n_extract_per_class": int, "n_eval_per_class": int, "judge": str,
  "gate": str,                                   # the headline gate (config.GATE)
  "schedule": {...headline gate schedule...},    # legacy slot (plots/summary)
  "arms": {                                      # unsteered/fixed are gate-free;
    "<unsteered|fixed|contextual>": {            # contextual == headline gate
       "harmful_refusal_rate": float, "harmful_gibberish_rate": float,
       "benign_over_refusal_rate": float, "benign_gibberish_rate": float,
       "mean_alpha_harmful": float, "mean_alpha_benign": float, "n": int },
    ...
  },
  "gates": {                                     # per-gate detail for BOTH gates
    "<diffmean_cos|probe>": {
       "gate": str, "signal": "probe_logit|diffmean_cosine",
       "schedule": {"tau", "ref", "cap", "benign_percentile",
                    "harmful_percentile", "harmful_proj_mean", ...},
       "ramp_degenerate": bool,                  # ref <= tau guard
       "separation": {"cohens_d": float, "auc": float,
                      "harmful_proj_mean": float, "benign_proj_mean": float, ...},
       "arm": {...contextual-arm stats incl mean_alpha_harmful/benign, rates...},
       "eval_projections": {"harmful": [float], "benign": [float]} },
    ...
  },
  "eval_projections": {"harmful": [float], "benign": [float]},   # headline gate
  "examples": [ {"prompt": str, "class": "harmful|benign",
                 "proj": float, "fixed_alpha": float, "fixed": str,
                 "fixed_verdict": str, "ctx_alpha": float, "contextual": str,
                 "ctx_verdict": str}, ... ],
  "plots": {"comparison": "fixed_vs_contextual.png",
            "schedule": "alpha_schedule.png",
            "schedule_<gate>": "alpha_schedule_<gate>.png"}
}
"""
from __future__ import annotations

import json
import sys

import numpy as np

from . import config as C
from .contextual import (
    ContextualSteerer,
    ProbeGate,
    calibrate_schedule,
    contextual_alpha,
    cosine_projection,
)


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


def _plot_schedule(proj_harmful, proj_benign, sched, alpha_base, cap, path,
                   signal_name: str = "cos(mean_pool(h), v_unit)") -> None:
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
    ax.set_xlabel(f"per-prompt signal  {signal_name}")
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


def _cohens_d(a, b) -> float:
    """Standardized mean gap (harmful − benign) / pooled-sd — the per-instance
    separation of the gate signal. ~0 means the gate cannot tell the classes
    apart (the diff-of-means cosine's failure); large positive means it can."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.size < 2 or b.size < 2:
        return 0.0
    sp = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2.0)
    return float((a.mean() - b.mean()) / sp) if sp > 0 else 0.0


def _auc(pos, neg) -> float:
    """P(signal(harmful) > signal(benign)) — the gate's ranking AUC, tie-corrected.
    Pairwise (eval sets are ~150/class, so 150×150 is trivial)."""
    pos = np.asarray(pos, dtype=np.float64)
    neg = np.asarray(neg, dtype=np.float64)
    if pos.size == 0 or neg.size == 0:
        return 0.5
    diff = pos[:, None] - neg[None, :]
    gt = float((diff > 0).sum())
    eq = float((diff == 0).sum())
    return (gt + 0.5 * eq) / (pos.size * neg.size)


def _separation(harm_proj, benign_proj) -> dict:
    """Does the gate signal separate harmful from benign per-instance? The single
    number that decides whether the ramp CAN spare benign."""
    hp = np.asarray(harm_proj, dtype=np.float64)
    bp = np.asarray(benign_proj, dtype=np.float64)
    return {
        "cohens_d": _cohens_d(hp, bp),
        "auc": _auc(hp, bp),
        "harmful_proj_mean": float(hp.mean()) if hp.size else 0.0,
        "benign_proj_mean": float(bp.mean()) if bp.size else 0.0,
        "harmful_proj_std": float(hp.std()) if hp.size else 0.0,
        "benign_proj_std": float(bp.std()) if bp.size else 0.0,
    }


def _gate_table(results: dict) -> str:
    """Per-gate summary: separation + contextual-arm alphas + rates for BOTH gates.
    Makes the honest-negative (diffmean_cos) vs the fix (probe) legible at a glance.
    """
    lines = ["", "-" * 78,
             "PER-GATE COMPARISON  (does the gate separate? does the ramp spare benign?)",
             "-" * 78,
             f"  {'gate':>13} {'sep_d':>6} {'auc':>5} {'a_harm':>7} {'a_ben':>7} "
             f"{'harm_ref':>9} {'ben_oref':>9} {'degen':>6}"]
    for name in results.get("gates", {}):
        g = results["gates"][name]
        sep, arm = g["separation"], g["arm"]
        lines.append(
            f"  {name:>13} {sep['cohens_d']:>6.2f} {sep['auc']:>5.2f} "
            f"{arm['mean_alpha_harmful']:>7.3f} {arm['mean_alpha_benign']:>7.3f} "
            f"{arm['harmful_refusal_rate']:>9.2f} {arm['benign_over_refusal_rate']:>9.2f} "
            f"{str(bool(g['ramp_degenerate'])):>6}")
    lines += ["-" * 78, ""]
    return "\n".join(lines)


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

    # --- 2. Build BOTH gates' per-prompt signals + calibrate each -------------
    #     (on the EXTRACT split only — never the eval split, which would leak).
    #
    # We report TWO gates so the comparison is honest:
    #   * "diffmean_cos" — cos(mean_pool(h) - benign_mean, v_unit). The benign-mean
    #     CENTER removes Gemma's huge prompt-independent common component; even so,
    #     this cosine barely separates the classes per-instance (Cohen's d ~ 0.1),
    #     so tau lands above the harmful mean, ref <= tau, and the ramp DEGENERATES
    #     to a hard step that cannot spare benign. This is the honest NEGATIVE.
    #   * "probe" — the trained lesson-1 probe's logit (the LEARNED sensing vector
    #     CLAS prescribes; AUC ~0.87). Harmful logits sit well above benign, so
    #     ref > tau, the ramp OPERATES, and benign prompts get alpha ~ 0. The FIX.
    #
    # Pool each extract prompt ONCE (at `layer`), then derive BOTH signals from the
    # cached activations — the probe reads the same layer, so no second forward.
    pooled_harm = [mean_pool_activation(model, tok, p, layer) for p in ex_harm]
    pooled_ben = [mean_pool_activation(model, tok, p, layer) for p in ex_ben]
    center = np.mean(np.stack(pooled_ben), axis=0) if pooled_ben else None

    # The probe gate (loads the lesson-1 checkpoint, mirroring flas's HarmGate).
    probe_gate = ProbeGate(model, tok, probe_path=C.PROBE_PATH)
    print(f"[probe-gate] loaded {C.PROBE_PATH} (probe layer={probe_gate.layer})",
          file=sys.stderr)
    reuse_pooled = probe_gate.layer == layer   # same layer => reuse cached activations

    ref_pct = C.REF_HARMFUL_PERCENTILE

    def extract_signal(gate: str):
        """Per-extract-prompt (harm, benign) signals for a gate, from cached pools."""
        if gate == "diffmean_cos":
            h = [cosine_projection(ph, v_unit, center) for ph in pooled_harm]
            b = [cosine_projection(pb, v_unit, center) for pb in pooled_ben]
        else:  # "probe"
            if reuse_pooled:
                h = [probe_gate.logit_from_pooled(ph) for ph in pooled_harm]
                b = [probe_gate.logit_from_pooled(pb) for pb in pooled_ben]
            else:  # probe reads a different layer — forward fresh at that layer
                h = [probe_gate.logit(p) for p in ex_harm]
                b = [probe_gate.logit(p) for p in ex_ben]
        return h, b

    steerers: dict[str, ContextualSteerer] = {}
    scheds: dict[str, dict] = {}
    degen: dict[str, bool] = {}
    for gate in C.GATES_TO_RUN:
        ex_h, ex_b = extract_signal(gate)
        sched = calibrate_schedule(ex_h, ex_b, C.TAU_BENIGN_PERCENTILE, ref_pct)
        sched["cap"] = C.CAP_MULT
        is_degen = sched["ref"] <= sched["tau"]
        degen[gate] = is_degen
        scheds[gate] = sched
        if is_degen:
            # Guard the degenerate branch (contextual_alpha then collapses to a hard
            # step at tau). Expected for diffmean_cos here; a RED FLAG for probe.
            print(f"[WARN] gate={gate}: ref ({sched['ref']:.3f}) <= tau "
                  f"({sched['tau']:.3f}) -> DEGENERATE ramp (hard step at tau, no "
                  f"linear region). The signal does not separate the classes.",
                  file=sys.stderr)
        print(f"[schedule/{gate}] tau={sched['tau']:.3f} ref={sched['ref']:.3f} "
              f"cap={C.CAP_MULT} degenerate={is_degen}", file=sys.stderr)
        steerers[gate] = ContextualSteerer(
            model, tok, v_unit, layer,
            alpha_base=C.ALPHA_BASE, tau=sched["tau"], ref=sched["ref"],
            cap=C.CAP_MULT, center=center, gate=gate,
            probe_gate=(probe_gate if gate == "probe" else None),
        )

    # --- 3. Cache each gate's EVAL projections (one forward per prompt/gate) ---
    eval_proj: dict[str, dict] = {}
    for gate in C.GATES_TO_RUN:
        st = steerers[gate]
        eval_proj[gate] = {
            "harmful": [st.projection(p) for p in ev_harm],
            "benign": [st.projection(p) for p in ev_ben],
        }
        sep = _separation(eval_proj[gate]["harmful"], eval_proj[gate]["benign"])
        print(f"[separation/{gate}] cohens_d={sep['cohens_d']:.2f} "
              f"auc={sep['auc']:.2f}", file=sys.stderr)

    # --- 4. Run the arms. unsteered/fixed are gate-independent (alpha fixed) so
    #        we run them ONCE; the contextual arm runs once per gate. --------------
    head = C.GATE                     # headline gate for the legacy result slots
    base_steerer = steerers[head]

    def run_arm(name: str, steerer, projs_harm, projs_ben):
        """Generate + judge every eval prompt under one arm; return verdicts+alphas.

        name in {"unsteered","fixed","contextual"}. For contextual, ``projs_*`` are
        that gate's cached eval projections; unused for the fixed-alpha arms.
        """
        h_verd, b_verd, h_alpha, b_alpha, recs = [], [], [], [], []
        for cls, prompts, projs, verd_list, alpha_list in (
            ("harmful", ev_harm, projs_harm, h_verd, h_alpha),
            ("benign", ev_ben, projs_ben, b_verd, b_alpha),
        ):
            for i, prompt in enumerate(prompts):
                if name == "unsteered":
                    alpha, proj_i = 0.0, None
                elif name == "fixed":
                    alpha, proj_i = steerer.alpha_base, None
                else:  # contextual — reuse the cached projection, don't recompute.
                    proj_i = projs[i]
                    alpha = steerer.alpha_for(proj_i)
                resp = steerer._generate(prompt, alpha, C.MAX_NEW_TOKENS)
                verd_list.append(judge.verdict(prompt, resp))
                alpha_list.append(alpha)
                recs.append({"class": cls, "prompt": prompt,
                             "proj": (None if proj_i is None else float(proj_i)),
                             "alpha": float(alpha), "response": resp,
                             "verdict": verd_list[-1]})
                if (i + 1) % 10 == 0:
                    print(f"[{name}/{cls}] {i + 1}/{len(prompts)}", file=sys.stderr)
        stats = _arm_stats(h_verd, b_verd, h_alpha, b_alpha)
        print(f"[{name}] harm_ref={stats['harmful_refusal_rate']:.2f} "
              f"ben_oref={stats['benign_over_refusal_rate']:.2f}", file=sys.stderr)
        return stats, recs

    arms, per_arm_recs = {}, {}
    for name in ("unsteered", "fixed"):
        arms[name], per_arm_recs[name] = run_arm(name, base_steerer, None, None)

    # The contextual arm, once per gate.
    gates_out: dict[str, dict] = {}
    ctx_recs_by_gate: dict[str, list] = {}
    for gate in C.GATES_TO_RUN:
        stats, recs = run_arm("contextual", steerers[gate],
                              eval_proj[gate]["harmful"], eval_proj[gate]["benign"])
        ctx_recs_by_gate[gate] = recs
        gates_out[gate] = {
            "gate": gate,
            "signal": ("probe_logit" if gate == "probe" else "diffmean_cosine"),
            "schedule": scheds[gate],
            "ramp_degenerate": bool(degen[gate]),
            "separation": _separation(eval_proj[gate]["harmful"],
                                      eval_proj[gate]["benign"]),
            "arm": stats,
            "eval_projections": eval_proj[gate],
        }

    # The headline gate fills the legacy `contextual` arm / schedule slots so the
    # existing plots, summary table, README and dashboard keep reading them.
    arms["contextual"] = gates_out[head]["arm"]

    # --- 5. Side-by-side fixed vs contextual examples (headline gate) ----------
    examples = []
    fixed_by_prompt = {r["prompt"]: r for r in per_arm_recs["fixed"]}
    for r in ctx_recs_by_gate[head]:
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
        "gate": head,                              # headline gate name
        "schedule": scheds[head],                  # legacy slot = headline schedule
        "arms": arms,                              # unsteered / fixed / contextual(head)
        "gates": gates_out,                        # per-gate detail for BOTH gates
        "eval_projections": eval_proj[head],       # legacy slot = headline projections
        "examples": examples,
        "plots": {"comparison": C.COMPARISON_PNG.name,
                  "schedule": C.SCHEDULE_PNG.name,
                  **{f"schedule_{g}": f"alpha_schedule_{g}.png" for g in C.GATES_TO_RUN}},
    }

    C.RESULTS_PATH.write_text(json.dumps(results, indent=2), encoding="utf-8")
    _plot_comparison(arms, C.COMPARISON_PNG)
    # One schedule plot per gate; the headline gate also writes the legacy name.
    for gate in C.GATES_TO_RUN:
        signal_name = ("probe logit" if gate == "probe"
                       else "cos(mean_pool(h) - benign_mean, v_unit)")
        _plot_schedule(eval_proj[gate]["harmful"], eval_proj[gate]["benign"],
                       scheds[gate], C.ALPHA_BASE, C.CAP_MULT,
                       C.ARTIFACTS / f"alpha_schedule_{gate}.png",
                       signal_name=signal_name)
    _plot_schedule(eval_proj[head]["harmful"], eval_proj[head]["benign"],
                   scheds[head], C.ALPHA_BASE, C.CAP_MULT, C.SCHEDULE_PNG,
                   signal_name=("probe logit" if head == "probe"
                                else "cos(mean_pool(h) - benign_mean, v_unit)"))
    print(f"[save] {C.RESULTS_PATH}", file=sys.stderr)
    print(f"[save] {C.COMPARISON_PNG}", file=sys.stderr)
    print(f"[save] {C.SCHEDULE_PNG}", file=sys.stderr)
    print(_summary_table(results))
    print(_gate_table(results))
    return results


if __name__ == "__main__":
    main()
