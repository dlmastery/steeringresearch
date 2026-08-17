"""run_near_orthogonal.py — the NEAR-ORTHOGONAL arm of the sec.9 decision rule.

    near-orthogonal DIRECTIONS ... STACK **until the norm budget is spent**

This is the clause ``run_stacking`` never tested and ``run_hillclimb`` only
touched at its endpoint (one exactly-orthogonal direction, added once, with no
stopping condition anywhere in the ladder). Two arms close it:

**ARM 1 — the cosine sweep.** "Near-orthogonal" is turned into a dial. For each
target cosine ``t`` we build ``W(t)`` at EXACTLY that cosine to the refusal
direction (``near_orthogonal.rotate_toward``) and measure the same-site pair
``[A, W(t)]`` against both of its constituents. As ``t`` runs 0 -> 1 the pair
walks continuously out of the STACK clause (orthogonal directions) and into the
COMPETE clause (same direction, which at ``t=1`` is literally A at double
strength). If the sec.9 boundary is real, it has a location, and this arm is
where it would show up.

**ARM 2 — the budget-limited additive ladder.** Starting from ``[A]``, each rung
adds exactly ONE more near-orthogonal direction at the same site. Before any
generation, each candidate must pass :func:`~.near_orthogonal.admit_direction`
(near-orthogonality bar + predicted-budget ceiling); after measurement, each rung
must pass the four pre-registered gates in
:data:`~.near_orthogonal.STOP_RULES` (budget / coherence / competition /
selectivity). A rung that fails is **DROPPED and reverted** — the next candidate
is judged against the last KEPT rung, exactly as if the failure had never
happened — and a budget break stops the ladder outright. The forbidden "all-on
hybrid" is therefore unreachable: no configuration exists in this run that
carries a prior which failed its own gate.

DATA FLOOR. Both arms default to the rubric floor of >=500 harmful and >=500
benign held-out prompts (``data_floor.FLOOR_PER_CLASS``). The measured pool caps
a BALANCED load at 792/class, so the split planner trims the extract slice
300 -> 292 to buy exactly 500 eval rows per class; whatever is achieved is
stamped into ``results["data_floor"]`` with ``pool_capped`` / ``env_capped``
distinguished, and a shortfall prints a loud warning. Env caps exist for a
foreground screening run and are recorded as such — never laundered into a
caveat.

RESULTS SCHEMA (``artifacts/near_ortho_results.json``)
------------------------------------------------------
{
  "meta": {...},                 # model, layers, alpha, cosines, caps, judge stamp
  "data_floor": {...},           # achieved n per class + why, see data_floor.py
  "directions": {"cos_to_refusal": {...}, "gram_ladder": [[...]], ...},
  "preflight": [ {...admit_direction diagnostics per candidate...} ],
  "configs": { "<key>": {"label","priors","added","cos_to_refusal",
                         "predicted_budget","norm_budget",
                         "harmful": {...}, "benign": {...}|null} },
  "ladder": {"rows":[...], "kept":[...], "stopped_at":..., "binding_constraint":...},
  "sweep":  [ {"cos_target","cos_measured","pair_refusal","standalone_refusal",
               "a_alone_refusal","vs_best_constituent","stacks",
               "predicted_budget","norm_budget"} ],
  "notes": [ ... ]
}

Run it (see the README section 8c for the capped screening variant):

    STEER_JUDGE_MODEL=Qwen/Qwen2.5-3B-Instruct \\
      python -m steering_tutorials.stacking.run_near_orthogonal

    python -m steering_tutorials.stacking.run_near_orthogonal --report    # no GPU
    python -m steering_tutorials.stacking.run_near_orthogonal --selftest  # no model
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np

from . import config as C
from . import data_floor as DF
from .near_orthogonal import (
    NEAR_ORTHO_MAX_COS,
    admit_direction,
    cosine_family,
    gram_cosines,
    max_abs_offdiag,
    orthogonal_capacity,
    orthonormal_complement_basis,
    predicted_budget,
    replay_ladder,
    rotate_toward,
)
from .stacking import Prior

# --- run-size knobs. Defaults sit AT the rubric floor; the env caps exist so a
# --- RAM-constrained host can take a labelled screening slice in one window.
N_HARM = int(os.environ.get("NORTHO_N_HARM") or DF.FLOOR_PER_CLASS)
N_BENIGN = int(os.environ.get("NORTHO_N_BENIGN") or DF.FLOOR_PER_CLASS)
N_EXTRACT_PREF = int(os.environ.get("NORTHO_N_EXTRACT") or C.N_EXTRACT)
N_BUDGET = int(os.environ.get("NORTHO_BUDGET_N") or C.N_NORM_BUDGET_PROMPTS)

# --- the near-orthogonality axis ------------------------------------------- #
# LADDER_K candidate directions, each built at NEAR_COS to the refusal axis and
# mutually near-orthogonal by construction (cos(w_i,w_j) = NEAR_COS^2). NEAR_COS
# is deliberately NON-zero: an exactly-orthogonal control is what run_hillclimb
# already measured, and "near-orthogonal" is the clause's actual wording.
LADDER_K = int(os.environ.get("NORTHO_LADDER_K") or 4)
NEAR_COS = float(os.environ.get("NORTHO_NEAR_COS") or 0.20)
COS_GRID = [float(x) for x in (os.environ.get("NORTHO_COS_GRID")
                               or "0.0,0.25,0.5,0.75,0.95").split(",") if x.strip()]

# --- the norm budget ------------------------------------------------------- #
# Ceiling on the measured N5 displacement at the injection site. 0.20 admits
# floor(0.20^2 / 0.08^2) = 6 exactly-orthogonal alpha=0.08 steps, so with K=4 the
# ladder is NOT expected to be stopped by the budget -- the pre-registered
# prediction is that COHERENCE binds first. If the budget rule never fires, that
# is the finding, and it is reported as such.
BUDGET_CEILING = float(os.environ.get("NORTHO_BUDGET_CEILING") or 0.20)
COHERENCE_TOL = float(os.environ.get("NORTHO_COHERENCE_TOL") or 0.05)
COMPETE_TOL = float(os.environ.get("NORTHO_COMPETE_TOL") or 0.0)
SELECTIVITY_TOL = float(os.environ.get("NORTHO_SELECTIVITY_TOL") or 0.10)

# "ladder", "sweep", or both (comma separated). The single 4090 serialises GPU
# work, so running one arm per window is the normal mode on this host.
ARMS = [a.strip() for a in (os.environ.get("NORTHO_ARMS") or "ladder,sweep").split(",")
        if a.strip()]
MAX_CONFIGS = int(os.environ.get("NORTHO_MAX_CONFIGS") or 0)

PARTIAL_PATH = C.ARTIFACTS / "near_ortho_partial.json"
RESULTS_PATH = C.ARTIFACTS / "near_ortho_results.json"
PLOT_PATH = C.ARTIFACTS / "near_ortho.png"
DIRECTIONS_PATH = C.ARTIFACTS / "near_ortho_directions.npz"


# --------------------------------------------------------------------------- #
# Pure helpers (no model) — unit-testable.
# --------------------------------------------------------------------------- #
def summarize(records: "list[dict]") -> dict:
    """Collapse per-prompt judge records into rates + the continuous score.

    Items the judge could not score carry ``label=None``; they are COUNTED in
    ``n_judge_skipped`` and EXCLUDED from every rate — never defaulted to a
    verdict, which would quietly manufacture refusals or compliances.
    """
    scored = [r for r in records if r.get("label")]
    n = len(scored)
    if n == 0:
        return {"n": 0, "refusal_rate": None, "compliance_rate": None,
                "gibberish_rate": None, "refusal_score": None,
                "n_judge_skipped": len(records)}
    return {
        "n": n,
        "refusal_rate": sum(r["label"] == "REFUSAL" for r in scored) / n,
        "compliance_rate": sum(r["label"] == "COMPLIANCE" for r in scored) / n,
        "gibberish_rate": sum(r["label"] == "GIBBERISH" for r in scored) / n,
        "refusal_score": float(np.mean([r["p_refusal"] for r in scored])),
        "n_judge_skipped": len(records) - n,
    }


def sweep_table(configs: dict) -> "list[dict]":
    """Read the cosine sweep: does ``[A, W(t)]`` beat its best constituent?

    Two criteria, both reported, neither swapped for the other after the fact:

    ``vs_best_constituent``
        the README's own bar — a stack that does not beat the better of its two
        constituents has not stacked, whatever the marginal says.
    ``marginal_vs_standalone``
        the pre-registered bar used by ``run_hillclimb`` — W(t) delivers less
        inside the pair than it does alone => it competes.

    Rows are ordered by MEASURED cosine, so the table reads as the dial it is.
    """
    def h(key, field):
        c = configs.get(key)
        return None if not c or not c.get("harmful") else c["harmful"].get(field)

    def sub(a, b):
        return None if (a is None or b is None) else float(a - b)

    a_alone = h("L0", "refusal_rate")
    unsteered = h("R0", "refusal_rate")
    rows = []
    for key, cfg in configs.items():
        if not key.startswith("SW_PAIR"):
            continue
        idx = key.split("_")[-1]
        solo_key = f"SW_SOLO_{idx}"
        pair = h(key, "refusal_rate")
        solo = h(solo_key, "refusal_rate")
        constituents = {k: v for k, v in
                        (("A", a_alone), ("W", solo)) if v is not None}
        best_k = max(constituents, key=constituents.get) if constituents else None
        rows.append({
            "key": key,
            "cos_target": cfg.get("cos_target"),
            "cos_measured": cfg.get("cos_to_refusal"),
            "pair_refusal": pair,
            "pair_gibberish": h(key, "gibberish_rate"),
            "standalone_refusal": solo,
            "standalone_gibberish": h(solo_key, "gibberish_rate"),
            "a_alone_refusal": a_alone,
            "unsteered_refusal": unsteered,
            "best_constituent": best_k,
            "vs_best_constituent": (None if best_k is None
                                    else sub(pair, constituents[best_k])),
            "stacks_vs_best_constituent": (None if (best_k is None or pair is None)
                                           else bool(pair > constituents[best_k])),
            "marginal_of_W": sub(pair, a_alone),
            "standalone_of_W": sub(solo, unsteered),
            "competes_marginal_vs_standalone": (
                None if (sub(pair, a_alone) is None or sub(solo, unsteered) is None)
                else bool(sub(pair, a_alone) < sub(solo, unsteered))),
            "predicted_budget": cfg.get("predicted_budget"),
            "norm_budget": cfg.get("norm_budget"),
        })
    rows.sort(key=lambda r: (r["cos_measured"] if r["cos_measured"] is not None else 0))
    return rows


def apply_preflight_stop(state: dict, admissions: "list[dict]") -> dict:
    """Fold a PRE-FLIGHT budget refusal into the ladder's binding constraint.

    ``replay_ladder`` only sees rungs that were measured, so a ladder halted by
    the admission test — the budget clause binding at its cheapest possible
    moment, before any generation — would otherwise report
    ``binding_constraint = NONE``. That reads as "nothing stopped it", which is
    the opposite of what happened. Mutates and returns ``state``.
    """
    if state.get("stopped_at"):
        return state
    for a in admissions or []:
        if a.get("reason") == "BUDGET_EXCEEDED":
            state["preflight_stop"] = a.get("candidate")
            state["binding_constraint"] = (
                f"BUDGET (at PRE-FLIGHT: candidate {a.get('candidate')} refused "
                f"before generation, predicted {a.get('predicted_budget_after')} "
                f"> ceiling {a.get('budget_ceiling')})")
            break
    return state


def build_notes(res: dict) -> "list[str]":
    """Plain-language findings derived ONLY from the measured cells.

    Every note names the numbers it rests on so a reader can refuse it. Notes are
    generated, not written by hand, so they cannot drift away from the JSON that
    is sitting beside them.
    """
    out: list[str] = []
    lad = res.get("ladder") or {}
    rows = {r["rung"]: r for r in lad.get("rows", [])}
    cfg = res.get("configs", {})

    # A candidate refused BEFORE generation is the budget clause binding at its
    # cheapest possible moment — worth stating first, because it is the one place
    # this lesson's mechanism claim costs no GPU time to check.
    for a in res.get("admissions", []):
        if a.get("admit") is False:
            out.append(
                f"PRE-FLIGHT REFUSAL of {a['candidate']} on {a['reason']}: "
                f"max|cos|={a['max_abs_cos']:.3f} (bar {a['max_cos_bar']}), "
                f"predicted budget {a['predicted_budget_before']:.4f} -> "
                f"{a['predicted_budget_after']:.4f} against a ceiling of "
                f"{a['budget_ceiling']}, measured against {a.get('against')}. "
                f"The norm-budget clause bound here without a single generation.")

    kept = [k for k in lad.get("kept", []) if k != "L0"]
    if lad.get("rows"):
        if lad.get("stopped_at"):
            out.append(
                f"LADDER STOPPED at {lad['stopped_at']} on rule "
                f"{lad['binding_constraint']}. Directions kept beyond the base: "
                f"{len(kept)} of {LADDER_K} candidates.")
        else:
            refused_budget = any(a.get("reason") == "BUDGET_EXCEEDED"
                                 for a in res.get("admissions", []))
            tail = (" -- but a later candidate was refused at PRE-FLIGHT on "
                    "budget, so the clause bound before measurement rather than "
                    "after it."
                    if refused_budget else
                    f", so 'stack until the norm budget is spent' was NOT the "
                    f"operative clause at alpha={C.STACK_ALPHA} on this model.")
            out.append(
                f"LADDER ENDED WITHOUT A MEASURED STOP: every candidate that was "
                f"generated passed its gates or was dropped and reverted; kept "
                f"{len(kept)}. No MEASURED rung exceeded the budget ceiling "
                f"{BUDGET_CEILING}" + tail)
        fired = [r for r in lad["rows"] if r.get("failed_rules")]
        for r in fired:
            out.append(
                f"{r['rung']} DROPPED on {'+'.join(r['failed_rules'])}: "
                f"refusal {r.get('refusal_rate')} vs reference "
                f"{r.get('reference_rung')}, marginal {r.get('marginal_refusal')}, "
                f"gibberish marginal {r.get('marginal_gibberish')}, budget "
                f"{r.get('norm_budget')} (ceiling {BUDGET_CEILING}).")

    # Budget accounting: does the measured N5 follow the sqrt(k) prediction?
    pred_obs = [(r.get("predicted_budget"), r.get("norm_budget"))
                for r in lad.get("rows", [])
                if r.get("predicted_budget") and r.get("norm_budget")]
    if len(pred_obs) >= 2:
        ratios = [o / p for p, o in pred_obs if p]
        out.append(
            f"BUDGET ACCOUNTING: measured/predicted N5 ratio ranges "
            f"{min(ratios):.2f}-{max(ratios):.2f} across the kept rungs. The "
            f"first-order prediction alpha*sqrt(1^T G 1) drops an O(alpha^2) "
            f"term from the sequential hooks; a ratio persistently above 1 is "
            f"that compounding, not a bookkeeping error.")

    sw = res.get("sweep") or []
    stacked = [r for r in sw if r.get("stacks_vs_best_constituent")]
    if sw:
        out.append(
            f"COSINE SWEEP: {len(stacked)} of {len(sw)} cosine cells beat their "
            f"best constituent. Cosines measured: "
            + ", ".join(f"{r['cos_measured']:.2f}" for r in sw if
                        r.get("cos_measured") is not None) + ".")
        if not stacked:
            out.append(
                "NO CELL STACKED AT ANY COSINE. The sec.9 near-orthogonal clause "
                "predicts a stacking regime at low cosine; on this substrate none "
                "appears, so the clause's boundary cannot be located here - the "
                "same conclusion the site clause reached in run_hillclimb.")

    r0 = (cfg.get("R0") or {}).get("harmful") or {}
    if r0.get("gibberish_rate") is not None and r0["gibberish_rate"] > 0.3:
        out.append(
            f"SUBSTRATE CAVEAT: the UNSTEERED gibberish rate is "
            f"{r0['gibberish_rate']:.3f}. Every rung is measured on a base with "
            f"little coherence headroom, so a coherence-driven DROP says as much "
            f"about the substrate as about the added direction.")
    return out


def _fmt(x, nd=3):
    return "   n/a" if x is None else f"{x:>6.{nd}f}"


def summary_text(res: dict) -> str:
    m, d = res["meta"], res["data_floor"]
    L = ["", "=" * 100,
         "NEAR-ORTHOGONAL ARM - sec.9 clause 3: stack until the norm budget is spent",
         "=" * 100,
         f"model {m['model_id']}   site L{m['primary_layer']}   alpha={m['stack_alpha']}",
         f"near_cos={m['near_cos']}  K={m['ladder_k']}  ceiling={m['budget_ceiling']}  "
         f"cos_bar={m['near_ortho_max_cos']}  judge={m['judge_id']}",
         f"data floor: harmful n={d['achieved_n_harmful']} benign n={d['achieved_n_benign']} "
         f"(floor {d['floor_per_class']}, meets={d['meets_floor']}, "
         f"pool_capped={d['pool_capped']}, env_capped={d['env_capped']})",
         f"split: {d['split_plan']['note']}",
         ""]

    # Prefer the LIVE admissions (judged against the rungs actually kept); fall
    # back to the static plan when only the plan exists (e.g. sweep-only run).
    adm = res.get("admissions") or res.get("preflight")
    if adm:
        L += [f"  ADMISSION, {'LIVE (vs kept rungs)' if res.get('admissions') else 'PLANNED (all-kept assumption)'}"
              f" - no generation involved",
              f"  {'cand':<6} {'maxcos':>7} {'pred_before':>12} {'pred_after':>11} "
              f"{'cap_left':>9} {'admit':>6} {'reason':<22}"]
        for p in adm:
            L.append(f"  {p['candidate']:<6} {_fmt(p['max_abs_cos'])} "
                     f"{_fmt(p['predicted_budget_before']):>12} "
                     f"{_fmt(p['predicted_budget_after']):>11} "
                     f"{p['orthogonal_capacity_left']:>9} "
                     f"{str(p['admit']):>6} {str(p['reason'] or '-'):<22}")
        L.append("")

    lad = res.get("ladder") or {}
    if lad.get("rows"):
        L += ["  LADDER (each rung adds ONE near-orthogonal direction; DROP => reverted)",
              f"  {'rung':<5} {'added':<8} {'cos_v':>7} {'maxcos':>7} {'refus':>7} "
              f"{'gibb':>7} {'ben_ref':>8} {'budget':>7} {'pred':>7} {'dRefus':>7} "
              f"{'verdict':<12} {'failed':<24}"]
        for r in lad["rows"]:
            L.append(
                f"  {r['rung']:<5} {str(r.get('added') or '-'):<8} "
                f"{_fmt(r.get('cos_to_refusal'))} "
                f"{_fmt(r.get('max_abs_cos_within_stack'))} "
                f"{_fmt(r.get('refusal_rate'))} {_fmt(r.get('gibberish_rate'))} "
                f"{_fmt(r.get('benign_refusal_rate')):>8} "
                f"{_fmt(r.get('norm_budget'))} {_fmt(r.get('predicted_budget'))} "
                f"{_fmt(r.get('marginal_refusal'))} {str(r.get('verdict')):<12} "
                f"{'+'.join(r.get('failed_rules') or []) or '-':<24}")
        L += ["", f"  kept: {lad.get('kept')}   stopped_at: {lad.get('stopped_at')}   "
                  f"binding: {lad.get('binding_constraint')}", ""]

    if res.get("sweep"):
        L += ["  COSINE SWEEP  [A, W(t)] at one site, vs each constituent",
              f"  {'cos':>6} {'pair':>7} {'W alone':>8} {'A alone':>8} {'vs best':>8} "
              f"{'stacks':>7} {'budget':>7} {'pred':>7}"]
        for r in res["sweep"]:
            L.append(f"  {_fmt(r['cos_measured'], 2)} {_fmt(r['pair_refusal'])} "
                     f"{_fmt(r['standalone_refusal']):>8} {_fmt(r['a_alone_refusal']):>8} "
                     f"{_fmt(r['vs_best_constituent']):>8} "
                     f"{str(r['stacks_vs_best_constituent']):>7} "
                     f"{_fmt(r['norm_budget'])} {_fmt(r['predicted_budget'])}")
        L.append("")

    for n in res.get("notes", []):
        L.append(f"  [NOTE] {n}")
    L += ["=" * 100, ""]
    return "\n".join(L)


# --------------------------------------------------------------------------- #
# Plot — matplotlib Agg (headless), PNG per the dashboard mandate.
# --------------------------------------------------------------------------- #
def _plot(res: dict, path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(16, 4.4))

    sw = res.get("sweep") or []
    if sw:
        xs = [r["cos_measured"] for r in sw]
        a1.plot(xs, [r["pair_refusal"] for r in sw], "o-", color="#37a",
                label="[A, W(t)] pair")
        a1.plot(xs, [r["standalone_refusal"] for r in sw], "s--", color="#c93",
                label="W(t) alone")
        if sw[0]["a_alone_refusal"] is not None:
            a1.axhline(sw[0]["a_alone_refusal"], color="#2a7", ls=":",
                       label="A alone")
        if sw[0]["unsteered_refusal"] is not None:
            a1.axhline(sw[0]["unsteered_refusal"], color="#666", ls="-.",
                       label="unsteered")
        a1.axvline(NEAR_ORTHO_MAX_COS, color="#c33", ls="--", lw=1,
                   label=f"near-ortho bar {NEAR_ORTHO_MAX_COS}")
    a1.set_xlabel("cos(W, refusal direction)  [measured]")
    a1.set_ylabel("refusal rate (harmful)")
    a1.set_title("Clause 3 as a dial:\northogonal (STACK) -> same direction (COMPETE)")
    a1.legend(fontsize=7)
    a1.grid(alpha=.3)

    rows = [r for r in (res.get("ladder") or {}).get("rows", [])
            if r.get("refusal_rate") is not None]
    if rows:
        labels = [r["rung"] for r in rows]
        colors = {"BASE": "#666", "KEEP": "#2a7", "DROP": "#c33",
                  "NOT_EVALUATED": "#ccc", "NOT_MEASURED": "#ccc"}
        a2.bar(labels, [r["refusal_rate"] for r in rows],
               color=[colors.get(r["verdict"], "#999") for r in rows],
               edgecolor="k", lw=.6)
        a2.plot(labels, [r["gibberish_rate"] for r in rows], "^--", color="#c33",
                label="gibberish (harmful)")
        ben = [r.get("benign_refusal_rate") for r in rows]
        if any(b is not None for b in ben):
            a2.plot(labels, [b if b is not None else np.nan for b in ben], "o:",
                    color="#93c", label="benign refusal (selectivity)")
        a2.set_title("Additive ladder\ngreen=KEEP  red=DROP (reverted)")
        a2.set_ylim(-0.02, 1.05)
        a2.legend(fontsize=7)
        a2.grid(axis="y", alpha=.3)

        a3.bar(labels, [r.get("norm_budget") or 0.0 for r in rows], color="#963",
               edgecolor="k", lw=.6, label="measured N5")
        a3.plot(labels, [r.get("predicted_budget") or np.nan for r in rows], "D--",
                color="#37a", label="predicted alpha*sqrt(1'G1)")
        a3.axhline(res["meta"]["budget_ceiling"], color="#c33", ls="--",
                   label=f"ceiling {res['meta']['budget_ceiling']}")
        a3.set_title("Norm budget: the stopping condition")
        a3.legend(fontsize=7)
        a3.grid(axis="y", alpha=.3)

    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# The run — everything below loads / runs the model.
# --------------------------------------------------------------------------- #
def main() -> dict:
    import random

    import torch

    from steering_tutorials.hello_world_steering.model_utils import (
        load_model, num_layers, last_token_activations,
    )
    from steering_tutorials.hello_world_steering.judge import Judge, JudgeUnavailable
    from steering_tutorials.common.data import build_harmful_benign

    from .hillclimb import apply_config, measure_norm_budget

    t0 = time.time()
    random.seed(C.SEED)
    np.random.seed(C.SEED)
    torch.manual_seed(C.SEED)

    # --- 1. Data FIRST: the floor is planned before a model is touched --------
    # Ask for extract + the larger eval slice; the loader returns a BALANCED set
    # and reports the pool it drew from, so the plan is made against reality.
    want_per_class = N_EXTRACT_PREF + max(N_HARM, N_BENIGN)
    rec = build_harmful_benign(n_per_class=want_per_class, seed=C.SEED)
    harmful_all = [r["prompt"] for r in rec["harmful"]]
    benign_all = [r["prompt"] for r in rec["benign"]]
    header = rec["header"]
    pool = min(len(harmful_all), len(benign_all))

    plan = DF.plan_split(pool, N_EXTRACT_PREF, max(N_HARM, N_BENIGN))
    n_extract = plan["n_extract"]
    ex_h = harmful_all[:n_extract]
    ex_b = benign_all[:n_extract]
    ev_h = harmful_all[n_extract:n_extract + N_HARM]
    ev_b = benign_all[n_extract:n_extract + N_BENIGN]

    floor = DF.floor_report(
        len(ev_h), len(ev_b), plan,
        requested_harmful=N_HARM, requested_benign=N_BENIGN,
        pool_harmful_raw=header.get("harmful_pool_after_topup"),
        pool_benign_raw=header.get("benign_pool"))
    print(f"[split] extract {len(ex_h)}h/{len(ex_b)}b   eval {len(ev_h)}h/{len(ev_b)}b   "
          f"pool {pool}/class", file=sys.stderr)
    print(f"[split] {plan['note']}", file=sys.stderr)
    DF.warn_if_below_floor(floor)

    # --- 2. Model + directions ----------------------------------------------
    model, tok = load_model(C.MODEL_ID)
    primary = min(C.PRIMARY_LAYER, num_layers(model) - 1)

    stamp = (f"{C.MODEL_ID}|L{primary}|n{len(ex_h)}|seed{C.SEED}|K{LADDER_K}"
             f"|near{NEAR_COS}|grid{','.join(str(g) for g in COS_GRID)}")
    cached = None
    if DIRECTIONS_PATH.exists():
        try:
            z = np.load(DIRECTIONS_PATH, allow_pickle=False)
            if str(z["stamp"].item() if z["stamp"].ndim == 0 else z["stamp"][0]) == stamp:
                cached = z
            else:
                z.close()
        except Exception:
            cached = None

    if cached is not None:
        v_unit = cached["v_unit"].astype(np.float32)
        basis = cached["basis"].astype(np.float32)
        cached.close()          # Windows will not unlink an npz with an open handle
        print(f"[cache] directions reused ({stamp})", file=sys.stderr)
    else:
        acts_h = last_token_activations(model, tok, ex_h, primary)
        acts_b = last_token_activations(model, tok, ex_b, primary)
        v_raw = (acts_h.mean(axis=0) - acts_b.mean(axis=0)).astype(np.float32)
        v_unit = (v_raw / np.linalg.norm(v_raw)).astype(np.float32)
        basis = orthonormal_complement_basis(
            np.vstack([acts_h, acts_b]), v_unit, k=max(LADDER_K, 1))
        np.savez(DIRECTIONS_PATH, stamp=np.array(stamp), v_unit=v_unit, basis=basis)
        print(f"[dirs] extracted and cached ({stamp})", file=sys.stderr)

    # Ladder candidates: near-orthogonal, NOT exactly orthogonal. Built so that
    # cos(w_i, v) = NEAR_COS exactly and cos(w_i, w_j) = NEAR_COS^2 -- a geometry
    # that is known in closed form and therefore CHECKABLE against gram_cosines.
    ladder_dirs = [rotate_toward(v_unit, basis[i], NEAR_COS) for i in range(LADDER_K)]
    gram_ladder = gram_cosines([v_unit] + ladder_dirs)
    print(f"[dirs] cos(w_i, v)={[round(float(np.dot(w.astype(np.float64), v_unit.astype(np.float64))), 4) for w in ladder_dirs]}",
          file=sys.stderr)
    print(f"[dirs] max |cos| off-diagonal over [v]+ladder = "
          f"{max_abs_offdiag(gram_ladder):.4f}", file=sys.stderr)

    sweep_dirs = cosine_family(v_unit, basis[0], COS_GRID)

    A = Prior(f"A refusal@L{primary}", v_unit, primary, C.STACK_ALPHA, "relative_add")

    judge = Judge(model, tok)
    print(f"[judge] {judge.judge_id}", file=sys.stderr)

    # --- 3. Pre-flight admission (pure arithmetic, BEFORE any generation) -----
    # The budget account decides which candidates may even be ATTEMPTED, and says
    # so in numbers before a single token is generated. This static table is the
    # PLAN: it assumes every candidate is kept. The live ladder re-runs the same
    # test against the rungs actually KEPT (a dropped rung frees its budget), and
    # both are reported -- the plan as the pre-registration, the live decisions as
    # what was executed.
    preflight = []
    admitted_vecs = [v_unit]
    for i, w in enumerate(ladder_dirs, start=1):
        diag = admit_direction(w, admitted_vecs, C.STACK_ALPHA, BUDGET_CEILING)
        diag["candidate"] = f"w{i}"
        diag["against"] = ["v_refusal"] + [f"w{j}" for j in range(1, i)]
        diag["cos_to_refusal"] = float(np.dot(w.astype(np.float64),
                                              v_unit.astype(np.float64)))
        preflight.append(diag)
        if diag["admit"]:
            admitted_vecs.append(w)
    print(f"[preflight] plan admits {sum(p['admit'] for p in preflight)}/{LADDER_K}; "
          f"an EXACTLY-orthogonal stack at alpha={C.STACK_ALPHA} would fit "
          f"{orthogonal_capacity(C.STACK_ALPHA, BUDGET_CEILING)} steps under "
          f"ceiling {BUDGET_CEILING} -- the shortfall is the near-orthogonality "
          f"premium at cos={NEAR_COS}", file=sys.stderr)

    # --- 4. Measurement machinery -------------------------------------------
    partial: dict = {}
    if PARTIAL_PATH.exists():
        try:
            partial = json.loads(PARTIAL_PATH.read_text(encoding="utf-8"))
            print(f"[resume] {len(partial)} config(s) already on disk", file=sys.stderr)
        except Exception:
            partial = {}

    def _score(prompt, text):
        try:
            s = judge.verdict_scores(prompt, text)
        except JudgeUnavailable:
            return {"label": None, "p_refusal": None}
        return {"label": max(s, key=s.get), "p_refusal": float(s["REFUSAL"])}

    def _run_arm(prompts, priors, tag):
        recs = []
        for i, p in enumerate(prompts):
            text, _fired = apply_config(model, tok, p, priors, C.MAX_NEW_TOKENS,
                                        clamp_cap=None, gate=None)
            r = _score(p, text)
            r["prompt"], r["response"] = p, text
            recs.append(r)
            if (i + 1) % 25 == 0:
                print(f"    [{tag}] {i + 1}/{len(prompts)} ({time.time() - t0:.0f}s)",
                      file=sys.stderr)
        return recs

    n_new = 0
    budget_cap_hit = {"v": False}

    def measure(key, label, priors, do_benign, extra=None):
        """Measure one configuration, checkpointing it the moment it completes.

        The signature encodes everything that could change a number (run sizes,
        alpha, the exact prior list, the extract size). A cell measured under a
        different signature is stale and is dropped at assembly time rather than
        silently mixed in.
        """
        nonlocal n_new
        names = "+".join(p.name for p in priors) or "none"
        sig = (f"{key}|nh{len(ev_h)}|nb{len(ev_b) if do_benign else 0}|"
               f"a{C.STACK_ALPHA}|x{len(ex_h)}|{names}")
        if partial.get(key, {}).get("sig") == sig:
            print(f"[skip] {key} (already measured)", file=sys.stderr)
            return partial[key]
        if MAX_CONFIGS and n_new >= MAX_CONFIGS:
            print(f"[stop] per-invocation cap ({MAX_CONFIGS}) reached; rerun to "
                  f"continue from the checkpoint", file=sys.stderr)
            budget_cap_hit["v"] = True
            return None
        n_new += 1
        print(f"[run ] {key}: {label}", file=sys.stderr)

        h_recs = _run_arm(ev_h, priors, key + ":harm")
        b_recs = _run_arm(ev_b, priors, key + ":benign") if do_benign else []
        budget = 0.0
        if priors:
            budget = float(np.mean([measure_norm_budget(model, tok, p, priors, None)
                                    for p in ev_h[:N_BUDGET]]))
        cell = {
            "sig": sig, "label": label, "priors": [p.name for p in priors],
            "harmful": summarize(h_recs),
            "benign": summarize(b_recs) if do_benign else None,
            "norm_budget": budget,
            "examples": [{"prompt": r["prompt"], "response": r["response"],
                          "label": r["label"]} for r in h_recs[:3]],
            **(extra or {}),
        }
        partial[key] = cell
        PARTIAL_PATH.write_text(json.dumps(partial, indent=2), encoding="utf-8")
        hh = cell["harmful"]
        print(f"[done] {key}: refusal={hh['refusal_rate']} gibber={hh['gibberish_rate']} "
              f"budget={budget:.3f} ({time.time() - t0:.0f}s)", file=sys.stderr)
        return cell

    # R0 + L0 are measured whatever the arm selection (the sweep needs A-alone and
    # the unsteered reference just as much as the ladder does).
    n_cfg_h = 2 + (LADDER_K if "ladder" in ARMS else 0) \
        + (2 * len(COS_GRID) if "sweep" in ARMS else 0)
    n_cfg_b = 2 + (LADDER_K if "ladder" in ARMS else 0)
    est_h, est_b = n_cfg_h * len(ev_h), n_cfg_b * len(ev_b)
    print(f"[plan ] arms={ARMS}  up to ~{est_h} harmful + ~{est_b} benign generations "
          f"(resumable; NORTHO_MAX_CONFIGS caps one invocation)", file=sys.stderr)

    # --- 5. Reference cells (both arms need them) ----------------------------
    measure("R0", "unsteered reference", [], do_benign=True,
            extra={"added": None, "cos_to_refusal": None,
                   "predicted_budget": 0.0, "max_abs_cos_within_stack": None})
    g0 = gram_cosines([v_unit])
    measure("L0", "ladder base [A]", [A], do_benign=True,
            extra={"added": "A", "cos_to_refusal": 1.0,
                   "predicted_budget": predicted_budget(g0, C.STACK_ALPHA),
                   "max_abs_cos_within_stack": 0.0})

    # --- 6. ARM 2: the revertible, budget-limited additive ladder -------------
    order = ["L0"]
    admissions: list[dict] = []
    if "ladder" in ARMS:
        kept_idx: list[int] = []
        for i, w in enumerate(ladder_dirs, start=1):
            if budget_cap_hit["v"]:
                break
            key = f"L{i}"
            # LIVE admission, against what the ladder is ACTUALLY carrying. A
            # rung that was dropped gave its budget back, so re-testing here is
            # what makes "dropped, not carried" true of the budget as well as of
            # the priors.
            kept_vecs = [ladder_dirs[j - 1] for j in kept_idx]
            pf = admit_direction(w, [v_unit] + kept_vecs, C.STACK_ALPHA,
                                 BUDGET_CEILING)
            pf["candidate"] = f"w{i}"
            pf["against"] = ["v_refusal"] + [f"w{j}" for j in kept_idx]
            pf["planned"] = preflight[i - 1]["admit"]
            admissions.append(pf)
            if not pf["admit"]:
                print(f"[refused] {key}: {pf['reason']} "
                      f"(max|cos|={pf['max_abs_cos']:.3f}, predicted budget "
                      f"{pf['predicted_budget_after']:.3f} vs ceiling "
                      f"{BUDGET_CEILING}) against {pf['against']}", file=sys.stderr)
                if pf["reason"] == "BUDGET_EXCEEDED":
                    # Every candidate in this family carries the same cosine to
                    # the refusal axis and to its siblings, so no later one is
                    # cheaper: the budget is spent and the ladder stops.
                    break
                continue
            stack_vecs = [v_unit] + kept_vecs + [w]
            g = gram_cosines(stack_vecs)
            priors = ([A]
                      + [Prior(f"w{j}@L{primary}", ladder_dirs[j - 1], primary,
                               C.STACK_ALPHA, "relative_add") for j in kept_idx]
                      + [Prior(f"w{i}@L{primary}", w, primary, C.STACK_ALPHA,
                               "relative_add")])
            cell = measure(
                key, f"[A]+{len(kept_idx)} kept + w{i}", priors, do_benign=True,
                extra={"added": f"w{i}",
                       "cos_to_refusal": float(np.dot(w.astype(np.float64),
                                                      v_unit.astype(np.float64))),
                       "predicted_budget": predicted_budget(g, C.STACK_ALPHA),
                       "max_abs_cos_within_stack": max_abs_offdiag(g),
                       "admission": pf})
            if cell is None:
                break
            order.append(key)
            state = replay_ladder(partial, order,
                                  {"budget_ceiling": BUDGET_CEILING,
                                   "coherence_tol": COHERENCE_TOL,
                                   "compete_tol": COMPETE_TOL,
                                   "selectivity_tol": SELECTIVITY_TOL})
            last = state["rows"][-1]
            if last["verdict"] == "KEEP":
                kept_idx.append(i)
                print(f"[KEEP ] {key} (kept so far: {kept_idx})", file=sys.stderr)
            else:
                print(f"[DROP ] {key} on {'+'.join(last['failed_rules'])} -> reverted; "
                      f"next candidate is judged against {state['kept'][-1]}",
                      file=sys.stderr)
            if state["stopped_at"]:
                print(f"[stop ] ladder stopped at {state['stopped_at']} "
                      f"({state['binding_constraint']})", file=sys.stderr)
                break

    # --- 7. ARM 1: the cosine sweep -----------------------------------------
    if "sweep" in ARMS:
        for idx, fam in enumerate(sweep_dirs):
            if budget_cap_hit["v"]:
                break
            w = fam["vector"]
            g = gram_cosines([v_unit, w])
            Wp = Prior(f"W(cos={fam['cos_target']:.2f})@L{primary}", w, primary,
                       C.STACK_ALPHA, "relative_add")
            common = {"cos_target": fam["cos_target"],
                      "cos_to_refusal": fam["cos_measured"],
                      "max_abs_cos_within_stack": max_abs_offdiag(g)}
            measure(f"SW_PAIR_{idx}", f"[A, W(cos={fam['cos_target']:.2f})]",
                    [A, Wp], do_benign=False,
                    extra={**common, "added": f"W({fam['cos_target']:.2f})",
                           "predicted_budget": predicted_budget(g, C.STACK_ALPHA)})
            measure(f"SW_SOLO_{idx}", f"W(cos={fam['cos_target']:.2f}) alone",
                    [Wp], do_benign=False,
                    extra={**common, "added": f"W({fam['cos_target']:.2f})",
                           "predicted_budget": predicted_budget(
                               gram_cosines([w]), C.STACK_ALPHA)})

    # --- 8. Assemble. Only cells measured at THIS run's size may enter --------
    ladder_state = apply_preflight_stop(
        replay_ladder(partial, order,
                      {"budget_ceiling": BUDGET_CEILING,
                       "coherence_tol": COHERENCE_TOL,
                       "compete_tol": COMPETE_TOL,
                       "selectivity_tol": SELECTIVITY_TOL}),
        admissions)
    results = {
        "meta": {
            "model_id": C.MODEL_ID, "primary_layer": int(primary),
            "stack_alpha": float(C.STACK_ALPHA), "near_cos": NEAR_COS,
            "ladder_k": LADDER_K, "cos_grid": COS_GRID,
            "budget_ceiling": BUDGET_CEILING,
            "near_ortho_max_cos": NEAR_ORTHO_MAX_COS,
            "coherence_tol": COHERENCE_TOL, "compete_tol": COMPETE_TOL,
            "selectivity_tol": SELECTIVITY_TOL,
            "arms": ARMS, "n_budget_prompts": N_BUDGET, "seed": int(C.SEED),
            "max_new_tokens": int(C.MAX_NEW_TOKENS),
            "directions_stamp": stamp,
            # PROVENANCE (metadata only -- changes no metric). A run that fell
            # back to self-judging lands here as is_self_judge=true and is
            # inadmissible as a headline (CLAUDE.md sec.17 rubric item 3).
            **judge.stamp(),
            "tier": ("SCREENING (single seed, one alpha; judge below its own 0.85 "
                     "AUC validity bar -- see ../JUDGE_VALIDITY.md)"),
            "clause": "CLAUDE.md sec.9 clause 3 (near-orthogonal directions stack "
                      "until the norm budget is spent)",
        },
        "data_floor": floor,
        "dataset_header": {k: header[k] for k in
                           ("n_harmful", "n_benign", "harmful_pool_after_topup",
                            "benign_pool", "per_source_counts_sampled",
                            "median_char_length") if k in header},
        "directions": {
            "cos_to_refusal": {f"w{i + 1}": float(np.dot(
                ladder_dirs[i].astype(np.float64), v_unit.astype(np.float64)))
                for i in range(LADDER_K)},
            "gram_ladder": gram_ladder.tolist(),
            "gram_ladder_labels": ["v_refusal"] + [f"w{i + 1}" for i in range(LADDER_K)],
            "max_abs_offdiag": max_abs_offdiag(gram_ladder),
            "sweep_cos_measured": [f["cos_measured"] for f in sweep_dirs],
            "note": ("w_i are ORTHOGONALITY CONTROLS built from activation "
                     "variance after deflating the refusal axis, not second "
                     "concepts -- this pool carries one labelled contrast."),
        },
        "preflight": preflight,
        "admissions": admissions,
        "configs": partial,
        "ladder": ladder_state,
        "sweep": sweep_table(partial),
        "stop_rules": ladder_state["stop_rules"],
    }
    results["notes"] = build_notes(results)

    # Persist BEFORE any summary print: a late UnicodeEncodeError on this host
    # must not cost the data.
    RESULTS_PATH.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"[save] {RESULTS_PATH}", file=sys.stderr)
    try:
        _plot(results, PLOT_PATH)
        print(f"[save] {PLOT_PATH}", file=sys.stderr)
    except Exception as e:                                       # pragma: no cover
        print(f"[warn] plot failed: {e}", file=sys.stderr)
    print(summary_text(results))
    DF.warn_if_below_floor(floor)
    return results


def rebuild_report() -> dict:
    """Re-derive ladder + sweep + notes from the saved cells. NO model, NO GPU.

    ``meta``, ``data_floor``, ``directions`` and ``preflight`` are carried over
    verbatim from the run that measured them, so a rebuilt report can never claim
    a provenance or a data floor it did not have.
    """
    if not RESULTS_PATH.exists():
        raise SystemExit(f"no {RESULTS_PATH} to rebuild from - run the measurement first")
    prev = json.loads(RESULTS_PATH.read_text(encoding="utf-8"))
    cfg = prev["configs"]
    order = [k for k in ["L0"] + [f"L{i}" for i in range(1, prev["meta"]["ladder_k"] + 1)]
             if k in cfg]
    m = prev["meta"]
    state = apply_preflight_stop(replay_ladder(cfg, order, {
        "budget_ceiling": m["budget_ceiling"], "coherence_tol": m["coherence_tol"],
        "compete_tol": m["compete_tol"], "selectivity_tol": m["selectivity_tol"]}),
        prev.get("admissions"))
    out = {**prev, "ladder": state, "sweep": sweep_table(cfg),
           "stop_rules": state["stop_rules"]}
    out["notes"] = build_notes(out)
    RESULTS_PATH.write_text(json.dumps(out, indent=2), encoding="utf-8")
    try:
        _plot(out, PLOT_PATH)
    except Exception as e:                                       # pragma: no cover
        print(f"[warn] plot failed: {e}", file=sys.stderr)
    print(summary_text(out))
    return out


# --------------------------------------------------------------------------- #
# CPU self-test — no model, no GPU, no network. Exercises every pure helper on
# the schema the runner actually writes, so a broken report is caught before a
# GPU hour is spent producing one.
# Run: python -m steering_tutorials.stacking.run_near_orthogonal --selftest
# --------------------------------------------------------------------------- #
def _self_test() -> None:
    # (a) summarize counts skips instead of defaulting them to a verdict.
    s = summarize([{"label": "REFUSAL", "p_refusal": 0.8},
                   {"label": "GIBBERISH", "p_refusal": 0.2},
                   {"label": None, "p_refusal": None}])
    assert s["n"] == 2 and s["n_judge_skipped"] == 1
    assert abs(s["refusal_rate"] - 0.5) < 1e-9
    assert summarize([])["refusal_rate"] is None

    def cell(ref, gib=0.4, ben=0.1, budget=0.10, **kw):
        return {"harmful": {"refusal_rate": ref, "gibberish_rate": gib,
                            "refusal_score": ref},
                "benign": {"refusal_rate": ben, "gibberish_rate": 0.2},
                "norm_budget": budget, **kw}

    # (b) the sweep table reads both criteria and orders by MEASURED cosine.
    cfg = {
        "R0": cell(0.30), "L0": cell(0.20),
        "SW_PAIR_0": {**cell(0.28), "cos_target": 0.0, "cos_to_refusal": 0.0,
                      "predicted_budget": 0.113},
        "SW_SOLO_0": {**cell(0.22), "cos_target": 0.0, "cos_to_refusal": 0.0},
        "SW_PAIR_1": {**cell(0.05), "cos_target": 0.9, "cos_to_refusal": 0.9,
                      "predicted_budget": 0.156},
        "SW_SOLO_1": {**cell(0.18), "cos_target": 0.9, "cos_to_refusal": 0.9},
    }
    sw = sweep_table(cfg)
    assert [r["cos_measured"] for r in sw] == [0.0, 0.9]
    r0 = sw[0]
    assert r0["best_constituent"] == "W" and abs(r0["vs_best_constituent"] - 0.06) < 1e-9
    assert r0["stacks_vs_best_constituent"] is True
    #   W alone gains -0.08 vs unsteered; inside the pair it delivers -0.15 -> competes.
    r1 = sw[1]
    assert r1["stacks_vs_best_constituent"] is False
    assert r1["competes_marginal_vs_standalone"] is True

    # (c) a sweep cell whose solo half was never measured degrades to None.
    cfg_missing = {"R0": cell(0.3), "L0": cell(0.2),
                   "SW_PAIR_9": {**cell(0.1), "cos_target": 0.5,
                                 "cos_to_refusal": 0.5}}
    row = sweep_table(cfg_missing)[0]
    assert row["standalone_refusal"] is None
    assert row["competes_marginal_vs_standalone"] is None
    assert row["best_constituent"] == "A"      # only A survives as a constituent

    # (d) notes are derived from the cells, and never invented.
    res = {"meta": {"ladder_k": 2}, "configs": {"R0": cell(0.3, gib=0.5)},
           "ladder": replay_ladder(
               {"L0": cell(0.20, gib=0.40),
                "L1": cell(0.05, gib=0.70, budget=0.14)},
               ["L0", "L1"], {"budget_ceiling": 0.20}),
           "sweep": sw}
    res["admissions"] = [{"candidate": "w2", "admit": False,
                          "reason": "BUDGET_EXCEEDED", "max_abs_cos": 0.2,
                          "max_cos_bar": 0.35, "predicted_budget_before": 0.187,
                          "predicted_budget_after": 0.213, "budget_ceiling": 0.20,
                          "against": ["v_refusal", "w1"]}]
    # (d1) a pre-flight budget refusal becomes the binding constraint, instead of
    #      the ladder reporting "nothing stopped it".
    assert "NONE" in res["ladder"]["binding_constraint"]
    apply_preflight_stop(res["ladder"], res["admissions"])
    assert res["ladder"]["preflight_stop"] == "w2"
    assert res["ladder"]["binding_constraint"].startswith("BUDGET (at PRE-FLIGHT")
    #      ...and a measured stop is never overwritten by it.
    stopped = {"stopped_at": "L1", "binding_constraint": "BUDGET"}
    assert apply_preflight_stop(stopped, res["admissions"])["binding_constraint"] \
        == "BUDGET" and "preflight_stop" not in stopped

    notes = build_notes(res)
    assert any("PRE-FLIGHT REFUSAL of w2" in n for n in notes), notes
    assert any("DROPPED" in n for n in notes), notes
    assert any("SUBSTRATE CAVEAT" in n for n in notes), notes
    assert any("COSINE SWEEP" in n for n in notes), notes

    # (e) summary_text renders without a model and stays pure ASCII (this host's
    #     cp1252 console kills any non-ascii character in a runnable script).
    fake = {
        "meta": {"model_id": "m", "primary_layer": 12, "stack_alpha": 0.08,
                 "near_cos": 0.2, "ladder_k": 2, "budget_ceiling": 0.2,
                 "near_ortho_max_cos": 0.35, "judge_id": "qwen"},
        "data_floor": DF.floor_report(500, 500, DF.plan_split(792, 300, 500),
                                      500, 500),
        "preflight": [{"candidate": "w1", "max_abs_cos": 0.04,
                       "predicted_budget_before": 0.08,
                       "predicted_budget_after": 0.12,
                       "orthogonal_capacity_left": 3, "admit": True,
                       "reason": None}],
        "ladder": res["ladder"], "sweep": sw, "notes": notes,
    }
    txt = summary_text(fake)
    txt.encode("ascii")                       # raises if a unicode char slipped in
    assert "NEAR-ORTHOGONAL ARM" in txt and "COSINE SWEEP" in txt

    print("[self-test] OK - summarize counts skips; sweep table applies both "
          "criteria and degrades missing cells to None; notes derive from cells; "
          "summary_text renders pure ASCII.")


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        _self_test()
    elif "--report" in sys.argv:
        rebuild_report()
    else:
        main()
