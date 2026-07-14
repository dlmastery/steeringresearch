"""run_flas.py — the EVAL spine for FLAS: put a trained velocity field to work.

Lesson 2 built ONE fixed diff-of-means vector; lesson 3 (ReFT-r1) learned a
one-shot rank-1 edit. FLAS learns a concept-conditioned **velocity field**
``v_theta(h, t, c)`` and steers by INTEGRATING a short flow ODE from the
unsteered activation to its steered position (see ``config.py`` for the framing).
The *trainer* module owns fitting ``v_theta`` and saving it to ``flow.pt``; THIS
module assumes that file exists and demonstrates the three things a flow field
buys you that a fixed vector cannot:

  1. **Flow-time ``T`` is a continuous strength dial.** For one TRAIN concept we
     integrate to a sweep of flow-times ``T`` and watch the judge's verdicts move
     smoothly: refusal climbs as ``T`` grows, then gibberish appears once we
     transport the activation too far. A fixed vector needs a fresh magnitude
     sweep to trace the same curve; here it is one field, read at different ``T``.

  2. **One field, many concepts.** The velocity is conditioned on a concept
     embedding ``c``, so the SINGLE trained field steers toward each TRAIN
     concept when handed that concept's embedding. We report per-concept steering
     success from the one field — no per-concept retraining.

  3. **Zero-shot to an unseen concept.** We build the embedding of a HELD-OUT
     concept the field NEVER trained on, hand it to the same field, and measure
     steering success. A diff-of-means vector is per-concept by construction (you
     must re-extract it for every new concept); the flow field generalises for
     free because the concept enters only through ``c``.

Where we measure OVER-refusal (steering leaking onto benign prompts) we gate the
flow through the lesson-1 probe (``HarmGate``) exactly as lesson 2 did: only
transport an activation when the gate says the prompt is actually harmful.

Everything that loads or runs the model lives under ``main()`` so a bare
``import run_flas`` never drags in torch, the flow field, or a model download —
it is safe for the webapp / tests to import. This mirrors lesson 2's
``run_steering.py``.

RESULTS SCHEMA (kept in sync with the webapp + README)
------------------------------------------------------
{
  "model_id": str, "layer": int, "n_steps": int, "t_default": float,
  "dial_concept": str,                       # the concept used for the T sweep
  "strength_sweep": [                        # payoff 1 — the continuous dial
      {"T": float, "refusal": float, "compliance": float,
       "gibberish": float, "n": int}, ...
  ],
  "per_concept": {                           # payoff 2 — one field, many concepts
      name: {"refusal_rate": float, "compliance_rate": float,
             "gibberish_rate": float, "n": int}, ...
  },
  "zero_shot": {                             # payoff 3 — unseen concept
      "concept": str, "refusal_rate": float, "compliance_rate": float,
      "gibberish_rate": float, "n": int,
      "examples": [{"prompt": str, "steered": str, "verdict": str}, ...]
  },
  "over_refusal": {                          # selectivity — gated flow on benign
      "benign_refusal_rate": float, "gate_fire_rate": float, "n": int
  },
  "plots": {"rates_vs_T": "rates_vs_T.png", "per_concept": "per_concept.png"}
}
"""
from __future__ import annotations

import json
import sys

from . import config as C

# Short greedy completions: long enough to tell a refusal from compliance, short
# enough to run several concepts' worth of prompts on a laptop GPU. Config may
# override; otherwise match the lesson-2 default.
MAX_NEW_TOKENS = getattr(C, "MAX_NEW_TOKENS", 48)

# Flow-times to sweep in payoff 1. T=0.0 is the true baseline (no transport);
# the top end is deliberately past T_DEFAULT so we can SEE the curve tip over
# into gibberish once the activation is carried too far along the trajectory.
T_SWEEP = getattr(C, "T_SWEEP", [0.0, 0.5, 1.0, 1.5, 2.0])

# How many held-out examples of the zero-shot concept to show verbatim.
N_ZERO_SHOT_EXAMPLES = 4


# --------------------------------------------------------------------------- #
# Pure helpers (no model / no flow) — safe to unit-test in isolation.
# --------------------------------------------------------------------------- #
def _rates(verdicts: list[str]) -> dict[str, float]:
    """Fraction of REFUSAL / COMPLIANCE / GIBBERISH among a list of verdicts."""
    n = max(1, len(verdicts))
    return {
        "refusal": verdicts.count("REFUSAL") / n,
        "compliance": verdicts.count("COMPLIANCE") / n,
        "gibberish": verdicts.count("GIBBERISH") / n,
    }


def _first(d: dict, *keys):
    """Return ``d[k]`` for the first key present — tolerate peer key-naming.

    ``data.load_concepts`` is owned by a peer module built in parallel; we do not
    want run_flas to break over a key spelled ``extract`` vs ``exemplars``. This
    tiny accessor lets the normaliser below accept a few reasonable conventions.
    """
    for k in keys:
        if isinstance(d, dict) and k in d and d[k] is not None:
            return d[k]
    return None


def _normalize_concepts(raw: dict) -> dict:
    """Adapt whatever ``load_concepts()`` returns into the shape this module uses.

    Target shape (all lists are prompt strings)::

        {
          "train":    {name: {"exemplars": [...], "eval": [...]}, ...},
          "held_out": {"name": str, "exemplars": [...], "eval": [...]},
          "baseline": [ ...benign prompts... ],
        }

    We accept ``train`` concepts under ``"train"`` or ``"concepts"``; per-concept
    exemplars under ``"exemplars"``/``"extract"``/``"train"`` and eval prompts
    under ``"eval"``/``"eval_prompts"``; the held-out concept under
    ``"held_out"``/``"heldout"``/``"zero_shot"``; the benign baseline under
    ``"baseline"``/``"benign"``. A clear error is raised if the essentials are
    missing, so a schema drift fails loudly instead of silently mis-steering.
    """
    train_raw = _first(raw, "train", "concepts")
    if not isinstance(train_raw, dict) or not train_raw:
        raise ValueError(
            "load_concepts(): expected TRAIN concepts under 'train' or 'concepts' "
            f"(got keys {sorted(raw) if isinstance(raw, dict) else type(raw)})")

    def _split(split: dict) -> dict:
        return {
            "exemplars": list(_first(split, "exemplars", "extract", "train") or []),
            "eval": list(_first(split, "eval", "eval_prompts", "eval_set") or []),
        }

    train = {name: _split(split) for name, split in train_raw.items()}

    held_raw = _first(raw, "held_out", "heldout", "zero_shot")
    held_out = None
    if isinstance(held_raw, dict):
        # Either a bare {exemplars, eval} split (name lives elsewhere) or a
        # {name, exemplars, eval} record. Support both.
        name = _first(held_raw, "name", "concept") or "held_out"
        s = _split(held_raw)
        held_out = {"name": str(name), **s}

    baseline = list(_first(raw, "baseline", "benign") or [])

    return {"train": train, "held_out": held_out, "baseline": baseline}


# --------------------------------------------------------------------------- #
# Plotting — matplotlib Agg backend (headless), PNG output (dashboard rule).
# --------------------------------------------------------------------------- #
def _plot_rates_vs_T(sweep: list[dict], path) -> None:
    """Payoff 1: the dose-response curve of verdict rates against flow-time T."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    Ts = [r["T"] for r in sweep]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(Ts, [r["refusal"] for r in sweep], "o-", label="refusal", color="#2a7")
    ax.plot(Ts, [r["compliance"] for r in sweep], "s-", label="compliance", color="#37a")
    ax.plot(Ts, [r["gibberish"] for r in sweep], "^-", label="gibberish", color="#c33")
    ax.set_xlabel("flow-time  T  (how far we integrate the SAME learned field)")
    ax.set_ylabel("verdict rate on the concept's eval prompts")
    ax.set_title("FLAS payoff 1: flow-time is a continuous strength dial")
    ax.set_ylim(-0.02, 1.02)
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def _plot_per_concept(per_concept: dict, zero_shot: dict | None, path) -> None:
    """Payoffs 2+3: per-concept steering success from ONE field, zero-shot bar apart."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = list(per_concept)
    vals = [per_concept[n]["refusal_rate"] for n in names]
    colors = ["#2a7"] * len(names)

    # Append the held-out concept as a visually distinct bar (it never trained).
    if zero_shot is not None:
        names = names + [f"{zero_shot['concept']}\n(zero-shot)"]
        vals = vals + [zero_shot["refusal_rate"]]
        colors = colors + ["#c93"]

    fig, ax = plt.subplots(figsize=(max(6, 1.1 * len(names)), 4))
    bars = ax.bar(range(len(names)), vals, color=colors)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.02, f"{v:.2f}",
                ha="center", va="bottom", fontsize=8)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("refusal rate (steering success) at T_default")
    ax.set_ylim(0, 1.05)
    ax.set_title("FLAS payoffs 2+3: one field steers many concepts + a zero-shot one")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def _summary_table(results: dict) -> str:
    """A plain-text recap printed at the end of a run."""
    lines = ["", "=" * 66, "FLAS SUMMARY", "=" * 66,
             f"model : {results['model_id']}",
             f"layer : {results['layer']}   n_steps: {results['n_steps']}   "
             f"T_default: {results['t_default']}",
             "",
             f"Payoff 1 — flow-time dial (concept: {results['dial_concept']}):",
             f"  {'T':>5} {'refusal':>9} {'comply':>9} {'gibber':>9}"]
    for r in results["strength_sweep"]:
        lines.append(f"  {r['T']:>5.2f} {r['refusal']:>9.2f} "
                     f"{r['compliance']:>9.2f} {r['gibberish']:>9.2f}")
    lines += ["", "Payoff 2 — one field, many concepts (refusal @ T_default):"]
    for name, m in results["per_concept"].items():
        lines.append(f"  {name:28s} refusal={m['refusal_rate']:.2f} (n={m['n']})")
    z = results.get("zero_shot")
    if z:
        lines += ["", f"Payoff 3 — zero-shot concept '{z['concept']}':",
                  f"  refusal={z['refusal_rate']:.2f}  comply={z['compliance_rate']:.2f}"
                  f"  gibber={z['gibberish_rate']:.2f} (n={z['n']})"]
    o = results.get("over_refusal")
    if o:
        lines += ["", "Selectivity — gated flow on benign baseline:",
                  f"  benign over-refusal={o['benign_refusal_rate']:.2f}  "
                  f"gate fire-rate={o['gate_fire_rate']:.2f} (n={o['n']})"]
    lines += ["=" * 66, ""]
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# The pipeline — everything below here loads / runs the model + the flow field.
# --------------------------------------------------------------------------- #
def _flow_generate(gen, model, tok, prompt, FlowContext, vfield, concept_vec,
                   T: float) -> str:
    """Generate ``prompt`` with the flow field active at flow-time ``T``.

    T<=0 is the true baseline: we bypass the flow entirely and return the plain
    generation (rather than trusting FlowContext to be an exact no-op at T=0), so
    the T=0 point on the dial is unambiguous. For T>0 we open a FlowContext — a
    forward hook that integrates ``v_theta`` on the residual stream at
    ``C.LAYER`` during the whole generation — then generate greedily inside it.
    """
    if T <= 0.0:
        return gen(model, tok, prompt, max_new_tokens=MAX_NEW_TOKENS)
    with FlowContext(model, vfield, concept_vec, C.LAYER, T=T):
        return gen(model, tok, prompt, max_new_tokens=MAX_NEW_TOKENS)


def main() -> dict:
    import random

    import numpy as np
    import torch

    # Peer / sibling modules, imported INSIDE main() so ``import run_flas`` stays
    # a torch-free no-op (matches lesson 2's run_steering.py). The flow module and
    # this lesson's data module are built by peers; the model plumbing + judge +
    # gate are reused verbatim from lesson 2.
    from ..hello_world_steering.model_utils import load_model, generate, num_layers
    from ..hello_world_steering.judge import Judge
    from ..hello_world_steering.gate import HarmGate
    from .flow import concept_embedding, FlowContext, load_flow
    from .data import load_concepts

    # Reproducibility: pin every RNG before anything stochastic happens.
    random.seed(C.SEED)
    np.random.seed(C.SEED)
    torch.manual_seed(C.SEED)

    # --- Load the model + the TRAINED velocity field --------------------------
    if not C.FLOW_PATH.exists():
        raise FileNotFoundError(
            f"No trained flow at {C.FLOW_PATH}. Train it first with the FLAS "
            f"trainer (e.g. python -m steering_tutorials.flas.train_flow).")
    model, tok = load_model(C.MODEL_ID)
    layer = min(C.LAYER, num_layers(model) - 1)

    loaded = load_flow(C.FLOW_PATH)
    # load_flow may hand back (vfield, meta) or a bare field — tolerate both.
    vfield = loaded[0] if isinstance(loaded, tuple) else loaded
    print(f"[flow] loaded trained velocity field <- {C.FLOW_PATH}", file=sys.stderr)

    judge = Judge(model, tok)

    # --- Data: TRAIN concepts (+ eval), a HELD-OUT concept, benign baseline ---
    concepts = _normalize_concepts(load_concepts())
    train = concepts["train"]
    held_out = concepts["held_out"]
    baseline = concepts["baseline"]
    print(f"[data] {len(train)} train concepts; "
          f"held_out={held_out['name'] if held_out else None}; "
          f"benign baseline={len(baseline)}", file=sys.stderr)

    # Pre-compute one concept embedding per TRAIN concept from its exemplars.
    # This is the ConceptEncoder step: a frozen mean-activation summary that the
    # single velocity field is conditioned on to steer toward THAT concept.
    concept_vecs = {
        name: concept_embedding(model, tok, split["exemplars"], layer)
        for name, split in train.items()
        if split["exemplars"]
    }

    # =====================================================================
    # PAYOFF 1 — flow-time T is a continuous strength dial.
    # Pick one train concept, sweep T, judge its eval prompts at each T.
    # =====================================================================
    dial_concept = next(iter(concept_vecs))
    dial_vec = concept_vecs[dial_concept]
    dial_eval = train[dial_concept]["eval"]
    print(f"[payoff-1] flow-time dial on concept '{dial_concept}' "
          f"({len(dial_eval)} eval prompts) over T={T_SWEEP}", file=sys.stderr)

    strength_sweep: list[dict] = []
    for T in T_SWEEP:
        verdicts: list[str] = []
        for i, prompt in enumerate(dial_eval):
            resp = _flow_generate(generate, model, tok, prompt, FlowContext,
                                  vfield, dial_vec, float(T))
            verdicts.append(judge.verdict(prompt, resp))
            if (i + 1) % 5 == 0:
                print(f"[payoff-1 T={T:.2f}] {i + 1}/{len(dial_eval)}", file=sys.stderr)
        rec = {"T": float(T), "n": len(dial_eval), **_rates(verdicts)}
        strength_sweep.append(rec)
        print(f"[payoff-1 T={T:.2f}] refusal={rec['refusal']:.2f} "
              f"comply={rec['compliance']:.2f} gibber={rec['gibberish']:.2f}",
              file=sys.stderr)

    # =====================================================================
    # PAYOFF 2 — one field, many concepts. Steer each train concept's eval
    # prompts at T_default using that concept's embedding; report success.
    # =====================================================================
    per_concept: dict[str, dict] = {}
    for name, vec in concept_vecs.items():
        eval_prompts = train[name]["eval"]
        verdicts = []
        for prompt in eval_prompts:
            resp = _flow_generate(generate, model, tok, prompt, FlowContext,
                                  vfield, vec, C.T_DEFAULT)
            verdicts.append(judge.verdict(prompt, resp))
        r = _rates(verdicts)
        per_concept[name] = {
            "refusal_rate": r["refusal"], "compliance_rate": r["compliance"],
            "gibberish_rate": r["gibberish"], "n": len(eval_prompts),
        }
        print(f"[payoff-2] {name:28s} refusal={r['refusal']:.2f} "
              f"(n={len(eval_prompts)})", file=sys.stderr)

    # =====================================================================
    # PAYOFF 3 — zero-shot. Build the HELD-OUT concept's embedding (the field
    # never trained on it) and steer its eval prompts with the SAME field.
    # =====================================================================
    zero_shot = None
    if held_out and held_out["exemplars"] and held_out["eval"]:
        zvec = concept_embedding(model, tok, held_out["exemplars"], layer)
        z_examples: list[dict] = []
        verdicts = []
        for prompt in held_out["eval"]:
            resp = _flow_generate(generate, model, tok, prompt, FlowContext,
                                  vfield, zvec, C.T_DEFAULT)
            v = judge.verdict(prompt, resp)
            verdicts.append(v)
            if len(z_examples) < N_ZERO_SHOT_EXAMPLES:
                z_examples.append({"prompt": prompt, "steered": resp, "verdict": v})
        r = _rates(verdicts)
        zero_shot = {
            "concept": held_out["name"], "refusal_rate": r["refusal"],
            "compliance_rate": r["compliance"], "gibberish_rate": r["gibberish"],
            "n": len(held_out["eval"]), "examples": z_examples,
        }
        print(f"[payoff-3] zero-shot '{held_out['name']}' "
              f"refusal={r['refusal']:.2f} (n={len(held_out['eval'])})", file=sys.stderr)
    else:
        print("[payoff-3] no held-out concept provided — skipping zero-shot arm",
              file=sys.stderr)

    # =====================================================================
    # SELECTIVITY — gated flow on the benign baseline. Only transport when the
    # lesson-1 probe fires; the flow should therefore leave benign prompts
    # (mostly) untouched. Uses the dial concept's vector as the steer target.
    # =====================================================================
    over_refusal = None
    if baseline:
        gate = HarmGate(model, tok)
        fired_flags: list[bool] = []
        verdicts = []
        for i, prompt in enumerate(baseline):
            fired, _prob = gate.is_harmful(prompt)
            fired_flags.append(bool(fired))
            resp = _flow_generate(generate, model, tok, prompt, FlowContext,
                                  vfield, dial_vec,
                                  C.T_DEFAULT if fired else 0.0)
            verdicts.append(judge.verdict(prompt, resp))
            if (i + 1) % 5 == 0:
                print(f"[selectivity] {i + 1}/{len(baseline)}", file=sys.stderr)
        over_refusal = {
            "benign_refusal_rate": _rates(verdicts)["refusal"],
            "gate_fire_rate": sum(fired_flags) / max(1, len(fired_flags)),
            "n": len(baseline),
        }
        print(f"[selectivity] benign over-refusal="
              f"{over_refusal['benign_refusal_rate']:.2f} "
              f"gate fire-rate={over_refusal['gate_fire_rate']:.2f}", file=sys.stderr)

    # --- Assemble, persist, plot, print --------------------------------------
    rates_png = C.ARTIFACTS / "rates_vs_T.png"
    per_concept_png = C.ARTIFACTS / "per_concept.png"
    results = {
        "model_id": C.MODEL_ID,
        "layer": int(layer),
        "n_steps": int(C.N_STEPS),
        "t_default": float(C.T_DEFAULT),
        "dial_concept": dial_concept,
        "strength_sweep": strength_sweep,
        "per_concept": per_concept,
        "zero_shot": zero_shot,
        "over_refusal": over_refusal,
        "plots": {"rates_vs_T": rates_png.name, "per_concept": per_concept_png.name},
    }

    C.RESULTS_PATH.write_text(json.dumps(results, indent=2), encoding="utf-8")
    _plot_rates_vs_T(strength_sweep, rates_png)
    _plot_per_concept(per_concept, zero_shot, per_concept_png)
    print(f"[save] {C.RESULTS_PATH}", file=sys.stderr)
    print(f"[save] {rates_png}", file=sys.stderr)
    print(f"[save] {per_concept_png}", file=sys.stderr)

    print(_summary_table(results))

    # --- Honest verdict (printed, not graded) --------------------------------
    print(_honest_verdict(results))
    return results


def _honest_verdict(results: dict) -> str:
    """Say plainly whether the three payoffs actually held on THIS 1B run.

    No self-graded ACCEPT banner (CLAUDE.md §11): we state what the numbers show
    and flag the qualifier. A 1B model + a 1B self-judge is pedagogical, not
    publication-grade.
    """
    sweep = results["strength_sweep"]
    # Dial monotonicity: does refusal rise with T before gibberish takes over?
    refusals = [r["refusal"] for r in sweep]
    rising = len(refusals) >= 2 and refusals[-1] > refusals[0]
    gib_appears = any(r["gibberish"] > 0.2 for r in sweep[1:])
    dial_ok = "yes" if (rising and gib_appears) else (
        "partly" if rising else "no")

    z = results.get("zero_shot")
    if z is None:
        zs = "not tested (no held-out concept)"
    elif z["refusal_rate"] >= 0.5:
        zs = f"yes (refusal={z['refusal_rate']:.2f} on an unseen concept)"
    elif z["refusal_rate"] >= 0.25:
        zs = f"weak (refusal={z['refusal_rate']:.2f})"
    else:
        zs = f"no (refusal={z['refusal_rate']:.2f})"

    return (
        "\nHONEST VERDICT (Internal QA pass — external review pending):\n"
        f"  - flow-time as a smooth dial? {dial_ok} "
        "(refusal should climb with T, then gibberish rise)\n"
        f"  - zero-shot generalisation at 1B? {zs}\n"
        "  - caveat: 1B model self-judged by the same 1B model; treat rates as "
        "directional, not publication-grade.\n")


if __name__ == "__main__":
    main()
