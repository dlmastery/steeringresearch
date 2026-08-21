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

EVERY RATE CARRIES ITS DENOMINATOR
----------------------------------
A rate with no ``n`` beside it is not a measurement, it is a decoration: 0.333
could be 1-of-3 or 15-of-45 and only one of those is readable. So each rung
records the exact denominator behind each rate, and the header records the
concept pool sizes, the achieved per-concept eval ``n``, whether the pool capped
us below the CLAUDE.md sec.17 rubric-1 target, and the judge stamp that says what
actually graded these generations. ``main()`` REFUSES to start if any concept's
held-out split is under ``config.MIN_EVAL_PER_CONCEPT``.

Note the shape of ``success``: it is the UNWEIGHTED MEAN of the K active
concepts' per-concept refusal rates, so it has no single denominator. We report
the per-concept ``n`` and rate that went into it, plus their sum as
``n_items``, and never present the mean as if it came from one pooled sample.

RESULTS SCHEMA (kept in sync with README)
-----------------------------------------
{
  "judge_id": str, "is_self_judge": bool,      # from Judge.stamp() -- provenance
  "judge_model_id": str, "off_family": bool,
  "seed": int,
  "model_id": str, "steer_layer": int, "concepts": [str, ...],
  "per_concept_alpha": float,
  "eval_n": {                                  # the denominators, up front
      "floor_per_concept": int,                # hard gate main() enforces
      "target_per_class": int,                 # rubric-1 standard (500)
      "requested_per_concept": int,            # config.N_PER_CONCEPT
      "pool_capped": bool,                     # pool < target => True
      "min_achieved_eval_n": int,
      "per_concept": {name: {"pool_available": int, "extract_n": int,
                             "achieved_n": int, "pool_capped": bool}, ...},
      "note": str},
  "cosine_matrix": {"labels": [...], "matrix": [[...]]},   # raw-dir overlap
  "ladder": [
      {"k": int, "active": [str,...], "inactive": [str,...],
       "n": {"per_concept": {name: int}, "n_items": int,
             "crosstalk_concept": str|None, "crosstalk_n": int|None,
             "success_is_unweighted_mean_of_per_concept_rates": true},
       "raw":  {"success": float, "crosstalk": float, "gibberish": float,
                "budget": float, "n_items": int, "n_crosstalk": int|None,
                "per_concept": {name: {"n": int, "success": float,
                                       "gibberish": float}}},
       "ortho":{... same keys ...}}, ...
  ],
  "examples": [ {"concept": str, "k": int, "arm": "raw"|"ortho",
                 "prompt": str, "response": str, "verdict": str}, ... ],
  "examples_cap": int,          # examples is a SAMPLE, never a denominator
  "plots": {"success_vs_k": "success_vs_k.png"}
}
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

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
# Run-size knob. The full ladder is 3 rungs x 2 arms x (active + cross-talk)
# concept slices = ~480 generations at MAX_NEW_TOKENS, each followed by a judge
# call -- more than one comfortable foreground window on a RAM-contended host.
# MULTI_INTENT_N_EVAL takes a labelled slice of that. The default IS the
# config, so an
# unset var reproduces the pre-registered run exactly.
#
# Note that it cannot be used to make the run arbitrarily cheap:
# ``assert_eval_floor`` still enforces MIN_EVAL_PER_CONCEPT (30), the sec.17
# rubric floor, and raises before any GPU work. That is deliberate -- the knob
# is for fitting a legitimate run into a window, not for shrinking it below the
# reportable floor.
# --------------------------------------------------------------------------- #
N_EVAL_PER_CONCEPT = int(
    os.environ.get("MULTI_INTENT_N_EVAL") or C.N_EVAL_PER_CONCEPT)

# A CAPPED run must not land on the FULL run's path. artifacts/results.json is
# the Jul-24 full-ladder artifact the README quotes; a 30-per-concept screening
# pass agrees with it on every other key and would replace it in place, leaving
# no way to tell which one is on disk. Keyed output, per CLAUDE.md 18.8.
_CAPPED = N_EVAL_PER_CONCEPT != C.N_EVAL_PER_CONCEPT
RESULTS_PATH: Path = (
    C.ARTIFACTS / f"results_n{N_EVAL_PER_CONCEPT}.json" if _CAPPED
    else C.RESULTS_PATH)
LADDER_PNG: Path = (
    C.ARTIFACTS / f"success_vs_k_n{N_EVAL_PER_CONCEPT}.png" if _CAPPED
    else C.LADDER_PNG)


# --------------------------------------------------------------------------- #
# Pure helpers (no model) — safe to unit-test in isolation.
# --------------------------------------------------------------------------- #
def _refusal_rate(verdicts: list[str]) -> float:
    """Fraction of REFUSAL verdicts (the steering-success signal).

    Raises on an empty list. The previous ``max(1, len(verdicts))`` guard turned
    "we graded nothing" into a confident 0.0 — a silent, plausible, wrong number,
    which is the exact failure mode CLAUDE.md sec.18.8 catalogues. An empty
    verdict list is a bug upstream and must surface as one.
    """
    if not verdicts:
        raise ValueError("no verdicts to rate: a rate needs a nonzero denominator")
    return verdicts.count("REFUSAL") / len(verdicts)


def _gibberish_rate(verdicts: list[str]) -> float:
    """Fraction of GIBBERISH verdicts. Raises on an empty list (see above)."""
    if not verdicts:
        raise ValueError("no verdicts to rate: a rate needs a nonzero denominator")
    return verdicts.count("GIBBERISH") / len(verdicts)


# --------------------------------------------------------------------------- #
# Denominator bookkeeping. Pure functions over the loaded prompt splits, so the
# floor is checked (and the header block built) BEFORE a GPU is touched.
# --------------------------------------------------------------------------- #
def build_eval_n_block(concept_splits: dict, requested: int, floor: int,
                       target_per_class: int) -> dict:
    """Summarise the real denominators for ``results.json``.

    ``concept_splits`` is ``data["concepts"]`` -- {name: {extract, eval,
    n_available}}. Returns the ``eval_n`` header described in the module
    docstring: what we asked for, what the pool could give, what we got.
    """
    per_concept = {}
    for name, split in concept_splits.items():
        pool = int(split.get("n_available", len(split["extract"]) + len(split["eval"])))
        per_concept[name] = {
            "pool_available": pool,
            "extract_n": len(split["extract"]),
            "achieved_n": len(split["eval"]),
            "pool_capped": pool < target_per_class,
        }
    achieved = [c["achieved_n"] for c in per_concept.values()]
    capped = any(c["pool_capped"] for c in per_concept.values())
    return {
        "floor_per_concept": int(floor),
        "target_per_class": int(target_per_class),
        "requested_per_concept": int(requested),
        "pool_capped": bool(capped),
        "min_achieved_eval_n": int(min(achieved)) if achieved else 0,
        "per_concept": per_concept,
        "note": (
            "Every rate in 'ladder' is count/achieved_n on a concept's held-out "
            "eval split. These toxic-chat harm concepts are POOL-LIMITED, so the "
            f"rubric-1 target of {target_per_class}/class is unreachable here; we "
            "request the whole pool and report what it gave. 'success' is the "
            "unweighted mean of the active concepts' rates, not a pooled rate."
        ),
    }


def assert_eval_floor(eval_n: dict) -> None:
    """Abort the run if any concept's eval split is under the hard floor.

    Fails LOUD and BEFORE the GPU work, because the alternative -- discovering it
    after four hours of generation -- is how a sub-floor rate ends up shipped
    "just this once".
    """
    floor = eval_n["floor_per_concept"]
    short = {n: c["achieved_n"] for n, c in eval_n["per_concept"].items()
             if c["achieved_n"] < floor}
    if short:
        raise RuntimeError(
            "ABORT: concept eval splits below the hard floor of "
            f"{floor}/concept: {short}. A per-concept rate built on fewer than "
            f"{floor} examples is not reportable (CLAUDE.md sec.17 rubric item "
            "2). Raise MULTI_INTENT_N_PER_CONCEPT, or drop the starved concept "
            "from config.CONCEPTS -- do not lower the floor."
        )


def warn_if_pool_capped(eval_n: dict) -> bool:
    """Print the pool ceiling explicitly. Returns True when capped.

    Rubric item 2 says a pool-limited lesson must SAY SO rather than quietly ship
    a small n. Saying so in the artifact is necessary; saying so on the console
    of the run that produces it is what stops the caveat getting lost.
    """
    if not eval_n["pool_capped"]:
        return False
    rows = ", ".join(
        f"{n}=pool {c['pool_available']}/eval {c['achieved_n']}"
        for n, c in eval_n["per_concept"].items()
    )
    print(
        "[run][POOL-CAPPED] the toxic-chat harm-concept pools are smaller than "
        f"the {eval_n['target_per_class']}/class rubric target, so this lesson "
        "reports the MAXIMUM the pool allows, not the standard: " + rows +
        ". Every rate below is SCREENING tier and is labelled pool_capped=true "
        "in results.json.",
        file=sys.stderr, flush=True,
    )
    return True


def summarize_ladder(ladder: list[dict]) -> str:
    """One-line-per-rung text table of the K-ladder (used by the console report)."""
    lines = [f"{'K':>2}  {'arm':<6} {'success':>8} {'n':>5} {'crosstalk':>10} "
             f"{'n_xt':>5} {'gibber':>7} {'budget':>7}"]
    for rung in ladder:
        for arm in ("raw", "ortho"):
            a = rung[arm]
            # n is printed beside every rate on purpose: a bare 0.33 in a console
            # log is the same defect as a bare 0.33 in the artifact.
            n_items = a.get("n_items", "?")
            n_xt = a.get("n_crosstalk")
            n_xt = "-" if n_xt is None else n_xt
            lines.append(f"{rung['k']:>2}  {arm:<6} {a['success']:>8.2f} "
                         f"{str(n_items):>5} {a['crosstalk']:>10.2f} "
                         f"{str(n_xt):>5} {a['gibberish']:>7.2f} "
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
    from steering_tutorials.common.judge_gate import (
        assert_publishable, require_off_family_judge)

    # JUDGE PROVENANCE FIRST -- before the data, before the model, before a
    # single token. Every headline here (per-concept success, cross-talk,
    # gibberish) is a judged refusal rate, so a self-judged run has nothing to
    # salvage.
    require_off_family_judge("multi_intent")

    # DATA FIRST, then the denominator gate, THEN the model. Loading prompts is
    # cheap and CPU-only; a run that cannot clear the floor should fail in
    # seconds, not after the weights are resident and the generations are spent.
    print("[run] loading data ...", file=sys.stderr)
    data = load_multi_intent(
        C.CONCEPTS,
        n_per_concept=C.N_PER_CONCEPT,
        n_eval_per_concept=N_EVAL_PER_CONCEPT,
        n_benign_baseline=C.N_BENIGN_BASELINE,
        seed=C.SEED,
    )
    concepts = list(data["concepts"])            # preserves config order
    baseline = data["baseline"]

    eval_n = build_eval_n_block(
        data["concepts"], requested=C.N_PER_CONCEPT,
        floor=C.MIN_EVAL_PER_CONCEPT, target_per_class=C.TARGET_PER_CLASS,
    )
    assert_eval_floor(eval_n)                    # raises before any GPU work
    warn_if_pool_capped(eval_n)

    print("[run] loading model ...", file=sys.stderr)
    model, tok = load_model(C.MODEL_ID)
    judge = Judge(model, tok)

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
            if len(examples) < C.EXAMPLES_CAP:
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

        # Denominators for this rung, known before a single token is generated.
        n_per_concept = {name: len(data["concepts"][name]["eval"])
                         for name in active}
        xt_concept = inactive[0] if inactive else None
        n_xt = (len(data["concepts"][xt_concept]["eval"]) if xt_concept else None)

        rung = {
            "k": k, "active": active, "inactive": inactive,
            "n": {
                "per_concept": n_per_concept,
                "n_items": int(sum(n_per_concept.values())),
                "crosstalk_concept": xt_concept,
                "crosstalk_n": n_xt,
                # Stated, not implied: 'success' averages RATES, so it has no one
                # denominator. n_items is the total graded, not the divisor.
                "success_is_unweighted_mean_of_per_concept_rates": True,
            },
        }
        for arm, vecs in (("raw", raw_k), ("ortho", ortho_k)):
            # success: mean refusal over the active concepts' eval prompts.
            succ, gib, per_concept = [], [], {}
            for name in active:
                vs = eval_concept(vecs, alphas, name, k, arm)
                s, g = _refusal_rate(vs), _gibberish_rate(vs)
                succ.append(s)
                gib.append(g)
                # The per-concept breakdown the mean is built from, each with the
                # n it was computed over -- so a reader can reconstruct the mean.
                per_concept[name] = {"n": len(vs), "success": s, "gibberish": g}
            # cross-talk: refusal on ONE inactive concept (if any remain).
            if inactive:
                xt = eval_concept(vecs, alphas, inactive[0], k, arm)
                crosstalk = _refusal_rate(xt)
                n_crosstalk = len(xt)
            else:
                crosstalk = float("nan")
                n_crosstalk = None
            rung[arm] = {
                "success": float(np.mean(succ)),
                "gibberish": float(np.mean(gib)),
                "crosstalk": crosstalk,
                "budget": norm_budget(vecs, alphas),
                "n_items": int(sum(c["n"] for c in per_concept.values())),
                "n_crosstalk": n_crosstalk,
                "per_concept": per_concept,
            }
        ladder.append(rung)
        print(f"[run] K={k} (n={rung['n']['n_items']} graded/arm)  "
              f"raw.success={rung['raw']['success']:.2f} "
              f"ortho.success={rung['ortho']['success']:.2f} "
              f"raw.gib={rung['raw']['gibberish']:.2f} "
              f"ortho.gib={rung['ortho']['gibberish']:.2f}", file=sys.stderr)

    # 3) REPORT. The JSON is written FIRST -- before the plot, before the summary
    #    table -- so a matplotlib failure or a cp1252 console blow-up on the last
    #    line still leaves every measured number on disk.
    results = {
        # PROVENANCE (metadata only -- changes no metric). judge.stamp() reports
        # what ACTUALLY graded these generations, taken from the judge object
        # rather than from the README or the env var. A run that fell back to
        # self-judging lands here as judge_id="self" / is_self_judge=true and is
        # therefore inadmissible as a headline (CLAUDE.md sec.17, rubric item 3).
        **judge.stamp(),
        "seed": int(C.SEED),
        "model_id": C.MODEL_ID,
        "steer_layer": C.STEER_LAYER,
        "concepts": concepts,
        "per_concept_alpha": C.PER_CONCEPT_ALPHA,
        # DENOMINATORS. Sits next to the provenance because they answer the same
        # question -- "is this rate readable?" -- and a rate missing either one
        # is not. See build_eval_n_block().
        "eval_n": eval_n,
        "cosine_matrix": {"labels": concepts, "matrix": cmat.tolist()},
        "ladder": ladder,
        "examples": examples,
        "examples_cap": int(C.EXAMPLES_CAP),
        "plots": {"success_vs_k": LADDER_PNG.name},
    }
    # Publish gate IN the write path (CLAUDE.md sec.18.8).
    assert_publishable(results, "multi_intent")
    RESULTS_PATH.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"[run] wrote {RESULTS_PATH}", file=sys.stderr)

    _plot_ladder(ladder, LADDER_PNG)
    # ASCII only below this line -- the Windows cp1252 console kills a run on a
    # stray alpha/Delta/norm-bar, and this is the last thing main() does.
    print("\n" + summarize_ladder(ladder))
    print(f"\njudge={results['judge_id']} (self_judge={results['is_self_judge']})  "
          f"seed={results['seed']}")
    print("eval n per concept: " + ", ".join(
        f"{name}={c['achieved_n']}/pool {c['pool_available']}"
        for name, c in eval_n["per_concept"].items()))
    if eval_n["pool_capped"]:
        print(f"POOL-CAPPED: below the {eval_n['target_per_class']}/class target; "
              "SCREENING tier only.")


# --------------------------------------------------------------------------- #
# CPU unit for the pure helpers — NO model. Run: python -m ...run_multi_intent
# --------------------------------------------------------------------------- #
def _self_test() -> None:
    assert _refusal_rate(["REFUSAL", "COMPLIANCE", "REFUSAL", "GIBBERISH"]) == 0.5
    assert _gibberish_rate(["GIBBERISH", "GIBBERISH", "REFUSAL", "COMPLIANCE"]) == 0.5

    # An empty verdict list must RAISE, not return a confident 0.0.
    for fn in (_refusal_rate, _gibberish_rate):
        try:
            fn([])
        except ValueError:
            pass
        else:                                    # pragma: no cover - regression
            raise AssertionError(f"{fn.__name__}([]) must raise, not return a rate")

    # --- denominator block + floor gate ---------------------------------- #
    # Fixture names are deliberately NOT real concept names, so the POOL-CAPPED
    # line this test provokes can never be mistaken for a measurement.
    splits = {
        "FIXTURE_big": {"extract": ["x"] * 272, "eval": ["e"] * 116,
                        "n_available": 388},
        "FIXTURE_mid": {"extract": ["x"] * 100, "eval": ["e"] * 43,
                        "n_available": 143},
        "FIXTURE_small": {"extract": ["x"] * 78, "eval": ["e"] * 33,
                          "n_available": 111},
    }
    blk = build_eval_n_block(splits, requested=500, floor=30, target_per_class=500)
    assert blk["min_achieved_eval_n"] == 33
    assert blk["pool_capped"] is True             # every pool is under 500
    assert blk["per_concept"]["FIXTURE_big"]["achieved_n"] == 116
    assert blk["per_concept"]["FIXTURE_big"]["pool_available"] == 388
    assert_eval_floor(blk)                        # 33 >= 30 -> passes
    assert warn_if_pool_capped(blk) is True

    starved = {"FIXTURE_tiny": {"extract": ["x"] * 60, "eval": ["e"] * 29,
                                "n_available": 89}}
    try:
        assert_eval_floor(build_eval_n_block(starved, requested=500, floor=30,
                                             target_per_class=500))
    except RuntimeError as exc:
        assert "29" in str(exc) and "floor" in str(exc)
    else:                                         # pragma: no cover - regression
        raise AssertionError("a 29-item eval split must ABORT the run, not ship")

    # --- summary table prints n beside every rate ------------------------ #
    fake = [{"k": 1,
             "n": {"per_concept": {"sexual": 116}, "n_items": 116,
                   "crosstalk_concept": "harassment", "crosstalk_n": 43},
             "raw": {"success": 1.0, "crosstalk": 0.0, "gibberish": 0.0,
                     "budget": 0.06, "n_items": 116, "n_crosstalk": 43,
                     "per_concept": {"sexual": {"n": 116, "success": 1.0,
                                                "gibberish": 0.0}}},
             "ortho": {"success": 1.0, "crosstalk": 0.0, "gibberish": 0.0,
                       "budget": 0.06, "n_items": 116, "n_crosstalk": 43,
                       "per_concept": {"sexual": {"n": 116, "success": 1.0,
                                                  "gibberish": 0.0}}}}]
    table = summarize_ladder(fake)
    assert "success" in table and "budget" in table
    assert "116" in table and "43" in table, "every rate must print its n"
    print("[self-test] OK - rate helpers, denominator block, floor gate and "
          "ladder summary behave as expected.")


def _preflight() -> None:
    """CPU-only: load the prompt splits, print the denominators, run the floor gate.

    The cheap half of ``main()`` -- everything up to the model load. Run this
    before booking the GPU: it answers "what n will this run actually report?"
    in seconds and fails loudly if a concept is under the floor.
    """
    from .data import load_multi_intent

    data = load_multi_intent(
        C.CONCEPTS,
        n_per_concept=C.N_PER_CONCEPT,
        n_eval_per_concept=N_EVAL_PER_CONCEPT,
        n_benign_baseline=C.N_BENIGN_BASELINE,
        seed=C.SEED,
    )
    eval_n = build_eval_n_block(
        data["concepts"], requested=C.N_PER_CONCEPT,
        floor=C.MIN_EVAL_PER_CONCEPT, target_per_class=C.TARGET_PER_CLASS,
    )
    assert_eval_floor(eval_n)
    warn_if_pool_capped(eval_n)
    for name, c in eval_n["per_concept"].items():
        print(f"[preflight] {name:12s} pool={c['pool_available']:4d} "
              f"extract={c['extract_n']:4d} eval_n={c['achieved_n']:4d}")
    print(f"[preflight] floor={eval_n['floor_per_concept']} "
          f"min_achieved={eval_n['min_achieved_eval_n']} "
          f"pool_capped={eval_n['pool_capped']} -- OK")

    # What the judge WILL be, resolved from the env without loading anything.
    from steering_tutorials.hello_world_steering.judge import resolve_judge_id
    jid = resolve_judge_id()
    print(f"[preflight] judge={jid}"
          + ("  <-- SELF-JUDGE: set STEER_JUDGE_MODEL=Qwen/Qwen2.5-3B-Instruct"
             " before a reportable run" if jid == "self" else "  (off-family)"))


if __name__ == "__main__":
    # Bare `python -m ...run_multi_intent` is the CPU-only unit test the README
    # advertises -- it must never load a model. `--preflight` adds the (still
    # CPU-only) data + denominator check; `--run` is the opt-in GPU experiment.
    _self_test()
    if "--preflight" in sys.argv:
        _preflight()
    if "--run" in sys.argv:
        main()
