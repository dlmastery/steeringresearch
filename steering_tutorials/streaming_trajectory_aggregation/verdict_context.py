"""verdict_context.py -- the REPORTED CONTEXT that sits beside a pre-registered verdict.

WHY THIS FILE EXISTS
--------------------
README section 8 names three defects in how this lesson's falsifiers are read, and all
three are defects of CONTEXT, not of the registered logic:

  (e) F1, F2 and F3 never consult `confound.binding_bar`. Only F4 does. So a method can
      "hold" a falsifier while sitting BELOW the bar that prices content/length shortcuts
      -- 0.9039 on SHADE, 0.9960 on AgentDojo, where almost nothing clears it. A reader
      who sees "F3 HOLDS" and nothing else will read it as "this beats content". It does
      not say that and never did.
  (e, second half) F0's shuffle runs INSIDE `common.confound`, on bag-of-words features.
      It certifies the confound module, not the embedding-to-aggregator ladder. No
      shuffle control was ever run on the ladder itself.
  (f) The 10,000 bootstrap resamples never enter a verdict. Every falsifier is an
      UNPAIRED point-estimate compared to a 0.02 threshold, even though
      `AggregatorResult.scores` is index-aligned across methods and a paired CI is one
      loop away -- a loop `horizon.py` already wrote.
  (f, second half) F3 takes the best of 8 methods against `last_step` with no
      multiple-comparison correction. A max over 8 is a selection, and an uncorrected
      selection is biased upward.

THE ONE RULE THIS MODULE OBEYS
-------------------------------
**It never changes a verdict.** `config.FALSIFIERS` is the pre-registration; its text is
verbatim and its holds/fails arithmetic in `run_sta._falsifier_verdicts` is untouched.
Everything here is ADDITIONAL reporting attached alongside, under its own keys, so that:

  * "F3 HOLDS" keeps meaning exactly what it was registered to mean, and
  * "F3 HOLDS *and its best method sits 0.0 above / below the content bar, and the
    selection does not survive Holm*" is visible in the same object.

Rewriting the criterion to match what we later understood would destroy the audit trail
-- the same reason `config.FALSIFIER_ADDENDA` exists rather than an edit to F2's text.
`attach_context()` ASSERTS that no registered field moved (CLAUDE.md section 18.8,
"assert your anchors"): if attaching context would alter a `holds` or an `applicable`,
it raises instead of shipping a quietly-rewritten verdict.

WHAT IS REUSED, NOT REWRITTEN
------------------------------
  * the paired bootstrap  -> `horizon.paired_delta_ci` (the SAME helper the within-corpus
                             horizon control uses; a second implementation of a paired
                             bootstrap in one lesson is how two numbers under one name
                             start disagreeing)
  * the bar pricing       -> `steering_tutorials.common.confound.margin_over_bar`
  * Holm-Bonferroni       -> `steering.stats.holm_bonferroni` (src/steering/stats.py).
                             `steering` is not pip-installed on this host, so `_holm()`
                             puts `src/` on sys.path and imports the REAL one. If that
                             fails it RAISES -- it never falls back to a local copy.
  * AUC                   -> `horizon.fast_auc` (asserted equal to sklearn on every run)
  * the CV / ladder       -> `evaluate.evaluate_offline`, unchanged, for the shuffle arm

CPU-only. Loads NO model, downloads nothing. ASCII stdout only (Windows cp1252).

    python -m steering_tutorials.streaming_trajectory_aggregation.verdict_context
        -> the synthetic self-test (no corpus, no model, seconds)
"""
from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path
from typing import Sequence

import numpy as np

from steering_tutorials.common.confound import margin_over_bar
from steering_tutorials.streaming_trajectory_aggregation import config as C
from steering_tutorials.streaming_trajectory_aggregation import evaluate
from steering_tutorials.streaming_trajectory_aggregation.horizon import (
    fast_auc, paired_delta_ci,
)
from steering_tutorials.streaming_trajectory_aggregation.types import Corpus

__all__ = [
    "REGISTERED_MARGIN",
    "bar_context",
    "bar_context_many",
    "paired_margin",
    "holm_vs_reference",
    "ladder_shuffle_control",
    "build_verdict_context",
    "attach_context",
]

# The threshold the falsifiers were registered against (config.FALSIFIERS: "margin >
# 0.02"). Repeated here ONLY so the reported context can say whether a CI clears the same
# number the point estimate was compared to. It is not a new criterion.
REGISTERED_MARGIN = 0.02

CONTEXT_DISCLAIMER = (
    "REPORTED CONTEXT, NOT A VERDICT. The pre-registered holds/fails logic in "
    "config.FALSIFIERS and run_sta._falsifier_verdicts is unchanged and unaffected by "
    "anything in this block. It is here because a bare HOLDS was being read as a claim "
    "the falsifier never made (README section 8(e),(f))."
)


def _eprint(*args) -> None:
    msg = " ".join(str(a) for a in args)
    try:
        print(msg, file=sys.stderr)
    except Exception:
        print(msg.encode("ascii", "replace").decode("ascii"), file=sys.stderr)


def _num(x):
    """JSON-safe float (NaN -> None), matching run_sta.py / horizon.py convention."""
    if x is None:
        return None
    try:
        x = float(x)
    except (TypeError, ValueError):
        return None
    return None if (x != x) else x


def _finite(x) -> bool:
    return isinstance(x, float) and x == x


# ======================================================================================
# Holm -- imported from the repo's own implementation, never re-derived here
# ======================================================================================
def _holm():
    """-> (holm_bonferroni, provenance_string). RAISES if the real one cannot be found.

    `steering` (src/steering) is not pip-installed on this host, so the plain import
    fails and `src/` is added to sys.path. A silent local reimplementation would be a
    second Holm in a repo that already has one -- exactly the drift this lesson's own
    audits keep finding -- so the failure path raises.
    """
    try:
        from steering.stats import holm_bonferroni

        return holm_bonferroni, "steering.stats.holm_bonferroni"
    except Exception:
        src = Path(__file__).resolve().parents[2] / "src"
        if src.is_dir() and str(src) not in sys.path:
            sys.path.insert(0, str(src))
        from steering.stats import holm_bonferroni  # raises if genuinely unavailable

        return holm_bonferroni, "steering.stats.holm_bonferroni (src/ added to sys.path)"


# ======================================================================================
# 1. THE CONFOUND BAR, ATTACHED TO A VERDICT  (README section 8(e))
# ======================================================================================
def bar_context(method: str, auc, confound: dict, auc_ci=None) -> dict:
    """Price ONE method's AUC against `confound.binding_bar`.

    Delegates to `common.confound.margin_over_bar` -- the same function every other
    lesson prices with -- and adds the CI reading, because "the point estimate is above
    the bar" and "the interval is above the bar" are different claims and the lesson
    has been making the first while sounding like the second.
    """
    if not _finite(_num(auc)):
        return {"method": method, "auc": None, "clears": None,
                "reason": "method AUC is missing or NaN in this run"}
    out = margin_over_bar(float(auc), confound)
    out["method"] = method
    # `margin_over_bar` names the method's own AUC `method_auc`; mirror it as `auc` so a
    # bar cell reads the same way as every other AUC-carrying dict in results.json.
    out["auc"] = out["method_auc"]
    out = {k: (_num(v) if isinstance(v, float) else v) for k, v in out.items()}
    lo = hi = None
    if auc_ci is not None and len(auc_ci) == 2:
        lo, hi = _num(auc_ci[0]), _num(auc_ci[1])
    out["auc_ci"] = [lo, hi]
    out["ci_lo_clears_bar"] = bool(_finite(lo) and lo > out["binding_bar"])
    out["ci_straddles_bar"] = bool(_finite(lo) and _finite(hi)
                                   and lo <= out["binding_bar"] <= hi)
    return out


def bar_context_many(aucs: dict, confound: dict, methods: Sequence[str] = None,
                     cis: dict = None) -> dict:
    """`bar_context` over several methods + the best-of summary a verdict needs.

    `best_clears_bar` is the field README section 8(e) asks for: it makes "F3 HOLDS"
    unreadable as "this beats content" when the best method does not, in fact, beat it.
    """
    methods = list(methods if methods is not None else aucs.keys())
    cis = cis or {}
    per = {m: bar_context(m, aucs.get(m), confound, cis.get(m)) for m in methods}
    scored = {m: v for m, v in per.items() if _finite(v.get("auc"))}
    best = max(scored, key=lambda m: scored[m]["auc"]) if scored else None
    return {
        "binding_bar": _num(confound.get("binding_bar")),
        "binding_bar_name": confound.get("binding_bar_name"),
        "per_method": per,
        "best_method": best,
        "best_auc": (scored[best]["auc"] if best else None),
        "best_clears_bar": (bool(scored[best]["clears"]) if best else None),
        "n_methods_clearing_bar": int(sum(1 for v in scored.values() if v["clears"])),
        "n_methods_considered": len(scored),
        "note": CONTEXT_DISCLAIMER,
    }


# ======================================================================================
# 2. THE PAIRED BOOTSTRAP, ATTACHED TO A VERDICT  (README section 8(f), first half)
# ======================================================================================
def paired_margin(labels, scores_by_method: dict, hi: str, lo: str,
                  n: int = None, seed: int = 0,
                  registered_margin: float = REGISTERED_MARGIN) -> dict:
    """95% PAIRED bootstrap CI on AUC[hi] - AUC[lo], the margin a falsifier thresholds.

    Straight delegation to `horizon.paired_delta_ci` -- the pairing is legitimate for
    exactly the reason that helper documents: both score vectors are out-of-fold scores
    for the SAME trajectories in the SAME order, so per-trajectory difficulty cancels.

    The added fields are readings of the interval against the number the registered
    verdict used: `ci_lo_above_registered_margin` is the strong form of the F1/F3 claim
    ("the whole interval clears 0.02"), `ci_excludes_zero` the weak one ("there is a gap
    at all"). A verdict can hold on the point estimate with neither being true.
    """
    n = C.VERDICT_BOOTSTRAP if n is None else n
    for name in (hi, lo):
        if name not in scores_by_method:
            return {"applicable": False,
                    "reason": "method %r absent from this run's scores" % name,
                    "hi": hi, "lo": lo}
    d = paired_delta_ci(np.asarray(labels), np.asarray(scores_by_method[hi]),
                        np.asarray(scores_by_method[lo]), n=n, seed=seed)
    ci_lo, ci_hi = (d.get("ci") or [None, None])
    out = dict(d)
    out.update({
        "applicable": True,
        "hi": hi,
        "lo": lo,
        "statistic": "AUC[%s] - AUC[%s], paired percentile bootstrap" % (hi, lo),
        "registered_margin": float(registered_margin),
        "ci_lo_above_registered_margin": bool(_finite(_num(ci_lo))
                                              and float(ci_lo) > registered_margin),
        "ci_straddles_registered_margin": bool(
            _finite(_num(ci_lo)) and _finite(_num(ci_hi))
            and float(ci_lo) <= registered_margin <= float(ci_hi)),
        # The reading an F2-style "the margin is SMALL" verdict most needs, and the one a
        # point estimate cannot express: the gap is REAL (interval clear of zero) and the
        # whole interval still sits under the registered threshold. That is a much
        # stronger statement than "0.0084 < 0.02" -- and it is not the same as "no gap".
        "gap_excludes_zero_but_ci_entirely_under_registered_margin": bool(
            d.get("excludes_zero") and _finite(_num(ci_lo)) and _finite(_num(ci_hi))
            and float(ci_lo) > 0.0 and float(ci_hi) < registered_margin),
        "note": CONTEXT_DISCLAIMER,
    })
    return out


# ======================================================================================
# 3. HOLM ACROSS THE MAX-OVER-8 SELECTION  (README section 8(f), second half)
# ======================================================================================
def holm_vs_reference(labels, scores_by_method: dict, reference: str = "last_step",
                      methods: Sequence[str] = None, n: int = None, seed: int = 0,
                      alpha: float = 0.05,
                      registered_margin: float = REGISTERED_MARGIN) -> dict:
    """Holm-Bonferroni over EVERY method-vs-`reference` comparison in the ladder.

    F3 as registered takes the max over the ladder and compares it to `last_step`. That
    is a selection over m candidates reported at the significance of ONE, which inflates
    it. Here every comparison in the family gets a two-sided paired-bootstrap p-value
    (`horizon.paired_delta_ci`'s `p_two_sided`) and the family goes through
    `steering.stats.holm_bonferroni`.

    FAMILY SIZE, STATED PLAINLY. The selection ranges over all 8 ladder rows, but one of
    them IS the reference: its margin is identically 0 and its comparison carries no
    information, so the testable family is the other m = 7. Counting it would inflate m
    for free. Both readings are reported -- `family_size` (7, primary) and
    `including_reference_self_comparison` (8, strictly more conservative) -- so nobody
    has to take the choice on trust.

    A bootstrap p-value has a RESOLUTION FLOOR of 2/(B+1). If that floor exceeds Holm's
    tightest threshold alpha/m, no effect of any size could be rejected and the test is
    vacuous; that is checked and reported (`resolution_floor_blocks_rejection`), in the
    same spirit as `steering.stats.power_note`'s Holm-feasibility leg.
    """
    n = C.VERDICT_BOOTSTRAP if n is None else n
    holm_fn, holm_impl = _holm()
    all_methods = list(methods if methods is not None else scores_by_method.keys())
    if reference not in scores_by_method:
        return {"applicable": False,
                "reason": "reference method %r absent from this run's scores" % reference,
                "reference": reference}
    family_methods = [m for m in all_methods if m != reference and m in scores_by_method]
    if not family_methods:
        return {"applicable": False, "reason": "no non-reference methods to compare",
                "reference": reference}

    rows = []
    for m in family_methods:
        d = paired_delta_ci(np.asarray(labels), np.asarray(scores_by_method[m]),
                            np.asarray(scores_by_method[reference]), n=n, seed=seed)
        p = d.get("p_two_sided")
        rows.append({
            "method": m,
            "delta_vs_reference": d.get("delta"),
            "delta_ci": d.get("ci"),
            "ci_excludes_zero": d.get("excludes_zero"),
            "p_two_sided": p,
            "p_method": d.get("p_method"),
            "n_paired": d.get("n_paired"),
            "beats_registered_margin_pointwise": bool(_finite(_num(d.get("delta")))
                                                      and float(d["delta"]) > registered_margin),
            # a comparison the bootstrap could not evaluate (e.g. every resample
            # single-class) enters the family at p=1 rather than being dropped: dropping
            # it would shrink m and make the SURVIVING comparisons easier to reject
            "p_missing_treated_as_1": not _finite(_num(p)),
            "unevaluable_reason": d.get("reason"),
        })

    pvals = [(r["p_two_sided"] if _finite(_num(r["p_two_sided"])) else 1.0) for r in rows]
    h = holm_fn(pvals, alpha=alpha)
    for r, rej, adj in zip(rows, h["reject"], h["adjusted"]):
        r["holm_reject"] = bool(rej)
        r["holm_adjusted_p"] = _num(adj)
        r["uncorrected_reject"] = bool(_finite(_num(r["p_two_sided"]))
                                       and float(r["p_two_sided"]) <= alpha)
        r["holm_changes_this_row"] = bool(r["uncorrected_reject"] and not r["holm_reject"])

    # the m = 8 reading: the reference's own comparison, p = 1 by construction
    h8 = holm_fn(pvals + [1.0], alpha=alpha)
    conservative = [{"method": m, "holm_reject": bool(rj), "holm_adjusted_p": _num(ad)}
                    for m, rj, ad in zip(family_methods + [reference], h8["reject"], h8["adjusted"])]

    scored = [r for r in rows if _finite(_num(r["delta_vs_reference"]))]
    selected = max(scored, key=lambda r: r["delta_vs_reference"]) if scored else None
    m_fam = len(rows)
    floor = 2.0 / (float(n) + 1.0)
    tightest = alpha / max(1, m_fam)

    return {
        "applicable": True,
        "reference": reference,
        "family_definition": (
            "every ladder method vs %r, one paired-bootstrap comparison each. The F3 "
            "selection ranges over %d ladder rows, but the reference's own row has a "
            "margin of identically 0 and is excluded from the testable family, so "
            "m = %d." % (reference, m_fam + 1, m_fam)),
        "family_size": m_fam,
        "alpha": float(alpha),
        "holm_tightest_threshold": _num(tightest),
        "holm_impl": holm_impl,
        "bootstrap_resamples": int(n),
        "p_resolution_floor": _num(floor),
        "resolution_floor_blocks_rejection": bool(floor > tightest),
        "family": rows,
        "n_uncorrected_reject": int(sum(1 for r in rows if r["uncorrected_reject"])),
        "n_holm_reject": int(sum(1 for r in rows if r["holm_reject"])),
        "n_rows_flipped_by_holm": int(sum(1 for r in rows if r["holm_changes_this_row"])),
        "holm_changes_the_picture": bool(any(r["holm_changes_this_row"] for r in rows)),
        "selected_definition": (
            "`selected_*` is the argmax of delta-vs-reference ACROSS THE FAMILY -- the "
            "row an uncorrected max-over-the-ladder would pick. It need not be the "
            "highest-AUC row: when the reference itself is the best method on the corpus, "
            "every family delta is negative and `selected_method` is merely the "
            "least-negative one. Compare it with `bar.best_method`, which is the plain "
            "AUC argmax over the whole ladder including the reference."),
        "selected_method": (selected["method"] if selected else None),
        "selected_delta": (selected["delta_vs_reference"] if selected else None),
        "selected_p_two_sided": (selected["p_two_sided"] if selected else None),
        "selected_holm_adjusted_p": (selected["holm_adjusted_p"] if selected else None),
        "selected_survives_holm": (bool(selected["holm_reject"]) if selected else None),
        "including_reference_self_comparison": {
            "family_size": m_fam + 1,
            "note": "strictly more conservative; reported so the m=7 choice is auditable",
            "rows": conservative,
        },
        "note": CONTEXT_DISCLAIMER,
    }


# ======================================================================================
# 4. A SHUFFLE CONTROL ON THE LADDER ITSELF  (README section 8(e), second half)
# ======================================================================================
def _permuted_corpus(corpus: Corpus, rng: np.random.Generator, tag: str) -> Corpus:
    """A copy of `corpus` with the trajectory labels PERMUTED.

    `pool_fingerprint` is suffixed rather than recomputed, mirroring
    `horizon._subset_corpus`: a permuted artifact must never be mistakable for the real
    corpus it came from.

    The permutation is per-TRAJECTORY, matching what `common.confound.shuffle_control`
    does for F0, so the two shuffle numbers mean the same thing. Note what that implies:
    it destroys any label-group association as well as the label-feature one, so it tests
    "does this pipeline manufacture AUC from noise", not "is the signal only group
    identity". `label_pure_groups` in the returned dict is reported so a reader can see
    how much group-level structure the permutation dissolved.
    """
    labels = np.asarray([t.label for t in corpus.trajectories], dtype=int)
    perm = rng.permutation(len(labels))
    shuffled = labels[perm]
    trajs = [replace(t, label=int(y)) for t, y in zip(corpus.trajectories, shuffled)]
    return replace(corpus, name="%s_%s" % (corpus.name, tag), trajectories=trajs,
                   pool_fingerprint="%s#%s" % (corpus.pool_fingerprint, tag))


def ladder_shuffle_control(corpus: Corpus, embeddings: list, aggregator_factory,
                           methods: Sequence[str] = None, seed: int = None,
                           n_folds: int = None, bootstrap: int = None,
                           repeats: int = None, band=(0.45, 0.55)) -> dict:
    """Re-run the MAIN aggregators on PERMUTED labels. Every AUC must land near 0.5.

    F0 shuffles inside `common.confound`, on bag-of-words features -- it certifies the
    confound module and says nothing about the embedding-to-aggregator ladder, which is
    where the trained arms (`gru`, `query_token_compressor`) and the group-aware CV live
    and therefore where a leak would actually hide. This runs the same permutation
    through the real ladder.

    `aggregator_factory(name) -> Aggregator` must return a FRESH instance per call: an
    instance already fitted on the real labels must not be carried into a shuffled fit.

    This is REPORTED CONTEXT: it does not feed F0's registered verdict (whose subject is
    the confound bar), and a failure here is not an F0 failure. It is, however, the
    number that would invalidate the ladder, so it is printed and stored beside it.
    """
    methods = list(methods if methods is not None else C.LADDER_SHUFFLE_METHODS)
    seed = C.SEED if seed is None else seed
    n_folds = C.N_FOLDS if n_folds is None else n_folds
    bootstrap = C.LADDER_SHUFFLE_BOOTSTRAP if bootstrap is None else bootstrap
    repeats = C.LADDER_SHUFFLE_REPEATS if repeats is None else repeats

    groups = {}
    for t in corpus.trajectories:
        groups.setdefault(t.group_id, set()).add(int(t.label))
    n_pure = sum(1 for v in groups.values() if len(v) == 1)

    # HOW MUCH IS A BAND WORTH AT THIS n? A shuffled AUC is a random variable; under the
    # null its standard error is Hanley-McNeil sqrt((n_pos+n_neg+1)/(12*n_pos*n_neg)).
    # On a 500/500 corpus that is ~0.018, so [0.45,0.55] is a ~2.7-sigma check and means
    # something. On 60/60 it is ~0.053 and the same band is under one sigma -- it would
    # fail routinely on clean data. Reporting the SE (and the band in SE units) is what
    # stops `within_band=False` from being read as leakage when it is just a small corpus.
    n_pos = int(sum(1 for t in corpus.trajectories if t.label == 1))
    n_neg = len(corpus.trajectories) - n_pos
    null_se = (float(np.sqrt((n_pos + n_neg + 1.0) / (12.0 * max(1, n_pos) * max(1, n_neg))))
               if n_pos and n_neg else None)
    half_band = (float(band[1]) - float(band[0])) / 2.0

    out = {
        "methods_requested": methods,
        "n_permutations": int(repeats),
        "bootstrap": int(bootstrap),
        "n_folds": int(n_folds),
        "seed": int(seed),
        "band": [float(band[0]), float(band[1])],
        "null_auc_se_hanley_mcneil": _num(null_se),
        "band_half_width_in_null_se": (_num(half_band / null_se) if null_se else None),
        "band_is_meaningful_at_this_n": bool(null_se and (half_band / null_se) >= 2.0),
        "permutation_scheme": "per-trajectory label permutation (same scheme as "
                              "common.confound.shuffle_control, which F0 uses)",
        "label_pure_groups": {"n_pure": int(n_pure), "n_groups": len(groups)},
        "permutations": [],
        "note": ("A shuffle control on the LADDER, which F0 never covered (F0 shuffles "
                 "bag-of-words features inside common.confound). REPORTED CONTEXT: it "
                 "does not enter F0's registered verdict. An AUC outside the band here "
                 "means the embedding-to-aggregator pipeline manufactures separation "
                 "from permuted labels -- leakage, and every ladder number above it "
                 "would be uninterpretable."),
    }

    per_method_aucs: dict[str, list] = {m: [] for m in methods}
    for rep_i in range(int(repeats)):
        rng = np.random.default_rng(seed + 9973 + rep_i)
        shuffled = _permuted_corpus(corpus, rng, "shuffle%d" % rep_i)
        aggs, missing = [], []
        for m in methods:
            a = aggregator_factory(m)
            if a is None:
                missing.append(m)
            else:
                aggs.append(a)
        results = evaluate.evaluate_offline(shuffled, embeddings, aggs, seed=seed,
                                            n_folds=n_folds, bootstrap=bootstrap)
        cells = {}
        for r in results:
            auc = _num(r.auc)
            per_method_aucs.setdefault(r.method, []).append(auc)
            cells[r.method] = {
                "auc": auc,
                "auc_ci": [_num(r.auc_ci[0]), _num(r.auc_ci[1])],
                "within_band": bool(_finite(auc) and band[0] <= auc <= band[1]),
            }
            _eprint("[shuffle:ladder] perm=%d %-28s auc=%s within_band=%s"
                    % (rep_i, r.method, auc, cells[r.method]["within_band"]))
        out["permutations"].append({"permutation": rep_i, "missing_methods": missing,
                                     "cells": cells})

    summary = {}
    for m, vals in per_method_aucs.items():
        finite = [v for v in vals if _finite(v)]
        worst = max((abs(v - 0.5) for v in finite), default=None)
        summary[m] = {
            # A method the factory could not build produced no AUC at all. It is reported
            # as `no_result`, NOT as "outside the band": a missing number and a
            # chance-violating number mean opposite things and must not share a flag.
            "no_result": not bool(finite),
            "aucs": vals,
            "mean_auc": _num(np.mean(finite)) if finite else None,
            "max_abs_deviation_from_chance": _num(worst),
            # the same deviation in SE units -- the reading that survives a change of n
            "max_abs_deviation_in_null_se": (_num(worst / null_se)
                                             if (worst is not None and null_se) else None),
            "all_within_band": bool(finite) and all(band[0] <= v <= band[1] for v in finite),
        }
    out["per_method"] = summary
    scored = {m: v for m, v in summary.items() if not v["no_result"]}
    out["all_methods_within_band"] = bool(scored) and all(
        v["all_within_band"] for v in scored.values())
    out["methods_outside_band"] = [m for m, v in scored.items() if not v["all_within_band"]]
    out["methods_with_no_result"] = [m for m, v in summary.items() if v["no_result"]]
    out["worst_abs_deviation_in_null_se"] = _num(max(
        (v["max_abs_deviation_in_null_se"] for v in scored.values()
         if v["max_abs_deviation_in_null_se"] is not None), default=None))
    return out


# ======================================================================================
# 5. ASSEMBLY -- one context block per falsifier, attached without touching the verdict
# ======================================================================================
def build_verdict_context(corpus_name: str, aucs: dict, cis: dict, scores_by_method: dict,
                          labels, confound: dict, seed: int = 0, n: int = None,
                          ladder_shuffle: dict = None) -> dict:
    """-> {falsifier_tag: context_dict}. Pure reporting; computes nothing a verdict reads.

    F0 gets a pointer to the LADDER shuffle (its own registered subject is the confound
    module's bag-of-words shuffle, which is a different instrument).
    F1 gets the bar for mean/max/gru plus paired CIs on both margins it thresholds.
    F2 gets the same (its claim is that the margins are SMALL, which a CI can support or
    undermine just as sharply as it can a large one).
    F3 gets the bar for the whole ladder, the paired CI on best-minus-last_step, and the
    Holm block for the max-over-8 selection.
    """
    n = C.VERDICT_BOOTSTRAP if n is None else n
    labels = np.asarray(labels)
    ctx: dict = {}
    ladder = [m for m in aucs.keys()]

    f1_f2_methods = [m for m in ("mean_pool", "max_pool", "gru") if m in aucs]
    mean_max_gru_bar = bar_context_many(aucs, confound, f1_f2_methods, cis)
    paired_pair = {
        "max_pool_minus_mean_pool": paired_margin(labels, scores_by_method, "max_pool",
                                                   "mean_pool", n=n, seed=seed),
        "gru_minus_mean_pool": paired_margin(labels, scores_by_method, "gru",
                                              "mean_pool", n=n, seed=seed),
    }

    shared_note = (
        "The registered verdict is a POINT-ESTIMATE margin against 0.02 and consults no "
        "bar. Both readings it omits are here: whether the arms clear "
        "confound.binding_bar at all, and whether the 10,000-resample PAIRED bootstrap "
        "CI on the margin clears the same 0.02 the point estimate was compared to.")

    for tag, applicable_on in (("F1_mean_pool_collapses_long", "shade"),
                               ("F2_mean_pool_survives_short", "agentdojo")):
        ctx[tag] = {
            "applicable_on_this_corpus": bool(corpus_name == applicable_on),
            # flat aliases -- the two fields a reader (or a grep) most needs beside a
            # HOLDS, promoted out of the nested `bar` block so they cannot be missed
            "binding_bar": mean_max_gru_bar["binding_bar"],
            "binding_bar_name": mean_max_gru_bar["binding_bar_name"],
            "best_clears_bar": mean_max_gru_bar["best_clears_bar"],
            "bar": mean_max_gru_bar,
            "paired_margins": paired_pair,
            "reading": shared_note,
            "note": CONTEXT_DISCLAIMER,
        }

    f3_bar = bar_context_many(aucs, confound, ladder, cis)
    best_method = f3_bar.get("best_method")
    f3 = {
        "binding_bar": f3_bar["binding_bar"],
        "binding_bar_name": f3_bar["binding_bar_name"],
        "best_clears_bar": f3_bar["best_clears_bar"],
        "bar": f3_bar,
        "paired_best_minus_last_step": (
            paired_margin(labels, scores_by_method, best_method, "last_step", n=n, seed=seed)
            if best_method and best_method != "last_step" else
            {"applicable": False,
             "reason": ("the best ladder method IS last_step, so the F3 margin is "
                        "identically zero and no paired interval is defined")
                       if best_method == "last_step" else "no scored method"}),
        "holm": holm_vs_reference(labels, scores_by_method, reference="last_step",
                                  methods=ladder, n=n, seed=seed),
        "reading": (
            "F3's `best` is a MAX OVER THE LADDER: a selection reported at the "
            "significance of a single comparison. The Holm block prices that selection. "
            "The bar block answers the separate question the verdict never asked -- "
            "whether the winning method beats a model that never looked at trajectory "
            "structure at all."),
        "note": CONTEXT_DISCLAIMER,
    }
    ctx["F3_sequence_beats_laststep"] = f3

    if ladder_shuffle is not None:
        ctx["F0_shuffle_control"] = {
            "registered_subject": (
                "F0 as registered is the CONTENT bar re-run with permuted labels, i.e. a "
                "shuffle of common.confound's bag-of-words instrument. It certifies that "
                "instrument and nothing else."),
            "ladder_shuffle_control": ladder_shuffle,
            "reading": (
                "The block above is the shuffle F0 never ran: the same permutation pushed "
                "through the real embedding-to-aggregator ladder, where the trained arms "
                "and the group-aware CV live. It does NOT feed F0's verdict, and a "
                "failure here is not an F0 failure -- it is worse, because it would make "
                "every ladder AUC in this file uninterpretable."),
            "note": CONTEXT_DISCLAIMER,
        }
    return ctx


def attach_context(verdicts: dict, context_by_tag: dict) -> dict:
    """Attach `context` to each verdict IN PLACE, asserting no registered field moved.

    The assertion is the point. A context-attachment step that could silently alter a
    `holds` would be indistinguishable from rewriting the pre-registration, which is the
    one thing this module must never do -- so it is checked rather than intended
    (CLAUDE.md section 18.8: a replace that matches nothing must fail, not pass).
    """
    before = {k: (v.get("holds"), v.get("applicable")) for k, v in verdicts.items()}
    for tag, ctx in context_by_tag.items():
        if tag not in verdicts:
            _eprint("[verdict_context] WARNING: no verdict %r in this run; context for it "
                    "is dropped rather than invented" % tag)
            continue
        if "context" in verdicts[tag]:
            raise ValueError("verdict %r already carries a `context` key -- refusing to "
                             "overwrite it" % tag)
        verdicts[tag]["context"] = ctx
    after = {k: (v.get("holds"), v.get("applicable")) for k, v in verdicts.items()}
    if before != after:
        moved = [k for k in before if before[k] != after[k]]
        raise AssertionError("attaching context changed a REGISTERED verdict field on "
                             "%s -- refusing to ship a quietly-rewritten "
                             "pre-registration" % moved)
    return verdicts


# ======================================================================================
# 6. BACKFILL -- the same context for a run that already happened
# ======================================================================================
def _labels_for_results(results: dict, root: Path = None):
    """Recover the per-trajectory labels for an existing results.json, or raise.

    `results.json` stores per-method `scores` but no labels, so the labels come from the
    corpus disk cache in `artifacts/corpora/`. Two anchors, both required, because a
    silently misaligned label vector would produce confident wrong AUCs -- the exact
    shape of every defect in CLAUDE.md section 18.8:

      1. the cache's `uids_sha256` must EQUAL the run's `pool_fingerprint` (same pool),
      2. the caller re-derives every method's AUC from these labels and refuses unless it
         reproduces the stored AUC (same ORDER). Anchor 1 alone only proves set equality.
    """
    root = (C.ARTIFACTS if root is None else Path(root))
    fp = results.get("pool_fingerprint")
    corpus = results.get("corpus")
    cands = sorted((root / "corpora").glob("%s_*.json" % corpus))
    if not cands:
        raise FileNotFoundError("no corpus cache for %r under %s" % (corpus, root / "corpora"))
    import json as _json

    for path in cands:
        with open(path, "r", encoding="utf-8") as fh:
            obj = _json.load(fh)
        if obj.get("uids_sha256") != fp:
            continue
        labels = np.asarray([int(t["label"]) for t in obj["trajectories"]], dtype=int)
        return labels, str(path)
    raise ValueError("no cache under %s carries uids_sha256 == the run's pool_fingerprint "
                     "%r -- refusing to guess which pool produced these scores"
                     % (root / "corpora", fp))


def backfill_from_results(results_path, out_path=None, n: int = None, seed: int = None,
                          tol: float = 1e-9) -> dict:
    """Recompute the verdict context for a run ALREADY on disk, from its stored scores.

    Writes a SEPARATE `verdict_context_<corpus>.json` rather than editing the run record:
    a results.json is the run's own witness and a later process must not rewrite it in
    place. The output stamps `source_results` + `pool_fingerprint` so it can never be
    read beside a different run than the one it was computed from.

    The ladder shuffle is NOT part of a backfill -- it needs the embeddings and a real
    ladder re-run, not stored scores -- and is recorded as skipped with that reason
    rather than omitted.
    """
    import json as _json

    results_path = Path(results_path)
    n = C.VERDICT_BOOTSTRAP if n is None else n
    with open(results_path, "r", encoding="utf-8") as fh:
        results = _json.load(fh)
    seed = int(results.get("seed", 0)) if seed is None else seed

    labels, cache_path = _labels_for_results(results, root=results_path.parent)
    scores, aucs, cis, anchors = {}, {}, {}, []
    for row in results.get("main_ladder", []):
        s = np.asarray([np.nan if v is None else float(v) for v in row["scores"]],
                       dtype=np.float64)
        if s.size != labels.size:
            raise ValueError("method %r has %d scores but the cache has %d labels"
                             % (row["method"], s.size, labels.size))
        m = ~np.isnan(s)
        recomputed = fast_auc(labels[m], s[m])
        stored = row.get("auc")
        diff = abs(recomputed - float(stored)) if _finite(_num(stored)) else float("nan")
        anchors.append({"method": row["method"], "stored_auc": _num(stored),
                        "recomputed_auc": _num(recomputed), "abs_diff": _num(diff),
                        "matches": bool(diff < tol)})
        scores[row["method"]] = s
        aucs[row["method"]] = float(stored)
        cis[row["method"]] = tuple(row.get("auc_ci") or (None, None))

    bad = [a for a in anchors if not a["matches"]]
    if bad:
        raise AssertionError(
            "label alignment ANCHOR FAILED for %s -- the labels recovered from %s do not "
            "reproduce the stored AUCs, so they are not the labels these scores were "
            "measured against. Refusing to emit context computed from them."
            % ([a["method"] for a in bad], cache_path))

    ctx = build_verdict_context(results.get("corpus"), aucs, cis, scores, labels,
                                results.get("confound") or {}, seed=seed, n=n)
    out = {
        "generated_by": "verdict_context.backfill_from_results",
        "source_results": str(results_path),
        "source_corpus": results.get("corpus"),
        "source_pool_fingerprint": results.get("pool_fingerprint"),
        "label_source": cache_path,
        "seed": seed,
        "bootstrap": int(n),
        "label_alignment_anchor": {"tolerance": tol, "per_method": anchors,
                                   "all_match": True},
        "ladder_shuffle_control": {
            "skipped": True,
            "reason": ("a shuffle control re-runs the LADDER on permuted labels and needs "
                       "the step embeddings; stored scores cannot produce it. Run "
                       "run_sta.py with STA_LADDER_SHUFFLE=1 (the default) to get it.")},
        "context_by_falsifier": ctx,
        "note": ("REPORTED CONTEXT for a run that already happened. It does NOT edit the "
                 "run's own results.json and it does NOT change any verdict recorded "
                 "there; %s" % CONTEXT_DISCLAIMER),
    }
    out_path = (Path(out_path) if out_path is not None else
                results_path.parent / ("verdict_context_%s.json" % results.get("corpus")))
    with open(out_path, "w", encoding="utf-8") as fh:
        _json.dump(out, fh, indent=2)
    print("[backfill] wrote %s" % out_path)
    return out


def _print_backfill(out: dict) -> None:
    """ASCII summary of a backfill. Printed AFTER the JSON is written."""
    ctx = out["context_by_falsifier"]
    print("corpus=%s bootstrap=%d labels=%s" % (out["source_corpus"], out["bootstrap"],
                                                 out["label_source"]))
    print("label-alignment anchor: %d/%d methods reproduce their stored AUC to %g"
          % (sum(1 for a in out["label_alignment_anchor"]["per_method"] if a["matches"]),
             len(out["label_alignment_anchor"]["per_method"]),
             out["label_alignment_anchor"]["tolerance"]))
    for tag in ("F1_mean_pool_collapses_long", "F2_mean_pool_survives_short"):
        blk = ctx.get(tag) or {}
        if not blk.get("applicable_on_this_corpus"):
            continue
        bar = blk["bar"]
        print("")
        print("%s -- binding_bar=%.4f (%s)" % (tag, bar["binding_bar"], bar["binding_bar_name"]))
        for m, cell in bar["per_method"].items():
            print("  %-12s auc=%.4f margin_vs_bar=%+.4f clears_bar=%s"
                  % (m, cell["auc"], cell["margin"], cell["clears"]))
        print("  best_clears_bar=%s" % bar["best_clears_bar"])
        for name, pm in blk["paired_margins"].items():
            if pm.get("applicable"):
                print("  %-28s delta=%+.4f ci=[%+.4f,%+.4f] p=%.4g ci_lo>0.02=%s "
                      "real_but_under_0.02=%s"
                      % (name, pm["delta"], pm["ci"][0], pm["ci"][1], pm["p_two_sided"],
                         pm["ci_lo_above_registered_margin"],
                         pm["gap_excludes_zero_but_ci_entirely_under_registered_margin"]))
    f3 = ctx["F3_sequence_beats_laststep"]
    bar = f3["bar"]
    print("")
    print("F3_sequence_beats_laststep -- binding_bar=%.4f (%s)"
          % (bar["binding_bar"], bar["binding_bar_name"]))
    print("  best=%s auc=%.4f best_clears_bar=%s (%d/%d ladder methods clear it)"
          % (bar["best_method"], bar["best_auc"], bar["best_clears_bar"],
             bar["n_methods_clearing_bar"], bar["n_methods_considered"]))
    pm = f3["paired_best_minus_last_step"]
    if pm.get("applicable"):
        print("  paired best-minus-last_step delta=%+.4f ci=[%+.4f,%+.4f] p=%.4g "
              "ci_lo>0.02=%s" % (pm["delta"], pm["ci"][0], pm["ci"][1], pm["p_two_sided"],
                                  pm["ci_lo_above_registered_margin"]))
    else:
        print("  paired best-minus-last_step: N/A (%s)" % pm.get("reason"))
    h = f3["holm"]
    if h.get("applicable"):
        print("  HOLM (m=%d vs %s, alpha/m=%.5f, impl=%s):"
              % (h["family_size"], h["reference"], h["holm_tightest_threshold"],
                 h["holm_impl"]))
        for r in h["family"]:
            print("    %-28s delta=%+.4f p=%-10.4g adj_p=%-10.4g raw_rej=%-5s holm_rej=%s"
                  % (r["method"], r["delta_vs_reference"], r["p_two_sided"],
                     r["holm_adjusted_p"], r["uncorrected_reject"], r["holm_reject"]))
        print("  selected=%s survives_holm=%s holm_changes_the_picture=%s "
              "(resolution_floor=%.5f, blocks=%s)"
              % (h["selected_method"], h["selected_survives_holm"],
                 h["holm_changes_the_picture"], h["p_resolution_floor"],
                 h["resolution_floor_blocks_rejection"]))


# ======================================================================================
# CPU SELF-TEST -- synthetic scores only. No corpus, no model, no network.
# ======================================================================================
def _synthetic_scores(n=400, seed=0):
    """Labels + a score dict with a KNOWN ordering, so each check has a right answer.

    strong  : clearly separates       -> big margin over last_step
    control : `last_step` stand-in    -> the reference
    tie     : identical to control    -> delta exactly 0, p ~ 1
    marginal: a deliberately small edge, tuned so its uncorrected p lands under 0.05
              while its Holm-adjusted p does not. That cell is the whole point of the
              Holm block: without it the test could pass while the correction did
              nothing.
    """
    rng = np.random.default_rng(seed)
    y = np.concatenate([np.ones(n // 2, dtype=int), np.zeros(n // 2, dtype=int)])
    noise = rng.normal(size=n)
    control = y * 0.85 + noise
    strong = y * 2.20 + rng.normal(size=n)
    tie = control.copy()
    marginal = control + y * 0.16 + 0.02 * rng.normal(size=n)
    weak = control - y * 0.30 + 0.05 * rng.normal(size=n)
    return y, {"strong": strong, "last_step": control, "tie": tie,
               "marginal": marginal, "weak": weak}


def _self_test() -> None:
    print("verdict_context.py self-test -- CPU only, no model, no corpus")
    print("-" * 78)

    y, scores = _synthetic_scores()
    aucs = {m: fast_auc(y, s) for m, s in scores.items()}
    cis = {m: (a - 0.03, a + 0.03) for m, a in aucs.items()}
    for m, a in sorted(aucs.items(), key=lambda kv: -kv[1]):
        print("  %-12s auc=%.4f" % (m, a))

    # --- 1. bar context: a method can hold a falsifier and still sit BELOW the bar -----
    # A bar set above every method is the SHADE/AgentDojo situation in miniature.
    high_bar = {"binding_bar": 0.99, "binding_bar_name": "content",
                "worst_auc": 0.99, "worst_name": "content"}
    b = bar_context_many(aucs, high_bar, cis=cis)
    assert b["best_clears_bar"] is False, "a bar above every method must not be cleared"
    assert b["n_methods_clearing_bar"] == 0
    low_bar = {"binding_bar": 0.50, "binding_bar_name": "length",
               "worst_auc": 0.50, "worst_name": "length"}
    b2 = bar_context_many(aucs, low_bar, cis=cis)
    assert b2["best_clears_bar"] is True and b2["n_methods_clearing_bar"] == len(aucs)
    assert b2["per_method"]["strong"]["ci_lo_clears_bar"] is True
    print("OK  bar context: best_clears_bar False against a 0.99 bar, True against 0.50; "
          "margins delegate to common.confound.margin_over_bar")

    # --- 2. paired bootstrap on the margin --------------------------------------------
    pm = paired_margin(y, scores, "strong", "last_step", n=2000, seed=0)
    assert pm["applicable"] and pm["excludes_zero"], "a real gap must exclude zero"
    assert pm["ci_lo_above_registered_margin"], "strong-vs-control must clear 0.02"
    assert pm["p_two_sided"] < 0.01, "p=%s for an obvious gap" % pm["p_two_sided"]
    null = paired_margin(y, scores, "tie", "last_step", n=2000, seed=0)
    assert abs(null["delta"]) < 1e-12 and not null["excludes_zero"]
    assert null["p_two_sided"] > 0.5, "identical scores must not look significant"
    missing = paired_margin(y, scores, "nope", "last_step", n=200, seed=0)
    assert missing["applicable"] is False
    print("OK  paired margin: delta=%.4f ci=[%.4f,%.4f] p=%.4g (strong); "
          "delta=%.1g p=%.3f (identical arms); missing method degrades honestly"
          % (pm["delta"], pm["ci"][0], pm["ci"][1], pm["p_two_sided"],
             null["delta"], null["p_two_sided"]))

    # --- 3. Holm over the max-over-m selection -----------------------------------------
    h = holm_vs_reference(y, scores, reference="last_step", n=4000, seed=0)
    assert h["applicable"] and h["holm_impl"].startswith("steering.stats.holm_bonferroni")
    assert h["family_size"] == len(scores) - 1
    assert h["including_reference_self_comparison"]["family_size"] == len(scores)
    assert not h["resolution_floor_blocks_rejection"], \
        "4000 resamples must have the resolution to reject at alpha/m"
    assert h["selected_method"] == "strong" and h["selected_survives_holm"] is True
    by = {r["method"]: r for r in h["family"]}
    assert by["tie"]["holm_reject"] is False
    for r in h["family"]:
        assert r["holm_adjusted_p"] >= (r["p_two_sided"] or 0.0) - 1e-12, \
            "an adjusted p must never fall below its raw p"
    print("OK  Holm: impl=%s m=%d selected=%s raw_p=%.4g adj_p=%.4g survives=%s"
          % (h["holm_impl"], h["family_size"], h["selected_method"],
             h["selected_p_two_sided"], h["selected_holm_adjusted_p"],
             h["selected_survives_holm"]))

    # A family where the correction MUST FLIP A ROW, not merely inflate a p-value. Nine
    # null arms drag m to 10, and the one marginal arm is tuned to land under alpha
    # (0.05) raw and over Holm's tightest threshold (alpha/10 = 0.005). Without a cell
    # like this the Holm block could be inert and the test would still pass -- which is
    # the failure mode this whole module exists to catch elsewhere.
    rng = np.random.default_rng(7)
    y2 = np.concatenate([np.ones(300, dtype=int), np.zeros(300, dtype=int)])
    ref = y2 * 0.9 + rng.normal(size=600)
    off_label = rng.normal(size=600)  # noise NOT aligned with the label
    fam = {"last_step": ref, "marginal": ref + y2 * 0.34 + 0.6 * off_label}
    for i in range(9):
        fam["null_%d" % i] = ref + 0.01 * rng.normal(size=600)
    h2 = holm_vs_reference(y2, fam, reference="last_step", n=1500, seed=1)
    mrow = next(r for r in h2["family"] if r["method"] == "marginal")
    print("  marginal arm: delta=%.4f raw_p=%.4g adj_p=%.4g raw_reject=%s holm_reject=%s "
          "(m=%d, alpha/m=%.4f)"
          % (mrow["delta_vs_reference"], mrow["p_two_sided"], mrow["holm_adjusted_p"],
             mrow["uncorrected_reject"], mrow["holm_reject"], h2["family_size"],
             h2["holm_tightest_threshold"]))
    assert h2["family_size"] == 10
    assert mrow["holm_adjusted_p"] > mrow["p_two_sided"], \
        "Holm must inflate a p-value inside a family of 10"
    assert mrow["uncorrected_reject"] is True and mrow["holm_reject"] is False, \
        ("the tuned marginal arm must reject uncorrected and FAIL under Holm "
         "(raw p=%s, alpha/m=%s) -- otherwise this test never exercises the correction"
         % (mrow["p_two_sided"], h2["holm_tightest_threshold"]))
    assert h2["holm_changes_the_picture"] is True and h2["n_rows_flipped_by_holm"] == 1
    assert h2["n_holm_reject"] < h2["n_uncorrected_reject"], \
        "correction can only ever remove rejections, and here it must remove one"
    print("OK  Holm bites: the marginal arm rejects at raw p=%.4g and FAILS at "
          "alpha/m=%.4f; n_holm_reject=%d < n_uncorrected_reject=%d"
          % (mrow["p_two_sided"], h2["holm_tightest_threshold"], h2["n_holm_reject"],
             h2["n_uncorrected_reject"]))

    # A degenerate-resolution family: 20 resamples cannot resolve alpha/m at all.
    h3 = holm_vs_reference(y, scores, reference="last_step", n=20, seed=0)
    assert h3["resolution_floor_blocks_rejection"] is True, \
        "a 20-resample bootstrap must be flagged as unable to reject at alpha/m"
    print("OK  resolution floor: B=20 -> floor %.4f > alpha/m %.4f, flagged vacuous"
          % (h3["p_resolution_floor"], h3["holm_tightest_threshold"]))

    # --- 4. attach_context never moves a registered field ------------------------------
    verdicts = {
        "F1_mean_pool_collapses_long": {"holds": True, "mean_pool_auc": 0.74},
        "F3_sequence_beats_laststep": {"holds": True, "best_auc": 0.94},
    }
    snapshot = {k: dict(v) for k, v in verdicts.items()}
    attach_context(verdicts, {"F1_mean_pool_collapses_long": {"x": 1},
                              "F3_sequence_beats_laststep": {"x": 2},
                              "F_not_in_this_run": {"x": 3}})
    for k, v in snapshot.items():
        for field, val in v.items():
            assert verdicts[k][field] == val, "attach_context mutated %s.%s" % (k, field)
        assert "context" in verdicts[k]
    try:
        attach_context(verdicts, {"F1_mean_pool_collapses_long": {"x": 9}})
    except ValueError:
        pass
    else:
        raise AssertionError("attach_context must refuse to overwrite an existing context")
    print("OK  attach_context: holds/applicable untouched, unknown tags dropped not "
          "invented, double-attach refused")

    # --- 5. the ladder shuffle control, end to end on synthetic embeddings --------------
    from steering_tutorials.streaming_trajectory_aggregation.aggregators.pooling import (
        LastStep, MaxPool, MeanPool,
    )
    from steering_tutorials.streaming_trajectory_aggregation.horizon import (
        _synthetic_horizon_corpus,
    )

    corpus, embs = _synthetic_horizon_corpus(n_per_class=60, steps_range=(20, 41), dim=8,
                                              n_short=0, seed=0)
    factory = {"mean_pool": MeanPool, "max_pool": MaxPool, "last_step": LastStep}
    real = evaluate.evaluate_offline(corpus, embs, [MaxPool(seed=0)], seed=0, n_folds=4,
                                      bootstrap=200)
    print("  real-label max_pool auc=%.4f (signal is present by construction)" % real[0].auc)
    assert real[0].auc > 0.7, "the planted corpus must be separable before shuffling"

    sc = ladder_shuffle_control(
        corpus, embs, lambda nm: factory[nm](seed=0),
        methods=["mean_pool", "max_pool", "last_step"], seed=0, n_folds=4, bootstrap=200,
        repeats=2)
    for m, blk in sc["per_method"].items():
        print("  shuffled %-12s aucs=%s within_band=%s dev=%.3f (%.2f null SE)"
              % (m, [round(v, 4) for v in blk["aucs"]], blk["all_within_band"],
                 blk["max_abs_deviation_from_chance"],
                 blk["max_abs_deviation_in_null_se"]))
    assert sc["n_permutations"] == 2 and len(sc["permutations"]) == 2
    assert set(sc["per_method"]) == {"mean_pool", "max_pool", "last_step"}
    # ASSERT IN SE UNITS, NOT IN BAND UNITS, and say why: this synthetic is 60/60, where
    # the null AUC SE is ~0.053, so the [0.45,0.55] band is under one sigma wide and a
    # clean run fails it routinely. The band is calibrated for the ~500/class corpora
    # this lesson actually runs (SE ~0.018, so the band is ~2.7 sigma). Asserting the
    # band here would either fail on correct code or force a band that means nothing at
    # scale -- so the scale-free reading is what is locked, and the module reports both.
    assert sc["band_is_meaningful_at_this_n"] is False, \
        "60/60 must be flagged as too small for the band to test anything"
    for m, blk in sc["per_method"].items():
        assert blk["max_abs_deviation_in_null_se"] < 3.0, \
            "permuted-label %s sits %.2f null SEs from chance (aucs=%s) -- that is a leak" \
            % (m, blk["max_abs_deviation_in_null_se"], blk["aucs"])
    # a method the factory cannot build must be reported as `no_result`, never folded in
    # with the methods that ran and missed the band
    sc_missing = ladder_shuffle_control(
        corpus, embs, lambda nm: factory.get(nm, lambda **kw: None)(seed=0)
        if nm in factory else None,
        methods=["last_step", "not_a_method"], seed=0, n_folds=4, bootstrap=100, repeats=1)
    assert sc_missing["methods_with_no_result"] == ["not_a_method"], \
        "an unbuildable method must be reported as no_result: %s" % sc_missing["per_method"]
    assert "not_a_method" not in sc_missing["methods_outside_band"], \
        "a missing number must not be reported as a chance violation"
    print("OK  ladder shuffle: an unbuildable method lands in methods_with_no_result, "
          "not in methods_outside_band")

    # the permutation must actually permute -- a no-op would pass the band check trivially
    rng = np.random.default_rng(0)
    perm_corpus = _permuted_corpus(corpus, rng, "check")
    orig = [t.label for t in corpus.trajectories]
    new = [t.label for t in perm_corpus.trajectories]
    assert sorted(orig) == sorted(new), "permutation changed the class balance"
    assert orig != new, "permutation was a no-op"
    assert perm_corpus.pool_fingerprint.endswith("#check"), \
        "a permuted corpus must not carry the real corpus fingerprint unmarked"
    worst = max(b["max_abs_deviation_in_null_se"] for b in sc["per_method"].values())
    print("OK  ladder shuffle: %d permutations x 3 methods, worst %.2f null SE from "
          "chance, on a corpus that scores %.4f with real labels. Band NOT asserted at "
          "n=%d/%d (half-band %.2f null SE -- reported as band_is_meaningful_at_this_n="
          "%s); class balance preserved, fingerprint suffixed."
          % (sc["n_permutations"], worst, real[0].auc,
             sum(1 for t in corpus.trajectories if t.label == 1),
             sum(1 for t in corpus.trajectories if t.label == 0),
             sc["band_half_width_in_null_se"], sc["band_is_meaningful_at_this_n"]))

    print("-" * 78)
    print("OK -- verdict_context.py self-test passed CPU-only, no model loaded")


if __name__ == "__main__":
    argv = sys.argv[1:]
    if argv and argv[0] == "--from-results":
        # Backfill the context for a run already on disk:
        #   python -m ...verdict_context --from-results artifacts/results_shade.json
        # JSON is written BEFORE the summary print, so a cp1252 console cannot cost the
        # data (CLAUDE.md section 17, operational playbook item 3).
        _print_backfill(backfill_from_results(argv[1]))
    else:
        _self_test()
