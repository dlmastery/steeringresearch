"""horizon.py -- the WITHIN-CORPUS horizon control.

WHY THIS FILE EXISTS
--------------------
The lesson's central claim is that mean pooling collapses as the HORIZON grows. Until
this module, the only evidence offered for it was a BETWEEN-corpus contrast: F1 on SHADE
(median 131 steps) plus F2 on AgentDojo (median 8). That contrast cannot license the
conclusion. SHADE and AgentDojo differ in task suite, generating agent, tool inventory,
environment, the prose style of a rendered step, base difficulty, and the KIND of
positive (a covert secondary objective vs an injected instruction) -- all of it varying
together with the horizon. "F1 holds and F2 holds" is equally consistent with "AgentDojo's
prompt injections are blatant enough that any pooled representation finds them", which
has nothing to do with horizon length.

The fix (AUDIT_FIDELITY.md section 2.2, ranked fix 1) is to move the contrast INSIDE one
corpus, where task, generator, tools, prose and label machinery are held perfectly fixed:

    delta(k) = AUC[Truncate(k, MaxPool)] - AUC[Truncate(k, MeanPool)]

swept over k, on SHADE alone. Both arms see the IDENTICAL first-k window of the IDENTICAL
trajectories, so within a k the ONLY thing that differs is the pooling function, and
across k the ONLY thing that differs is how much horizon each arm must survive. If
delta(k) grows with k here, the horizon claim is established with one corpus, and
AgentDojo demotes from sole control to external replication.

TWO CONTROLS, AND THEY ARE NOT EQUALLY STRONG -- SAY SO
-------------------------------------------------------
1. `run_within_corpus_horizon` -- the TRUNCATION sweep above. The strong one: the same
   trajectory set at every k, so horizon is manipulated directly and nothing else moves.
2. `run_median_split_horizon` -- the corpus split at its OWN median step count, short half
   vs long half, both arms untruncated. Free (the data is already loaded) and it uses REAL
   long horizons rather than synthetic prefixes -- but it is OBSERVATIONAL, and this
   module's self-test shows it is weak enough to actively mislead: on a synthetic corpus
   with a PLANTED dilution effect that control 1 recovers cleanly (delta 0.15 -> rho 0.89,
   growth CI excluding zero), the median split returns the OPPOSITE SIGN. The mechanism is
   not noise: a median split leaves the lower half spanning the corpus minimum to the
   median (a wide length spread) and the upper half far more homogeneous, and a
   mean-pooled feature scales like signal/n, so the wide-spread half presents one linear
   threshold with a wide range of feature magnitudes and scores worse for reasons that
   have nothing to do with its mean horizon. So control 2 CORROBORATES at best. Per-stratum
   class balance, a length-only AUC and the length dispersion are all reported so a reader
   can price that rather than take our word for it -- and a stratum contrast pointing the
   other way is NOT evidence against control 1.

PRE-REGISTERED BEFORE THE SWEEP (config.HORIZON_PREREGISTERED / HORIZON_CLAIM), not
written after the numbers came in.

Reuses `evaluate.evaluate_offline` for every AUC -- same folds, same group-aware CV, same
bootstrap, same shared classifier. The only statistic added here is a PAIRED bootstrap on
the delta, which is possible precisely because `AggregatorResult.scores` is index-aligned
across methods (out-of-fold scores for the same trajectories) -- and which the existing
verdict logic wants anyway, since a 0.02 point-estimate threshold cannot say whether a
gap is outside noise.

CPU-only. Loads NO model, downloads nothing. ASCII stdout only (Windows cp1252).
"""
from __future__ import annotations

import sys
from dataclasses import replace
from typing import Callable, Sequence

import numpy as np

from steering_tutorials.streaming_trajectory_aggregation import config as C
from steering_tutorials.streaming_trajectory_aggregation import evaluate
from steering_tutorials.streaming_trajectory_aggregation.aggregators.pooling import (
    MaxPool, MeanPool, Truncate,
)
from steering_tutorials.streaming_trajectory_aggregation.types import Corpus

__all__ = [
    "INNER_FACTORIES",
    "fast_auc",
    "paired_delta_ci",
    "run_within_corpus_horizon",
    "run_median_split_horizon",
]


def _eprint(*args) -> None:
    msg = " ".join(str(a) for a in args)
    try:
        print(msg, file=sys.stderr)
    except Exception:
        print(msg.encode("ascii", "replace").decode("ascii"), file=sys.stderr)


def _num(x):
    """JSON-safe float (NaN -> None), matching run_sta.py's convention."""
    if x is None:
        return None
    x = float(x)
    return None if (x != x) else x


# Any `_PooledAggregator` composes with `Truncate`; these are the two the claim is about.
INNER_FACTORIES: dict[str, Callable[..., object]] = {
    "mean_pool": MeanPool,
    "max_pool": MaxPool,
}


# ======================================================================================
# AUC + paired bootstrap on the delta
# ======================================================================================
def fast_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Rank-based ROC-AUC (average ranks for ties). Identical to
    `sklearn.metrics.roc_auc_score`, ~100x cheaper per call -- which matters because the
    paired bootstrap evaluates several AUCs per resample.

    NOT a reimplementation anyone is asked to trust on faith: agreement with sklearn is
    asserted against the REAL scores on every run (`fast_auc_agrees_with_sklearn` in the
    returned dict) and against random data in this module's self-test.
    """
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score, dtype=np.float64)
    n_pos = int((y_true == 1).sum())
    n_neg = int((y_true == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(y_score, kind="mergesort")
    s_sorted = y_score[order]
    ranks = np.empty(len(y_score), dtype=np.float64)
    i = 0
    m = len(s_sorted)
    while i < m:
        j = i
        while j + 1 < m and s_sorted[j + 1] == s_sorted[i]:
            j += 1
        ranks[order[i:j + 1]] = 0.5 * (i + j) + 1.0  # average rank, 1-based
        i = j + 1
    sum_pos = float(ranks[y_true == 1].sum())
    return (sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def _common_valid(scores_by_name: dict[str, np.ndarray]) -> np.ndarray:
    """Indices scored (non-NaN) by EVERY method -- the paired comparison's index set.

    A trajectory that one arm failed to score cannot enter a paired delta for any arm;
    silently letting each arm use its own index set is how an unpaired difference gets
    reported under a paired label.
    """
    mask = None
    for s in scores_by_name.values():
        m = ~np.isnan(np.asarray(s, dtype=np.float64))
        mask = m if mask is None else (mask & m)
    return np.where(mask)[0] if mask is not None else np.array([], dtype=int)


def paired_delta_ci(labels: np.ndarray, hi_scores: np.ndarray, lo_scores: np.ndarray,
                    n: int = 10000, seed: int = 0) -> dict:
    """95% percentile bootstrap CI on the PAIRED delta AUC(hi) - AUC(lo).

    Paired because both arrays are out-of-fold scores for the SAME trajectories in the
    SAME order: each resample draws one index set and scores both arms on it, so the
    shared per-trajectory difficulty cancels instead of inflating the interval.
    """
    idx = _common_valid({"hi": hi_scores, "lo": lo_scores})
    y = np.asarray(labels)[idx].astype(int)
    hi = np.asarray(hi_scores, dtype=np.float64)[idx]
    lo = np.asarray(lo_scores, dtype=np.float64)[idx]
    out = {"n_paired": int(len(idx)), "n_resamples": int(n)}
    if len(idx) == 0 or len(np.unique(y)) < 2:
        out.update({"delta": None, "ci": [None, None], "excludes_zero": False,
                    "reason": "fewer than two classes among commonly-scored trajectories"})
        return out
    point = fast_auc(y, hi) - fast_auc(y, lo)
    rng = np.random.default_rng(seed)
    m = len(y)
    boots = []
    for _ in range(int(n)):
        r = rng.integers(0, m, m)
        yy = y[r]
        if len(np.unique(yy)) < 2:
            continue
        boots.append(fast_auc(yy, hi[r]) - fast_auc(yy, lo[r]))
    if not boots:
        out.update({"delta": _num(point), "ci": [_num(point), _num(point)],
                    "excludes_zero": False,
                    "reason": "every resample was single-class"})
        return out
    lo_ci = float(np.percentile(boots, 2.5))
    hi_ci = float(np.percentile(boots, 97.5))
    out.update({
        "delta": _num(point),
        "ci": [_num(lo_ci), _num(hi_ci)],
        "excludes_zero": bool(lo_ci > 0.0 or hi_ci < 0.0),
    })
    return out


def _paired_growth_ci(labels, hi_big, lo_big, hi_small, lo_small, n=10000, seed=0) -> dict:
    """CI on delta(k_max) - delta(k_min): does the GAP itself grow?

    All four score vectors are out-of-fold scores for the same trajectories, so one
    resampled index set serves all four -- the quantity under test is a difference of
    differences on a single paired sample, not two independent intervals compared by eye.
    """
    names = {"hb": hi_big, "lb": lo_big, "hs": hi_small, "ls": lo_small}
    idx = _common_valid(names)
    y = np.asarray(labels)[idx].astype(int)
    arrs = {k: np.asarray(v, dtype=np.float64)[idx] for k, v in names.items()}
    out = {"n_paired": int(len(idx)), "n_resamples": int(n)}
    if len(idx) == 0 or len(np.unique(y)) < 2:
        out.update({"growth": None, "ci": [None, None], "excludes_zero": False,
                    "reason": "fewer than two classes among commonly-scored trajectories"})
        return out

    def _stat(yy, a):
        return (fast_auc(yy, a["hb"]) - fast_auc(yy, a["lb"])) - \
               (fast_auc(yy, a["hs"]) - fast_auc(yy, a["ls"]))

    point = _stat(y, arrs)
    rng = np.random.default_rng(seed)
    m = len(y)
    boots = []
    for _ in range(int(n)):
        r = rng.integers(0, m, m)
        yy = y[r]
        if len(np.unique(yy)) < 2:
            continue
        boots.append(_stat(yy, {k: v[r] for k, v in arrs.items()}))
    if not boots:
        out.update({"growth": _num(point), "ci": [_num(point), _num(point)],
                    "excludes_zero": False, "reason": "every resample was single-class"})
        return out
    lo_ci = float(np.percentile(boots, 2.5))
    hi_ci = float(np.percentile(boots, 97.5))
    out.update({"growth": _num(point), "ci": [_num(lo_ci), _num(hi_ci)],
                "excludes_zero": bool(lo_ci > 0.0 or hi_ci < 0.0)})
    return out


def _spearman(xs, ys) -> dict:
    """Rank correlation of delta against k. scipy if present, plain Pearson-on-ranks
    otherwise -- the p-value is simply absent in the fallback rather than faked."""
    xs = np.asarray(xs, dtype=np.float64)
    ys = np.asarray(ys, dtype=np.float64)
    ok = np.isfinite(xs) & np.isfinite(ys)
    xs, ys = xs[ok], ys[ok]
    if len(xs) < 3:
        return {"rho": None, "p": None, "n": int(len(xs)),
                "note": "fewer than 3 usable k-cells; rank correlation undefined"}
    try:
        from scipy.stats import spearmanr

        rho, p = spearmanr(xs, ys)
        return {"rho": _num(rho), "p": _num(p), "n": int(len(xs)), "backend": "scipy"}
    except Exception:
        rx = np.argsort(np.argsort(xs)).astype(float)
        ry = np.argsort(np.argsort(ys)).astype(float)
        rho = float(np.corrcoef(rx, ry)[0, 1])
        return {"rho": _num(rho), "p": None, "n": int(len(xs)), "backend": "numpy-ranks",
                "note": "scipy unavailable; p-value not computed rather than approximated"}


# ======================================================================================
# 1. THE STRONG CONTROL -- truncation sweep on ONE corpus
# ======================================================================================
def run_within_corpus_horizon(corpus: Corpus, embeddings: list[np.ndarray],
                               ks: Sequence[int] = None,
                               inner_low: str = None, inner_high: str = None,
                               seed: int = None, n_folds: int = None,
                               bootstrap: int = None, delta_bootstrap: int = None,
                               main_ladder_aucs: dict = None) -> dict:
    """delta(k) = AUC[Truncate(k, high)] - AUC[Truncate(k, low)] over k, plus k = all.

    `main_ladder_aucs` (optional, {method: auc}) enables the k=all ANCHOR CHECK:
    `Truncate(None, MeanPool)` is the same estimator as the main ladder's `mean_pool` over
    the same folds and must reproduce its AUC exactly. A mismatch means the two paths have
    drifted apart and the sweep is not comparable to the ladder it is read beside -- so it
    is asserted and recorded, not assumed (CLAUDE.md section 18.8, "assert your anchors").
    """
    ks = list(C.HORIZON_KS if ks is None else ks)
    inner_low = C.HORIZON_INNER_LOW if inner_low is None else inner_low
    inner_high = C.HORIZON_INNER_HIGH if inner_high is None else inner_high
    seed = C.SEED if seed is None else seed
    n_folds = C.N_FOLDS if n_folds is None else n_folds
    bootstrap = C.BOOTSTRAP if bootstrap is None else bootstrap
    delta_bootstrap = C.HORIZON_BOOTSTRAP if delta_bootstrap is None else delta_bootstrap
    for nm in (inner_low, inner_high):
        if nm not in INNER_FACTORIES:
            raise ValueError("unknown inner aggregator %r; known: %s"
                             % (nm, sorted(INNER_FACTORIES)))

    labels = np.asarray([t.label for t in corpus.trajectories], dtype=int)
    ks_sorted = sorted({int(k) for k in ks if int(k) >= 1})
    cells = []
    scores_by_cell: dict[tuple, np.ndarray] = {}

    for k in ks_sorted + [None]:
        aggs = [Truncate(k, INNER_FACTORIES[nm](seed=seed),
                         name="trunc_%s_%s" % ("all" if k is None else k, nm))
                for nm in (inner_low, inner_high)]
        results = evaluate.evaluate_offline(corpus, embeddings, aggs, seed=seed,
                                            n_folds=n_folds, bootstrap=bootstrap)
        by_name = {r.method: r for r in results}
        per_inner = {}
        for agg, nm in zip(aggs, (inner_low, inner_high)):
            r = by_name[agg.name]
            scores_by_cell[(k, nm)] = np.asarray(r.scores, dtype=np.float64)
            per_inner[nm] = {
                "aggregator": agg.name,
                "auc": _num(r.auc),
                "auc_ci": [_num(r.auc_ci[0]), _num(r.auc_ci[1])],
                "is_causal": bool(agg.is_causal),
                "notes": r.notes,
            }
        cov = aggs[0].truncation_stats(embeddings, labels)
        # A k that binds nothing is Truncate(all) wearing a k label. Say so; do not let
        # it enter a monotonicity read as though it were an independent horizon cell.
        cov["degenerate_equals_all"] = bool(k is not None and cov["n_bound"] == 0)
        d = paired_delta_ci(labels, scores_by_cell[(k, inner_high)],
                            scores_by_cell[(k, inner_low)],
                            n=delta_bootstrap, seed=seed)
        cells.append({
            "k": k,
            "k_label": "all" if k is None else str(k),
            "coverage": cov,
            "per_inner": per_inner,
            "delta_name": "AUC[%s] - AUC[%s]" % (inner_high, inner_low),
            "delta": d["delta"],
            "delta_ci": d["ci"],
            "delta_excludes_zero": d["excludes_zero"],
            "n_paired": d["n_paired"],
        })
        _eprint("[horizon] k=%-4s %s=%.4f %s=%.4f delta=%s ci=%s bound=%d/%d"
                % (cells[-1]["k_label"], inner_low,
                   per_inner[inner_low]["auc"] or float("nan"), inner_high,
                   per_inner[inner_high]["auc"] or float("nan"),
                   cells[-1]["delta"], cells[-1]["delta_ci"],
                   cov["n_bound"], cov["n_total"]))

    # --- anchor 1: fast_auc must agree with sklearn on the REAL scores ----------------
    anchor_fast = None
    try:
        from sklearn.metrics import roc_auc_score

        s = scores_by_cell[(None, inner_low)]
        m = ~np.isnan(s)
        if m.sum() and len(np.unique(labels[m])) > 1:
            diff = abs(fast_auc(labels[m], s[m]) - float(roc_auc_score(labels[m], s[m])))
            anchor_fast = {"max_abs_diff": _num(diff), "agrees": bool(diff < 1e-9)}
            if diff >= 1e-9:
                _eprint("[horizon] WARNING: fast_auc disagrees with sklearn by %.3g" % diff)
    except Exception as exc:  # pragma: no cover -- sklearn is a hard dependency
        anchor_fast = {"agrees": None, "error": str(exc)}

    # --- anchor 2: Truncate(None, low) == the main ladder's own row -------------------
    anchor_all = None
    if main_ladder_aucs:
        ladder = main_ladder_aucs.get(inner_low)
        mine = next((c["per_inner"][inner_low]["auc"] for c in cells if c["k"] is None), None)
        if isinstance(ladder, float) and isinstance(mine, float):
            diff = abs(ladder - mine)
            anchor_all = {"main_ladder_auc": _num(ladder), "k_all_auc": _num(mine),
                          "abs_diff": _num(diff), "matches": bool(diff < 1e-9)}
            if diff >= 1e-9:
                _eprint("[horizon] WARNING: k=all %s AUC (%.6f) != main-ladder %s (%.6f) "
                        "-- the sweep and the ladder have drifted apart"
                        % (inner_low, mine, inner_low, ladder))

    # --- trend over the FINITE, NON-DEGENERATE k-cells --------------------------------
    usable = [c for c in cells
              if c["k"] is not None and not c["coverage"]["degenerate_equals_all"]
              and isinstance(c["delta"], float)]
    ks_u = [c["k"] for c in usable]
    ds_u = [c["delta"] for c in usable]
    monotone = bool(len(ds_u) >= 2 and all(b >= a - 1e-12 for a, b in zip(ds_u, ds_u[1:])))
    rank = _spearman(ks_u, ds_u)

    growth = {"applicable": False,
              "reason": "fewer than two non-degenerate finite k-cells to compare"}
    if len(usable) >= 2:
        k_small, k_big = usable[0]["k"], usable[-1]["k"]
        growth = _paired_growth_ci(
            labels,
            scores_by_cell[(k_big, inner_high)], scores_by_cell[(k_big, inner_low)],
            scores_by_cell[(k_small, inner_high)], scores_by_cell[(k_small, inner_low)],
            n=delta_bootstrap, seed=seed)
        growth.update({"applicable": True, "k_small": k_small, "k_big": k_big,
                       "statistic": "delta(k_big) - delta(k_small), paired"})

    rho = rank.get("rho")
    criterion_met = bool(
        isinstance(rho, float) and rho >= C.HORIZON_RHO_FLOOR
        and growth.get("applicable") and growth.get("excludes_zero")
        and isinstance(growth.get("growth"), float) and growth["growth"] > 0
    )

    return {
        "preregistered": C.HORIZON_PREREGISTERED,
        "claim": C.HORIZON_CLAIM,
        "success_criterion": C.HORIZON_CRITERION,
        "corpus": corpus.name,
        "inner_low": inner_low,
        "inner_high": inner_high,
        "ks": ks_sorted,
        "cells": cells,
        "monotone_nondecreasing": monotone,
        "rank_correlation_delta_vs_k": rank,
        "growth_k_big_minus_k_small": growth,
        "criterion_met": criterion_met,
        "anchors": {"fast_auc_agrees_with_sklearn": anchor_fast,
                    "k_all_matches_main_ladder": anchor_all},
        "short_trajectory_policy": (
            "use_available: a trajectory with fewer than k steps is passed through WHOLE "
            "-- never zero-padded, never dropped. Padding would invent steps and pull a "
            "pooled mean toward the pad value; dropping would make each k-cell score a "
            "different trajectory set, confounding horizon with sample composition. The "
            "price is that at large k the short trajectories are effectively untruncated, "
            "so coverage.n_bound / coverage.n_shorter_than_k (per class) are reported for "
            "every cell and a cell binding NOTHING is flagged degenerate_equals_all."),
        "reading": (
            "delta(k) rising with k, on ONE corpus, is the dilution claim with task, "
            "generator, tools, prose style and label machinery held fixed -- both arms see "
            "the identical first-k window of the identical trajectories. delta(k) flat is "
            "evidence AGAINST dilution on this substrate. This does NOT license a claim "
            "about any other corpus, embedder or seed: screening tier, one of each."),
    }


# ======================================================================================
# 2. THE FREE, WEAKER CONTROL -- the corpus split at its own median step count
# ======================================================================================
def _subset_corpus(corpus: Corpus, idx: Sequence[int], name_suffix: str) -> Corpus:
    """A Corpus over a subset of trajectories, with provenance carried through.

    `pool_fingerprint` is deliberately NOT recomputed to look like a fresh sample -- it is
    suffixed, so a stratum artifact can never be mistaken for the full corpus it came from.
    """
    subs = [corpus.trajectories[i] for i in idx]
    return replace(corpus,
                   name="%s_%s" % (corpus.name, name_suffix),
                   trajectories=subs,
                   pool_fingerprint="%s#%s" % (corpus.pool_fingerprint, name_suffix))


def run_median_split_horizon(corpus: Corpus, embeddings: list[np.ndarray],
                              inner_low: str = None, inner_high: str = None,
                              seed: int = None, n_folds: int = None,
                              bootstrap: int = None, delta_bootstrap: int = None) -> dict:
    """Split the corpus at its OWN median step count; run both arms on each half.

    Same corpus, same tasks, same generator, different horizon -- and, unlike the
    truncation sweep, REAL long trajectories rather than synthetic prefixes. The cost is
    that this is observational: episode length was not assigned, so a stratum gap
    confounds horizon with anything that covaries with length (task difficulty, tool
    count, how many retries the agent needed). Per-stratum class balance and a
    LENGTH-ONLY AUC are reported so that confound is priced rather than waved at.

    Lengths are taken from the EMBEDDING arrays, not `Trajectory.n_steps`, because the
    embeddings are what the ladder actually sees (config.MAX_STEPS may have capped them).
    """
    inner_low = C.HORIZON_INNER_LOW if inner_low is None else inner_low
    inner_high = C.HORIZON_INNER_HIGH if inner_high is None else inner_high
    seed = C.SEED if seed is None else seed
    n_folds = C.N_FOLDS if n_folds is None else n_folds
    bootstrap = C.BOOTSTRAP if bootstrap is None else bootstrap
    delta_bootstrap = C.HORIZON_BOOTSTRAP if delta_bootstrap is None else delta_bootstrap

    labels = np.asarray([t.label for t in corpus.trajectories], dtype=int)
    lengths = np.asarray([int(np.asarray(e).shape[0]) for e in embeddings], dtype=int)
    if len(lengths) != len(labels):
        raise ValueError("len(embeddings)=%d != len(corpus.trajectories)=%d"
                         % (len(lengths), len(labels)))
    median = float(np.median(lengths))
    strata_idx = {
        "short": [int(i) for i in np.where(lengths <= median)[0]],
        "long": [int(i) for i in np.where(lengths > median)[0]],
    }

    out = {
        "median_steps": median,
        "length_source": "embedding arrays (post config.MAX_STEPS cap)",
        "n_at_exactly_median": int((lengths == median).sum()),
        "tie_rule": "steps <= median -> short, steps > median -> long",
        "corpus_length_only_auc": _num(fast_auc(labels, lengths.astype(float))),
        "strata": {},
        "delta_long_minus_short": None,
        "caveat": (
            "OBSERVATIONAL, and MUCH weaker than the truncation sweep -- do not read a "
            "stratum contrast as a horizon measurement. Three named reasons: (1) episode "
            "length was not assigned, so long trajectories may also be harder tasks; "
            "(2) the strata are disjoint trajectory sets, so the long-minus-short "
            "difference cannot be given a paired CI -- each stratum's own delta CI is "
            "reported instead and they must NOT be compared by eyeballing overlap; "
            "(3) a median split leaves the two halves with very different length SPREAD "
            "(see strata[*].length_dispersion), and a mean-pooled feature scales like "
            "signal/n, so the wider-spread half can score worse for reasons unrelated to "
            "its mean horizon. Measured on a synthetic corpus with a PLANTED dilution "
            "effect that the truncation sweep recovers cleanly, this split returned the "
            "OPPOSITE sign (horizon.py self-test). It corroborates at best; it refutes "
            "nothing and establishes nothing on its own."),
    }

    for stratum, idx in strata_idx.items():
        sub_labels = labels[idx]
        n_pos = int((sub_labels == 1).sum())
        n_neg = int((sub_labels == 0).sum())
        sub_lengths = lengths[idx]
        blk = {
            "n": len(idx), "n_pos": n_pos, "n_neg": n_neg,
            "n_groups": len({corpus.trajectories[i].group_id for i in idx}),
            "steps": {"min": int(sub_lengths.min()) if len(idx) else None,
                      "median": _num(np.median(sub_lengths)) if len(idx) else None,
                      "mean": _num(np.mean(sub_lengths)) if len(idx) else None,
                      "max": int(sub_lengths.max()) if len(idx) else None},
            "length_only_auc": _num(fast_auc(sub_labels, sub_lengths.astype(float))),
            # A median split equalises nothing about SPREAD. The lower half runs from the
            # corpus minimum to the median (often a 10x range); the upper half is bounded
            # below by the median and is usually far more homogeneous. That matters:
            # a mean-pooled feature scales like signal/n, so a stratum with a wide length
            # spread presents one linear threshold with a wide range of feature
            # magnitudes and is classified WORSE for reasons that have nothing to do with
            # its mean horizon. Measured here so the comparison can be priced, because it
            # is a mechanism that can flip the sign of the stratum contrast.
            "length_dispersion": {
                "cv": _num(sub_lengths.std() / sub_lengths.mean()) if len(idx) else None,
                "max_over_min": _num(sub_lengths.max() / max(1, sub_lengths.min())) if len(idx) else None,
                "iqr": _num(np.percentile(sub_lengths, 75) - np.percentile(sub_lengths, 25)) if len(idx) else None,
            },
        }
        if n_pos < 2 or n_neg < 2:
            blk.update({"skipped": True,
                        "reason": "stratum has n_pos=%d n_neg=%d -- too few of one class "
                                  "to fit and score a group-aware CV; reported as skipped "
                                  "rather than scored on a degenerate split"
                                  % (n_pos, n_neg)})
            out["strata"][stratum] = blk
            continue

        sub_corpus = _subset_corpus(corpus, idx, stratum)
        sub_emb = [embeddings[i] for i in idx]
        aggs = [INNER_FACTORIES[nm](seed=seed) for nm in (inner_low, inner_high)]
        results = evaluate.evaluate_offline(sub_corpus, sub_emb, aggs, seed=seed,
                                            n_folds=n_folds, bootstrap=bootstrap)
        by_name = {r.method: r for r in results}
        per_inner, scores = {}, {}
        for nm in (inner_low, inner_high):
            r = by_name[nm]
            scores[nm] = np.asarray(r.scores, dtype=np.float64)
            per_inner[nm] = {"auc": _num(r.auc),
                             "auc_ci": [_num(r.auc_ci[0]), _num(r.auc_ci[1])],
                             "notes": r.notes}
        d = paired_delta_ci(sub_labels, scores[inner_high], scores[inner_low],
                            n=delta_bootstrap, seed=seed)
        blk.update({"skipped": False, "per_inner": per_inner,
                    "delta_name": "AUC[%s] - AUC[%s]" % (inner_high, inner_low),
                    "delta": d["delta"], "delta_ci": d["ci"],
                    "delta_excludes_zero": d["excludes_zero"], "n_paired": d["n_paired"]})
        out["strata"][stratum] = blk
        _eprint("[horizon:median_split] %-5s n=%d (pos=%d neg=%d) steps_med=%s delta=%s ci=%s"
                % (stratum, blk["n"], n_pos, n_neg, blk["steps"]["median"],
                   blk["delta"], blk["delta_ci"]))

    s, l = out["strata"].get("short", {}), out["strata"].get("long", {})
    if isinstance(s.get("delta"), float) and isinstance(l.get("delta"), float):
        out["delta_long_minus_short"] = _num(l["delta"] - s["delta"])
    return out


# ======================================================================================
# CPU SELF-TEST -- synthetic embeddings with a PLANTED horizon effect. Loads no model.
#   python -m steering_tutorials.streaming_trajectory_aggregation.horizon
# ======================================================================================
def _synthetic_horizon_corpus(n_per_class=150, steps_range=(60, 201), dim=16, spike=9.0,
                              n_short=20, short_len=12, seed=0):
    """Positives carry ONE anomalous step, always inside the first two steps.

    That placement is the whole point: the anomaly is fully present in EVERY prefix, so a
    k-cell never loses information as k grows -- only dilution differs. Mean-pool SNR then
    falls as (spike/k) / (sigma/sqrt(k)) = spike/(sigma*sqrt(k)), while a per-dimension max
    keeps the spike intact. So delta(k) must RISE. If this test ever fails, either the
    truncation wrapper stopped truncating or the delta statistic stopped measuring the gap.

    Lengths are DRAWN, not fixed, and from the same distribution for both classes -- so the
    median split has two populated strata to compare and a length-only AUC near chance,
    i.e. the one confound the median split is most exposed to is absent BY CONSTRUCTION
    here. (On a real corpus it is measured, not assumed; that is what `length_only_auc` in
    the returned dict is for.)

    `n_short` trajectories are built shorter than the largest k so the short-trajectory
    accounting is exercised too.
    """
    from steering_tutorials.streaming_trajectory_aggregation.types import (
        Corpus, Step, Trajectory,
    )

    rng = np.random.default_rng(seed)
    direction = rng.normal(size=dim)
    direction /= np.linalg.norm(direction)

    trajs, embs = [], []
    for cls in (1, 0):
        for i in range(n_per_class):
            short = i < (n_short // 2)
            n = short_len if short else int(rng.integers(*steps_range))
            a = rng.normal(0.0, 1.0, size=(n, dim)).astype(np.float32)
            if cls == 1:
                a[int(rng.integers(0, 2))] += (spike * direction).astype(np.float32)
            uid = "%s-%d" % ("pos" if cls else "neg", i)
            trajs.append(Trajectory(
                uid=uid,
                steps=tuple(Step(index=j, text="step %d of %s" % (j, uid)) for j in range(n)),
                label=cls, group_id=uid, source="synthetic-horizon"))
            embs.append(a)

    corpus = Corpus(name="synthetic_horizon", trajectories=trajs,
                    requested_n_per_class=n_per_class, pool_fingerprint="0" * 64,
                    licence="synthetic", label_provenance="planted by _synthetic_horizon_corpus")
    return corpus, embs


def _self_test() -> None:
    from sklearn.metrics import roc_auc_score

    # --- fast_auc == sklearn, ties included ------------------------------------------
    rng = np.random.default_rng(0)
    for _ in range(5):
        y = rng.integers(0, 2, 200)
        s = np.round(rng.normal(size=200), 1)  # rounding forces ties
        assert abs(fast_auc(y, s) - roc_auc_score(y, s)) < 1e-12, "fast_auc != sklearn"
    print("OK  fast_auc matches sklearn.roc_auc_score (ties included)")

    corpus, embs = _synthetic_horizon_corpus()
    ks = [2, 4, 8, 16, 32, 64, 128]
    res = run_within_corpus_horizon(corpus, embs, ks=ks, inner_low="mean_pool",
                                    inner_high="max_pool", seed=0, n_folds=4,
                                    bootstrap=200, delta_bootstrap=600)

    print("")
    print("%-6s %-10s %-10s %-9s %-22s %s" % ("k", "mean_pool", "max_pool", "delta",
                                               "delta 95% CI", "bound/total"))
    for c in res["cells"]:
        print("%-6s %-10.4f %-10.4f %-9.4f [%8.4f,%8.4f] %d/%d%s"
              % (c["k_label"], c["per_inner"]["mean_pool"]["auc"],
                 c["per_inner"]["max_pool"]["auc"], c["delta"],
                 c["delta_ci"][0], c["delta_ci"][1],
                 c["coverage"]["n_bound"], c["coverage"]["n_total"],
                 "  DEGENERATE(=all)" if c["coverage"]["degenerate_equals_all"] else ""))

    finite = [c for c in res["cells"] if c["k"] is not None]
    deltas = [c["delta"] for c in finite]
    max_aucs = [c["per_inner"]["max_pool"]["auc"] for c in finite]
    mean_aucs = [c["per_inner"]["mean_pool"]["auc"] for c in finite]

    print("")
    print("rank_correlation(delta, k) rho=%s (%s)"
          % (res["rank_correlation_delta_vs_k"]["rho"],
             res["rank_correlation_delta_vs_k"].get("backend")))
    g = res["growth_k_big_minus_k_small"]
    print("growth delta(k=%s)-delta(k=%s)=%.4f ci=[%.4f,%.4f] excludes_zero=%s"
          % (g["k_big"], g["k_small"], g["growth"], g["ci"][0], g["ci"][1],
             g["excludes_zero"]))
    print("monotone_nondecreasing=%s criterion_met=%s"
          % (res["monotone_nondecreasing"], res["criterion_met"]))

    # --- the planted effect must be RECOVERED ------------------------------------------
    assert deltas[-1] - deltas[0] > 0.05, \
        "planted horizon effect not recovered: delta(k=64)-delta(k=2)=%.4f" % (deltas[-1] - deltas[0])
    assert res["rank_correlation_delta_vs_k"]["rho"] >= 0.7, \
        "delta(k) is not rising with k (rho=%s)" % res["rank_correlation_delta_vs_k"]["rho"]
    assert g["excludes_zero"] and g["growth"] > 0, "growth CI failed to exclude zero"
    assert res["criterion_met"], "pre-registered criterion not met on a planted positive"
    # mean-pool must be the arm that decays; max-pool must decay MUCH less.
    # Not "max-pool is exactly flat": the max of n noise draws grows like sqrt(2*log n),
    # so a per-dimension max does lose a little as the window lengthens. That is real
    # behaviour of the estimator, not a defect, and asserting flatness to a hard
    # tolerance would be asserting something false. What the dilution mechanism actually
    # predicts -- and what is tested -- is that mean-pool decays substantially faster.
    # Observed on this synthetic at the time of writing: mean_drop 0.286, max_drop 0.104.
    mean_drop = mean_aucs[0] - mean_aucs[-1]
    max_drop = max_aucs[0] - max_aucs[-1]
    assert mean_drop > 0.05, \
        "mean_pool did not degrade with k (%.4f -> %.4f)" % (mean_aucs[0], mean_aucs[-1])
    assert mean_drop - max_drop > 0.10, \
        "max_pool degraded nearly as fast as mean_pool (%.4f vs %.4f) -- that is " \
        "horizon-wide information loss, not the dilution the sweep is built to detect" \
        % (max_drop, mean_drop)
    print("OK  planted horizon effect recovered: mean_pool drops %.4f, max_pool only "
          "%.4f, delta(k) rises" % (mean_drop, max_drop))

    # --- short-trajectory accounting is counted, per class, and adds up ----------------
    for c in res["cells"]:
        cov = c["coverage"]
        assert cov["n_bound"] + cov["n_shorter_than_k"] == cov["n_total"]
        assert cov["n_bound_class0"] + cov["n_bound_class1"] == cov["n_bound"]
    k16 = next(c for c in res["cells"] if c["k"] == 16)
    assert k16["coverage"]["n_shorter_than_k"] == 20, \
        "expected the 20 planted short trajectories to be unbound at k=16, got %d" \
        % k16["coverage"]["n_shorter_than_k"]
    assert res["cells"][-1]["k"] is None and res["cells"][-1]["coverage"]["n_bound"] == 0
    print("OK  short-trajectory accounting: %d/%d unbound at k=16, counted per class, "
          "never padded or dropped" % (k16["coverage"]["n_shorter_than_k"],
                                       k16["coverage"]["n_total"]))

    # --- k=all anchor: Truncate(None, MeanPool) IS the plain MeanPool row --------------
    plain = evaluate.evaluate_offline(corpus, embs, [MeanPool(seed=0)], seed=0, n_folds=4,
                                      bootstrap=200)
    res2 = run_within_corpus_horizon(corpus, embs, ks=[4], seed=0, n_folds=4, bootstrap=200,
                                     delta_bootstrap=100,
                                     main_ladder_aucs={"mean_pool": plain[0].auc})
    anchor = res2["anchors"]["k_all_matches_main_ladder"]
    assert anchor and anchor["matches"], "k=all anchor drifted from the main ladder: %s" % anchor
    assert res2["anchors"]["fast_auc_agrees_with_sklearn"]["agrees"]
    print("OK  anchors hold: k=all reproduces the plain MeanPool row exactly")

    # --- median split runs, reports its confound, and skips degenerate strata ----------
    ms = run_median_split_horizon(corpus, embs, seed=0, n_folds=4, bootstrap=200,
                                  delta_bootstrap=200)
    print("")
    print("median split at %.1f steps (corpus length_only_auc=%.3f):"
          % (ms["median_steps"], ms["corpus_length_only_auc"]))
    for stratum in ("short", "long"):
        b = ms["strata"][stratum]
        print("  %-5s n=%-4d pos=%-4d neg=%-4d steps med=%-6s len_auc=%-5.3f cv=%-5.3f %s"
              % (stratum, b["n"], b["n_pos"], b["n_neg"], b["steps"]["median"],
                 b["length_only_auc"], b["length_dispersion"]["cv"],
                 ("SKIPPED: %s" % b["reason"]) if b.get("skipped") else
                 ("delta=%.4f ci=[%.4f,%.4f]" % (b["delta"], b["delta_ci"][0],
                                                  b["delta_ci"][1]))))
    print("  delta_long_minus_short=%s" % ms["delta_long_minus_short"])
    assert set(ms["strata"]) == {"short", "long"}
    assert ms["strata"]["short"]["n"] + ms["strata"]["long"]["n"] == len(embs)
    assert not ms["strata"]["short"].get("skipped") and not ms["strata"]["long"].get("skipped")
    for b in ms["strata"].values():
        assert isinstance(b["delta"], float) and b["delta_ci"][0] is not None
        assert b["length_dispersion"]["cv"] is not None
    # The synthetic draws lengths class-independently, so length alone must be ~chance.
    # If this ever drifts, the split has become a label confound and its delta is not a
    # horizon reading at all.
    assert 0.40 <= ms["corpus_length_only_auc"] <= 0.60, \
        "length alone separates the synthetic classes (auc=%.3f) -- the median split " \
        "would be confounded" % ms["corpus_length_only_auc"]

    # NO assertion on the SIGN of delta_long_minus_short, deliberately. On this planted
    # positive the split comes out NEGATIVE while the truncation sweep comes out strongly
    # positive, and the reason is structural (see run_median_split_horizon's caveat): the
    # short half spans 12..median steps and the long half only median..max, so the halves
    # differ in length DISPERSION as well as mean horizon. Asserting the sign we wanted
    # would be pinning a synthetic artefact; asserting the sign observed would enshrine
    # one. What IS locked is that the split reports the confounds needed to see this.
    disp = {k: v["length_dispersion"]["cv"] for k, v in ms["strata"].items()}
    print("  NOTE: the median split DISSENTS from the planted ground truth here "
          "(long-minus-short < 0) while the truncation sweep recovers it. Length-spread "
          "cv short=%.3f long=%.3f -- the halves are not exchangeable in dispersion, "
          "which is why control 2 corroborates at best." % (disp["short"], disp["long"]))
    assert disp["short"] > disp["long"], \
        "expected the lower half of a median split to have the wider length spread"
    print("OK  median split: strata partition the corpus, per-stratum deltas + CIs and "
          "both confounds (length-only AUC, length dispersion) reported")

    print("")
    print("OK -- horizon.py self-test passed CPU-only, no model loaded")


if __name__ == "__main__":
    _self_test()
