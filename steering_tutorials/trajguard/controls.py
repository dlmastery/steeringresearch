"""controls.py -- the THREE confound controls trajguard was missing.

WHY THIS FILE EXISTS
--------------------
`README.md` section 12.3 ("What is still missing") named three gaps and
`AUDIT_2026-08.md` fixes #13 and #15 named two of them again. All three were *code*
gaps -- the controls did not exist, so the lesson's binding bar (`content` = 0.9103 on
the disguised substrate) was the worst of a set of bars that was itself incomplete:

  1. **MULTIVARIATE TRIVIAL BASELINE.** Every bar in `common/confound.py` and in
     `data.confound_report` is a SINGLE feature, and the binding bar is the worst of
     them one at a time. A method that beats each trivial feature individually can
     still be beaten by their COMBINATION -- and there is a concrete construction where
     that happens with every univariate bar sitting at chance (see :func:`_selftest`,
     case C: two features whose noise is shared and whose signal is antisymmetric).
     So "worst of univariate" is not "the strongest trivial baseline"; it is only the
     strongest trivial baseline *that uses one number*. This module adds a logistic
     regression over the joint feature vector, run under the SAME folds and the SAME
     bootstrap as the real methods, and folds its AUC into `worst_auc`. It is a peer in
     the ladder, not a footnote.

  2. **MATCHED-BIN CONTROL.** `shuffle` landed in the 2026-08-08 rewrite; matched-bin
     did not. The question it answers is the one a raw AUC cannot: does the method
     still separate the classes when completion LENGTH is held approximately fixed?
     Stratify into quantile bins of `charlen`, compute the AUC WITHIN each bin, and
     pool the bins by their pair counts (the stratified Mann-Whitney statistic). A
     method whose margin evaporates within bins was riding length, regardless of what
     the univariate length bar said -- a length bar at 0.5064 rules out a *monotone*
     length tell, not a length-conditional one.

  3. **PAIRED MARGIN CI AGAINST THE BINDING BAR.** `results_disguised.json` carries
     `vs_confound_paired_ci: null` on all four methods with a note explaining that the
     spine's `content` bar "exposes no per-item score". **That blocker was real but not
     fundamental, and it is solved here.** `common.confound.content_bar` computes a
     per-item out-of-fold score for every item (`scores[i] = <v, c_pos> - <v, c_neg>`)
     and then throws the vector away, returning only the pooled AUC.
     :func:`content_bar_scores` reproduces that loop using the spine's OWN primitives
     and returns the vector, then ASSERTS its pooled AUC equals the spine's to 1e-9. So
     the paired bootstrap CI is now computable against `content` and against
     `multivariate`, not only against the scalar bars.

     The clean long-term fix is a `return_scores=True` flag on
     `common.confound.content_bar`; `common/` is SHARED and this lesson does not edit
     it, so the reproduction-plus-anchor-assertion is the in-scope form. The assertion
     is the point: if the spine's content bar ever changes, this fails LOUDLY at the
     next run instead of silently pairing against a stale model (CLAUDE.md section
     18.8 -- "assert your anchors").

WHAT THIS FILE DOES NOT DO
--------------------------
It writes no parallel statistics stack. `kfold_indices`, `bootstrap_auc_ci` and
`paired_margin_ci` come from `run_trajguard`; `auc_raw`, `directionless`, `_tokenize`,
`_tfidf`, `_l2`, `_dot` and `content_bar` come from `common.confound`. The only new
primitives here are the quantile binning and the stratified (within-bin) AUC, because
nothing in the course computed either.

CPU-only. No model, no GPU, no torch. ASCII stdout (Windows cp1252 -- never print
alpha/Delta/norm glyphs).

Run the self-tests:      python -m steering_tutorials.trajguard.controls
Measure from the sidecar: python -m steering_tutorials.trajguard.controls --from-meta
"""
from __future__ import annotations

import json
import math
import random
from collections import Counter

import numpy as np

from steering_tutorials.common import confound as CF

from . import config as C

__all__ = [
    "TRIVIAL_FEATURES",
    "trivial_features",
    "trivial_features_from_meta",
    "multivariate_bar",
    "format_multivariate",
    "content_bar_scores",
    "bar_per_item_scores",
    "quantile_bins",
    "stratified_auc",
    "matched_bin_control",
    "matched_bin_block",
    "format_matched_bin",
]


# ---------------------------------------------------------------------------
# ITEM 1 -- the multivariate trivial baseline
# ---------------------------------------------------------------------------
# The four scalars the lesson already computes one at a time. `charlen` and
# `tokencount` are the spine's `length` and `count` bars; `mean_norm` and `final_norm`
# are this lesson's two geometry bars (`data._geometry_bars`). Every one of them is
# available with no model and no trajectory reading -- which is what makes their
# combination a TRIVIAL baseline rather than a rival method.
#
# DELIBERATELY EXCLUDED: the norm SLOPE (final_norm - first_norm) and any other
# difference across the trajectory. Those read how the residual stream MOVES, which is
# exactly the thing the detectors claim to read; putting them in the trivial bar would
# make the bar unbeatable by construction and would prove nothing about triviality.
# The exclusion is a judgement call and is recorded here rather than left implicit.
TRIVIAL_FEATURES = ("charlen", "tokencount", "mean_norm", "final_norm")


def _as_matrix(trajectories):
    """Normalise a ragged trajectory list to a list of 2-D [n_tokens, dim] arrays."""
    out = []
    for t in trajectories:
        a = np.asarray(t, dtype=np.float32)
        if a.ndim == 1:
            a = a[None, :]
        out.append(a)
    return out


def trivial_features(trajectories, completions):
    """The joint trivial feature matrix. Returns ``(names, X)`` with X of shape [n, 4].

    Column order is :data:`TRIVIAL_FEATURES` and is part of the contract -- the fitted
    coefficients are reported per name, so a silent reorder would mislabel them.
    """
    trajs = _as_matrix(trajectories)
    n = len(trajs)
    comps = list(completions) if completions is not None else [""] * n
    if len(comps) != n:
        raise ValueError("completions/trajectories length mismatch: %d vs %d"
                         % (len(comps), n))
    rows = []
    for a, c in zip(trajs, comps):
        norms = np.linalg.norm(a, axis=-1) if a.size else np.zeros(1, dtype=np.float32)
        rows.append([
            float(len(str(c))),
            float(a.shape[0]),
            float(norms.mean()),
            float(np.linalg.norm(a[-1])) if a.size else 0.0,
        ])
    return list(TRIVIAL_FEATURES), np.asarray(rows, dtype=float)


def trivial_features_from_meta(meta_path):
    """Same matrix, rebuilt from the COMMITTED text-free sidecar -- no GPU, no cache.

    `data.write_meta` records exactly these four scalars per completion, so the
    multivariate bar is reproducible from a fresh clone with no 100 MB hidden-state
    blob. Returns ``(names, X, labels)``.
    """
    blob = json.loads(open(str(meta_path), "r", encoding="utf-8").read())
    recs = blob.get("completions") or []
    if not recs:
        raise ValueError("sidecar %s carries no 'completions' records" % meta_path)
    X = np.asarray([[float(r["completion_chars"]), float(r["token_count"]),
                     float(r["mean_norm"]), float(r["final_norm"])] for r in recs],
                   dtype=float)
    y = np.asarray([int(r["label"]) for r in recs], dtype=int)
    return list(TRIVIAL_FEATURES), X, y


def multivariate_bar(X, labels, folds=None, seed: int = C.SEED, n_folds: int = C.N_FOLDS,
                     bootstrap: int = C.BOOTSTRAP, feature_names=None) -> dict:
    """Logistic regression over the JOINT trivial feature vector -- a peer in the ladder.

    Same CV folds as the methods when `folds` is supplied (the runner always supplies
    them), same pooled-out-of-fold scoring, same bootstrap CI machinery
    (`run_trajguard.bootstrap_auc_ci`). The scaler is fit on the TRAINING fold only
    (CLAUDE.md section 17: "fixed seeds with the scaler fit on train only") -- fitting it
    on all rows would leak the test fold's scale into the bar and inflate it, which is
    the one way a *confound* bar can cheat in the method's favour by being too strong.

    Returns the same shape as every other bar (``auc_raw`` / ``auc``) plus:
      ``auc_ci``      bootstrap 95% CI on the pooled out-of-fold AUC
      ``scores``      PER-ITEM out-of-fold scores in ORIGINAL index order -- this is what
                      makes a paired margin CI possible against this bar
      ``univariate``  each feature's own folded AUC, so "the combination beats each of
                      them individually" is checkable in the artifact, not asserted
      ``coef_mean``   mean standardised coefficient per feature (which trivial feature
                      is carrying the bar)
      ``sign_flipped`` True when the pooled raw AUC came back below 0.5 and the score
                      vector was negated so that higher == more harmful. Folding an AUC
                      is free for the bar; leaving the SCORES unfolded would silently
                      hand `paired_margin_ci` a backwards vector.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    from .run_trajguard import bootstrap_auc_ci, kfold_indices

    names = list(feature_names or TRIVIAL_FEATURES)
    X = np.asarray(X, dtype=float)
    y = np.asarray(labels).astype(int)
    n = len(y)
    if X.shape[0] != n:
        raise ValueError("X/labels length mismatch: %d vs %d" % (X.shape[0], n))
    if X.shape[1] != len(names):
        raise ValueError("X has %d columns but %d feature names were given"
                         % (X.shape[1], len(names)))
    if len(set(y.tolist())) < 2:
        return {"auc_raw": 0.5, "auc": 0.5, "note": "degenerate: a single class",
                "features": names}

    if folds is None:
        folds = kfold_indices(n, n_folds, seed, labels=y)

    scores = np.zeros(n, dtype=float)
    covered = np.zeros(n, dtype=bool)
    coefs = []
    for tr, te in folds:
        tr = np.asarray(tr)
        te = np.asarray(te)
        if len(set(y[tr].tolist())) < 2:
            continue
        scaler = StandardScaler().fit(X[tr])          # TRAIN ONLY -- see docstring
        lr = LogisticRegression(max_iter=2000, solver="lbfgs")
        lr.fit(scaler.transform(X[tr]), y[tr])
        scores[te] = lr.decision_function(scaler.transform(X[te]))
        covered[te] = True
        coefs.append(np.asarray(lr.coef_).reshape(-1))
    if not covered.all():
        raise RuntimeError("multivariate bar: %d/%d items got no out-of-fold score; the "
                           "fold set does not cover the data" % ((~covered).sum(), n))

    raw = CF.auc_raw(scores.tolist(), y.tolist())
    sign_flipped = bool(raw < 0.5)
    if sign_flipped:
        scores = -scores
        raw = 1.0 - raw
    point, lo, hi = bootstrap_auc_ci(y, scores, n=bootstrap, seed=seed)

    uni = {}
    for j, nm in enumerate(names):
        r = CF.auc_raw(X[:, j].tolist(), y.tolist())
        uni[nm] = {"auc_raw": float(r), "auc": float(CF.directionless(r))}
    best_uni_name = max(uni, key=lambda k: uni[k]["auc"])

    # COMPARABILITY. The univariate bars above are IN-SAMPLE: a fixed feature has no
    # parameters, so its AUC is unbiased and no CV is needed. The joint bar FITS five
    # numbers, so its honest estimate is the out-of-fold one -- which is penalised by
    # estimation noise and can legitimately land BELOW the best univariate bar at small
    # n. That is not a bug and it is not evidence the combination is useless; it means
    # the combination buys nothing beyond one feature AT THIS SAMPLE SIZE. To keep the
    # comparison from being read as like-for-like when it is not, the in-sample fit is
    # reported alongside and labelled OPTIMISTIC. `auc` -- the number that competes for
    # the binding bar -- is always the out-of-fold one; the optimistic figure never
    # binds anything.
    scaler_all = StandardScaler().fit(X)
    lr_all = LogisticRegression(max_iter=2000, solver="lbfgs")
    lr_all.fit(scaler_all.transform(X), y)
    raw_in = CF.auc_raw(lr_all.decision_function(scaler_all.transform(X)).tolist(),
                        y.tolist())

    coef_mean = (np.mean(np.vstack(coefs), axis=0) if coefs
                 else np.zeros(len(names), dtype=float))
    auc = float(CF.directionless(raw))
    return {
        "auc_raw": float(raw),
        "auc": auc,
        "auc_ci": [float(lo), float(hi)],
        "n_folds": len(folds),
        "features": names,
        "coef_mean": {nm: float(v) for nm, v in zip(names, coef_mean.tolist())},
        "auc_in_sample": float(CF.directionless(raw_in)),
        "univariate": uni,
        "univariate_estimator": "in-sample (a fixed feature has no parameters)",
        "joint_estimator": "pooled out-of-fold (5 fitted parameters)",
        "best_univariate": {"name": best_uni_name, "auc": uni[best_uni_name]["auc"]},
        "gain_over_best_univariate": auc - uni[best_uni_name]["auc"],
        "gain_over_best_univariate_in_sample": (float(CF.directionless(raw_in))
                                                - uni[best_uni_name]["auc"]),
        "comparability_note": (
            "`gain_over_best_univariate` compares an OUT-OF-FOLD joint AUC against "
            "IN-SAMPLE univariate AUCs. A negative gain therefore does NOT mean the "
            "combination is worse than its parts -- it means the combination buys "
            "nothing beyond the best single feature at this n, once the cost of "
            "estimating its coefficients is paid. The like-for-like in-sample "
            "comparison is `gain_over_best_univariate_in_sample`. Only `auc` (the "
            "out-of-fold figure) competes for the binding bar."),
        "sign_flipped": sign_flipped,
        "scores": [round(float(s), 6) for s in scores.tolist()],
        "note": "logistic regression on {%s}, scaler fit on TRAIN fold only, pooled "
                "out-of-fold, folded directionless. A TRIVIAL baseline: every input is "
                "available without reading the trajectory."
                % ", ".join(names),
    }


def format_multivariate(mv: dict) -> str:
    """ASCII summary. NEVER prints unicode -- the Windows cp1252 console dies on it."""
    if not isinstance(mv, dict) or "auc" not in mv:
        return "  multivariate: n/a (%s)" % (mv or {}).get("error", "not computed")
    lines = ["  -- multivariate trivial baseline (ITEM 1: the COMBINATION of the "
             "scalar bars) --"]
    for nm in mv.get("features", []):
        u = (mv.get("univariate") or {}).get(nm) or {}
        lines.append("  %-12s univariate %-7.4f   coef %+.3f"
                     % (nm, u.get("auc", float("nan")),
                        (mv.get("coef_mean") or {}).get(nm, float("nan"))))
    ci = mv.get("auc_ci") or [float("nan"), float("nan")]
    bu = (mv.get("best_univariate") or {}).get("name", "?")
    lines.append("  %-12s JOINT      %-7.4f   ci [%.4f, %.4f]  (out-of-fold -- this is "
                 "the number that competes for the binding bar)"
                 % ("multivariate", mv["auc"], ci[0], ci[1]))
    if "auc_in_sample" in mv:
        lines.append("  %-12s JOINT      %-7.4f   (in-sample, OPTIMISTIC -- shown only "
                     "because the univariate column above is also in-sample)"
                     % ("", mv["auc_in_sample"]))
        lines.append("  gain over best univariate (%s): %+.4f out-of-fold, %+.4f "
                     "like-for-like in-sample"
                     % (bu, mv.get("gain_over_best_univariate", float("nan")),
                        mv.get("gain_over_best_univariate_in_sample", float("nan"))))
    else:
        lines.append("  gain over best univariate (%s): %+.4f"
                     % (bu, mv.get("gain_over_best_univariate", float("nan"))))
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# ITEM 3 -- per-item scores for the CONTENT bar (the paired-CI blocker)
# ---------------------------------------------------------------------------
def content_bar_scores(texts, labels, seed: int = 0, n_folds: int = 5,
                       max_features: int = 20000) -> dict:
    """Per-item OUT-OF-FOLD scores for `common.confound.content_bar`, plus an ANCHOR.

    The spine computes `scores[i] = <tfidf_i, centroid_pos> - <tfidf_i, centroid_neg>`
    for every item under group-free K-fold CV and returns only the pooled AUC. This
    reproduces that loop -- same fold construction, same RNG seed, same vocabulary /
    IDF / centroid arithmetic, all of it through the spine's OWN helpers -- and returns
    the score vector, which is what a PAIRED bootstrap CI needs.

    Then it asserts equality with the spine's own pooled AUC to 1e-9. The reproduction
    is only admissible while that assertion holds: if `common/confound.py` changes, this
    raises instead of quietly pairing the methods against a different model than the one
    that set the bar.

    Returns ``{"auc_raw", "auc", "n_folds", "scores", "anchor_ok", "anchor_delta"}``.
    ``scores`` are in ORIGINAL index order and oriented so higher == more harmful.
    """
    labels = [int(y) for y in labels]
    docs = [CF._tokenize(CF._flatten(t)) for t in texts]
    n = len(docs)
    if n == 0 or len(set(labels)) < 2:
        return {"auc_raw": 0.5, "auc": 0.5, "n_folds": 0, "scores": [0.0] * n,
                "anchor_ok": True, "anchor_delta": 0.0,
                "note": "degenerate: single class or empty"}

    # --- the spine's fold construction, verbatim -----------------------------
    rng = random.Random(seed)
    idx = list(range(n))
    rng.shuffle(idx)
    folds = [idx[i::n_folds] for i in range(n_folds)]

    scores = [0.0] * n
    used = 0
    for f in range(n_folds):
        test = folds[f]
        train = [i for i in idx if i not in set(test)]
        if not test or not train:
            continue
        if len(set(labels[i] for i in train)) < 2:
            continue
        used += 1

        df = Counter()
        for i in train:
            for tok in set(docs[i]):
                df[tok] += 1
        vocab = {t: k for k, (t, _) in enumerate(df.most_common(max_features))}
        n_train = len(train)
        idf = {t: math.log((1.0 + n_train) / (1.0 + df[t])) + 1.0 for t in vocab}

        cpos, cneg = {}, {}
        npos = nneg = 0
        for i in train:
            v = CF._tfidf(docs[i], vocab, idf)
            tgt = cpos if labels[i] == 1 else cneg
            for k, val in v.items():
                tgt[k] = tgt.get(k, 0.0) + val
            if labels[i] == 1:
                npos += 1
            else:
                nneg += 1
        for c, cnt in ((cpos, npos), (cneg, nneg)):
            if cnt:
                for k in c:
                    c[k] /= cnt
        CF._l2(cpos)
        CF._l2(cneg)

        for i in test:
            v = CF._tfidf(docs[i], vocab, idf)
            CF._l2(v)
            scores[i] = CF._dot(v, cpos) - CF._dot(v, cneg)

    raw = CF.auc_raw(scores, labels)

    # --- THE ANCHOR: this reproduction must equal the spine, or it is not the bar ---
    ref = CF.content_bar(texts, labels, seed=seed, n_folds=n_folds,
                         max_features=max_features)
    delta = abs(float(raw) - float(ref.get("auc_raw", float("nan"))))
    if not (delta <= 1e-9):
        raise AssertionError(
            "content_bar_scores no longer reproduces common.confound.content_bar "
            "(auc_raw %.12f vs %.12f, delta %.3e). The per-item scores would be paired "
            "against a DIFFERENT model than the one setting the binding bar. Re-sync "
            "this function with the spine before trusting any paired CI."
            % (raw, ref.get("auc_raw", float("nan")), delta))

    sign_flipped = bool(raw < 0.5)
    if sign_flipped:
        scores = [-s for s in scores]
        raw = 1.0 - raw
    return {
        "auc_raw": float(raw),
        "auc": float(CF.directionless(raw)),
        "n_folds": used,
        "scores": [round(float(s), 8) for s in scores],
        "sign_flipped": sign_flipped,
        "anchor_ok": True,
        "anchor_delta": float(delta),
        "note": "per-item out-of-fold centroid-cosine scores reproduced from "
                "common.confound.content_bar and asserted equal to it (delta %.2e)"
                % delta,
    }


def bar_per_item_scores(name, trajectories, completions, labels=None, multivariate=None,
                        seed: int = C.SEED, n_folds: int = C.N_FOLDS):
    """Per-item scores for ANY binding bar, or ``None`` when the bar genuinely has none.

    Extends `run_trajguard._bar_feature` (which covers the four scalar bars) to the two
    bars that previously blocked the paired CI:

      ``content``      -> :func:`content_bar_scores` (ITEM 3: the blocker, solved)
      ``multivariate`` -> the pooled out-of-fold LR scores from :func:`multivariate_bar`

    Returns ``(scores, provenance)``. ``provenance`` names which bar the scores belong
    to so the caller can LABEL the CI -- a paired CI against `final_norm` and one against
    `content` are different claims and must never be printed under the same header.
    """
    from .run_trajguard import _bar_feature

    if not name:
        return None, None
    if name == "content":
        if completions is None or labels is None:
            return None, None
        blk = content_bar_scores([str(c) for c in completions], labels, seed=seed,
                                 n_folds=n_folds)
        return blk["scores"], "content"
    if name == "multivariate":
        if not isinstance(multivariate, dict) or "scores" not in multivariate:
            return None, None
        return multivariate["scores"], "multivariate"
    scalar = _bar_feature(name, trajectories, completions)
    return (scalar, name) if scalar is not None else (None, None)


# ---------------------------------------------------------------------------
# ITEM 2 -- the matched-bin control
# ---------------------------------------------------------------------------
def quantile_bins(values, n_bins: int):
    """Quantile bin index per item. Returns ``(bin_idx, edges)``.

    Ties collapse the edge set (a heavily tied feature cannot support `n_bins` distinct
    bins and pretending otherwise would produce empty strata), so the ACHIEVED bin count
    is ``len(edges) - 1`` and is reported, never assumed.
    """
    v = np.asarray(values, dtype=float)
    if v.size == 0:
        return np.zeros(0, dtype=int), np.zeros(0, dtype=float)
    qs = np.quantile(v, np.linspace(0.0, 1.0, int(max(2, n_bins)) + 1))
    edges = np.unique(qs)
    if edges.size < 3:                       # not enough distinct values to stratify
        return np.zeros(v.size, dtype=int), np.asarray([v.min(), v.max()], dtype=float)
    idx = np.clip(np.searchsorted(edges, v, side="left") - 1, 0, edges.size - 2)
    return idx.astype(int), edges


def _auc_fast(y_true, y_score):
    """Rank-based AUC on numpy arrays. Falls back to the spine when scipy is absent.

    Identical statistic to `common.confound.auc_raw` (Mann-Whitney U with average ranks
    for ties); this exists only because the matched-bin CI evaluates it 10k times.
    """
    y = np.asarray(y_true).astype(int)
    n_pos = int(y.sum())
    n_neg = int(y.size - n_pos)
    if n_pos == 0 or n_neg == 0:
        return 0.5
    try:
        from scipy.stats import rankdata
        r = rankdata(np.asarray(y_score, dtype=float))
    except Exception:                                       # pragma: no cover
        return CF.auc_raw(list(np.asarray(y_score, dtype=float)), list(y))
    u = float(r[y == 1].sum()) - n_pos * (n_pos + 1) / 2.0
    return float(u / (n_pos * n_neg))


def stratified_auc(y_true, y_score, bin_idx):
    """The pooled WITHIN-BIN AUC: sum of per-bin U over the sum of per-bin pair counts.

    This is the stratified Mann-Whitney statistic -- the probability that a random
    harmful item outranks a random benign item DRAWN FROM THE SAME BIN. It is the right
    pooling: averaging per-bin AUCs weights a 4-item bin like a 200-item one, and
    folding each bin about 0.5 before pooling would let every bin pick its own favourable
    sign and manufacture separation out of noise. So the fold happens ONCE, on the pooled
    statistic, and the per-bin folded values below are DISPLAY ONLY.

    Returns ``(pooled_raw, per_bin)``; ``pooled_raw`` is NaN when no bin has both classes.
    """
    y = np.asarray(y_true).astype(int)
    s = np.asarray(y_score, dtype=float)
    b = np.asarray(bin_idx).astype(int)
    num = 0.0
    den = 0.0
    per_bin = []
    for bb in sorted(set(b.tolist())):
        m = b == bb
        yt = y[m]
        n_pos = int(yt.sum())
        n_neg = int(yt.size - n_pos)
        cell = {"bin": int(bb), "n": int(yt.size), "n_pos": n_pos, "n_neg": n_neg}
        if n_pos == 0 or n_neg == 0:
            cell["auc_raw"] = None
            cell["auc"] = None
            cell["note"] = "single-class bin: contributes 0 pairs, excluded from the pool"
        else:
            a = _auc_fast(yt, s[m])
            cell["auc_raw"] = float(a)
            cell["auc"] = float(CF.directionless(a))   # display only -- see docstring
            num += a * n_pos * n_neg
            den += n_pos * n_neg
        per_bin.append(cell)
    pooled = float(num / den) if den > 0 else float("nan")
    return pooled, per_bin


def matched_bin_control(y_true, y_score, stratifier, n_bins: int = None,
                        n_boot: int = None, seed: int = C.SEED,
                        stratifier_name: str = "charlen") -> dict:
    """Does this score still separate the classes at approximately FIXED length?

    `stratifier` is the per-item value to hold fixed (the runner passes completion
    character length, aligned to the pooled out-of-fold order). Every argument is in the
    SAME item order.

    The CI resamples WITHIN bins, which is the resampling scheme that matches the
    statistic: a plain bootstrap over all items would reshuffle the bin composition and
    put length variation back into the interval the control exists to remove.

    Returns raw AUC, pooled within-bin AUC, their delta, per-bin n / AUC, and the CI.
    A large negative ``delta`` is the finding: the method was riding the stratifier.
    """
    y = np.asarray(y_true).astype(int)
    s = np.asarray(y_score, dtype=float)
    v = np.asarray(stratifier, dtype=float)
    if not (y.size == s.size == v.size):
        raise ValueError("matched_bin_control: length mismatch %d/%d/%d"
                         % (y.size, s.size, v.size))
    n_bins = int(n_bins or getattr(C, "MATCHED_BINS", 4))
    n_boot = int(n_boot if n_boot is not None else getattr(C, "MATCHED_BIN_BOOTSTRAP",
                                                           C.BOOTSTRAP))

    bin_idx, edges = quantile_bins(v, n_bins)
    raw_all = _auc_fast(y, s)
    pooled_raw, per_bin = stratified_auc(y, s, bin_idx)
    raw = float(CF.directionless(raw_all))
    pooled = float(CF.directionless(pooled_raw)) if pooled_raw == pooled_raw else float("nan")

    # --- within-bin bootstrap CI on the pooled statistic ---------------------
    ci = [float("nan"), float("nan")]
    if n_boot > 0 and pooled_raw == pooled_raw:
        rng = np.random.default_rng(seed)
        members = [np.flatnonzero(bin_idx == bb) for bb in sorted(set(bin_idx.tolist()))]
        boots = []
        for _ in range(n_boot):
            pick = np.concatenate([mem[rng.integers(0, mem.size, mem.size)]
                                   for mem in members if mem.size])
            p, _pb = stratified_auc(y[pick], s[pick], bin_idx[pick])
            if p == p:
                boots.append(CF.directionless(p))
        if boots:
            ci = [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))]

    return {
        "stratifier": stratifier_name,
        "n_bins_requested": n_bins,
        "n_bins_achieved": max(0, int(edges.size) - 1),
        "bin_edges": [float(e) for e in edges.tolist()],
        "auc_raw_unstratified": float(raw_all),
        "auc_unstratified": raw,
        "auc_within_bin_raw": float(pooled_raw),
        "auc_within_bin": pooled,
        "auc_within_bin_ci": ci,
        "delta_vs_unstratified": (pooled - raw) if pooled == pooled else float("nan"),
        "per_bin": per_bin,
        "n_boot": int(n_boot),
        "note": "pooled = stratified Mann-Whitney (sum of per-bin U / sum of per-bin "
                "pair counts), folded ONCE at the end; CI resamples WITHIN bins. A "
                "margin that evaporates here was riding %s." % stratifier_name,
    }


def matched_bin_block(method_scores, bar_scores, stratifier, n_bins: int = None,
                      n_boot: int = None, seed: int = C.SEED,
                      stratifier_name: str = "charlen") -> dict:
    """Run :func:`matched_bin_control` for every method AND for the binding bar.

    `method_scores` maps ``name -> (y_true, y_score)``; `bar_scores` is
    ``(name, y_true, y_score)`` or None. Running the bar through the SAME control is the
    point: within-bin numbers are only comparable to other within-bin numbers, and a
    within-bin method AUC beside an unstratified bar is not a margin.
    """
    out = {"stratifier": stratifier_name, "methods": {}, "bar": None}
    for name, (yt, ys) in (method_scores or {}).items():
        try:
            out["methods"][name] = matched_bin_control(
                yt, ys, stratifier, n_bins=n_bins, n_boot=n_boot, seed=seed,
                stratifier_name=stratifier_name)
        except Exception as exc:
            out["methods"][name] = {"error": str(exc)}
    if bar_scores is not None:
        bname, byt, bys = bar_scores
        try:
            blk = matched_bin_control(byt, bys, stratifier, n_bins=n_bins,
                                      n_boot=n_boot, seed=seed,
                                      stratifier_name=stratifier_name)
            blk["bar_name"] = bname
            out["bar"] = blk
        except Exception as exc:
            out["bar"] = {"bar_name": bname, "error": str(exc)}

    # the only comparison that is like-for-like
    bar_wb = ((out.get("bar") or {}).get("auc_within_bin"))
    if isinstance(bar_wb, float) and bar_wb == bar_wb:
        for name, cell in out["methods"].items():
            wb = cell.get("auc_within_bin")
            if isinstance(wb, float) and wb == wb:
                cell["within_bin_margin_over_bar"] = wb - bar_wb
                cell["within_bin_margin_bar_name"] = (out["bar"] or {}).get("bar_name")
    return out


def format_matched_bin(block: dict) -> str:
    """ASCII summary of the matched-bin control. No unicode."""
    if not isinstance(block, dict):
        return "  matched-bin: n/a"
    lines = ["  -- matched-bin control (ITEM 2: AUC WITHIN quantile bins of %s) --"
             % block.get("stratifier", "?")]
    lines.append("  %-20s %8s %8s %8s  %s"
                 % ("row", "raw", "within", "delta", "within-bin 95% CI"))

    def _row(tag, cell):
        if not isinstance(cell, dict) or "auc_within_bin" not in cell:
            return "  %-20s [FAILED: %s]" % (tag, (cell or {}).get("error", "n/a"))
        ci = cell.get("auc_within_bin_ci") or [float("nan"), float("nan")]
        return ("  %-20s %8.4f %8.4f %+8.4f  [%.4f, %.4f]"
                % (tag, cell["auc_unstratified"], cell["auc_within_bin"],
                   cell["delta_vs_unstratified"], ci[0], ci[1]))

    bar = block.get("bar")
    if bar:
        lines.append(_row("BAR:%s" % bar.get("bar_name", "?"), bar))
    for name, cell in (block.get("methods") or {}).items():
        lines.append(_row(name, cell))
    ref = bar or next(iter((block.get("methods") or {}).values()), None)
    if isinstance(ref, dict) and ref.get("per_bin"):
        lines.append("  bins (n / n_pos / n_neg): %s"
                     % "  ".join("[%d] %d/%d/%d" % (c["bin"], c["n"], c["n_pos"],
                                                    c["n_neg"])
                                 for c in ref["per_bin"]))
        lines.append("  edges: %s" % ", ".join("%.0f" % e
                                               for e in (ref.get("bin_edges") or [])))
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CPU SELF-TESTS -- synthetic data with a PLANTED confound. No model, no GPU.
# ---------------------------------------------------------------------------
def _synth_length_confound(n=240, seed=0):
    """Class 1 is longer; the 'method' score is a monotone function of length + noise.

    This is the failure the matched-bin control exists to catch: a detector that has
    learned nothing except 'harmful completions run long'.
    """
    rng = np.random.default_rng(seed)
    y = np.array([1] * (n // 2) + [0] * (n - n // 2))
    charlen = np.where(y == 1, rng.normal(900, 120, n), rng.normal(600, 120, n))
    charlen = np.clip(charlen, 50, None)
    score = charlen + rng.normal(0, 40, n)        # NOTHING but length
    return y, score, charlen


def _synth_real_signal(n=240, seed=1):
    """Class 1 is longer, but the 'method' score is INDEPENDENT of length.

    The control must NOT destroy this one -- a stratification that flattens genuine
    signal is not a control, it is a bug.
    """
    rng = np.random.default_rng(seed)
    y = np.array([1] * (n // 2) + [0] * (n - n // 2))
    charlen = np.where(y == 1, rng.normal(900, 120, n), rng.normal(600, 120, n))
    charlen = np.clip(charlen, 50, None)
    score = np.where(y == 1, rng.normal(1.0, 0.5, n), rng.normal(-1.0, 0.5, n))
    return y, score, charlen


def _synth_joint_confound(n=300, seed=2, dim=8):
    """Two trivial features, EACH at chance, whose COMBINATION separates perfectly.

    The exact construction ITEM 1 is about. Shared noise ``z`` with large variance
    dominates each feature marginally, while the class signal ``s`` enters with opposite
    sign, so ``charlen - mean_norm`` is a clean linear separator that no univariate bar
    can see. Returns (trajectories, completions, labels) so the REAL
    `data.confound_report` + `multivariate_bar` path is exercised, not a stand-in.
    """
    rng = np.random.default_rng(seed)
    y = np.array([1] * (n // 2) + [0] * (n - n // 2))
    z = rng.normal(0.0, 10.0, n)
    s = np.where(y == 1, 1.5, -1.5)
    a = z + s                                   # -> completion character length
    b = z - s                                   # -> mean/final hidden-state norm
    charlen = np.clip(np.round(600 + 12 * a), 20, None).astype(int)
    norm = 100.0 + 2.0 * b

    unit = np.zeros(dim, dtype=np.float32)
    unit[0] = 1.0
    trajs, comps = [], []
    for i in range(n):
        trajs.append(np.tile(unit * float(norm[i]), (6, 1)))   # fixed 6 tokens -> count bar 0.5
        # identical filler token, so TF-IDF sees one L2-normalised direction for every
        # item and the CONTENT bar cannot fire on lexical difference.
        reps = max(1, int(charlen[i]) // 5)
        comps.append(("word " * reps)[:int(charlen[i])])
    return trajs, comps, y


def _check(tag, ok, detail=""):
    print("  [%s] %s %s" % ("PASS" if ok else "FAIL", tag, detail))
    return bool(ok)


def _selftest() -> int:
    """Every assertion this module makes about itself. CPU-only, seconds."""
    from . import data as D

    print("=" * 78)
    print("trajguard/controls.py -- CPU SELF-TESTS (synthetic data, no model, no GPU)")
    print("=" * 78)
    ok = True
    boot = 400          # the self-test does not need 10k resamples to check a direction

    # --- A. matched-bin REMOVES a planted length confound --------------------
    print("\nA. planted length confound: the score IS length. matched-bin must kill it.")
    y, s, L = _synth_length_confound()
    mb = matched_bin_control(y, s, L, n_bins=4, n_boot=boot, seed=0)
    print(format_matched_bin({"stratifier": "charlen",
                              "methods": {"length_only_detector": mb}, "bar": None}))
    ok &= _check("raw AUC is high", mb["auc_unstratified"] > 0.85,
                 "raw=%.4f" % mb["auc_unstratified"])
    ok &= _check("within-bin AUC drops hard", mb["auc_within_bin"] < 0.80,
                 "within=%.4f" % mb["auc_within_bin"])
    ok &= _check("delta is a large drop", mb["delta_vs_unstratified"] < -0.15,
                 "delta=%+.4f" % mb["delta_vs_unstratified"])
    ok &= _check("achieved 4 bins", mb["n_bins_achieved"] == 4,
                 "bins=%d" % mb["n_bins_achieved"])

    # A2. The control's POWER IS BOUNDED BY BIN WIDTH, and that is a property to state,
    # not to hide behind a lucky threshold. A 4-bin control on a continuous confound
    # leaves a residual INSIDE each bin; refining the bins removes it. If a lesson
    # reports a within-bin AUC it must also report how many bins produced it.
    print("   resolution sweep (same data, same score -- only the bin count changes):")
    sweep = []
    for nb in (2, 4, 8, 12, 20):
        m = matched_bin_control(y, s, L, n_bins=nb, n_boot=0, seed=0)
        sweep.append(m["auc_within_bin"])
        print("     %2d bins -> within %.4f (raw %.4f)"
              % (m["n_bins_achieved"], m["auc_within_bin"], m["auc_unstratified"]))
    ok &= _check("refining bins monotonically removes more of the confound",
                 all(sweep[i] >= sweep[i + 1] - 1e-9 for i in range(len(sweep) - 1)),
                 "sweep=%s" % ["%.3f" % v for v in sweep])
    ok &= _check("at fine resolution the planted confound is gone", sweep[-1] < 0.60,
                 "within@20bins=%.4f" % sweep[-1])

    # --- B. matched-bin PRESERVES a length-independent signal ----------------
    print("\nB. real signal, same length gap. matched-bin must NOT destroy it.")
    y2, s2, L2 = _synth_real_signal()
    mb2 = matched_bin_control(y2, s2, L2, n_bins=4, n_boot=boot, seed=0)
    print(format_matched_bin({"stratifier": "charlen",
                              "methods": {"real_detector": mb2}, "bar": None}))
    ok &= _check("raw AUC is high", mb2["auc_unstratified"] > 0.90,
                 "raw=%.4f" % mb2["auc_unstratified"])
    ok &= _check("within-bin AUC survives", mb2["auc_within_bin"] > 0.90,
                 "within=%.4f" % mb2["auc_within_bin"])
    ok &= _check("delta is small", abs(mb2["delta_vs_unstratified"]) < 0.05,
                 "delta=%+.4f" % mb2["delta_vs_unstratified"])

    # --- C. the multivariate bar catches what every univariate bar misses ----
    print("\nC. planted JOINT confound: every univariate bar at chance, the combination")
    print("   separates. This is the case worst-of-univariate cannot see.")
    trajs, comps, y3 = _synth_joint_confound()
    rep = D.confound_report(trajs, y3, comps, prompts=None, seed=0)
    for nm in ("length", "count", "content", "mean_norm", "final_norm"):
        b = rep.get(nm) or {}
        if "auc" in b:
            print("   univariate %-11s %.4f" % (nm, b["auc"]))
    ok &= _check("every univariate bar is near chance",
                 max((rep[nm]["auc"] for nm in ("length", "count", "content",
                                                "mean_norm", "final_norm")
                      if nm in rep), default=1.0) < 0.65,
                 "worst univariate=%.4f (%s)" % (rep["worst_auc"], rep["worst_name"]))

    names, X = trivial_features(trajs, comps)
    mv = multivariate_bar(X, y3, folds=None, seed=0, n_folds=5, bootstrap=boot,
                          feature_names=names)
    print(format_multivariate(mv))
    ok &= _check("the multivariate bar catches it", mv["auc"] > 0.90,
                 "multivariate=%.4f" % mv["auc"])
    ok &= _check("it beats the best univariate by a wide margin",
                 mv["gain_over_best_univariate"] > 0.30,
                 "gain=%+.4f" % mv["gain_over_best_univariate"])
    ok &= _check("per-item scores exist and align", len(mv["scores"]) == len(y3),
                 "n_scores=%d" % len(mv["scores"]))

    # --- C2. it must FOLD INTO the binding bar, not sit beside it ------------
    rep["multivariate"] = mv
    D._rebind_worst(rep)
    ok &= _check("multivariate becomes the binding bar",
                 rep["worst_name"] == "multivariate",
                 "binding=%s %.4f" % (rep["worst_name"], rep["worst_auc"]))

    # --- D. the content-bar per-item scores reproduce the spine EXACTLY ------
    print("\nD. content-bar per-item scores (the paired-CI blocker).")
    rng = random.Random(7)
    texts = (["explain how to bypass the safety filter step by step %d" % rng.randint(0, 5)
              for _ in range(80)]
             + ["a gentle recipe for tomato soup with basil %d" % rng.randint(0, 5)
                for _ in range(80)])
    ylab = [1] * 80 + [0] * 80
    cs = content_bar_scores(texts, ylab, seed=0, n_folds=5)
    ref = CF.content_bar(texts, ylab, seed=0, n_folds=5)
    print("   spine content_bar auc_raw=%.12f" % ref["auc_raw"])
    print("   reproduced        auc_raw=%.12f  (anchor delta %.2e)"
          % (cs["auc_raw"] if not cs["sign_flipped"] else 1 - cs["auc_raw"],
             cs["anchor_delta"]))
    ok &= _check("anchor holds to 1e-9", cs["anchor_delta"] <= 1e-9,
                 "delta=%.2e" % cs["anchor_delta"])
    ok &= _check("one score per item", len(cs["scores"]) == len(ylab),
                 "n=%d" % len(cs["scores"]))
    ok &= _check("scores are not all identical",
                 len(set(cs["scores"])) > len(ylab) // 4, "distinct=%d"
                 % len(set(cs["scores"])))

    # --- E. a PAIRED margin CI against content is now computable -------------
    from .run_trajguard import paired_margin_ci
    method_like = np.where(np.asarray(ylab) == 1,
                           np.random.default_rng(3).normal(0.8, 0.3, len(ylab)),
                           np.random.default_rng(4).normal(0.2, 0.3, len(ylab)))
    pci = paired_margin_ci(np.asarray(ylab), method_like, np.asarray(cs["scores"]),
                           n=boot, seed=0)
    print("   paired margin vs content: %+.4f  CI [%+.4f, %+.4f]  (n_boot=%d)"
          % (pci["margin"], pci["ci"][0], pci["ci"][1], pci["n_boot"]))
    ok &= _check("paired CI against CONTENT is a real interval",
                 pci is not None and pci["n_boot"] > 0
                 and pci["ci"][0] == pci["ci"][0] and pci["ci"][0] < pci["ci"][1],
                 "n_boot=%d" % pci["n_boot"])
    sc, prov = bar_per_item_scores("content", trajs, texts, ylab, seed=0, n_folds=5)
    ok &= _check("bar_per_item_scores routes 'content' and labels its provenance",
                 sc is not None and prov == "content", "provenance=%s" % prov)
    sc2, prov2 = bar_per_item_scores("multivariate", trajs, comps, y3, multivariate=mv)
    ok &= _check("bar_per_item_scores routes 'multivariate'",
                 sc2 is not None and prov2 == "multivariate", "provenance=%s" % prov2)

    print("\n" + "=" * 78)
    print("SELF-TESTS: %s" % ("ALL PASS" if ok else "FAILURES ABOVE"))
    print("=" * 78)
    return 0 if ok else 1


def _from_meta() -> int:
    """Measure the multivariate bar from the COMMITTED sidecar. CPU-only, no GPU.

    `data.write_meta` stores exactly the four trivial scalars per completion, so ITEM 1
    is measurable on the shipped artifact today, with no hidden-state cache and no model
    load. The matched-bin control on the METHODS is not, because `results_*.json` stores
    per-method AUCs and not per-method out-of-fold score vectors -- that arm needs the
    full runner. Stated rather than approximated.
    """
    path = C.META_PATH
    if not path.exists():
        print("[from-meta] no sidecar at %s -- run the lesson first." % path)
        return 1
    names, X, y = trivial_features_from_meta(path)
    print("[from-meta] %s  n=%d (pos=%d neg=%d)" % (path.name, len(y), int(y.sum()),
                                                    int((y == 0).sum())))
    mv = multivariate_bar(X, y, folds=None, seed=C.SEED, n_folds=C.N_FOLDS,
                          bootstrap=C.BOOTSTRAP, feature_names=names)
    print(format_multivariate(mv))
    print("  READ: this is the TRIVIAL joint baseline. It is a bar the trajectory")
    print("  methods must clear, not a method. It does NOT include the content bar,")
    print("  which is measured separately from the completion text (GPU run required).")
    mb = matched_bin_control(y, np.asarray(mv["scores"], dtype=float), X[:, 0],
                             n_bins=getattr(C, "MATCHED_BINS", 4),
                             n_boot=C.BOOTSTRAP, seed=C.SEED)
    print(format_matched_bin({"stratifier": "charlen",
                              "methods": {"multivariate": mb}, "bar": None}))
    return 0


if __name__ == "__main__":
    import sys

    raise SystemExit(_from_meta() if "--from-meta" in sys.argv else _selftest())
