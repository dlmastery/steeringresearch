"""common/confound.py -- the ONE shared confound audit every detection lesson imports.

WHY THIS FILE EXISTS
--------------------
Four lessons (`biencoder_guard`, `cross_trajectory`, `multiturn_jailbreak`, `trajguard`)
each grew their own `confound_report`, and each implemented a DIFFERENT SUBSET of the
discipline. The 2026-08-08 audit found:

  lesson              fold max(auc,1-auc)   count bar   content bar   shuffle
  trajguard           yes                   n/a         no            no
  multiturn_jailbreak yes                   yes         no            no
  cross_trajectory    **NO**                yes         no            no
  biencoder_guard     **NO**                **absent**  no            no

and -- worse -- `CONFOUND_DISCIPLINE.md` section 4 printed a folded `confound_auc` and
declared "the pattern already exists as ``confound_report()`` in
``cross_trajectory/data.py`` -- this is the canonical form." **It was not.** That function
returns the RAW auc and never folds. The course document described code that had never
existed, and it is the document every future lesson copies from.

The live cost of the missing fold: `cross_trajectory`'s easy condition measured
``totalchar_auc = 0.10975``. Read raw that looks CLEANER than chance. Folded it is
**0.890** -- a near-perfect length tell with the sign flipped. The lesson README printed
AUCs of 0.991-0.998 against no bar at all, so a +0.10 margin was presented as +0.99.

THE RULE THIS FILE ENFORCES (CLAUDE.md section 17 rule 7)
---------------------------------------------------------
Claim only the margin ABOVE the larger of {method baseline, confound bar}. A confound bar
is **directionless**: a feature that predicts the NEGATIVE class perfectly is exactly as
damning as one that predicts the positive class perfectly, because a classifier is free to
learn the sign. Therefore every bar reported here is ``max(auc, 1 - auc)``.

FOUR BARS, NOT ONE
------------------
A length-only audit is not sufficient. `hello_world`'s reference recipe runs four controls
and the detection lessons run one. This module provides all four:

1. **length**  -- can raw character count separate the classes?
2. **count**   -- can the number of units (turns / trajectories / tokens) separate them?
3. **content** -- can a bag-of-words / TF-IDF model separate them? This is the bar that
                  actually matters and NO lesson had it. A "trajectory" detector that
                  cannot beat unigrams is not reading trajectories.
4. **shuffle** -- with labels permuted, does the pipeline still score above chance? A
                  shuffle AUC far from 0.5 means leakage, not signal.

CPU-only, dependency-free (pure Python + optional numpy). Loads NO model. ASCII stdout
only (Windows cp1252 console -- never print alpha/Delta/norm glyphs).
"""
from __future__ import annotations

import math
import random
import re
from collections import Counter

__all__ = [
    "auc_raw",
    "directionless",
    "length_bar",
    "count_bar",
    "content_bar",
    "shuffle_control",
    "confound_report",
    "margin_over_bar",
    "format_report",
]


# --- the primitive -----------------------------------------------------------
def auc_raw(scores, labels) -> float:
    """Mann-Whitney U / rank-based ROC-AUC. Ties get average ranks.

    RAW and DIRECTIONAL: 0.9 means the score predicts the positive class, 0.1 means it
    predicts the negative class just as strongly. Callers almost always want
    :func:`directionless` instead -- this is exposed so a report can show both.
    """
    scores = [float(s) for s in scores]
    labels = [int(y) for y in labels]
    if len(scores) != len(labels):
        raise ValueError("scores and labels differ in length: %d vs %d" % (len(scores), len(labels)))
    n_pos = sum(labels)
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.5  # degenerate: one class absent, no discrimination is measurable

    order = sorted(range(len(scores)), key=lambda i: scores[i])
    ranks = [0.0] * len(scores)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and scores[order[j + 1]] == scores[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0  # 1-indexed average rank across the tie block
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1

    sum_pos = sum(ranks[i] for i in range(len(labels)) if labels[i] == 1)
    u = sum_pos - n_pos * (n_pos + 1) / 2.0
    return float(u / (n_pos * n_neg))


def directionless(auc: float) -> float:
    """Fold an AUC about 0.5. THE function whose absence caused the 0.110-vs-0.890 bug.

    A confound that predicts the negative class perfectly (auc -> 0.0) is exactly as
    damning as one that predicts the positive class perfectly (auc -> 1.0), because a
    classifier can learn either sign for free.
    """
    return float(max(auc, 1.0 - auc))


# --- the four bars -----------------------------------------------------------
def length_bar(texts, labels) -> dict:
    """Can raw CHARACTER COUNT separate the classes?

    `texts` may be a list of strings OR a list of lists-of-strings (a conversation, a set
    of trajectories); in the latter case the total character count is used, matching what
    the multi-unit lessons measure.
    """
    lens = [_total_chars(t) for t in texts]
    raw = auc_raw(lens, labels)
    pos = [n for n, y in zip(lens, labels) if int(y) == 1]
    neg = [n for n, y in zip(lens, labels) if int(y) == 0]
    return {
        "auc_raw": raw,
        "auc": directionless(raw),
        "mean_pos": float(sum(pos) / len(pos)) if pos else 0.0,
        "mean_neg": float(sum(neg) / len(neg)) if neg else 0.0,
    }


def count_bar(units, labels) -> dict:
    """Can the NUMBER OF UNITS separate the classes? (turns, trajectories, tokens)

    `units` is a list of sequences; its per-item length is the feature. For a lesson that
    fixes K by construction this should come back at exactly 0.5 -- which is a PASS worth
    recording, not a reason to skip the measurement.
    """
    counts = [len(u) if hasattr(u, "__len__") and not isinstance(u, str) else 1 for u in units]
    raw = auc_raw(counts, labels)
    pos = [n for n, y in zip(counts, labels) if int(y) == 1]
    neg = [n for n, y in zip(counts, labels) if int(y) == 0]
    return {
        "auc_raw": raw,
        "auc": directionless(raw),
        "mean_pos": float(sum(pos) / len(pos)) if pos else 0.0,
        "mean_neg": float(sum(neg) / len(neg)) if neg else 0.0,
    }


def content_bar(texts, labels, seed: int = 0, n_folds: int = 5, max_features: int = 20000) -> dict:
    """Can a BAG-OF-WORDS model separate the classes? THE bar no lesson had.

    A centroid-cosine unigram classifier under group-free K-fold CV. Deliberately the
    dumbest reasonable content model: if a "trajectory" / "bi-encoder" / "aggregation"
    detector cannot beat *unigrams*, it is not reading what the lesson says it reads.

    Fit strictly inside each training fold (vocabulary, IDF and centroids all train-only),
    so this bar cannot itself leak.
    """
    labels = [int(y) for y in labels]
    docs = [_tokenize(_flatten(t)) for t in texts]
    n = len(docs)
    if n == 0 or len(set(labels)) < 2:
        return {"auc_raw": 0.5, "auc": 0.5, "n_folds": 0, "note": "degenerate: single class or empty"}

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
            v = _tfidf(docs[i], vocab, idf)
            tgt, _ = (cpos, 1) if labels[i] == 1 else (cneg, 0)
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
        _l2(cpos)
        _l2(cneg)

        for i in test:
            v = _tfidf(docs[i], vocab, idf)
            _l2(v)
            scores[i] = _dot(v, cpos) - _dot(v, cneg)

    raw = auc_raw(scores, labels)
    return {"auc_raw": raw, "auc": directionless(raw), "n_folds": used}


def shuffle_control(texts, labels, seed: int = 0, n_folds: int = 5) -> dict:
    """Re-run the content bar with PERMUTED labels. Must land at ~0.5.

    A shuffle AUC far from 0.5 means the pipeline is leaking (duplicate texts straddling
    folds, a group that spans the split), not that the data has signal.
    """
    rng = random.Random(seed + 9973)
    permuted = list(int(y) for y in labels)
    rng.shuffle(permuted)
    out = content_bar(texts, permuted, seed=seed, n_folds=n_folds)
    out["note"] = "labels permuted; expect ~0.5. Far from 0.5 => leakage, not signal."
    return out


# --- the report --------------------------------------------------------------
def confound_report(texts, labels, units=None, seed: int = 0, n_folds: int = 5,
                    run_content: bool = True, run_shuffle: bool = True) -> dict:
    """Run every bar and return the WORST (largest) as the binding bar.

    Returns a dict with ``length``/``count``/``content``/``shuffle`` sub-reports plus:
      ``worst_auc``  -- the binding directionless bar a method must beat
      ``worst_name`` -- which bar binds
    """
    labels = [int(y) for y in labels]
    rep = {"n": len(labels), "n_pos": sum(labels), "n_neg": len(labels) - sum(labels), "seed": seed}
    rep["length"] = length_bar(texts, labels)
    if units is not None:
        rep["count"] = count_bar(units, labels)
    if run_content:
        rep["content"] = content_bar(texts, labels, seed=seed, n_folds=n_folds)
    if run_shuffle:
        rep["shuffle"] = shuffle_control(texts, labels, seed=seed, n_folds=n_folds)

    # the shuffle control is a LEAKAGE DIAGNOSTIC, never a bar to clear
    bars = {k: v["auc"] for k, v in rep.items()
            if isinstance(v, dict) and "auc" in v and k != "shuffle"}
    if bars:
        rep["worst_name"] = max(bars, key=lambda k: bars[k])
        rep["worst_auc"] = float(bars[rep["worst_name"]])
    else:
        rep["worst_name"], rep["worst_auc"] = "none", 0.5
    return rep


def margin_over_bar(method_auc: float, report: dict, baseline_auc: float = None) -> dict:
    """The only number a lesson may headline: margin over max{confound bar, baseline}.

    CLAUDE.md section 17 rule 7 -- claim only the margin ABOVE the larger of
    {baseline, confound}.
    """
    bar = float(report.get("worst_auc", 0.5))
    bar_name = report.get("worst_name", "none")
    if baseline_auc is not None and float(baseline_auc) > bar:
        bar, bar_name = float(baseline_auc), "method_baseline"
    return {
        "method_auc": float(method_auc),
        "binding_bar": bar,
        "binding_bar_name": bar_name,
        "margin": float(method_auc) - bar,
        "clears": bool(float(method_auc) > bar),
    }


def format_report(report: dict) -> str:
    """ASCII-only summary. NEVER prints unicode -- the Windows cp1252 console dies on it."""
    lines = ["CONFOUND REPORT  n=%d (pos=%d neg=%d) seed=%d"
             % (report.get("n", 0), report.get("n_pos", 0), report.get("n_neg", 0), report.get("seed", 0)),
             "  %-9s %-9s %-9s %s" % ("bar", "raw", "directionless", "detail")]
    for name in ("length", "count", "content", "shuffle"):
        b = report.get(name)
        if not b:
            continue
        detail = ""
        if "mean_pos" in b:
            detail = "mean pos=%.1f neg=%.1f" % (b["mean_pos"], b["mean_neg"])
        elif "n_folds" in b:
            detail = "%d folds" % b["n_folds"]
        flag = "  <-- LEAKAGE?" if (name == "shuffle" and b["auc"] > 0.60) else ""
        lines.append("  %-9s %-9.4f %-9.4f %s%s" % (name, b["auc_raw"], b["auc"], detail, flag))
    lines.append("  BINDING BAR: %s = %.4f  (a method must beat THIS, not 0.5)"
                 % (report.get("worst_name", "none"), report.get("worst_auc", 0.5)))
    return "\n".join(lines)


# --- helpers -----------------------------------------------------------------
_TOKEN = re.compile(r"[a-z0-9']+")


def _flatten(t) -> str:
    if isinstance(t, str):
        return t
    if hasattr(t, "__iter__"):
        return " ".join(_flatten(x) for x in t)
    return str(t)


def _total_chars(t) -> int:
    return len(_flatten(t))


def _tokenize(text: str):
    return _TOKEN.findall(str(text).lower())


def _tfidf(tokens, vocab, idf) -> dict:
    tf = Counter(tok for tok in tokens if tok in vocab)
    return {vocab[t]: (1.0 + math.log(c)) * idf[t] for t, c in tf.items()}


def _l2(v: dict) -> None:
    nrm = math.sqrt(sum(x * x for x in v.values()))
    if nrm > 0:
        for k in v:
            v[k] /= nrm


def _dot(a: dict, b: dict) -> float:
    if len(a) > len(b):
        a, b = b, a
    return sum(val * b.get(k, 0.0) for k, val in a.items())


# --- CPU smoke: python -m steering_tutorials.common.confound -----------------
if __name__ == "__main__":
    rng = random.Random(0)
    # Class 1 is deliberately LONGER and lexically distinct -> every bar should fire.
    texts = (["how do i build a device that causes harm " * rng.randint(3, 6) for _ in range(60)]
             + ["what is a good recipe for soup " * rng.randint(1, 2) for _ in range(60)])
    labels = [1] * 60 + [0] * 60
    rep = confound_report(texts, labels, units=[t.split() for t in texts], seed=0)
    print(format_report(rep))
    print()
    m = margin_over_bar(0.95, rep)
    print("method 0.95 -> margin %+.4f over %s=%.4f  clears=%s"
          % (m["margin"], m["binding_bar_name"], m["binding_bar"], m["clears"]))

    # The fold is the whole point: a raw 0.11 IS a 0.89 confound.
    inv = auc_raw([1, 2, 3, 4], [0, 0, 1, 1])
    print("\nfold check: raw=%.4f directionless=%.4f (a 0.11 raw is a 0.89 bar)"
          % (1 - inv, directionless(1 - inv)))
