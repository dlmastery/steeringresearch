"""probes.py -- linear probes on trajectory activations, and the three controls
that decide whether any number they produce means anything.

THE HEADLINE IS NEVER THE HEADLINE
----------------------------------
A probe AUC on agent-trajectory activations is easy to get and hard to earn. The
same 0.9 can come from four different places, only one of which is the claim:

  1. the model's internal state really carries the failure signal   <- the claim
  2. the probe is reading the STEP INDEX. Failing trajectories run longer, so a
     position detector scores well while knowing nothing.           -> CONTROL 1
  3. the signal is in the surface text, and unigrams have it too.   -> CONTROL 2
  4. the probe has enough capacity to fit anything at this n.       -> CONTROL 3

So every `ProbeResult` this module emits carries all three controls beside the
AUC, and `notes` records which unit each control was computed on. A cell with
`auc_residualised=None` is not a cell with a good raw AUC; it is an unreadable
cell.

CONTROL 1 -- STEP-INDEX RESIDUALISATION (the centrepiece)
---------------------------------------------------------
Regress every feature on the step index and probe the RESIDUALS. What survives
is the part of the representation that is NOT a linear function of position.
Paper B reports 0.989 -> >= 0.939 under this control; a probe that collapses to
chance under it has measured position.

Two properties this implementation insists on:
  * the residualiser is fit on the TRAINING fold only and applied to both, so it
    cannot itself become a leak;
  * it is LINEAR, and honestly so. `_self_test` plants a linear position code and
    shows the control annihilates it (0.8385 -> 0.4948), then plants a QUADRATIC
    one and shows it does NOT (0.8416 -> 0.6232 at degree 1, and only 0.5191
    once the basis is raised to degree 2). That middle number is printed, not
    hidden: a linear control removes linear position information and nothing
    else, and a lesson that reports `auc_residualised` as though it had removed
    "position" in general is overclaiming.

A note on why the planted raw AUC is 0.84 and not 0.99: every trajectory starts
at k=0, so its early rows are genuinely ambiguous no matter how strong the plant.
That ceiling is why `StepIndexProbe` exists -- the number an activation probe has
to be read against is the turn counter's AUC on the SAME corpus, never 0.5.

CONTROL 2 -- THE CONTENT BAR
----------------------------
`common.confound.content_bar` on the SAME text the model read, compared at the
SAME unit. The probe scores rows; the bar scores trajectories; comparing the two
directly would be a category error, so the row scores are pooled to one score
per trajectory first and `clears_content_bar` is computed on that. Both AUCs go
into `notes`.

CONTROL 3 -- THE RANDOM-LABEL CEILING
-------------------------------------
Hewitt and Liang, EMNLP 2019, "Designing and Interpreting Probes with Control
Tasks" (arXiv:1909.03368). Refit on labels permuted BETWEEN trajectories (never
within: within-trajectory permutation would leave the group structure the probe
actually exploits intact and understate the ceiling). What comes back is what
this probe's capacity can reach on this data with no signal at all.

GROUP-AWARE CV, NON-NEGOTIABLE
------------------------------
Every row of one trajectory shares a `group_id`. Splitting by row would put step
3 of a trajectory in train and step 4 in test, which is not held-out anything.
Folds are built on groups and every fold ASSERTS the disjointness rather than
trusting the splitter.

CPU-only. Needs numpy + sklearn. Loads NO model. ASCII stdout (Windows cp1252).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

try:  # package form
    from .types import ActivationBundle, ProbeResult
    from ..common.confound import content_bar, directionless
except (ImportError, ValueError):  # pragma: no cover - direct-script form
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
    from steering_tutorials.traj_probes.types import ActivationBundle, ProbeResult
    from steering_tutorials.common.confound import content_bar, directionless

__all__ = [
    "EVAL_N_BOOT", "auc", "bootstrap_auc_ci", "group_folds",
    "StepResidualiser", "residualise_step_index",
    "LinearTrajProbe", "StepIndexProbe",
    "trajectory_scores", "content_bar_control", "random_label_control",
]

# CLAUDE.md section 7 puts the rigor floor at >= 10k resamples for the 95% CI on
# a delta. It is the DEFAULT here rather than a knob someone remembers to raise,
# because a check that must be remembered will eventually be skipped
# (CLAUDE.md 18.8).
EVAL_N_BOOT = 10000


def _print(*args) -> None:
    """ASCII-only stdout. Never print alpha/Delta/norm glyphs on this host."""
    msg = " ".join(str(a) for a in args)
    try:
        print(msg)
    except Exception:  # pragma: no cover
        print(msg.encode("ascii", "replace").decode("ascii"))


# ---------------------------------------------------------------------------
# 1. AUC and a clustered bootstrap CI
# ---------------------------------------------------------------------------
def auc(scores, y) -> float:
    """Rank-based ROC-AUC with average ranks for ties. RAW and DIRECTIONAL.

    A fitted probe is free to learn either sign, so its out-of-fold AUC is read
    as-is. Only a CONFOUND bar is folded about 0.5 (`confound.directionless`) --
    that asymmetry is deliberate and is the reason both live here side by side.
    """
    s = np.asarray(scores, dtype=np.float64)
    yy = np.asarray(y).astype(np.int64)
    n_pos = int(yy.sum())
    n_neg = int(len(yy) - n_pos)
    if n_pos == 0 or n_neg == 0:
        return 0.5
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s), dtype=np.float64)
    sorted_s = s[order]
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and sorted_s[j + 1] == sorted_s[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return float((ranks[yy == 1].sum() - n_pos * (n_pos + 1) / 2.0)
                 / (n_pos * n_neg))


def bootstrap_auc_ci(scores, y, groups, n_boot: int = EVAL_N_BOOT,
                     seed: int = 0, alpha: float = 0.05) -> tuple:
    """95% percentile CI on the AUC, resampling GROUPS not rows.

    Rows inside one trajectory are not independent draws -- step 4 is very nearly
    step 3 -- and the label attaches to the trajectory, not to the row. So the
    resampling unit must be the trajectory: a row bootstrap resamples something
    that is not the data-generating process.

    A caveat worth stating, because the obvious version of this claim is wrong.
    "Clustered bootstrap => wider CI" is the textbook intuition and it is NOT a
    theorem. It holds when rows within a group are near-duplicates (the
    self-test constructs that case and measures a ~2x widening). On real data
    where the within-trajectory variation is LARGE -- e.g. a score driven by
    step index, which sweeps its whole range inside every single trajectory --
    the row bootstrap can come out WIDER, because it scrambles a composition
    that group resampling holds fixed. The self-test measures that case too and
    reports it. Group resampling is used here because it is the right estimator,
    not because it reliably produces a bigger number.
    """
    s = np.asarray(scores, dtype=np.float64)
    yy = np.asarray(y).astype(np.int64)
    g = np.asarray([str(x) for x in groups])
    uniq = np.unique(g)
    idx_by_group = {u: np.flatnonzero(g == u) for u in uniq}
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(int(n_boot)):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([idx_by_group[u] for u in pick])
        if len(np.unique(yy[idx])) < 2:
            continue
        out.append(auc(s[idx], yy[idx]))
    if not out:
        return (float("nan"), float("nan"))
    out = np.asarray(out)
    return (float(np.quantile(out, alpha / 2.0)),
            float(np.quantile(out, 1.0 - alpha / 2.0)))


# ---------------------------------------------------------------------------
# 2. Group-aware folds, with the disjointness ASSERTED
# ---------------------------------------------------------------------------
def group_folds(groups, y, n_folds: int = 5, seed: int = 0) -> list:
    """-> list of (train_idx, test_idx), never splitting a group.

    Prefers `StratifiedGroupKFold` so class balance survives the split; falls
    back to `GroupKFold`. Either way the disjointness is CHECKED here rather than
    trusted, because a splitter silently doing the wrong thing is exactly the
    class of defect this repo keeps finding.
    """
    g = np.asarray([str(x) for x in groups])
    yy = np.asarray(y).astype(np.int64)
    n_groups = len(np.unique(g))
    k = int(min(n_folds, n_groups))
    if k < 2:
        raise ValueError("group-aware CV needs >= 2 groups, got %d. Rows from a "
                         "single trajectory cannot form a held-out fold."
                         % n_groups)
    try:
        from sklearn.model_selection import StratifiedGroupKFold
        splitter = StratifiedGroupKFold(n_splits=k, shuffle=True,
                                        random_state=seed)
        splits = list(splitter.split(np.zeros(len(g)), yy, groups=g))
    except Exception:  # pragma: no cover - old sklearn, or degenerate strata
        from sklearn.model_selection import GroupKFold
        splits = list(GroupKFold(n_splits=k).split(np.zeros(len(g)), yy, groups=g))

    for f, (tr, te) in enumerate(splits):
        overlap = set(g[tr]) & set(g[te])
        if overlap:
            raise AssertionError(
                "fold %d leaks %d group(s) across the split (e.g. %r). Rows from "
                "one trajectory straddling train and test is not held-out "
                "evaluation." % (f, len(overlap), sorted(overlap)[0]))
    return splits


# ---------------------------------------------------------------------------
# 3. CONTROL 1 -- step-index residualisation
# ---------------------------------------------------------------------------
class StepResidualiser:
    """Remove the linear step-index component of every feature.

    Fits, per feature j, the least-squares model  X[:, j] ~ B(k)  where B(k) is
    the polynomial basis [1, k, k^2, ...] of `degree` (default 1 = linear, which
    is what the series specifies), then returns  X - B(k) @ coef.

    `fit` and `transform` are separate so CV can fit on the TRAINING fold and
    apply to both. Residualising on the pooled data before splitting would use
    test-fold features to define the transform -- not a label leak, but not a
    clean held-out number either, and the distinction is cheap to keep.
    """

    def __init__(self, degree: int = 1):
        if int(degree) < 1:
            raise ValueError("degree must be >= 1; degree 0 removes only the "
                             "mean, which is not a position control")
        self.degree = int(degree)
        self.coef_ = None
        self.n_features_ = None

    @staticmethod
    def _basis(k, degree: int) -> np.ndarray:
        k = np.asarray(k, dtype=np.float64).reshape(-1)
        cols = [np.ones_like(k)] + [k ** d for d in range(1, degree + 1)]
        return np.stack(cols, axis=1)

    def fit(self, X, step_index) -> "StepResidualiser":
        X = np.asarray(X, dtype=np.float64)
        B = self._basis(step_index, self.degree)
        if len(np.unique(np.asarray(step_index))) < 2:
            raise ValueError(
                "step_index is constant over the training fold, so there is no "
                "position information to residualise and CONTROL 1 would be "
                "vacuous. Check that the bundle carries per-turn rows.")
        self.coef_, *_ = np.linalg.lstsq(B, X, rcond=None)
        self.n_features_ = X.shape[1]
        return self

    def transform(self, X, step_index) -> np.ndarray:
        if self.coef_ is None:
            raise RuntimeError("StepResidualiser.transform before fit")
        X = np.asarray(X, dtype=np.float64)
        if X.shape[1] != self.n_features_:
            raise ValueError("residualiser fit on %d features, got %d"
                             % (self.n_features_, X.shape[1]))
        return X - self._basis(step_index, self.degree) @ self.coef_

    def fit_transform(self, X, step_index) -> np.ndarray:
        return self.fit(X, step_index).transform(X, step_index)


def residualise_step_index(X, step_index, degree: int = 1) -> np.ndarray:
    """Convenience one-shot form of :class:`StepResidualiser`.

    Fits and applies on the SAME data, so use it for inspection and for the
    planted-signal test -- inside CV, use the class and fit on train only.
    """
    return StepResidualiser(degree=degree).fit_transform(X, step_index)


# ---------------------------------------------------------------------------
# 4. The probes
# ---------------------------------------------------------------------------
def _fit_fold(Xtr, ytr, Xte, C: float, seed: int) -> np.ndarray:
    """Standardise on train only, fit L2 logistic regression, score test."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    sc = StandardScaler().fit(Xtr)
    clf = LogisticRegression(C=C, max_iter=2000, solver="lbfgs",
                             random_state=seed)
    clf.fit(sc.transform(Xtr), ytr)
    return clf.decision_function(sc.transform(Xte))


def _oof(bundle, splits, residualise: bool, degree: int, C: float,
         seed: int, y_override=None) -> tuple:
    """-> (out-of-fold scores, n_folds_used, n_folds_skipped)."""
    X = np.asarray(bundle.X, dtype=np.float64)
    y = (np.asarray(bundle.y) if y_override is None
         else np.asarray(y_override)).astype(np.int64)
    k = np.asarray(bundle.step_index)
    scores = np.full(len(X), np.nan, dtype=np.float64)
    used = skipped = 0
    for tr, te in splits:
        if len(np.unique(y[tr])) < 2 or len(np.unique(y[te])) < 2:
            skipped += 1
            continue
        Xtr, Xte = X[tr], X[te]
        if residualise:
            r = StepResidualiser(degree=degree).fit(Xtr, k[tr])
            Xtr, Xte = r.transform(Xtr, k[tr]), r.transform(Xte, k[te])
        scores[te] = _fit_fold(Xtr, y[tr], Xte, C=C, seed=seed)
        used += 1
    return scores, used, skipped


def trajectory_scores(traj_uid, row_scores, y, how: str = "max") -> tuple:
    """Pool row scores to ONE score per trajectory. -> (uids, scores, labels).

    Needed because the content bar scores trajectories while the probe scores
    rows: comparing a row-level AUC against a trajectory-level bar would compare
    two different n on two different units. `max` is the default because it is
    the decision rule reproduction A's abort cascade actually uses -- fire if any
    step trips the gate.
    """
    uid = np.asarray([str(u) for u in traj_uid])
    s = np.asarray(row_scores, dtype=np.float64)
    yy = np.asarray(y).astype(np.int64)
    order, out_s, out_y = [], [], []
    for u in dict.fromkeys(uid.tolist()):       # first-seen order, stable
        m = (uid == u) & np.isfinite(s)
        if not m.any():
            continue
        v = s[m]
        val = (v.max() if how == "max" else
               v.mean() if how == "mean" else
               v[-1] if how == "last" else None)
        if val is None:
            raise ValueError("unknown pooling %r (max|mean|last)" % how)
        order.append(u)
        out_s.append(float(val))
        out_y.append(int(yy[uid == u][0]))
    return np.asarray(order), np.asarray(out_s), np.asarray(out_y)


class LinearTrajProbe:
    """L2 logistic regression on residual-stream rows. Implements `TrajProbe`.

    Deliberately low capacity. A probe rich enough to compute the task itself
    stops being evidence about the representation, which is the point of
    CONTROL 3 -- and of keeping this linear.
    """

    name = "linear_l2"

    def __init__(self, C: float = 1.0, residual_degree: int = 1,
                 traj_pool: str = "max"):
        self.C = float(C)
        self.residual_degree = int(residual_degree)
        self.traj_pool = str(traj_pool)

    def fit_predict_cv(self, bundle, n_folds: int = 5, seed: int = 0,
                       texts=None, n_boot: int = EVAL_N_BOOT,
                       run_controls: bool = True) -> ProbeResult:
        """-> a `ProbeResult` with every control populated (or an honest None).

        `texts` is an optional {traj_uid: text} mapping. Without it CONTROL 2
        cannot be run and `content_bar_auc` stays None -- which reads as "not
        measured", never as "cleared".
        """
        y = np.asarray(bundle.y).astype(np.int64)
        splits = group_folds(bundle.group_id, y, n_folds=n_folds, seed=seed)

        raw, used, skipped = _oof(bundle, splits, False, self.residual_degree,
                                  self.C, seed)
        ok = np.isfinite(raw)
        a_raw = auc(raw[ok], y[ok])
        ci = bootstrap_auc_ci(raw[ok], y[ok], np.asarray(bundle.group_id)[ok],
                              n_boot=n_boot, seed=seed)

        res_auc = None
        try:
            res, _u, _s = _oof(bundle, splits, True, self.residual_degree,
                               self.C, seed)
            okr = np.isfinite(res)
            res_auc = auc(res[okr], y[okr])
        except ValueError as exc:
            notes_res = "CONTROL 1 unavailable: %s" % exc
        else:
            notes_res = ("CONTROL 1: degree-%d step-index residualisation, fit "
                         "on the training fold only. It removes LINEAR position "
                         "information and nothing else."
                         % self.residual_degree)

        bar_auc = clears = None
        traj_auc = None
        if run_controls:
            uids, ts, ty = trajectory_scores(bundle.traj_uid, raw, y,
                                             how=self.traj_pool)
            traj_auc = auc(ts, ty)
            if texts:
                bar = content_bar([texts.get(str(u), "") for u in uids], ty,
                                  seed=seed, return_scores=True)
                bar_auc = float(bar["auc"])
                clears = bool(traj_auc > bar_auc)

        rand_auc = None
        if run_controls:
            rand_auc = random_label_control(bundle, splits, C=self.C, seed=seed,
                                            degree=self.residual_degree)

        notes = "; ".join([
            notes_res,
            "CONTROL 2 unit: trajectory-level probe AUC %s vs the content bar, "
            "pooled by '%s'; the row-level headline AUC is NOT comparable to a "
            "per-trajectory bar"
            % ("n/a" if traj_auc is None else "%.4f" % traj_auc, self.traj_pool),
            "CONTROL 3: labels permuted BETWEEN trajectories (Hewitt and Liang, "
            "EMNLP 2019, arXiv:1909.03368)",
            "CV: %d group folds used, %d skipped for a single-class fold; "
            "bootstrap resamples GROUPS, n_boot=%d" % (used, skipped, n_boot),
        ])
        return ProbeResult(
            method=self.name, layer=int(bundle.layer), auc=float(a_raw),
            auc_ci=ci, auc_residualised=res_auc, content_bar_auc=bar_auc,
            clears_content_bar=clears, random_label_auc=rand_auc,
            n_items=int(ok.sum()),
            n_trajectories=int(len(set(str(u) for u in bundle.traj_uid))),
            notes=notes)


class StepIndexProbe:
    """The position-only baseline: one feature, the step index. No activations.

    This is the number CONTROL 1 exists to guard against, made explicit. If it
    lands near the activation probe's AUC, the activation probe has not
    demonstrated that the residual stream carries anything the turn counter does
    not. Costs nothing to run and is the cheapest honest thing in the module.
    """

    name = "step_index_only"

    def __init__(self, C: float = 1.0):
        self.C = float(C)

    def fit_predict_cv(self, bundle, n_folds: int = 5, seed: int = 0,
                       n_boot: int = EVAL_N_BOOT, **_kw) -> ProbeResult:
        y = np.asarray(bundle.y).astype(np.int64)
        k = np.asarray(bundle.step_index, dtype=np.float64).reshape(-1, 1)
        splits = group_folds(bundle.group_id, y, n_folds=n_folds, seed=seed)
        scores = np.full(len(k), np.nan)
        for tr, te in splits:
            if len(np.unique(y[tr])) < 2 or len(np.unique(y[te])) < 2:
                continue
            scores[te] = _fit_fold(k[tr], y[tr], k[te], C=self.C, seed=seed)
        ok = np.isfinite(scores)
        a = auc(scores[ok], y[ok])
        ci = bootstrap_auc_ci(scores[ok], y[ok],
                              np.asarray(bundle.group_id)[ok],
                              n_boot=n_boot, seed=seed)
        return ProbeResult(
            method=self.name, layer=int(bundle.layer), auc=float(a), auc_ci=ci,
            auc_residualised=0.5,
            n_items=int(ok.sum()),
            n_trajectories=int(len(set(str(u) for u in bundle.traj_uid))),
            notes="position-only baseline: the single feature IS the step index, "
                  "so auc_residualised is 0.5 by construction. Any activation "
                  "probe must be read against THIS, not against 0.5.")


# ---------------------------------------------------------------------------
# 5. CONTROL 2 and CONTROL 3 as standalone, reusable functions
# ---------------------------------------------------------------------------
def content_bar_control(uids, labels, texts, seed: int = 0) -> dict:
    """CONTROL 2: the bag-of-words bar on the SAME text, at trajectory level.

    Thin wrapper over `common.confound.content_bar` that keeps the directionless
    fold (a bar predicting the NEGATIVE class perfectly is exactly as damning)
    and returns the per-item out-of-fold scores, so a PAIRED comparison against
    the probe is possible rather than two unpaired AUCs.
    """
    docs = [texts.get(str(u), "") for u in uids]
    rep = content_bar(docs, list(labels), seed=seed, return_scores=True)
    rep["auc"] = directionless(rep["auc_raw"])
    return rep


def random_label_control(bundle, splits=None, C: float = 1.0, seed: int = 0,
                         degree: int = 1, n_folds: int = 5) -> float:
    """CONTROL 3: refit on labels permuted BETWEEN trajectories. -> AUC.

    Permuting at the TRAJECTORY level, not the row level, is the load-bearing
    detail. A row-level permutation would leave each trajectory with a mixture of
    labels, destroying the very group structure the real probe exploits, and
    would report a ceiling far lower than the probe's actual capacity -- a
    control that flatters the thing it is meant to bound.
    """
    y = np.asarray(bundle.y).astype(np.int64)
    uid = np.asarray([str(u) for u in bundle.traj_uid])
    order = list(dict.fromkeys(uid.tolist()))
    lab = {u: int(y[uid == u][0]) for u in order}
    rng = np.random.default_rng(seed + 7919)
    shuffled = list(lab.values())
    rng.shuffle(shuffled)
    perm = {u: shuffled[i] for i, u in enumerate(order)}
    y_perm = np.asarray([perm[u] for u in uid], dtype=np.int64)

    if splits is None:
        splits = group_folds(bundle.group_id, y_perm, n_folds=n_folds, seed=seed)
    scores, _u, _s = _oof(bundle, splits, False, degree, C, seed,
                          y_override=y_perm)
    ok = np.isfinite(scores)
    if not ok.any() or len(np.unique(y_perm[ok])) < 2:
        return float("nan")
    return float(auc(scores[ok], y_perm[ok]))


# ---------------------------------------------------------------------------
# CPU self-test -- NO model, NO GPU, synthetic arrays only.
#   python -m steering_tutorials.traj_probes.probes
# ---------------------------------------------------------------------------
def _make_bundle(n_traj: int, hidden: int, seed: int, plant: str,
                 signal: float = 1.0):
    """Synthesise a bundle whose signal is a KNOWN function of step or content.

    Failing trajectories are drawn LONGER than passing ones -- the real-world
    correlation that makes CONTROL 1 necessary in the first place.
    """
    rng = np.random.default_rng(seed)
    A = rng.normal(size=(hidden,))
    Q = rng.normal(size=(hidden,))
    C = rng.normal(size=(hidden,))
    X, y, step, uid, grp = [], [], [], [], []
    for i in range(n_traj):
        label = i % 2
        n = int(rng.integers(12, 20) if label else rng.integers(3, 7))
        for k in range(n):
            h = rng.normal(scale=0.5, size=hidden)
            if plant in ("linear", "both"):
                h = h + signal * A * k
            if plant == "quadratic":
                h = h + signal * Q * (k ** 2) / 10.0
            if plant in ("content", "both"):
                h = h + signal * C * (2.0 * label - 1.0)
            X.append(h)
            y.append(label)
            step.append(k)
            uid.append("t%03d" % i)
            grp.append("g%03d" % i)
    return ActivationBundle(
        X=np.asarray(X, dtype=np.float32), y=np.asarray(y),
        step_index=np.asarray(step), traj_uid=np.asarray(uid),
        group_id=np.asarray(grp), layer=12, model_id="synthetic",
        behaviour_fingerprint="selftest")


def _self_test() -> None:  # noqa: C901 - a test, read top to bottom
    n_boot = 400          # the self-test is a smoke; EVAL_N_BOOT is the default

    # (0) AUC primitive agrees with sklearn, ties included.
    from sklearn.metrics import roc_auc_score
    rng = np.random.default_rng(0)
    s = np.round(rng.normal(size=400), 1)          # rounding forces ties
    yv = (rng.random(400) < 0.4).astype(int)
    assert abs(auc(s, yv) - roc_auc_score(yv, s)) < 1e-12
    _print("OK  auc(): matches sklearn to 1e-12 with ties present")

    # (1) group-aware CV never splits a trajectory, and the assert is real.
    b = _make_bundle(40, 16, seed=1, plant="linear")
    splits = group_folds(b.group_id, b.y, n_folds=5, seed=0)
    g = np.asarray([str(x) for x in b.group_id])
    assert all(not (set(g[tr]) & set(g[te])) for tr, te in splits)
    row_splits = [(np.arange(len(g))[::2], np.arange(len(g))[1::2])]
    leaked = set(g[row_splits[0][0]]) & set(g[row_splits[0][1]])
    assert leaked, "the row-wise split failed to demonstrate the leak"
    _print("OK  group CV: %d folds, 0 groups straddling; the same data split by "
           "ROW would leak %d groups" % (len(splits), len(leaked)))

    # (2) THE PLANTED-SIGNAL TEST -- the proof CONTROL 1 works.
    #     Features are a pure LINEAR function of step index plus noise, and the
    #     label is carried entirely by trajectory LENGTH. A probe reading
    #     position must therefore score high and must collapse to chance once
    #     linear position information is residualised out.
    probe = LinearTrajProbe(C=1.0)
    r_lin = probe.fit_predict_cv(b, n_folds=5, seed=0, n_boot=n_boot)
    sp_lin = StepIndexProbe().fit_predict_cv(b, n_folds=5, seed=0, n_boot=n_boot)

    # The raw AUC is capped well below 1.0 no matter how strong the plant,
    # because every trajectory starts at k=0 and its early rows are genuinely
    # ambiguous. So the test is NOT "is the raw AUC big" -- an arbitrary
    # threshold there would be numerology. It is the two statements that
    # actually constitute the claim:
    #   (a) a 1152-dim-style activation probe reaches EXACTLY what a one-feature
    #       turn counter reaches, because position is all there is to read;
    #   (b) residualising position away collapses it to chance.
    assert r_lin.auc > 0.70, ("the planted position signal was not detectable "
                              "(%.4f)" % r_lin.auc)
    assert abs(r_lin.auc - sp_lin.auc) < 0.05, (
        "the activation probe (%.4f) and the position-only probe (%.4f) should "
        "agree on a bundle whose only signal IS position" % (r_lin.auc, sp_lin.auc))
    assert 0.40 <= r_lin.auc_residualised <= 0.60, (
        "residualisation left %.4f of a PURE step-index signal -- CONTROL 1 is "
        "not removing what it claims to remove" % r_lin.auc_residualised)
    _print("OK  PLANTED step-index signal: raw AUC %.4f -> residualised %.4f "
           "(collapses to chance, as it must), and it matches the one-feature "
           "turn counter at %.4f -- the probe read position and nothing else"
           % (r_lin.auc, r_lin.auc_residualised, sp_lin.auc))

    # (2b) the control must not simply destroy everything. A genuine
    #      content signal, orthogonal to step, has to SURVIVE it.
    b_c = _make_bundle(40, 16, seed=2, plant="content")
    r_c = probe.fit_predict_cv(b_c, n_folds=5, seed=0, n_boot=n_boot)
    assert r_c.auc_residualised > 0.85, (
        "a step-independent signal was destroyed by the position control "
        "(%.4f) -- the control would then 'work' by erasing everything"
        % r_c.auc_residualised)
    _print("OK  step-INDEPENDENT signal survives it: raw %.4f -> residualised "
           "%.4f (the control is selective, not a shredder)"
           % (r_c.auc, r_c.auc_residualised))

    # (2c) THE HONEST LIMITATION, printed rather than buried. A linear control
    #      removes LINEAR position information. A quadratic position code walks
    #      straight through it, and any lesson reading `auc_residualised` as
    #      "position removed" in general is overclaiming.
    b_q = _make_bundle(40, 16, seed=3, plant="quadratic")
    r_q = probe.fit_predict_cv(b_q, n_folds=5, seed=0, n_boot=n_boot,
                               run_controls=False)
    r_q2 = LinearTrajProbe(C=1.0, residual_degree=2).fit_predict_cv(
        b_q, n_folds=5, seed=0, n_boot=n_boot, run_controls=False)
    _print("LIMITATION  a QUADRATIC position code: raw %.4f -> degree-1 "
           "residualised %.4f (survives); degree-2 residualised %.4f. "
           "'auc_residualised' means LINEAR position removed, nothing more."
           % (r_q.auc, r_q.auc_residualised, r_q2.auc_residualised))

    # (3) THE BAR IS NOT 0.5. In both synthetic bundles the failing trajectories
    #     are longer, so the one-feature turn counter scores ~0.85 even on the
    #     bundle whose ACTIVATIONS carry no position information at all. That is
    #     the real-world situation, and it is why a raw AUC near 0.85 here means
    #     nothing on its own. What separates the two bundles is CONTROL 1: the
    #     content-planted probe keeps everything it had once position is
    #     residualised away, the position-planted one keeps none of it.
    sp_c = StepIndexProbe().fit_predict_cv(b_c, n_folds=5, seed=0, n_boot=n_boot)
    assert sp_c.auc > 0.70, ("the synthetic corpus was meant to give failing "
                             "trajectories more turns; the counter only reached "
                             "%.4f" % sp_c.auc)
    assert r_c.auc_residualised > sp_c.auc, (
        "the content-planted probe's residualised AUC (%.4f) does not beat the "
        "turn counter (%.4f) -- with that ordering there would be no evidence "
        "the activations added anything" % (r_c.auc_residualised, sp_c.auc))
    assert r_lin.auc_residualised < sp_lin.auc
    _print("OK  the bar is NOT 0.5: the turn counter alone reaches %.4f on the "
           "position bundle and %.4f on the content bundle. What tells them "
           "apart is CONTROL 1 -- residualised %.4f (position: gone) vs %.4f "
           "(content: intact, and above the counter)"
           % (sp_lin.auc, sp_c.auc, r_lin.auc_residualised,
              r_c.auc_residualised))

    # (4) CONTROL 2 at a matched unit.
    texts = {}
    for u in dict.fromkeys(np.asarray(b_c.traj_uid).tolist()):
        lab = int(np.asarray(b_c.y)[np.asarray(b_c.traj_uid) == u][0])
        texts[u] = ("the agent looped and retried and failed " * 4 if lab
                    else "the agent fetched the record and finished " * 4)
    r_t = probe.fit_predict_cv(b_c, n_folds=5, seed=0, texts=texts,
                               n_boot=n_boot)
    assert r_t.content_bar_auc is not None and r_t.clears_content_bar is not None
    # These texts are trivially separable by unigrams, so the bar saturates and
    # a probe that merely TIES it has demonstrated nothing -- clears must be
    # False. `clears_content_bar` has to be capable of both answers or it is
    # decoration, so the same probe is re-run against uninformative text where
    # the bar collapses and it must flip to True.
    assert r_t.clears_content_bar is False, (
        "the probe tied a saturated unigram bar (%.4f vs %.4f) and was still "
        "recorded as clearing it" % (r_t.auc, r_t.content_bar_auc))
    flat_texts = {u: "the agent ran some steps and then stopped " * 4
                  for u in texts}
    r_f = probe.fit_predict_cv(b_c, n_folds=5, seed=0, texts=flat_texts,
                               n_boot=n_boot)
    assert r_f.clears_content_bar is True, r_f.content_bar_auc
    _print("OK  CONTROL 2 goes BOTH ways: bar %.4f -> clears=%s on separable "
           "text, bar %.4f -> clears=%s on uninformative text (compared at the "
           "TRAJECTORY unit, never against the row-level headline)"
           % (r_t.content_bar_auc, r_t.clears_content_bar,
              r_f.content_bar_auc, r_f.clears_content_bar))
    r_nt = probe.fit_predict_cv(b_c, n_folds=5, seed=0, n_boot=n_boot)
    assert r_nt.content_bar_auc is None and r_nt.clears_content_bar is None
    _print("OK  with no texts the bar reads None (NOT MEASURED), never 'cleared'")

    # (5) CONTROL 3: the random-label ceiling lands near chance on real signal,
    #     and trajectory-level permutation is what makes it honest.
    assert 0.30 <= r_t.random_label_auc <= 0.70, r_t.random_label_auc
    _print("OK  CONTROL 3: random-label ceiling %.4f (probe %.4f) -- capacity "
           "alone does not reach the result" % (r_t.random_label_auc, r_t.auc))

    # (6) THE BOOTSTRAP UNIT. The textbook line is "clustered resampling gives a
    #     wider CI", and it is not a theorem -- so it is tested where it is
    #     actually true and MEASURED where it is not.
    #
    #     (a) the case it is true: rows within a trajectory are exact
    #         duplicates, so there are really n_groups observations and a row
    #         bootstrap treats them as n_rows. Its CI must be too tight.
    rng6 = np.random.default_rng(11)
    n_g, reps = 40, 12
    ty6 = np.repeat([0, 1], n_g // 2)
    ts6 = rng6.normal(loc=ty6 * 0.9, scale=1.0)
    dup_s = np.repeat(ts6, reps)
    dup_y = np.repeat(ty6, reps)
    dup_g = np.repeat(["g%02d" % i for i in range(n_g)], reps)
    lo_g, hi_g = bootstrap_auc_ci(dup_s, dup_y, dup_g, n_boot=n_boot, seed=0)
    lo_r, hi_r = bootstrap_auc_ci(dup_s, dup_y, np.arange(len(dup_s)),
                                  n_boot=n_boot, seed=0)
    assert (hi_g - lo_g) > (hi_r - lo_r), (
        "with rows that are exact within-group duplicates the GROUP CI "
        "[%.4f,%.4f] must be wider than the ROW CI [%.4f,%.4f]; it is not, so "
        "the resampling unit is not being honoured"
        % (lo_g, hi_g, lo_r, hi_r))
    _print("OK  bootstrap unit: with %d groups x %d duplicate rows, group CI "
           "width %.4f vs row CI width %.4f (%.1fx) -- a row bootstrap would "
           "count %d observations where there are %d"
           % (n_g, reps, hi_g - lo_g, hi_r - lo_r,
              (hi_g - lo_g) / max(hi_r - lo_r, 1e-9), n_g * reps, n_g))

    #     (b) the case it is NOT true, measured rather than assumed. On the
    #         position bundle the score sweeps its whole range INSIDE every
    #         trajectory, so row resampling scrambles a composition that group
    #         resampling holds fixed, and the row CI comes out wider. Group
    #         resampling is still the right estimator -- the label attaches to
    #         the trajectory -- but it is not the conservative one here.
    raw, _u, _s = _oof(b, group_folds(b.group_id, b.y, 5, 0), False, 1, 1.0, 0)
    ok = np.isfinite(raw)
    yv2 = np.asarray(b.y)[ok]
    g2 = np.asarray(b.group_id)[ok]
    lo_g2, hi_g2 = bootstrap_auc_ci(raw[ok], yv2, g2, n_boot=n_boot, seed=0)
    lo_r2, hi_r2 = bootstrap_auc_ci(raw[ok], yv2, np.arange(int(ok.sum())),
                                    n_boot=n_boot, seed=0)
    _print("MEASURED    on the position bundle the ordering REVERSES: group CI "
           "[%.4f,%.4f] width %.4f vs row CI [%.4f,%.4f] width %.4f. "
           "'Clustered is wider' is an intuition, not a guarantee; the reason "
           "to resample groups is that the label attaches to the trajectory."
           % (lo_g2, hi_g2, hi_g2 - lo_g2, lo_r2, hi_r2, hi_r2 - lo_r2))

    # (7) a constant step_index makes CONTROL 1 vacuous, and that is reported
    #     as unavailable rather than silently passed.
    flat = ActivationBundle(X=b_c.X, y=b_c.y,
                            step_index=np.zeros(len(b_c.X), dtype=int),
                            traj_uid=b_c.traj_uid, group_id=b_c.group_id,
                            layer=12, model_id="synthetic",
                            behaviour_fingerprint="selftest")
    r_flat = probe.fit_predict_cv(flat, n_folds=5, seed=0, n_boot=n_boot,
                                  run_controls=False)
    assert r_flat.auc_residualised is None and "unavailable" in r_flat.notes
    _print("OK  a constant step_index yields auc_residualised=None plus a note, "
           "not a quietly-passing control")

    _print("")
    _print("OK -- probes.py: group-aware CV asserted, CONTROL 1 annihilates a "
           "planted LINEAR position signal while sparing a step-independent "
           "one, CONTROL 2 is compared at a matched unit and can answer both "
           "ways, CONTROL 3 permutes whole trajectories, and the CI resamples "
           "groups.")


if __name__ == "__main__":
    _self_test()
