"""pooling.py -- the fixed-feature aggregator arm of the offline aggregator ladder.

THE CLAIM UNDER TEST (see ../types.py module docstring for the full framing)
------------------------------------------------------------------------------
Advani, L., 2 Jan 2026, AAAI TrustAgent 2026, "Trajectory Guard -- A Lightweight,
Sequence-Aware Model for Real-Time Anomaly Detection in Agentic AI" (arXiv:2601.00516)
states the hypothesis this whole ladder exists to test, in its own abstract: "mean-
pooling embeddings dilutes anomalous steps, while contrastive-only approaches ignore
sequential structure." This repo's `cross_trajectory` lesson measured its `mean_agg`
as the TOP method (0.9358), above `attn_pool` (0.8635) and `gnn_agg` (0.8121) --
apparently contradicting that claim on our substrate.

**But `cross_trajectory`'s `mean_agg` is not a mean pool.** Per `cross_trajectory/
models.py:198` (`MeanAgg._features`) and `cross_trajectory/README.md` line 140 ("`
mean_agg` is not a mean-pool"), it concatenates `[mean, max, std]` over the K
embeddings -- three times the dimension, with a MAX term already baked in. Part of why
"mean pooling wins" there may just be that a max-pool feature was never absent. This
module keeps `MeanPool` (a TRUE mean, one term) and `MeanMaxStdPool` (the exact
`cross_trajectory` recipe) as two DISTINCT, separately-scored aggregators specifically
so that conflation cannot happen here.

NO THUMB ON THE SCALE
----------------------
Every aggregator below reduces a trajectory to a fixed-size feature vector and is then
scored by the IDENTICAL classifier: a `sklearn.linear_model.LogisticRegression` fit on
a `StandardScaler`-transformed feature, both fit on TRAIN ONLY (see `_PooledAggregator.
fit`). Same solver, same max_iter, same seed policy, same fitting budget, for all six
classes. If mean pooling wins on our corpora, the harness lets it win -- the only
degree of freedom between subclasses is the pooling function `_pool()`.

THE CONTROL (do not remove `LastStep`)
---------------------------------------
Per the lead's brief: without a final-step-only baseline, "trajectory detection" is
indistinguishable from "the final step is just harmful". That is the exact ambiguity
that made `multiturn_jailbreak`'s headline undecidable. `LastStep` exists so every
other aggregator's margin can be measured ABOVE this control, not above chance.

CPU-only to import. Loads NO model, downloads nothing. ASCII stdout only.
"""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

__all__ = [
    "MeanPool",
    "MaxPool",
    "MeanMaxStdPool",
    "DeviationWeighted",
    "LastStep",
    "Truncate",
    "FirstKSteps",
]


def _softmax(x: np.ndarray) -> np.ndarray:
    """Numerically stable softmax over a 1-D array. Falls back to uniform on all-zero
    input (a trajectory whose every step sits exactly on the centroid)."""
    x = np.asarray(x, dtype=np.float64)
    shifted = x - np.max(x)
    e = np.exp(shifted)
    total = e.sum()
    if total <= 0.0 or not np.isfinite(total):
        return np.full_like(x, 1.0 / max(len(x), 1))
    return e / total


class _PooledAggregator:
    """Shared scaffolding: every subclass supplies ONLY `_pool(trajectory) -> vector`.

    `fit` pools every training trajectory into a feature matrix, then fits a
    StandardScaler + LogisticRegression on TRAIN ONLY. `score` pools the query
    trajectory through the SAME fitted scaler+classifier. This is what makes the
    aggregator comparison honest: capacity, regularization (`C=1.0` default) and
    optimizer budget are held fixed across every subclass in this file.
    """

    name = "pooled_aggregator"
    is_causal = True

    def __init__(self, seed: int = 0):
        self.seed = seed
        self.scaler_: StandardScaler | None = None
        self.clf_: LogisticRegression | None = None

    # -- subclasses implement this ONE method --------------------------------
    def _pool(self, trajectory: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    # -- Aggregator protocol --------------------------------------------------
    def fit(self, trajectories, labels) -> None:
        labels = [int(y) for y in labels]
        if len(trajectories) == 0:
            raise ValueError("%s.fit called with zero trajectories" % self.name)
        if len(set(labels)) < 2:
            raise ValueError(
                "%s.fit called with a single class present (labels=%s) -- "
                "LogisticRegression cannot fit this; fix the caller's split, "
                "do not silently degrade to a constant predictor." % (self.name, sorted(set(labels)))
            )
        X = np.stack(
            [self._pool(np.asarray(t, dtype=np.float32)) for t in trajectories], axis=0
        ).astype(np.float64)
        self.scaler_ = StandardScaler()
        Xs = self.scaler_.fit_transform(X)
        self.clf_ = LogisticRegression(max_iter=2000, random_state=self.seed)
        self.clf_.fit(Xs, labels)

    def score(self, trajectory) -> float:
        if self.scaler_ is None or self.clf_ is None:
            raise RuntimeError("%s.fit must be called before score" % self.name)
        x = self._pool(np.asarray(trajectory, dtype=np.float32)).astype(np.float64)
        xs = self.scaler_.transform(x.reshape(1, -1))
        return float(self.clf_.predict_proba(xs)[0, 1])


# --- 1. MeanPool ---------------------------------------------------------------
class MeanPool(_PooledAggregator):
    """TRUE mean over steps -- the literature's claimed failure mode, isolated.

    One term, `dim`-wide. This is the aggregator arXiv:2601.00516's abstract is
    describing when it says "mean-pooling embeddings dilutes anomalous steps" --
    unlike `cross_trajectory`'s `mean_agg`, there is no max or std term riding along.
    """

    name = "mean_pool"
    is_causal = True  # mean of steps 0..t never needs steps > t

    def _pool(self, trajectory: np.ndarray) -> np.ndarray:
        return trajectory.mean(axis=0)


# --- 2. MaxPool ------------------------------------------------------------------
class MaxPool(_PooledAggregator):
    """Per-dimension max over steps -- the literature's preferred alternative to mean
    pooling for sparse/rare-event signals (a single anomalous step should not be
    averaged away by many benign ones)."""

    name = "max_pool"
    is_causal = True  # max of steps 0..t never needs steps > t

    def _pool(self, trajectory: np.ndarray) -> np.ndarray:
        return trajectory.max(axis=0)


# --- 3. MeanMaxStdPool -------------------------------------------------------------
class MeanMaxStdPool(_PooledAggregator):
    """The `[mean, max, std]` concatenation `cross_trajectory/models.py:MeanAgg`
    actually computes under the name "mean_agg". Three terms, `3*dim`-wide. Kept
    separate from `MeanPool` on purpose: if this class outperforms `MeanPool` by a
    wide margin, that is direct evidence that `cross_trajectory`'s past "mean pooling
    wins" headline was substantially a MAX-pooling result wearing a mean-pooling
    label, not a refutation of the dilution claim.
    """

    name = "mean_max_std_pool"
    is_causal = True  # each term is a running statistic of steps 0..t only

    def _pool(self, trajectory: np.ndarray) -> np.ndarray:
        mean = trajectory.mean(axis=0)
        mx = trajectory.max(axis=0)
        std = trajectory.std(axis=0) if trajectory.shape[0] > 1 else np.zeros_like(mean)
        return np.concatenate([mean, mx, std], axis=0)


# --- 4. DeviationWeighted ----------------------------------------------------------
class DeviationWeighted(_PooledAggregator):
    """Weight each step by its distance from the TRAIN benign centroid, softmax-
    normalize the weights, then take the weighted mean. The deviation-aware weighting
    family's mechanism: a step that looks like ordinary benign behavior contributes
    little; a step that is far from "normal" dominates the pooled vector -- the
    opposite failure mode from plain mean pooling.

    Centroid: the mean step embedding over every step of every TRAIN trajectory with
    label 0 (benign), fit once in `fit()` and frozen for every subsequent `score()`
    call -- never derived from the query trajectory or from test labels.

    `is_causal = False`, and DELIBERATELY not flipped to True by fiat: as implemented,
    `score()` computes a softmax over ALL steps in the array it is handed in a single
    batched pass, and is not an incremental, O(1)-per-step update (contrast `GRUAggregator`'s
    hidden state, which IS such an update). Called on a genuine prefix it would not
    peek at future steps -- but this class does not implement the bounded-state
    `StreamingAggregator` interface from `../types.py`, and marking it causal without
    that would overstate what has actually been built. A running-centroid variant that
    updates online as steps arrive would earn `is_causal = True`; that variant is not
    implemented here.
    """

    name = "deviation_weighted"
    is_causal = False

    def __init__(self, temperature: float = 1.0, seed: int = 0):
        super().__init__(seed=seed)
        if temperature <= 0:
            raise ValueError("temperature must be > 0, got %r" % (temperature,))
        self.temperature = float(temperature)
        self.centroid_: np.ndarray | None = None

    def fit(self, trajectories, labels) -> None:
        trajectories = [np.asarray(t, dtype=np.float32) for t in trajectories]
        labels_int = [int(y) for y in labels]
        benign_steps = [
            step
            for traj, y in zip(trajectories, labels_int)
            if y == 0
            for step in traj
        ]
        if not benign_steps:
            # No benign trajectory in this fold -- fall back to the global step mean
            # rather than crashing; the weighting degrades toward uniform, which is
            # the honest behavior when the "benign" reference is undefined.
            benign_steps = [step for traj in trajectories for step in traj]
        self.centroid_ = np.mean(np.stack(benign_steps, axis=0), axis=0).astype(np.float32)
        super().fit(trajectories, labels_int)

    def _pool(self, trajectory: np.ndarray) -> np.ndarray:
        if self.centroid_ is None:
            raise RuntimeError("%s.fit must be called before score/_pool" % self.name)
        dist = np.linalg.norm(trajectory - self.centroid_[None, :], axis=1)
        weights = _softmax(dist / self.temperature)
        return (weights[:, None] * trajectory).sum(axis=0)


# --- 5. LastStep -- THE CONTROL ------------------------------------------------------
class LastStep(_PooledAggregator):
    """Score from the FINAL step alone. THE control.

    Without this baseline, a high AUC from any other aggregator in this file is
    consistent with "trajectory-level detection works" AND with "the last step alone
    already gives it away" -- the harness cannot distinguish the two hypotheses without
    measuring this one. Do not omit it and do not read a small margin over it as
    unimportant: a small margin means the trajectory-level structure is adding little
    over judging the final action.
    """

    name = "last_step"
    is_causal = True  # trivially: uses only the last step of whatever prefix it is given

    def _pool(self, trajectory: np.ndarray) -> np.ndarray:
        return trajectory[-1]


# --- 6. Truncate -- the WITHIN-CORPUS horizon control ----------------------------------
class Truncate:
    """Restrict every trajectory to its first `k` steps, then delegate to `inner`.

    WHY THIS IS A WRAPPER AND NOT A POOLING FUNCTION
    ------------------------------------------------
    The previous shape of this control (`FirstKSteps`, a `_PooledAggregator` whose
    `_pool` was `prefix.mean(axis=0)`) was MeanPool-at-k and nothing else. Sweeping it
    answers "how much signal sits in an early prefix" -- a curve that FALLS both when a
    prefix carries less information AND when a longer window dilutes a signal that is
    fully present in the prefix. Those two mechanisms are indistinguishable in it, and
    only the second one is the dilution claim this lesson exists to test.

    As a wrapper, the same truncation composes with ANY aggregator, so a single corpus
    yields the statistic the claim actually needs:

        delta(k) = AUC[Truncate(k, MaxPool)] - AUC[Truncate(k, MeanPool)]

    Both arms see the IDENTICAL first-k window of the IDENTICAL trajectories, so the
    information available is held fixed by construction and the only thing varying
    within a k is the pooling function. If delta(k) grows with k on ONE corpus, task,
    generator, tool inventory, prose style, kind-of-positive and label machinery are all
    held perfectly fixed -- which is exactly what a between-corpus SHADE-vs-AgentDojo
    contrast cannot do (see ../types.py). `horizon.py` computes that sweep.

    CAUSAL SAFETY
    -------------
    `_cut` is `array[:k]` -- a prefix. It indexes no element at position >= k, so
    truncation can never introduce lookahead, and `fit` truncates too (the classifier is
    fit in the same k-step world it is scored in; fitting on full trajectories and
    scoring on prefixes would be a train/test mismatch, not a horizon control).

    Truncation cannot REPAIR an acausal inner, so `is_causal` is inherited from `inner`
    rather than asserted True: `Truncate(k, QueryTokenCompressor)` is still acausal
    (unmasked attention over the whole window), and this class does not pretend otherwise.

    TRAJECTORIES SHORTER THAN k
    ---------------------------
    Policy: USE WHAT EXISTS. A trajectory with fewer than `k` steps is passed through
    whole -- never zero-padded (padding invents steps and drags a pooled mean toward the
    pad value) and never dropped (dropping makes the k-cells score different trajectory
    sets, so delta(k) would confound horizon with sample composition). The consequence is
    real and must be reported, not assumed away: at large k a corpus's short trajectories
    are effectively untruncated, so Truncate(k) converges to `inner` itself. Call
    `truncation_stats()` for the per-k counts; `horizon.py` writes them into results.json
    beside every AUC so a reader can see how many trajectories a given k actually bound.

    `k=None` means NO truncation (`array[:None]` is the whole array) -- the `k = all`
    cell of the sweep, and an anchor: `Truncate(None, MeanPool)` must reproduce the main
    ladder's `mean_pool` AUC exactly.
    """

    def __init__(self, k: int | None, inner, name: str | None = None):
        if k is not None:
            k = int(k)
            if k < 1:
                raise ValueError("k must be >= 1 or None (= no truncation), got %r" % (k,))
        self.k = k
        self.inner = inner
        self.k_label = "all" if k is None else str(k)
        self.name = name or ("trunc_%s_%s" % (self.k_label, getattr(inner, "name", "inner")))
        # Inherited, never asserted: truncation adds no lookahead but removes none either.
        self.is_causal = bool(getattr(inner, "is_causal", False))

    def _cut(self, trajectory) -> np.ndarray:
        a = np.asarray(trajectory, dtype=np.float32)
        return a[: self.k]  # k=None -> the whole array; k>=1 -> a strict prefix

    # -- Aggregator protocol (delegated; the wrapper holds no fitted state) --------
    def fit(self, trajectories, labels) -> None:
        self.inner.fit([self._cut(t) for t in trajectories], labels)

    def score(self, trajectory) -> float:
        return float(self.inner.score(self._cut(trajectory)))

    def truncation_stats(self, trajectories, labels=None) -> dict:
        """How many trajectories this k actually binds, and how many it leaves whole.

        `n_bound` = trajectories with >= k steps (genuinely truncated to k).
        `n_shorter_than_k` = trajectories with < k steps, passed through WHOLE under the
        use-what-exists policy above. Reported per class when `labels` is given, because
        a k that binds only the positives (or only the negatives) is a confound, not a
        horizon knob.
        """
        counts = [int(np.asarray(t).shape[0]) for t in trajectories]
        n = len(counts)
        if self.k is None:
            bound = [False] * n
            used = counts
        else:
            bound = [c >= self.k for c in counts]
            used = [min(c, self.k) for c in counts]
        out = {
            "k": self.k,
            "k_label": self.k_label,
            "short_policy": "use_available (no padding, no dropping)",
            "n_total": n,
            "n_bound": int(sum(bound)),
            "n_shorter_than_k": int(n - sum(bound)),
            "mean_steps_used": float(np.mean(used)) if n else 0.0,
            "median_steps_used": float(np.median(used)) if n else 0.0,
        }
        if labels is not None:
            lab = [int(y) for y in labels]
            for cls in (0, 1):
                idx = [i for i, y in enumerate(lab) if y == cls]
                out["n_bound_class%d" % cls] = int(sum(bound[i] for i in idx))
                out["n_shorter_than_k_class%d" % cls] = int(len(idx) - sum(bound[i] for i in idx))
        return out


class FirstKSteps(Truncate):
    """`Truncate(k, MeanPool)` under its historical name -- prefix-only TRUE mean.

    Kept as a named subclass so the existing `first_<k>_steps` rows in
    `artifacts/results_*.json` (and the `_plot_horizon` parser that reads `k` out of that
    name) stay valid across this refactor. It is now literally MeanPool-at-k rather than
    a separate pooling function that happened to be a mean, so the "mean-pool arm" of the
    within-corpus sweep and this curve are the SAME estimator by construction.

    On its own this curve still cannot separate "an early prefix carries less
    information" from "a long window dilutes it" -- that is what the MaxPool arm in
    `horizon.py` is for. Read it as a coverage curve, not as dilution evidence.
    """

    def __init__(self, k: int = 1, seed: int = 0):
        if k is None:
            raise ValueError("FirstKSteps requires an integer k; use Truncate(None, ...) "
                             "for the untruncated cell")
        super().__init__(k=k, inner=MeanPool(seed=seed), name="first_%d_steps" % int(k))
        self.seed = seed


# --- CPU smoke: python -m streaming_trajectory_aggregation.aggregators.pooling -------
if __name__ == "__main__":
    rng = np.random.default_rng(0)
    dim = 8

    def _make_traj(n_steps, spike):
        a = rng.normal(0.0, 1.0, size=(n_steps, dim)).astype(np.float32)
        if spike is not None:
            a[spike] += 6.0  # one clearly anomalous step
        return a

    trajs, labels = [], []
    for i in range(30):
        n = rng.integers(5, 15)
        trajs.append(_make_traj(n, spike=rng.integers(0, n)))
        labels.append(1)
    for i in range(30):
        n = rng.integers(5, 15)
        trajs.append(_make_traj(n, spike=None))
        labels.append(0)

    for agg in (
        MeanPool(),
        MaxPool(),
        MeanMaxStdPool(),
        DeviationWeighted(),
        LastStep(),
        FirstKSteps(k=3),
        Truncate(3, MaxPool()),
        Truncate(None, MeanPool()),
    ):
        agg.fit(trajs, labels)
        scores = [agg.score(t) for t in trajs]
        # crude self-consistency check only -- NOT a claimed AUC, just plumbing proof
        pos = [s for s, y in zip(scores, labels) if y == 1]
        neg = [s for s, y in zip(scores, labels) if y == 0]
        print(
            "%-24s is_causal=%-5s mean_score pos=%.3f neg=%.3f"
            % (agg.name, agg.is_causal, float(np.mean(pos)), float(np.mean(neg)))
        )

    # Truncate(None, inner) must BE inner, not merely resemble it -- assert the anchor.
    plain, wrapped = MeanPool(), Truncate(None, MeanPool())
    plain.fit(trajs, labels)
    wrapped.fit(trajs, labels)
    assert np.allclose([plain.score(t) for t in trajs], [wrapped.score(t) for t in trajs]), \
        "Truncate(None, MeanPool) diverged from MeanPool -- the k=all anchor is broken"

    # FirstKSteps(k) must BE Truncate(k, MeanPool) after the refactor.
    fk, tk = FirstKSteps(k=3), Truncate(3, MeanPool())
    fk.fit(trajs, labels)
    tk.fit(trajs, labels)
    assert np.allclose([fk.score(t) for t in trajs], [tk.score(t) for t in trajs]), \
        "FirstKSteps(k) diverged from Truncate(k, MeanPool)"

    # Causality: the score of a prefix must not move when later steps are rewritten.
    probe = Truncate(3, MaxPool())
    probe.fit(trajs, labels)
    t0 = trajs[0].copy()
    mangled = t0.copy()
    mangled[3:] += 50.0
    assert abs(probe.score(t0) - probe.score(mangled)) < 1e-9, \
        "Truncate leaked information from steps >= k"

    # Short-trajectory accounting is COUNTED, not silent.
    stats = Truncate(12, MeanPool()).truncation_stats(trajs, labels)
    assert stats["n_bound"] + stats["n_shorter_than_k"] == stats["n_total"]
    print("truncation_stats(k=12): n_bound=%d n_shorter=%d of %d (policy=%s)"
          % (stats["n_bound"], stats["n_shorter_than_k"], stats["n_total"],
             stats["short_policy"]))
    print("OK -- pooling.py aggregators fit/score end to end; Truncate anchors hold")
