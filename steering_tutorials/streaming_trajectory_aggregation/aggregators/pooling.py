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


# --- 6. FirstKSteps -- the horizon-truncation curve -----------------------------------
class FirstKSteps(_PooledAggregator):
    """Prefix-only: mean-pool just the first `k` steps (or all of them, if the
    trajectory is shorter than `k`). Sweeping `k` across a run of this class produces
    the horizon-truncation curve -- how much AUC is recoverable from an early prefix
    before the rest of a long trajectory (e.g. SHADE's median 131 steps) has even
    happened. Uses the same TRUE-mean aggregation as `MeanPool` so a `FirstKSteps(k)`
    vs `MeanPool` (= `FirstKSteps(inf)`) comparison isolates the horizon effect from
    the pooling-function effect.
    """

    is_causal = True  # a fixed-length prefix by construction never sees later steps

    def __init__(self, k: int = 1, seed: int = 0):
        super().__init__(seed=seed)
        if k < 1:
            raise ValueError("k must be >= 1, got %r" % (k,))
        self.k = int(k)
        self.name = "first_%d_steps" % self.k

    def _pool(self, trajectory: np.ndarray) -> np.ndarray:
        prefix = trajectory[: self.k]
        return prefix.mean(axis=0)


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

    for cls, kwargs in (
        (MeanPool, {}),
        (MaxPool, {}),
        (MeanMaxStdPool, {}),
        (DeviationWeighted, {}),
        (LastStep, {}),
        (FirstKSteps, {"k": 3}),
    ):
        agg = cls(**kwargs)
        agg.fit(trajs, labels)
        scores = [agg.score(t) for t in trajs]
        # crude self-consistency check only -- NOT a claimed AUC, just plumbing proof
        pos = [s for s, y in zip(scores, labels) if y == 1]
        neg = [s for s, y in zip(scores, labels) if y == 0]
        print(
            "%-20s is_causal=%-5s mean_score pos=%.3f neg=%.3f"
            % (agg.name, agg.is_causal, float(np.mean(pos)), float(np.mean(neg)))
        )
    print("OK -- pooling.py aggregators fit/score end to end")
