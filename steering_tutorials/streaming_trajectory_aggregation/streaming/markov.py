"""markov.py -- SafetyDriftMonitor: cumulative safety state as an ABSORBING MARKOV CHAIN.

Reference: Aditya Dhodapkar, Farhaan Pishori, "SafetyDrift: Predicting When AI Agents
Cross the Line Before They Actually Do" (arXiv:2603.27148, 28 Mar 2026). Verified in
CITATIONS_VERIFIED.md. The paper is SUBMITTED to COLM ("sent to COLM conference" per its
own comments field) -- NOT accepted -- and must never be described as a COLM paper here.
The paper models agent safety trajectories as absorbing Markov chains over a hand-designed
(data exposure, tool escalation, reversibility) risk-state triple, fit from labelled
trajectories that actually reach a violation, and reports catching 94.7% of violations
with 3.7 steps of advance warning -- both verbatim in its abstract. "<0.001 ms/step" does
NOT appear in the abstract (only "lightweight" does) -- [NEEDS VERIFICATION] per
CITATIONS_VERIFIED.md -- and is not claimed here.

THIS IS AN ADAPTATION of the mechanism, not a reproduction of the paper's method or its
94.7% / 3.7-step numbers. Two departures, stated plainly:

  1. Our per-step "risk state" is a K-MEANS cluster of Gemma step EMBEDDINGS, fit on TRAIN
     data only and ordered by distance from the benign centroid -- not the paper's
     hand-designed (data exposure, tool escalation, reversibility) triple. This lesson has
     no such labelled telemetry; clustering embeddings is our stand-in, and it is a weaker,
     unvalidated substitute for the paper's engineered features.
  2. The paper's transition matrix is fit from trajectories that are labelled and actually
     reach "violated". This monitor is trained ONE-CLASS -- on label == 0 (benign) only,
     per this lesson's non-negotiable -- so it NEVER observes a real transition into the
     absorbing state. The absorbing state's incoming probability is instead assigned by a
     MONOTONE HEURISTIC over cluster risk-rank (the riskiest cluster gets the largest, but
     still small, per-step escape probability). That heuristic is a design choice standing
     in for data we deliberately never fit on -- not a learned quantity -- which is exactly
     why this monitor is a mechanism ADAPTATION, not a validated reproduction.

Runtime scoring is a lookup, per the paper's own framing: cluster assignment (nearest
centroid, cost independent of trajectory length) then a table read of a finite-horizon
absorption probability P(violated within `horizon` steps | state), precomputed once in
fit() via matrix powers of the transient sub-stochastic transition matrix Q:

    P(absorbed within h steps | state i) = 1 - sum_j (Q^h)[i, j]

CPU-only to import. Loads NO model. ASCII stdout only (Windows cp1252).
"""
from __future__ import annotations

import time
from typing import Sequence

import numpy as np
from sklearn.cluster import KMeans

from ..types import StreamingAggregator

__all__ = ["SafetyDriftMonitor"]


class SafetyDriftMonitor:
    """Discretised risk state + absorbing-chain lookup, one-class (benign-only) fit.

    Scoring at runtime is O(1) in trajectory length: one nearest-centroid lookup (cost
    depends only on n_states, not on steps seen) followed by one array read from a
    precomputed [n_states] table. All the Markov-chain algebra (transition-matrix
    estimation, matrix powers) happens once, inside fit(), never at scoring time.
    """

    name = "safetydrift"
    is_causal = True

    def __init__(
        self,
        n_states: int = 8,
        horizon: int = 10,
        max_absorb_rate: float = 0.15,
        alarm_percentile: float = 95.0,
        laplace_alpha: float = 1.0,
        seed: int = 0,
    ) -> None:
        self.n_states = n_states
        self.horizon = horizon
        self.max_absorb_rate = max_absorb_rate
        self.alarm_percentile = alarm_percentile
        self.laplace_alpha = laplace_alpha
        self.seed = seed

        self._kmeans: KMeans | None = None
        self._risk_rank: np.ndarray | None = None            # cluster idx -> risk rank 0..K-1
        self._absorb_prob_table: np.ndarray | None = None     # [n_states] P(violated | state)
        self._threshold: float | None = None
        self._fitted = False

        self.reset()

    # -- StreamingAggregator protocol ---------------------------------------------------
    def reset(self) -> None:
        self._current_state = -1
        self._peak = 0.0
        self._step_count = np.int64(0)
        self.alarm_step: int | None = None            # first step exceeding threshold, if any

    def _discretize(self, x: np.ndarray) -> int:
        x = np.asarray(x, dtype=np.float64).reshape(1, -1)
        return int(self._kmeans.predict(x)[0])

    def update(self, step_vec: np.ndarray) -> float:
        """Ingest ONE step. O(1) in trajectory length: a nearest-centroid lookup + a table read."""
        if self._kmeans is None or self._absorb_prob_table is None:
            raise RuntimeError("SafetyDriftMonitor.update called before fit()")
        s = self._discretize(step_vec)
        self._current_state = s
        score = float(self._absorb_prob_table[s])
        self._peak = max(self._peak, score)
        self._step_count += 1
        if self.alarm_step is None and self._threshold is not None and score > self._threshold:
            self.alarm_step = int(self._step_count) - 1     # 0-indexed step of first alarm
        return score

    def state_bytes(self) -> int:
        """Current cluster id + running peak + a step counter. Constant in steps seen."""
        return int(
            np.asarray(self._current_state).nbytes
            + np.asarray(self._peak).nbytes
            + np.asarray(self._step_count).nbytes
        )

    def fit(self, trajectories: Sequence[np.ndarray], labels: Sequence[int]) -> None:
        """Discretise, fit the transient transition matrix, and precompute the absorption
        table -- ALL from label==0 (benign) trajectories only. Positive-class rows passed
        in are filtered out before anything is estimated."""
        benign = [np.asarray(t, dtype=np.float64) for t, y in zip(trajectories, labels) if y == 0]
        if not benign:
            raise ValueError("SafetyDriftMonitor.fit requires at least one benign (label=0) trajectory")

        all_steps = np.concatenate(benign, axis=0)
        self._kmeans = KMeans(n_clusters=self.n_states, random_state=self.seed, n_init=10)
        self._kmeans.fit(all_steps)

        # Order clusters by risk = distance of each centroid from the benign global mean
        # ("safe centre"). Rank 0 = closest = safest; rank n_states-1 = furthest = riskiest.
        safe_center = all_steps.mean(axis=0)
        centroid_dist = np.linalg.norm(self._kmeans.cluster_centers_ - safe_center, axis=1)
        order = np.argsort(centroid_dist)
        self._risk_rank = np.empty(self.n_states, dtype=np.int64)
        self._risk_rank[order] = np.arange(self.n_states)     # cluster idx -> risk rank

        # Transient transition matrix Q over the n_states risk clusters, Laplace-smoothed,
        # estimated from benign-only consecutive-step transitions.
        counts = np.full((self.n_states, self.n_states), self.laplace_alpha, dtype=np.float64)
        for traj in benign:
            state_seq = self._kmeans.predict(traj)
            for a, b in zip(state_seq[:-1], state_seq[1:]):
                counts[a, b] += 1.0
        Q_raw = counts / counts.sum(axis=1, keepdims=True)

        # ADAPTATION (see module/class docstring): the absorbing "violated" state is never
        # observed in benign-only training, so its incoming probability per cluster is a
        # monotone heuristic of risk rank, not a fit quantity -- rising linearly with rank,
        # capped at max_absorb_rate for the single riskiest cluster.
        rank_frac = self._risk_rank.astype(np.float64) / max(1, self.n_states - 1)
        p_absorb = rank_frac * self.max_absorb_rate
        Q = Q_raw * (1.0 - p_absorb)[:, None]   # each row now sums to (1 - p_absorb[i])

        # Finite-horizon absorption probability, analytically, ONCE:
        #   P(absorbed within h steps | state i) = 1 - sum_j (Q^h)[i, j]
        Qh = np.linalg.matrix_power(Q, self.horizon)
        self._absorb_prob_table = 1.0 - Qh.sum(axis=1)

        # Calibrate the alarm threshold from the distribution of per-trajectory PEAK
        # absorption-probability lookups on the benign train set -- the same design-time
        # false-alarm-budget discipline ESNCusum uses for its CUSUM threshold.
        self._threshold = float("inf")
        peaks = []
        for traj in benign:
            self.reset()
            for step_vec in traj:
                self.update(step_vec)
            peaks.append(self._peak)
        self._threshold = float(np.percentile(peaks, self.alarm_percentile))
        self.reset()
        self._fitted = True

    def score(self, trajectory: np.ndarray) -> float:
        """Offline convenience: reset, stream every step through update(), return the peak."""
        self.reset()
        for step_vec in np.asarray(trajectory, dtype=np.float64):
            self.update(step_vec)
        return self._peak


# --- CPU smoke -------------------------------------------------------------------------
if __name__ == "__main__":
    rng = np.random.default_rng(7)
    dim = 16
    step_std = 0.12
    # A single fixed "risk direction" shared by every trajectory (train and eval alike) --
    # a persistent random walk with zero drift stays a bounded benign blob near the
    # origin; a walk with steady drift along this SAME direction is what a monitor with
    # any temporal structure should be able to catch, unlike i.i.d. per-step noise which
    # would make every cluster equally likely regardless of drift.
    risk_direction = rng.normal(0.0, 1.0, size=dim)
    risk_direction /= np.linalg.norm(risk_direction)

    def make_traj(n_steps: int, drift_per_step: float) -> np.ndarray:
        # Synthetic step embeddings for a CPU smoke ONLY -- not a claim about real agent
        # trajectories. drift_per_step=0.0 is a small bounded random walk around the
        # origin (benign); drift_per_step > 0 steadily pushes the walk along
        # risk_direction, away from the benign training manifold (malicious).
        steps = np.zeros((n_steps, dim))
        x = np.zeros(dim)
        for t in range(n_steps):
            x = x + rng.normal(0.0, step_std, size=dim) + drift_per_step * risk_direction
            steps[t] = x
        return steps

    train_trajs = [make_traj(int(rng.integers(30, 70)), drift_per_step=0.0) for _ in range(30)]
    train_labels = [0] * len(train_trajs)

    mon = SafetyDriftMonitor(n_states=8, horizon=10, seed=0)
    assert isinstance(mon, StreamingAggregator), "SafetyDriftMonitor must satisfy StreamingAggregator"
    assert mon.is_causal is True

    mon.fit(train_trajs, train_labels)

    eval_benign = make_traj(40, drift_per_step=0.0)
    eval_bad = make_traj(40, drift_per_step=0.4)
    print("benign eval score=%.4f alarm_step=%s" % (mon.score(eval_benign), mon.alarm_step))
    print("malicious eval score=%.4f alarm_step=%s (higher score = more drift toward the"
          " absorbing state; this toy calibration may or may not cross the alarm"
          " threshold -- both outcomes are reported honestly, not massaged)"
          % (mon.score(eval_bad), mon.alarm_step))

    # O(1) state check: state_bytes() must be FLAT between step 10 and step 500, and
    # latency is OUR measurement on this machine, never substituted for the paper's number.
    mon.reset()
    long_traj = make_traj(500, drift_per_step=0.2)
    state_at_10 = None
    state_at_500 = None
    t0 = time.perf_counter()
    for i, step_vec in enumerate(long_traj):
        mon.update(step_vec)
        if i == 9:
            state_at_10 = mon.state_bytes()
        if i == 499:
            state_at_500 = mon.state_bytes()
    elapsed_s = time.perf_counter() - t0
    us_per_step = (elapsed_s / len(long_traj)) * 1e6

    print("state_bytes at step 10  = %d" % state_at_10)
    print("state_bytes at step 500 = %d" % state_at_500)
    flat = state_at_10 == state_at_500
    print("O(1) state check: %s (flat=%s)" % ("PASS" if flat else "FAIL", flat))
    print("measured latency: %.2f microseconds/step (this run, this machine, NOT the paper's number)"
          % us_per_step)
    print("OK -- markov smoke complete")
