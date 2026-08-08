"""esn_cusum.py -- ESNCusum: Echo-State Network reservoir + per-channel CUSUM.

Reference: Sunny Dubey, "Real-Time Detection and Repair of LLM Agent Failures"
(arXiv:2608.02464, 3 Aug 2026). Verified in CITATIONS_VERIFIED.md -- the id is real, but
the brief that spawned this lesson called the system "AgentTrajectorySentinel"; that
string is NOT this paper's title and must never be printed as one. The paper's own
one-class ESN ensemble + CUSUM alarms reach AUROC 0.872 at a 5% false-alarm budget and,
closed into a rollback/repair loop, raise task success 52% -> 73% at ~200
microseconds/step -- all three figures verbatim in its abstract. The abstract states the
52% -> 73% figure as OVERALL task success; whether it is scoped to a "booking agent" is
[NEEDS VERIFICATION] per CITATIONS_VERIFIED.md and that scoping is not claimed here.

THIS IS AN ADAPTATION, not a reproduction: our reservoir runs on Gemma step EMBEDDINGS
(continuous vectors), not the paper's raw agent telemetry fields, and our CUSUM null and
threshold are calibrated on our own synthetic/benchmark benign trajectories, not the
paper's 2,823-episode corpus. Only the mechanism -- a fixed random reservoir feeding a
per-channel CUSUM over a healthy-only null, fused by channel-max -- is reproduced. The
"~200 microseconds/step" number quoted above is the PAPER's; this file's own smoke test
measures OUR latency independently and never substitutes the paper's number for it.

CPU-only to import. Loads NO model. ASCII stdout only (Windows cp1252).
"""
from __future__ import annotations

import time
from typing import Sequence

import numpy as np

from ..types import StreamingAggregator

__all__ = ["ESNCusum"]


class ESNCusum:
    """Fixed random reservoir + per-channel CUSUM change detection, one-class (benign-only).

    State update (leaky-integrator ESN, reservoir weights FIXED at construction, never
    trained):

        h_t = (1 - a) * h_{t-1}  +  a * tanh(W_in @ x_t + W_res @ h_{t-1})

    ``fit()`` never touches W_in / W_res -- it only estimates the healthy-only CUSUM null
    (per-channel mean/std of reservoir activations) and calibrates an alarm threshold,
    and it does so ONLY on trajectories labelled benign (label == 0); any positive-class
    rows passed in are filtered out before anything is estimated, so this monitor can
    never be fit on the class it exists to detect.

    CUSUM (one-sided, per reservoir channel c, against the healthy null mu_c/sigma_c):

        z_t[c] = (h_t[c] - mu_c) / sigma_c
        g_t[c] = max(0, g_{t-1}[c] + z_t[c] - k)      # k = cusum_slack

    fused score = max_c g_t[c]   (channel-max fusion -- ANY channel drifting off-null
    fires the alarm, so a monitor need not know in advance which reservoir channel will
    carry a given failure mode).
    """

    name = "esn_cusum"
    is_causal = True

    def __init__(
        self,
        reservoir_dim: int = 100,
        spectral_radius: float = 0.9,
        leak_rate: float = 0.3,
        input_scale: float = 0.5,
        sparsity: float = 0.1,
        cusum_slack: float = 0.5,
        alarm_percentile: float = 95.0,
        seed: int = 0,
    ) -> None:
        self.reservoir_dim = reservoir_dim
        self.spectral_radius = spectral_radius
        self.leak_rate = leak_rate
        self.input_scale = input_scale
        self.sparsity = sparsity
        self.cusum_slack = cusum_slack
        self.alarm_percentile = alarm_percentile
        self.seed = seed

        self._rng = np.random.default_rng(seed)
        self._W_in: np.ndarray | None = None
        self._W_res: np.ndarray | None = None
        self._mu: np.ndarray | None = None          # [reservoir_dim] healthy-null channel mean
        self._sigma: np.ndarray | None = None        # [reservoir_dim] healthy-null channel std
        self._threshold: float | None = None
        self._fitted = False

        self.reset()

    # -- reservoir construction: fixed random weights, drawn once, never trained --------
    def _init_reservoir(self, input_dim: int) -> None:
        self._W_in = self._rng.uniform(-1.0, 1.0, size=(self.reservoir_dim, input_dim))
        self._W_in *= self.input_scale
        mask = self._rng.random((self.reservoir_dim, self.reservoir_dim)) < self.sparsity
        W = self._rng.uniform(-1.0, 1.0, size=(self.reservoir_dim, self.reservoir_dim)) * mask
        eigvals = np.linalg.eigvals(W)
        radius = float(np.max(np.abs(eigvals)))
        if radius > 1e-12:
            W = W * (self.spectral_radius / radius)
        self._W_res = W

    def _reservoir_step(self, x: np.ndarray) -> np.ndarray:
        pre = self._W_in @ x + self._W_res @ self._h
        self._h = (1.0 - self.leak_rate) * self._h + self.leak_rate * np.tanh(pre)
        return self._h

    # -- StreamingAggregator protocol ---------------------------------------------------
    def reset(self) -> None:
        """Clear per-trajectory state. Reservoir weights and the CUSUM null are NOT reset."""
        dim = self.reservoir_dim
        self._h = np.zeros(dim, dtype=np.float64)
        self._g = np.zeros(dim, dtype=np.float64)     # per-channel CUSUM accumulator
        self._peak = 0.0
        self._step_count = np.int64(0)
        self.alarm_step: int | None = None            # first step exceeding threshold, if any

    def update(self, step_vec: np.ndarray) -> float:
        """Ingest ONE step. O(1) in trajectory length: cost depends only on reservoir_dim."""
        x = np.asarray(step_vec, dtype=np.float64)
        if self._W_res is None:
            self._init_reservoir(x.shape[-1])
        h = self._reservoir_step(x)
        mu = self._mu if self._mu is not None else np.zeros_like(h)
        sigma = self._sigma if self._sigma is not None else np.ones_like(h)
        z = (h - mu) / sigma
        self._g = np.maximum(0.0, self._g + z - self.cusum_slack)
        score = float(np.max(self._g))
        self._peak = max(self._peak, score)
        self._step_count += 1
        if self.alarm_step is None and self._threshold is not None and score > self._threshold:
            self.alarm_step = int(self._step_count) - 1     # 0-indexed step of first alarm
        return score

    def state_bytes(self) -> int:
        """Reservoir state h + CUSUM accumulator g + a step counter. Constant in steps seen."""
        return int(self._h.nbytes + self._g.nbytes + np.asarray(self._step_count).nbytes)

    def fit(self, trajectories: Sequence[np.ndarray], labels: Sequence[int]) -> None:
        """Fit the healthy-only CUSUM null and alarm threshold. NEVER fits on label==1."""
        benign = [np.asarray(t, dtype=np.float64) for t, y in zip(trajectories, labels) if y == 0]
        if not benign:
            raise ValueError("ESNCusum.fit requires at least one benign (label=0) trajectory")
        if self._W_res is None:
            self._init_reservoir(benign[0].shape[-1])

        # pass 1: run the FIXED reservoir over all benign trajectories and collect
        # per-channel activation stats -- the healthy-only null the CUSUM is set against.
        acts = []
        for traj in benign:
            self.reset()
            for step_vec in traj:
                acts.append(self._reservoir_step(np.asarray(step_vec, dtype=np.float64)).copy())
        A = np.stack(acts, axis=0)
        self._mu = A.mean(axis=0)
        self._sigma = A.std(axis=0) + 1e-6

        # pass 2: with the null fixed, run CUSUM over the same benign trajectories and
        # calibrate the alarm threshold from the distribution of per-trajectory PEAK
        # scores -- a design-time false-alarm budget (~(100 - alarm_percentile)% on the
        # benign train set), the same discipline SafetyDriftMonitor uses for its threshold.
        self._threshold = float("inf")   # disable alarms during calibration
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
    rng = np.random.default_rng(42)
    dim = 16

    def make_traj(n_steps: int, drift: float) -> np.ndarray:
        # Synthetic step embeddings for a CPU smoke ONLY -- not a claim about real agent
        # trajectories. Benign trajectories wander around 0; "malicious" trajectories ramp
        # steadily along a fixed random direction, giving the monitor something to catch.
        base = rng.normal(0.0, 1.0, size=(n_steps, dim))
        ramp = np.linspace(0.0, drift, n_steps)[:, None]
        direction = rng.normal(0.0, 1.0, size=(1, dim))
        direction /= np.linalg.norm(direction)
        return base + ramp * direction

    train_trajs = [make_traj(int(rng.integers(30, 70)), drift=0.0) for _ in range(30)]
    train_labels = [0] * len(train_trajs)

    mon = ESNCusum(reservoir_dim=64, seed=0)
    assert isinstance(mon, StreamingAggregator), "ESNCusum must satisfy StreamingAggregator"
    assert mon.is_causal is True

    mon.fit(train_trajs, train_labels)

    eval_benign = make_traj(40, drift=0.0)
    eval_bad = make_traj(40, drift=6.0)
    print("benign eval score=%.4f alarm_step=%s" % (mon.score(eval_benign), mon.alarm_step))
    print("malicious eval score=%.4f alarm_step=%s" % (mon.score(eval_bad), mon.alarm_step))

    # O(1) state check: state_bytes() must be FLAT between step 10 and step 500, and
    # latency is OUR measurement on this machine, never substituted for the paper's number.
    mon.reset()
    long_traj = make_traj(500, drift=3.0)
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
    print("OK -- esn_cusum smoke complete")
