"""intent_gate.py — calibrated, per-category intent detector (the "WHEN").

This is AXIS 9 (the CONDITION) specialised to *multiple* intent categories. Where
``gate.LogisticGate`` is a single binary harm-relevance gate, ``IntentGate`` is a
bank of one-vs-rest logistic gates — one per intent category (e.g. "self-harm",
"weapons", "privacy", "benign") — over the multi-layer condition features
produced by ``gate.condition_features``. It answers, per prompt: *which* intents
are present, with a per-category probability and a per-category firing threshold.

Two things make it deployable rather than a bare classifier:

  * ``calibrate_thresholds`` — picks each category's firing threshold at a TARGET
    false-positive rate on the negatives (everything-not-this-category). This is
    the over-refusal control knob: a low ``target_fpr`` keeps benign prompts from
    tripping a safety intent, trading recall for selectivity (CLAUDE.md §10,
    axis 5). Without it a 0.5 cutoff over-fires on the long benign tail.
  * ``expected_calibration_error`` — reports whether the predicted probabilities
    are trustworthy (do prompts the gate calls "0.9 harmful" turn out harmful
    ~90% of the time). A miscalibrated gate makes any threshold meaningless.

It reuses ``gate.LogisticGate`` verbatim for each one-vs-rest head and
``gate.condition_features`` (re-exposed as ``IntentGate.extract_features``) for
the model-touching feature step, so the numpy↔torch seam and the convex,
L2-regularised training are inherited, not re-implemented.

All math is numpy float32. Deterministic given the LogisticGate seed.

STATUS: BUILT but NOT YET VALIDATED on real models / benchmarks. The unit tests
exercise firing/abstaining, the calibrated FPR, and ECE on synthetic separable
features; no real intent taxonomy has been fit on Gemma activations yet.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import torch.nn as nn

from .gate import LogisticGate, condition_features

_EPS = 1e-8


class IntentGate:
    """A bank of one-vs-rest logistic gates, one per intent category.

    Categories are discovered from the training labels (any hashable label —
    ``int`` or ``str``) and stored in sorted order as ``self.categories``. Each
    category gets its own ``gate.LogisticGate`` trained one-vs-rest on the shared
    feature matrix; ``predict_proba`` returns the per-category probabilities
    side by side (columns aligned to ``self.categories``).
    """

    def __init__(self) -> None:
        self.categories: list = []
        self.gates_: dict[object, LogisticGate] = {}
        self.thresholds_: dict[object, float] = {}
        self._fitted: bool = False

    # ---- feature extraction (model-touching; thin reuse of gate.py) --------- #
    @staticmethod
    def extract_features(
        model: nn.Module,
        tok,
        prompts: Sequence[str],
        layers: Sequence[int],
        condition_vectors: dict[int, np.ndarray],
    ) -> np.ndarray:
        """Multi-layer condition features for ``prompts`` -> ``[n, n_layers]``.

        A thin pass-through to ``gate.condition_features`` so callers can build
        IntentGate features without reaching into ``gate`` directly (the
        ``condition_features(model,tok,prompts,layers,vectors)`` reuse named in
        the method contract).
        """
        return condition_features(model, tok, prompts, layers, condition_vectors)

    # ---- training ----------------------------------------------------------- #
    def fit(
        self,
        features: np.ndarray,
        labels: Sequence,
        l2: float = 0.1,
        epochs: int = 500,
        lr: float = 0.5,
        seed: int = 0,
    ) -> "IntentGate":
        """Fit one ``LogisticGate`` per category, one-vs-rest.

        Parameters
        ----------
        features : ``[n, k]`` condition features (1-D treated as k=1).
        labels   : length-n sequence of category labels (int or str).
        l2       : L2 weight decay passed to each LogisticGate.
        epochs/lr/seed : training hyper-parameters passed straight through.

        Sets ``self.categories`` (sorted unique labels) and ``self.gates_``.
        Thresholds default to 0.5 until ``calibrate_thresholds`` is called.
        """
        x = np.asarray(features, dtype=np.float32)
        if x.ndim == 1:
            x = x.reshape(-1, 1)
        y = np.asarray(list(labels))
        if x.shape[0] != y.shape[0]:
            raise ValueError("features and labels must have equal length")
        # Sorted unique categories (works for int and str labels alike).
        self.categories = sorted(set(y.tolist()))
        self.gates_ = {}
        self.thresholds_ = {}
        for cat in self.categories:
            yc = (y == cat).astype(np.float64)
            gate = LogisticGate().fit(x, yc, l2=l2, epochs=epochs, lr=lr, seed=seed)
            self.gates_[cat] = gate
            self.thresholds_[cat] = 0.5
        self._fitted = True
        return self

    def _check_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError("IntentGate.predict_* called before fit()")

    # ---- inference ---------------------------------------------------------- #
    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        """Per-category probabilities ``[n, K]`` (columns aligned to ``categories``).

        Each column is the one-vs-rest sigmoid probability from that category's
        gate; columns do NOT sum to 1 (independent detectors, by design — several
        intents may be present at once).
        """
        self._check_fitted()
        x = np.asarray(features, dtype=np.float32)
        if x.ndim == 1:
            x = x.reshape(-1, 1)
        out = np.zeros((x.shape[0], len(self.categories)), dtype=np.float32)
        for j, cat in enumerate(self.categories):
            out[:, j] = self.gates_[cat].score(x)
        return out

    def predict(self, features: np.ndarray) -> np.ndarray:
        """Per-category firing decisions ``[n, K]`` int ``{0,1}`` at the (calibrated)
        thresholds. Fires category ``c`` where ``predict_proba[:, c] > τ_c``."""
        proba = self.predict_proba(features)
        thr = np.array(
            [self.thresholds_.get(cat, 0.5) for cat in self.categories],
            dtype=np.float32,
        )
        return (proba > thr).astype(np.int64)

    # ---- calibration -------------------------------------------------------- #
    def calibrate_thresholds(
        self,
        features: np.ndarray,
        labels: Sequence,
        target_fpr: float = 0.01,
    ) -> dict:
        """Set each category's threshold at a target false-positive rate.

        For category ``c`` the negatives are all rows whose label != c. The
        threshold is the ``(1 - target_fpr)`` quantile of the negatives' scores,
        so (on this calibration set) only ~``target_fpr`` of benign/other prompts
        exceed it and trip the intent. This is the over-refusal control: smaller
        ``target_fpr`` ⇒ more selective gate. Updates and returns
        ``self.thresholds_`` ``{category: threshold}``.
        """
        self._check_fitted()
        if not 0.0 < target_fpr < 1.0:
            raise ValueError("target_fpr must be in (0, 1)")
        proba = self.predict_proba(features)
        y = np.asarray(list(labels))
        q = 1.0 - float(target_fpr)
        for j, cat in enumerate(self.categories):
            neg_scores = proba[y != cat, j]
            if neg_scores.size == 0:
                # No negatives to bound the FPR against; fall back to 0.5.
                self.thresholds_[cat] = 0.5
            else:
                # nextafter nudges strictly above the quantile so a negative
                # sitting exactly at it does NOT fire (FPR <= target, not >).
                thr = float(np.quantile(neg_scores, q))
                self.thresholds_[cat] = float(np.nextafter(thr, np.float64(1.0)))
        return dict(self.thresholds_)

    def expected_calibration_error(
        self,
        features: np.ndarray,
        labels: Sequence,
        n_bins: int = 10,
    ) -> float:
        """Expected Calibration Error of the top-1 prediction (lower = better).

        The per-category probabilities are L1-normalised into a distribution; the
        top-1 category's probability is the confidence and the bin accuracy is how
        often that top-1 category is the true label. ECE is the confidence-
        weighted average gap between confidence and accuracy across ``n_bins``
        equal-width confidence bins (the standard Guo et al. 2017 estimator):

            ECE = Σ_b (n_b / n) · | acc(b) − conf(b) |
        """
        self._check_fitted()
        proba = self.predict_proba(features)
        y = np.asarray(list(labels))
        # L1-normalise the independent detector scores into a distribution so the
        # "confidence" of the top-1 category is a probability in [0, 1].
        row_sum = proba.sum(axis=1, keepdims=True)
        dist = proba / (row_sum + _EPS)
        conf = dist.max(axis=1)
        top_idx = dist.argmax(axis=1)
        cats = np.array(self.categories, dtype=object)
        pred_cat = cats[top_idx]
        correct = (pred_cat == y).astype(np.float64)

        n = conf.shape[0]
        if n == 0:
            return 0.0
        edges = np.linspace(0.0, 1.0, int(n_bins) + 1)
        ece = 0.0
        for b in range(int(n_bins)):
            lo, hi = edges[b], edges[b + 1]
            # Last bin is closed on the right so confidence==1.0 is counted.
            if b == int(n_bins) - 1:
                in_bin = (conf >= lo) & (conf <= hi)
            else:
                in_bin = (conf >= lo) & (conf < hi)
            count = int(in_bin.sum())
            if count == 0:
                continue
            acc_bin = float(correct[in_bin].mean())
            conf_bin = float(conf[in_bin].mean())
            ece += (count / n) * abs(acc_bin - conf_bin)
        return float(ece)
