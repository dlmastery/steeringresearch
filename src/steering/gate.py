"""gate.py — AXIS-9 conditional gating: the learned-gate-vs-fixed-threshold infra (E15).

This module is the implementation backbone of hypothesis **E15** ("a learned
logistic gate on multi-layer activations beats a fixed cosine threshold under
distribution shift"; see
`hypotheses/B_conditional/E15_learned_gate_vs_fixed_threshold.md`). It is AXIS 9
(the CONDITION — *when* to steer) in the 12-axis taxonomy: a gate decides, per
prompt, whether the input is harm-relevant and therefore whether the steering
operation should fire at all. It does NOT edit the residual stream itself
(that is `hooks.py`, AXIS 4); it produces the on/off decision that a CAST-style
conditional wrapper consumes.

Two gate families are provided, mirroring the E15 comparison:

  * ``CosineGate``  — the fixed single-layer threshold baseline (E9 / CAST).
    A prompt's feature is the dot product of its mean-pooled activation at one
    layer with that layer's unit condition vector; the gate fires when that
    scalar exceeds a threshold chosen on a development set (max balanced
    accuracy / Youden's J). One scalar parameter, no training loop.

  * ``LogisticGate`` — the learned multi-layer gate (the E15 contribution).
    A logistic regression over the k-dimensional multi-layer feature vector
    (one cosine-style feature per layer, §5.1), trained by gradient descent on
    binary-cross-entropy with L2 weight decay (§5.2). Standardises features
    using train-set statistics. This is the small, convex, overfit-resistant
    "training infrastructure the hypothesis needs".

Metrics (``pr_auc``, ``roc_auc``) are implemented in PURE numpy on purpose: the
gate-quality axis (§7, gate PR-AUC; §6 predicted-delta is in AUC terms) must not
add or depend on a heavier classifier library, and a from-scratch implementation
keeps the metric auditable and the offline test path dependency-light.

Everything here is offline-capable and deterministic: ``condition_features``
reads activations through ``hooks.probe_activations`` (works on BOTH
``FakeResidualLM`` and real Gemma); the gates train with fixed seeds; the metrics
are pure functions of (scores, labels). The model-touching part
(``condition_features``) and the pure part (the gates, metrics, ``evaluate_gates``)
are separated so the scientific comparison is unit-testable without a model.

All math is float32 numpy / torch.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import torch
import torch.nn as nn

from .hooks import probe_activations
from .model import encode_to_device

_EPS = 1e-8


# --------------------------------------------------------------------------- #
# Feature extraction (the model-touching part)                                #
# --------------------------------------------------------------------------- #
def condition_features(
    model: nn.Module,
    tokenizer,
    prompts: Sequence[str],
    layers: Sequence[int],
    condition_vectors: dict[int, np.ndarray],
) -> np.ndarray:
    """Multi-layer condition feature matrix for a batch of prompts (E15 §5.1).

    For each prompt we run ONE forward pass, mean-pool the residual activation at
    every requested layer (over the sequence axis), and dot that pooled vector
    with the layer's UNIT condition vector. The result is the multi-layer feature
    vector ``[h_L1 @ v_L1, ..., h_Lk @ v_Lk]`` that both gates consume — a high
    value on a layer means "this prompt's activation points along that layer's
    condition direction".

    Parameters
    ----------
    model             : FakeResidualLM or real Gemma.
    tokenizer         : matching tokenizer (real HF or the offline _FakeTokenizer).
    prompts           : list of prompt strings.
    layers            : layer indices to read (e.g. (6, 10, 14, 18) from §5.2).
    condition_vectors : {layer_idx: vector [dim]}. Each is unit-normalised here,
                        so the feature is a cosine-like projection regardless of
                        the raw vector norm. A vector for every requested layer
                        must be present.

    Returns
    -------
    np.ndarray ``[n_prompts, n_layers]`` float32, column j aligned to
    ``layers[j]``. Always finite for finite inputs.
    """
    layers = list(layers)
    missing = [li for li in layers if li not in condition_vectors]
    if missing:
        raise KeyError(
            f"condition_vectors is missing a vector for layer(s) {missing}; "
            f"have {sorted(condition_vectors)}, need {layers}."
        )

    # Pre-unit-normalise the condition vectors once (float32).
    unit_vecs: dict[int, np.ndarray] = {}
    for li in layers:
        v = np.asarray(condition_vectors[li], dtype=np.float32).reshape(-1)
        unit_vecs[li] = v / (float(np.linalg.norm(v)) + _EPS)

    feats = np.zeros((len(prompts), len(layers)), dtype=np.float32)
    for row, text in enumerate(prompts):
        input_ids = encode_to_device(tokenizer, text, model)
        acts = probe_activations(model, input_ids, layers)
        for col, li in enumerate(layers):
            # acts[li]: [batch=1, seq, dim] -> mean over seq -> [dim]
            pooled = acts[li][0].mean(dim=0).float().cpu().numpy()
            feats[row, col] = float(np.dot(pooled, unit_vecs[li]))
    return feats


# --------------------------------------------------------------------------- #
# Pure-numpy ranking metrics (no sklearn — keep the metric auditable + light)  #
# --------------------------------------------------------------------------- #
def _as_1d(scores: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    s = np.asarray(scores, dtype=np.float64).reshape(-1)
    y = np.asarray(labels, dtype=np.float64).reshape(-1)
    if s.shape != y.shape:
        raise ValueError(f"scores {s.shape} and labels {y.shape} must have equal length")
    return s, y


def roc_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """ROC-AUC (area under the ROC curve) in pure numpy.

    Computed via the Mann-Whitney U statistic with proper tie handling
    (rank-average): AUC = (mean rank of positives - (n_pos+1)/2) / n_neg. Returns
    0.5 for a degenerate single-class input (no ranking is possible).
    """
    s, y = _as_1d(scores, labels)
    pos = y == 1
    n_pos = int(pos.sum())
    n_neg = int((~pos).sum())
    if n_pos == 0 or n_neg == 0:
        return 0.5
    # Average ranks (1..N), ties share the mean rank.
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s), dtype=np.float64)
    sorted_s = s[order]
    i = 0
    n = len(s)
    while i < n:
        j = i
        while j + 1 < n and sorted_s[j + 1] == sorted_s[i]:
            j += 1
        avg_rank = (i + j) / 2.0 + 1.0  # 1-based average rank for the tie block
        ranks[order[i : j + 1]] = avg_rank
        i = j + 1
    sum_ranks_pos = float(ranks[pos].sum())
    auc = (sum_ranks_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)


def pr_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """PR-AUC = average precision (area under the precision-recall curve), pure numpy.

    Average precision: sum over thresholds of (recall_k - recall_{k-1}) *
    precision_k, walking the examples in descending score order (the §7 gate
    metric). Ties in score are processed as a group so the curve does not depend
    on the arbitrary order of equal-scored examples. For a degenerate input with
    no positives, returns 0.0.
    """
    s, y = _as_1d(scores, labels)
    n_pos = int((y == 1).sum())
    if n_pos == 0:
        return 0.0
    order = np.argsort(-s, kind="mergesort")  # descending score
    s_sorted = s[order]
    y_sorted = y[order]

    ap = 0.0
    tp = 0
    fp = 0
    prev_recall = 0.0
    n = len(s_sorted)
    i = 0
    while i < n:
        # consume a whole block of equal scores at once (tie handling)
        j = i
        while j + 1 < n and s_sorted[j + 1] == s_sorted[i]:
            j += 1
        block = y_sorted[i : j + 1]
        tp += int((block == 1).sum())
        fp += int((block == 0).sum())
        recall = tp / n_pos
        precision = tp / (tp + fp) if (tp + fp) > 0 else 1.0
        ap += (recall - prev_recall) * precision
        prev_recall = recall
        i = j + 1
    return float(ap)


# --------------------------------------------------------------------------- #
# Gates                                                                        #
# --------------------------------------------------------------------------- #
class CosineGate:
    """Fixed single-layer cosine-threshold gate (E9 / CAST baseline).

    Operates on ONE feature column (the cosine-style projection at a single
    layer). ``fit`` scans candidate thresholds (the midpoints between sorted
    feature values plus open ends) and keeps the one maximising balanced accuracy
    (equivalently Youden's J = TPR + TNR - 1, which is maximised at the same
    threshold). Fires when ``feature > threshold``.

    This is intentionally a one-parameter model: it is the fixed-threshold
    alternative the E15 logistic gate must beat under distribution shift.
    """

    def __init__(self) -> None:
        self.threshold: float = 0.0
        self.column: int = 0
        self._fitted: bool = False

    def _column_view(self, features: np.ndarray) -> np.ndarray:
        f = np.asarray(features, dtype=np.float64)
        if f.ndim == 1:
            return f
        return f[:, self.column]

    def fit(self, features: np.ndarray, labels: np.ndarray, column: int = 0) -> "CosineGate":
        """Pick the balanced-accuracy-optimal threshold on a single feature column.

        ``features`` may be 1-D (the column itself) or 2-D (``column`` selects it).
        """
        self.column = int(column)
        x = self._column_view(features)
        y = np.asarray(labels, dtype=np.float64).reshape(-1)
        if x.shape[0] != y.shape[0]:
            raise ValueError("features and labels must have equal length")
        n_pos = float((y == 1).sum())
        n_neg = float((y == 0).sum())
        if n_pos == 0 or n_neg == 0:
            # Degenerate: nothing to separate; threshold below all -> fire on all.
            self.threshold = float(x.min()) - 1.0
            self._fitted = True
            return self

        uniq = np.unique(x)
        # Candidate thresholds: midpoints + one below the min (fire on all)
        # + one above the max (fire on none).
        mids = (uniq[:-1] + uniq[1:]) / 2.0 if uniq.size > 1 else np.array([], dtype=np.float64)
        candidates = np.concatenate(([uniq[0] - 1.0], mids, [uniq[-1] + 1.0]))

        best_thr = float(candidates[0])
        best_bal = -1.0
        for thr in candidates:
            pred = x > thr
            tpr = float((pred & (y == 1)).sum()) / n_pos
            tnr = float((~pred & (y == 0)).sum()) / n_neg
            bal = 0.5 * (tpr + tnr)  # balanced accuracy; argmax == Youden's J argmax
            if bal > best_bal:
                best_bal = bal
                best_thr = float(thr)
        self.threshold = best_thr
        self._fitted = True
        return self

    def score(self, features: np.ndarray) -> np.ndarray:
        """Raw cosine feature (the single-layer projection), float32 ``[n]``."""
        return self._column_view(features).astype(np.float32)

    def predict(self, features: np.ndarray) -> np.ndarray:
        """Binary gate decision ``feature > threshold`` as int ``{0,1}`` ``[n]``."""
        return (self._column_view(features) > self.threshold).astype(np.int64)


class LogisticGate:
    """Learned multi-layer logistic gate, trained by gradient descent (E15 §5.2).

    A logistic regression ``sigmoid(W @ z + b)`` over the standardised k-layer
    feature vector ``z`` (per-layer cosine-style projections). Trained on binary
    cross-entropy with L2 weight decay via full-batch gradient descent. Features
    are standardised using TRAIN-set mean/std, and the SAME statistics are applied
    at score/predict time (no leakage). This is the convex, L2-regularised,
    overfit-resistant model the hypothesis specifies; the multi-layer weight
    vector is what lets it beat a single-layer cosine threshold when no single
    layer separates the classes but a linear combination does.
    """

    def __init__(self) -> None:
        self.W: Optional[np.ndarray] = None  # [k] float32
        self.b: float = 0.0
        self.mean_: Optional[np.ndarray] = None  # [k] train feature mean
        self.std_: Optional[np.ndarray] = None  # [k] train feature std
        self._fitted: bool = False

    def _standardize(self, features: np.ndarray) -> np.ndarray:
        f = np.asarray(features, dtype=np.float64)
        if f.ndim == 1:
            f = f.reshape(-1, 1)
        assert self.mean_ is not None and self.std_ is not None
        return (f - self.mean_) / self.std_

    def fit(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        l2: float = 0.1,
        epochs: int = 500,
        lr: float = 0.5,
        seed: int = 0,
    ) -> "LogisticGate":
        """Train the logistic gate with BCE-with-logits + L2 weight decay.

        Parameters
        ----------
        features : ``[n, k]`` multi-layer condition features (1-D treated as k=1).
        labels   : ``[n]`` in ``{0,1}``.
        l2       : L2 weight-decay coefficient (applied to W, NOT to the bias).
        epochs   : full-batch gradient-descent steps.
        lr       : learning rate.
        seed     : determinism seed for the (tiny) weight initialisation.

        Standardisation statistics are computed here and stored for inference.
        Deterministic: same (features, labels, hyperparams, seed) -> same fit.
        """
        f = np.asarray(features, dtype=np.float64)
        if f.ndim == 1:
            f = f.reshape(-1, 1)
        y_np = np.asarray(labels, dtype=np.float64).reshape(-1)
        if f.shape[0] != y_np.shape[0]:
            raise ValueError("features and labels must have equal length")

        # Train-set standardisation (guard zero-variance columns).
        self.mean_ = f.mean(axis=0)
        std = f.std(axis=0)
        std[std < _EPS] = 1.0
        self.std_ = std
        z_np = (f - self.mean_) / self.std_

        torch.manual_seed(int(seed))
        x = torch.tensor(z_np, dtype=torch.float32)
        y = torch.tensor(y_np, dtype=torch.float32)
        k = x.shape[1]
        # Small deterministic init so the loop starts near zero (convex anyway).
        w = (torch.randn(k, generator=torch.Generator().manual_seed(int(seed))) * 0.01)
        w = w.clone().requires_grad_(True)
        b = torch.zeros(1, requires_grad=True)

        opt = torch.optim.SGD([w, b], lr=float(lr))
        loss_fn = nn.BCEWithLogitsLoss()
        for _ in range(int(epochs)):
            opt.zero_grad()
            logits = x @ w + b
            loss = loss_fn(logits, y) + float(l2) * (w * w).sum()
            loss.backward()
            opt.step()

        self.W = w.detach().numpy().astype(np.float32)
        self.b = float(b.detach().item())
        self._fitted = True
        return self

    def score(self, features: np.ndarray) -> np.ndarray:
        """Sigmoid probability of the harmful class, float32 ``[n]``."""
        if self.W is None:
            raise RuntimeError("LogisticGate.score called before fit()")
        z = self._standardize(features)
        logits = z @ self.W.astype(np.float64) + self.b
        probs = 1.0 / (1.0 + np.exp(-logits))
        return probs.astype(np.float32)

    def predict(self, features: np.ndarray) -> np.ndarray:
        """Binary gate decision ``prob > 0.5`` as int ``{0,1}`` ``[n]``."""
        return (self.score(features) > 0.5).astype(np.int64)


# --------------------------------------------------------------------------- #
# Convenience evaluator (pure — unit-testable without a model)                 #
# --------------------------------------------------------------------------- #
def evaluate_gates(
    feats_indist: np.ndarray,
    labels_indist: np.ndarray,
    feats_ood: np.ndarray,
    labels_ood: np.ndarray,
    metric: str = "pr_auc",
    l2: float = 0.1,
    epochs: int = 500,
    lr: float = 0.5,
    seed: int = 0,
    falsifier_gap: float = 0.06,
) -> dict:
    """Train both gates on in-distribution features and report the E15 comparison.

    Both gates are FIT on (``feats_indist``, ``labels_indist``) and EVALUATED on
    both the in-distribution and the OOD sets. The cosine gate is the per-column
    best: every feature column is fit as a single-layer CosineGate and the column
    with the best in-distribution metric is selected (the "best fixed cosine
    threshold (E9 optimal)" the falsifier compares against). This function is
    pure — it takes already-extracted feature arrays — so the scientific
    comparison is unit-testable with no model.

    Returns a dict with per-condition AUCs for both gates, the OOD gap
    (``logistic_ood - best_cosine_ood``), and a verdict against ``falsifier_gap``
    (default 0.06, the §3 falsifier): "SUPPORTED" iff the learned gate beats the
    best fixed cosine threshold on OOD by at least the gap, else "FALSIFIED_OOD".
    """
    metric_fn = pr_auc if metric == "pr_auc" else roc_auc

    fi = np.asarray(feats_indist, dtype=np.float64)
    if fi.ndim == 1:
        fi = fi.reshape(-1, 1)
    fo = np.asarray(feats_ood, dtype=np.float64)
    if fo.ndim == 1:
        fo = fo.reshape(-1, 1)
    k = fi.shape[1]

    # Best single-layer cosine gate: fit each column, pick the best on in-dist.
    best_col = 0
    best_col_auc = -1.0
    cosine_gates: list[CosineGate] = []
    for col in range(k):
        g = CosineGate().fit(fi, labels_indist, column=col)
        auc_col = metric_fn(g.score(fi), np.asarray(labels_indist))
        cosine_gates.append(g)
        if auc_col > best_col_auc:
            best_col_auc = auc_col
            best_col = col
    cosine = cosine_gates[best_col]

    logistic = LogisticGate().fit(
        fi, labels_indist, l2=l2, epochs=epochs, lr=lr, seed=seed
    )

    auc_indist_cosine = metric_fn(cosine.score(fi), np.asarray(labels_indist))
    auc_indist_logistic = metric_fn(logistic.score(fi), np.asarray(labels_indist))
    auc_ood_cosine = metric_fn(cosine.score(fo), np.asarray(labels_ood))
    auc_ood_logistic = metric_fn(logistic.score(fo), np.asarray(labels_ood))
    gap = float(auc_ood_logistic - auc_ood_cosine)

    return {
        "metric": metric,
        "best_cosine_column": int(best_col),
        "auc_indist_cosine": float(auc_indist_cosine),
        "auc_indist_logistic": float(auc_indist_logistic),
        "auc_ood_cosine": float(auc_ood_cosine),
        "auc_ood_logistic": float(auc_ood_logistic),
        "auc_gap_ood": gap,
        "falsifier_gap": float(falsifier_gap),
        "verdict": "SUPPORTED" if gap >= falsifier_gap else "FALSIFIED_OOD",
    }
