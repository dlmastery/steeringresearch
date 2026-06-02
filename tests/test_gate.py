"""Unit checks for gate.py — the E15 learned-gate-vs-fixed-threshold infra.

All offline, no network, fast. The headline scientific test is
``test_logistic_beats_cosine_when_no_single_layer_separates``: the multi-layer
logistic gate must beat the best single-layer cosine gate on a dataset where NO
single feature separates the classes but a linear combination does — the E15
mechanism in miniature.
"""

import json
from pathlib import Path

import numpy as np

from steering.fakelm import make_fake_lm
from steering.gate import (
    CosineGate,
    LogisticGate,
    condition_features,
    evaluate_gates,
    pr_auc,
    roc_auc,
)
from steering.model import _FakeTokenizer


# --------------------------------------------------------------------------- #
# Metrics: hand-computed values on tiny labelled arrays                        #
# --------------------------------------------------------------------------- #
def test_auc_perfect_separation():
    # Positives all score above negatives -> ranking is perfect.
    scores = np.array([0.1, 0.2, 0.8, 0.9])
    labels = np.array([0, 0, 1, 1])
    assert abs(roc_auc(scores, labels) - 1.0) < 1e-9
    assert abs(pr_auc(scores, labels) - 1.0) < 1e-9


def test_auc_inverted_separation():
    # Perfectly WRONG ranking -> ROC-AUC 0.0.
    scores = np.array([0.9, 0.8, 0.2, 0.1])
    labels = np.array([0, 0, 1, 1])
    assert abs(roc_auc(scores, labels) - 0.0) < 1e-9


def test_roc_auc_constant_scores_is_half():
    # All-equal scores -> every pair is a tie -> ROC-AUC exactly 0.5.
    scores = np.array([0.5, 0.5, 0.5, 0.5])
    labels = np.array([0, 1, 0, 1])
    assert abs(roc_auc(scores, labels) - 0.5) < 1e-9


def test_roc_auc_random_is_near_half():
    rng = np.random.default_rng(0)
    scores = rng.normal(size=2000)
    labels = rng.integers(0, 2, size=2000)
    assert abs(roc_auc(scores, labels) - 0.5) < 0.05


def test_roc_auc_known_mixed_case():
    # One positive ranked below one negative out of a 2x2 -> AUC = 0.75.
    # scores: neg=0.1, pos=0.2, neg=0.3, pos=0.4  -> the neg at 0.3 beats pos 0.2.
    scores = np.array([0.1, 0.2, 0.3, 0.4])
    labels = np.array([0, 1, 0, 1])
    # Mann-Whitney: pairs (pos,neg): (0.2 vs 0.1)=win, (0.2 vs 0.3)=loss,
    # (0.4 vs 0.1)=win, (0.4 vs 0.3)=win -> 3/4 = 0.75.
    assert abs(roc_auc(scores, labels) - 0.75) < 1e-9


def test_pr_auc_known_value():
    # Descending order of scores: labels seen as [1, 0, 1, 1].
    # AP = sum over each positive of precision-at-that-positive.
    #   pos1 at rank1: precision 1/1 = 1.0
    #   pos2 at rank3: precision 2/3
    #   pos3 at rank4: precision 3/4
    # AP = (1.0 + 2/3 + 3/4) / 3 ... using the (recall step * precision) form:
    #   contributions: (1/3)*1.0 + 0 + (1/3)*(2/3) + (1/3)*(3/4)
    scores = np.array([0.9, 0.8, 0.7, 0.6])
    labels = np.array([1, 0, 1, 1])
    expected = (1.0 / 3) * 1.0 + (1.0 / 3) * (2.0 / 3) + (1.0 / 3) * (3.0 / 4)
    assert abs(pr_auc(scores, labels) - expected) < 1e-9


def test_pr_auc_no_positives_is_zero():
    scores = np.array([0.1, 0.2, 0.3])
    labels = np.array([0, 0, 0])
    assert pr_auc(scores, labels) == 0.0


# --------------------------------------------------------------------------- #
# CosineGate: fits a sensible threshold on a single separable column           #
# --------------------------------------------------------------------------- #
def test_cosine_gate_single_column_separation():
    rng = np.random.default_rng(1)
    pos = rng.normal(2.0, 0.3, size=50)
    neg = rng.normal(-2.0, 0.3, size=50)
    x = np.concatenate([neg, pos])
    y = np.concatenate([np.zeros(50), np.ones(50)])
    gate = CosineGate().fit(x, y)
    pred = gate.predict(x)
    acc = float((pred == y).mean())
    assert acc > 0.95, f"cosine gate should separate a clean 1-D set, acc={acc}"
    # threshold sits between the two clusters
    assert -2.0 < gate.threshold < 2.0


# --------------------------------------------------------------------------- #
# LogisticGate LEARNS: high train AUC on a linearly-separable multi-feature set #
# --------------------------------------------------------------------------- #
def _linsep_multifeature(n=200, k=4, seed=0):
    """A k-feature set separable by a known linear combination of all features."""
    rng = np.random.default_rng(seed)
    w_true = np.array([1.0, -1.0, 0.5, 0.8])[:k]
    X = rng.normal(0, 1.0, size=(n, k))
    margin = X @ w_true
    y = (margin > 0).astype(np.float64)
    return X, y


def test_logistic_gate_learns_linearly_separable():
    X, y = _linsep_multifeature()
    gate = LogisticGate().fit(X, y, l2=0.0, epochs=800, lr=0.5, seed=0)
    auc = roc_auc(gate.score(X), y)
    assert auc > 0.95, f"logistic gate should learn a linearly-separable set, AUC={auc}"


# --------------------------------------------------------------------------- #
# THE KEY E15 TEST: multi-layer logistic beats single-layer cosine when NO     #
# single feature separates but a linear combination does.                      #
# --------------------------------------------------------------------------- #
def _xor_like_no_single_feature_separates(n=400, seed=0):
    """Two features whose SUM cleanly separates but neither does so ALONE.

    Construction: the informative direction is s = a + b, drawn well-separated by
    class (s ~ +M for harmful, -M for benign); the orthogonal direction o = a - b
    carries a LARGE class-independent nuisance variance. We set a = (s + o)/2,
    b = (s - o)/2. Because o's spread dwarfs the per-axis share of M, each single
    column a or b is only weakly predictive (the nuisance swamps the signal on
    that axis), yet the logistic gate recovers s = a + b and separates the classes
    almost perfectly. A single-column cosine threshold can only use ONE column.
    Two extra pure-noise columns force layer-selection to find the right pair.
    """
    rng = np.random.default_rng(seed)
    half = n // 2
    margin = 2.0  # class separation along the s = a+b direction
    nuisance = 6.0  # large class-independent spread along o = a-b
    s = np.concatenate([rng.normal(margin, 1.0, half), rng.normal(-margin, 1.0, n - half)])
    o = rng.normal(0.0, nuisance, size=n)
    y = np.concatenate([np.ones(half), np.zeros(n - half)])
    a = (s + o) / 2.0
    b = (s - o) / 2.0
    c = rng.normal(0, 1.0, size=n)  # pure noise
    d = rng.normal(0, 1.0, size=n)  # pure noise
    X = np.stack([a, b, c, d], axis=1)
    # shuffle so class order is not positional
    perm = rng.permutation(n)
    return X[perm], y[perm]


def test_logistic_beats_cosine_when_no_single_layer_separates():
    X, y = _xor_like_no_single_feature_separates(n=600, seed=3)

    # Best single-layer cosine gate: best AUC over all columns.
    best_cos_auc = -1.0
    for col in range(X.shape[1]):
        g = CosineGate().fit(X, y, column=col)
        best_cos_auc = max(best_cos_auc, roc_auc(g.score(X), y))

    log_gate = LogisticGate().fit(X, y, l2=0.0, epochs=1000, lr=0.5, seed=0)
    log_auc = roc_auc(log_gate.score(X), y)

    # No single feature separates: best cosine should be clearly sub-perfect.
    assert best_cos_auc < 0.80, f"a single feature should NOT separate, got {best_cos_auc}"
    # The multi-layer logistic gate exploits the combination -> much higher AUC.
    assert log_auc > 0.90, f"logistic should exploit the combination, got {log_auc}"
    assert log_auc - best_cos_auc > 0.10, (
        f"E15 mechanism: multi-layer must beat single-layer "
        f"(logistic {log_auc:.3f} vs best cosine {best_cos_auc:.3f})"
    )


def test_evaluate_gates_reports_e15_comparison():
    # In-dist and OOD both follow the same "needs both features" structure, so the
    # logistic gate should win on OOD -> SUPPORTED verdict + populated fields.
    Xi, yi = _xor_like_no_single_feature_separates(n=600, seed=1)
    Xo, yo = _xor_like_no_single_feature_separates(n=400, seed=2)
    res = evaluate_gates(Xi, yi, Xo, yo, metric="roc_auc", l2=0.0, epochs=1000, lr=0.5)
    for key in (
        "auc_indist_cosine",
        "auc_indist_logistic",
        "auc_ood_cosine",
        "auc_ood_logistic",
        "auc_gap_ood",
        "verdict",
    ):
        assert key in res
    assert res["auc_ood_logistic"] > res["auc_ood_cosine"]
    assert res["auc_gap_ood"] >= 0.06
    assert res["verdict"] == "SUPPORTED"


# --------------------------------------------------------------------------- #
# L2 regularization shrinks the weight norm                                    #
# --------------------------------------------------------------------------- #
def test_l2_shrinks_weights():
    X, y = _linsep_multifeature(n=200, k=4, seed=5)
    # Modest lr keeps the strong-L2 fit numerically stable (the L2 gradient is
    # 2*l2*W; lr*2*l2 must stay < 1 for the decay step to contract rather than
    # diverge). Both fits share lr/epochs/seed so only l2 differs.
    low = LogisticGate().fit(X, y, l2=0.0, epochs=600, lr=0.1, seed=0)
    high = LogisticGate().fit(X, y, l2=1.0, epochs=600, lr=0.1, seed=0)
    assert low.W is not None and high.W is not None
    norm_low = float(np.linalg.norm(low.W))
    norm_high = float(np.linalg.norm(high.W))
    assert norm_high < norm_low, (
        f"higher L2 must shrink ||W||: l2=5 -> {norm_high:.4f}, l2=0 -> {norm_low:.4f}"
    )


# --------------------------------------------------------------------------- #
# Determinism: same seed -> identical fit                                      #
# --------------------------------------------------------------------------- #
def test_logistic_fit_is_deterministic():
    X, y = _linsep_multifeature(n=150, k=4, seed=7)
    g1 = LogisticGate().fit(X, y, l2=0.1, epochs=300, lr=0.3, seed=42)
    g2 = LogisticGate().fit(X, y, l2=0.1, epochs=300, lr=0.3, seed=42)
    assert g1.W is not None and g2.W is not None
    assert np.allclose(g1.W, g2.W)
    assert abs(g1.b - g2.b) < 1e-9


# --------------------------------------------------------------------------- #
# condition_features end-to-end on FakeResidualLM                              #
# --------------------------------------------------------------------------- #
def test_condition_features_shape_and_orientation():
    model = make_fake_lm(seed=0)
    tok = _FakeTokenizer(model.vocab_size)
    layers = [0, 1, 2]
    dim = model.dim
    # Unit condition vectors per layer (arbitrary but fixed).
    rng = np.random.default_rng(0)
    cond = {li: rng.normal(size=dim).astype(np.float32) for li in layers}

    prompts = ["alpha prompt", "beta prompt longer", "g"]
    feats = condition_features(model, tok, prompts, layers, cond)

    assert feats.shape == (len(prompts), len(layers))
    assert np.isfinite(feats).all()

    # Orientation: the feature equals dot(mean-pooled activation, unit condition).
    # Recompute layer-0 feature for the first prompt directly and compare.
    from steering.hooks import probe_activations
    from steering.model import encode_to_device

    ids = encode_to_device(tok, prompts[0], model)
    acts = probe_activations(model, ids, [0])
    pooled = acts[0][0].mean(dim=0).float().cpu().numpy()
    v = cond[0] / (np.linalg.norm(cond[0]) + 1e-8)
    expected = float(np.dot(pooled, v))
    assert abs(float(feats[0, 0]) - expected) < 1e-4


def test_condition_features_missing_vector_raises():
    model = make_fake_lm(seed=0)
    tok = _FakeTokenizer(model.vocab_size)
    cond = {0: np.ones(model.dim, dtype=np.float32)}  # layer 1 missing
    import pytest

    with pytest.raises(KeyError):
        condition_features(model, tok, ["x"], [0, 1], cond)


# --------------------------------------------------------------------------- #
# The OOD json loads and parses                                               #
# --------------------------------------------------------------------------- #
def test_ood_json_loads_and_parses():
    path = Path("src/steering/data/ood_harmful_mini.json")
    data = json.loads(path.read_text(encoding="utf-8"))
    assert "harmful" in data and "benign" in data
    assert len(data["harmful"]) >= 12
    assert len(data["benign"]) >= 12
    assert all(isinstance(p, str) and p for p in data["harmful"])
    assert all(isinstance(p, str) and p for p in data["benign"])
