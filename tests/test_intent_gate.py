"""Unit checks for intent_gate.py — calibrated per-category intent detector.

Offline, deterministic. Headline contracts:
  * the gate FIRES on harmful features and ABSTAINS on benign ones (separable);
  * ``calibrate_thresholds`` hits ~the target false-positive rate on negatives;
  * ``expected_calibration_error`` is computed and small on a well-fit detector.
"""

import numpy as np

from steering.fakelm import make_fake_lm
from steering.intent_gate import IntentGate
from steering.model import _FakeTokenizer


def _two_class_separable(n=400, k=3, seed=0):
    """Two categories ("benign"=0, "harmful"=1) separable by a linear combo."""
    rng = np.random.default_rng(seed)
    half = n // 2
    w = np.array([1.2, -0.8, 0.6])[:k]
    X = rng.normal(0, 1.0, size=(n, k)).astype(np.float32)
    margin = X @ w
    # Push the two classes apart so each is cleanly separable.
    X[:half] += 1.5 * w  # harmful cluster
    X[half:] -= 1.5 * w  # benign cluster
    y = np.array(["harmful"] * half + ["benign"] * (n - half))
    perm = rng.permutation(n)
    return X[perm], y[perm], margin  # margin unused, kept for clarity


def test_gate_fires_on_harmful_not_benign():
    X, y, _ = _two_class_separable(n=400, seed=1)
    gate = IntentGate().fit(X, y, l2=0.0, epochs=600, lr=0.5, seed=0)
    assert set(gate.categories) == {"benign", "harmful"}

    proba = gate.predict_proba(X)
    h_col = gate.categories.index("harmful")
    harmful_scores = proba[y == "harmful", h_col]
    benign_scores = proba[y == "benign", h_col]
    # The harmful detector scores harmful prompts far above benign ones.
    assert harmful_scores.mean() > 0.8
    assert benign_scores.mean() < 0.2
    assert harmful_scores.mean() - benign_scores.mean() > 0.6


def test_predict_uses_calibrated_thresholds():
    X, y, _ = _two_class_separable(n=400, seed=2)
    gate = IntentGate().fit(X, y, l2=0.0, epochs=600, lr=0.5, seed=0)
    gate.calibrate_thresholds(X, y, target_fpr=0.05)
    pred = gate.predict(X)  # [n, K] {0,1}
    h_col = gate.categories.index("harmful")
    # Recall on harmful is high; benign rarely trips the harmful column.
    harmful_fired = pred[y == "harmful", h_col].mean()
    benign_fired = pred[y == "benign", h_col].mean()
    assert harmful_fired > 0.8
    assert benign_fired <= 0.10


def test_calibrate_thresholds_hits_target_fpr():
    X, y, _ = _two_class_separable(n=800, seed=3)
    gate = IntentGate().fit(X, y, l2=0.0, epochs=600, lr=0.5, seed=0)
    target = 0.05
    thr = gate.calibrate_thresholds(X, y, target_fpr=target)
    assert set(thr.keys()) == {"benign", "harmful"}

    proba = gate.predict_proba(X)
    for cat in gate.categories:
        col = gate.categories.index(cat)
        neg = proba[y != cat, col]
        empirical_fpr = float((neg > thr[cat]).mean())
        # The (1 - target) quantile gives an FPR at or just below the target.
        assert empirical_fpr <= target + 0.02, (
            f"category {cat}: empirical FPR {empirical_fpr:.3f} should be ~{target}"
        )


def test_lower_target_fpr_gives_higher_threshold():
    X, y, _ = _two_class_separable(n=800, seed=4)
    gate = IntentGate().fit(X, y, l2=0.0, epochs=600, lr=0.5, seed=0)
    loose = gate.calibrate_thresholds(X, y, target_fpr=0.20)["harmful"]
    strict = gate.calibrate_thresholds(X, y, target_fpr=0.01)["harmful"]
    assert strict >= loose, "a stricter FPR target must not lower the threshold"


def test_expected_calibration_error_is_computed_and_small():
    X, y, _ = _two_class_separable(n=600, seed=5)
    gate = IntentGate().fit(X, y, l2=0.1, epochs=600, lr=0.5, seed=0)
    ece = gate.expected_calibration_error(X, y, n_bins=10)
    assert 0.0 <= ece <= 1.0
    # A well-fit detector on separable data is reasonably calibrated.
    assert ece < 0.25, f"ECE should be small on a clean fit, got {ece}"


def test_predict_proba_shape_and_finiteness():
    X, y, _ = _two_class_separable(n=200, seed=6)
    gate = IntentGate().fit(X, y, l2=0.1, epochs=300, lr=0.5, seed=0)
    proba = gate.predict_proba(X)
    assert proba.shape == (200, 2)
    assert np.isfinite(proba).all()
    assert ((proba >= 0.0) & (proba <= 1.0)).all()


def test_predict_before_fit_raises():
    import pytest

    gate = IntentGate()
    with pytest.raises(RuntimeError):
        gate.predict_proba(np.zeros((2, 3), dtype=np.float32))


def test_extract_features_matches_gate_condition_features():
    """IntentGate.extract_features is a thin reuse of gate.condition_features."""
    from steering.gate import condition_features

    model = make_fake_lm(seed=0)
    tok = _FakeTokenizer(model.vocab_size)
    layers = [0, 1]
    rng = np.random.default_rng(0)
    cond = {li: rng.normal(size=model.dim).astype(np.float32) for li in layers}
    prompts = ["alpha", "beta gamma"]

    f1 = IntentGate.extract_features(model, tok, prompts, layers, cond)
    f2 = condition_features(model, tok, prompts, layers, cond)
    assert np.allclose(f1, f2)
    assert f1.shape == (2, 2)
