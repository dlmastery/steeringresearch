"""Unit + scientific checks for hypersteer.py (E45 description->vector net).

All offline: the description encoder runs on FakeResidualLM; the hypernetwork
trains on synthetic data with a KNOWN smooth description->vector mapping so we
can assert generalization (held-out cosine high) against a shuffled-label
control (held-out cosine ~ 0). No network, fast, deterministic.
"""

import numpy as np
import torch

from steering.fakelm import make_fake_lm
from steering.hypersteer import (
    HyperNet,
    cosine,
    encode_descriptions,
    evaluate_holdout,
    predict_vector,
    train_hypernet,
    train_hypernet_with_history,
)
from steering.model import _FakeTokenizer


def _synthetic_mapping(
    n=80, embed_dim=24, vector_dim=16, seed=0
):
    """A KNOWN smooth (linear + mild nonlinear) map embeddings -> target vectors.

    targets = tanh(E @ W1) @ W2 + E @ W3  — a fixed, smooth, deterministic
    function of the embeddings. A network with enough capacity should recover it
    and GENERALIZE to held-out rows; a shuffled-label control cannot.
    """
    rng = np.random.default_rng(seed)
    emb = rng.normal(0, 1.0, size=(n, embed_dim)).astype(np.float32)
    w1 = rng.normal(0, 1.0 / np.sqrt(embed_dim), size=(embed_dim, 32)).astype(np.float32)
    w2 = rng.normal(0, 1.0 / np.sqrt(32), size=(32, vector_dim)).astype(np.float32)
    w3 = rng.normal(0, 1.0 / np.sqrt(embed_dim), size=(embed_dim, vector_dim)).astype(np.float32)
    targets = (np.tanh(emb @ w1) @ w2 + emb @ w3).astype(np.float32)
    return emb, targets


def _split(emb, targets, n_train):
    return emb[:n_train], targets[:n_train], emb[n_train:], targets[n_train:]


# --------------------------------------------------------------------------- #
# THE KEY SCIENTIFIC TEST: generalization vs shuffled-label control.
# --------------------------------------------------------------------------- #
def test_hypernet_generalizes_real_vs_shuffled_control():
    emb, targets = _synthetic_mapping(n=80, seed=0)
    tr_e, tr_t, ho_e, ho_t = _split(emb, targets, n_train=64)

    # Real: train on the true (embedding, target) pairs.
    net = train_hypernet(tr_e, tr_t, epochs=600, lr=1e-3, l2=1e-5, hidden=256, depth=2, seed=0)
    real = evaluate_holdout(net, ho_e, ho_t)["mean_cosine"]

    # Control: shuffle the TRAINING labels so the map is destroyed. The held-out
    # set (and its true targets) are untouched, so a memoriser scores ~0 there.
    rng = np.random.default_rng(123)
    perm = rng.permutation(tr_t.shape[0])
    net_shuf = train_hypernet(
        tr_e, tr_t[perm], epochs=600, lr=1e-3, l2=1e-5, hidden=256, depth=2, seed=0
    )
    shuffled = evaluate_holdout(net_shuf, ho_e, ho_t)["mean_cosine"]

    assert real > 0.8, f"held-out cosine should be high on the real mapping, got {real:.3f}"
    assert shuffled < 0.3, f"shuffled-label control should not generalize, got {shuffled:.3f}"
    assert real - shuffled > 0.5, (
        f"real must clearly beat the control: real={real:.3f} shuffled={shuffled:.3f}"
    )


def test_training_loss_decreases():
    emb, targets = _synthetic_mapping(n=48, seed=1)
    _, history = train_hypernet_with_history(
        emb, targets, epochs=300, lr=1e-3, l2=1e-5, seed=0
    )
    assert len(history) == 300
    # Optimization actually happens: final loss is well below the initial loss.
    assert history[-1] < history[0], "loss must decrease over training"
    assert history[-1] < 0.5 * history[0], (
        f"loss should at least halve: start={history[0]:.4f} end={history[-1]:.4f}"
    )


def test_determinism_same_seed_same_weights():
    emb, targets = _synthetic_mapping(n=40, seed=2)
    net_a = train_hypernet(emb, targets, epochs=50, seed=7)
    net_b = train_hypernet(emb, targets, epochs=50, seed=7)
    for pa, pb in zip(net_a.parameters(), net_b.parameters()):
        assert torch.equal(pa, pb), "same seed must reproduce identical weights"


def test_higher_l2_shrinks_weight_norm():
    emb, targets = _synthetic_mapping(n=40, seed=3)
    net_lo = train_hypernet(emb, targets, epochs=400, l2=1e-5, seed=0)
    net_hi = train_hypernet(emb, targets, epochs=400, l2=1e-1, seed=0)

    def _wnorm(net):
        return float(
            sum((p.detach() ** 2).sum() for p in net.parameters()).sqrt()
        )

    assert _wnorm(net_hi) < _wnorm(net_lo), (
        f"higher L2 must shrink weight norm: lo={_wnorm(net_lo):.3f} hi={_wnorm(net_hi):.3f}"
    )


def test_encode_descriptions_end_to_end_on_fakelm():
    model = make_fake_lm(seed=0)
    tok = _FakeTokenizer(model.vocab_size)
    descs = [
        "make the text more formal and professional",
        "express strong anger and frustration",
        "describe the ocean and the deep sea",
    ]
    layer = 2
    emb = encode_descriptions(model, tok, descs, layer)
    assert emb.shape == (3, model.dim), f"expected [3, {model.dim}], got {emb.shape}"
    assert np.isfinite(emb).all(), "embeddings must be finite"
    assert emb.dtype == np.float32


def test_predict_vector_shape_and_smoothness():
    emb, targets = _synthetic_mapping(n=40, embed_dim=24, vector_dim=16, seed=4)
    net = train_hypernet(emb, targets, epochs=200, seed=0)

    pred = predict_vector(net, emb[0])
    assert pred.shape == (16,), f"predicted vector should be [16], got {pred.shape}"
    assert np.isfinite(pred).all()

    # Smoothness: a near-duplicate description embedding -> near-identical vector.
    near = emb[0] + np.random.default_rng(9).normal(0, 1e-4, size=emb[0].shape).astype(np.float32)
    pred_near = predict_vector(net, near)
    assert cosine(pred, pred_near) > 0.999, (
        f"near-duplicate input should map to near-identical output, "
        f"cos={cosine(pred, pred_near):.5f}"
    )


def test_hypernet_forward_shape():
    net = HyperNet(embed_dim=24, vector_dim=16, hidden=128, depth=3)
    out = net(torch.zeros(5, 24))
    assert out.shape == (5, 16)


def test_evaluate_holdout_keys():
    emb, targets = _synthetic_mapping(n=30, seed=5)
    net = train_hypernet(emb[:24], targets[:24], epochs=100, seed=0)
    res = evaluate_holdout(net, emb[24:], targets[24:])
    assert set(res.keys()) == {"per_behavior", "mean_cosine"}
    assert len(res["per_behavior"]) == 6
    assert isinstance(res["mean_cosine"], float)
