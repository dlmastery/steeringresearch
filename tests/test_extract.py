"""Unit checks for extract.py on a synthetic separable set."""

import numpy as np

from steering.extract import (
    build_vector_bank,
    cosine,
    diffmean_vector,
    fisher_ratio,
    pca_top1_vector,
)


def _planted_set(dim=16, n=40, planted_axis=3, gap=4.0, noise=0.3, seed=0):
    """pos clusters at +gap along planted_axis, neg at -gap; isotropic noise."""
    rng = np.random.default_rng(seed)
    direction = np.zeros(dim)
    direction[planted_axis] = 1.0
    pos = rng.normal(0, noise, size=(n, dim)) + gap * direction
    neg = rng.normal(0, noise, size=(n, dim)) - gap * direction
    return pos, neg, direction


def test_diffmean_recovers_planted_direction():
    pos, neg, direction = _planted_set()
    dm = diffmean_vector(pos, neg)
    assert cosine(dm, direction) > 0.9, "DiffMean must recover the planted direction"


def test_pca_top1_aligns_with_diffmean():
    pos, neg, _ = _planted_set()
    dm = diffmean_vector(pos, neg)
    pca = pca_top1_vector(pos, neg)
    assert cosine(dm, pca) > 0.8, "PCA-top1 must align with DiffMean"


def test_fisher_peaks_at_planted_layer():
    # Build per-layer activations: only the "planted layer" is separable.
    dim = 16
    layers = {}
    planted_layer = 2
    for li in range(5):
        if li == planted_layer:
            pos, neg, _ = _planted_set(dim=dim, gap=5.0, noise=0.3, seed=li)
        else:
            # non-separable: pos and neg drawn from the SAME distribution
            rng = np.random.default_rng(100 + li)
            pos = rng.normal(0, 1.0, size=(40, dim))
            neg = rng.normal(0, 1.0, size=(40, dim))
        layers[li] = {"pos": pos, "neg": neg}

    fishers = {li: fisher_ratio(d["pos"], d["neg"]) for li, d in layers.items()}
    best = max(fishers, key=lambda li: fishers[li])
    assert best == planted_layer, f"Fisher should peak at planted layer, got {best}: {fishers}"


def test_build_vector_bank_shapes():
    dim = 16
    layers = {0: {"pos": np.random.randn(20, dim), "neg": np.random.randn(20, dim)}}
    bank = build_vector_bank(layers)
    assert bank[0]["diffmean"].shape == (dim,)
    assert bank[0]["pca"].shape == (dim,)
    assert "cosine_dm_pca" in bank[0] and "fisher" in bank[0]
