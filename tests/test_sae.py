"""Unit + scientific checks for sae.py (E20 SAE-TS vs DiffMean).

All offline, fast, hermetic — no network, no Gemma, no GemmaScope. We train a
small `SparseAutoencoder` on synthetic sparse-dictionary activations and then
exercise the SAE-TS steering-vector optimizer and the Gram-mass orthogonality
machinery that the E20 hypothesis turns on.
"""

import numpy as np
import torch

from steering.sae import (
    SparseAutoencoder,
    feature_activation,
    gram_mass,
    pairwise_cosines,
    sae_ts_vector,
    train_sae,
)


def _dictionary(dim=24, n_features=40, seed=0):
    """A random unit-norm overcomplete dictionary D : [n_features, dim]."""
    rng = np.random.default_rng(seed)
    d = rng.normal(size=(n_features, dim)).astype(np.float32)
    d /= np.linalg.norm(d, axis=1, keepdims=True) + 1e-8
    return d


def _sparse_activations(dictionary, n=600, k_active=3, seed=0):
    """Generate activations as SPARSE non-negative combos of dictionary atoms.

    Each sample activates exactly ``k_active`` random atoms with positive
    coefficients — the ground-truth generative model an SAE is meant to recover.
    """
    rng = np.random.default_rng(seed)
    n_features, dim = dictionary.shape
    acts = np.zeros((n, dim), dtype=np.float32)
    for i in range(n):
        atoms = rng.choice(n_features, size=k_active, replace=False)
        coeffs = rng.uniform(0.5, 2.0, size=k_active).astype(np.float32)
        acts[i] = (coeffs[:, None] * dictionary[atoms]).sum(axis=0)
    return acts


# --------------------------------------------------------------------------- #
# 1. The SAE LEARNS: reconstruction drops, codes are sparse.
# --------------------------------------------------------------------------- #
def test_sae_learns_and_is_sparse():
    dim, n_features = 24, 48
    dictionary = _dictionary(dim=dim, n_features=n_features, seed=1)
    acts = _sparse_activations(dictionary, n=600, k_active=3, seed=2)
    x = torch.as_tensor(acts, dtype=torch.float32)

    # Untrained baseline: reconstruction error AND code L1 (the sparsity proxy
    # the L1 penalty actually drives down).
    untrained = SparseAutoencoder(dim, n_features, seed=0)
    with torch.no_grad():
        base_err = torch.mean((x - untrained(x)) ** 2).item()
        base_l1 = float(torch.sum(torch.abs(untrained.encode(x)), dim=-1).mean())

    sae = train_sae(acts, n_features=n_features, l1=5e-3, epochs=400, lr=1e-2, seed=0)
    with torch.no_grad():
        z = sae.encode(x)
        trained_err = torch.mean((x - sae.decode(z)) ** 2).item()
        trained_l1 = float(torch.sum(torch.abs(z), dim=-1).mean())
        mean_l0 = float((z > 1e-6).float().sum(dim=-1).mean())

    # The SAE LEARNS: reconstruction collapses far below the untrained baseline.
    assert trained_err < 0.25 * base_err, (
        f"trained recon {trained_err:.4f} must be well below untrained {base_err:.4f}"
    )
    # Codes are SPARSE: the L1 penalty (what train_sae optimizes) more than halves
    # the mean code L1 vs the untrained SAE, and L0 stays below the dictionary size.
    assert trained_l1 < 0.5 * base_l1, (
        f"trained code L1 {trained_l1:.3f} must be << untrained {base_l1:.3f} (sparsity)"
    )
    assert mean_l0 < n_features, f"codes saturate the dictionary: L0={mean_l0:.1f}/{n_features}"


# --------------------------------------------------------------------------- #
# 2. THE KEY SCIENTIFIC TEST: SAE-TS reduces Gram mass vs feature-sharing raw
#    vectors (the E20 mechanism).
# --------------------------------------------------------------------------- #
def test_sae_ts_orthogonalizes_vs_shared_feature_raw_vectors():
    dim, n_features = 24, 48
    dictionary = _dictionary(dim=dim, n_features=n_features, seed=3)
    acts = _sparse_activations(dictionary, n=800, k_active=3, seed=4)
    sae = train_sae(acts, n_features=n_features, l1=1e-3, epochs=500, lr=1e-2, seed=0)

    # Two DISJOINT target-feature sets: "behavior A" and "behavior B".
    targets_a = [2, 5, 9]
    targets_b = [20, 27, 33]

    v_a = sae_ts_vector(sae, targets_a, lam=1.0, steps=400, lr=2e-2, seed=0)
    v_b = sae_ts_vector(sae, targets_b, lam=1.0, steps=400, lr=2e-2, seed=0)

    # (a) Each optimized vector activates ITS OWN targets more than the other's.
    fa = feature_activation(sae, v_a)
    fb = feature_activation(sae, v_b)
    assert fa[targets_a].mean() > fa[targets_b].mean(), (
        "v_a must drive its own targets above behavior-B's targets"
    )
    assert fb[targets_b].mean() > fb[targets_a].mean(), (
        "v_b must drive its own targets above behavior-A's targets"
    )

    # Build "raw" (DiffMean-style) vectors that SHARE features: each is a blend of
    # its own targets PLUS a common shared atom — the cross-feature contamination
    # SAE-TS is designed to remove. These mimic DiffMean vectors that overlap in
    # activation space.
    shared = [40, 41]
    raw_a = dictionary[targets_a + shared].sum(axis=0)
    raw_b = dictionary[targets_b + shared].sum(axis=0)

    cos_saets = float(np.abs(pairwise_cosines([v_a, v_b])[0]))
    cos_raw = float(np.abs(pairwise_cosines([raw_a, raw_b])[0]))

    # (b) THE CLAIM: SAE-TS vectors are MORE ORTHOGONAL (lower |cos|) than the
    # feature-sharing raw vectors.
    assert cos_saets < cos_raw, (
        f"SAE-TS |cos|={cos_saets:.4f} must be < raw |cos|={cos_raw:.4f} "
        "(E20: side-effect suppression increases orthogonality)"
    )
    # And the Gram mass of the SAE-TS stack is lower (single pair ⇒ M == |cos|).
    assert gram_mass([v_a, v_b]) < gram_mass([raw_a, raw_b]), (
        "SAE-TS 3-stack must have lower off-diagonal Gram mass than raw vectors"
    )


# --------------------------------------------------------------------------- #
# 3. gram_mass correctness on hand-constructed vectors.
# --------------------------------------------------------------------------- #
def test_gram_mass_orthogonal_identical_and_mixed():
    e0 = np.array([1.0, 0.0, 0.0])
    e1 = np.array([0.0, 1.0, 0.0])
    e2 = np.array([0.0, 0.0, 1.0])

    # Orthonormal set ⇒ every |cos| = 0 ⇒ M = 0.
    assert abs(gram_mass([e0, e1, e2]) - 0.0) < 1e-6

    # Identical (and anti-parallel) directions ⇒ every pair |cos| = 1 ⇒ n_pairs.
    assert abs(gram_mass([e0, e0, e0]) - 3.0) < 1e-6  # C(3,2) = 3 pairs
    assert abs(gram_mass([e0, -e0, e0]) - 3.0) < 1e-6  # |cos| = 1 each

    # Known mixed case: 45-degree pair ⇒ |cos| = 1/sqrt(2); orthogonal pairs ⇒ 0.
    a = np.array([1.0, 0.0])
    b = np.array([1.0, 1.0])  # cos(a, b) = 1/sqrt(2)
    expected = 1.0 / np.sqrt(2.0)
    assert abs(gram_mass([a, b]) - expected) < 1e-6


def test_pairwise_cosines_shape_and_values():
    e0 = np.array([1.0, 0.0])
    e1 = np.array([0.0, 1.0])
    a = np.array([1.0, 1.0])
    cos = pairwise_cosines([e0, e1, a])
    assert cos.shape == (3,)  # C(3,2) pairs
    # (e0,e1)=0, (e0,a)=1/sqrt2, (e1,a)=1/sqrt2
    assert abs(cos[0] - 0.0) < 1e-6
    assert abs(cos[1] - 1.0 / np.sqrt(2.0)) < 1e-6
    assert abs(cos[2] - 1.0 / np.sqrt(2.0)) < 1e-6
    # Single vector ⇒ no pairs.
    assert pairwise_cosines([e0]).shape == (0,)


# --------------------------------------------------------------------------- #
# 4. sae_ts_vector returns a finite unit-norm [dim] vector that increases the
#    target-feature score vs a random vector.
# --------------------------------------------------------------------------- #
def test_sae_ts_vector_is_finite_unit_norm_and_raises_target_score():
    dim, n_features = 20, 40
    dictionary = _dictionary(dim=dim, n_features=n_features, seed=5)
    acts = _sparse_activations(dictionary, n=500, k_active=3, seed=6)
    sae = train_sae(acts, n_features=n_features, l1=1e-3, epochs=400, lr=1e-2, seed=0)

    targets = [7, 12, 18]
    v = sae_ts_vector(sae, targets, lam=1.0, steps=400, lr=2e-2, seed=0)

    assert v.shape == (dim,)
    assert np.all(np.isfinite(v)), "optimized vector must be finite"
    assert abs(float(np.linalg.norm(v)) - 1.0) < 1e-4, "vector must be unit-norm"

    # Target-feature score must beat a matched random unit vector.
    rng = np.random.default_rng(0)
    rand = rng.normal(size=dim).astype(np.float32)
    rand /= np.linalg.norm(rand) + 1e-8
    opt_score = feature_activation(sae, v)[targets].mean()
    rand_score = feature_activation(sae, rand)[targets].mean()
    assert opt_score > rand_score, (
        f"SAE-TS target score {opt_score:.4f} must exceed random {rand_score:.4f}"
    )


# --------------------------------------------------------------------------- #
# 5. Determinism: same seed ⇒ same vector.
# --------------------------------------------------------------------------- #
def test_sae_ts_vector_is_deterministic():
    dim, n_features = 20, 40
    dictionary = _dictionary(dim=dim, n_features=n_features, seed=7)
    acts = _sparse_activations(dictionary, n=400, k_active=3, seed=8)
    sae = train_sae(acts, n_features=n_features, l1=1e-3, epochs=200, lr=1e-2, seed=0)

    targets = [3, 11, 19]
    v1 = sae_ts_vector(sae, targets, lam=1.0, steps=200, lr=2e-2, seed=42)
    v2 = sae_ts_vector(sae, targets, lam=1.0, steps=200, lr=2e-2, seed=42)
    assert np.allclose(v1, v2, atol=1e-7), "same seed must give the same vector"


def test_train_sae_is_deterministic():
    dim, n_features = 16, 32
    dictionary = _dictionary(dim=dim, n_features=n_features, seed=9)
    acts = _sparse_activations(dictionary, n=300, k_active=3, seed=10)
    s1 = train_sae(acts, n_features=n_features, epochs=100, lr=1e-2, seed=0)
    s2 = train_sae(acts, n_features=n_features, epochs=100, lr=1e-2, seed=0)
    x = torch.as_tensor(acts, dtype=torch.float32)
    with torch.no_grad():
        assert torch.allclose(s1(x), s2(x), atol=1e-6)
