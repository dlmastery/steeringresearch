"""Unit checks for multi_intent.py — gram_schmidt / compose / interference mass.

Offline, pure numpy, deterministic. Headline contracts: Gram-Schmidt produces a
mutually-orthogonal basis (pairwise cos ~0), and the interference Gram mass is 0
for an orthonormal set and ``sqrt(2)`` for a duplicated direction.
"""

import numpy as np

from steering.multi_intent import compose, gram_schmidt, interference_gram_mass


def _cos(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def test_gram_schmidt_orthonormalizes():
    rng = np.random.default_rng(0)
    vecs = [rng.normal(size=8).astype(np.float32) for _ in range(4)]
    basis = gram_schmidt(vecs)
    assert len(basis) == 4
    for b in basis:
        assert abs(float(np.linalg.norm(b)) - 1.0) < 1e-5, "each basis vector unit-norm"
    # pairwise orthogonality
    for i in range(len(basis)):
        for j in range(i + 1, len(basis)):
            assert abs(_cos(basis[i], basis[j])) < 1e-5, (
                f"basis[{i}] and basis[{j}] must be orthogonal"
            )


def test_gram_schmidt_drops_dependent_vectors():
    v = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    vecs = [v, 2.0 * v, np.array([0.0, 1.0, 0.0], dtype=np.float32)]
    basis = gram_schmidt(vecs)
    # The second vector is collinear with the first -> dropped.
    assert len(basis) == 2
    assert abs(_cos(basis[0], basis[1])) < 1e-5


def test_gram_schmidt_preserves_first_direction():
    a = np.array([3.0, 4.0, 0.0], dtype=np.float32)
    b = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    basis = gram_schmidt([a, b])
    # First vector keeps its direction (just normalised).
    assert _cos(basis[0], a) > 0.999


def test_compose_is_weighted_sum():
    a = np.array([1.0, 0.0], dtype=np.float32)
    b = np.array([0.0, 1.0], dtype=np.float32)
    out = compose([a, b], [2.0, 3.0])
    assert np.allclose(out, np.array([2.0, 3.0], dtype=np.float32))


def test_compose_equal_weights_sums_directions():
    a = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    b = np.array([4.0, 5.0, 6.0], dtype=np.float32)
    out = compose([a, b], [1.0, 1.0])
    assert np.allclose(out, a + b)


def test_compose_alpha_mismatch_raises():
    import pytest

    with pytest.raises(ValueError):
        compose([np.ones(3, dtype=np.float32)], [1.0, 2.0])


def test_interference_mass_zero_for_orthonormal():
    e1 = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    e2 = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    e3 = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    mass = interference_gram_mass([e1, e2, e3])
    assert abs(mass) < 1e-6, f"orthonormal set must have zero interference, got {mass}"


def test_interference_mass_duplicate_is_sqrt2():
    v = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    # two identical (after unit-norm) directions: off-diagonal entries are 1, 1
    # -> Frobenius mass sqrt(1^2 + 1^2) = sqrt(2).
    mass = interference_gram_mass([v, v.copy()])
    assert abs(mass - np.sqrt(2.0)) < 1e-5, f"expected sqrt(2), got {mass}"


def test_interference_mass_single_vector_is_zero():
    mass = interference_gram_mass([np.array([1.0, 2.0, 3.0], dtype=np.float32)])
    assert mass == 0.0


def test_interference_mass_matches_pairwise_cosine_formula():
    rng = np.random.default_rng(3)
    vecs = [rng.normal(size=5).astype(np.float32) for _ in range(3)]
    mass = interference_gram_mass(vecs)
    # sqrt(2 * sum_{i<j} cos^2)
    units = [v / np.linalg.norm(v) for v in vecs]
    s = 0.0
    for i in range(3):
        for j in range(i + 1, 3):
            s += _cos(units[i], units[j]) ** 2
    expected = np.sqrt(2.0 * s)
    assert abs(mass - expected) < 1e-5
