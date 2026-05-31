"""Unit checks for geometry.py."""

import torch

from steering.geometry import (
    effective_rank,
    norm_budget,
    offshell_displacement,
    participation_ratio,
)


def _rank_k_batch(k=3, dim=16, n=200, seed=0):
    """A batch whose activations span exactly k orthogonal equal-energy axes."""
    torch.manual_seed(seed)
    basis = torch.eye(dim)[:k]  # k orthonormal directions
    coeffs = torch.randn(n, k)  # equal-variance coefficients
    return coeffs @ basis  # [n, dim], rank k


def test_effective_rank_of_rank_k_batch():
    for k in (2, 3, 5):
        x = _rank_k_batch(k=k)
        er = effective_rank(x)
        assert abs(er - k) < 0.6, f"effective rank ~{k}, got {er}"


def test_participation_ratio_of_rank_k_batch():
    for k in (2, 4):
        x = _rank_k_batch(k=k)
        pr = participation_ratio(x)
        assert abs(pr - k) < 0.6, f"participation ratio ~{k}, got {pr}"


def test_norm_budget_accumulates():
    torch.manual_seed(0)
    dim = 16
    h_base = torch.randn(10, dim)
    base_norm = h_base.norm(dim=-1).mean()
    # single delta of known norm
    delta = torch.zeros(10, dim)
    delta[:, 0] = 1.0  # each row delta has norm 1
    nb = norm_budget(delta, h_base)
    expected = 1.0 / float(base_norm)
    assert abs(nb - expected) < 1e-4, f"norm budget {nb} vs expected {expected}"

    # multi-step accumulation: 3 identical steps -> 3x the single-step budget
    steps = torch.stack([delta, delta, delta], dim=0)  # [3, 10, dim]
    nb3 = norm_budget(steps, h_base)
    assert abs(nb3 - 3 * expected) < 1e-4, f"3-step budget {nb3} vs {3*expected}"


def test_offshell_displacement_zero_when_unchanged():
    torch.manual_seed(0)
    h = torch.randn(8, 16)
    assert offshell_displacement(h, h) == 0.0
    # scaling up the norm raises displacement
    assert offshell_displacement(h, 2.0 * h) > 0.5


def test_angular_displacement_orthogonal_and_parallel():
    import torch
    from steering.geometry import angular_displacement
    h = torch.randn(4, 8)
    # identical -> 0
    assert angular_displacement(h, h) < 1e-5
    # orthogonal -> ~1
    a = torch.zeros(1, 3); a[0, 0] = 1.0
    b = torch.zeros(1, 3); b[0, 1] = 1.0
    assert abs(angular_displacement(a, b) - 1.0) < 1e-5
    # anti-parallel -> ~2
    assert abs(angular_displacement(a, -a) - 2.0) < 1e-5
