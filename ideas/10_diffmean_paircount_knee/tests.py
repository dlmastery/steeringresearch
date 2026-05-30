"""
Tests for H<NN> — <one-line idea title>

Correctness tests that must pass (VERIFY.md) before any experiment run.
These tests run in < 30 seconds on CPU; they do NOT load the full model.

Run with:
    pytest ideas/<NN>_<name>/tests.py -v

All tests use tiny mock tensors, not real model activations, to keep CI fast.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def tiny_hidden_size() -> int:
    """Small hidden size for unit tests (not 2304)."""
    return 64


@pytest.fixture()
def mock_vector(tiny_hidden_size: int) -> torch.Tensor:
    """A unit-normed mock steering vector."""
    v = torch.randn(tiny_hidden_size)
    return v / v.norm()


@pytest.fixture()
def mock_activation(tiny_hidden_size: int) -> torch.Tensor:
    """A mock activation tensor of shape (seq_len, hidden_size)."""
    return torch.randn(10, tiny_hidden_size)


# ---------------------------------------------------------------------------
# Plumbing tests (Rung 0 — UNIT)
# ---------------------------------------------------------------------------


class TestUnitPlumbing:
    """Rung 0 tests: verify that extract/apply/evaluate don't silently corrupt state."""

    def test_vector_is_unit_normed(self, mock_vector: torch.Tensor) -> None:
        """The steering vector must be unit-normed (alpha controls magnitude)."""
        norm = mock_vector.norm().item()
        assert abs(norm - 1.0) < 1e-5, f"Expected unit norm, got {norm:.6f}"

    def test_zero_alpha_is_identity(
        self,
        mock_vector: torch.Tensor,
        mock_activation: torch.Tensor,
    ) -> None:
        """At alpha=0, applying the steering vector must leave activations unchanged."""
        # Stub: once apply() is implemented, replace this with a real call.
        # For now, assert that zero-vector addition is identity.
        h = mock_activation.clone()
        delta = 0.0 * mock_vector  # alpha=0
        h_steered = h + delta.unsqueeze(0)  # broadcast over seq_len
        assert torch.allclose(h, h_steered), "Zero alpha must be identity"

    def test_additive_linearity(
        self,
        mock_vector: torch.Tensor,
        mock_activation: torch.Tensor,
    ) -> None:
        """Additive steering must scale linearly with alpha (baseline sanity)."""
        h = mock_activation.clone()
        alpha1, alpha2 = 0.5, 1.0
        delta1 = alpha1 * mock_vector
        delta2 = alpha2 * mock_vector
        ratio = delta2.norm() / delta1.norm()
        expected_ratio = alpha2 / alpha1
        assert abs(ratio.item() - expected_ratio) < 1e-4, (
            f"Expected linear scaling {expected_ratio:.4f}, got {ratio:.4f}"
        )

    def test_no_inplace_mutation_of_activation(
        self,
        mock_vector: torch.Tensor,
        mock_activation: torch.Tensor,
    ) -> None:
        """The apply() function must NOT mutate the original activation tensor."""
        h_original = mock_activation.clone()
        # Simulate what apply() should do: create a new tensor, not modify h.
        h_steered = mock_activation + 0.8 * mock_vector.unsqueeze(0)
        assert torch.allclose(mock_activation, h_original), (
            "apply() must not mutate the input activation tensor"
        )
        assert not torch.allclose(h_steered, h_original), (
            "steered activation should differ from original at alpha=0.8"
        )


# ---------------------------------------------------------------------------
# Idea-specific tests (replace stubs with real tests after implementation)
# ---------------------------------------------------------------------------


class TestIdeaSpecific:
    """Tests that verify the core claim of H<NN>. Replace stubs."""

    @pytest.mark.skip(reason="implement after implementation.py is written")
    def test_primary_metric_moves_in_expected_direction(self) -> None:
        """<primary metric> should increase / decrease vs baseline at alpha=0.8."""
        raise NotImplementedError

    @pytest.mark.skip(reason="implement after implementation.py is written")
    def test_falsifier_not_fired_on_mock_data(self) -> None:
        """On mock data, the falsifier threshold should not fire (basic sanity)."""
        raise NotImplementedError

    @pytest.mark.skip(reason="implement after implementation.py is written")
    def test_composite_formula_fingerprint(self) -> None:
        """Composite formula SHA-256 must match the registered fingerprint."""
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Data-split audit (no eval leakage)
# ---------------------------------------------------------------------------


class TestDataSplit:
    """Verify that extraction pairs are disjoint from the eval set."""

    def test_extraction_eval_disjoint(self) -> None:
        """Placeholder: once datasets are wired, assert disjoint prompt indices."""
        # Real implementation should call src/steering/extract.py:audit_or_die()
        # For now, assert the stub never accidentally uses the same split.
        extraction_indices = set(range(0, 50))   # first 50 pairs
        eval_indices = set(range(50, 150))         # next 100 for eval
        assert extraction_indices.isdisjoint(eval_indices), (
            "Extraction and eval splits must be disjoint"
        )
