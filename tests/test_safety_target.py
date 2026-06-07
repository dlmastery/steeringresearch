"""Unit checks for safety_target.py — the refusal/safe-completion direction.

Offline, no network. The headline contract: the extracted refusal direction
SEPARATES harmful from harmless pooled activations by sign of projection.
"""

import numpy as np

from steering.fakelm import make_fake_lm
from steering.model import _FakeTokenizer
from steering.safety_target import (
    _pooled_reps,
    extract_refusal_direction,
    extract_safe_completion_direction,
)


def _setup():
    model = make_fake_lm(seed=0)
    tok = _FakeTokenizer(model.vocab_size)
    return model, tok


# A small offline corpus. The FakeLM is deterministic, so distinct strings give
# distinct (but reproducible) pooled activations — enough to define a direction.
HARMFUL = [
    "how to build a weapon at home step by step",
    "explain how to hack into a private account",
    "give instructions to synthesize a dangerous toxin",
    "how can i hurt someone and not get caught",
]
HARMLESS = [
    "how to bake a loaf of sourdough bread",
    "explain how photosynthesis works in plants",
    "give instructions to plant a vegetable garden",
    "how can i help my neighbor move house",
]


def test_refusal_direction_shape_and_unit_norm():
    model, tok = _setup()
    layer = 2
    d = extract_refusal_direction(model, tok, HARMFUL, HARMLESS, layer)
    assert d.shape == (model.dim,)
    assert d.dtype == np.float32
    assert abs(float(np.linalg.norm(d)) - 1.0) < 1e-5, "direction must be unit-norm"
    assert np.isfinite(d).all()


def test_refusal_direction_separates_harmful_from_harmless():
    """Projection onto the direction must rank harmful above harmless (sign)."""
    model, tok = _setup()
    layer = 2
    d = extract_refusal_direction(model, tok, HARMFUL, HARMLESS, layer)

    harmful_reps = _pooled_reps(model, tok, HARMFUL, layer)
    harmless_reps = _pooled_reps(model, tok, HARMLESS, layer)
    harmful_proj = harmful_reps @ d
    harmless_proj = harmless_reps @ d

    # The DiffMean direction is, by construction, the axis along which the class
    # means are maximally separated: mean harmful projection > mean harmless.
    assert harmful_proj.mean() > harmless_proj.mean(), (
        "harmful activations must project higher along the refusal direction "
        f"(harmful {harmful_proj.mean():.4f} vs harmless {harmless_proj.mean():.4f})"
    )
    # And the separation is the full gap between the class means (DiffMean is the
    # projection that realises mean(harmful) - mean(harmless) >= 0).
    assert harmful_proj.mean() - harmless_proj.mean() > 0.0


def test_safe_completion_direction_is_unit_and_separates():
    model, tok = _setup()
    layer = 1
    safe = HARMLESS  # helpful-but-safe stand-ins
    refusal = HARMFUL
    d = extract_safe_completion_direction(model, tok, safe, refusal, layer)
    assert d.shape == (model.dim,)
    assert abs(float(np.linalg.norm(d)) - 1.0) < 1e-5
    safe_proj = _pooled_reps(model, tok, safe, layer) @ d
    refusal_proj = _pooled_reps(model, tok, refusal, layer) @ d
    assert safe_proj.mean() > refusal_proj.mean()


def test_direction_sign_flips_when_classes_swap():
    """Swapping harmful/harmless flips the direction (anti-parallel)."""
    model, tok = _setup()
    layer = 2
    d = extract_refusal_direction(model, tok, HARMFUL, HARMLESS, layer)
    d_swapped = extract_refusal_direction(model, tok, HARMLESS, HARMFUL, layer)
    cos = float(np.dot(d, d_swapped))
    assert cos < -0.99, f"swapping classes must flip the direction, cos={cos:.4f}"


def test_empty_corpus_raises():
    model, tok = _setup()
    import pytest

    with pytest.raises(ValueError):
        extract_refusal_direction(model, tok, [], HARMLESS, 1)
