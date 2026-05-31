"""Rung-0 UNIT checks for hooks.py — all offline via FakeResidualLM."""

import torch

from steering.fakelm import make_fake_lm
from steering.hooks import (
    SteeringContext,
    apply_operation,
    build_position_mask,
    probe_activations,
)
from steering.model import get_residual_layers


def _make():
    model = make_fake_lm(seed=0)
    ids = torch.tensor([[1, 5, 6, 7, 8, 9]])  # 1 == BOS (special)
    return model, ids


def test_intervention_changes_hidden_state():
    model, ids = _make()
    layer = 2
    dim = model.dim
    v = torch.ones(dim)

    h_base = probe_activations(model, ids, [layer])[layer]
    with SteeringContext(model, v, [layer], operation="add", alpha=3.0):
        h_steer = probe_activations(model, ids, [layer])[layer]

    delta = (h_steer - h_base).norm()
    assert float(delta) > 0.0, "steering must change the residual stream (||Δh||>0)"


def test_logits_move_under_steering():
    model, ids = _make()
    layer = 1
    v = torch.ones(model.dim)

    with torch.no_grad():
        base_logits = model(ids).logits.clone()
    with SteeringContext(model, v, [layer], operation="add", alpha=5.0):
        with torch.no_grad():
            steer_logits = model(ids).logits.clone()

    assert float((steer_logits - base_logits).abs().max()) > 0.0, "logits must move"


def test_context_exit_restores_state_exactly():
    model, ids = _make()
    layer = 2
    v = torch.ones(model.dim)

    with torch.no_grad():
        before = model(ids).logits.clone()

    with SteeringContext(model, v, [layer], operation="add", alpha=4.0):
        pass  # enter + exit

    # No lingering hooks anywhere.
    for li in get_residual_layers(model):
        assert len(li._forward_hooks) == 0, "all hooks must be removed on exit"

    with torch.no_grad():
        after = model(ids).logits.clone()
    assert torch.allclose(before, after, atol=0.0), "state must restore EXACTLY"


def test_special_token_positions_unmodified():
    model, ids = _make()
    layer = 2
    v = torch.ones(model.dim)
    # BOS id == 1 sits at position 0; mask must exclude it.
    mask = build_position_mask(ids, model.special_token_ids())
    assert mask[0, 0].item() is False or not bool(mask[0, 0])

    h_base = probe_activations(model, ids, [layer])[layer]
    with SteeringContext(model, v, [layer], operation="add", alpha=6.0, position_mask=mask):
        h_steer = probe_activations(model, ids, [layer])[layer]

    # special position (0) unchanged; a non-special position changed.
    assert torch.allclose(h_steer[0, 0], h_base[0, 0]), "special token must be untouched"
    assert not torch.allclose(h_steer[0, 1], h_base[0, 1]), "non-special token must change"


def test_project_out_makes_projection_zero():
    torch.manual_seed(0)
    dim = 16
    h = torch.randn(3, 4, dim)
    v = torch.randn(dim)
    v_hat = v / v.norm()

    out = apply_operation(h, v, "project_out", alpha=1.0)
    proj = out @ v_hat  # [3, 4]
    assert float(proj.abs().max()) < 1e-5, "project_out must zero the v-component"


def test_rotate_preserves_norm():
    torch.manual_seed(1)
    dim = 16
    h = torch.randn(5, dim)
    v = torch.randn(dim)
    out = apply_operation(h, v, "rotate", alpha=0.5)
    assert torch.allclose(h.norm(dim=-1), out.norm(dim=-1), atol=1e-4), \
        "rotate must be norm-preserving"


def test_special_token_positions_unmodified_across_incremental_forwards():
    """Regression for the silent-mask-drop bug: a position mask built once from
    the prompt must keep protecting the prompt's special-token position across
    SUBSEQUENT forwards of DIFFERENT sequence lengths (the KV-cache decode case),
    instead of being silently dropped on a shape mismatch and steering ALL
    positions."""
    model = make_fake_lm(seed=0)
    layer = 2
    v = torch.ones(model.dim)
    prompt_ids = torch.tensor([[1, 5, 6, 7, 8, 9]])  # 1 == BOS (special) at pos 0
    mask = build_position_mask(prompt_ids, model.special_token_ids())

    # Baselines (captured OUTSIDE the steering context) for the prompt, a LONGER
    # continuation (prompt + new tokens), and a single-token decode step.
    longer_ids = torch.tensor([[1, 5, 6, 7, 8, 9, 10, 11]])
    step_ids = torch.tensor([[12]])
    base_prompt = probe_activations(model, prompt_ids, [layer])[layer]
    base_longer = probe_activations(model, longer_ids, [layer])[layer]
    base_step = probe_activations(model, step_ids, [layer])[layer]

    with SteeringContext(
        model, v, [layer], operation="add", alpha=6.0, position_mask=mask
    ):
        # 1) Same-length prompt forward: special pos untouched, others changed.
        steer_prompt = probe_activations(model, prompt_ids, [layer])[layer]
        # 2) Longer forward (mask rebuilt per forward): the prompt's BOS at pos 0
        #    is STILL protected; appended generated positions ARE steered.
        steer_longer = probe_activations(model, longer_ids, [layer])[layer]
        # 3) A bare incremental decode step (seq len 1, a brand-new token): it is
        #    not a prompt special token, so it is steerable — and must not crash.
        steer_step = probe_activations(model, step_ids, [layer])[layer]

    # Prompt BOS protected in BOTH the same-length and longer forwards.
    assert torch.allclose(steer_prompt[0, 0], base_prompt[0, 0]), \
        "BOS must be untouched on same-length forward"
    assert torch.allclose(steer_longer[0, 0], base_longer[0, 0]), \
        "BOS must STILL be untouched on a longer forward (mask rebuilt, not dropped)"
    # Non-special prompt position changed.
    assert not torch.allclose(steer_prompt[0, 1], base_prompt[0, 1]), \
        "non-special token must change"
    # Appended generated positions (index >= prompt len) ARE steered.
    assert not torch.allclose(steer_longer[0, 6], base_longer[0, 6]), \
        "generated positions beyond the prompt must be steered"
    # The single-token decode step is steered (it is a generated token).
    assert not torch.allclose(steer_step[0, 0], base_step[0, 0]), \
        "a fresh decode-step token must be steerable, not silently skipped"
