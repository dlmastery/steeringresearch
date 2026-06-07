"""Unit checks for cast.py — THE conditional multi-intent safety steerer.

Offline via FakeResidualLM. The KEY test is the conditional identity:
  * when a registered intent's condition is PRESENT (a matching activation), the
    gate fires and a steer is applied (output differs from unsteered);
  * when the condition is ABSENT, NO steer fires and the output is IDENTICAL to
    the unsteered model.

We make the condition "present" by setting the intent's condition_vector to the
prompt's own pooled activation at layer_condition (cos == 1 > threshold), and
"absent" by using a direction orthogonal to it (cos ~= 0 < threshold).
"""

import numpy as np

from steering.cast import CASTSteerer, UnconditionalSteerer, _GateState
from steering.fakelm import make_fake_lm
from steering.hooks import ProbeHook, build_position_mask, probe_activations
from steering.model import _FakeTokenizer, encode_to_device, get_residual_layers

LAYER_C = 1
LAYER_W = 2
PROMPT = "tell me something interesting"


def _setup():
    model = make_fake_lm(seed=0)
    tok = _FakeTokenizer(model.vocab_size)
    return model, tok


def _pooled_condition(model, tok, prompt, layer):
    """The prompt's mean-pooled activation at ``layer`` (a perfectly-matching
    condition vector: cos with itself == 1)."""
    ids = encode_to_device(tok, prompt, model)
    acts = probe_activations(model, ids, [layer])
    return acts[layer][0].mean(dim=0).float().cpu().numpy().astype(np.float32)


def _orthogonal_to(u, seed=0):
    """A unit vector orthogonal to ``u`` (cos ~= 0)."""
    rng = np.random.default_rng(seed)
    uh = u / (np.linalg.norm(u) + 1e-8)
    r = rng.normal(size=u.shape).astype(np.float32)
    o = r - float(np.dot(r, uh)) * uh
    return (o / (np.linalg.norm(o) + 1e-8)).astype(np.float32)


def _safety_vec(dim, seed=1):
    rng = np.random.default_rng(seed)
    v = rng.normal(size=dim).astype(np.float32)
    return (v / np.linalg.norm(v)).astype(np.float32)


def _post_write_activation(steerer, prompt, alpha=12.0, operation="add"):
    """White-box: run the steerer's own hooks on the prompt and capture the
    POST-steer residual at the write layer (a ProbeHook registered after the
    write hook). Returns (steered_write_activation, fired_list)."""
    ids = encode_to_device(steerer.tokenizer, prompt, steerer.model)
    state = _GateState()
    prompt_mask = build_position_mask(ids, steerer.model.special_token_ids())
    read_hook, write_hook = steerer._make_hooks(state, prompt_mask, alpha, operation)
    layers = get_residual_layers(steerer.model)
    probe = ProbeHook()
    handles = [
        layers[steerer.layer_condition].register_forward_hook(read_hook),
        layers[steerer.layer_write].register_forward_hook(write_hook),
        layers[steerer.layer_write].register_forward_hook(probe),
    ]
    try:
        steerer.model(ids)
    finally:
        for hd in handles:
            hd.remove()
    return probe.activations, list(state.fired)


# --------------------------------------------------------------------------- #
# Construction guards                                                          #
# --------------------------------------------------------------------------- #
def test_condition_layer_must_precede_write_layer():
    model, tok = _setup()
    import pytest

    with pytest.raises(ValueError):
        CASTSteerer(model, tok, layer_condition=3, layer_write=1)


def test_duplicate_intent_name_raises():
    model, tok = _setup()
    s = CASTSteerer(model, tok, LAYER_C, LAYER_W)
    cv = np.ones(model.dim, dtype=np.float32)
    sv = _safety_vec(model.dim)
    s.add_intent("harm", cv, 0.5, sv)
    import pytest

    with pytest.raises(ValueError):
        s.add_intent("harm", cv, 0.5, sv)


# --------------------------------------------------------------------------- #
# THE KEY CONDITIONAL TEST                                                     #
# --------------------------------------------------------------------------- #
def test_gate_fires_when_condition_present():
    model, tok = _setup()
    cond = _pooled_condition(model, tok, PROMPT, LAYER_C)
    sv = _safety_vec(model.dim)

    s = CASTSteerer(model, tok, LAYER_C, LAYER_W)
    s.add_intent("harm", condition_vector=cond, threshold=0.5, safety_vector=sv)

    res = s.generate(PROMPT, alpha=12.0, max_new_tokens=6, operation="add")
    assert res["fired_intents"] == ["harm"], "matching condition must fire the gate"
    assert res["gate_scores"]["harm"] > 0.9, "cos with own pooled activation ~= 1"


def test_no_steer_when_condition_absent_is_identical_to_unsteered():
    """KEY: an unfired gate leaves generation BIT-IDENTICAL to the baseline."""
    model, tok = _setup()
    cond_present = _pooled_condition(model, tok, PROMPT, LAYER_C)
    cond_absent = _orthogonal_to(cond_present, seed=2)
    sv = _safety_vec(model.dim)

    # Baseline steerer with NO intents -> never writes -> plain greedy decode.
    baseline = CASTSteerer(model, tok, LAYER_C, LAYER_W)
    base_text = baseline.generate(PROMPT, alpha=12.0, max_new_tokens=6, operation="add")["text"]

    # Steerer WITH an intent whose condition is orthogonal -> gate must NOT fire.
    s_absent = CASTSteerer(model, tok, LAYER_C, LAYER_W)
    s_absent.add_intent("harm", condition_vector=cond_absent, threshold=0.5, safety_vector=sv)
    res_absent = s_absent.generate(PROMPT, alpha=12.0, max_new_tokens=6, operation="add")

    assert res_absent["fired_intents"] == [], "orthogonal condition must NOT fire"
    assert res_absent["gate_scores"]["harm"] < 0.5
    assert res_absent["text"] == base_text, (
        "an unfired gate must leave generation identical to the unsteered model"
    )


def test_fired_steer_changes_output():
    """A fired gate must actually change the generation vs the unsteered baseline."""
    model, tok = _setup()
    cond = _pooled_condition(model, tok, PROMPT, LAYER_C)
    sv = _safety_vec(model.dim)

    baseline = CASTSteerer(model, tok, LAYER_C, LAYER_W)
    base_text = baseline.generate(PROMPT, alpha=12.0, max_new_tokens=6, operation="add")["text"]

    s = CASTSteerer(model, tok, LAYER_C, LAYER_W)
    s.add_intent("harm", condition_vector=cond, threshold=0.5, safety_vector=sv)
    res = s.generate(PROMPT, alpha=12.0, max_new_tokens=6, operation="add")

    assert res["fired_intents"] == ["harm"]
    assert res["text"] != base_text, "a fired steer must change the generated tokens"


# --------------------------------------------------------------------------- #
# Multi-intent: several gates fire and their safety vectors compose            #
# --------------------------------------------------------------------------- #
def test_multiple_intents_fire_and_compose():
    model, tok = _setup()
    cond = _pooled_condition(model, tok, PROMPT, LAYER_C)
    sv1 = _safety_vec(model.dim, seed=1)
    sv2 = _safety_vec(model.dim, seed=2)

    # Two intents whose conditions both match (cos ~= 1) -> both fire.
    s2 = CASTSteerer(model, tok, LAYER_C, LAYER_W)
    s2.add_intent("a", condition_vector=cond, threshold=0.5, safety_vector=sv1)
    s2.add_intent("b", condition_vector=cond, threshold=0.5, safety_vector=sv2)
    res2 = s2.generate(PROMPT, alpha=12.0, max_new_tokens=6, operation="add")
    assert set(res2["fired_intents"]) == {"a", "b"}, "both matching intents must fire"

    # Composing two directions differs from steering with only one. Compared at
    # the residual level (the post-write activation): the large-alpha argmax can
    # saturate to the same token for nearby directions, but the written residual
    # itself must differ when a second safety vector is composed in.
    s1 = CASTSteerer(model, tok, LAYER_C, LAYER_W)
    s1.add_intent("a", condition_vector=cond, threshold=0.5, safety_vector=sv1)

    import torch

    act2, fired2 = _post_write_activation(s2, PROMPT)
    act1, fired1 = _post_write_activation(s1, PROMPT)
    assert set(fired2) == {"a", "b"} and fired1 == ["a"]
    assert not torch.allclose(act1, act2), (
        "composed multi-intent write must differ from the single-intent write"
    )


def test_partial_firing_only_matching_intents():
    """Of two intents, only the one whose condition matches fires."""
    model, tok = _setup()
    cond = _pooled_condition(model, tok, PROMPT, LAYER_C)
    cond_off = _orthogonal_to(cond, seed=7)
    sv = _safety_vec(model.dim)

    s = CASTSteerer(model, tok, LAYER_C, LAYER_W)
    s.add_intent("on", condition_vector=cond, threshold=0.5, safety_vector=sv)
    s.add_intent("off", condition_vector=cond_off, threshold=0.5, safety_vector=sv)
    res = s.generate(PROMPT, alpha=12.0, max_new_tokens=6, operation="add")
    assert res["fired_intents"] == ["on"]


# --------------------------------------------------------------------------- #
# UnconditionalSteerer ablation: ALWAYS fires regardless of the condition      #
# --------------------------------------------------------------------------- #
def test_unconditional_steerer_always_fires():
    model, tok = _setup()
    cond = _pooled_condition(model, tok, PROMPT, LAYER_C)
    cond_off = _orthogonal_to(cond, seed=3)  # would NOT fire under CAST
    sv = _safety_vec(model.dim)

    baseline = CASTSteerer(model, tok, LAYER_C, LAYER_W)
    base_text = baseline.generate(PROMPT, alpha=12.0, max_new_tokens=6, operation="add")["text"]

    u = UnconditionalSteerer(model, tok, LAYER_C, LAYER_W)
    u.add_intent("harm", condition_vector=cond_off, threshold=0.5, safety_vector=sv)
    res = u.generate(PROMPT, alpha=12.0, max_new_tokens=6, operation="add")

    # Despite the orthogonal (non-matching) condition, the unconditional steerer
    # fires anyway and changes the output -> the gate is what saves benign prompts.
    assert res["fired_intents"] == ["harm"]
    assert res["text"] != base_text


def test_unconditional_vs_conditional_on_absent_condition():
    """The ablation isolates the gate: same orthogonal condition, CAST stays a
    no-op while Unconditional steers."""
    model, tok = _setup()
    cond = _pooled_condition(model, tok, PROMPT, LAYER_C)
    cond_off = _orthogonal_to(cond, seed=5)
    sv = _safety_vec(model.dim)

    cast = CASTSteerer(model, tok, LAYER_C, LAYER_W)
    cast.add_intent("h", cond_off, 0.5, sv)
    uncond = UnconditionalSteerer(model, tok, LAYER_C, LAYER_W)
    uncond.add_intent("h", cond_off, 0.5, sv)

    r_cast = cast.generate(PROMPT, alpha=12.0, max_new_tokens=6, operation="add")
    r_uncond = uncond.generate(PROMPT, alpha=12.0, max_new_tokens=6, operation="add")

    assert r_cast["fired_intents"] == []
    assert r_uncond["fired_intents"] == ["h"]
    assert r_cast["text"] != r_uncond["text"]


# --------------------------------------------------------------------------- #
# Special-token protection in the write hook                                   #
# --------------------------------------------------------------------------- #
def test_write_protects_special_tokens():
    """The masked write must not steer the prompt's BOS (special) position.

    White-box: build the steerer's own (read, write) hooks, register them on the
    layers exactly as ``generate`` does, then register a ProbeHook AFTER the write
    hook on the write layer so it captures the POST-steer activation. The BOS at
    position 0 (a special token) must be untouched; a non-special position must
    change.
    """
    model, tok = _setup()
    cond = _pooled_condition(model, tok, PROMPT, LAYER_C)
    sv = _safety_vec(model.dim)
    ids = encode_to_device(tok, PROMPT, model)
    assert int(ids[0, 0]) == model.bos_token_id  # BOS sits at position 0

    base_w = probe_activations(model, ids, [LAYER_W])[LAYER_W]

    s = CASTSteerer(model, tok, LAYER_C, LAYER_W)
    s.add_intent("h", cond, 0.5, sv)

    state = _GateState()
    prompt_mask = build_position_mask(ids, model.special_token_ids())
    read_hook, write_hook = s._make_hooks(state, prompt_mask, alpha=12.0, operation="add")

    layers = get_residual_layers(model)
    probe = ProbeHook()
    handles = [
        layers[LAYER_C].register_forward_hook(read_hook),
        layers[LAYER_W].register_forward_hook(write_hook),
        layers[LAYER_W].register_forward_hook(probe),  # runs AFTER write -> post-steer
    ]
    try:
        model(ids)
    finally:
        for hd in handles:
            hd.remove()

    assert state.fired == ["h"], "matching condition must fire so the write executes"
    steered_w = probe.activations
    assert steered_w is not None
    # BOS (special, position 0) untouched; a non-special position changed.
    import torch

    assert torch.allclose(steered_w[0, 0], base_w[0, 0]), "BOS must not be steered"
    assert not torch.allclose(steered_w[0, 1], base_w[0, 1]), "non-special pos must steer"
