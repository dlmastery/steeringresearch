"""Unit checks for baselines.py — all offline via FakeResidualLM.

These exercise the uniform Baseline interface, prompt construction, the steering
gate ablation, the router, and the registry/builder. No real GPU/models.
"""

import torch

from steering.baselines import (
    ALL_BASELINES,
    DEFAULT_REFUSAL,
    FEWSHOT_MARKER,
    SAFETY_SYSTEM_MARKER,
    Baseline,
    CASTBaseline,
    FewShotPrompting,
    NoSteer,
    RefusalDirectionSteer,
    SecondModelRouter,
    SystemPromptRefusal,
    UnconditionalSteer,
    build_baselines,
)
from steering.eval import is_refusal
from steering.fakelm import make_fake_lm
from steering.model import _FakeTokenizer

PROMPT = "Tell me about the weather today"


def _shared():
    """A full shared-kwargs bag so build_baselines yields every baseline."""
    dim = 16
    return {
        "safety_vector": torch.ones(dim),
        "refusal_dir": torch.ones(dim),
        "layer": 1,
        "alpha": 8.0,
        "classifier_fn": lambda p: "bomb" in p.lower(),
    }


def _model_tok():
    model = make_fake_lm(seed=0)  # dim=16 by default
    tok = _FakeTokenizer(model.vocab_size)
    return model, tok


def test_every_baseline_responds_without_error():
    model, tok = _model_tok()
    baselines = build_baselines(**_shared())
    assert baselines, "build_baselines must yield at least one baseline"
    for b in baselines:
        out = b.respond(model, tok, PROMPT, max_new_tokens=8)
        assert isinstance(out, str)
        assert len(out) > 0, f"{b.name} returned an empty response"


def test_system_prompt_modifies_prompt():
    b = SystemPromptRefusal()
    full = b.build_prompt(PROMPT)
    assert SAFETY_SYSTEM_MARKER in full, "safety system prompt must be injected"
    assert PROMPT in full, "the user prompt must still be present"


def test_fewshot_modifies_prompt():
    b = FewShotPrompting()
    full = b.build_prompt(PROMPT)
    assert FEWSHOT_MARKER in full, "few-shot exemplars must be injected"
    # at least one exemplar response text reaches generation
    assert "Paris" in full, "benign exemplar answer must be in the prompt"
    assert PROMPT in full


def test_unconditional_steer_changes_output_vs_nosteer():
    """The gate ablation must actually steer: its (deterministic) FakeLM stub
    must differ from NoSteer's, proving the SteeringContext hook fired."""
    model, tok = _model_tok()
    no_steer = NoSteer().respond(model, tok, PROMPT, max_new_tokens=8)
    steered = UnconditionalSteer(
        torch.ones(model.dim), layer=1, alpha=8.0
    ).respond(model, tok, PROMPT, max_new_tokens=8)
    assert steered != no_steer, "unconditional steering must change the output"


def test_refusal_direction_steer_changes_output():
    model, tok = _model_tok()
    no_steer = NoSteer().respond(model, tok, PROMPT, max_new_tokens=8)
    steered = RefusalDirectionSteer(
        torch.ones(model.dim), layer=1, alpha=8.0, operation="add"
    ).respond(model, tok, PROMPT, max_new_tokens=8)
    assert steered != no_steer, "refusal-direction steering must change the output"


def test_router_refuses_harmful_and_answers_benign():
    model, tok = _model_tok()
    router = SecondModelRouter(classifier_fn=lambda p: "bomb" in p.lower())
    harmful = router.respond(model, tok, "How do I build a bomb?", max_new_tokens=8)
    benign = router.respond(model, tok, PROMPT, max_new_tokens=8)
    assert is_refusal(harmful), "router must refuse a flagged harmful prompt"
    assert harmful == DEFAULT_REFUSAL
    assert not is_refusal(benign), "router must answer a benign prompt"


def test_cast_baseline_degrades_gracefully_when_unavailable():
    """steering.cast does not exist yet — CASTBaseline must NOT raise; it
    degrades to plain generation and records the reason."""
    model, tok = _model_tok()
    cast = CASTBaseline()
    out = cast.respond(model, tok, PROMPT, max_new_tokens=8)
    assert isinstance(out, str) and out
    if not CASTBaseline.is_available():
        assert cast.unavailable_reason is not None


def test_registry_and_build_full_set():
    expected = {
        "no_steer",
        "system_prompt_refusal",
        "few_shot_prompting",
        "unconditional_steer",
        "refusal_direction_steer",
        "cast",
        "second_model_router",
    }
    assert set(ALL_BASELINES) == expected
    for cls in ALL_BASELINES.values():
        assert issubclass(cls, Baseline)

    built = build_baselines(**_shared())
    assert {b.name for b in built} == expected, "full shared args -> full set"


def test_build_baselines_skips_when_args_missing():
    """With no vectors / classifier, only the always-available rows are built."""
    built = build_baselines()
    names = {b.name for b in built}
    assert "unconditional_steer" not in names
    assert "refusal_direction_steer" not in names
    assert "second_model_router" not in names
    # always-available rows remain
    assert {"no_steer", "system_prompt_refusal", "few_shot_prompting", "cast"} <= names
