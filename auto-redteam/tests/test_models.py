"""test_models.py -- the spine vocabulary validates + `config_hash` is stable.

These tests touch only `autoredteam.models` (the fixed spine) so they always
run -- no sibling required.
"""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from autoredteam.models import (
    AttackGoal,
    AttackProposal,
    CampaignConfig,
    EvalResult,
    GenerationResult,
    Message,
    ModelConfig,
    Role,
    Trajectory,
    Turn,
    Verdict,
)


# --------------------------------------------------------------------------- #
# Enums                                                                        #
# --------------------------------------------------------------------------- #
def test_role_and_verdict_enum_values():
    assert Role("system") is Role.SYSTEM
    assert Role.ASSISTANT.value == "assistant"
    assert Verdict.VIOLATION.value == "violation"
    # str-enum: comparable to its string value
    assert Verdict.REFUSAL == "refusal"


# --------------------------------------------------------------------------- #
# Chat primitives                                                              #
# --------------------------------------------------------------------------- #
def test_message_validates_and_rejects_bad_role():
    m = Message(role=Role.USER, content="hi")
    assert m.role is Role.USER and m.content == "hi" and m.meta == {}
    with pytest.raises(ValidationError):
        Message(role="not-a-role", content="x")


def test_generation_result_defaults():
    g = GenerationResult(text="ok", model="mock-model")
    assert g.text == "ok"
    assert g.prompt_tokens is None and g.completion_tokens is None
    assert g.raw == {}


# --------------------------------------------------------------------------- #
# Goals + evaluation + proposals                                              #
# --------------------------------------------------------------------------- #
def test_attack_goal_defaults():
    goal = AttackGoal(id="g1", goal="do X", category="cat.leaf")
    assert goal.seed_prompts == [] and goal.severity == "medium"


def test_eval_result_and_proposal_defaults():
    ev = EvalResult(verdict=Verdict.REFUSAL, success=False)
    assert ev.score == 0.0 and ev.confidence == 0.0 and ev.judge == ""
    prop = AttackProposal(prompt="p", strategy="single_turn")
    assert prop.confidence == 0.0 and prop.parent_index is None and prop.critique == ""


# --------------------------------------------------------------------------- #
# Trajectory nesting                                                          #
# --------------------------------------------------------------------------- #
def test_trajectory_nesting_roundtrip():
    goal = AttackGoal(id="g1", goal="do X", category="cat.leaf")
    turn = Turn(
        index=0,
        strategy="single_turn",
        attacker_prompt="please do X",
        defender_response="I cannot help with that.",
        eval=EvalResult(verdict=Verdict.REFUSAL, success=False, score=0.1),
    )
    traj = Trajectory(id="t1", goal=goal, turns=[turn], messages=[Message(role=Role.USER, content="please do X")])
    dumped = traj.model_dump(mode="json")
    restored = Trajectory.model_validate(dumped)
    assert restored.turns[0].eval.verdict is Verdict.REFUSAL
    assert restored.goal.id == "g1"
    assert restored.succeeded is False and restored.best_score == 0.0


# --------------------------------------------------------------------------- #
# CampaignConfig: required fields + config_hash                               #
# --------------------------------------------------------------------------- #
def _minimal_config(**overrides) -> CampaignConfig:
    mc = ModelConfig(provider="mock", model="mock-model")
    base = dict(attacker=mc, defender=mc)
    base.update(overrides)
    return CampaignConfig(**base)


def test_campaign_config_requires_attacker_and_defender():
    with pytest.raises(ValidationError):
        CampaignConfig()  # attacker + defender are required


def test_config_hash_is_stable_and_sensitive():
    a = _minimal_config(seed=42)
    b = _minimal_config(seed=42)
    assert a.config_hash() == b.config_hash()          # deterministic
    assert len(a.config_hash()) == 64                  # sha-256 hex
    c = _minimal_config(seed=43)
    assert a.config_hash() != c.config_hash()          # sensitive to any change


def test_campaign_config_roundtrip_preserves_hash():
    cfg = _minimal_config(name="rt", seed=7)
    restored = CampaignConfig.model_validate(cfg.model_dump(mode="json"))
    assert restored.config_hash() == cfg.config_hash()
