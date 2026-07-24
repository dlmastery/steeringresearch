"""test_config.py -- the authorization gate + `load_config` deep-merge precedence.

The authorization-gate tests use only the spine (`autoredteam.banner`) and
always run. The `load_config` tests depend on the AR-CONFIG sibling and SKIP via
`require(...)` until it lands.

Deep-merge precedence contract (increasing precedence):
    config/default.yaml  <  user --config yaml  <  HARNESS__A__B env  <  overrides (CLI)
"""
from __future__ import annotations

import textwrap

import pytest

from autoredteam.banner import AuthorizationError, assert_authorized
from .conftest import require


# --------------------------------------------------------------------------- #
# Authorization gate (spine -- always runs)                                   #
# --------------------------------------------------------------------------- #
def test_authorization_gate_raises_when_unconfirmed():
    with pytest.raises(AuthorizationError):
        assert_authorized({"confirmed": False, "scope": "a valid long scope string"})
    with pytest.raises(AuthorizationError):
        assert_authorized({})          # missing entirely
    with pytest.raises(AuthorizationError):
        assert_authorized(None)


def test_authorization_gate_requires_a_real_scope():
    # confirmed but scope too short (<10 chars) -> refuse
    with pytest.raises(AuthorizationError):
        assert_authorized({"confirmed": True, "scope": "short"})


def test_authorization_gate_returns_scope_when_valid():
    scope = "internal safety eval of our own mock deployment, ticket SAFE-1"
    assert assert_authorized({"confirmed": True, "scope": scope}) == scope


# --------------------------------------------------------------------------- #
# load_config: defaults + deep-merge precedence (AR-CONFIG sibling)           #
# --------------------------------------------------------------------------- #
def test_load_config_defaults_are_mock_and_authorized_is_false():
    config = require("autoredteam.config")
    cfg = config.load_config()
    assert cfg.attacker.provider == "mock"
    assert cfg.defender.provider == "mock"
    assert cfg.seed == 42                       # default.yaml value
    assert cfg.authorization.confirmed is False  # must be explicitly affirmed
    assert len(cfg.config_hash()) == 64


def test_loaded_default_config_is_blocked_by_the_gate():
    """End-to-end: the shipped default is NOT authorized, so the gate refuses it."""
    config = require("autoredteam.config")
    cfg = config.load_config()
    with pytest.raises(AuthorizationError):
        assert_authorized(cfg.authorization.model_dump())


def test_load_config_deep_merge_precedence(tmp_path):
    config = require("autoredteam.config")

    user_yaml = tmp_path / "user.yaml"
    user_yaml.write_text(textwrap.dedent("""\
        seed: 100
        """), encoding="ascii")

    # default only -> 42
    assert config.load_config().seed == 42
    # file over default -> 100
    assert config.load_config(str(user_yaml)).seed == 100
    # env over file -> 200 (HARNESS__<PATH> double-underscore, string coerced)
    assert config.load_config(str(user_yaml), env={"HARNESS__SEED": "200"}).seed == 200
    # overrides (CLI) win over everything -> 300
    assert config.load_config(
        str(user_yaml),
        overrides={"seed": 300},
        env={"HARNESS__SEED": "200"},
    ).seed == 300


def test_load_goals_returns_goal_objects():
    config = require("autoredteam.config")
    goals = config.load_goals("config/goals_example.yaml")
    assert isinstance(goals, list) and len(goals) >= 1
    assert all(g.id and g.category for g in goals)
