"""test_orchestrator_e2e.py -- a full 2-goal MOCK campaign end to end.

Depends on the whole sibling ecosystem (providers + strategies + selection +
evaluator + persistence + orchestrator); SKIPS until they all land. Uses the
mock provider on both sides, so it is offline + deterministic. Asserts the
AR-TESTS contract: a `CampaignResult` with metrics AND both a success and a
non-success trajectory, with the authorization scope recorded.
"""
from __future__ import annotations

import pytest

from autoredteam.models import CampaignResult
from .conftest import require


def _build_orchestrator(config):
    """Construct the Orchestrator from a config, tolerating minor ctor shape.

    Contract: `Orchestrator(config, ...factories injected or built via
    factories)`. The offline path builds every sibling from config internally,
    so a config-only construction is the expected entry point.
    """
    orchestrator = require("autoredteam.orchestrator")
    Orchestrator = orchestrator.Orchestrator
    try:
        return Orchestrator(config)
    except TypeError:
        return Orchestrator(config=config)


async def test_two_goal_mock_campaign_end_to_end(mock_campaign_config, toy_goals):
    # Require the full ecosystem up front so a missing piece SKIPS (not errors).
    for mod in ("providers", "strategies", "selection", "evaluator", "persistence", "orchestrator"):
        require(f"autoredteam.{mod}")

    orch = _build_orchestrator(mock_campaign_config)
    result = await orch.run(toy_goals)

    # Shape
    assert isinstance(result, CampaignResult)
    assert result.campaign_name == mock_campaign_config.name
    assert result.config_hash == mock_campaign_config.config_hash()

    # Authorization scope recorded for disclosure
    assert result.scope and len(result.scope) >= 10

    # Metrics computed
    assert isinstance(result.metrics, dict) and result.metrics

    # One trajectory per goal, each with at least one turn
    assert len(result.trajectories) == len(toy_goals)
    for traj in result.trajectories:
        assert len(traj.turns) >= 1

    # The mock ecosystem must yield BOTH a success and a non-success trajectory.
    outcomes = {traj.succeeded for traj in result.trajectories}
    assert True in outcomes, "expected at least one successful trajectory"
    assert False in outcomes, "expected at least one non-successful trajectory"


async def test_campaign_refuses_without_authorization(mock_campaign_config, toy_goals):
    """The orchestrator must gate on authorization before any generation."""
    from autoredteam.banner import AuthorizationError

    for mod in ("providers", "strategies", "selection", "evaluator", "persistence", "orchestrator"):
        require(f"autoredteam.{mod}")

    unauthorized = mock_campaign_config.model_copy(deep=True)
    unauthorized.authorization.confirmed = False

    orch = _build_orchestrator(unauthorized)
    with pytest.raises(AuthorizationError):
        await orch.run(toy_goals)
