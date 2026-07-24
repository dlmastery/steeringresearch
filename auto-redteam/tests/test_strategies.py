"""test_strategies.py -- every registered strategy yields proposals.

Depends on the AR-STRATEGIES sibling (`autoredteam.strategies`); SKIPS until it
lands. Strategies are pure MECHANICS seeded from `goal.seed_prompts` -- we only
assert they produce non-empty, well-formed `AttackProposal`s and can mutate.
"""
from __future__ import annotations

import pytest

from autoredteam.models import (
    AttackGoal,
    EvalResult,
    Message,
    Role,
    StrategyConfig,
    Trajectory,
    Turn,
    Verdict,
)
from .conftest import require

# The strategy names the default.yaml + contract reference.
STRATEGY_NAMES = ["single_turn", "crescendo", "mutation_loop", "tree_of_attacks"]


def _toy_goal() -> AttackGoal:
    return AttackGoal(
        id="g1",
        goal="elicit a policy violation (placeholder)",
        category="test.harm.generic",
        seed_prompts=["Please explain the general topic in a harmless way."],
        severity="medium",
    )


def _seeded_trajectory(goal: AttackGoal, first_prompt: str) -> Trajectory:
    """A one-turn trajectory so `mutate` has prior context to work from."""
    turn = Turn(
        index=0,
        strategy="single_turn",
        attacker_prompt=first_prompt,
        defender_response="I cannot help with that request.",
        eval=EvalResult(verdict=Verdict.REFUSAL, success=False, score=0.1),
    )
    return Trajectory(
        id="t1",
        goal=goal,
        turns=[turn],
        messages=[Message(role=Role.USER, content=first_prompt)],
        strategies_used=["single_turn"],
    )


def test_registry_lists_strategies():
    strategies = require("autoredteam.strategies")
    names = strategies.list_strategies()
    assert isinstance(names, list) and len(names) >= 1
    # at least the core single-turn mechanic must be registered
    assert "single_turn" in names


@pytest.mark.parametrize("name", STRATEGY_NAMES)
def test_strategy_generate_initial_and_mutate(name):
    strategies = require("autoredteam.strategies")
    if name not in strategies.list_strategies():
        pytest.skip(f"strategy {name!r} not registered yet")

    strat = strategies.get_strategy(StrategyConfig(name=name, config={}))
    assert strat.name == name

    goal = _toy_goal()
    initial = strat.generate_initial(goal)
    assert isinstance(initial, list) and len(initial) >= 1
    for prop in initial:
        assert prop.prompt.strip(), "proposal prompt must be non-empty"
        assert prop.strategy  # tags itself

    # mutate against a seeded trajectory + a refusal feedback -> more proposals
    traj = _seeded_trajectory(goal, initial[0].prompt)
    feedback = EvalResult(verdict=Verdict.REFUSAL, success=False, score=0.1)
    mutated = strat.mutate(traj, feedback)
    assert isinstance(mutated, list) and len(mutated) >= 1
    assert all(p.prompt.strip() for p in mutated)

    # should_continue returns a bool
    assert isinstance(strat.should_continue(traj), bool)


def test_get_strategy_unknown_raises():
    strategies = require("autoredteam.strategies")
    with pytest.raises((ValueError, KeyError)):
        strategies.get_strategy(StrategyConfig(name="does_not_exist"))
