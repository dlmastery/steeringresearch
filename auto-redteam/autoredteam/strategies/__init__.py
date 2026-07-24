"""strategies -- the pluggable AttackStrategy registry.

Add a strategy by writing a class that satisfies `interfaces.AttackStrategy`
and registering it in `STRATEGIES` below -- the orchestrator resolves it by name
through `get_strategy`, with no core changes. Every strategy here is a PUBLISHED
red-teaming *mechanic* (single-shot, Crescendo escalation, evolutionary mutation,
TAP branching) with content-agnostic framing operators; concrete attack seeds
arrive at runtime via `AttackGoal.seed_prompts` from the authorized-testing YAML.
"""
from __future__ import annotations

from ..interfaces import AttackStrategy
from ..models import StrategyConfig
from .base import BaseStrategy
from .crescendo import CrescendoStrategy
from .mutation_loop import MutationLoopStrategy
from .single_turn import SingleTurnStrategy
from .tree_of_attacks import TreeOfAttacksStrategy

# name -> class. Keys are the stable identifiers used in config + the bandit.
STRATEGIES: dict[str, type[BaseStrategy]] = {
    SingleTurnStrategy.name: SingleTurnStrategy,
    CrescendoStrategy.name: CrescendoStrategy,
    MutationLoopStrategy.name: MutationLoopStrategy,
    TreeOfAttacksStrategy.name: TreeOfAttacksStrategy,
}


def list_strategies() -> list[str]:
    """Registered strategy names (stable order)."""
    return list(STRATEGIES.keys())


def get_strategy(sc: StrategyConfig) -> AttackStrategy:
    """Factory: build the strategy named by `sc.name`, passing `sc.config` knobs.

    Unknown names raise ValueError (fail loud -- a typo in the config must not
    silently fall back to a different attack pattern).
    """
    cls = STRATEGIES.get(sc.name)
    if cls is None:
        raise ValueError(
            f"unknown strategy '{sc.name}'; available: {list_strategies()}"
        )
    return cls(sc.config)


__all__ = [
    "STRATEGIES",
    "list_strategies",
    "get_strategy",
    "BaseStrategy",
    "SingleTurnStrategy",
    "CrescendoStrategy",
    "MutationLoopStrategy",
    "TreeOfAttacksStrategy",
]


if __name__ == "__main__":  # pragma: no cover
    # Self-test: every strategy yields non-empty proposals from a toy goal.
    from ..models import AttackGoal

    goal = AttackGoal(
        id="t1",
        goal="demonstrate the harness plumbing",
        category="test.category",
        seed_prompts=["placeholder benign seed prompt"],
    )
    print("strategies:", list_strategies())
    for name in list_strategies():
        strat = get_strategy(StrategyConfig(name=name, config={"seed": 7}))
        proposals = strat.generate_initial(goal)
        assert proposals, f"{name} produced no initial proposals"
        assert all(p.strategy == name for p in proposals)
        print(f"  {name:16s} -> {len(proposals)} initial proposal(s)")
    print("OK: all strategies produce proposals")
