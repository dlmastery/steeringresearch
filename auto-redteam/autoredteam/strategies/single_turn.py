"""single_turn.py -- the simplest strategy: one-shot direct asks.

Mechanic: fire each seed prompt (or a neutral template around the goal) once,
optionally with a couple of light rephrase retries. This is the baseline every
multi-turn strategy is measured against -- if a single direct ask already wins,
the adaptive machinery earned nothing.
"""
from __future__ import annotations

from ..models import AttackGoal, AttackProposal, EvalResult, Trajectory
from .base import BaseStrategy, _op_reask


class SingleTurnStrategy(BaseStrategy):
    """Direct, memoryless asks. `max_variants` seeds fired up front; `max_turns`
    caps optional rephrase retries."""

    name = "single_turn"

    def __init__(self, config=None) -> None:
        super().__init__(config)
        self.max_variants: int = int(self.cfg.get("max_variants", 1))
        # single-turn is, by design, short-lived.
        self.max_turns = int(self.cfg.get("max_turns", 3))

    def generate_initial(self, goal: AttackGoal) -> list[AttackProposal]:
        n = max(1, self.max_variants)
        return [
            self._mk(self._seed(goal, i), confidence=0.4, variant=i)
            for i in range(n)
        ]

    def mutate(
        self, trajectory: Trajectory, feedback: EvalResult
    ) -> list[AttackProposal]:
        # A partial (hedged) response is worth a single rephrase; a hard refusal
        # means this strategy has little more to offer -- return nothing so the
        # orchestrator/selector can switch strategies.
        goal = trajectory.goal
        prior = self._last_prompt(trajectory, goal)
        if feedback.score <= 0.05:
            return []
        return [self._mk(_op_reask(prior, self.rng), confidence=0.35, rephrase=True)]
