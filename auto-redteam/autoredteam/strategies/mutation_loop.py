"""mutation_loop.py -- genetic / evolutionary mutation of the prior prompt.

Published lineage: evolutionary jailbreak search such as AutoDAN
(Liu et al., 2023, arXiv:2310.04451) and GPTFuzzer (Yu et al., 2023,
arXiv:2309.10253) -- keep the best-scoring prompt so far and breed variants of it
with a pool of mutation operators, selecting on the judge's soft score.

Mechanic only: the operators (persona framing, hypothetical framing, format
coercion, request splitting, rephrase) live in `base.MUTATION_OPS` and are
content-agnostic *wrappers* around whatever prompt they receive. The "genome" is
seeded from `goal.seed_prompts`; the loop never invents payloads, it only
re-frames and re-tries the runtime seeds.
"""
from __future__ import annotations

from ..models import AttackGoal, AttackProposal, EvalResult, Trajectory
from .base import MUTATION_OPS, BaseStrategy


class MutationLoopStrategy(BaseStrategy):
    """Hill-climb by score: each turn mutates the best prompt seen so far into
    `offspring` candidate variants."""

    name = "mutation_loop"

    def __init__(self, config=None) -> None:
        super().__init__(config)
        self.offspring: int = int(self.cfg.get("offspring", 2))
        self.max_turns = int(self.cfg.get("max_turns", 8))

    def _breed(self, parent: str, k: int) -> list[AttackProposal]:
        out: list[AttackProposal] = []
        for _ in range(max(1, k)):
            op = self.rng.choice(MUTATION_OPS)
            child = op(parent, self.rng)
            out.append(self._mk(child, confidence=0.45, operator=op.__name__))
        return out

    def generate_initial(self, goal: AttackGoal) -> list[AttackProposal]:
        # Generation 0 = the raw seeds themselves (the initial population).
        seeds = goal.seed_prompts or [self._seed(goal, 0)]
        return [self._mk(s, confidence=0.35, gen=0) for s in seeds]

    def mutate(
        self, trajectory: Trajectory, feedback: EvalResult
    ) -> list[AttackProposal]:
        # Elitist selection: breed from the highest-scoring prompt observed,
        # NOT merely the last one -- that is what makes it a hill-climb.
        best = self._best_turn(trajectory)
        parent = best.attacker_prompt if best else self._last_prompt(trajectory, trajectory.goal)
        return self._breed(parent, self.offspring)
