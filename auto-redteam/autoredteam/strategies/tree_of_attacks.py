"""tree_of_attacks.py -- TAP-style branching search with pruning.

Published technique: "Tree of Attacks with Pruning" (TAP; Mehrotra et al., 2023,
arXiv:2312.02119) -- maintain a tree of candidate attacks, expand the most
promising leaves into several children, and prune low-scoring branches so the
search stays width-bounded.

This is the one strategy that legitimately returns *multiple* proposals per step
(branching). Node scores come from the judge (`feedback.score` / each turn's
`eval.score`); we expand the current best node and set `parent_index` on the
children so the trajectory tree is reconstructable. Operators are the shared,
content-agnostic wrappers from `base.MUTATION_OPS` -- no payloads baked in.
"""
from __future__ import annotations

from ..models import AttackGoal, AttackProposal, EvalResult, Trajectory
from .base import MUTATION_OPS, BaseStrategy


class TreeOfAttacksStrategy(BaseStrategy):
    """Branch `width` ways from the best node each round, up to `max_depth`."""

    name = "tree_of_attacks"

    def __init__(self, config=None) -> None:
        super().__init__(config)
        self.width: int = int(self.cfg.get("width", 3))          # branching factor
        self.max_depth: int = int(self.cfg.get("max_depth", 4))
        # Turn budget for the tree is depth-bounded.
        self.max_turns = int(self.cfg.get("max_turns", self.max_depth))

    def _distinct_children(self, parent: str, k: int, parent_index: int | None):
        # Use DISTINCT operators (sampled without replacement where possible) so
        # sibling branches explore genuinely different framings, not near-dupes.
        k = max(1, k)
        ops = list(MUTATION_OPS)
        self.rng.shuffle(ops)
        chosen = (ops * ((k // len(ops)) + 1))[:k]
        out: list[AttackProposal] = []
        for i, op in enumerate(chosen):
            child = op(parent, self.rng)
            out.append(
                self._mk(
                    child,
                    confidence=0.4,
                    parent_index=parent_index,
                    branch=i,
                    operator=op.__name__,
                )
            )
        return out

    def generate_initial(self, goal: AttackGoal) -> list[AttackProposal]:
        # Root layer: `width` distinct framings of the seed (the initial fan-out).
        root = self._seed(goal, 0)
        return self._distinct_children(root, self.width, parent_index=None)

    def mutate(
        self, trajectory: Trajectory, feedback: EvalResult
    ) -> list[AttackProposal]:
        # Prune-and-expand: pick the best scoring node so far and branch from it.
        best = self._best_turn(trajectory)
        if best is None:
            return self.generate_initial(trajectory.goal)
        return self._distinct_children(best.attacker_prompt, self.width, parent_index=best.index)

    def should_continue(self, trajectory: Trajectory) -> bool:
        own = self._own_turns(trajectory)
        if any(t.eval.success for t in own):
            return False
        return len(own) < self.max_depth
