"""selection.py -- strategy selectors: static weights and online bandits.

The orchestrator asks a `StrategySelector` which attack strategy to try next,
then reports back the outcome via `update`. Two families implement the same
`interfaces.StrategySelector` Protocol:

  * WeightedSelector      -- static (mode "fixed"/"weighted"): sample by fixed
                             weights; `update` only bookkeeps, weights never move.
  * EpsilonGreedySelector -- explore uniformly with prob epsilon, else exploit the
                             arm with the best mean reward.
  * UCBSelector           -- UCB1: exploit mean + an optimism-under-uncertainty bonus.
  * ThompsonSelector      -- Beta-Bernoulli posterior sampling (reward in [0,1]
                             folded into a Beta(alpha,beta) per arm).

Why bandits: against a fixed defender, some strategies simply work better. A
bandit LEARNS that online -- concentrating pulls on the productive strategy while
still exploring -- instead of burning a fixed mixture of compute on losers.

Reproducibility: all randomness comes from a `random.Random(seed)` threaded in at
construction. `stats()` returns a JSON-serializable per-arm dict; `get_selector`
can warm-start from a previously persisted `stats()` blob so a resumed campaign
keeps everything the bandit already learned.
"""
from __future__ import annotations

import math
import random
from typing import Any

from .models import SelectionConfig


# --------------------------------------------------------------------------- #
# Per-arm accounting                                                          #
# --------------------------------------------------------------------------- #
class _Arm:
    """Sufficient statistics for one strategy arm (all JSON-serializable)."""

    def __init__(self) -> None:
        self.pulls: int = 0
        self.total_reward: float = 0.0
        self.total_turns: int = 0
        # Beta-Bernoulli posterior params (used by Thompson; harmless for others).
        # Start at the uniform prior Beta(1,1).
        self.alpha: float = 1.0
        self.beta: float = 1.0

    @property
    def mean(self) -> float:
        return self.total_reward / self.pulls if self.pulls else 0.0

    @property
    def avg_turns(self) -> float:
        return self.total_turns / self.pulls if self.pulls else 0.0

    def record(self, reward: float, turns: int) -> None:
        r = min(1.0, max(0.0, float(reward)))   # clamp to [0,1] -- rewards are rates
        self.pulls += 1
        self.total_reward += r
        self.total_turns += int(turns)
        # Fold the (possibly soft) reward into the Beta posterior.
        self.alpha += r
        self.beta += 1.0 - r

    def to_dict(self) -> dict[str, Any]:
        return {
            "pulls": self.pulls,
            "total_reward": round(self.total_reward, 6),
            "total_turns": self.total_turns,
            "mean_reward": round(self.mean, 6),
            "avg_turns": round(self.avg_turns, 6),
            "alpha": round(self.alpha, 6),
            "beta": round(self.beta, 6),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "_Arm":
        a = cls()
        a.pulls = int(d.get("pulls", 0))
        a.total_reward = float(d.get("total_reward", 0.0))
        a.total_turns = int(d.get("total_turns", 0))
        a.alpha = float(d.get("alpha", 1.0))
        a.beta = float(d.get("beta", 1.0))
        return a


# --------------------------------------------------------------------------- #
# Base selector                                                               #
# --------------------------------------------------------------------------- #
class BaseSelector:
    """Shared arm bookkeeping, RNG, and (de)serialization for all selectors."""

    mode: str = "base"

    def __init__(
        self,
        strategy_names: list[str],
        seed: int = 0,
        warm_stats: dict[str, Any] | None = None,
    ) -> None:
        if not strategy_names:
            raise ValueError("selector needs at least one strategy name")
        self.names: list[str] = list(strategy_names)
        self.rng = random.Random(seed)
        self.arms: dict[str, _Arm] = {n: _Arm() for n in self.names}
        if warm_stats:
            for n, d in warm_stats.items():
                if n in self.arms and isinstance(d, dict):
                    self.arms[n] = _Arm.from_dict(d)

    # -- helpers ------------------------------------------------------------- #
    def _avail(self, available: list[str] | None) -> list[str]:
        names = [n for n in (available or self.names) if n in self.arms]
        if not names:
            raise ValueError("no available strategies known to this selector")
        return names

    @property
    def _total_pulls(self) -> int:
        return sum(a.pulls for a in self.arms.values())

    def _argmax(self, names: list[str], key) -> str:
        """Deterministic-under-seed argmax: ties broken by an RNG shuffle so no
        strategy is structurally favoured by dict order."""
        shuffled = list(names)
        self.rng.shuffle(shuffled)
        return max(shuffled, key=key)

    # -- Protocol surface ---------------------------------------------------- #
    def update(self, strategy: str, reward: float, turns: int) -> None:
        if strategy in self.arms:
            self.arms[strategy].record(reward, turns)

    def stats(self) -> dict[str, Any]:
        return {n: self.arms[n].to_dict() for n in self.names}

    def select(self, available: list[str]) -> str:  # pragma: no cover - overridden
        raise NotImplementedError


# --------------------------------------------------------------------------- #
# Static weighted selector (modes: fixed / weighted)                          #
# --------------------------------------------------------------------------- #
class WeightedSelector(BaseSelector):
    """Sample by fixed weights. `update` bookkeeps but weights never change --
    this is the backward-compatible, non-learning baseline."""

    mode = "weighted"

    def __init__(
        self,
        strategy_names: list[str],
        seed: int = 0,
        weights: dict[str, float] | None = None,
        warm_stats: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(strategy_names, seed, warm_stats)
        w = weights or {}
        # Default to uniform; unknown/missing names get weight 1.0.
        self.weights: dict[str, float] = {n: float(w.get(n, 1.0)) for n in self.names}

    def select(self, available: list[str]) -> str:
        names = self._avail(available)
        wts = [max(0.0, self.weights[n]) for n in names]
        if sum(wts) <= 0:
            return self.rng.choice(names)
        return self.rng.choices(names, weights=wts, k=1)[0]


# --------------------------------------------------------------------------- #
# Epsilon-greedy bandit                                                        #
# --------------------------------------------------------------------------- #
class EpsilonGreedySelector(BaseSelector):
    """With prob epsilon explore uniformly; otherwise exploit the best mean."""

    mode = "epsilon_greedy"

    def __init__(
        self,
        strategy_names: list[str],
        seed: int = 0,
        epsilon: float = 0.15,
        warm_stats: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(strategy_names, seed, warm_stats)
        self.epsilon = float(epsilon)

    def select(self, available: list[str]) -> str:
        names = self._avail(available)
        # Always try an unpulled arm first (optimistic cold-start).
        unpulled = [n for n in names if self.arms[n].pulls == 0]
        if unpulled:
            return self.rng.choice(unpulled)
        if self.rng.random() < self.epsilon:
            return self.rng.choice(names)
        return self._argmax(names, key=lambda n: self.arms[n].mean)


# --------------------------------------------------------------------------- #
# UCB1 bandit                                                                  #
# --------------------------------------------------------------------------- #
class UCBSelector(BaseSelector):
    """UCB1: pick argmax( mean + c*sqrt(ln(total)/pulls) ). Unpulled arms win
    outright (infinite bonus), guaranteeing each is tried once."""

    mode = "ucb"

    def __init__(
        self,
        strategy_names: list[str],
        seed: int = 0,
        c: float = 1.4,
        warm_stats: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(strategy_names, seed, warm_stats)
        self.c = float(c)

    def _ucb(self, name: str, total: int) -> float:
        arm = self.arms[name]
        if arm.pulls == 0:
            return math.inf
        return arm.mean + self.c * math.sqrt(math.log(max(1, total)) / arm.pulls)

    def select(self, available: list[str]) -> str:
        names = self._avail(available)
        total = self._total_pulls
        return self._argmax(names, key=lambda n: self._ucb(n, total))


# --------------------------------------------------------------------------- #
# Thompson sampling (Beta-Bernoulli)                                          #
# --------------------------------------------------------------------------- #
class ThompsonSelector(BaseSelector):
    """Posterior sampling: draw theta_a ~ Beta(alpha_a, beta_a) per arm, pick the
    argmax draw. Naturally balances explore/exploit and tends to dominate the
    other bandits empirically."""

    mode = "thompson"

    def select(self, available: list[str]) -> str:
        names = self._avail(available)
        # betavariate is part of random.Random -> fully reproducible under seed.
        def draw(n: str) -> float:
            a = self.arms[n]
            return self.rng.betavariate(a.alpha, a.beta)

        # No RNG-shuffle tie-break needed: the Beta draws already randomize order.
        return max(names, key=draw)


# --------------------------------------------------------------------------- #
# Factory                                                                      #
# --------------------------------------------------------------------------- #
_MODES = {
    "fixed": WeightedSelector,
    "weighted": WeightedSelector,
    "epsilon_greedy": EpsilonGreedySelector,
    "ucb": UCBSelector,
    "thompson": ThompsonSelector,
}


def get_selector(
    sel: SelectionConfig,
    strategy_names: list[str],
    warm_stats: dict[str, Any] | None = None,
    seed: int = 0,
) -> "BaseSelector":
    """Build the selector named by `sel.mode`, warm-starting from `warm_stats`.

    `fixed`/`weighted` -> WeightedSelector (using `sel.initial_weights`); the three
    bandit modes read their exploration knobs off the SelectionConfig. Unknown
    modes raise ValueError. `seed` makes the whole selection stream reproducible.
    """
    mode = sel.mode
    cls = _MODES.get(mode)
    if cls is None:
        raise ValueError(
            f"unknown selection mode '{mode}'; available: {sorted(_MODES)}"
        )
    if cls is WeightedSelector:
        return WeightedSelector(
            strategy_names, seed=seed,
            weights=dict(sel.initial_weights) or None,
            warm_stats=warm_stats,
        )
    if cls is EpsilonGreedySelector:
        return EpsilonGreedySelector(
            strategy_names, seed=seed, epsilon=sel.epsilon, warm_stats=warm_stats
        )
    if cls is UCBSelector:
        return UCBSelector(
            strategy_names, seed=seed, c=sel.ucb_c, warm_stats=warm_stats
        )
    return ThompsonSelector(strategy_names, seed=seed, warm_stats=warm_stats)


__all__ = [
    "BaseSelector",
    "WeightedSelector",
    "EpsilonGreedySelector",
    "UCBSelector",
    "ThompsonSelector",
    "get_selector",
]


# --------------------------------------------------------------------------- #
# Self-test: a bandit must converge to the higher-reward arm                   #
# --------------------------------------------------------------------------- #
def _simulate(selector: "BaseSelector", true_rates: dict[str, float],
              n: int, reward_seed: int) -> dict[str, int]:
    """Run `n` pulls against a stationary Bernoulli environment; return pick counts."""
    arms = list(true_rates.keys())
    rrng = random.Random(reward_seed)           # environment RNG, separate from policy
    picks = {a: 0 for a in arms}
    for _ in range(n):
        choice = selector.select(arms)
        picks[choice] += 1
        reward = 1.0 if rrng.random() < true_rates[choice] else 0.0
        selector.update(choice, reward, turns=1)
    return picks


if __name__ == "__main__":  # pragma: no cover
    # Two arms: "good" pays off 0.75 of the time, "bad" only 0.25. A working
    # bandit should learn to pull "good" the majority of the time over ~200 pulls.
    TRUE = {"good": 0.75, "bad": 0.25}
    N = 200
    print(f"convergence test: {TRUE}, N={N} pulls each\n")

    configs = {
        "weighted (baseline)": SelectionConfig(mode="weighted"),
        "epsilon_greedy": SelectionConfig(mode="epsilon_greedy", epsilon=0.15),
        "ucb": SelectionConfig(mode="ucb", ucb_c=1.4),
        "thompson": SelectionConfig(mode="thompson"),
    }
    results: dict[str, float] = {}
    for label, cfg in configs.items():
        sel = get_selector(cfg, list(TRUE.keys()), seed=1)
        picks = _simulate(sel, TRUE, N, reward_seed=99)
        frac = picks["good"] / N
        results[label] = frac
        print(f"  {label:22s} good={picks['good']:3d} bad={picks['bad']:3d}  "
              f"good_fraction={frac:.3f}")

    # The three BANDITS must beat 50%; the static weighted baseline is expected
    # to hover near its fixed mixture (~50%) and is reported for contrast only.
    for label in ("epsilon_greedy", "ucb", "thompson"):
        assert results[label] > 0.5, (
            f"{label} failed to converge on the better arm: {results[label]:.3f}"
        )
    print("\nOK: all bandits converged to the better arm (picked >50%).")
