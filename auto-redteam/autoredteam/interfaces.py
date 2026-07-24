"""interfaces.py -- the pluggable Protocols the orchestrator depends on.

The orchestrator knows ONLY these interfaces, never a concrete implementation. A
new provider / strategy / judge is added by writing a class that satisfies the
matching Protocol and registering it -- no core changes. This is what keeps the
harness "driven almost entirely by configuration".

    ModelProvider     wraps any chat model (local Gemma, Gemini, OpenAI, Anthropic, mock)
    AttackStrategy    turns a goal + feedback into candidate prompts (Crescendo, TAP, ...)
    AttackerSwarm     the optional Generator(+Critic) multi-agent attacker (SOTA)
    StrategySelector  picks the next strategy (fixed weights or a learning bandit)
    Judge             scores a defender response (rule-based or LLM-as-judge)
    Persistence       stores trajectories + bandit stats (jsonl / sqlite)
    Reporter          renders a CampaignResult to HTML / Markdown / CSV
"""
from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from .models import (
    AttackGoal,
    AttackProposal,
    CampaignResult,
    EvalResult,
    GenerationResult,
    Message,
    Trajectory,
)


@runtime_checkable
class ModelProvider(Protocol):
    """Any chat model behind a uniform async surface. `name` identifies it in logs."""

    name: str

    async def generate(self, messages: list[Message], **kwargs: Any) -> GenerationResult: ...

    def get_config(self) -> dict[str, Any]:
        """Non-secret config for the run manifest (NEVER return the API key)."""
        ...


@runtime_checkable
class AttackStrategy(Protocol):
    """A named attack pattern. Stateless w.r.t. the campaign; state lives in Trajectory."""

    name: str

    def generate_initial(self, goal: AttackGoal) -> list[AttackProposal]:
        """The opening prompt(s) for a fresh goal."""
        ...

    def mutate(self, trajectory: Trajectory, feedback: EvalResult) -> list[AttackProposal]:
        """Given the trajectory so far and the last eval, propose the next attack(s)."""
        ...

    def should_continue(self, trajectory: Trajectory) -> bool:
        """Whether this strategy has more to try on this trajectory."""
        ...


@runtime_checkable
class AttackerSwarm(Protocol):
    """The optional multi-agent attacker: a Generator proposes, a Critic reflects.

    When the swarm is disabled the orchestrator uses AttackStrategy directly. When
    enabled, the swarm may consult the selected strategy as a hint and return
    several proposals (branching). Implementations own their own model providers.
    """

    async def propose_next(
        self,
        goal: AttackGoal,
        trajectory: Trajectory,
        strategy_hint: str | None = None,
    ) -> list[AttackProposal]: ...


@runtime_checkable
class StrategySelector(Protocol):
    """Picks the next strategy name. `update` lets bandit selectors learn online."""

    def select(self, available: list[str]) -> str: ...

    def update(self, strategy: str, reward: float, turns: int) -> None:
        """Record the outcome of a trajectory that used `strategy` (reward in [0,1])."""
        ...

    def stats(self) -> dict[str, Any]:
        """Per-strategy statistics, persisted for resume + reporting."""
        ...


@runtime_checkable
class Judge(Protocol):
    """Scores a single defender response against a goal. Rule-based or LLM-as-judge."""

    name: str

    async def evaluate(
        self,
        goal: AttackGoal,
        attacker_prompt: str,
        defender_response: str,
    ) -> EvalResult: ...


@runtime_checkable
class Persistence(Protocol):
    """Durable storage for trajectories and selector stats (jsonl / sqlite)."""

    def save_trajectory(self, campaign: str, trajectory: Trajectory) -> None: ...

    def save_result(self, result: CampaignResult) -> None: ...

    def load_stats(self, campaign: str) -> dict[str, Any]:
        """Return persisted bandit/selector stats for warm-start, or {} if none."""
        ...

    def save_stats(self, campaign: str, stats: dict[str, Any]) -> None: ...


@runtime_checkable
class Reporter(Protocol):
    """Renders a finished campaign to a human-readable artifact."""

    def render(self, result: CampaignResult, out_dir: str) -> list[str]:
        """Write report file(s); return the paths written."""
        ...
