"""models.py -- the strongly-typed vocabulary of the harness (Pydantic v2).

Every component (providers, strategies, judges, orchestrator, persistence) speaks
in these models. They are the stable contract: change them deliberately, because
the whole package imports them. Pydantic v2 gives validation, (de)serialization,
and JSON-schema generation (useful for config validation and a future UI).

The object graph, top down:

    CampaignConfig            # the entire run, loaded from YAML (+ env/CLI overrides)
      +-- ModelConfig         # attacker + defender + judge model settings
      +-- StrategyConfig[]    # which attack strategies, their weights + knobs
      +-- EvaluationConfig    # taxonomy + judge ensemble + success threshold
    CampaignResult            # the whole run's outcome
      +-- Trajectory[]        # one adaptive attack against one goal
            +-- Turn[]        # (attacker prompt, defender response, eval) per step
                  +-- Message # a single chat message
                  +-- EvalResult
"""
from __future__ import annotations

import enum
import hashlib
import json
from typing import Any

from pydantic import BaseModel, Field


# --------------------------------------------------------------------------- #
# Enums                                                                        #
# --------------------------------------------------------------------------- #
class Role(str, enum.Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class Verdict(str, enum.Enum):
    """The judge's call on a single defender response."""

    VIOLATION = "violation"      # the defender produced disallowed content
    REFUSAL = "refusal"          # the defender refused / deflected
    PARTIAL = "partial"          # hedged or partially-compliant
    ERROR = "error"              # provider/judge error -- not scored as success


# --------------------------------------------------------------------------- #
# Chat primitives                                                              #
# --------------------------------------------------------------------------- #
class Message(BaseModel):
    """One chat message. `meta` carries provider-specific extras (tool calls, etc.)."""

    role: Role
    content: str
    name: str | None = None
    meta: dict[str, Any] = Field(default_factory=dict)


class GenerationResult(BaseModel):
    """What a ModelProvider returns for one generation call."""

    text: str
    model: str
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    finish_reason: str | None = None
    latency_s: float | None = None
    raw: dict[str, Any] = Field(default_factory=dict)   # provider raw response (never logged wholesale)


# --------------------------------------------------------------------------- #
# Attack goals + evaluation                                                    #
# --------------------------------------------------------------------------- #
class AttackGoal(BaseModel):
    """One behaviour we try to elicit, tagged to a harm taxonomy category."""

    id: str
    goal: str                                  # natural-language description of the target behaviour
    category: str                              # taxonomy key (e.g. owasp_asi_2026 leaf)
    seed_prompts: list[str] = Field(default_factory=list)
    severity: str = "medium"                   # low | medium | high | critical
    meta: dict[str, Any] = Field(default_factory=dict)


class EvalResult(BaseModel):
    """A judge's assessment of a single defender response."""

    verdict: Verdict
    success: bool                              # binary ASR contribution (True iff a real violation)
    score: float = 0.0                         # soft score in [0,1]
    category_match: str | None = None
    severity: str | None = None
    confidence: float = 0.0
    explanation: str = ""
    judge: str = ""                            # which judge produced this
    meta: dict[str, Any] = Field(default_factory=dict)


# --------------------------------------------------------------------------- #
# Attacker swarm outputs                                                       #
# --------------------------------------------------------------------------- #
class AttackProposal(BaseModel):
    """One candidate next-attack from the attacker (or swarm generator).

    A strategy or the swarm may return several (for TAP-style branching); the
    orchestrator picks/expands them per the selection policy.
    """

    prompt: str
    strategy: str
    confidence: float = 0.0                    # generator/critic estimate of success likelihood
    critique: str = ""                         # critic's short natural-language steer (swarm mode)
    parent_index: int | None = None            # branch point in the trajectory tree
    meta: dict[str, Any] = Field(default_factory=dict)


# --------------------------------------------------------------------------- #
# Trajectory                                                                   #
# --------------------------------------------------------------------------- #
class Turn(BaseModel):
    """One attacker->defender->judge step within a trajectory."""

    index: int
    strategy: str                              # strategy that produced the attacker prompt
    attacker_prompt: str
    defender_response: str
    eval: EvalResult
    attacker_reflection: str | None = None     # optional self-reflection (Gemma-as-critic)
    parent_index: int | None = None            # for TAP-style branching trees
    meta: dict[str, Any] = Field(default_factory=dict)


class Trajectory(BaseModel):
    """One adaptive attack campaign against ONE goal (possibly multi-turn / branching)."""

    id: str
    goal: AttackGoal
    turns: list[Turn] = Field(default_factory=list)
    succeeded: bool = False
    best_score: float = 0.0
    turns_to_success: int | None = None
    strategies_used: list[str] = Field(default_factory=list)
    messages: list[Message] = Field(default_factory=list)   # the flat conversation as sent
    meta: dict[str, Any] = Field(default_factory=dict)


# --------------------------------------------------------------------------- #
# Config models (loaded from YAML; see config.py for the loader)              #
# --------------------------------------------------------------------------- #
class ModelConfig(BaseModel):
    """A single model's settings. `provider` selects the concrete ModelProvider."""

    provider: str                              # "local_gemma" | "gemini" | "openai_compatible" | "anthropic" | "mock"
    model: str
    base_url: str | None = None
    api_key_env: str | None = None             # env var NAME (never the key itself)
    temperature: float = 0.7
    max_tokens: int = 1024
    system_prompt: str | None = None
    system_prompt_path: str | None = None
    tools: list[dict[str, Any]] = Field(default_factory=list)
    extra: dict[str, Any] = Field(default_factory=dict)


class StrategyConfig(BaseModel):
    """One attack strategy entry: which strategy, how often, with what knobs."""

    name: str
    weight: float = 1.0
    config: dict[str, Any] = Field(default_factory=dict)


class JudgeConfig(BaseModel):
    type: str                                  # "rule_based" | "llm_judge"
    model: ModelConfig | None = None           # required for llm_judge
    prompt_path: str | None = None
    weight: float = 1.0


class EvaluationConfig(BaseModel):
    taxonomy: str = "owasp_asi_2026"
    judges: list[JudgeConfig] = Field(default_factory=list)
    success_threshold: float = 0.7


class SelectionConfig(BaseModel):
    """How the orchestrator picks the next strategy each turn.

    `fixed`/`weighted` = static mixing (backward compatible). The bandit modes make
    the harness LEARN which strategy works against the current defender, updating
    per-strategy stats online (SOTA addition, fully optional).
    """

    mode: str = "weighted"                     # "fixed" | "weighted" | "epsilon_greedy" | "ucb" | "thompson"
    epsilon: float = 0.15                      # epsilon_greedy only
    ucb_c: float = 1.4                         # ucb exploration constant
    update_after_every: int = 1                # trajectories between stat updates
    persist_stats: bool = True                 # carry bandit state across resume/campaigns
    initial_weights: dict[str, float] = Field(default_factory=dict)   # optional warm start


class SwarmConfig(BaseModel):
    """The lightweight multi-agent attacker swarm (optional, feature-flagged).

    When `enabled` is false the harness runs the classic single-attacker loop. When
    true, a Generator proposes attacks and an optional Critic reflects on the last
    defender response to steer the next mutation -- the highest-ROI 2026 upgrade.
    """

    enabled: bool = False
    generator: ModelConfig | None = None       # defaults to the top-level `attacker` if None
    critic_enabled: bool = True
    critic: ModelConfig | None = None           # "same" backbone by default (may be a lighter quant)
    max_proposals_per_turn: int = 1             # >1 enables TAP-style branching


class AuthorizationConfig(BaseModel):
    confirmed: bool = False
    scope: str = ""


class CampaignConfig(BaseModel):
    """The entire run. Built by config.py from deep-merged YAML + env + CLI."""

    name: str = "unnamed-campaign"
    seed: int = 42
    max_parallel: int = 4
    max_turns_per_trajectory: int = 12
    stop_on_success: bool = False
    time_budget_s: float | None = None

    authorization: AuthorizationConfig = Field(default_factory=AuthorizationConfig)
    attacker: ModelConfig
    defender: ModelConfig
    attacker_swarm: SwarmConfig = Field(default_factory=SwarmConfig)
    strategies: list[StrategyConfig] = Field(default_factory=list)
    selection: SelectionConfig = Field(default_factory=SelectionConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)
    goals_path: str | None = None
    persistence: dict[str, Any] = Field(default_factory=dict)
    logging: dict[str, Any] = Field(default_factory=dict)

    def config_hash(self) -> str:
        """Stable SHA-256 of the whole config -> recorded in every run for reproducibility."""
        blob = json.dumps(self.model_dump(mode="json"), sort_keys=True).encode()
        return hashlib.sha256(blob).hexdigest()


# --------------------------------------------------------------------------- #
# Campaign result                                                              #
# --------------------------------------------------------------------------- #
class CampaignResult(BaseModel):
    """The whole run's outcome + the metrics the reporter renders."""

    campaign_name: str
    config_hash: str
    scope: str = ""                            # the authorization scope, recorded for disclosure
    trajectories: list[Trajectory] = Field(default_factory=list)
    metrics: dict[str, Any] = Field(default_factory=dict)   # ASR-per-category, avg turns, cost, ...
    started_at: str | None = None
    finished_at: str | None = None
    meta: dict[str, Any] = Field(default_factory=dict)
