"""auto-redteam -- a configurable end-to-end auto red-teaming harness.

An ablated/uncensored LOCAL Gemma model plays the adaptive attacker (prompt
generator, mutator, strategist) against a fully configurable frontier DEFENDER
(default example: Gemini Flash via the Google API, pluggable to any
OpenAI-compatible, Anthropic, or local model). Single-turn, multi-turn, and simple
agentic trajectories are supported, with reproducible runs and rich metrics.

For AUTHORIZED AI-safety research only -- see `autoredteam.banner`.

The orchestrator depends only on the Protocol INTERFACES in `autoredteam.interfaces`;
concrete providers/strategies/judges are pluggable behind them. The typed data
models in `autoredteam.models` are the vocabulary every component speaks.
"""
from __future__ import annotations

__version__ = "0.1.0"

from .banner import BANNER, AuthorizationError, assert_authorized

__all__ = ["BANNER", "AuthorizationError", "assert_authorized", "__version__"]
