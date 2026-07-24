"""defender.py -- the target-under-test wrapper.

The Defender is deliberately thin: it is *any* ModelProvider (a frontier API, a
local server, or the offline mock) dressed with a system prompt and an optional
tool list, exposing the single coroutine the orchestrator needs:

    async respond(messages) -> GenerationResult

Keeping this a thin wrapper (rather than baking model logic in) is what lets the
harness point at an arbitrary defender purely from config: swap the `provider`
and you have swapped the system under test. The Defender NEVER inspects or logs
the provider's API key -- that stays inside the provider (see providers/base.py).
"""
from __future__ import annotations

from typing import Any

from .interfaces import ModelProvider
from .models import GenerationResult, Message, Role


class Defender:
    """A configurable target model: provider + injected system prompt + tools."""

    def __init__(
        self,
        provider: ModelProvider,
        system_prompt: str | None = None,
        tools: list[dict[str, Any]] | None = None,
    ) -> None:
        self.provider = provider
        self.system_prompt = system_prompt
        self.tools = list(tools or [])

    @property
    def name(self) -> str:
        # Providers expose a `name`; fall back gracefully if a stub does not.
        return getattr(self.provider, "name", "defender")

    async def respond(self, messages: list[Message], **kwargs: Any) -> GenerationResult:
        """Answer the running conversation.

        The defender's own system prompt is prepended once (if configured and not
        already present as the leading message). Everything else -- the attacker
        turns and prior defender turns -- is passed through verbatim so the target
        sees a natural multi-turn chat. Tools, if any, ride along as a kwarg for
        providers that support tool-calling; providers that do not simply ignore it.
        """
        msgs = list(messages)
        if self.system_prompt and not (msgs and msgs[0].role == Role.SYSTEM):
            msgs = [Message(role=Role.SYSTEM, content=self.system_prompt), *msgs]

        call_kwargs = dict(kwargs)
        if self.tools:
            call_kwargs.setdefault("tools", self.tools)
        return await self.provider.generate(msgs, **call_kwargs)

    def get_config(self) -> dict[str, Any]:
        """Non-secret description of the defender for the run manifest."""
        cfg: dict[str, Any] = {"name": self.name, "has_system_prompt": bool(self.system_prompt)}
        if hasattr(self.provider, "get_config"):
            cfg["provider"] = self.provider.get_config()
        cfg["n_tools"] = len(self.tools)
        return cfg
