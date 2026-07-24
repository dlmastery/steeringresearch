"""anthropic.py -- Anthropic Claude defender/attacker via the `anthropic` SDK.

Lazy SDK import (`pip install auto-redteam[anthropic]`). Anthropic takes the system
prompt as a separate top-level argument (not a chat turn), so we split it out with
the base helper. Key from `os.environ[cfg.api_key_env]`; never stored or logged.
"""
from __future__ import annotations

from typing import Any

from ..models import GenerationResult, Message
from .base import BaseProvider, ProviderConfigError


class AnthropicProvider(BaseProvider):
    """Wraps a Claude chat model via the async `anthropic` client."""

    provider_key = "anthropic"

    async def _generate(self, messages: list[Message], **kwargs: Any) -> GenerationResult:
        try:
            from anthropic import AsyncAnthropic  # lazy: only needed for real calls
        except ImportError as exc:  # pragma: no cover
            raise ProviderConfigError(
                "anthropic is not installed. Run `pip install auto-redteam[anthropic]`."
            ) from exc

        api_key = self._require_api_key()
        client = AsyncAnthropic(api_key=api_key)

        system_text, chat = self._split_system(messages)
        # Claude wants alternating user/assistant turns; TOOL turns fold into user.
        payload = [
            {
                "role": ("assistant" if m.role.value == "assistant" else "user"),
                "content": m.content,
            }
            for m in chat
        ]
        resp = await client.messages.create(
            model=self.cfg.model,
            system=system_text or None,
            messages=payload,
            temperature=kwargs.get("temperature", self.cfg.temperature),
            max_tokens=kwargs.get("max_tokens", self.cfg.max_tokens),
        )
        # Concatenate any text blocks in the response content.
        text = "".join(
            getattr(block, "text", "") for block in getattr(resp, "content", [])
        )
        usage = getattr(resp, "usage", None)
        return GenerationResult(
            text=text,
            model=self.cfg.model,
            prompt_tokens=getattr(usage, "input_tokens", None) if usage else None,
            completion_tokens=getattr(usage, "output_tokens", None) if usage else None,
            finish_reason=getattr(resp, "stop_reason", None),
            raw={"provider": "anthropic"},
        )
