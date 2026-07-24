"""openai_compat.py -- any OpenAI-Chat-Completions-compatible endpoint.

Works with OpenAI itself and with the many servers that speak the same schema
(vLLM, Together, Groq, LM Studio, ...). The `openai` SDK is imported LAZILY so the
module loads without the `openai` extra installed. `cfg.base_url` points the client
at a non-OpenAI endpoint; the key is read from `os.environ[cfg.api_key_env]`.
"""
from __future__ import annotations

from typing import Any

from ..models import GenerationResult, Message
from .base import BaseProvider, ProviderConfigError


class OpenAICompatibleProvider(BaseProvider):
    """Wraps any OpenAI-compatible chat endpoint via the async `openai` client."""

    provider_key = "openai_compatible"

    async def _generate(self, messages: list[Message], **kwargs: Any) -> GenerationResult:
        try:
            from openai import AsyncOpenAI       # lazy: only needed for real calls
        except ImportError as exc:  # pragma: no cover
            raise ProviderConfigError(
                "openai is not installed. Run `pip install auto-redteam[openai]`."
            ) from exc

        api_key = self._require_api_key()
        client = AsyncOpenAI(api_key=api_key, base_url=self.cfg.base_url)

        resp = await client.chat.completions.create(
            model=self.cfg.model,
            messages=self._to_openai_messages(messages),
            temperature=kwargs.get("temperature", self.cfg.temperature),
            max_tokens=kwargs.get("max_tokens", self.cfg.max_tokens),
        )
        choice = resp.choices[0]
        usage = getattr(resp, "usage", None)
        return GenerationResult(
            text=choice.message.content or "",
            model=self.cfg.model,
            prompt_tokens=getattr(usage, "prompt_tokens", None) if usage else None,
            completion_tokens=getattr(usage, "completion_tokens", None) if usage else None,
            finish_reason=getattr(choice, "finish_reason", None),
            raw={"provider": "openai_compatible"},
        )
