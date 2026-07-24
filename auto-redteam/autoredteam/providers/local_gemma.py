"""local_gemma.py -- the LOCAL attacker model served by Ollama (or any OpenAI-compat server).

In real runs the adaptive attacker is an ablated/uncensored Gemma running LOCALLY,
with no network egress beyond its own model server (see banner.py). This provider
targets an Ollama server at `cfg.base_url` via the `ollama` SDK, imported LAZILY
(`pip install auto-redteam[local]`). No API key is required for a local server; if
one is configured it is read from the environment by name.
"""
from __future__ import annotations

from typing import Any

from ..models import GenerationResult, Message
from .base import BaseProvider, ProviderConfigError


class LocalGemmaProvider(BaseProvider):
    """Wraps a locally-served chat model (Ollama) via the async `ollama` client."""

    provider_key = "local_gemma"

    async def _generate(self, messages: list[Message], **kwargs: Any) -> GenerationResult:
        try:
            from ollama import AsyncClient        # lazy: only needed for real calls
        except ImportError as exc:  # pragma: no cover
            raise ProviderConfigError(
                "ollama is not installed. Run `pip install auto-redteam[local]`."
            ) from exc

        # A local server usually needs no key; honor one only if configured.
        headers = None
        if self.cfg.api_key_env:
            headers = {"Authorization": f"Bearer {self._require_api_key()}"}

        client = AsyncClient(host=self.cfg.base_url, headers=headers)
        resp = await client.chat(
            model=self.cfg.model,
            messages=self._to_openai_messages(messages),
            options={
                "temperature": kwargs.get("temperature", self.cfg.temperature),
                "num_predict": kwargs.get("max_tokens", self.cfg.max_tokens),
            },
        )
        # ollama responses are dict-like; support attr or key access defensively.
        msg = resp["message"] if isinstance(resp, dict) else getattr(resp, "message", {})
        text = (msg.get("content", "") if isinstance(msg, dict) else getattr(msg, "content", "")) or ""

        def _get(obj: Any, key: str) -> Any:
            return obj.get(key) if isinstance(obj, dict) else getattr(obj, key, None)

        return GenerationResult(
            text=text,
            model=self.cfg.model,
            prompt_tokens=_get(resp, "prompt_eval_count"),
            completion_tokens=_get(resp, "eval_count"),
            finish_reason=_get(resp, "done_reason") or "stop",
            raw={"provider": "local_gemma"},
        )
