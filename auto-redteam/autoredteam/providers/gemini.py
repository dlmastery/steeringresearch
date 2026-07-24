"""gemini.py -- Google Gemini defender/attacker via the `google-genai` SDK.

The SDK is imported LAZILY inside `_generate` so this module imports fine even when
the `gemini` extra is not installed (`pip install auto-redteam[gemini]`). The API
key is read from `os.environ[cfg.api_key_env]` at call time and never stored.

This provider is pure transport -- it forwards messages to the model and returns
the text. No exploit content lives here.
"""
from __future__ import annotations

from typing import Any

from ..models import GenerationResult, Message
from .base import BaseProvider, ProviderConfigError


class GeminiProvider(BaseProvider):
    """Wraps a Gemini chat model. Requires the `google-genai` package at call time."""

    provider_key = "gemini"

    async def _generate(self, messages: list[Message], **kwargs: Any) -> GenerationResult:
        try:
            from google import genai            # lazy: only needed for real calls
            from google.genai import types
        except ImportError as exc:  # pragma: no cover - exercised only without the extra
            raise ProviderConfigError(
                "google-genai is not installed. Run `pip install auto-redteam[gemini]`."
            ) from exc

        api_key = self._require_api_key()
        client = genai.Client(api_key=api_key)

        system_text, chat = self._split_system(messages)
        # Gemini uses role "model" for assistant turns; system goes in the config.
        contents = [
            types.Content(
                role=("model" if m.role.value == "assistant" else "user"),
                parts=[types.Part.from_text(text=m.content)],
            )
            for m in chat
        ]
        config = types.GenerateContentConfig(
            temperature=kwargs.get("temperature", self.cfg.temperature),
            max_output_tokens=kwargs.get("max_tokens", self.cfg.max_tokens),
            system_instruction=system_text or None,
        )

        # The SDK call is sync; run it off the event loop so we stay async-friendly.
        import asyncio

        def _call() -> Any:
            return client.models.generate_content(
                model=self.cfg.model, contents=contents, config=config
            )

        resp = await asyncio.to_thread(_call)

        usage = getattr(resp, "usage_metadata", None)
        return GenerationResult(
            text=getattr(resp, "text", "") or "",
            model=self.cfg.model,
            prompt_tokens=getattr(usage, "prompt_token_count", None) if usage else None,
            completion_tokens=getattr(usage, "candidates_token_count", None) if usage else None,
            finish_reason="stop",
            raw={"provider": "gemini"},
        )
