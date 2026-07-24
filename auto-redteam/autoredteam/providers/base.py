"""base.py -- the shared machinery every concrete provider reuses.

A `ModelProvider` (see `interfaces.ModelProvider`) is just transport: it turns a
list of `Message`s into one `GenerationResult`. Everything that is the SAME across
providers -- timing, bounded retry with exponential backoff, message-shape
conversion, secret-safe config reporting -- lives here so each concrete provider
only has to implement the one call that actually talks to its backend.

Design (template-method pattern):

    BaseProvider.generate(messages, **kw)      # public: times + retries
        -> self._generate(messages, **kw)      # subclass: the real backend call

Subclasses override `_generate` only. `MockProvider` overrides it with a canned,
deterministic response; the real providers override it with a LAZY SDK call so the
module still imports when the SDK extra is not installed.

SAFETY: `get_config()` never returns the API key. Keys are read from the process
environment by NAME (`cfg.api_key_env`) at call time and are never stored on the
instance, logged, or echoed back.
"""
from __future__ import annotations

import asyncio
import os
import time
from typing import Any

from ..models import GenerationResult, Message, ModelConfig, Role


class ProviderConfigError(RuntimeError):
    """A misconfiguration that retrying cannot fix (missing SDK, missing key, ...).

    Raised eagerly and NOT retried by `BaseProvider.generate` -- backing off and
    trying again would only waste time when the env var is simply not set.
    """


# Roles that map straight through to the OpenAI-style chat schema.
_ROLE_TO_STR: dict[Role, str] = {
    Role.SYSTEM: "system",
    Role.USER: "user",
    Role.ASSISTANT: "assistant",
    Role.TOOL: "tool",
}


class BaseProvider:
    """Common base for all providers. Satisfies `interfaces.ModelProvider`.

    Subclasses set `provider_key` (for the registry) and implement `_generate`.
    """

    provider_key: str = "base"

    def __init__(self, cfg: ModelConfig) -> None:
        self.cfg = cfg
        # A stable, human-readable id for logs/manifests, e.g. "gemini:gemini-1.5-flash".
        self.name: str = f"{cfg.provider}:{cfg.model}"
        # Retry knobs are overridable per-model via `extra` without touching code.
        self.max_retries: int = int(cfg.extra.get("max_retries", 3))
        self.backoff_base_s: float = float(cfg.extra.get("backoff_base_s", 0.5))

    # ---- secret-safe manifest ------------------------------------------- #
    def get_config(self) -> dict[str, Any]:
        """Non-secret config for the run manifest.

        Returns provider/model/temperature (and other harmless fields). NEVER the
        API key -- only the env var NAME, which is not itself a secret.
        """
        return {
            "name": self.name,
            "provider": self.cfg.provider,
            "model": self.cfg.model,
            "temperature": self.cfg.temperature,
            "max_tokens": self.cfg.max_tokens,
            "base_url": self.cfg.base_url,
            "api_key_env": self.cfg.api_key_env,   # the NAME only, never the value
        }

    # ---- public template method ----------------------------------------- #
    async def generate(self, messages: list[Message], **kwargs: Any) -> GenerationResult:
        """Time + bounded-retry wrapper around the subclass `_generate`.

        Config errors (`ProviderConfigError`, `ImportError`) fail fast -- they are
        not transient. Anything else (network blips, rate limits) is retried with
        exponential backoff up to `max_retries`.
        """
        last_exc: Exception | None = None
        for attempt in range(max(1, self.max_retries)):
            t0 = time.perf_counter()
            try:
                result = await self._generate(messages, **kwargs)
                if result.latency_s is None:
                    result.latency_s = time.perf_counter() - t0
                return result
            except (ProviderConfigError, ImportError):
                # Not transient -- surface immediately so the operator can fix config.
                raise
            except Exception as exc:  # noqa: BLE001 -- providers raise SDK-specific types
                last_exc = exc
                if attempt >= self.max_retries - 1:
                    break
                await asyncio.sleep(self.backoff_base_s * (2 ** attempt))
        assert last_exc is not None
        raise last_exc

    async def _generate(self, messages: list[Message], **kwargs: Any) -> GenerationResult:
        """The one backend call. Subclasses MUST override."""
        raise NotImplementedError

    # ---- shared helpers for subclasses ---------------------------------- #
    def _require_api_key(self) -> str:
        """Read the key from `os.environ[cfg.api_key_env]` or raise a clear error.

        The key is returned to the local call frame only; it is never stored on the
        instance nor included in `get_config()`.
        """
        env_name = self.cfg.api_key_env
        if not env_name:
            raise ProviderConfigError(
                f"{self.name}: no api_key_env configured -- set `api_key_env` to the "
                f"NAME of the environment variable holding the key."
            )
        key = os.environ.get(env_name)
        if not key:
            raise ProviderConfigError(
                f"{self.name}: environment variable {env_name!r} is not set."
            )
        return key

    @staticmethod
    def _to_openai_messages(messages: list[Message]) -> list[dict[str, str]]:
        """Flatten to the OpenAI-style `[{role, content}, ...]` list."""
        return [{"role": _ROLE_TO_STR[m.role], "content": m.content} for m in messages]

    @staticmethod
    def _split_system(messages: list[Message]) -> tuple[str, list[Message]]:
        """Split SYSTEM messages (joined into one string) from the rest.

        Anthropic and Gemini take the system instruction as a separate argument
        rather than as a chat turn, so those providers use this helper.
        """
        system_parts = [m.content for m in messages if m.role is Role.SYSTEM]
        chat = [m for m in messages if m.role is not Role.SYSTEM]
        return ("\n\n".join(system_parts), chat)
