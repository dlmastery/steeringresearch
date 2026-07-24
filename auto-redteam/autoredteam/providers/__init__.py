"""providers -- the pluggable ModelProvider registry + factory.

Adding a new backend is a two-line change: write a class that satisfies
`interfaces.ModelProvider` (subclass `BaseProvider` for free timing/retry) and add
it to `PROVIDERS` below. The orchestrator only ever calls `get_provider(cfg)`.

Importing this package is CHEAP and side-effect-free: the concrete provider modules
import their heavy SDKs LAZILY (inside `_generate`), so `import autoredteam.providers`
works with only the core deps installed. `MockProvider` needs nothing beyond the
standard library and makes the whole harness runnable offline.
"""
from __future__ import annotations

from ..models import ModelConfig
from .anthropic import AnthropicProvider
from .base import BaseProvider, ProviderConfigError
from .gemini import GeminiProvider
from .local_gemma import LocalGemmaProvider
from .mock import MockProvider
from .openai_compat import OpenAICompatibleProvider

# Canonical provider key -> class. Keys match `ModelConfig.provider` values.
PROVIDERS: dict[str, type[BaseProvider]] = {
    MockProvider.provider_key: MockProvider,
    GeminiProvider.provider_key: GeminiProvider,
    OpenAICompatibleProvider.provider_key: OpenAICompatibleProvider,
    AnthropicProvider.provider_key: AnthropicProvider,
    LocalGemmaProvider.provider_key: LocalGemmaProvider,
}

# Friendly aliases so common spellings resolve to the canonical provider.
_ALIASES: dict[str, str] = {
    "openai": "openai_compatible",
    "openai_compat": "openai_compatible",
    "ollama": "local_gemma",
    "gemma": "local_gemma",
    "google": "gemini",
    "claude": "anthropic",
}


def get_provider(cfg: ModelConfig) -> BaseProvider:
    """Build the concrete provider for a `ModelConfig`.

    Dispatches on `cfg.provider` (case-insensitive, alias-aware). Unknown provider
    -> ValueError listing the registered names.
    """
    key = (cfg.provider or "").strip().lower()
    key = _ALIASES.get(key, key)
    cls = PROVIDERS.get(key)
    if cls is None:
        known = ", ".join(sorted(PROVIDERS)) + ", " + ", ".join(sorted(_ALIASES))
        raise ValueError(
            f"Unknown provider {cfg.provider!r}. Registered providers: {known}."
        )
    return cls(cfg)


__all__ = [
    "PROVIDERS",
    "get_provider",
    "BaseProvider",
    "ProviderConfigError",
    "MockProvider",
    "GeminiProvider",
    "OpenAICompatibleProvider",
    "AnthropicProvider",
    "LocalGemmaProvider",
]
