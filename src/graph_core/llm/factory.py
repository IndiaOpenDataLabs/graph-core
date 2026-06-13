"""LLM provider selection."""

from __future__ import annotations

import hashlib
import threading

from graph_core.config import settings
from graph_core.llm.interface import LLMProvider
from graph_core.llm.openai_provider import LocalEchoLLMProvider, OpenAILLMProvider
from graph_core.provider_base_url import normalize_provider_base_url

# Singleton cache: config hash -> provider instance.
_llm_provider_cache: dict[str, LLMProvider] = {}
_cache_lock = threading.Lock()


def _cache_key(
    provider_name: str,
    model: str,
    api_key: str | None,
    base_url: str | None,
    profile_id: str | None,
    max_concurrent_calls: int | None,
    max_output_tokens: int | None,
) -> str:
    raw = (
        f"{provider_name}|{model}|"
        f"{hashlib.sha256((api_key or '').encode()).hexdigest()}|"
        f"{base_url}|{profile_id}|{max_concurrent_calls}|{max_output_tokens}"
    )
    return hashlib.sha256(raw.encode()).hexdigest()


def get_llm_provider(
    *,
    provider_name: str | None = None,
    model: str | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
    profile_id: str | None = None,
    max_concurrent_calls: int | None = None,
    max_output_tokens: int | None = None,
) -> LLMProvider:
    provider_name = provider_name or settings.default_llm_provider
    model = model or settings.default_llm_model

    if provider_name == "local_echo":
        return LocalEchoLLMProvider()
    if provider_name == "openai":
        effective_api_key = api_key or settings.openai_api_key
        if not effective_api_key:
            return LocalEchoLLMProvider()

        key = _cache_key(
            provider_name,
            model,
            effective_api_key,
            normalize_provider_base_url(base_url or settings.openai_base_url),
            profile_id,
            max_concurrent_calls,
            max_output_tokens,
        )

        with _cache_lock:
            cached = _llm_provider_cache.get(key)
            if cached is not None:
                return cached

            instance = OpenAILLMProvider(
                api_key=effective_api_key,
                model=model,
                base_url=normalize_provider_base_url(
                    base_url or settings.openai_base_url
                ),
                profile_id=profile_id,
                max_concurrent_calls=max_concurrent_calls,
                max_output_tokens=max_output_tokens,
            )
            _llm_provider_cache[key] = instance
            return instance

    raise ValueError(f"Unsupported llm provider: {provider_name}")
