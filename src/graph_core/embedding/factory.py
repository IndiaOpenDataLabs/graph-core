"""Embedding provider selection."""

from __future__ import annotations

import hashlib
import threading
from typing import Any

from graph_core.config import settings
from graph_core.embedding.hash_provider import HashEmbeddingProvider
from graph_core.embedding.interface import EmbeddingProvider
from graph_core.embedding.openai_provider import OpenAIEmbeddingProvider
from graph_core.provider_base_url import normalize_provider_base_url

# Singleton cache: config hash -> provider instance.
# Reuses AsyncOpenAI clients (and their underlying httpx connection pools)
# across queries instead of creating a new client per call.
_embedding_provider_cache: dict[str, EmbeddingProvider] = {}
_cache_lock = threading.Lock()


def _cache_key(
    provider_name: str,
    model: str,
    dimensions: int,
    api_key: str | None,
    base_url: str | None,
    profile_id: str | None,
    max_concurrent_calls: int | None,
) -> str:
    """Produce a deterministic cache key from provider configuration.

    The raw api_key is hashed so it never appears in the dict keys,
    but identical keys still map to the same cached client.
    """
    raw = (
        f"{provider_name}|{model}|{dimensions}|"
        f"{hashlib.sha256((api_key or '').encode()).hexdigest()}|"
        f"{base_url}|{profile_id}|{max_concurrent_calls}"
    )
    return hashlib.sha256(raw.encode()).hexdigest()


def get_embedding_provider(
    *,
    provider_name: str | None = None,
    model: str | None = None,
    dimensions: int | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
    profile_id: str | None = None,
    max_concurrent_calls: int | None = None,
) -> EmbeddingProvider:
    provider_name = provider_name or settings.default_embedding_provider
    model = model or settings.default_embedding_model
    dimensions = dimensions or settings.default_embedding_dimensions

    if provider_name == "local_hash":
        return HashEmbeddingProvider(dimensions=dimensions)
    if provider_name == "openai":
        effective_api_key = api_key or settings.openai_api_key
        if not effective_api_key:
            return HashEmbeddingProvider(dimensions=dimensions)

        key = _cache_key(
            provider_name,
            model,
            dimensions,
            effective_api_key,
            normalize_provider_base_url(base_url or settings.openai_base_url),
            profile_id,
            max_concurrent_calls,
        )

        with _cache_lock:
            cached = _embedding_provider_cache.get(key)
            if cached is not None:
                return cached

            instance = OpenAIEmbeddingProvider(
                api_key=effective_api_key,
                model=model,
                dimensions=dimensions,
                base_url=normalize_provider_base_url(
                    base_url or settings.openai_base_url
                ),
                profile_id=profile_id,
                max_concurrent_calls=max_concurrent_calls,
            )
            _embedding_provider_cache[key] = instance
            return instance

    raise ValueError(f"Unsupported embedding provider: {provider_name}")
