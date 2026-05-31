"""Embedding provider selection."""

from graph_core.config import settings
from graph_core.embedding.hash_provider import HashEmbeddingProvider
from graph_core.embedding.interface import EmbeddingProvider
from graph_core.embedding.openai_provider import OpenAIEmbeddingProvider
from graph_core.provider_base_url import normalize_provider_base_url


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
        return OpenAIEmbeddingProvider(
            api_key=effective_api_key,
            model=model,
            dimensions=dimensions,
            base_url=normalize_provider_base_url(
                base_url or settings.openai_base_url
            ),
            profile_id=profile_id,
            max_concurrent_calls=max_concurrent_calls,
        )

    raise ValueError(f"Unsupported embedding provider: {provider_name}")
