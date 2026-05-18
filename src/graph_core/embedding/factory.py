"""Embedding provider selection."""

from graph_core.config import settings
from graph_core.embedding.hash_provider import HashEmbeddingProvider
from graph_core.embedding.interface import EmbeddingProvider


def get_embedding_provider() -> EmbeddingProvider:
    provider_name = settings.default_embedding_provider

    if provider_name == "local_hash":
        return HashEmbeddingProvider(dimensions=settings.default_embedding_dimensions)
    if provider_name == "openai":
        return HashEmbeddingProvider(dimensions=settings.default_embedding_dimensions)

    raise ValueError(f"Unsupported embedding provider: {provider_name}")
