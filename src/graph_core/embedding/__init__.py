"""Embedding provider interfaces and implementations."""

from graph_core.embedding.factory import get_embedding_provider
from graph_core.embedding.hash_provider import HashEmbeddingProvider
from graph_core.embedding.interface import EmbeddingProvider
from graph_core.embedding.openai_provider import OpenAIEmbeddingProvider

__all__ = [
    "EmbeddingProvider",
    "HashEmbeddingProvider",
    "OpenAIEmbeddingProvider",
    "get_embedding_provider",
]
