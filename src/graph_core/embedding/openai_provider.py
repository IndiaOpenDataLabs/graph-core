"""OpenAI embedding provider."""

import math

from openai import AsyncOpenAI

from graph_core.embedding.interface import EmbeddingProvider
from graph_core.provider_semaphore import embedding_call_slot


class OpenAIEmbeddingProvider(EmbeddingProvider):
    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        dimensions: int,
        base_url: str | None = None,
        profile_id: str | None = None,
        max_concurrent_calls: int | None = None,
    ):
        self._client = AsyncOpenAI(api_key=api_key, base_url=base_url)
        self._model = model
        self._dimensions = dimensions
        self._profile_id = profile_id
        self._max_concurrent_calls = max_concurrent_calls

    @staticmethod
    def _is_finite_embedding(embedding: list[float]) -> bool:
        return bool(embedding) and all(math.isfinite(float(v)) for v in embedding)

    async def _embed_many(self, texts: list[str]) -> list[list[float]]:
        async with embedding_call_slot(
            scope=self._profile_id,
            max_concurrent_calls=self._max_concurrent_calls,
        ):
            response = await self._client.embeddings.create(
                model=self._model,
                input=texts,
                dimensions=self._dimensions,
            )
        return [list(item.embedding) for item in response.data]

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        embeddings = await self._embed_many(texts)
        fixed: list[list[float]] = []
        for text, embedding in zip(texts, embeddings, strict=True):
            if self._is_finite_embedding(embedding):
                fixed.append(embedding)
                continue
            retried = await self._embed_many([" " + text])
            retry_embedding = retried[0]
            if not self._is_finite_embedding(retry_embedding):
                raise ValueError("Embedding contains non-finite values")
            fixed.append(retry_embedding)
        return fixed

    async def embed_query(self, text: str) -> list[float]:
        embedding = (await self._embed_many([text]))[0]
        if self._is_finite_embedding(embedding):
            return embedding
        retried = (await self._embed_many([" " + text]))[0]
        if not self._is_finite_embedding(retried):
            raise ValueError("Embedding contains non-finite values")
        return retried

    @property
    def dimensions(self) -> int:
        return self._dimensions
