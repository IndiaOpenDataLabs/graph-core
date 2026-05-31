"""OpenAI embedding provider."""

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

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
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

    async def embed_query(self, text: str) -> list[float]:
        async with embedding_call_slot(
            scope=self._profile_id,
            max_concurrent_calls=self._max_concurrent_calls,
        ):
            response = await self._client.embeddings.create(
                model=self._model,
                input=[text],
                dimensions=self._dimensions,
            )
        return list(response.data[0].embedding)

    @property
    def dimensions(self) -> int:
        return self._dimensions
