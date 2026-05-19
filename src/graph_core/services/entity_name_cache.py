"""Redis-backed entity name → ID cache for deduplicating resolve_entity calls
across parallel chunk workers during a single ingest run.

Keys: ingest:{collection_id}:entity:{NormalizedName} → UUID string
TTL:  600s
"""

import uuid
from typing import Optional

import redis.asyncio as aioredis

from graph_core.config import settings


class EntityNameCache:
    """Shared entity name registry across parallel chunk workers.

    Before calling resolve_entity, check the cache. A hit means another worker
    already resolved this entity in this run. Only newly created entities are
    written (is_new=True), since pre-existing entities are found via DB alias
    lookup in resolve_entity anyway.
    """

    def __init__(self, collection_id: str, ttl: int = 600) -> None:
        url = settings.redis_semaphore_url
        self._redis = aioredis.from_url(url, decode_responses=True)
        self._prefix = f"ingest:{collection_id}:entity:"
        self._ttl = ttl

    def _key(self, name: str) -> str:
        return f"{self._prefix}{name.strip().title()}"

    async def get(self, name: str) -> Optional[uuid.UUID]:
        val = await self._redis.get(self._key(name))
        return uuid.UUID(val) if val else None

    async def set_many(self, names: list[str], entity_id: uuid.UUID) -> None:
        """Register multiple name variants via pipeline. SET NX — first writer wins."""
        pipe = self._redis.pipeline()
        for name in names:
            pipe.set(self._key(name), str(entity_id), nx=True, ex=self._ttl)
        await pipe.execute()
