"""Redis-backed semaphores for provider call concurrency limits."""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import redis.asyncio as aioredis

from graph_core.config import settings

logger = logging.getLogger(__name__)

_ACQUIRE_SCRIPT = """
local key = KEYS[1]
local token = ARGV[1]
local now = tonumber(ARGV[2])
local expires_at = tonumber(ARGV[3])
local limit = tonumber(ARGV[4])

redis.call('ZREMRANGEBYSCORE', key, '-inf', now)
if redis.call('ZCARD', key) < limit then
    redis.call('ZADD', key, expires_at, token)
    return 1
end
return 0
"""

_RELEASE_SCRIPT = """
local key = KEYS[1]
local token = ARGV[1]
redis.call('ZREM', key, token)
return 1
"""


class _RedisSemaphore:
    def __init__(self, key: str, limit: int) -> None:
        self._key = key
        self._limit = limit
        self._lease_ms = max(settings.provider_semaphore_lease_seconds, 1) * 1000
        self._poll_seconds = max(settings.provider_semaphore_poll_interval_ms, 1) / 1000
        self._redis = aioredis.from_url(settings.redis_semaphore_url)

    async def acquire(self) -> str | None:
        if self._limit <= 0:
            return None
        token = str(uuid.uuid4())
        while True:
            now_ms = int(time.time() * 1000)
            acquired = await self._redis.eval(
                _ACQUIRE_SCRIPT,
                1,
                self._key,
                token,
                now_ms,
                now_ms + self._lease_ms,
                self._limit,
            )
            if acquired:
                return token
            await asyncio.sleep(self._poll_seconds)

    async def release(self, token: str | None) -> None:
        if self._limit <= 0 or not token:
            return
        try:
            await self._redis.eval(_RELEASE_SCRIPT, 1, self._key, token)
        except Exception:
            logger.exception("Failed to release provider semaphore %s", self._key)


_llm_semaphore = _RedisSemaphore(
    key="provider-semaphore:llm",
    limit=settings.llm_max_concurrent_calls,
)
_embedding_semaphore = _RedisSemaphore(
    key="provider-semaphore:embedding",
    limit=settings.embedding_max_concurrent_calls,
)


@asynccontextmanager
async def llm_call_slot() -> AsyncIterator[None]:
    token = await _llm_semaphore.acquire()
    try:
        yield
    finally:
        await _llm_semaphore.release(token)


@asynccontextmanager
async def embedding_call_slot() -> AsyncIterator[None]:
    token = await _embedding_semaphore.acquire()
    try:
        yield
    finally:
        await _embedding_semaphore.release(token)
