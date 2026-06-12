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
    def __init__(self, key_prefix: str) -> None:
        self._key_prefix = key_prefix
        self._lease_ms = max(settings.provider_semaphore_lease_seconds, 1) * 1000
        self._poll_seconds = max(settings.provider_semaphore_poll_interval_ms, 1) / 1000
        self._redis = aioredis.from_url(settings.redis_semaphore_url)

    async def acquire(self, scope: str, limit: int) -> str | None:
        if limit <= 0:
            return None
        token = str(uuid.uuid4())
        key = f"{self._key_prefix}:{scope}"
        timeout = max(settings.provider_semaphore_acquire_timeout_seconds, 1)
        deadline = time.monotonic() + timeout
        while True:
            now_ms = int(time.time() * 1000)
            acquired = await self._redis.eval(
                _ACQUIRE_SCRIPT,
                1,
                key,
                token,
                now_ms,
                now_ms + self._lease_ms,
                limit,
            )
            if acquired:
                return token
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(
                    f"Timed out after {timeout}s waiting for provider semaphore "
                    f"slot on {key} (limit={limit})"
                )
            wait = min(self._poll_seconds, remaining)
            await asyncio.sleep(wait)

    async def release(self, scope: str, token: str | None, limit: int) -> None:
        if limit <= 0 or not token:
            return
        key = f"{self._key_prefix}:{scope}"
        try:
            await self._redis.eval(_RELEASE_SCRIPT, 1, key, token)
        except Exception:
            logger.exception("Failed to release provider semaphore %s", key)

_llm_semaphore = _RedisSemaphore(key_prefix="provider-semaphore:llm")
_embedding_semaphore = _RedisSemaphore(key_prefix="provider-semaphore:embedding")
_llm_chunk_semaphore = _RedisSemaphore(key_prefix="ingestion-semaphore:llm")


@asynccontextmanager
async def llm_call_slot(
    scope: str | None = None,
    max_concurrent_calls: int | None = None,
) -> AsyncIterator[None]:
    limit = (
        max_concurrent_calls
        if max_concurrent_calls is not None
        else settings.llm_max_concurrent_calls
    )
    semaphore_scope = scope or "default"
    token = await _llm_semaphore.acquire(semaphore_scope, limit)
    try:
        yield
    finally:
        await _llm_semaphore.release(semaphore_scope, token, limit)


@asynccontextmanager
async def embedding_call_slot(
    scope: str | None = None,
    max_concurrent_calls: int | None = None,
) -> AsyncIterator[None]:
    limit = (
        max_concurrent_calls
        if max_concurrent_calls is not None
        else settings.embedding_max_concurrent_calls
    )
    semaphore_scope = scope or "default"
    token = await _embedding_semaphore.acquire(semaphore_scope, limit)
    try:
        yield
    finally:
        await _embedding_semaphore.release(semaphore_scope, token, limit)


@asynccontextmanager
async def llm_chunk_slot(
    scope: str | None = None,
    max_concurrent_calls: int | None = None,
) -> AsyncIterator[None]:
    limit = (
        max_concurrent_calls
        if max_concurrent_calls is not None
        else settings.llm_max_concurrent_calls
    )
    semaphore_scope = scope or "default"
    token = await _llm_chunk_semaphore.acquire(semaphore_scope, limit)
    try:
        yield
    finally:
        await _llm_chunk_semaphore.release(semaphore_scope, token, limit)
