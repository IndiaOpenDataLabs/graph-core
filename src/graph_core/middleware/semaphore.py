"""Distributed semaphore middleware for Dramatiq.

Limits fleet-wide concurrency for chunk processing using Redis.
"""

from __future__ import annotations

import logging
import time

import dramatiq
import redis.asyncio as aioredis

from graph_core.config import settings

logger = logging.getLogger(__name__)


# Lua script: atomic semaphore acquire with timeout
_ACQUIRE_SCRIPT = """
local key = KEYS[1]
local value = ARGV[1]
local ttl = tonumber(ARGV[2])
local max_slots = tonumber(ARGV[3])

local current = redis.call('SCARD', key)
if current < max_slots then
    redis.call('SADD', key, value)
    redis.call('EXPIRE', key, ttl)
    return 1
end
return 0
"""

# Lua script: atomic semaphore release
_RELEASE_SCRIPT = """
local key = KEYS[1]
local value = ARGV[1]
redis.call('SREM', key, value)
"""


class DistributedSemaphore(dramatiq.Middleware):
    """Limits concurrent message processing fleet-wide using Redis sorted set."""

    def __init__(self, max_concurrent: int = 5) -> None:
        url = settings.redis_semaphore_url
        self._redis = aioredis.from_url(url)
        self._max = max_concurrent
        self._key = f"semaphore:run_chunk"
        self._ttl = 300  # 5 minutes per slot

    @property
    def actor_options(self):
        return {"semaphore"}

    async def _acquire(self, message_id: str) -> bool:
        result = await self._redis.eval(
            _ACQUIRE_SCRIPT, 1, self._key, message_id, self._ttl, self._max
        )
        return bool(result)

    async def _release(self, message_id: str) -> None:
        await self._redis.eval(_RELEASE_SCRIPT, 1, self._key, message_id)

    def before_process_message(self, worker, message):
        import asyncio
        loop = asyncio.get_event_loop()
        acquired = loop.run_until_complete(self._acquire(message.message_id))
        if not acquired:
            logger.info("Semaphore full, retiring message %s", message.message_id)
            raise dramatiq.Retire()

    def after_process_message(self, worker, message, *args):
        import asyncio
        loop = asyncio.get_event_loop()
        try:
            loop.run_until_complete(self._release(message.message_id))
        except Exception as e:
            logger.warning("Failed to release semaphore for %s: %s", message.message_id, e)

    def before_publish_message(self, broker, message):
        pass

    def after_publish_message(self, broker, message):
        pass
