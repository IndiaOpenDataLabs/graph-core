"""Tests for ingestion queue cleanup."""

from __future__ import annotations

import uuid

import pytest
from dramatiq.message import Message

from graph_core.services.graph.ingestion import document_pipeline


class _FakeRedis:
    def __init__(self, hashes: dict[str, dict[str, bytes]], zsets: dict[str, dict[str, int]]):
        self.hashes = hashes
        self.zsets = zsets
        self.sets: dict[str, set[str]] = {}
        self.expirations: dict[str, int] = {}
        self.closed = False

    async def sadd(self, key_name, *members):  # noqa: ANN001
        table = self.sets.setdefault(key_name, set())
        added = 0
        for member in members:
            if member not in table:
                table.add(member)
                added += 1
        return added

    async def expire(self, key_name, ttl):  # noqa: ANN001
        self.expirations[key_name] = ttl
        return True

    async def sismember(self, key_name, member):  # noqa: ANN001
        return member in self.sets.get(key_name, set())

    async def scan(self, cursor=0, match=None, count=None):  # noqa: ANN001
        import fnmatch

        keys = [key for key in self.hashes if match is None or fnmatch.fnmatch(key, match)]
        return 0, [key.encode() for key in keys]

    async def hscan(self, key_name, cursor=0, count=None):  # noqa: ANN001
        return 0, {field.encode(): value for field, value in self.hashes.get(key_name, {}).items()}

    async def hdel(self, key_name, *fields):  # noqa: ANN001
        removed = 0
        table = self.hashes.get(key_name, {})
        for field in fields:
            if field in table:
                del table[field]
                removed += 1
        return removed

    async def zrem(self, key_name, *members):  # noqa: ANN001
        removed = 0
        table = self.zsets.get(key_name, {})
        for member in members:
            if member in table:
                del table[member]
                removed += 1
        return removed

    async def aclose(self):
        self.closed = True


@pytest.mark.asyncio
async def test_purge_queued_job_messages_removes_matching_dramatiq_entries(monkeypatch):
    target_job_id = uuid.uuid4()
    other_job_id = uuid.uuid4()

    target_message = Message(
        queue_name="ingestion_control",
        actor_name="run_ingestion",
        args=(str(target_job_id),),
        kwargs={},
        options={},
    ).encode()
    other_message = Message(
        queue_name="ingestion_control",
        actor_name="run_ingestion",
        args=(str(other_job_id),),
        kwargs={},
        options={},
    ).encode()

    fake_redis = _FakeRedis(
        hashes={
            "dramatiq:ingestion_control.XQ.msgs": {
                "target": target_message,
                "other": other_message,
            }
        },
        zsets={
            "dramatiq:ingestion_control.XQ": {
                "target": 1,
                "other": 2,
            }
        },
    )

    monkeypatch.setattr(document_pipeline.aioredis, "from_url", lambda *_args, **_kwargs: fake_redis)

    removed = await document_pipeline.purge_queued_job_messages([target_job_id])

    assert removed == 1
    assert "target" not in fake_redis.hashes["dramatiq:ingestion_control.XQ.msgs"]
    assert "target" not in fake_redis.zsets["dramatiq:ingestion_control.XQ"]
    assert "other" in fake_redis.hashes["dramatiq:ingestion_control.XQ.msgs"]
    assert "other" in fake_redis.zsets["dramatiq:ingestion_control.XQ"]
    assert str(target_job_id) in fake_redis.sets[document_pipeline._CANCELLED_JOBS_KEY]
    assert fake_redis.closed is True
