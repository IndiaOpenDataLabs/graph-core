"""Offline coverage for chunk retries and transaction ownership; no model server."""

import inspect
import json
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from sqlalchemy import select
from sqlalchemy.exc import DBAPIError

from graph_core.models.chunk import IngestionChunk
from graph_core.models.graph_rag import GraphEntity
from graph_core.models.job import Job, JobEvent
from graph_core.services.graph.ingestion import chunk_status
from graph_core.services.graph_rag import entity_resolver
from graph_core.services.graph_rag.entity_resolver import (
    EntityResolutionResult,
    IncrementalEntityResolver,
)
from graph_core.storage import graph_rag_vectors
from graph_core.workers import ingestion


async def make_job(
    session, namespace_id, statuses, *, status="failed", job_type="ingest_document"
):
    job = Job(
        id=uuid.uuid4(),
        namespace_id=namespace_id,
        job_type=job_type,
        status=status,
        chunks_total=len(statuses),
        chunks_completed=len(statuses),
        error="old job error",
        completed_at=datetime.now(timezone.utc),
    )
    session.add(job)
    await session.flush()
    for i, state in enumerate(statuses):
        session.add(
            IngestionChunk(
                job_id=job.id,
                chunk_index=i,
                text="PRIVATE ORIGINAL DOCUMENT",
                status=state,
                error="chunk error" if state == "failed" else None,
                completed_at=datetime.now(timezone.utc),
            )
        )
    await session.commit()
    return job


@pytest.mark.asyncio
async def test_chunk_inspection_is_scoped_paginated_and_omits_source_text(
    db_session, test_namespace
):
    job = await make_job(
        db_session, test_namespace.id, ["completed", "failed", "failed"]
    )
    snapshot = await chunk_status.list_job_chunks(
        job.id, test_namespace.id, status="failed", offset=1, limit=1
    )
    assert snapshot["counts"]["completed"] == 1
    assert snapshot["counts"]["failed"] == 2
    assert snapshot["total"] == 3 and snapshot["filtered_total"] == 2
    assert snapshot["chunks"][0]["chunk_index"] == 2
    assert "PRIVATE ORIGINAL DOCUMENT" not in json.dumps(snapshot)
    with pytest.raises(LookupError):
        await chunk_status.list_job_chunks(job.id, uuid.uuid4())


@pytest.mark.asyncio
@pytest.mark.parametrize("indices,expected", [(None, [1, 2]), ([2], [2])])
async def test_retry_resets_only_selected_failures_and_preserves_successes(
    db_session,
    test_namespace,
    monkeypatch,
    indices,
    expected,
):
    job = await make_job(
        db_session, test_namespace.id, ["completed", "failed", "failed"]
    )
    send = Mock()
    monkeypatch.setattr(ingestion.dispatch_retried_chunks, "send", send)
    result = await chunk_status.retry_failed_chunks(job.id, test_namespace.id, indices)
    assert result["retried_chunks"] == expected
    send.assert_called_once_with(str(job.id))
    await db_session.refresh(job)
    rows = (
        await db_session.scalars(
            select(IngestionChunk)
            .where(IngestionChunk.job_id == job.id)
            .order_by(IngestionChunk.chunk_index)
        )
    ).all()
    assert rows[0].status == "completed" and rows[0].completed_at is not None
    assert job.status == "running" and job.error is None and job.completed_at is None
    assert job.chunks_completed == 3 - len(expected)
    for row in rows:
        if row.chunk_index in expected:
            assert (
                row.status == "pending"
                and row.error is None
                and row.completed_at is None
            )
        elif row.chunk_index:
            assert row.status == "failed" and row.error == "chunk error"
    event = await db_session.scalar(select(JobEvent).where(JobEvent.job_id == job.id))
    assert event.payload["chunk_indices"] == expected
    assert event.payload["previous_errors"] == {str(i): "chunk error" for i in expected}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "statuses,status,indices",
    [
        (["completed", "failed"], "failed", [0]),
        (["processing", "failed"], "running", None),
        (["failed"], "cancelled", None),
        (["pending", "failed"], "running", [1]),
        (["completed"], "completed", None),
    ],
)
async def test_invalid_retry_does_not_schedule_or_modify_rows(
    db_session,
    test_namespace,
    monkeypatch,
    statuses,
    status,
    indices,
):
    job = await make_job(db_session, test_namespace.id, statuses, status=status)
    send = Mock()
    monkeypatch.setattr(ingestion.dispatch_retried_chunks, "send", send)
    with pytest.raises(chunk_status.ChunkRetryConflict):
        await chunk_status.retry_failed_chunks(job.id, test_namespace.id, indices)
    send.assert_not_called()
    await db_session.refresh(job)
    assert job.status == status


@pytest.mark.asyncio
async def test_schedule_failure_leaves_saved_retry_resumable(
    db_session, test_namespace, monkeypatch
):
    job = await make_job(db_session, test_namespace.id, ["completed", "failed"])
    send = Mock(side_effect=RuntimeError("broker unavailable"))
    monkeypatch.setattr(ingestion.dispatch_retried_chunks, "send", send)
    with pytest.raises(RuntimeError, match="broker unavailable"):
        await chunk_status.retry_failed_chunks(job.id, test_namespace.id)
    await db_session.refresh(job)
    assert job.status == "running" and job.chunks_completed == 1
    send.side_effect = None
    result = await chunk_status.retry_failed_chunks(job.id, test_namespace.id)
    assert result["retried_chunks"] == []
    assert send.call_count == 2


@pytest.mark.asyncio
async def test_chunk_api_status_validation_and_tenant_isolation(
    async_client, db_session, test_namespace, monkeypatch
):
    job = await make_job(db_session, test_namespace.id, ["completed", "failed"])
    send = Mock()
    monkeypatch.setattr(ingestion.dispatch_retried_chunks, "send", send)
    response = await async_client.get(f"/jobs/{job.id}/chunks?status=failed")
    assert response.status_code == 200 and response.json()["filtered_total"] == 1
    for body in [{"chunk_indices": []}, {"chunk_indices": [-1]}]:
        assert (
            await async_client.post(f"/jobs/{job.id}/retry-failed-chunks", json=body)
        ).status_code == 422
    assert (
        await async_client.post(
            f"/jobs/{job.id}/retry-failed-chunks", json={"chunk_indices": [0]}
        )
    ).status_code == 409
    other = uuid.uuid4()
    assert (
        await async_client.get(
            f"/jobs/{job.id}/chunks", headers={"X-Namespace-ID": str(other)}
        )
    ).status_code == 404
    assert (
        await async_client.post(
            f"/jobs/{job.id}/retry-failed-chunks",
            json={},
            headers={"X-Namespace-ID": str(other)},
        )
    ).status_code == 404
    assert (
        await async_client.post(
            f"/jobs/{job.id}/retry-failed-chunks", json={"chunk_indices": [1]}
        )
    ).status_code == 202
    send.assert_called_once()


@pytest.mark.asyncio
@pytest.mark.parametrize("owned", [True, False])
@pytest.mark.parametrize("fail", [True, False])
async def test_embedding_upsert_flushes_parent_and_respects_session_ownership(
    monkeypatch, owned, fail
):
    events = []
    session = SimpleNamespace(
        flush=AsyncMock(side_effect=lambda: events.append("flush")),
        execute=AsyncMock(
            side_effect=RuntimeError("write failed")
            if fail
            else lambda *a: events.append("execute")
        ),
        commit=AsyncMock(),
        close=AsyncMock(),
    )
    factory = Mock(return_value=session)
    monkeypatch.setattr(graph_rag_vectors, "AsyncSessionLocal", factory)
    monkeypatch.setattr(
        graph_rag_vectors, "get_collection_dimensions", AsyncMock(return_value=3)
    )

    async def upsert():
        await graph_rag_vectors.GraphRAGVectorStore().upsert_relationship_embedding(
            uuid.uuid4(),
            uuid.uuid4(),
            "Source",
            "Target",
            "Fact",
            [0.1, 0.2, 0.3],
            session=None if owned else session,
        )

    if fail:
        with pytest.raises(RuntimeError, match="write failed"):
            await upsert()
    else:
        await upsert()
        assert events == ["flush", "execute"]
    assert session.commit.await_count == int(owned and not fail)
    assert session.close.await_count == int(owned)
    assert factory.call_count == int(owned)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "statuses,status,capacity,waits",
    [
        (["pending"], "running", False, True),
        (["pending", "processing"], "running", False, False),
        (["pending"], "running", True, False),
        (["pending"], "failed", False, False),
    ],
)
async def test_retry_dispatch_waits_only_when_pending_work_cannot_start(
    db_session,
    test_namespace,
    monkeypatch,
    statuses,
    status,
    capacity,
    waits,
):
    job = await make_job(db_session, test_namespace.id, statuses, status=status)
    dispatch = AsyncMock(return_value=capacity)
    monkeypatch.setattr(ingestion, "dispatch_pending_chunks", dispatch)
    run = inspect.unwrap(ingestion.dispatch_retried_chunks.fn)
    if waits:
        with pytest.raises(RuntimeError, match="Waiting for provider capacity"):
            await run(str(job.id))
    else:
        await run(str(job.id))
    assert dispatch.await_count == int(status == "running")


class DatabaseFailure(Exception):
    def __init__(self, code):
        self.sqlstate = code


def resolver(collection_id):
    return IncrementalEntityResolver(
        SimpleNamespace(
            dimensions=3, embed_query=AsyncMock(return_value=[0.1, 0.2, 0.3])
        ),
        collection_id,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "code,failures,attempts", [("40P01", 1, 2), ("40P01", 3, 3), ("23505", 1, 1)]
)
async def test_entity_retries_only_deadlocks_with_fresh_sessions(
    monkeypatch, code, failures, attempts
):
    caller = SimpleNamespace(bind=object(), commit=AsyncMock())
    sessions = []

    @asynccontextmanager
    async def factory(**kwargs):
        assert kwargs["bind"] is caller.bind
        caller.commit.assert_awaited_once()
        session = SimpleNamespace(commit=AsyncMock(), rollback=AsyncMock())
        sessions.append(session)
        yield session

    sleep = AsyncMock()
    monkeypatch.setattr(entity_resolver, "AsyncSessionLocal", factory)
    monkeypatch.setattr(entity_resolver.asyncio, "sleep", sleep)
    r = resolver(uuid.uuid4())
    expected = EntityResolutionResult(True, uuid.uuid4(), "Entity")
    failure = DBAPIError("UPDATE", {}, DatabaseFailure(code))
    r._resolve_entity_once = AsyncMock(side_effect=[failure] * failures + [expected])
    if failures >= 3 or code != "40P01":
        with pytest.raises(DBAPIError):
            await r.resolve_entity(caller, "Entity", "CONCEPT", "Fact", "chunk")
    else:
        assert (
            await r.resolve_entity(caller, "Entity", "CONCEPT", "Fact", "chunk")
            == expected
        )
    assert len(sessions) == attempts
    assert len({id(s) for s in sessions}) == attempts
    assert sleep.await_count == attempts - 1
    for i, session in enumerate(sessions):
        assert session.rollback.await_count == int(i < failures)
        assert session.commit.await_count == int(i >= failures)


@pytest.mark.asyncio
async def test_deadlock_rollback_preserves_previously_committed_work(
    db_session, test_graph_rag_collection, monkeypatch
):
    collection_id = test_graph_rag_collection.id
    earlier = GraphEntity(
        id=uuid.uuid4(),
        collection_id=collection_id,
        canonical_name="Earlier",
        primary_type="CONCEPT",
    )
    db_session.add(earlier)
    r = resolver(collection_id)
    attempts = 0

    async def once(session, *args):
        nonlocal attempts
        attempts += 1
        entity = GraphEntity(
            id=uuid.uuid4(),
            collection_id=collection_id,
            canonical_name="Rolled back" if attempts == 1 else "Retried",
            primary_type="CONCEPT",
        )
        session.add(entity)
        await session.flush()
        if attempts == 1:
            raise DBAPIError("UPDATE", {}, DatabaseFailure("40P01"))
        return EntityResolutionResult(True, entity.id, entity.canonical_name)

    monkeypatch.setattr(r, "_resolve_entity_once", once)
    monkeypatch.setattr(entity_resolver.asyncio, "sleep", AsyncMock())
    await r.resolve_entity(db_session, "Retried", "CONCEPT", "Fact", "chunk")
    rows = (
        await db_session.scalars(
            select(GraphEntity).where(GraphEntity.collection_id == collection_id)
        )
    ).all()
    assert {row.canonical_name for row in rows} == {"Earlier", "Retried"}


@pytest.mark.asyncio
async def test_new_relationship_embedding_uses_caller_session_and_finishes_commit():
    r = resolver(uuid.uuid4())
    session = SimpleNamespace(
        execute=AsyncMock(
            return_value=SimpleNamespace(scalar_one_or_none=lambda: None)
        ),
        get=AsyncMock(return_value=SimpleNamespace(canonical_name="Endpoint")),
        add=Mock(),
        commit=AsyncMock(),
    )
    r._resolve_rel_type = AsyncMock(
        return_value=SimpleNamespace(
            canonical_type="CONNECTS_TO", relationship_type_id=uuid.uuid4()
        )
    )
    r._vstore.upsert_relationship_embedding = AsyncMock()
    await r.resolve_relationship(
        session, uuid.uuid4(), uuid.uuid4(), "Fact", [], 1.0, "chunk"
    )
    assert (
        r._vstore.upsert_relationship_embedding.await_args.kwargs["session"] is session
    )
    assert session.commit.await_count == 2
