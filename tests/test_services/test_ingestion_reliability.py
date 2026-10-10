"""Offline coverage for chunk retries and transaction ownership; no model server."""

import inspect
import json
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from sqlalchemy import select
from sqlalchemy.exc import DBAPIError

from graph_core.models.chunk import IngestionChunk
from graph_core.models.graph_rag import GraphEntity
from graph_core.models.job import Job, JobEvent
from graph_core.services.graph.ingestion import chunk_status, document_pipeline
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


@pytest.mark.asyncio
@pytest.mark.parametrize("fail_at", [0, 1])
async def test_chunk_message_publish_failure_restores_unpublished_rows_and_bootstrap_retries(
    db_session,
    test_namespace,
    test_graph_rag_collection,
    monkeypatch,
    fail_at,
):
    job = await make_job(db_session, test_namespace.id, ["failed"] * 3)
    job.collection_id = test_graph_rag_collection.id
    await db_session.commit()
    job_id = job.id
    monkeypatch.setattr(ingestion.dispatch_retried_chunks, "send", Mock())
    await chunk_status.retry_failed_chunks(job_id, test_namespace.id)
    monkeypatch.setattr(
        document_pipeline, "is_job_cancelled", AsyncMock(return_value=False)
    )
    monkeypatch.setattr(
        document_pipeline, "_resolve_chunk_dispatch_limit", lambda *args: 3
    )
    monkeypatch.setattr(document_pipeline.settings, "llm_max_concurrent_calls", 3)
    reserve = AsyncMock(side_effect=[f"token-{i}" for i in range(6)])
    release = AsyncMock()
    monkeypatch.setattr(document_pipeline, "try_reserve_llm_call_slot", reserve)
    monkeypatch.setattr(document_pipeline, "release_llm_call_slot", release)
    monkeypatch.setattr(
        ingestion, "dispatch_pending_chunks", document_pipeline.dispatch_pending_chunks
    )
    successful_publications = []
    attempts = 0

    def publish(*args):
        nonlocal attempts
        attempt = attempts
        attempts += 1
        if attempt == fail_at:
            raise RuntimeError("chunk broker unavailable")
        successful_publications.append(args)

    monkeypatch.setattr(ingestion.run_chunk, "send", Mock(side_effect=publish))
    run = inspect.unwrap(ingestion.dispatch_retried_chunks.fn)
    with pytest.raises(RuntimeError, match="chunk broker unavailable"):
        await run(str(job_id))

    async def snapshot():
        async with document_pipeline.AsyncSessionLocal() as session:
            return (
                await session.scalars(
                    select(IngestionChunk)
                    .where(IngestionChunk.job_id == job_id)
                    .order_by(IngestionChunk.chunk_index)
                )
            ).all()

    rows = await snapshot()
    assert [c.status for c in rows] == ["processing"] * fail_at + ["pending"] * (
        3 - fail_at
    )
    for chunk in rows[fail_at:]:
        assert chunk.processing_started_at is None
        assert chunk.lease_expires_at is None
    assert [call.kwargs["token"] for call in release.await_args_list] == [
        f"token-{i}" for i in range(fail_at, 3)
    ]
    # This is Dramatiq's next bootstrap attempt: no lease expiry or manual
    # API intervention is needed, even when a previous message was published.
    await run(str(job_id))
    assert [args[1] for args in successful_publications] == [0, 1, 2]
    assert all(c.status == "processing" for c in await snapshot())
    assert release.await_count == 3 - fail_at
    if fail_at:
        assert successful_publications[0][-1] == "token-0"
        assert rows[0].lease_expires_at is not None


@pytest.mark.asyncio
async def test_publish_recovery_does_not_reset_terminal_or_reassigned_leases(
    db_session,
    test_namespace,
):
    job = await make_job(
        db_session,
        test_namespace.id,
        ["completed", "processing", "processing"],
        status="running",
    )
    old_lease = datetime.now(timezone.utc) + timedelta(minutes=1)
    newer_lease = old_lease + timedelta(minutes=1)
    rows = (
        await db_session.scalars(
            select(IngestionChunk)
            .where(IngestionChunk.job_id == job.id)
            .order_by(IngestionChunk.chunk_index)
        )
    ).all()
    rows[1].lease_expires_at = newer_lease
    rows[2].lease_expires_at = old_lease
    await db_session.commit()
    await document_pipeline._restore_unpublished_chunks(
        job.id,
        [
            (SimpleNamespace(id=row.id, lease_expires_at=old_lease), None)
            for row in rows
        ],
    )
    for row in rows:
        await db_session.refresh(row)
    assert [row.status for row in rows] == ["completed", "processing", "pending"]
    assert rows[1].lease_expires_at is not None
    assert rows[2].lease_expires_at is None


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
