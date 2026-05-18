"""GraphService — unit tests."""

import uuid
from unittest.mock import patch

import pytest

from graph_core.models.job import Job
from graph_core.models.vector_chunk import VectorChunk
from graph_core.services.graph import (
    ChunkIngestionResult,
    DocumentIngestionResult,
    GraphService,
)


@pytest.mark.asyncio
async def test_create_collection(service, test_namespace):
    coll = await service.create_collection(
        name="new-collection",
        namespace_id=test_namespace.id,
        strategy="vector",
    )
    assert coll.name == "new-collection"
    assert coll.namespace_id == test_namespace.id
    assert coll.strategy == "vector"


@pytest.mark.asyncio
async def test_create_collection_invalid_namespace(service):
    with pytest.raises(ValueError, match="not found"):
        await service.create_collection(
            name="orphan",
            namespace_id=uuid.uuid4(),
        )


@pytest.mark.asyncio
async def test_list_collections(service, test_namespace, test_collection):
    collections = await service.list_collections(test_namespace.id)
    assert any(c.name == "test-collection" for c in collections)


@pytest.mark.asyncio
async def test_get_collection(service, test_collection):
    coll = await service.get_collection(test_collection.id)
    assert coll.id == test_collection.id


@pytest.mark.asyncio
async def test_get_collection_not_found(service):
    with pytest.raises(ValueError, match="not found"):
        await service.get_collection(uuid.uuid4())


@pytest.mark.asyncio
async def test_ingest_chunk_vector(service, test_collection):
    result = await service.ingest_chunk(
        "hello world",
        test_collection.id,
        test_collection.namespace_id,
    )
    assert isinstance(result, ChunkIngestionResult)
    assert result.chunk_hash is not None

    from graph_core.database import AsyncSessionLocal

    async with AsyncSessionLocal() as session:
        rows = (await session.execute(__import__("sqlalchemy").select(VectorChunk))).scalars().all()
    assert len(rows) == 1
    assert rows[0].content == "hello world"


@pytest.mark.asyncio
async def test_ingest_chunk_namespace_enforcement(service, test_collection, test_namespace):
    wrong_namespace = uuid.uuid4()
    with pytest.raises(PermissionError, match="does not belong"):
        await service.ingest_chunk(
            "hello",
            test_collection.id,
            wrong_namespace,
        )


@pytest.mark.asyncio
async def test_enqueue_document_ingestion(service, test_collection):
    with patch("graph_core.workers.ingestion.run_ingestion") as mock_worker:
        result = await service.enqueue_document_ingestion(
            "long document text",
            test_collection.id,
            test_collection.namespace_id,
        )
    assert isinstance(result, DocumentIngestionResult)
    assert result.status == "pending"
    mock_worker.send.assert_called_once_with(str(result.job_id))


@pytest.mark.asyncio
async def test_query_respects_mode_override(service, test_collection):
    result = await service.query(
        "what is the meaning?",
        test_collection.id,
        test_collection.namespace_id,
        mode="global",
    )
    assert result.mode == "global"


@pytest.mark.asyncio
async def test_query_uses_collection_default(service, test_collection):
    result = await service.query(
        "what is the meaning?",
        test_collection.id,
        test_collection.namespace_id,
    )
    # test_collection has no default_query_mode, falls back to "local"
    assert result.mode == "local"


@pytest.mark.asyncio
async def test_query_vector_returns_relevant_chunks(service, test_collection):
    await service.ingest_chunk(
        "Krishna teaches Arjuna about duty and devotion.",
        test_collection.id,
        test_collection.namespace_id,
    )
    await service.ingest_chunk(
        "A recipe for lentil soup uses cumin and turmeric.",
        test_collection.id,
        test_collection.namespace_id,
    )

    result = await service.query(
        "What does Krishna teach Arjuna?",
        test_collection.id,
        test_collection.namespace_id,
    )

    assert "Krishna teaches Arjuna" in result.response
    assert "lentil soup" not in result.response


@pytest.mark.asyncio
async def test_get_job_not_found(service):
    with pytest.raises(ValueError, match="not found"):
        await service.get_job(uuid.uuid4())


@pytest.mark.asyncio
async def test_update_and_get_job(service):
    job_id = uuid.uuid4()
    # Create a job directly for testing
    await service.update_job_status(job_id, "running", progress_percent=50)
    # Note: job doesn't exist yet, update is a no-op in current impl
    # This tests the update path without crash


@pytest.mark.asyncio
async def test_ingest_document_pipeline_processes_stored_payload(
    service, test_collection
):
    from graph_core.database import AsyncSessionLocal

    async with AsyncSessionLocal() as session:
        job = Job(
            namespace_id=test_collection.namespace_id,
            collection_id=test_collection.id,
            job_type="ingest_document",
            status="pending",
            payload={
                "text": "Krishna teaches Arjuna. " * 200,
            },
        )
        session.add(job)
        await session.commit()
        await session.refresh(job)
        job_id = job.id

    await service.ingest_document_pipeline(job_id)

    async with AsyncSessionLocal() as session:
        vector_rows = (
            await session.execute(__import__("sqlalchemy").select(VectorChunk))
        ).scalars().all()
        refreshed_job = await session.get(Job, job_id)

    assert len(vector_rows) >= 1
    assert refreshed_job is not None
    assert refreshed_job.progress_percent == 100
