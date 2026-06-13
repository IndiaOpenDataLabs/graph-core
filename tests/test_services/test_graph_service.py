"""GraphService — unit tests."""

import uuid
from unittest.mock import AsyncMock
from unittest.mock import patch

import pytest
from sqlalchemy import select

from graph_core.database import AsyncSessionLocal
from graph_core.models.collection import Collection
from graph_core.models.ingestion import IngestionRecord
from graph_core.models.job import Job
from graph_core.models.profile import Profile
from graph_core.services.graph import (
    ChunkIngestionResult,
    DocumentIngestionResult,
    GraphService,
)
from graph_core.services.graph.ingestion.document_pipeline import (
    process_single_chunk,
    update_chunk_status,
)


def _has_pgvector_tables() -> bool:
    """Check if pgvector tables are available (Postgres, not SQLite)."""
    from graph_core.database import engine
    return "postgresql" in engine.url.drivername


@pytest.mark.asyncio
async def test_create_collection(service, test_namespace):
    async with AsyncSessionLocal() as session:
        profile = Profile(
            namespace_id=test_namespace.id,
            kind="embedding",
            provider="local_hash",
            model="hash-256",
            dimensions=16,
            distance_metric="cosine",
        )
        session.add(profile)
        await session.commit()
        await session.refresh(profile)

    coll = await service.create_collection(
        name="new-collection",
        namespace_id=test_namespace.id,
        strategy="vector",
        embedding_profile_id=profile.id,
    )
    assert coll.name == "new-collection"
    assert coll.namespace_id == test_namespace.id
    assert coll.strategy == "vector"


@pytest.mark.asyncio
async def test_update_collection_renames_meta_collection(service, test_namespace):
    base_id = uuid.uuid4()
    meta_id = uuid.uuid4()
    async with AsyncSessionLocal() as session:
        session.add(
            Collection(
                id=base_id,
                namespace_id=test_namespace.id,
                name="base",
                strategy="vector",
                embedding_dimensions=256,
            )
        )
        session.add(
            Collection(
                id=meta_id,
                namespace_id=test_namespace.id,
                name="base__meta",
                strategy="custom_graph_rag",
                embedding_dimensions=256,
            )
        )
        await session.commit()

    service._migrate_collection_graph_if_needed = AsyncMock()  # type: ignore[method-assign]

    updated = await service.update_collection(
        base_id,
        test_namespace.id,
        name="renamed",
    )

    assert updated.name == "renamed"
    async with AsyncSessionLocal() as session:
        meta = await session.get(Collection, meta_id)
    assert meta is not None
    assert meta.name == "renamed__meta__l1"


@pytest.mark.asyncio
async def test_update_collection_rejects_direct_meta_rename(service, test_namespace):
    meta_id = uuid.uuid4()
    async with AsyncSessionLocal() as session:
        session.add(
            Collection(
                id=meta_id,
                namespace_id=test_namespace.id,
                name="base__meta__l1",
                strategy="custom_graph_rag",
                embedding_dimensions=256,
            )
        )
        await session.commit()

    with pytest.raises(ValueError, match="cannot be renamed directly"):
        await service.update_collection(
            meta_id,
            test_namespace.id,
            name="renamed-meta",
        )


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
@pytest.mark.skipif(not _has_pgvector_tables(), reason="requires Postgres pgvector")
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
        rows = (await session.execute(select(IngestionRecord))).scalars().all()
    assert len(rows) == 1
    assert rows[0].chunk_hash == result.chunk_hash


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
@pytest.mark.skipif(not _has_pgvector_tables(), reason="requires Postgres pgvector")
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
@pytest.mark.skipif(not _has_pgvector_tables(), reason="requires Postgres pgvector")
async def test_query_respects_mode_override(service, test_collection):
    result = await service.query(
        "what is the meaning?",
        test_collection.id,
        test_collection.namespace_id,
        mode="global",
    )
    assert result.mode == "global"


@pytest.mark.asyncio
@pytest.mark.skipif(not _has_pgvector_tables(), reason="requires Postgres pgvector")
async def test_query_uses_collection_default(service, test_collection):
    result = await service.query(
        "what is the meaning?",
        test_collection.id,
        test_collection.namespace_id,
    )
    # test_collection has no default_query_mode, falls back to "local"
    assert result.mode == "local"


@pytest.mark.asyncio
@pytest.mark.skipif(not _has_pgvector_tables(), reason="requires Postgres pgvector")
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
@pytest.mark.skipif(not _has_pgvector_tables(), reason="requires Postgres pgvector")
async def test_collection_embedding_profile_is_used(service, test_namespace):
    from graph_core.database import AsyncSessionLocal

    async with AsyncSessionLocal() as session:
        profile = Profile(
            namespace_id=test_namespace.id,
            kind="embedding",
            provider="local_hash",
            model="hash-256",
            dimensions=16,
            distance_metric="cosine",
        )
        session.add(profile)
        await session.commit()
        await session.refresh(profile)

    collection = await service.create_collection(
        name="profile-bound",
        namespace_id=test_namespace.id,
        strategy="vector",
        embedding_profile_id=profile.id,
    )

    result = await service.ingest_chunk(
        "Rama walks through the forest with Sita.",
        collection.id,
        test_namespace.id,
    )
    assert result.chunk_hash


@pytest.mark.asyncio
async def test_get_job_not_found(service):
    with pytest.raises(ValueError, match="not found"):
        await service.get_job(uuid.uuid4())


@pytest.mark.asyncio
async def test_resolve_enhance_region_batch_size_uses_profile_min(
    service, test_namespace
):
    async with AsyncSessionLocal() as session:
        embedding_profile = Profile(
            namespace_id=test_namespace.id,
            kind="embedding",
            provider="local_hash",
            model="hash-256",
            dimensions=16,
            distance_metric="cosine",
            max_concurrent_calls=7,
        )
        llm_profile = Profile(
            namespace_id=test_namespace.id,
            kind="llm",
            provider="local_echo",
            model="echo-v1",
            max_concurrent_calls=3,
        )
        session.add_all([embedding_profile, llm_profile])
        await session.commit()
        await session.refresh(embedding_profile)
        await session.refresh(llm_profile)
        collection = Collection(
            id=uuid.uuid4(),
            namespace_id=test_namespace.id,
            name="base",
            strategy="custom_graph_rag",
            embedding_dimensions=256,
            embedding_profile_id=embedding_profile.id,
            llm_profile_id=llm_profile.id,
        )
        session.add(collection)
        await session.commit()
        await session.refresh(collection)

    batch_size = await service._resolve_enhance_region_batch_size(collection)
    assert batch_size == 3


@pytest.mark.asyncio
async def test_enhance_stops_when_no_candidate_regions(test_namespace):
    service = GraphService()
    base_collection = Collection(
        id=uuid.uuid4(),
        namespace_id=test_namespace.id,
        name="base",
        strategy="custom_graph_rag",
        embedding_dimensions=256,
    )
    service.get_collection = AsyncMock(return_value=base_collection)  # type: ignore[method-assign]
    service._resolve_collection_llm_provider = AsyncMock(return_value=None)  # type: ignore[method-assign]
    service._resolve_enhance_region_batch_size = AsyncMock(return_value=1)  # type: ignore[method-assign]
    service._get_collection_by_names = AsyncMock(return_value=None)  # type: ignore[method-assign]
    service.create_collection = AsyncMock()  # type: ignore[method-assign]
    service._materialize_meta_collection = AsyncMock()  # type: ignore[method-assign]

    with patch(
        "graph_core.services.graph.analyze_collection_graph",
        AsyncMock(return_value={"totals": {}, "role_groups": []}),
    ), patch(
        "graph_core.services.graph.build_collection_understanding",
        AsyncMock(
            return_value={
                "nodes": [],
                "edges": [],
                "chunks": [],
                "candidate_region_count": 0,
            }
        ),
    ):
        with pytest.raises(
            ValueError,
            match="No candidate regions found for further enhancement",
        ):
            await service.build_collection_understanding(
                base_collection.id,
                test_namespace.id,
                levels=100,
            )

    service.create_collection.assert_not_awaited()
    service._materialize_meta_collection.assert_not_awaited()


@pytest.mark.asyncio
async def test_enhance_stops_after_single_concept_level(test_namespace):
    service = GraphService()
    base_collection = Collection(
        id=uuid.uuid4(),
        namespace_id=test_namespace.id,
        name="base",
        strategy="custom_graph_rag",
        embedding_dimensions=256,
        embedding_profile_id=uuid.uuid4(),
    )
    level_one = Collection(
        id=uuid.uuid4(),
        namespace_id=test_namespace.id,
        name="base__meta__l1",
        strategy="custom_graph_rag",
        embedding_dimensions=256,
    )
    service.get_collection = AsyncMock(return_value=base_collection)  # type: ignore[method-assign]
    service._resolve_collection_llm_provider = AsyncMock(return_value=None)  # type: ignore[method-assign]
    service._resolve_enhance_region_batch_size = AsyncMock(return_value=1)  # type: ignore[method-assign]
    service._get_collection_by_names = AsyncMock(return_value=None)  # type: ignore[method-assign]
    service._prepare_meta_collection = AsyncMock(return_value=level_one)  # type: ignore[method-assign]
    service._resolve_collection_embedding_provider = AsyncMock(return_value=object())  # type: ignore[method-assign]
    service._graph_storage = lambda collection: object()  # type: ignore[method-assign]
    service._materialize_region_concept = AsyncMock()  # type: ignore[method-assign]
    service._materialize_meta_edges = AsyncMock()  # type: ignore[method-assign]
    service._graph_name = lambda collection: f"graph_{collection.name}"  # type: ignore[method-assign]

    async def _fake_build_collection_understanding(*args, **kwargs):
        on_region_concept = kwargs.get("on_region_concept")
        if on_region_concept is not None:
            await on_region_concept(
                {"region_id": "role_group_1", "source_ids": ["node-1"]},
                {
                    "label": "concept-1",
                    "concept_type": "derived_concept",
                    "description": "concept description",
                    "aliases": [],
                    "importance_reason": "important",
                    "member_entity_names": [],
                    "evidence_region_ids": ["role_group_1"],
                },
            )
        return {
            "nodes": [{"id": "concept-1", "type": "derived_concept"}],
            "edges": [],
            "chunks": [],
            "candidate_region_count": 1,
        }

    with patch(
        "graph_core.services.graph.analyze_collection_graph",
        AsyncMock(return_value={"totals": {}, "role_groups": [{}]}),
    ), patch(
        "graph_core.services.graph.build_collection_understanding",
        AsyncMock(side_effect=_fake_build_collection_understanding),
    ):
        result = await service.build_collection_understanding(
            base_collection.id,
            test_namespace.id,
            levels=100,
        )

    service._prepare_meta_collection.assert_awaited_once()
    service._materialize_region_concept.assert_awaited_once()
    service._materialize_meta_edges.assert_awaited_once()
    assert len(result["generated_levels"]) == 1
    assert result["generated_levels"][0]["level"] == 1
    assert result["generated_levels"][0]["node_count"] == 1


@pytest.mark.asyncio
async def test_update_and_get_job(service):
    job_id = uuid.uuid4()
    # Create a job directly for testing
    await service.update_job_status(job_id, "running", progress_percent=50)
    # Note: job doesn't exist yet, update is a no-op in current impl
    # This tests the update path without crash


@pytest.mark.asyncio
async def test_process_single_chunk_ignores_missing_chunk_row(service, test_collection):
    async with AsyncSessionLocal() as session:
        job = Job(
            namespace_id=test_collection.namespace_id,
            collection_id=test_collection.id,
            job_type="ingest_document",
            status="running",
            payload={"text": "ignored"},
        )
        session.add(job)
        await session.commit()
        await session.refresh(job)

    await process_single_chunk(str(job.id), 99)


@pytest.mark.asyncio
async def test_update_chunk_status_ignores_missing_chunk_row(service):
    await update_chunk_status(uuid.uuid4(), 42, "failed", error="stale")


@pytest.mark.asyncio
@pytest.mark.skipif(not _has_pgvector_tables(), reason="requires Postgres pgvector")
async def test_ingest_document_pipeline_processes_stored_payload(
    service, test_collection
):
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
        ingestion_rows = (await session.execute(select(IngestionRecord))).scalars().all()
        refreshed_job = await session.get(Job, job_id)

    assert len(ingestion_rows) >= 1
    assert refreshed_job is not None
    assert refreshed_job.progress_percent == 100


@pytest.mark.asyncio
@pytest.mark.skipif(not _has_pgvector_tables(), reason="requires Postgres pgvector")
async def test_query_accepts_llm_profile_id(service, test_namespace, test_collection):
    from graph_core.database import AsyncSessionLocal

    async with AsyncSessionLocal() as session:
        llm_profile = Profile(
            namespace_id=test_namespace.id,
            kind="llm",
            provider="local_echo",
            model="echo-v1",
            label="local-echo",
        )
        session.add(llm_profile)
        await session.commit()
        await session.refresh(llm_profile)

    await service.ingest_chunk(
        "Bhishma explains duty from the bed of arrows.",
        test_collection.id,
        test_collection.namespace_id,
    )
    result = await service.query(
        "What does Bhishma explain?",
        test_collection.id,
        test_collection.namespace_id,
        llm_profile_id=llm_profile.id,
    )
    assert "Bhishma explains duty" in result.response
