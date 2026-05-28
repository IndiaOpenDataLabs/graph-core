"""Ingest API — integration tests."""

from unittest.mock import patch

import pytest


def _has_pgvector_tables() -> bool:
    """Check if pgvector tables are available (Postgres, not SQLite)."""
    from graph_core.database import engine
    return "postgresql" in engine.url.drivername


@pytest.mark.asyncio
@pytest.mark.skipif(not _has_pgvector_tables(), reason="requires Postgres pgvector")
async def test_ingest_chunk(async_client, test_collection):
    resp = await async_client.post(
        f"/collections/{test_collection.id}/ingest/chunk",
        json={"text": "hello world from API"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert "chunk_hash" in data
    assert data["entity_count"] == 0  # placeholder returns 0


@pytest.mark.asyncio
async def test_ingest_document_returns_job_id(async_client, test_collection):
    with patch("graph_core.workers.ingestion.run_ingestion"):
        resp = await async_client.post(
            f"/collections/{test_collection.id}/ingest/doc",
            json={"text": "this is a longer document that gets queued"},
        )
    assert resp.status_code == 200
    data = resp.json()
    assert "job_id" in data
    assert data["status"] == "pending"


@pytest.mark.asyncio
@pytest.mark.skipif(not _has_pgvector_tables(), reason="requires Postgres pgvector")
async def test_ingest_chunk_then_query_returns_vector_context(async_client, test_collection):
    ingest_resp = await async_client.post(
        f"/collections/{test_collection.id}/ingest/chunk",
        json={"text": "Krishna teaches Arjuna about dharma."},
    )
    assert ingest_resp.status_code == 200

    query_resp = await async_client.post(
        f"/collections/{test_collection.id}/query",
        json={"question": "What does Krishna teach Arjuna?"},
    )
    assert query_resp.status_code == 200
    assert "Krishna teaches Arjuna" in query_resp.json()["response"]


@pytest.mark.asyncio
async def test_ingest_chunk_wrong_namespace(async_client, test_collection):
    import uuid

    wrong_ns = uuid.uuid4()
    resp = await async_client.post(
        f"/collections/{test_collection.id}/ingest/chunk",
        json={"text": "hello"},
        headers={"X-Namespace-ID": str(wrong_ns)},
    )
    assert resp.status_code == 403
