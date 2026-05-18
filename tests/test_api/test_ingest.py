"""Ingest API — integration tests."""

import pytest


@pytest.mark.asyncio
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
    resp = await async_client.post(
        f"/collections/{test_collection.id}/ingest/doc",
        json={"text": "this is a longer document that gets queued"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert "job_id" in data
    assert data["status"] == "pending"


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
