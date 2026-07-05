"""Jobs API — integration tests."""

import uuid
from unittest.mock import AsyncMock, patch

import pytest

from graph_core.models.job import Job


@pytest.mark.asyncio
async def test_get_job_not_found(async_client):
    resp = await async_client.get(f"/jobs/{uuid.uuid4()}")
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_cancel_job(async_client, db_session, test_namespace):
    job_id = uuid.uuid4()
    db_session.add(
        Job(
            id=job_id,
            namespace_id=test_namespace.id,
            collection_id=None,
            job_type="enhance",
            status="running",
            progress_percent=6,
            chunks_completed=41,
            chunks_total=666,
        )
    )
    await db_session.commit()

    with (
        patch("graph_core.services.graph.mark_jobs_cancelled", new=AsyncMock()),
        patch("graph_core.services.graph.finalize_cancelled_jobs", new=AsyncMock()),
        patch("graph_core.services.graph.purge_queued_job_messages", new=AsyncMock()),
        patch("graph_core.services.graph.cancel_processing_chunks", new=AsyncMock()),
    ):
        resp = await async_client.post(f"/jobs/{job_id}/cancel")

    assert resp.status_code == 200
    data = resp.json()
    assert data["id"] == str(job_id)
    assert data["status"] == "cancelled"
    assert data["progress_label"] == "meta_entities"
    assert data["progress_completed"] == 41
    assert data["progress_total"] == 666


@pytest.mark.asyncio
async def test_stream_job_events_returns_sse(async_client):
    resp = await async_client.get(f"/jobs/{uuid.uuid4()}/stream")
    assert resp.status_code == 200
    assert resp.headers["content-type"] == "text/event-stream; charset=utf-8"
