"""Jobs API — integration tests."""

import pytest


@pytest.mark.asyncio
async def test_get_job_not_found(async_client):
    import uuid

    resp = await async_client.get(f"/jobs/{uuid.uuid4()}")
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_stream_job_events_returns_sse(async_client):
    import uuid

    resp = await async_client.get(f"/jobs/{uuid.uuid4()}/stream")
    assert resp.status_code == 200
    assert resp.headers["content-type"] == "text/event-stream; charset=utf-8"
