"""FastAPI router — job status and SSE streaming."""

import uuid
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import StreamingResponse

from graph_core.api.auth import get_namespace_id
from graph_core.services.graph import GraphService

router = APIRouter(prefix="/jobs", tags=["jobs"])
service = GraphService()


@router.get("/")
async def list_jobs(
    namespace_id: Annotated[uuid.UUID, Depends(get_namespace_id)],
    limit: int = Query(default=20, ge=1, le=100),
    collection_id: uuid.UUID | None = Query(default=None),
) -> list[dict]:
    """List recent jobs for the current namespace."""
    return await service.list_jobs(
        namespace_id,
        limit=limit,
        collection_id=collection_id,
    )


@router.get("/{job_id}")
async def get_job(job_id: uuid.UUID) -> dict:
    """Get durable job status from Postgres."""
    try:
        return await service.get_job(job_id)
    except ValueError:
        raise HTTPException(status_code=404, detail="Job not found")


@router.get("/{job_id}/result")
async def get_job_result(job_id: uuid.UUID) -> dict:
    """Get the final result payload for a completed job."""
    try:
        return await service.get_job_result(job_id)
    except ValueError as exc:
        message = str(exc)
        if "not completed" in message:
            raise HTTPException(status_code=409, detail=message)
        raise HTTPException(status_code=404, detail=message)


@router.get("/{job_id}/stream")
async def stream_job_events(job_id: uuid.UUID):
    """SSE stream of transient job events via Redis pubsub."""
    # TODO: subscribe to Redis channel f"job:{job_id}" and yield SSE events
    async def event_generator():
        import json
        payload = json.dumps({"status": "subscribed", "job_id": str(job_id)})
        yield f"data: {payload}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")
