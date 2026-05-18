"""FastAPI router — job status and SSE streaming."""

import uuid

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from graph_core.services.graph import GraphService


router = APIRouter(prefix="/jobs", tags=["jobs"])
service = GraphService()


@router.get("/{job_id}")
async def get_job(job_id: uuid.UUID) -> dict:
    """Get durable job status from Postgres."""
    try:
        return await service.get_job(job_id)
    except ValueError:
        raise HTTPException(status_code=404, detail="Job not found")


@router.get("/{job_id}/stream")
async def stream_job_events(job_id: uuid.UUID):
    """SSE stream of transient job events via Redis pubsub."""
    # TODO: subscribe to Redis channel f"job:{job_id}" and yield SSE events
    async def event_generator():
        yield f"data: {{"status": "subscribed", "job_id": "{job_id}"}}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")
