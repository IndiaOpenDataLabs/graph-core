"""Ingestion worker — thin Dramatiq actor wrapping GraphService."""

import uuid

import dramatiq

from graph_core.services.graph import GraphService


@dramatiq.actor(max_retries=3, max_age=3600000)
async def run_ingestion(job_id: str):
    """Thin wrapper — all pipeline logic lives in GraphService."""
    service = GraphService()
    job_uuid = uuid.UUID(job_id)

    await service.update_job_status(job_uuid, "running")
    await service.append_job_event(job_uuid, "started")

    try:
        await service.ingest_document_pipeline(job_uuid)
        await service.append_job_event(job_uuid, "completed")
        await service.update_job_status(job_uuid, "completed", progress_percent=100)
    except Exception as e:
        await service.append_job_event(job_uuid, "error", {"error": str(e)})
        await service.update_job_status(job_uuid, "failed", error=str(e))
        raise
