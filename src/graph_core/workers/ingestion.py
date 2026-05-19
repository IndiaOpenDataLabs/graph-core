"""Ingestion workers — Dramatiq actors for document and chunk processing."""

import logging
import uuid

import dramatiq

from graph_core.services.graph import GraphService

logger = logging.getLogger(__name__)


@dramatiq.actor(max_retries=3, max_age=3600000)
async def run_ingestion(job_id: str):
    """Parent actor — dispatches chunk workers for graph RAG collections."""
    service = GraphService()
    job_uuid = uuid.UUID(job_id)

    await service.update_job_status(job_uuid, "running")
    await service.append_job_event(job_uuid, "started")

    try:
        await service.ingest_document_pipeline(job_uuid)
        await service.append_job_event(job_uuid, "completed")
    except Exception as e:
        await service.append_job_event(job_uuid, "error", {"error": str(e)})
        await service.update_job_status(job_uuid, "failed", error=str(e))
        raise


@dramatiq.actor(max_retries=3, max_age=1800000)
async def run_chunk(job_id: str, chunk_index: int):
    """Child actor — processes a single chunk with full Graph RAG pipeline."""
    service = GraphService()

    try:
        await service.process_single_chunk(job_id, chunk_index)
    except Exception as e:
        logger.error("run_chunk failed: job=%s chunk=%d error=%s", job_id, chunk_index, e)
        await service.update_chunk_status(
            uuid.UUID(job_id), chunk_index, "failed", error=str(e)
        )
        raise
