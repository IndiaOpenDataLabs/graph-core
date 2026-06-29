"""Ingestion workers — Dramatiq actors for document and chunk processing."""

import logging
import uuid

import dramatiq

import graph_core.broker  # noqa: F401
from graph_core.config import settings
from graph_core.services.graph import GraphService
from graph_core.services.graph.ingestion.document_pipeline import (
    dispatch_pending_chunks,
    increment_chunk_counter,
    is_job_cancelled,
)

logger = logging.getLogger(__name__)


@dramatiq.actor(
    queue_name="ingestion_control",
    max_retries=3,
    max_age=604800000,
    time_limit=float("inf"),
)
async def run_ingestion(job_id: str):
    """Parent actor — dispatches chunk workers for graph RAG collections."""
    if await is_job_cancelled(job_id):
        logger.info("Skipping cancelled ingestion job: %s", job_id)
        return

    service = GraphService()
    job_uuid = uuid.UUID(job_id)

    await service.update_job_status(job_uuid, "running")
    await service.append_job_event(job_uuid, "started")

    try:
        await service.ingest_document_pipeline(job_uuid)
        await dispatch_pending_chunks(job_uuid)
        await service.append_job_event(job_uuid, "chunks_dispatched")
    except Exception as e:
        await service.append_job_event(job_uuid, "error", {"error": str(e)})
        await service.update_job_status(job_uuid, "failed", error=str(e))
        raise


@dramatiq.actor(
    queue_name="ingestion_chunks",
    max_retries=3,
    max_age=settings.ingest_chunk_max_age_ms,
    time_limit=settings.ingest_chunk_time_limit_ms,
)
async def run_chunk(
    job_id: str,
    chunk_index: int,
    llm_scope: str | None = None,
    llm_limit: int | None = None,
    llm_slot_token: str | None = None,
):
    """Child actor — processes a single chunk with full Graph RAG pipeline."""
    if await is_job_cancelled(job_id):
        logger.info("Skipping cancelled chunk job: %s chunk=%d", job_id, chunk_index)
        return

    service = GraphService()

    try:
        await service.process_single_chunk(
            job_id,
            chunk_index,
            llm_scope=llm_scope,
            llm_limit=llm_limit,
            llm_slot_token=llm_slot_token,
        )
    except Exception as e:
        logger.exception(
            "run_chunk failed: job=%s chunk=%d error=%s",
            job_id,
            chunk_index,
            e,
        )
        job_uuid = uuid.UUID(job_id)
        await service.update_chunk_status(job_uuid, chunk_index, "failed", error=str(e))
        await increment_chunk_counter(job_uuid)
        await service.append_job_event(
            job_uuid,
            "chunk_failed",
            {
                "chunk_index": chunk_index,
                "error": str(e),
            },
        )
        await dispatch_pending_chunks(job_uuid, slots=1)
