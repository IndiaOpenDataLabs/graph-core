"""Ingestion workers — Dramatiq actors for document and chunk processing."""

import logging
import uuid

import dramatiq

import graph_core.broker  # noqa: F401
from graph_core.config import settings
from graph_core.database import AsyncSessionLocal
from graph_core.models.collection import Collection
from graph_core.models.job import Job
from graph_core.models.profile import Profile
from graph_core.provider_semaphore import llm_chunk_slot
from graph_core.services.graph import GraphService
from graph_core.services.graph.ingestion.document_pipeline import (
    dispatch_pending_chunks,
    increment_chunk_counter,
)

logger = logging.getLogger(__name__)


async def _resolve_llm_chunk_scope_and_limit(job_uuid: uuid.UUID) -> tuple[str, int] | None:
    async with AsyncSessionLocal() as session:
        job = await session.get(Job, job_uuid)
        if not job or not job.collection_id:
            return None
        collection = await session.get(Collection, job.collection_id)
        if not collection:
            return None
        llm_profile = None
        if collection.llm_profile_id is not None:
            llm_profile = await session.get(Profile, collection.llm_profile_id)

    limit = (
        llm_profile.max_concurrent_calls
        if llm_profile and llm_profile.max_concurrent_calls
        else settings.llm_max_concurrent_calls
    )
    scope = str(collection.llm_profile_id) if collection.llm_profile_id else "default"
    return scope, max(1, int(limit))


@dramatiq.actor(
    queue_name="ingestion_control",
    max_retries=3,
    max_age=604800000,
    time_limit=float("inf"),
)
async def run_ingestion(job_id: str):
    """Parent actor — dispatches chunk workers for graph RAG collections."""
    service = GraphService()
    job_uuid = uuid.UUID(job_id)

    async with AsyncSessionLocal() as session:
        job = await session.get(Job, job_uuid)
        if not job:
            logger.info("Skipping stale run_ingestion message for missing job=%s", job_uuid)
            return

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
async def run_chunk(job_id: str, chunk_index: int):
    """Child actor — processes a single chunk with full Graph RAG pipeline."""
    service = GraphService()
    job_uuid = uuid.UUID(job_id)

    try:
        llm_chunk_scope = await _resolve_llm_chunk_scope_and_limit(job_uuid)
        if llm_chunk_scope is None:
            logger.info(
                "Skipping stale run_chunk message for missing job/collection: job=%s chunk=%s",
                job_uuid,
                chunk_index,
            )
            return

        scope, limit = llm_chunk_scope
        async with llm_chunk_slot(scope, max_concurrent_calls=limit):
            await service.process_single_chunk(job_id, chunk_index)
    except Exception as e:
        logger.exception(
            "run_chunk failed: job=%s chunk=%d error=%s",
            job_id,
            chunk_index,
            e,
        )
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
