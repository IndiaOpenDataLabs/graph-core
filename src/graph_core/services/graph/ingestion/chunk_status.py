"""Namespace-scoped chunk inspection and targeted ingestion retries."""

from __future__ import annotations

import uuid
from typing import Any

from sqlalchemy import func, select

from graph_core.database import AsyncSessionLocal
from graph_core.models.chunk import IngestionChunk
from graph_core.models.job import Job, JobEvent


class ChunkRetryConflict(ValueError):
    """The requested chunks cannot currently be retried."""


async def list_job_chunks(
    job_id: uuid.UUID,
    namespace_id: uuid.UUID,
    *,
    offset: int = 0,
    limit: int = 50,
    status: str | None = None,
) -> dict[str, Any]:
    async with AsyncSessionLocal() as session:
        job = await session.scalar(
            select(Job).where(
                Job.id == job_id,
                Job.namespace_id == namespace_id,
            )
        )
        if job is None:
            raise LookupError("Job not found")
        counts = dict(
            (
                await session.execute(
                    select(IngestionChunk.status, func.count())
                    .where(IngestionChunk.job_id == job_id)
                    .group_by(IngestionChunk.status)
                )
            ).all()
        )
        query = select(IngestionChunk).where(IngestionChunk.job_id == job_id)
        if status is not None:
            query = query.where(IngestionChunk.status == status)
        chunks = (
            (
                await session.execute(
                    query.order_by(IngestionChunk.chunk_index)
                    .offset(offset)
                    .limit(limit)
                )
            )
            .scalars()
            .all()
        )
        return {
            "job_id": str(job_id),
            "job_status": job.status,
            "counts": {
                state: int(counts.get(state, 0))
                for state in (
                    "pending",
                    "processing",
                    "completed",
                    "failed",
                    "cancelled",
                )
            },
            "total": sum(counts.values()),
            "offset": offset,
            "limit": limit,
            "filtered_total": counts.get(status, 0) if status else sum(counts.values()),
            "chunks": [
                {
                    "id": str(chunk.id),
                    "chunk_index": chunk.chunk_index,
                    "status": chunk.status,
                    "error": chunk.error,
                    "processing_started_at": chunk.processing_started_at.isoformat()
                    if chunk.processing_started_at
                    else None,
                    "completed_at": chunk.completed_at.isoformat()
                    if chunk.completed_at
                    else None,
                }
                for chunk in chunks
            ],
        }


async def retry_failed_chunks(
    job_id: uuid.UUID,
    namespace_id: uuid.UUID,
    chunk_indices: list[int] | None = None,
) -> dict[str, Any]:
    async with AsyncSessionLocal() as session:
        job = await session.scalar(
            select(Job)
            .where(
                Job.id == job_id,
                Job.namespace_id == namespace_id,
            )
            .with_for_update()
        )
        if job is None:
            raise LookupError("Job not found")
        if job.job_type != "ingest_document" or job.status == "cancelled":
            raise ChunkRetryConflict(
                "Only failed document-ingestion chunks can be retried"
            )
        chunks = (
            (
                await session.execute(
                    select(IngestionChunk)
                    .where(IngestionChunk.job_id == job_id)
                    .with_for_update()
                )
            )
            .scalars()
            .all()
        )
        failed = [chunk for chunk in chunks if chunk.status == "failed"]
        pending = [chunk for chunk in chunks if chunk.status == "pending"]
        if any(chunk.status == "processing" for chunk in chunks):
            raise ChunkRetryConflict("Wait for the current chunk processing to finish")
        if pending and job.status == "running" and chunk_indices is None:
            failed = []  # Resume a previously saved retry without resetting more rows.
        elif pending and failed:
            raise ChunkRetryConflict(
                "Wait for the pending chunks to finish before retrying"
            )
        if chunk_indices is not None:
            requested = set(chunk_indices)
            failed = [chunk for chunk in failed if chunk.chunk_index in requested]
            if {chunk.chunk_index for chunk in failed} != requested:
                raise ChunkRetryConflict("Only failed chunks can be selected for retry")
        if failed:
            if job.status != "failed":
                raise ChunkRetryConflict(
                    "Wait for the ingestion job to finish before retrying"
                )
            previous_errors = {str(chunk.chunk_index): chunk.error for chunk in failed}
            for chunk in failed:
                chunk.status = "pending"
                chunk.error = None
                chunk.processing_started_at = None
                chunk.lease_expires_at = None
                chunk.completed_at = None
            # The existing counter counts terminal attempts, including failures.
            # Subtract only the reset chunks; leave successful chunks untouched.
            job.chunks_total = len(chunks)
            job.chunks_completed = sum(
                chunk.status in {"completed", "failed", "cancelled"} for chunk in chunks
            )
            job.progress_percent = int(job.chunks_completed * 100 / len(chunks))
            job.status = "running"
            job.error = None
            job.completed_at = None
            session.add(
                JobEvent(
                    job_id=job_id,
                    event_type="chunks_retry_requested",
                    payload={
                        "chunk_indices": [chunk.chunk_index for chunk in failed],
                        "previous_errors": previous_errors,
                    },
                )
            )
        elif not (pending and job.status == "running" and chunk_indices is None):
            raise ChunkRetryConflict("No failed chunks to retry")
        retried = [chunk.chunk_index for chunk in failed]
        await session.commit()

    # Schedule after commit, so workers can see the reset rows. A separate
    # bootstrap actor retries dispatch when another job holds the provider slots.
    from graph_core.workers.ingestion import dispatch_retried_chunks

    dispatch_retried_chunks.send(str(job_id))
    return {"job_id": str(job_id), "retried_chunks": retried, "status": "queued"}
