"""Document ingestion pipeline functions extracted from GraphService.

Module-level async functions that handle the full document ingestion lifecycle:
job enqueueing, chunking, fan-out, per-chunk processing, and progress tracking.
"""

import uuid
from dataclasses import dataclass
from datetime import UTC, datetime

from sqlalchemy import select, text

from graph_core.config import settings
from graph_core.database import AsyncSessionLocal
from graph_core.models.chunk import IngestionChunk
from graph_core.models.collection import Collection
from graph_core.models.job import Job, JobEvent
from graph_core.services.chunking import TokenChunker
from graph_core.services.graph.ingestion.chunk_processor import ingest_collection_chunk

UTC = UTC

_chunker = TokenChunker(
    chunk_size_tokens=settings.chunk_size_tokens,
    chunk_overlap_tokens=settings.chunk_overlap_tokens,
)


# ── Result type ──


@dataclass
class DocumentIngestionResult:
    job_id: uuid.UUID
    status: str


# ── Job status helpers ──


async def _update_job_status(
    job_id: uuid.UUID,
    status: str,
    progress_percent: int | None = None,
    error: str | None = None,
) -> None:
    async with AsyncSessionLocal() as session:
        job = await session.get(Job, job_id)
        if not job:
            return
        job.status = status
        if progress_percent is not None:
            job.progress_percent = progress_percent
        if error:
            job.error = error
        if status == "running" and not job.started_at:
            job.started_at = datetime.now(UTC)
        if status in ("completed", "failed", "cancelled"):
            job.completed_at = datetime.now(UTC)
        await session.commit()


async def _append_job_event(
    job_id: uuid.UUID,
    event_type: str,
    payload: dict | None = None,
) -> None:
    async with AsyncSessionLocal() as session:
        event = JobEvent(job_id=job_id, event_type=event_type, payload=payload)
        session.add(event)
        await session.commit()


# ── Job enqueueing ──


async def enqueue_document_ingestion_job(
    text: str,
    collection_id: uuid.UUID,
    namespace_id: uuid.UUID,
) -> DocumentIngestionResult:
    """Create a pending ingest_document Job and return its result wrapper."""
    async with AsyncSessionLocal() as session:
        collection = await session.get(Collection, collection_id)
        if not collection:
            raise ValueError(f"Collection {collection_id} not found")
        if collection.namespace_id != namespace_id:
            raise ValueError(
                f"Collection {collection_id} does not belong to namespace "
                f"{namespace_id}"
            )

        job = Job(
            namespace_id=namespace_id,
            collection_id=collection_id,
            job_type="ingest_document",
            status="pending",
            payload={"text": text},
        )
        session.add(job)
        await session.commit()
        await session.refresh(job)

    return DocumentIngestionResult(job_id=job.id, status="pending")


# ── Document pipeline ──


async def ingest_document_pipeline(job_id: uuid.UUID) -> None:
    """Main pipeline — dispatches chunks based on collection strategy."""
    async with AsyncSessionLocal() as session:
        job = await session.get(Job, job_id)
        if not job:
            raise ValueError(f"Job {job_id} not found")
        if not job.payload or "text" not in job.payload:
            raise ValueError(f"Job {job_id} does not contain input text")

        collection = await session.get(Collection, job.collection_id)
        if not collection:
            raise ValueError(f"Collection {job.collection_id} not found")

        text = str(job.payload["text"])

    chunks = _chunker.chunk_text(text)
    total_chunks = max(len(chunks), 1)

    if not chunks:
        await _update_job_status(job_id, "completed", progress_percent=100)
        return

    # For custom_graph_rag and light_rag: fan-out chunks to parallel workers
    if collection.strategy in ("custom_graph_rag", "light_rag"):
        await fan_out_chunks(job_id, collection.id, chunks)
    else:
        # Vector strategy: sequential processing
        for index, chunk in enumerate(chunks, start=1):
            await ingest_collection_chunk(
                text=chunk,
                collection=collection,
                namespace_id=collection.namespace_id,
                chunk_index=index - 1,
            )
            progress = int(index * 100 / total_chunks)
            await _update_job_status(job_id, "running", progress_percent=progress)
            await _append_job_event(
                job_id, "chunk_completed",
                {"chunk_index": index - 1, "total_chunks": total_chunks},
            )
        await _update_job_status(job_id, "completed", progress_percent=100)


# ── Chunk fan-out ──


async def fan_out_chunks(
    job_id: uuid.UUID, collection_id: uuid.UUID, chunks: list[str]
) -> None:
    """Create chunk records. Worker is responsible for enqueuing."""
    async with AsyncSessionLocal() as session:
        for index, chunk_text in enumerate(chunks):
            chunk = IngestionChunk(
                job_id=job_id,
                chunk_index=index,
                text=chunk_text,
                status="pending",
            )
            session.add(chunk)

        await session.execute(
            text("UPDATE jobs SET chunks_total = :total WHERE id = :jid"),
            {"total": len(chunks), "jid": str(job_id).replace("-", "")},
        )
        await session.commit()


# ── Single-chunk processing ──


async def process_single_chunk(job_id: str, chunk_index: int) -> None:
    """Process a single chunk — called by run_chunk worker."""
    job_uuid = uuid.UUID(job_id)

    async with AsyncSessionLocal() as session:
        chunk = await session.execute(
            select(IngestionChunk).where(
                IngestionChunk.job_id == job_uuid,
                IngestionChunk.chunk_index == chunk_index,
            )
        )
        chunk = chunk.scalar_one()
        chunk.status = "processing"  # type: ignore[assignment]
        await session.commit()

        job = await session.get(Job, job_uuid)
        collection = await session.get(Collection, job.collection_id)
        text = chunk.text

    result = await ingest_collection_chunk(
        text=text,
        collection=collection,
        namespace_id=collection.namespace_id,
        chunk_index=chunk_index,
    )

    await update_chunk_status(job_uuid, chunk_index, "completed")

    await increment_chunk_counter(job_uuid)
    await _append_job_event(
        job_uuid,
        "chunk_completed",
        {
            "chunk_index": chunk_index,
            "entity_count": result.entity_count,
            "relationship_count": result.relationship_count,
        },
    )


# ── Chunk status tracking ──


async def update_chunk_status(
    job_id: uuid.UUID, chunk_index: int, status: str, error: str | None = None
) -> None:
    async with AsyncSessionLocal() as session:
        chunk = await session.execute(
            select(IngestionChunk).where(
                IngestionChunk.job_id == job_id,
                IngestionChunk.chunk_index == chunk_index,
            )
        )
        chunk = chunk.scalar_one()
        chunk.status = status  # type: ignore[assignment]
        if error:
            chunk.error = error  # type: ignore[attr-defined]
        if status in ("completed", "failed"):
            chunk.completed_at = datetime.now(UTC)  # type: ignore[attr-defined]
        await session.commit()


async def increment_chunk_counter(job_id: uuid.UUID) -> int:
    """Atomically increment chunks_completed and return progress percent."""
    async with AsyncSessionLocal() as session:
        result = await session.execute(
            text(
                "UPDATE jobs SET chunks_completed = chunks_completed + 1, "
                "progress_percent = CAST("
                "((chunks_completed + 1)::float / NULLIF(chunks_total, 0) * 100) "
                "AS integer), "
                "status = CASE "
                "WHEN chunks_completed + 1 >= chunks_total "
                "THEN CAST('completed' AS job_status) "
                "ELSE CAST('running' AS job_status) END "
                "WHERE id = :jid "
                "RETURNING chunks_completed, chunks_total, status"
            ),
            {"jid": job_id},
        )
        row = result.fetchone()
        if row:
            completed, total, status = row
            if total and completed >= total:
                await session.execute(
                    text(
                        "UPDATE jobs SET completed_at = :now "
                        "WHERE id = :jid AND status = 'completed'"
                    ),
                    {"now": datetime.now(UTC), "jid": str(job_id).replace("-", "")},
                )
                await session.commit()
            await session.commit()
            return int((completed / total * 100) if total else 0)
        return 0
