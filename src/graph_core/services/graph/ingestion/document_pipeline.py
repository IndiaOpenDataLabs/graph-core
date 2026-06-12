"""Document ingestion pipeline functions extracted from GraphService.

Module-level async functions that handle the full document ingestion lifecycle:
job enqueueing, chunking, fan-out, per-chunk processing, and progress tracking.
"""

import uuid
from dataclasses import dataclass
from datetime import UTC, datetime

from sqlalchemy import func, select, text

from graph_core.config import settings
from graph_core.database import AsyncSessionLocal
from graph_core.models.chunk import IngestionChunk
from graph_core.models.collection import Collection
from graph_core.models.job import Job, JobEvent
from graph_core.models.profile import Profile
from graph_core.provider_semaphore import (
    release_llm_dispatch_slot,
    try_acquire_llm_dispatch_slot,
)
from graph_core.services.chunking import DocumentChunker
from graph_core.services.graph.ingestion.chunk_processor import ingest_collection_chunk

UTC = UTC

_chunker = DocumentChunker(
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
    domain: str | None = None,
) -> DocumentIngestionResult:
    """Create a pending ingest_document Job and return its result wrapper."""
    if not text.strip():
        raise ValueError("Cannot ingest an empty document")
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
            payload={"text": text, "domain": domain},
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
        domain = job.payload.get("domain") if isinstance(job.payload, dict) else None

    chunks = _chunker.chunk_text(text, domain=domain)
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
                domain=domain,
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


def _resolve_chunk_dispatch_limit(
    collection: Collection,
    embedding_profile: Profile | None,
    llm_profile: Profile | None,
) -> int:
    limits = []

    if embedding_profile and embedding_profile.max_concurrent_calls:
        limits.append(embedding_profile.max_concurrent_calls)
    elif collection.embedding_profile_id is not None:
        limits.append(settings.embedding_max_concurrent_calls)

    if llm_profile and llm_profile.max_concurrent_calls:
        limits.append(llm_profile.max_concurrent_calls)
    elif collection.llm_profile_id is not None:
        limits.append(settings.llm_max_concurrent_calls)

    if not limits:
        return 1

    return max(1, min(limits))


def _llm_dispatch_scope_and_limit(
    collection: Collection,
    llm_profile: Profile | None,
) -> tuple[str, int]:
    limit = (
        llm_profile.max_concurrent_calls
        if llm_profile and llm_profile.max_concurrent_calls
        else settings.llm_max_concurrent_calls
    )
    if collection.llm_profile_id is not None:
        return str(collection.llm_profile_id), max(1, int(limit))
    return "default", max(1, int(limit))


async def release_chunk_dispatch_slot(job_id: uuid.UUID, chunk_index: int) -> None:
    async with AsyncSessionLocal() as session:
        job = await session.get(Job, job_id)
        if not job or not job.collection_id:
            return
        collection = await session.get(Collection, job.collection_id)
        if not collection:
            return
        llm_profile = None
        if collection.llm_profile_id is not None:
            llm_profile = await session.get(Profile, collection.llm_profile_id)

    dispatch_scope, llm_dispatch_limit = _llm_dispatch_scope_and_limit(
        collection,
        llm_profile,
    )
    await release_llm_dispatch_slot(
        dispatch_scope,
        f"{job_id}:{chunk_index}",
        max_concurrent_calls=llm_dispatch_limit,
    )


async def dispatch_pending_chunks(job_id: uuid.UUID, slots: int | None = None) -> int:
    """Reserve and enqueue the next bounded window of pending chunks."""
    async with AsyncSessionLocal() as session:
        job = await session.get(Job, job_id)
        if not job or not job.collection_id:
            return 0

        collection = await session.get(Collection, job.collection_id)
        if not collection:
            return 0

        embedding_profile = None
        if collection.embedding_profile_id is not None:
            embedding_profile = await session.get(
                Profile,
                collection.embedding_profile_id,
            )

        llm_profile = None
        if collection.llm_profile_id is not None:
            llm_profile = await session.get(Profile, collection.llm_profile_id)

        dispatch_limit = _resolve_chunk_dispatch_limit(
            collection,
            embedding_profile,
            llm_profile,
        )
        dispatch_scope, llm_dispatch_limit = _llm_dispatch_scope_and_limit(
            collection,
            llm_profile,
        )

        active_count = await session.scalar(
            select(func.count())
            .select_from(IngestionChunk)
            .where(
                IngestionChunk.job_id == job_id,
                IngestionChunk.status == "processing",
            )
        )
        available = dispatch_limit - int(active_count or 0)
        if slots is not None:
            available = min(available, slots)
        if available <= 0:
            return 0

        pending_chunks = (
            await session.execute(
                select(IngestionChunk)
                .where(
                    IngestionChunk.job_id == job_id,
                    IngestionChunk.status == "pending",
                )
                .order_by(IngestionChunk.chunk_index)
                .limit(available)
                .with_for_update(skip_locked=True)
            )
        ).scalars().all()

        if not pending_chunks:
            return 0

        dispatch_indices: list[int] = []
        reserved_tokens: list[str] = []
        for chunk in pending_chunks:
            token = f"{job_id}:{chunk.chunk_index}"
            acquired = await try_acquire_llm_dispatch_slot(
                dispatch_scope,
                token,
                max_concurrent_calls=llm_dispatch_limit,
            )
            if not acquired:
                continue
            chunk.status = "processing"  # type: ignore[assignment]
            dispatch_indices.append(chunk.chunk_index)
            reserved_tokens.append(token)

        if not dispatch_indices:
            await session.rollback()
            return 0

        try:
            await session.commit()
        except Exception:
            for token in reserved_tokens:
                await release_llm_dispatch_slot(
                    dispatch_scope,
                    token,
                    max_concurrent_calls=llm_dispatch_limit,
                )
            raise

    from graph_core.workers.ingestion import run_chunk

    for chunk_index in dispatch_indices:
        run_chunk.send(str(job_id), chunk_index)  # type: ignore[attr-defined]

    return len(dispatch_indices)


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

        job = await session.get(Job, job_uuid)
        collection = await session.get(Collection, job.collection_id)
        text = chunk.text
        llm_profile = None
        if collection.llm_profile_id is not None:
            llm_profile = await session.get(Profile, collection.llm_profile_id)

    dispatch_scope, llm_dispatch_limit = _llm_dispatch_scope_and_limit(
        collection,
        llm_profile,
    )
    dispatch_token = f"{job_uuid}:{chunk_index}"

    try:
        result = await ingest_collection_chunk(
            text=text,
            collection=collection,
            namespace_id=collection.namespace_id,
            chunk_index=chunk_index,
            domain=job.payload.get("domain") if isinstance(job.payload, dict) else None,
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
        await dispatch_pending_chunks(job_uuid, slots=1)
    finally:
        await release_llm_dispatch_slot(
            dispatch_scope,
            dispatch_token,
            max_concurrent_calls=llm_dispatch_limit,
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
                "WHEN chunks_completed + 1 >= chunks_total AND EXISTS ("
                "  SELECT 1 FROM ingestion_chunks "
                "  WHERE job_id = :jid AND status = 'failed'"
                ") THEN CAST('failed' AS job_status) "
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
                        "WHERE id = :jid AND status IN ('completed', 'failed')"
                    ),
                    {"now": datetime.now(UTC), "jid": str(job_id).replace("-", "")},
                )
                await session.commit()
            await session.commit()
            return int((completed / total * 100) if total else 0)
        return 0
