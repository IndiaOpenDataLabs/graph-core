"""Graph RAG service — thin public API over ingestion/ and query/ submodules.

GraphService delegates to ingestion/ and query/ submodules. All domain
logic lives there; this package provides the familiar class API used by
API routes and workers.
"""

import uuid
from datetime import UTC, datetime
from typing import Any, Literal

from sqlalchemy import delete, select

from graph_core.database import AsyncSessionLocal
from graph_core.models.collection import Collection
from graph_core.models.job import Job, JobEvent
from graph_core.models.namespace import Namespace
from graph_core.models.profile import Profile
from graph_core.services.graph.ingestion import (
    deterministic_uuid,
    fan_out_chunks,
    get_graph_storage,
    increment_chunk_counter,
)
from graph_core.services.graph.ingestion.chunk_processor import (
    ChunkIngestionResult,
    ingest_collection_chunk,
)
from graph_core.services.graph.ingestion.document_pipeline import (
    DocumentIngestionResult,
    enqueue_document_ingestion_job,
    ingest_document_pipeline,
    process_single_chunk,
)
from graph_core.services.graph.ingestion.document_pipeline import (
    update_chunk_status as _update_chunk_status,
)
from graph_core.services.graph.query import (
    extract_keywords,
    fallback_keywords,
    generate_vector_answer,
)
from graph_core.services.graph.query.graph_rag import graph_rag_query
from graph_core.services.graph.query.lightrag import lightrag_query
from graph_core.services.graph.query.vector import (
    QueryResult,
    vector_query,
)
from graph_core.services.sanitizer import TextSanitizer
from graph_core.storage.graph_rag_vectors import GraphRAGVectorStore
from graph_core.storage.vector_store import VectorStore
from graph_core.storage.vector_tables import (
    create_all_tables,
    drop_all_tables,
)


class GraphService:
    """Thin orchestration class. All logic delegates to submodules."""

    def __init__(self):
        self._sanitizer = TextSanitizer()
        self._vector_store = VectorStore()
        self._graph_rag_vectors = GraphRAGVectorStore()

    # ── Collections ──

    async def create_collection(
        self,
        name: str,
        namespace_id: uuid.UUID,
        strategy: Literal["vector", "custom_graph_rag", "light_rag"] = "vector",
        embedding_profile_id: uuid.UUID | None = None,
        llm_profile_id: uuid.UUID | None = None,
        default_query_mode: str | None = None,
    ) -> Collection:
        async with AsyncSessionLocal() as session:
            ns = await session.get(Namespace, namespace_id)
            if not ns:
                raise ValueError(f"Namespace {namespace_id} not found")

            if embedding_profile_id is None:
                raise ValueError("Embedding profile is required to create a collection")

            dimensions = None
            profile = await session.get(Profile, embedding_profile_id)
            if not profile:
                raise ValueError(f"Embedding profile {embedding_profile_id} not found")
            if profile.namespace_id != namespace_id:
                raise ValueError("Embedding profile does not belong to namespace")
            if profile.kind != "embedding":
                raise ValueError("Profile kind must be embedding")
            if profile.dimensions is None:
                raise ValueError("Embedding profile dimensions are required")
            dimensions = profile.dimensions

            if llm_profile_id is not None:
                llm_profile = await session.get(Profile, llm_profile_id)
                if not llm_profile:
                    raise ValueError(f"LLM profile {llm_profile_id} not found")
                if llm_profile.namespace_id != namespace_id:
                    raise ValueError("LLM profile does not belong to namespace")
                if llm_profile.kind != "llm":
                    raise ValueError("LLM profile kind must be llm")

            collection = Collection(
                name=name,
                namespace_id=namespace_id,
                strategy=strategy,
                embedding_profile_id=embedding_profile_id,
                llm_profile_id=llm_profile_id,
                default_query_mode=default_query_mode,
                embedding_dimensions=dimensions,
            )
            session.add(collection)
            await session.commit()
            await session.refresh(collection)

        if dimensions is not None:
            await create_all_tables(collection.id, dimensions)

        return collection

    async def update_collection(
        self,
        collection_id: uuid.UUID,
        namespace_id: uuid.UUID,
        *,
        name: str | None = None,
        strategy: Literal["vector", "custom_graph_rag", "light_rag"] | None = None,
        embedding_profile_id: uuid.UUID | None = None,
        llm_profile_id: uuid.UUID | None = None,
        default_query_mode: str | None = None,
        clear_llm_profile: bool = False,
        clear_default_query_mode: bool = False,
    ) -> Collection:
        async with AsyncSessionLocal() as session:
            collection = await session.get(Collection, collection_id)
            if not collection:
                raise ValueError(f"Collection {collection_id} not found")
            self._enforce_namespace(collection, namespace_id)

            if name is not None:
                collection.name = name
            if strategy is not None:
                collection.strategy = strategy

            if embedding_profile_id is not None:
                profile = await session.get(Profile, embedding_profile_id)
                if not profile:
                    raise ValueError(
                        f"Embedding profile {embedding_profile_id} not found"
                    )
                if profile.namespace_id != namespace_id:
                    raise ValueError("Embedding profile does not belong to namespace")
                if profile.kind != "embedding":
                    raise ValueError("Profile kind must be embedding")
                if profile.dimensions is None:
                    raise ValueError("Embedding profile dimensions are required")
                collection.embedding_profile_id = embedding_profile_id
                collection.embedding_dimensions = profile.dimensions

            if clear_llm_profile:
                collection.llm_profile_id = None
            elif llm_profile_id is not None:
                llm_profile = await session.get(Profile, llm_profile_id)
                if not llm_profile:
                    raise ValueError(f"LLM profile {llm_profile_id} not found")
                if llm_profile.namespace_id != namespace_id:
                    raise ValueError("LLM profile does not belong to namespace")
                if llm_profile.kind != "llm":
                    raise ValueError("LLM profile kind must be llm")
                collection.llm_profile_id = llm_profile_id

            if clear_default_query_mode:
                collection.default_query_mode = None
            elif default_query_mode is not None:
                collection.default_query_mode = default_query_mode

            await session.commit()
            await session.refresh(collection)
            return collection

    async def delete_collection(self, collection_id: uuid.UUID) -> None:
        await self.get_collection(collection_id)
        await drop_all_tables(collection_id)
        from graph_core.storage.graph_storage import FalkorDBGraphStorage
        graph_name = f"collection_{str(collection_id).replace('-', '')}"
        graph_storage = FalkorDBGraphStorage(graph_name)
        await graph_storage.drop()
        async with AsyncSessionLocal() as session:
            await session.execute(
                delete(Collection).where(Collection.id == collection_id)
            )
            await session.commit()

    async def list_collections(self, namespace_id: uuid.UUID) -> list[Collection]:
        async with AsyncSessionLocal() as session:
            result = await session.execute(
                select(Collection).where(Collection.namespace_id == namespace_id)
            )
            return list(result.scalars().all())

    async def get_collection(self, collection_id: uuid.UUID) -> Collection:
        async with AsyncSessionLocal() as session:
            collection = await session.get(Collection, collection_id)
            if not collection:
                raise ValueError(f"Collection {collection_id} not found")
            return collection

    # ── Ingestion ──

    async def ingest_chunk(
        self,
        text: str,
        collection_id: uuid.UUID,
        namespace_id: uuid.UUID,
    ) -> ChunkIngestionResult:
        collection = await self.get_collection(collection_id)
        return await ingest_collection_chunk(
            text=text,
            collection=collection,
            namespace_id=namespace_id,
            chunk_index=0,
        )

    async def enqueue_document_ingestion(
        self,
        text: str,
        collection_id: uuid.UUID,
        namespace_id: uuid.UUID,
    ) -> DocumentIngestionResult:
        collection = await self.get_collection(collection_id)
        self._enforce_namespace(collection, namespace_id)
        return await enqueue_document_ingestion_job(
            text=text,
            collection_id=collection_id,
            namespace_id=namespace_id,
        )

    async def ingest_document_pipeline(self, job_id: uuid.UUID):
        """Main pipeline — delegates to ingestion submodule."""
        await ingest_document_pipeline(job_id)

    async def process_single_chunk(
        self, job_id: str, chunk_index: int
    ) -> None:
        """Process a single chunk — called by run_chunk worker."""
        await process_single_chunk(job_id, chunk_index)

    async def update_chunk_status(
        self, job_id: uuid.UUID, chunk_index: int, status: str, error: str | None = None
    ) -> None:
        await _update_chunk_status(job_id, chunk_index, status, error=error)

    # ── Query ──

    async def query(
        self,
        question: str,
        collection_id: uuid.UUID,
        namespace_id: uuid.UUID,
        mode: str | None = None,
        llm_profile_id: uuid.UUID | None = None,
    ) -> QueryResult:
        collection = await self.get_collection(collection_id)
        self._enforce_namespace(collection, namespace_id)
        effective_mode = mode or collection.default_query_mode or "local"
        effective_llm_profile_id = llm_profile_id or collection.llm_profile_id

        if collection.strategy == "vector":
            return await vector_query(
                question, collection, namespace_id, effective_mode,
                llm_profile_id=effective_llm_profile_id,
            )
        if collection.strategy == "custom_graph_rag":
            return await graph_rag_query(
                question,
                collection,
                namespace_id,
                llm_profile_id=effective_llm_profile_id,
            )
        if collection.strategy == "light_rag":
            return await lightrag_query(
                question, collection, namespace_id, effective_mode,
                llm_profile_id=effective_llm_profile_id,
            )

        return QueryResult(
            response="",
            entities_used=[],
            relationships_used=[],
            mode=effective_mode,
        )

    # ── Jobs ──

    async def get_job(self, job_id: uuid.UUID) -> dict[str, Any]:
        async with AsyncSessionLocal() as session:
            job = await session.get(Job, job_id)
            if not job:
                raise ValueError(f"Job {job_id} not found")
            return {
                "id": str(job.id),
                "type": job.job_type,
                "status": job.status,
                "progress_percent": job.progress_percent,
                "error": job.error,
                "created_at": job.created_at.isoformat() if job.created_at else None,
                "started_at": job.started_at.isoformat() if job.started_at else None,
                "completed_at": (
                    job.completed_at.isoformat() if job.completed_at else None
                ),
                "chunks_total": job.chunks_total,
                "chunks_completed": job.chunks_completed,
            }

    async def list_jobs(
        self,
        namespace_id: uuid.UUID,
        *,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        async with AsyncSessionLocal() as session:
            result = await session.execute(
                select(Job)
                .where(Job.namespace_id == namespace_id)
                .order_by(Job.created_at.desc())
                .limit(limit)
            )
            jobs = list(result.scalars().all())
            return [
                {
                    "id": str(job.id),
                    "type": job.job_type,
                    "status": job.status,
                    "progress_percent": job.progress_percent,
                    "chunks_total": job.chunks_total,
                    "chunks_completed": job.chunks_completed,
                    "collection_id": (
                        str(job.collection_id) if job.collection_id else None
                    ),
                    "created_at": (
                        job.created_at.isoformat() if job.created_at else None
                    ),
                    "error": job.error,
                }
                for job in jobs
            ]

    async def update_job_status(
        self,
        job_id: uuid.UUID,
        status: str,
        progress_percent: int | None = None,
        error: str | None = None,
    ):
        async with AsyncSessionLocal() as session:
            job = await session.get(Job, job_id)
            if not job:
                return
            job.status = status  # type: ignore[assignment]
            if progress_percent is not None:
                job.progress_percent = progress_percent
            if error:
                job.error = error
            if status == "running" and not job.started_at:
                job.started_at = datetime.now(UTC)
            if status in ("completed", "failed", "cancelled"):
                job.completed_at = datetime.now(UTC)
            await session.commit()

    async def append_job_event(
        self, job_id: uuid.UUID, event_type: str, payload: dict | None = None
    ):
        async with AsyncSessionLocal() as session:
            event = JobEvent(job_id=job_id, event_type=event_type, payload=payload)
            session.add(event)
            await session.commit()

    # ── Internal ──

    def _enforce_namespace(self, collection: Collection, namespace_id: uuid.UUID):
        if collection.namespace_id != namespace_id:
            raise PermissionError(
                f"Collection {collection.id} does not belong to namespace "
                f"{namespace_id}"
            )

__all__ = [
    "GraphService",
    "ChunkIngestionResult",
    "DocumentIngestionResult",
    "QueryResult",
    "ingest_collection_chunk",
    "ingest_document_pipeline",
    "process_single_chunk",
    "update_chunk_status",
    "enqueue_document_ingestion_job",
    "fan_out_chunks",
    "increment_chunk_counter",
    "deterministic_uuid",
    "get_graph_storage",
    "graph_rag_query",
    "lightrag_query",
    "vector_query",
    "generate_vector_answer",
    "extract_keywords",
    "fallback_keywords",
]
