"""GraphService — internal orchestration for all graph operations.

This class has no transport dependencies. It is called by API routes,
MCP tools, and background workers. All dependencies are injected.
"""

import uuid
from dataclasses import dataclass
from typing import Any, Literal

from sqlalchemy import select
from sqlalchemy.orm import selectinload

from graph_core.config import settings
from graph_core.database import AsyncSessionLocal
from graph_core.embedding import get_embedding_provider
from graph_core.llm import LocalEchoLLMProvider, get_llm_provider
from graph_core.models.collection import Collection
from graph_core.models.credential import Credential
from graph_core.models.ingestion import IngestionRecord
from graph_core.models.job import Job, JobEvent
from graph_core.models.namespace import Namespace
from graph_core.models.profile import Profile
from graph_core.services.chunking import TokenChunker
from graph_core.services.crypto import CredentialCrypto
from graph_core.services.sanitizer import TextSanitizer
from graph_core.storage.vector_store import VectorStore


@dataclass
class ChunkIngestionResult:
    chunk_hash: str
    entity_count: int
    relationship_count: int


@dataclass
class DocumentIngestionResult:
    job_id: uuid.UUID
    status: str


@dataclass
class QueryResult:
    response: str
    entities_used: list[str]
    relationships_used: list[str]
    mode: str


class GraphService:
    def __init__(self):
        self._sanitizer = TextSanitizer()
        self._chunker = TokenChunker(
            chunk_size_tokens=settings.chunk_size_tokens,
            chunk_overlap_tokens=settings.chunk_overlap_tokens,
        )
        self._crypto = CredentialCrypto()
        self._vector_store = VectorStore()

    # ── Collections ──

    async def create_collection(
        self,
        name: str,
        namespace_id: uuid.UUID,
        strategy: Literal["vector", "custom_graph_rag", "light_rag"] = "vector",
        embedding_profile_id: uuid.UUID | None = None,
        default_query_mode: str | None = None,
    ) -> Collection:
        """Create a new collection bound to a namespace and embedding profile."""
        async with AsyncSessionLocal() as session:
            # Verify namespace exists
            ns = await session.get(Namespace, namespace_id)
            if not ns:
                raise ValueError(f"Namespace {namespace_id} not found")
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

            collection = Collection(
                name=name,
                namespace_id=namespace_id,
                strategy=strategy,
                embedding_profile_id=embedding_profile_id,
                default_query_mode=default_query_mode,
            )
            session.add(collection)
            await session.commit()
            await session.refresh(collection)
            return collection

    async def list_collections(self, namespace_id: uuid.UUID) -> list[Collection]:
        async with AsyncSessionLocal() as session:
            result = await session.execute(
                select(Collection).where(Collection.namespace_id == namespace_id)
            )
            return list(result.scalars().all())

    async def get_collection(self, collection_id: uuid.UUID) -> Collection:
        async with AsyncSessionLocal() as session:
            collection = await session.get(Collection, collection_id, options=[selectinload(Collection.namespace)])
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
        """Ingest a single chunk of text. Synchronous from caller's perspective."""
        collection = await self.get_collection(collection_id)
        return await self._ingest_collection_chunk(
            text=text,
            collection=collection,
            namespace_id=namespace_id,
            chunk_index=0,
        )

    async def _ingest_collection_chunk(
        self,
        text: str,
        collection: Collection,
        namespace_id: uuid.UUID,
        chunk_index: int,
    ) -> ChunkIngestionResult:
        self._enforce_namespace(collection, namespace_id)

        # Sanitize
        sanitized_text, report = self._sanitizer.sanitize(text, str(namespace_id))
        chunk_hash = self._sanitizer.chunk_hash(sanitized_text)

        # Strategy dispatch
        if collection.strategy == "vector":
            result = await self._ingest_vector_chunk(
                sanitized_text,
                collection,
                chunk_hash,
                report,
                chunk_index=chunk_index,
            )
        elif collection.strategy == "light_rag":
            result = await self._ingest_lightrag_chunk(sanitized_text, collection, chunk_hash, report)
        else:
            result = await self._ingest_graph_chunk(sanitized_text, collection, chunk_hash, report)

        # Write ledger record
        await self._write_ledger(collection, chunk_hash, report, result)
        return result

    async def enqueue_document_ingestion(
        self,
        text: str,
        collection_id: uuid.UUID,
        namespace_id: uuid.UUID,
    ) -> DocumentIngestionResult:
        """Queue a document ingestion job. Returns immediately with job_id."""
        collection = await self.get_collection(collection_id)
        self._enforce_namespace(collection, namespace_id)

        async with AsyncSessionLocal() as session:
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

        # Enqueue Dramatiq worker
        from graph_core.workers.ingestion import run_ingestion

        run_ingestion.send(str(job.id))

        return DocumentIngestionResult(job_id=job.id, status="pending")

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
        if collection.strategy == "vector":
            return await self._query_vector(
                question,
                collection,
                namespace_id,
                effective_mode,
                llm_profile_id=llm_profile_id,
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
                "completed_at": job.completed_at.isoformat() if job.completed_at else None,
            }

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
            from datetime import UTC, datetime

            if status == "running" and not job.started_at:
                job.started_at = datetime.now(UTC)
            if status in ("completed", "failed", "cancelled"):
                job.completed_at = datetime.now(UTC)
            await session.commit()

    async def append_job_event(self, job_id: uuid.UUID, event_type: str, payload: dict | None = None):
        async with AsyncSessionLocal() as session:
            event = JobEvent(job_id=job_id, event_type=event_type, payload=payload)
            session.add(event)
            await session.commit()

    async def ingest_document_pipeline(self, job_id: uuid.UUID):
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

        chunks = self._chunker.chunk_text(text)
        total_chunks = max(len(chunks), 1)

        if not chunks:
            await self.update_job_status(job_id, "completed", progress_percent=100)
            return

        for index, chunk in enumerate(chunks, start=1):
            await self._ingest_collection_chunk(
                text=chunk,
                collection=collection,
                namespace_id=collection.namespace_id,
                chunk_index=index - 1,
            )
            progress = int(index * 100 / total_chunks)
            await self.update_job_status(job_id, "running", progress_percent=progress)
            await self.append_job_event(
                job_id,
                "chunk_completed",
                {"chunk_index": index - 1, "total_chunks": total_chunks},
            )

    # ── Internal ──

    def _enforce_namespace(self, collection: Collection, namespace_id: uuid.UUID):
        if collection.namespace_id != namespace_id:
            raise PermissionError(
                f"Collection {collection.id} does not belong to namespace {namespace_id}"
            )

    async def _ingest_vector_chunk(
        self,
        text: str,
        collection: Collection,
        chunk_hash: str,
        report,
        chunk_index: int,
    ) -> ChunkIngestionResult:
        embedding_provider = await self._get_embedding_provider_for_collection(
            collection
        )
        embedding = await embedding_provider.embed_query(text)
        token_count = len(text.split())
        await self._vector_store.upsert_chunks(
            namespace_id=collection.namespace_id,
            collection_id=collection.id,
            chunks=[
                {
                    "chunk_hash": chunk_hash,
                    "chunk_index": chunk_index,
                    "content": text,
                    "token_count": token_count,
                    "metadata": {
                        "strategy": collection.strategy,
                        "default_query_mode": collection.default_query_mode,
                    },
                    "embedding": embedding,
                }
            ],
        )
        return ChunkIngestionResult(chunk_hash=chunk_hash, entity_count=0, relationship_count=0)

    async def _ingest_lightrag_chunk(
        self, text: str, collection: Collection, chunk_hash: str, report
    ) -> ChunkIngestionResult:
        # TODO: LightRAG.insert(text) → library-managed storage → returns extraction summary
        return ChunkIngestionResult(chunk_hash=chunk_hash, entity_count=0, relationship_count=0)

    async def _ingest_graph_chunk(
        self, text: str, collection: Collection, chunk_hash: str, report
    ) -> ChunkIngestionResult:
        # TODO: sanitize → chunk → LLM extraction → entity resolution → embed → FalkorDB + ChromaDB
        return ChunkIngestionResult(chunk_hash=chunk_hash, entity_count=0, relationship_count=0)

    async def _write_ledger(
        self,
        collection: Collection,
        chunk_hash: str,
        report,
        result: ChunkIngestionResult,
    ):
        async with AsyncSessionLocal() as session:
            record = IngestionRecord(
                collection_id=collection.id,
                chunk_hash=chunk_hash,
                strategy=collection.strategy,
                entity_count=result.entity_count,
                relationship_count=result.relationship_count,
                sanitization_flags={"severity": report.severity, "details": report.details} if report.severity != "none" else None,
            )
            session.add(record)
            await session.commit()

    async def _query_vector(
        self,
        question: str,
        collection: Collection,
        namespace_id: uuid.UUID,
        mode: str,
        llm_profile_id: uuid.UUID | None = None,
    ) -> QueryResult:
        embedding_provider = await self._get_embedding_provider_for_collection(
            collection
        )
        query_embedding = await embedding_provider.embed_query(question)
        results = await self._vector_store.query_chunks(
            collection_id=collection.id,
            query_embedding=query_embedding,
            top_k=settings.vector_query_top_k,
        )
        chunks = [result.content for result in results]
        response = await self._generate_vector_answer(
            question=question,
            chunks=chunks,
            namespace_id=namespace_id,
            llm_profile_id=llm_profile_id,
        )
        return QueryResult(
            response=response,
            entities_used=[],
            relationships_used=[],
            mode=mode,
        )

    async def _get_embedding_provider_for_collection(
        self,
        collection: Collection,
    ):
        if collection.embedding_profile_id is None:
            return get_embedding_provider()

        async with AsyncSessionLocal() as session:
            profile = await session.get(Profile, collection.embedding_profile_id)
            if not profile:
                raise ValueError(
                    f"Embedding profile {collection.embedding_profile_id} not found"
                )
            api_key = await self._get_profile_api_key(session, profile)
            return get_embedding_provider(
                provider_name=profile.provider,
                model=profile.model,
                dimensions=profile.dimensions,
                api_key=api_key,
            )

    async def _generate_vector_answer(
        self,
        *,
        question: str,
        chunks: list[str],
        namespace_id: uuid.UUID,
        llm_profile_id: uuid.UUID | None,
    ) -> str:
        if not chunks:
            return ""

        llm_provider = await self._get_llm_provider(
            namespace_id=namespace_id,
            llm_profile_id=llm_profile_id,
        )
        if isinstance(llm_provider, LocalEchoLLMProvider):
            return chunks[0]

        context = "\n\n".join(
            f"Chunk {index + 1}:\n{chunk}"
            for index, chunk in enumerate(chunks)
        )
        return await llm_provider.chat(
            [
                {
                    "role": "system",
                    "content": (
                        "Answer the question using only the provided context. "
                        "If the answer is not present, say so."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Question:\n{question}\n\n"
                        f"Context:\n{context}\n\n"
                        "Answer using the context above."
                    ),
                },
            ]
        )

    async def _get_llm_provider(
        self,
        *,
        namespace_id: uuid.UUID,
        llm_profile_id: uuid.UUID | None,
    ):
        if llm_profile_id is None:
            return get_llm_provider()

        async with AsyncSessionLocal() as session:
            profile = await session.get(Profile, llm_profile_id)
            if not profile or profile.namespace_id != namespace_id:
                raise ValueError("LLM profile not found in namespace")
            if profile.kind != "llm":
                raise ValueError("Profile kind must be llm")
            api_key = await self._get_profile_api_key(session, profile)
            return get_llm_provider(
                provider_name=profile.provider,
                model=profile.model,
                api_key=api_key,
            )

    async def _get_profile_api_key(self, session, profile: Profile) -> str | None:
        if profile.credential_id is None:
            return None

        credential = await session.get(Credential, profile.credential_id)
        if not credential:
            raise ValueError(f"Credential {profile.credential_id} not found")
        return self._crypto.decrypt(credential.encrypted_secret)
