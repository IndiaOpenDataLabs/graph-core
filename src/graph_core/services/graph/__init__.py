"""Graph RAG service — thin public API over ingestion/ and query/ submodules.

GraphService delegates to ingestion/ and query/ submodules. All domain
logic lives there; this package provides the familiar class API used by
API routes and workers.
"""

import uuid
from datetime import UTC, datetime
from typing import Any, Literal

from sqlalchemy import delete, func, select

from graph_core.database import AsyncSessionLocal
from graph_core.embedding import get_embedding_provider
from graph_core.models.chat import ChatSession, ChatTurn
from graph_core.models.collection import Collection
from graph_core.models.credential import Credential
from graph_core.models.job import Job, JobEvent
from graph_core.models.namespace import Namespace
from graph_core.models.profile import Profile
from graph_core.services.crypto import CredentialCrypto
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
from graph_core.storage.graph_storage import FalkorDBGraphStorage
from graph_core.storage.vector_store import VectorStore
from graph_core.storage.vector_tables import (
    create_all_tables,
    drop_all_tables,
)

_crypto = CredentialCrypto()


async def _resolve_credential(
    session, profile: Profile
) -> tuple[str | None, str | None]:
    if profile.credential_id is None:
        return None, None
    credential = await session.get(Credential, profile.credential_id)
    if not credential:
        raise ValueError(f"Credential {profile.credential_id} not found")
    return _crypto.decrypt(credential.encrypted_secret), credential.base_url


class GraphService:
    """Thin orchestration class. All logic delegates to submodules."""

    def __init__(self):
        self._sanitizer = TextSanitizer()
        self._vector_store = VectorStore()
        self._graph_rag_vectors = GraphRAGVectorStore()

    # ── Collections ──

    @staticmethod
    def _chat_graph_name(chat_id: uuid.UUID) -> str:
        return f"chat_{str(chat_id).replace('-', '')}"

    def _chat_storage(self, chat_id: uuid.UUID) -> FalkorDBGraphStorage:
        return FalkorDBGraphStorage(self._chat_graph_name(chat_id))

    @staticmethod
    def _chat_turn_content(question: str, response: str) -> str:
        return f"Question:\n{question}\n\nResponse:\n{response}"

    async def create_chat_session(
        self,
        collection_id: uuid.UUID,
        namespace_id: uuid.UUID,
        *,
        title: str | None = None,
    ) -> ChatSession:
        collection = await self.get_collection(collection_id)
        self._enforce_namespace(collection, namespace_id)
        async with AsyncSessionLocal() as session:
            chat = ChatSession(
                collection_id=collection_id,
                namespace_id=namespace_id,
                title=title,
            )
            session.add(chat)
            await session.commit()
            await session.refresh(chat)
            return chat

    async def list_chat_sessions(
        self,
        collection_id: uuid.UUID,
        namespace_id: uuid.UUID,
        *,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        collection = await self.get_collection(collection_id)
        self._enforce_namespace(collection, namespace_id)
        async with AsyncSessionLocal() as session:
            result = await session.execute(
                select(
                    ChatSession,
                    func.count(ChatTurn.id).label("turn_count"),
                )
                .outerjoin(ChatTurn, ChatTurn.chat_id == ChatSession.id)
                .where(ChatSession.collection_id == collection_id)
                .where(ChatSession.namespace_id == namespace_id)
                .group_by(ChatSession.id)
                .order_by(ChatSession.updated_at.desc())
                .limit(limit)
            )
            rows = result.all()
            return [
                {
                    "id": str(chat.id),
                    "collection_id": str(chat.collection_id),
                    "title": chat.title,
                    "turn_count": int(turn_count or 0),
                    "created_at": (
                        chat.created_at.isoformat() if chat.created_at else None
                    ),
                    "updated_at": (
                        chat.updated_at.isoformat() if chat.updated_at else None
                    ),
                }
                for chat, turn_count in rows
            ]

    async def _get_chat_session(
        self,
        chat_id: uuid.UUID,
        collection_id: uuid.UUID,
        namespace_id: uuid.UUID,
    ) -> ChatSession:
        async with AsyncSessionLocal() as session:
            chat = await session.get(ChatSession, chat_id)
            if not chat:
                raise ValueError(f"Chat session {chat_id} not found")
            if chat.collection_id != collection_id or chat.namespace_id != namespace_id:
                raise PermissionError("Chat session does not belong to collection")
            return chat

    async def _load_chat_context(
        self,
        collection: Collection,
        question: str,
        chat_id: uuid.UUID,
    ) -> str:
        embedding_provider = await self._resolve_collection_embedding_provider(
            collection
        )
        query_embedding = await embedding_provider.embed_query(question)
        hits = await self._vector_store.query_chunks(
            collection_id=collection.id,
            query_embedding=query_embedding,
            top_k=3,
            metadata_filters={
                "memory_type": "chat_turn",
                "chat_id": str(chat_id),
            },
        )
        if not hits:
            return ""

        memory_lines: list[str] = []
        seen_turn_ids: set[str] = set()
        turn_ids: list[str] = []
        for hit in hits:
            metadata = hit.get("metadata") or {}
            turn_id = str(metadata.get("turn_id") or "").strip()
            if turn_id and turn_id not in seen_turn_ids:
                seen_turn_ids.add(turn_id)
                turn_ids.append(turn_id)
                memory_lines.append(hit["content"])

        storage = self._chat_storage(chat_id)
        neighbor_ids: set[str] = set()
        for turn_id in turn_ids:
            for source_id, target_id in await storage.get_node_edges(turn_id):
                neighbor = target_id if source_id == turn_id else source_id
                if neighbor and neighbor not in seen_turn_ids:
                    neighbor_ids.add(neighbor)

        neighbor_rows: list[tuple[int, str]] = []
        for neighbor_id in neighbor_ids:
            node = await storage.get_node(neighbor_id)
            if not node:
                continue
            question_text = str(node.get("question") or "").strip()
            response_text = str(node.get("response") or "").strip()
            if not question_text and not response_text:
                continue
            turn_index = int(node.get("turn_index") or 0)
            neighbor_rows.append(
                (
                    turn_index,
                    self._chat_turn_content(question_text, response_text),
                )
            )

        for _, content in sorted(neighbor_rows, key=lambda item: item[0])[:2]:
            if content not in memory_lines:
                memory_lines.append(content)

        return "\n\n".join(memory_lines[:4])

    async def _record_chat_turn(
        self,
        collection: Collection,
        namespace_id: uuid.UUID,
        chat_id: uuid.UUID,
        question: str,
        response: str,
        mode: str | None,
    ) -> None:
        embedding_provider = await self._resolve_collection_embedding_provider(
            collection
        )
        content = self._chat_turn_content(question, response)
        async with AsyncSessionLocal() as session:
            chat = await session.get(ChatSession, chat_id)
            if not chat:
                raise ValueError(f"Chat session {chat_id} not found")
            if chat.collection_id != collection.id or chat.namespace_id != namespace_id:
                raise PermissionError("Chat session does not belong to collection")

            max_turn = await session.scalar(
                select(func.max(ChatTurn.turn_index)).where(ChatTurn.chat_id == chat_id)
            )
            turn_index = int(max_turn or 0) + 1
            turn = ChatTurn(
                chat_id=chat_id,
                collection_id=collection.id,
                turn_index=turn_index,
                question=question,
                response=response,
                mode=mode,
            )
            chat.updated_at = datetime.now(UTC)
            session.add(turn)
            await session.commit()
            await session.refresh(turn)

            previous_turn_id = await session.scalar(
                select(ChatTurn.id)
                .where(ChatTurn.chat_id == chat_id)
                .where(ChatTurn.turn_index == turn_index - 1)
            )

        embedding = await embedding_provider.embed_query(content)
        await self._vector_store.upsert_chunks(
            namespace_id=namespace_id,
            collection_id=collection.id,
            chunks=[
                {
                    "chunk_hash": f"chat:{chat_id}:{turn_index}",
                    "chunk_index": 0,
                    "content": content,
                    "token_count": len(content.split()),
                    "metadata": {
                        "memory_type": "chat_turn",
                        "chat_id": str(chat_id),
                        "turn_id": str(turn.id),
                        "turn_index": str(turn_index),
                    },
                    "embedding": embedding,
                }
            ],
        )

        storage = self._chat_storage(chat_id)
        await storage.upsert_node(
            str(turn.id),
            {
                "id": str(turn.id),
                "name": f"Turn {turn_index}",
                "collection_id": str(collection.id),
                "type": "turn",
                "description": question[:256],
                "question": question,
                "response": response,
                "chat_id": str(chat_id),
                "turn_index": turn_index,
            },
        )
        if previous_turn_id:
            prev_id = str(previous_turn_id)
            await storage.upsert_edge(
                prev_id,
                str(turn.id),
                {
                    "id": f"{prev_id}__{turn.id}",
                    "weight": 1,
                    "collection_id": str(collection.id),
                    "description": "next turn",
                    "keywords": ["chat", "next"],
                },
            )

    async def _resolve_collection_embedding_provider(
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
            api_key, cred_base_url = await _resolve_credential(session, profile)
            base_url = profile.base_url or cred_base_url
            return get_embedding_provider(
                provider_name=profile.provider,
                model=profile.model,
                dimensions=profile.dimensions,
                api_key=api_key,
                base_url=base_url,
                profile_id=str(profile.id),
                max_concurrent_calls=profile.max_concurrent_calls,
            )

    async def create_collection(
        self,
        name: str,
        namespace_id: uuid.UUID,
        strategy: Literal["vector", "custom_graph_rag", "light_rag"] = "vector",
        embedding_profile_id: uuid.UUID | None = None,
        llm_profile_id: uuid.UUID | None = None,
        default_query_mode: str | None = None,
        gleaning_passes: int = 1,
    ) -> Collection:
        if gleaning_passes < 0:
            raise ValueError("Gleaning passes must be 0 or greater")
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
                gleaning_passes=gleaning_passes,
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
        gleaning_passes: int | None = None,
        clear_llm_profile: bool = False,
        clear_default_query_mode: bool = False,
    ) -> Collection:
        async with AsyncSessionLocal() as session:
            collection = await session.get(Collection, collection_id)
            if not collection:
                raise ValueError(f"Collection {collection_id} not found")
            self._enforce_namespace(collection, namespace_id)
            if gleaning_passes is not None and gleaning_passes < 0:
                raise ValueError("Gleaning passes must be 0 or greater")

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
            if gleaning_passes is not None:
                collection.gleaning_passes = gleaning_passes

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
        chat_id: uuid.UUID | None = None,
    ) -> QueryResult:
        collection = await self.get_collection(collection_id)
        self._enforce_namespace(collection, namespace_id)
        if collection.strategy == "custom_graph_rag":
            default_mode = "mix"
        else:
            default_mode = "local"
        effective_mode = mode or collection.default_query_mode or default_mode
        effective_llm_profile_id = llm_profile_id or collection.llm_profile_id
        chat_context = ""
        if chat_id is not None:
            await self._get_chat_session(chat_id, collection_id, namespace_id)
            chat_context = await self._load_chat_context(collection, question, chat_id)
        retrieval_question = question
        if chat_context:
            retrieval_question = (
                "Relevant prior chat context:\n"
                f"{chat_context}\n\n"
                f"Current question:\n{question}"
            )

        if collection.strategy == "vector":
            result = await vector_query(
                retrieval_question,
                collection,
                namespace_id,
                effective_mode,
                llm_profile_id=effective_llm_profile_id,
            )
        elif collection.strategy == "custom_graph_rag":
            result = await graph_rag_query(
                retrieval_question,
                collection,
                namespace_id,
                effective_mode,
                llm_profile_id=effective_llm_profile_id,
            )
        elif collection.strategy == "light_rag":
            result = await lightrag_query(
                retrieval_question,
                collection,
                namespace_id,
                effective_mode,
                llm_profile_id=effective_llm_profile_id,
            )
        else:
            result = QueryResult(
                response="",
                entities_used=[],
                relationships_used=[],
                mode=effective_mode,
            )

        if chat_id is not None:
            await self._record_chat_turn(
                collection=collection,
                namespace_id=namespace_id,
                chat_id=chat_id,
                question=question,
                response=result.response,
                mode=result.mode,
            )
            result.chat_id = str(chat_id)
        return result

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
