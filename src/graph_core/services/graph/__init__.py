"""Graph RAG service — thin public API over ingestion/ and query/ submodules.

GraphService delegates to ingestion/ and query/ submodules. All domain
logic lives there; this package provides the familiar class API used by
API routes and workers.
"""

import hashlib
import json
import uuid
from datetime import UTC, datetime
from typing import Any, Literal

from sqlalchemy import delete, func, select

from graph_core.database import AsyncSessionLocal
from graph_core.embedding import get_embedding_provider
from graph_core.llm import get_llm_provider
from graph_core.models.chat import ChatMessage, ChatSession
from graph_core.models.collection import Collection
from graph_core.models.credential import Credential
from graph_core.models.job import Job, JobEvent
from graph_core.models.namespace import Namespace
from graph_core.models.profile import Profile
from graph_core.services.crypto import CredentialCrypto
from graph_core.services.graph.analytics import analyze_collection_graph
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
from graph_core.services.graph_rag.extractor import (
    ExtractedRelationship,
    LLMGraphExtractor,
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
                    func.count(ChatMessage.id)
                    .filter(ChatMessage.role == "user")
                    .label("turn_count"),
                )
                .outerjoin(ChatMessage, ChatMessage.chat_id == ChatSession.id)
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

    @staticmethod
    def _format_chat_message(role: str, content: str, turn_index: int) -> str:
        speaker = "User" if role == "user" else "Assistant"
        return f"{speaker} (turn {turn_index}):\n{content}"

    @staticmethod
    def _parse_json_list(value: Any) -> list[str]:
        if value is None:
            return []
        if isinstance(value, list):
            return [str(item) for item in value if str(item).strip()]
        if isinstance(value, str):
            try:
                parsed = json.loads(value)
            except json.JSONDecodeError:
                return [value] if value.strip() else []
            if isinstance(parsed, list):
                return [str(item) for item in parsed if str(item).strip()]
        return []

    @staticmethod
    def _semantic_entity_id(name: str) -> str:
        normalized = " ".join(name.strip().lower().split())
        return f"entity:{normalized}"

    @staticmethod
    def _semantic_relationship_id(
        source_name: str,
        target_name: str,
    ) -> str:
        return (
            f"{GraphService._semantic_entity_id(source_name)}"
            f"__{GraphService._semantic_entity_id(target_name)}"
        )

    @staticmethod
    def _chat_chunk_hash(*parts: object) -> str:
        raw = "::".join(str(part) for part in parts)
        return hashlib.md5(raw.encode("utf-8")).hexdigest()

    async def _load_recent_chat_messages(
        self,
        chat_id: uuid.UUID,
        *,
        limit: int = 6,
    ) -> list[ChatMessage]:
        async with AsyncSessionLocal() as session:
            messages = (
                await session.execute(
                    select(ChatMessage)
                    .where(ChatMessage.chat_id == chat_id)
                    .order_by(ChatMessage.message_index.desc())
                    .limit(limit)
                )
            ).scalars().all()
        return list(reversed(messages))

    async def _load_chat_message_rows(
        self,
        chat_id: uuid.UUID,
        message_ids: set[str],
    ) -> list[ChatMessage]:
        if not message_ids:
            return []
        async with AsyncSessionLocal() as session:
            rows = (
                await session.execute(
                    select(ChatMessage)
                    .where(ChatMessage.chat_id == chat_id)
                    .where(ChatMessage.id.in_([uuid.UUID(mid) for mid in message_ids]))
                    .order_by(ChatMessage.message_index)
                )
            ).scalars().all()
        return rows

    @staticmethod
    def _format_semantic_entity(name: str, description: str) -> str:
        return f"Entity: {name}. {description}".strip()

    @staticmethod
    def _format_semantic_relationship(rel: ExtractedRelationship) -> str:
        keywords = ", ".join(rel.keywords[:5])
        suffix = f" Keywords: {keywords}." if keywords else ""
        return (
            f"Relationship: {rel.source_name} -> {rel.target_name}. "
            f"{rel.description}{suffix}"
        ).strip()

    async def _extract_chat_semantics(
        self,
        collection: Collection,
        text: str,
        llm_profile_id: uuid.UUID | None,
    ):
        llm_provider = await self._resolve_collection_llm_provider(
            collection,
            llm_profile_id,
        )
        extractor = LLMGraphExtractor(llm_provider)
        return await extractor.extract_with_gleaning(
            text=text,
            max_gleaning=max(0, int(collection.gleaning_passes or 0)),
        )

    async def _merge_chat_semantic_node(
        self,
        storage: FalkorDBGraphStorage,
        collection: Collection,
        *,
        node_id: str,
        name: str,
        description: str,
        source_message_id: str,
        source_role: str,
    ) -> None:
        existing = await storage.get_node(node_id)
        source_message_ids = set(
            self._parse_json_list(existing.get("source_message_ids"))
            if existing
            else []
        )
        source_roles = set(
            self._parse_json_list(existing.get("source_roles")) if existing else []
        )
        source_message_ids.add(source_message_id)
        source_roles.add(source_role)
        existing_description = (
            str(existing.get("description") or "") if existing else ""
        )
        final_description = (
            description
            if len(description) >= len(existing_description)
            else existing_description
        )
        await storage.upsert_node(
            node_id,
            {
                "id": node_id,
                "name": name,
                "collection_id": str(collection.id),
                "type": "semantic_entity",
                "description": final_description,
                "source_message_ids": json.dumps(sorted(source_message_ids)),
                "source_roles": json.dumps(sorted(source_roles)),
            },
        )

    async def _merge_chat_semantic_edge(
        self,
        storage: FalkorDBGraphStorage,
        collection: Collection,
        *,
        relationship: ExtractedRelationship,
        source_message_id: str,
        source_role: str,
    ) -> None:
        source_id = self._semantic_entity_id(relationship.source_name)
        target_id = self._semantic_entity_id(relationship.target_name)
        existing = await storage.get_edge(source_id, target_id)
        source_message_ids = set(
            self._parse_json_list(existing.get("source_message_ids"))
            if existing
            else []
        )
        source_roles = set(
            self._parse_json_list(existing.get("source_roles")) if existing else []
        )
        source_message_ids.add(source_message_id)
        source_roles.add(source_role)
        existing_keywords = (
            self._parse_json_list(existing.get("keywords")) if existing else []
        )
        merged_keywords = sorted(
            {
                *(keyword.strip() for keyword in existing_keywords if keyword.strip()),
                *(
                    keyword.strip()
                    for keyword in relationship.keywords
                    if keyword.strip()
                ),
            }
        )
        existing_description = (
            str(existing.get("description") or "") if existing else ""
        )
        final_description = (
            relationship.description
            if len(relationship.description) >= len(existing_description)
            else existing_description
        )
        await storage.upsert_edge(
            source_id,
            target_id,
            {
                "id": self._semantic_relationship_id(
                    relationship.source_name,
                    relationship.target_name,
                ),
                "weight": max(
                    float(existing.get("weight", 0.0)) if existing else 0.0,
                    float(relationship.weight),
                ),
                "collection_id": str(collection.id),
                "description": final_description,
                "keywords": merged_keywords,
                "source_message_ids": json.dumps(sorted(source_message_ids)),
                "source_roles": json.dumps(sorted(source_roles)),
            },
        )

    async def _store_chat_semantic_memory(
        self,
        collection: Collection,
        namespace_id: uuid.UUID,
        chat_id: uuid.UUID,
        message: ChatMessage,
        embedding_provider,
        llm_profile_id: uuid.UUID | None,
    ) -> None:
        extraction = await self._extract_chat_semantics(
            collection,
            message.content,
            llm_profile_id,
        )
        if not extraction.entities and not extraction.relationships:
            return

        storage = self._chat_storage(chat_id)
        vector_chunks: list[dict[str, Any]] = []

        for index, entity in enumerate(extraction.entities):
            node_id = self._semantic_entity_id(entity.name)
            await self._merge_chat_semantic_node(
                storage,
                collection,
                node_id=node_id,
                name=entity.name,
                description=entity.description,
                source_message_id=str(message.id),
                source_role=message.role,
            )
            entity_content = self._format_semantic_entity(
                entity.name,
                entity.description,
            )
            vector_chunks.append(
                {
                    "chunk_hash": self._chat_chunk_hash(
                        "chat_semantic",
                        chat_id,
                        message.id,
                        "entity",
                        index,
                    ),
                    "chunk_index": index,
                    "content": entity_content,
                    "token_count": len(entity_content.split()),
                    "metadata": {
                        "memory_type": "chat_semantic",
                        "chat_id": str(chat_id),
                        "source_message_id": str(message.id),
                        "source_role": message.role,
                        "semantic_kind": "entity",
                        "semantic_id": node_id,
                        "entity_name": entity.name,
                    },
                    "embedding": await embedding_provider.embed_query(entity_content),
                }
            )

        relationship_offset = len(vector_chunks)
        for index, relationship in enumerate(extraction.relationships):
            await self._merge_chat_semantic_node(
                storage,
                collection,
                node_id=self._semantic_entity_id(relationship.source_name),
                name=relationship.source_name,
                description="",
                source_message_id=str(message.id),
                source_role=message.role,
            )
            await self._merge_chat_semantic_node(
                storage,
                collection,
                node_id=self._semantic_entity_id(relationship.target_name),
                name=relationship.target_name,
                description="",
                source_message_id=str(message.id),
                source_role=message.role,
            )
            await self._merge_chat_semantic_edge(
                storage,
                collection,
                relationship=relationship,
                source_message_id=str(message.id),
                source_role=message.role,
            )
            rel_content = self._format_semantic_relationship(relationship)
            vector_chunks.append(
                {
                    "chunk_hash": self._chat_chunk_hash(
                        "chat_semantic",
                        chat_id,
                        message.id,
                        "relationship",
                        index,
                    ),
                    "chunk_index": relationship_offset + index,
                    "content": rel_content,
                    "token_count": len(rel_content.split()),
                    "metadata": {
                        "memory_type": "chat_semantic",
                        "chat_id": str(chat_id),
                        "source_message_id": str(message.id),
                        "source_role": message.role,
                        "semantic_kind": "relationship",
                        "semantic_id": self._semantic_relationship_id(
                            relationship.source_name,
                            relationship.target_name,
                        ),
                    },
                    "embedding": await embedding_provider.embed_query(rel_content),
                }
            )

        if vector_chunks:
            await self._vector_store.upsert_chunks(
                namespace_id=namespace_id,
                collection_id=collection.id,
                chunks=vector_chunks,
            )

    async def _load_semantic_region(
        self,
        chat_id: uuid.UUID,
        source_message_ids: set[str],
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        storage = self._chat_storage(chat_id)
        nodes_by_id: dict[str, dict[str, Any]] = {}
        edges_by_id: dict[str, dict[str, Any]] = {}
        for message_id in source_message_ids:
            for node in await storage.get_nodes_by_source_message_id(message_id):
                node_id = str(node.get("id") or "")
                if node_id:
                    nodes_by_id[node_id] = node
            for edge in await storage.get_edges_by_source_message_id(message_id):
                edge_id = str(edge.get("id") or "")
                if edge_id:
                    edges_by_id[edge_id] = edge
                    source_id = str(edge.get("source_id") or "")
                    target_id = str(edge.get("target_id") or "")
                    if source_id and source_id not in nodes_by_id:
                        node = await storage.get_node(source_id)
                        if node:
                            nodes_by_id[source_id] = node
                    if target_id and target_id not in nodes_by_id:
                        node = await storage.get_node(target_id)
                        if node:
                            nodes_by_id[target_id] = node

        for node_id in list(nodes_by_id):
            for source_id, target_id in await storage.get_node_edges(node_id):
                edge = await storage.get_edge(source_id, target_id)
                if not edge:
                    continue
                edge_id = str(edge.get("id") or "")
                if edge_id:
                    edge["source_id"] = source_id
                    edge["target_id"] = target_id
                    edges_by_id[edge_id] = edge
                neighbor_id = target_id if source_id == node_id else source_id
                if neighbor_id and neighbor_id not in nodes_by_id:
                    neighbor = await storage.get_node(neighbor_id)
                    if neighbor:
                        nodes_by_id[neighbor_id] = neighbor
        nodes = sorted(
            nodes_by_id.values(),
            key=lambda node: str(node.get("name") or ""),
        )
        edges = sorted(
            edges_by_id.values(),
            key=lambda edge: (
                str(edge.get("source_id") or ""),
                str(edge.get("target_id") or ""),
            ),
        )
        return nodes, edges

    def _format_semantic_context(
        self,
        nodes: list[dict[str, Any]],
        edges: list[dict[str, Any]],
    ) -> str:
        lines: list[str] = []
        for node in nodes[:8]:
            name = str(node.get("name") or "").strip()
            description = str(node.get("description") or "").strip()
            if name and description:
                lines.append(f"Entity: {name}. {description}")
            elif name:
                lines.append(f"Entity: {name}")
        for edge in edges[:10]:
            source_id = str(edge.get("source_id") or "")
            target_id = str(edge.get("target_id") or "")
            source_name = source_id.removeprefix("entity:")
            target_name = target_id.removeprefix("entity:")
            description = str(edge.get("description") or "").strip()
            if source_name and target_name and description:
                lines.append(
                    f"Relationship: {source_name} -> {target_name}. {description}"
                )
        return "\n".join(lines)

    async def _rewrite_chat_question(
        self,
        collection: Collection,
        question: str,
        chronology_context: str,
        semantic_context: str,
        llm_profile_id: uuid.UUID | None,
    ) -> str:
        if not chronology_context.strip() and not semantic_context.strip():
            return question
        llm_provider = await self._resolve_collection_llm_provider(
            collection,
            llm_profile_id,
        )
        rewritten = await llm_provider.chat(
            [
                {
                    "role": "system",
                    "content": (
                        "Rewrite the user's latest message into a standalone "
                        "retrieval query using the provided chronology and semantic "
                        "memory from the chat. Resolve omitted references like "
                        "it/that/this/trip/place/person from context. Preserve the "
                        "user's intent. If the latest message is a correction or "
                        "disagreement, make the rewritten query explicitly compare "
                        "what the user said with what the assistant inferred. "
                        "Return only the rewritten query."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Recent chat messages:\n{chronology_context}\n\n"
                        f"Semantic memory:\n{semantic_context}\n\n"
                        f"Latest user message:\n{question}"
                    ),
                },
            ]
        )
        candidate = rewritten.strip()
        return candidate or question

    async def _load_chat_context(
        self,
        collection: Collection,
        question: str,
        chat_id: uuid.UUID,
        llm_profile_id: uuid.UUID | None,
    ) -> tuple[str, str]:
        recent_messages = await self._load_recent_chat_messages(chat_id, limit=6)
        embedding_provider = await self._resolve_collection_embedding_provider(
            collection
        )
        query_embedding = await embedding_provider.embed_query(question)
        hits = await self._vector_store.query_chunks(
            collection_id=collection.id,
            query_embedding=query_embedding,
            top_k=6,
            metadata_filters={
                "memory_type": "chat_semantic",
                "chat_id": str(chat_id),
            },
        )
        chronology_context = "\n\n".join(
            self._format_chat_message(
                message.role,
                message.content,
                message.turn_index,
            )
            for message in recent_messages[-4:]
        )

        selected_message_ids: set[str] = set()
        top_score = 0.0
        for hit in hits:
            top_score = max(top_score, float(hit.get("score") or 0.0))
            metadata = hit.get("metadata") or {}
            message_id = str(metadata.get("source_message_id") or "").strip()
            if message_id:
                selected_message_ids.add(message_id)

        if top_score < 0.55 or not selected_message_ids:
            for message in recent_messages[-2:]:
                selected_message_ids.add(str(message.id))

        nodes, edges = await self._load_semantic_region(chat_id, selected_message_ids)
        semantic_context = self._format_semantic_context(nodes, edges)
        rewritten = await self._rewrite_chat_question(
            collection,
            question,
            chronology_context,
            semantic_context,
            llm_profile_id,
        )
        combined_context = chronology_context
        if semantic_context:
            combined_context = (
                f"{chronology_context}\n\nSemantic memory:\n{semantic_context}"
            )
        return combined_context, rewritten

    async def _record_chat_exchange(
        self,
        collection: Collection,
        namespace_id: uuid.UUID,
        chat_id: uuid.UUID,
        question: str,
        response: str,
        mode: str | None,
        llm_profile_id: uuid.UUID | None,
    ) -> None:
        embedding_provider = await self._resolve_collection_embedding_provider(
            collection
        )
        async with AsyncSessionLocal() as session:
            chat = await session.get(ChatSession, chat_id)
            if not chat:
                raise ValueError(f"Chat session {chat_id} not found")
            if chat.collection_id != collection.id or chat.namespace_id != namespace_id:
                raise PermissionError("Chat session does not belong to collection")

            max_turn = await session.scalar(
                select(func.max(ChatMessage.turn_index)).where(
                    ChatMessage.chat_id == chat_id
                )
            )
            turn_index = int(max_turn or 0) + 1
            max_message_index = await session.scalar(
                select(func.max(ChatMessage.message_index)).where(
                    ChatMessage.chat_id == chat_id
                )
            )
            user_message = ChatMessage(
                chat_id=chat_id,
                collection_id=collection.id,
                role="user",
                turn_index=turn_index,
                message_index=int(max_message_index or 0) + 1,
                content=question,
                mode=None,
            )
            assistant_message = ChatMessage(
                chat_id=chat_id,
                collection_id=collection.id,
                role="assistant",
                turn_index=turn_index,
                message_index=int(max_message_index or 0) + 2,
                content=response,
                mode=mode,
            )
            chat.updated_at = datetime.now(UTC)
            session.add(user_message)
            session.add(assistant_message)
            await session.commit()
            await session.refresh(user_message)
            await session.refresh(assistant_message)

        user_embedding = await embedding_provider.embed_query(
            self._format_chat_message("user", question, turn_index)
        )
        assistant_embedding = await embedding_provider.embed_query(
            self._format_chat_message("assistant", response, turn_index)
        )
        await self._vector_store.upsert_chunks(
            namespace_id=namespace_id,
            collection_id=collection.id,
            chunks=[
                {
                    "chunk_hash": self._chat_chunk_hash(
                        "chat_message",
                        chat_id,
                        user_message.message_index,
                    ),
                    "chunk_index": 0,
                    "content": self._format_chat_message("user", question, turn_index),
                    "token_count": len(question.split()),
                    "metadata": {
                        "memory_type": "chat_message",
                        "chat_id": str(chat_id),
                        "message_id": str(user_message.id),
                        "role": "user",
                        "turn_index": str(turn_index),
                        "message_index": str(user_message.message_index),
                    },
                    "embedding": user_embedding,
                },
                {
                    "chunk_hash": self._chat_chunk_hash(
                        "chat_message",
                        chat_id,
                        assistant_message.message_index,
                    ),
                    "chunk_index": 0,
                    "content": self._format_chat_message(
                        "assistant",
                        response,
                        turn_index,
                    ),
                    "token_count": len(response.split()),
                    "metadata": {
                        "memory_type": "chat_message",
                        "chat_id": str(chat_id),
                        "message_id": str(assistant_message.id),
                        "role": "assistant",
                        "turn_index": str(turn_index),
                        "message_index": str(assistant_message.message_index),
                    },
                    "embedding": assistant_embedding,
                },
            ],
        )
        await self._store_chat_semantic_memory(
            collection,
            namespace_id,
            chat_id,
            user_message,
            embedding_provider,
            llm_profile_id,
        )
        await self._store_chat_semantic_memory(
            collection,
            namespace_id,
            chat_id,
            assistant_message,
            embedding_provider,
            llm_profile_id,
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

    async def _resolve_collection_llm_provider(
        self,
        collection: Collection,
        llm_profile_id: uuid.UUID | None,
    ):
        profile_id = llm_profile_id or collection.llm_profile_id
        if profile_id is None:
            return get_llm_provider()
        async with AsyncSessionLocal() as session:
            profile = await session.get(Profile, profile_id)
            if not profile:
                raise ValueError(f"LLM profile {profile_id} not found")
            api_key, cred_base_url = await _resolve_credential(session, profile)
            base_url = profile.base_url or cred_base_url
            return get_llm_provider(
                provider_name=profile.provider,
                model=profile.model,
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
            chat_context, retrieval_question = await self._load_chat_context(
                collection,
                question,
                chat_id,
                effective_llm_profile_id,
            )
        else:
            retrieval_question = question

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
            await self._record_chat_exchange(
                collection=collection,
                namespace_id=namespace_id,
                chat_id=chat_id,
                question=question,
                response=result.response,
                mode=result.mode,
                llm_profile_id=effective_llm_profile_id,
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

    async def analyze_collection_graph(
        self,
        collection_id: uuid.UUID,
        namespace_id: uuid.UUID,
        *,
        min_edge_strength: float = 0.2,
        min_community_size: int = 2,
        max_anchors: int = 12,
        max_path_depth: int = 4,
        max_connector_paths: int = 20,
    ) -> dict[str, Any]:
        collection = await self.get_collection(collection_id)
        self._enforce_namespace(collection, namespace_id)
        return await analyze_collection_graph(
            collection_id,
            min_edge_strength=min_edge_strength,
            min_community_size=min_community_size,
            max_anchors=max_anchors,
            max_path_depth=max_path_depth,
            max_connector_paths=max_connector_paths,
        )

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
    "analyze_collection_graph",
]
