"""GraphService — internal orchestration for all graph operations.

This class has no transport dependencies. It is called by API routes,
MCP tools, and background workers. All dependencies are injected.
"""

import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Literal

from sqlalchemy import select, text
from sqlalchemy.dialects.postgresql import insert as pg_insert

from graph_core.config import settings
from graph_core.database import AsyncSessionLocal
from graph_core.embedding import get_embedding_provider
from graph_core.embedding.interface import EmbeddingProvider
from graph_core.llm import LocalEchoLLMProvider, get_llm_provider
from graph_core.llm.interface import LLMProvider
from graph_core.models.chunk import IngestionChunk
from graph_core.models.collection import Collection
from graph_core.models.credential import Credential
from graph_core.models.graph_rag import (
    EntityAlias,
    EntityDescription,
    GraphEntity,
    GraphRelationship,
    RelationshipDescription,
    RawChunkExtraction,
)
from graph_core.models.ingestion import IngestionRecord
from graph_core.models.job import Job, JobEvent
from graph_core.models.namespace import Namespace
from graph_core.models.profile import Profile
from graph_core.services.chunking import TokenChunker
from graph_core.services.crypto import CredentialCrypto
from graph_core.services.graph_rag.entity_resolver import (
    IncrementalEntityResolver,
)
from graph_core.services.graph_rag.extractor import (
    LLMGraphExtractor,
    ExtractionResult,
)
from graph_core.services.sanitizer import TextSanitizer
from graph_core.storage.graph_rag_vectors import GraphRAGVectorStore
from graph_core.storage.vector_store import VectorStore
from graph_core.storage.vector_tables import (
    create_all_tables,
    drop_all_tables,
    table_name,
)


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
    mode: str | None = None


class GraphService:
    def __init__(self):
        self._sanitizer = TextSanitizer()
        self._chunker = TokenChunker(
            chunk_size_tokens=settings.chunk_size_tokens,
            chunk_overlap_tokens=settings.chunk_overlap_tokens,
        )
        self._crypto = CredentialCrypto()
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

            dimensions = None
            if embedding_profile_id is not None:
                profile = await session.get(Profile, embedding_profile_id)
                if not profile:
                    raise ValueError(f"Embedding profile {embedding_profile_id} not found")
                if profile.namespace_id != namespace_id:
                    raise ValueError("Embedding profile does not belong to namespace")
                if profile.kind != "embedding":
                    raise ValueError("Profile kind must be embedding")
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

        # Create per-collection vector tables after the collection exists
        if dimensions is not None:
            await create_all_tables(collection.id, dimensions)

        return collection

    async def delete_collection(self, collection_id: uuid.UUID) -> None:
        """Delete a collection and drop its per-collection vector tables + FalkorDB graph."""
        collection = await self.get_collection(collection_id)
        await drop_all_tables(collection_id)
        graph_storage = self._get_graph_storage(collection_id)
        await graph_storage.drop()
        async with AsyncSessionLocal() as session:
            await session.delete(collection)
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
        sanitized_text, report = self._sanitizer.sanitize(text, str(namespace_id))
        chunk_hash = self._sanitizer.chunk_hash(sanitized_text)

        if collection.strategy == "vector":
            result = await self._ingest_vector_chunk(
                sanitized_text, collection, chunk_hash, report, chunk_index=chunk_index,
            )
        elif collection.strategy == "light_rag":
            result = await self._ingest_lightrag_chunk(
                sanitized_text, collection, chunk_hash, report,
            )
        else:
            result = await self._ingest_graph_chunk(
                sanitized_text, collection, chunk_hash, report,
            )

        await self._write_ledger(collection, chunk_hash, report, result)
        return result

    async def enqueue_document_ingestion(
        self,
        text: str,
        collection_id: uuid.UUID,
        namespace_id: uuid.UUID,
    ) -> DocumentIngestionResult:
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

        return DocumentIngestionResult(job_id=job.id, status="pending")

    # ── Document pipeline ──

    async def ingest_document_pipeline(self, job_id: uuid.UUID):
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

        chunks = self._chunker.chunk_text(text)
        total_chunks = max(len(chunks), 1)

        if not chunks:
            await self.update_job_status(job_id, "completed", progress_percent=100)
            return

        # For custom_graph_rag and light_rag: fan-out chunks to parallel workers
        if collection.strategy in ("custom_graph_rag", "light_rag"):
            await self._fan_out_chunks(job_id, collection.id, chunks)
        else:
            # Vector strategy: sequential processing
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
                    job_id, "chunk_completed",
                    {"chunk_index": index - 1, "total_chunks": total_chunks},
                )
            await self.update_job_status(job_id, "completed", progress_percent=100)

    async def _fan_out_chunks(
        self, job_id: uuid.UUID, collection_id: uuid.UUID, chunks: list[str]
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
                {"total": len(chunks), "jid": job_id},
            )
            await session.commit()

    async def process_single_chunk(
        self, job_id: str, chunk_index: int
    ) -> None:
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

        result = await self._ingest_collection_chunk(
            text=text,
            collection=collection,
            namespace_id=collection.namespace_id,
            chunk_index=chunk_index,
        )

        await self.update_chunk_status(job_uuid, chunk_index, "completed")

        progress = await self._increment_chunk_counter(job_uuid)
        await self.append_job_event(
            job_uuid, "chunk_completed",
            {"chunk_index": chunk_index, "entity_count": result.entity_count, "relationship_count": result.relationship_count},
        )

    async def update_chunk_status(
        self, job_id: uuid.UUID, chunk_index: int, status: str, error: str | None = None
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

    async def _increment_chunk_counter(self, job_id: uuid.UUID) -> int:
        """Atomically increment chunks_completed and return progress percent."""
        async with AsyncSessionLocal() as session:
            result = await session.execute(
                text(
                    "UPDATE jobs SET chunks_completed = chunks_completed + 1, "
                    "progress_percent = CAST(((chunks_completed + 1)::float / NULLIF(chunks_total, 0) * 100) AS integer), "
                    "status = CASE WHEN chunks_completed + 1 >= chunks_total THEN 'completed' ELSE 'running' END "
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
                        text("UPDATE jobs SET completed_at = :now WHERE id = :jid AND status = 'completed'"),
                        {"now": datetime.now(UTC), "jid": job_id},
                    )
                    await session.commit()
                await session.commit()
                return int((completed / total * 100) if total else 0)
            return 0

    # ── Graph RAG ingestion pipeline ──

    def _get_graph_storage(self, collection_id: uuid.UUID):
        """Return a FalkorDBGraphStorage scoped to the collection's own graph."""
        from graph_core.storage.graph_storage import FalkorDBGraphStorage
        graph_name = f"collection_{str(collection_id).replace('-', '')}"
        return FalkorDBGraphStorage(graph_name)

    async def _ingest_graph_chunk(
        self,
        text: str,
        collection: Collection,
        chunk_hash: str,
        report,
    ) -> ChunkIngestionResult:
        """Full Graph RAG pipeline: extract → resolve → store."""
        embedding_provider = await self._get_embedding_provider_for_collection(collection)
        llm_provider = await self._get_llm_provider(namespace_id=collection.namespace_id, llm_profile_id=collection.llm_profile_id)

        # Check raw extraction cache
        cached = await self._get_raw_extraction(chunk_hash, collection.id)
        if cached:
            return ChunkIngestionResult(
                chunk_hash=chunk_hash,
                entity_count=len(cached.entities),
                relationship_count=len(cached.relationships),
            )

        # LLM extraction
        extractor = LLMGraphExtractor(llm=llm_provider)
        extraction = await extractor.extract_with_gleaning(text=text, max_gleaning=1)

        # Save raw extraction cache
        await self._save_raw_extraction(
            chunk_hash=chunk_hash,
            collection_id=collection.id,
            extraction=extraction,
        )

        # Embed and store chunk for naive retrieval fallback
        chunk_embedding = await embedding_provider.embed_query(text)
        await self._graph_rag_vectors.upsert_chunk_embedding(
            collection_id=collection.id,
            chunk_hash=chunk_hash,
            chunk_index=0,
            content=text,
            embedding=chunk_embedding,
        )

        if not extraction.entities and not extraction.relationships:
            return ChunkIngestionResult(
                chunk_hash=chunk_hash, entity_count=0, relationship_count=0,
            )

        # Entity resolution + storage
        from graph_core.services.entity_name_cache import EntityNameCache

        resolver = IncrementalEntityResolver(
            embedding_provider=embedding_provider,
            collection_id=collection.id,
        )
        name_cache = EntityNameCache(str(collection.id))

        resolved_entity_ids: dict[str, uuid.UUID] = {}

        async with AsyncSessionLocal() as session:
            for entity in extraction.entities:
                # Check cache first
                cached_id = await name_cache.get(entity.name)
                if cached_id:
                    resolved_entity_ids[entity.name] = cached_id
                    continue

                result = await resolver.resolve_entity(
                    session=session,
                    name=entity.name,
                    entity_type=entity.entity_type,
                    description=entity.description,
                    source_chunk_hash=chunk_hash,
                )
                resolved_entity_ids[entity.name] = result.entity_id

                if result.is_new:
                    await name_cache.set_many(
                        [entity.name, result.canonical_name], result.entity_id
                    )

                await session.commit()

            # Resolve relationships + FalkorDB upsert
            nodes_to_upsert = []
            edges_to_upsert = []

            for rel in extraction.relationships:
                source_id = resolved_entity_ids.get(rel.source_name)
                target_id = resolved_entity_ids.get(rel.target_name)
                if not source_id or not target_id:
                    continue

                rel_result = await resolver.resolve_relationship(
                    session=session,
                    source_entity_id=source_id,
                    target_entity_id=target_id,
                    description=rel.description,
                    keywords=rel.keywords,
                    weight=rel.weight,
                    source_chunk_hash=chunk_hash,
                )
                await session.commit()

                nodes_to_upsert.append({
                    "id": str(source_id),
                    "name": rel.source_name,
                    "collection_id": str(collection.id),
                })
                nodes_to_upsert.append({
                    "id": str(target_id),
                    "name": rel.target_name,
                    "collection_id": str(collection.id),
                })
                edges_to_upsert.append({
                    "source_id": str(source_id),
                    "target_id": str(target_id),
                    "id": str(rel_result.relationship_id),
                    "weight": int(rel.weight * 10),
                    "keywords": rel.keywords,
                    "collection_id": str(collection.id),
                })

        # Deduplicate nodes
        unique_nodes = {n["id"]: n for n in nodes_to_upsert}.values()

        # Batch upsert to FalkorDB
        graph_storage = self._get_graph_storage(collection.id)
        if unique_nodes:
            await graph_storage.upsert_nodes(list(unique_nodes))
        if edges_to_upsert:
            await graph_storage.upsert_edges(edges_to_upsert)

        return ChunkIngestionResult(
            chunk_hash=chunk_hash,
            entity_count=len(extraction.entities),
            relationship_count=len(extraction.relationships),
        )

    async def _save_raw_extraction(
        self,
        chunk_hash: str,
        collection_id: uuid.UUID,
        extraction: ExtractionResult,
    ) -> None:
        async with AsyncSessionLocal() as session:
            record = RawChunkExtraction(
                chunk_content_hash=chunk_hash,
                collection_id=collection_id,
                entities_json=[
                    {"name": e.name, "type": e.entity_type, "description": e.description}
                    for e in extraction.entities
                ],
                relationships_json=[
                    {
                        "source_name": r.source_name,
                        "target_name": r.target_name,
                        "description": r.description,
                        "keywords": r.keywords,
                        "weight": r.weight,
                    }
                    for r in extraction.relationships
                ],
            )
            session.add(record)
            try:
                await session.commit()
            except Exception:
                await session.rollback()

    async def _get_raw_extraction(
        self, chunk_hash: str, collection_id: uuid.UUID
    ) -> ExtractionResult | None:
        from graph_core.services.graph_rag.extractor import (
            ExtractedEntity,
            ExtractedRelationship,
        )

        async with AsyncSessionLocal() as session:
            result = await session.execute(
                select(RawChunkExtraction).where(
                    RawChunkExtraction.chunk_content_hash == chunk_hash,
                    RawChunkExtraction.collection_id == collection_id,
                )
            )
            record = result.scalar_one_or_none()
            if not record:
                return None

            entities = [
                ExtractedEntity(name=e["name"], entity_type=e["type"], description=e["description"])
                for e in (record.entities_json or [])
            ]
            relationships = [
                ExtractedRelationship(
                    source_name=r["source_name"],
                    target_name=r["target_name"],
                    description=r["description"],
                    keywords=r.get("keywords", []),
                    weight=r.get("weight", 1.0),
                )
                for r in (record.relationships_json or [])
            ]
            return ExtractionResult(entities=entities, relationships=relationships)

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
                question, collection, namespace_id, effective_mode, llm_profile_id=llm_profile_id,
            )
        if collection.strategy == "custom_graph_rag":
            return await self._query_graph_rag(
                question, collection, namespace_id, llm_profile_id=llm_profile_id,
            )
        if collection.strategy == "light_rag":
            return await self._query_lightrag(
                question, collection, namespace_id, effective_mode, llm_profile_id=llm_profile_id,
            )

        return QueryResult(
            response="",
            entities_used=[],
            relationships_used=[],
            mode=effective_mode,
        )

    # ── Graph RAG Query ──

    async def _query_graph_rag(
        self,
        question: str,
        collection: Collection,
        namespace_id: uuid.UUID,
        llm_profile_id: uuid.UUID | None = None,
    ) -> QueryResult:
        """Full Graph RAG query pipeline with energy-decay DFS traversal.

        Steps:
        1. Embed query
        2. Seed entity search (pgvector entity embeddings + alias ILIKE)
        3. Score seeds by relationship relevance
        4. Energy-decay DFS traversal in FalkorDB
        5. Fetch EntityDescriptions from Postgres
        6. Fetch RelationshipDescriptions from Postgres
        7. Build context and call LLM
        """
        embedding_provider = await self._get_embedding_provider_for_collection(collection)

        # Step 1: Embed query
        query_embedding = await embedding_provider.embed_query(question)

        # Step 2: Seed entity search
        TOP_K = 10
        MIN_EDGE_SIM = settings.graph_rag_min_edge_similarity
        MIN_SEED_SIM = 0.4
        ENERGY_BUDGET = 7.0
        MAX_DEPTH = 8
        MAX_ENTITIES = 10
        MAX_ENTITY_DESCS = 4
        MAX_REL_DESCS = 4

        entity_hits = await self._graph_rag_vectors.search_entity_embeddings(
            collection_id=collection.id,
            query_embedding=query_embedding,
            top_k=TOP_K,
        )

        seed_entity_ids: list[str] = []
        entity_relevance: dict[str, float] = {}

        for hit in entity_hits:
            meta = hit.metadata
            entity_id_str = meta.get("entity_id", "")
            sim = 1.0 - hit.distance
            if entity_id_str and entity_id_str not in seed_entity_ids:
                seed_entity_ids.append(entity_id_str)
                entity_relevance[entity_id_str] = sim

        # Step 2b: Alias lookup — catch entities missed by vector search
        async with AsyncSessionLocal() as session:
            import string
            stop_words = {
                "the", "a", "an", "and", "or", "in", "on", "at", "to",
                "for", "of", "with", "is", "what", "how", "why", "who",
                "i", "me", "my",
            }
            tokens = [w.strip(string.punctuation).lower() for w in question.split()]
            keywords = [w for w in tokens if w and w not in stop_words and len(w) > 2]
            keywords = list(dict.fromkeys([question] + keywords))

            for kw in keywords[:5]:
                alias_result = await session.execute(
                    select(EntityAlias).join(
                        GraphEntity, GraphEntity.id == EntityAlias.entity_id
                    ).where(
                        EntityAlias.alias_name.ilike(f"%{kw}%"),
                        GraphEntity.collection_id == collection.id,
                    ).limit(5)
                )
                for alias in alias_result.scalars().all():
                    eid = str(alias.entity_id)
                    if eid not in seed_entity_ids:
                        seed_entity_ids.append(eid)
                        entity_relevance[eid] = 1.0

        # Step 3: Score seed entities by relationship relevance
        rel_hits = await self._graph_rag_vectors.search_relationship_embeddings(
            collection_id=collection.id,
            query_embedding=query_embedding,
            top_k=max(TOP_K * 5, 50),
        )

        # Build name -> entity_id map
        async with AsyncSessionLocal() as session:
            seed_entity_rows = await session.execute(
                select(GraphEntity).where(
                    GraphEntity.collection_id == collection.id
                )
            )
            name_to_eid = {
                e.canonical_name.lower(): str(e.id)
                for e in seed_entity_rows.scalars().all()
            }

        seed_rel_scores: dict[str, float] = {eid: 0.0 for eid in seed_entity_ids}
        for hit in rel_hits:
            meta = hit.metadata
            sim = 1.0 - hit.distance
            if sim < MIN_EDGE_SIM:
                continue
            for name_field in ("source_name", "target_name"):
                name = meta.get(name_field, "").lower()
                eid = name_to_eid.get(name)
                if eid and sim > seed_rel_scores.get(eid, 0.0):
                    seed_rel_scores[eid] = sim

        # Step 4: Energy-decay DFS traversal in FalkorDB
        graph_storage = self._get_graph_storage(collection.id)
        visited = set(seed_entity_ids)
        traversed_rel_ids: list[str] = []
        discovered_entity_ids = list(seed_entity_ids)
        energy = ENERGY_BUDGET
        rel_score_cache: dict[str, float] = {}

        best_seed_sim = max(entity_relevance.values()) if entity_relevance else 0.0
        effective_depth = 1 if best_seed_sim < MIN_SEED_SIM else MAX_DEPTH

        # Sort seeds by relationship relevance ascending so highest-rel seeds
        # sit on top of stack (LIFO = explored first)
        sorted_seeds = sorted(
            seed_entity_ids,
            key=lambda e: seed_rel_scores.get(e, 0.0),
        )
        stack = [(node_id, 0) for node_id in sorted_seeds]

        async with AsyncSessionLocal() as session:
            while stack and energy > 0:
                node_id, depth = stack.pop()
                if depth >= effective_depth:
                    continue

                edges = await graph_storage.get_node_edges(node_id)
                scored_edges: list[tuple[float, str, str]] = []

                for src, tgt in edges:
                    neighbor = tgt if src == node_id else src
                    if neighbor in visited:
                        continue

                    edge_props = await graph_storage.get_edge(src, tgt)
                    if not edge_props:
                        edge_props = await graph_storage.get_edge(tgt, src)
                    if not (edge_props and edge_props.get("id")):
                        continue

                    rel_id_str = str(edge_props["id"])

                    # Score edge by relationship embedding similarity
                    rel_vdb_hits = await self._graph_rag_vectors.search_relationship_embeddings(
                        collection_id=collection.id,
                        query_embedding=query_embedding,
                        top_k=MAX_REL_DESCS,
                        relationship_id=uuid.UUID(rel_id_str),
                    )
                    sim = max(1.0 - r.distance for r in rel_vdb_hits) if rel_vdb_hits else 0.0
                    rel_score_cache[rel_id_str] = sim

                    if sim >= MIN_EDGE_SIM:
                        scored_edges.append((sim, neighbor, rel_id_str))

                # Push low-sim edges first so high-sim edges are on top of stack
                for sim, neighbor, rel_id_str in sorted(scored_edges, key=lambda x: x[0]):
                    cost = max(0.05, 1.0 - sim)
                    if energy - cost > 0:
                        energy -= cost
                        visited.add(neighbor)
                        stack.append((neighbor, depth + 1))
                        if rel_id_str and rel_id_str not in traversed_rel_ids:
                            traversed_rel_ids.append(rel_id_str)
                        if neighbor not in discovered_entity_ids:
                            discovered_entity_ids.append(neighbor)
                        edge_sim = 1.0 - cost
                        if neighbor not in entity_relevance or edge_sim > entity_relevance[neighbor]:
                            entity_relevance[neighbor] = edge_sim

            # Step 5: Fetch EntityDescriptions from Postgres
            ranked_entity_ids = sorted(
                discovered_entity_ids,
                key=lambda eid: entity_relevance.get(eid, 0.0),
                reverse=True,
            )

            entity_context_parts: list[str] = []
            entities_used: list[str] = []
            for eid_str in ranked_entity_ids[:MAX_ENTITIES]:
                try:
                    eid = uuid.UUID(eid_str)
                except ValueError:
                    continue
                entity = await session.get(GraphEntity, eid)
                if not entity:
                    continue
                descs_result = await session.execute(
                    select(EntityDescription)
                    .where(EntityDescription.entity_id == eid)
                    .order_by(EntityDescription.weight.desc())
                    .limit(MAX_ENTITY_DESCS)
                )
                descs = descs_result.scalars().all()
                if descs:
                    desc_texts = " | ".join(d.description for d in descs)
                    entity_context_parts.append(
                        f"{entity.canonical_name} ({entity.primary_type or 'unknown'}): {desc_texts}"
                    )
                    entities_used.append(entity.canonical_name)

            # Step 6: Fetch RelationshipDescriptions from Postgres
            rel_context_parts: list[str] = []
            relationships_used: list[str] = []
            for rel_id_str in traversed_rel_ids[:50]:
                try:
                    rel_uuid = uuid.UUID(rel_id_str)
                except ValueError:
                    continue
                rel = await session.get(GraphRelationship, rel_uuid)
                if not rel:
                    continue
                src_entity = await session.get(GraphEntity, rel.source_entity_id)
                tgt_entity = await session.get(GraphEntity, rel.target_entity_id)
                src_name = src_entity.canonical_name if src_entity else "?"
                tgt_name = tgt_entity.canonical_name if tgt_entity else "?"
                descs_result = await session.execute(
                    select(RelationshipDescription)
                    .where(RelationshipDescription.relationship_id == rel_uuid)
                    .order_by(RelationshipDescription.weight.desc())
                    .limit(MAX_REL_DESCS)
                )
                descs = descs_result.scalars().all()
                sim = rel_score_cache.get(rel_id_str, 0.0)
                for d in descs:
                    rel_text = f"{src_name} \u2192 {tgt_name}: {d.description}"
                    rel_context_parts.append((sim, rel_text))
                    relationships_used.append(f"{src_name} \u2192 {tgt_name}")

            rel_context_parts.sort(key=lambda x: x[0], reverse=True)

            # Step 7: Build context and call LLM
            entity_context = "\n".join(entity_context_parts)
            rel_context = "\n".join(text for _, text in rel_context_parts)

            context = f"""Context:
Entities:
{entity_context or "(none)"}
Relationships:
{rel_context or "(none)"}"""

            llm_provider = await self._get_llm_provider(
                namespace_id=namespace_id, llm_profile_id=llm_profile_id,
            )
            if isinstance(llm_provider, LocalEchoLLMProvider):
                response = entity_context or rel_context or "No relevant context found."
            else:
                response = await llm_provider.chat([
                    {
                        "role": "system",
                        "content": (
                            "Use the context below to answer the question. "
                            "Draw on the entities and relationships to reason through your answer - "
                            "explain, connect, and illuminate rather than just report. "
                            "Write in natural prose. If the context is insufficient for part of the "
                            "question, acknowledge it briefly without making it the focus."
                        ),
                    },
                    {
                        "role": "user",
                        "content": f"{context}\n\nQuestion: {question}",
                    },
                ])

            return QueryResult(
                response=response,
                entities_used=entities_used,
                relationships_used=list(dict.fromkeys(relationships_used)),
            )

    # ── LightRAG Query ──

    async def _query_lightrag(
        self,
        question: str,
        collection: Collection,
        namespace_id: uuid.UUID,
        mode: str,
        llm_profile_id: uuid.UUID | None = None,
    ) -> QueryResult:
        """LightRAG query with keyword-driven retrieval.

        Supports modes: local, global, hybrid, naive.
        1. Extract high/low level keywords from query
        2. Mode-specific retrieval (entity/relationship/chunk vector search)
        3. Graph traversal for connected entities/relationships
        4. Token budget management
        5. LLM answer generation
        """
        embedding_provider = await self._get_embedding_provider_for_collection(collection)
        llm_provider = await self._get_llm_provider(
            namespace_id=namespace_id, llm_profile_id=llm_profile_id,
        )

        keywords = await self._extract_keywords(question, llm_provider)

        if mode == "naive":
            return await self._lightrag_query_naive(
                question, collection, embedding_provider, llm_provider,
            )
        elif mode == "local":
            return await self._lightrag_query_local(
                question, collection, keywords, embedding_provider, llm_provider,
            )
        elif mode == "global":
            return await self._lightrag_query_global(
                question, collection, keywords, embedding_provider, llm_provider,
            )
        elif mode == "hybrid":
            return await self._lightrag_query_hybrid(
                question, collection, keywords, embedding_provider, llm_provider,
            )
        elif mode == "mix":
            return await self._lightrag_query_mix(
                question, collection, keywords, embedding_provider, llm_provider,
            )
        else:
            return await self._lightrag_query_local(
                question, collection, keywords, embedding_provider, llm_provider,
            )

    async def _extract_keywords(
        self, query: str, llm_provider: LLMProvider
    ) -> tuple[list[str], list[str]]:
        """Extract high-level and low-level keywords from query.

        Returns (high_level, low_level) keyword lists.
        Falls back to word-level extraction on failure.
        """
        if not query or not query.strip():
            return [], []

        _KW_SCHEMA = {
            "type": "object",
            "properties": {
                "high_level_keywords": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Conceptual, abstract keywords for broad topic search",
                },
                "low_level_keywords": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Specific, concrete keywords for precise entity search",
                },
            },
            "required": ["high_level_keywords", "low_level_keywords"],
        }

        try:
            result = await llm_provider.structured_extract(
                prompt=(
                    "Extract keywords from this query for knowledge graph search.\n\n"
                    "Return JSON with two arrays:\n"
                    "- high_level_keywords: conceptual terms describing the topic/theme\n"
                    "- low_level_keywords: specific entity names, places, or concrete terms\n\n"
                    f"Query: {query}"
                ),
                schema=_KW_SCHEMA,
            )
            hl = result.get("high_level_keywords", [])
            ll = result.get("low_level_keywords", [])
            if isinstance(hl, list) and isinstance(ll, list):
                hl = [str(k) for k in hl if k]
                ll = [str(k) for k in ll if k]
                if hl and ll:
                    return hl, ll
        except Exception:
            pass

        return self._fallback_keywords(query), self._fallback_keywords(query)

    @staticmethod
    def _fallback_keywords(query: str) -> list[str]:
        import string
        stop_words = {
            "the", "a", "an", "and", "or", "in", "on", "at", "to",
            "for", "of", "with", "is", "what", "how", "why", "who",
            "i", "me", "my", "are", "was", "were", "be", "been",
            "have", "has", "had", "do", "does", "did", "will", "would",
            "this", "that", "these", "those", "there", "here",
        }
        words = query.lower().split()
        words = [w.strip(string.punctuation) for w in words]
        words = [w for w in words if w and w not in stop_words and len(w) > 2]
        return words if words else [w for w in words if w]

    async def _lightrag_query_naive(
        self,
        question: str,
        collection: Collection,
        embedding_provider: EmbeddingProvider,
        llm_provider: LLMProvider,
    ) -> QueryResult:
        """Naive mode: pure vector search on chunk embeddings."""
        query_embedding = await embedding_provider.embed_query(question)
        hits = await self._graph_rag_vectors.search_chunk_embeddings(
            collection_id=collection.id,
            query_embedding=query_embedding,
            top_k=settings.vector_query_top_k,
        )
        chunks = [h.content for h in hits]
        response = await self._generate_vector_answer(
            question=question,
            chunks=chunks,
            namespace_id=collection.namespace_id,
            llm_profile_id=None,
        )
        if isinstance(llm_provider, LocalEchoLLMProvider):
            response = chunks[0] if chunks else ""
        return QueryResult(
            response=response, entities_used=[], relationships_used=[], mode="naive",
        )

    async def _lightrag_query_local(
        self,
        question: str,
        collection: Collection,
        keywords: tuple[list[str], list[str]],
        embedding_provider: EmbeddingProvider,
        llm_provider: LLMProvider,
    ) -> QueryResult:
        """Local mode: entity-focused retrieval + graph traversal.

        1. Search entity embeddings with low-level keywords
        2. For each entity, get connected edges from graph
        3. Collect source chunks from entities and relationships
        4. Build context with token budgets
        5. Call LLM
        """
        high_level, low_level = keywords
        search_terms = low_level if low_level else [question]
        search_text = " ".join(search_terms)
        query_embedding = await embedding_provider.embed_query(search_text)

        entity_hits = await self._graph_rag_vectors.search_entity_embeddings(
            collection_id=collection.id,
            query_embedding=query_embedding,
            top_k=20,
        )

        entities: list[dict[str, Any]] = []
        entity_names: list[str] = []
        graph_storage = self._get_graph_storage(collection.id)
        collection_id_str = str(collection.id)

        for hit in entity_hits:
            name = hit.metadata.get("name", "")
            if name and name not in entity_names:
                entity_names.append(name)
                node = await graph_storage.get_lightrag_node(name, collection_id_str)
                if node:
                    entities.append(node)

        relationships: list[dict[str, Any]] = []
        rel_ids_seen: set[str] = set()
        for entity in entities:
            name = entity.get("name", "")
            if not name:
                continue
            try:
                edges = await graph_storage.get_lightrag_node_edges(name, collection_id_str)
                for src, tgt in edges:
                    edge_data = await graph_storage.get_lightrag_edge(src, tgt, collection_id_str)
                    if edge_data:
                        edge_id = edge_data.get("id", f"{src}__{tgt}")
                        if edge_id not in rel_ids_seen:
                            rel_ids_seen.add(edge_id)
                            relationships.append(edge_data)
            except Exception:
                continue

        chunk_ids = set()
        for entity in entities:
            chunk_ids.update(entity.get("source_ids") or [])
        for rel in relationships:
            chunk_ids.update(rel.get("source_ids") or [])

        chunks = await self._get_chunks_by_hashes(collection.id, list(chunk_ids))

        entity_context, entities_used = self._build_budgeted_context(
            [f"{e.get('name', '?')} ({e.get('type', '?')}): {e.get('description', '')}" for e in entities],
            [e.get("name", "") for e in entities],
            6000,
        )

        rel_context, rels_used = self._build_budgeted_context(
            [f"{r.get('id', '?')}: {r.get('description', '')}" for r in relationships],
            [r.get("id", "") for r in relationships],
            8000,
        )

        chunk_context, _ = self._build_budgeted_context(
            [c["content"] for c in chunks],
            [c["id"] for c in chunks],
            30000,
        )

        context = f"""Context:
Entities:
{entity_context or "(none)"}
Relationships:
{rel_context or "(none)"}
Source Text:
{chunk_context or "(none)"}"""

        response = await self._generate_lightrag_response(context, question, llm_provider)

        return QueryResult(
            response=response,
            entities_used=entities_used,
            relationships_used=rels_used,
            mode="local",
        )

    async def _lightrag_query_global(
        self,
        question: str,
        collection: Collection,
        keywords: tuple[list[str], list[str]],
        embedding_provider: EmbeddingProvider,
        llm_provider: LLMProvider,
    ) -> QueryResult:
        """Global mode: relationship-focused retrieval.

        1. Search relationship embeddings with high-level keywords
        2. For each relationship, get connected entities
        3. Collect source chunks
        4. Build context with token budgets
        5. Call LLM
        """
        high_level, low_level = keywords
        search_terms = high_level if high_level else [question]
        search_text = " ".join(search_terms)
        query_embedding = await embedding_provider.embed_query(search_text)

        rel_hits = await self._graph_rag_vectors.search_relationship_embeddings(
            collection_id=collection.id,
            query_embedding=query_embedding,
            top_k=30,
        )

        relationships: list[dict[str, Any]] = []
        rel_ids: list[str] = []
        graph_storage = self._get_graph_storage(collection.id)
        collection_id_str = str(collection.id)

        for hit in rel_hits:
            meta = hit.metadata
            src_name = meta.get("source_name", "")
            tgt_name = meta.get("target_name", "")
            if src_name and tgt_name:
                rel_id = f"{src_name}__{tgt_name}"
                if rel_id not in rel_ids:
                    rel_ids.append(rel_id)
                    edge = await graph_storage.get_lightrag_edge(src_name, tgt_name, collection_id_str)
                    if edge:
                        relationships.append(edge)

        entity_ids_set: set[str] = set()
        entities: list[dict[str, Any]] = []
        for rel in relationships:
            for entity_name in (rel.get("source_name"), rel.get("target_name")):
                if entity_name and entity_name not in entity_ids_set:
                    entity_ids_set.add(entity_name)
                    node = await graph_storage.get_lightrag_node(entity_name, collection_id_str)
                    if node:
                        entities.append(node)

        chunk_ids = set()
        for rel in relationships:
            chunk_ids.update(rel.get("source_ids") or [])
        for entity in entities:
            chunk_ids.update(entity.get("source_ids") or [])

        chunks = await self._get_chunks_by_hashes(collection.id, list(chunk_ids))

        rel_context, rels_used = self._build_budgeted_context(
            [f"{r.get('id', '?')}: {r.get('description', '')}" for r in relationships],
            [r.get("id", "") for r in relationships],
            8000,
        )

        entity_context, entities_used = self._build_budgeted_context(
            [f"{e.get('name', '?')} ({e.get('type', '?')}): {e.get('description', '')}" for e in entities],
            [e.get("name", "") for e in entities],
            6000,
        )

        chunk_context, _ = self._build_budgeted_context(
            [c["content"] for c in chunks],
            [c["id"] for c in chunks],
            30000,
        )

        context = f"""Context:
Relationships:
{rel_context or "(none)"}
Entities:
{entity_context or "(none)"}
Source Text:
{chunk_context or "(none)"}"""

        response = await self._generate_lightrag_response(context, question, llm_provider)

        return QueryResult(
            response=response,
            entities_used=entities_used,
            relationships_used=rels_used,
            mode="global",
        )

    async def _lightrag_query_hybrid(
        self,
        question: str,
        collection: Collection,
        keywords: tuple[list[str], list[str]],
        embedding_provider: EmbeddingProvider,
        llm_provider: LLMProvider,
    ) -> QueryResult:
        """Hybrid mode: merge local + global retrieval."""
        local_result = await self._lightrag_query_local(
            question, collection, keywords, embedding_provider, llm_provider,
        )
        global_result = await self._lightrag_query_global(
            question, collection, keywords, embedding_provider, llm_provider,
        )

        merged_entities = self._merge_unique(local_result.entities_used, global_result.entities_used)
        merged_rels = self._merge_unique(local_result.relationships_used, global_result.relationships_used)
        merged_response = local_result.response

        if global_result.response and global_result.response != local_result.response:
            if isinstance(llm_provider, LocalEchoLLMProvider):
                merged_response = local_result.response
            else:
                merged_response = local_result.response

        return QueryResult(
            response=merged_response,
            entities_used=merged_entities,
            relationships_used=merged_rels,
            mode="hybrid",
        )

    async def _lightrag_query_mix(
        self,
        question: str,
        collection: Collection,
        keywords: tuple[list[str], list[str]],
        embedding_provider: EmbeddingProvider,
        llm_provider: LLMProvider,
    ) -> QueryResult:
        """Mix mode: combine local + global + naive retrieval.

        Runs all three strategies, deduplicates entities/relationships,
        and merges chunks with token budgets.
        """
        local_result = await self._lightrag_query_local(
            question, collection, keywords, embedding_provider, llm_provider,
        )
        global_result = await self._lightrag_query_global(
            question, collection, keywords, embedding_provider, llm_provider,
        )
        naive_result = await self._lightrag_query_naive(
            question, collection, embedding_provider, llm_provider,
        )

        merged_entities = self._merge_unique(
            local_result.entities_used, global_result.entities_used,
        )
        merged_rels = self._merge_unique(
            local_result.relationships_used, global_result.relationships_used,
        )

        response = local_result.response
        if isinstance(llm_provider, LocalEchoLLMProvider):
            response = local_result.response

        return QueryResult(
            response=response,
            entities_used=merged_entities,
            relationships_used=merged_rels,
            mode="mix",
        )

    @staticmethod
    def _merge_unique(first: list[str], second: list[str]) -> list[str]:
        return list(dict.fromkeys(first + second))

    @staticmethod
    def _build_budgeted_context(
        texts: list[str],
        ids: list[str],
        max_tokens: int,
    ) -> tuple[str, list[str]]:
        """Build context string respecting approximate token budget."""
        if not texts:
            return "", []

        used_ids = []
        used_parts = []
        total_tokens = 0

        for t, item_id in zip(texts, ids):
            tokens = max(1, len(t.split()))
            if total_tokens + tokens <= max_tokens:
                used_parts.append(t)
                used_ids.append(item_id)
                total_tokens += tokens
            else:
                remaining = max_tokens - total_tokens
                if remaining > 10:
                    max_chars = remaining * 4
                    truncated = t[:max_chars] + "..."
                    used_parts.append(truncated)
                break

        return "\n\n".join(used_parts), used_ids

    async def _get_chunks_by_hashes(
        self, collection_id: uuid.UUID, chunk_hashes: list[str]
    ) -> list[dict]:
        """Retrieve chunk contents from per-collection chunk_embeddings by hashes."""
        if not chunk_hashes:
            return []

        tbl = table_name(collection_id, "chunk_embeddings")
        async with AsyncSessionLocal() as session:
            placeholders = ",".join(f"'{h}'" for h in chunk_hashes)
            result = await session.execute(
                text(
                    f"SELECT id::text, content FROM {tbl} "
                    f"WHERE chunk_hash IN ({placeholders}) "
                    f"ORDER BY chunk_index"
                )
            )
            return [{"id": row[0], "content": row[1]} for row in result]

    async def _generate_lightrag_response(
        self, context: str, question: str, llm_provider: LLMProvider
    ) -> str:
        if isinstance(llm_provider, LocalEchoLLMProvider):
            return context
        return await llm_provider.chat([
            {
                "role": "system",
                "content": (
                    "Use the provided context to answer the question. "
                    "Draw on the entities, relationships, and source text to reason through your answer. "
                    "Write in natural prose. If the context is insufficient, acknowledge it briefly."
                ),
            },
            {
                "role": "user",
                "content": f"{context}\n\nQuestion: {question}",
            },
        ])

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
        embedding_provider = await self._get_embedding_provider_for_collection(collection)
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
        """LightRAG ingestion: extract → store in FalkorDB + pgvector.

        Unlike custom_graph_rag, LightRAG uses entity NAME as the node ID,
        skips incremental entity resolution, and stores full metadata on
        FalkorDB nodes/edges directly.
        """
        embedding_provider = await self._get_embedding_provider_for_collection(collection)
        llm_provider = await self._get_llm_provider(namespace_id=collection.namespace_id, llm_profile_id=collection.llm_profile_id)

        cached = await self._get_raw_extraction(chunk_hash, collection.id)
        if cached:
            return ChunkIngestionResult(
                chunk_hash=chunk_hash,
                entity_count=len(cached.entities),
                relationship_count=len(cached.relationships),
            )

        extractor = LLMGraphExtractor(llm=llm_provider)
        extraction = await extractor.extract_with_gleaning(text=text, max_gleaning=1)

        await self._save_raw_extraction(
            chunk_hash=chunk_hash,
            collection_id=collection.id,
            extraction=extraction,
        )

        chunk_embedding = await embedding_provider.embed_query(text)
        await self._graph_rag_vectors.upsert_chunk_embedding(
            collection_id=collection.id,
            chunk_hash=chunk_hash,
            chunk_index=0,
            content=text,
            embedding=chunk_embedding,
        )

        if not extraction.entities and not extraction.relationships:
            return ChunkIngestionResult(
                chunk_hash=chunk_hash, entity_count=0, relationship_count=0,
            )

        collection_id_str = str(collection.id)
        graph_storage = self._get_graph_storage(collection.id)

        entity_ids_resolved: dict[str, str] = {}

        for entity in extraction.entities:
            name = entity.name
            entity_ids_resolved[name] = name
            entity_uuid = self._deterministic_uuid(collection.id, name)

            if not await graph_storage.has_lightrag_node(name, collection_id_str):
                await graph_storage.upsert_lightrag_node(
                    node_name=name,
                    collection_id=collection_id_str,
                    properties={
                        "type": entity.entity_type,
                        "description": entity.description,
                        "source_ids": [chunk_hash],
                    },
                )
            else:
                existing = await graph_storage.get_lightrag_node(name, collection_id_str)
                if existing:
                    source_ids = existing.get("source_ids") or []
                    if chunk_hash not in source_ids:
                        source_ids.append(chunk_hash)
                    existing_desc = existing.get("description", "")
                    merged_desc = (
                        existing_desc + "; " + entity.description
                        if existing_desc and entity.description
                        else (existing_desc or entity.description)
                    )
                    await graph_storage.upsert_lightrag_node(
                        node_name=name,
                        collection_id=collection_id_str,
                        properties={
                            "type": entity.entity_type,
                            "description": merged_desc,
                            "source_ids": source_ids[:300],
                        },
                    )

            # Ensure entity exists in graph_entities for FK constraint
            async with AsyncSessionLocal() as session:
                await session.execute(
                    pg_insert(GraphEntity)
                    .values(
                        id=entity_uuid,
                        canonical_name=name,
                        primary_type=entity.entity_type,
                        description_count=0,
                        collection_id=collection.id,
                    )
                    .on_conflict_do_nothing(
                        constraint="uq_graph_entities_canonical_name_collection_id"
                    )
                )
                await session.commit()

            desc_embedding = await embedding_provider.embed_query(entity.description)
            desc_id = self._deterministic_uuid(collection.id, f"desc:{name}:{chunk_hash}")
            await self._graph_rag_vectors.upsert_entity_embedding(
                entity_id=entity_uuid,
                collection_id=collection.id,
                name=name,
                description=entity.description,
                description_id=desc_id,
                embedding=desc_embedding,
            )

        for rel in extraction.relationships:
            source_name = rel.source_name
            target_name = rel.target_name

            if source_name not in entity_ids_resolved or target_name not in entity_ids_resolved:
                continue

            rel_id_str = f"{source_name}__{target_name}"
            rel_uuid = self._deterministic_uuid(collection.id, rel_id_str)
            source_entity_uuid = self._deterministic_uuid(collection.id, source_name)
            target_entity_uuid = self._deterministic_uuid(collection.id, target_name)

            # Ensure relationship exists in graph_relationships for FK constraint
            async with AsyncSessionLocal() as session:
                await session.execute(
                    pg_insert(GraphRelationship)
                    .values(
                        id=rel_uuid,
                        source_entity_id=source_entity_uuid,
                        target_entity_id=target_entity_uuid,
                        weight=int(rel.weight * 10),
                        keywords=rel.keywords,
                        collection_id=collection.id,
                    )
                    .on_conflict_do_nothing(index_elements=["id"])
                )
                await session.commit()

            rel_embedding = await embedding_provider.embed_query(rel.description)
            await self._graph_rag_vectors.upsert_relationship_embedding(
                relationship_id=rel_uuid,
                collection_id=collection.id,
                source_name=source_name,
                target_name=target_name,
                description=rel.description,
                embedding=rel_embedding,
            )

            await graph_storage.upsert_lightrag_edge(
                source_name=source_name,
                target_name=target_name,
                collection_id=collection_id_str,
                properties={
                    "id": rel_id_str,
                    "description": rel.description,
                    "keywords": rel.keywords,
                    "weight": int(rel.weight * 10),
                    "source_ids": [chunk_hash],
                },
            )

        return ChunkIngestionResult(
            chunk_hash=chunk_hash,
            entity_count=len(extraction.entities),
            relationship_count=len(extraction.relationships),
        )

    @staticmethod
    def _deterministic_uuid(collection_id: uuid.UUID, name: str) -> uuid.UUID:
        """Generate a deterministic UUID scoped to a collection."""
        import hashlib
        return uuid.UUID(hashlib.md5(f"{collection_id}:{name}".encode()).hexdigest())

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
                sanitization_flags=(
                    {"severity": report.severity, "details": report.details}
                    if report.severity != "none" else None
                ),
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
        embedding_provider = await self._get_embedding_provider_for_collection(collection)
        query_embedding = await embedding_provider.embed_query(question)
        results = await self._vector_store.query_chunks(
            collection_id=collection.id,
            query_embedding=query_embedding,
            top_k=settings.vector_query_top_k,
        )
        chunks = [r["content"] for r in results]
        response = await self._generate_vector_answer(
            question=question,
            chunks=chunks,
            namespace_id=namespace_id,
            llm_profile_id=llm_profile_id,
        )
        return QueryResult(
            response=response, entities_used=[], relationships_used=[], mode=mode,
        )

    async def _get_embedding_provider_for_collection(
        self, collection: Collection,
    ) -> EmbeddingProvider:
        if collection.embedding_profile_id is None:
            return get_embedding_provider()
        async with AsyncSessionLocal() as session:
            profile = await session.get(Profile, collection.embedding_profile_id)
            if not profile:
                raise ValueError(f"Embedding profile {collection.embedding_profile_id} not found")
            api_key, cred_base_url = await self._get_profile_credential_info(session, profile)
            base_url = profile.base_url or cred_base_url
            return get_embedding_provider(
                provider_name=profile.provider,
                model=profile.model,
                dimensions=profile.dimensions,
                api_key=api_key,
                base_url=base_url,
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
            namespace_id=namespace_id, llm_profile_id=llm_profile_id,
        )
        if isinstance(llm_provider, LocalEchoLLMProvider):
            return chunks[0]
        context = "\n\n".join(f"Chunk {i + 1}:\n{c}" for i, c in enumerate(chunks))
        return await llm_provider.chat([
            {"role": "system", "content": "Answer using only the provided context."},
            {"role": "user", "content": f"Question:\n{question}\n\nContext:\n{context}"},
        ])

    async def _get_llm_provider(
        self,
        *,
        namespace_id: uuid.UUID,
        llm_profile_id: uuid.UUID | None = None,
    ) -> LLMProvider:
        if llm_profile_id is None:
            return get_llm_provider()
        async with AsyncSessionLocal() as session:
            profile = await session.get(Profile, llm_profile_id)
            if not profile or profile.namespace_id != namespace_id:
                raise ValueError("LLM profile not found in namespace")
            if profile.kind != "llm":
                raise ValueError("Profile kind must be llm")
            api_key, cred_base_url = await self._get_profile_credential_info(session, profile)
            base_url = profile.base_url or cred_base_url
            return get_llm_provider(
                provider_name=profile.provider, model=profile.model, api_key=api_key, base_url=base_url,
            )

    async def _get_profile_credential_info(self, session, profile: Profile) -> tuple[str | None, str | None]:
        if profile.credential_id is None:
            return None, None
        credential = await session.get(Credential, profile.credential_id)
        if not credential:
            raise ValueError(f"Credential {profile.credential_id} not found")
        return self._crypto.decrypt(credential.encrypted_secret), credential.base_url
