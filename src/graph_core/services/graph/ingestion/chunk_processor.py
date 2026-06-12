"""Chunk ingestion processing functions extracted from GraphService.

Module-level async functions that handle the per-chunk ingestion pipeline
for vector, custom_graph_rag, and light_rag strategies.
"""

import hashlib
import uuid
from dataclasses import dataclass

from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert as pg_insert

from graph_core.database import AsyncSessionLocal
from graph_core.embedding import get_embedding_provider
from graph_core.embedding.interface import EmbeddingProvider
from graph_core.llm import get_llm_provider
from graph_core.llm.interface import LLMProvider
from graph_core.models.collection import Collection
from graph_core.models.credential import Credential
from graph_core.models.graph_rag import (
    GraphEntity,
    GraphRelationship,
    RawChunkExtraction,
)
from graph_core.models.ingestion import IngestionRecord
from graph_core.models.profile import Profile
from graph_core.services.crypto import CredentialCrypto
from graph_core.services.entity_name_cache import EntityNameCache
from graph_core.services.graph_rag.entity_resolver import (
    IncrementalEntityResolver,
)
from graph_core.services.graph_rag.extractor import (
    ExtractionResult,
    LLMGraphExtractor,
)
from graph_core.services.sanitizer import TextSanitizer
from graph_core.storage.graph_rag_vectors import GraphRAGVectorStore
from graph_core.storage.vector_store import VectorStore


@dataclass
class ChunkIngestionResult:
    chunk_hash: str
    entity_count: int
    relationship_count: int


# ── Module-level singleton dependencies ──

_sanitizer = TextSanitizer()
_vector_store = VectorStore()
_graph_rag_vectors = GraphRAGVectorStore()
_crypto = CredentialCrypto()


# ── Credential / provider resolution helpers ──


async def _resolve_credential(
    session, profile: Profile
) -> tuple[str | None, str | None]:
    """Decrypt a profile's credential, returning (api_key, base_url)."""
    if profile.credential_id is None:
        return None, None
    credential = await session.get(Credential, profile.credential_id)
    if not credential:
        raise ValueError(f"Credential {profile.credential_id} not found")
    return _crypto.decrypt(credential.encrypted_secret), credential.base_url


async def _resolve_embedding_provider(collection: Collection) -> EmbeddingProvider:
    """Resolve the embedding provider for a collection."""
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


async def _resolve_llm_provider(
    namespace_id: uuid.UUID,
    llm_profile_id: uuid.UUID | None = None,
) -> LLMProvider:
    """Resolve the LLM provider for a namespace."""
    if llm_profile_id is None:
        return get_llm_provider()
    async with AsyncSessionLocal() as session:
        profile = await session.get(Profile, llm_profile_id)
        if not profile or profile.namespace_id != namespace_id:
            raise ValueError("LLM profile not found in namespace")
        if profile.kind != "llm":
            raise ValueError("Profile kind must be llm")
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


# ── Utility functions ──


def deterministic_uuid(collection_id: uuid.UUID, name: str) -> uuid.UUID:
    """Generate a deterministic UUID scoped to a collection."""
    return uuid.UUID(
        hashlib.md5(f"{collection_id}:{name}".encode()).hexdigest()
    )


def _enforce_namespace(collection: Collection, namespace_id: uuid.UUID) -> None:
    """Raise if the collection does not belong to the given namespace."""
    if collection.namespace_id != namespace_id:
        raise PermissionError(
            f"Collection {collection.id} does not belong to namespace {namespace_id}"
        )


def get_graph_storage(collection_id: uuid.UUID):
    """Return a FalkorDBGraphStorage scoped to the collection's own graph."""
    from graph_core.storage.graph_storage import FalkorDBGraphStorage

    graph_name = f"collection_{str(collection_id).replace('-', '')}"
    return FalkorDBGraphStorage(graph_name)


# ── Core ingestion entry point ──


async def ingest_collection_chunk(
    text: str,
    collection: Collection,
    namespace_id: uuid.UUID,
    chunk_index: int,
    domain: str | None = None,
) -> ChunkIngestionResult:
    """Route a sanitized text chunk to the appropriate strategy handler."""
    _enforce_namespace(collection, namespace_id)
    sanitized_text, report = _sanitizer.sanitize(text, str(namespace_id))
    chunk_hash = _sanitizer.chunk_hash(sanitized_text)

    if collection.strategy == "vector":
        result = await _ingest_vector_chunk(
            sanitized_text, collection, chunk_hash, report, chunk_index=chunk_index,
        )
    elif collection.strategy == "light_rag":
        result = await _ingest_lightrag_chunk(
            sanitized_text, collection, chunk_hash, report, domain=domain,
        )
    else:
        result = await _ingest_graph_chunk(
            sanitized_text, collection, chunk_hash, report, domain=domain,
        )

    await _write_ledger(collection, chunk_hash, report, result)
    return result


# ── Strategy-specific ingestion ──


async def _ingest_vector_chunk(
    text: str,
    collection: Collection,
    chunk_hash: str,
    report,
    chunk_index: int,
) -> ChunkIngestionResult:
    """Ingest a chunk using pure vector embedding strategy."""
    embedding_provider = await _resolve_embedding_provider(collection)
    embedding = await embedding_provider.embed_query(text)
    token_count = len(text.split())
    await _vector_store.upsert_chunks(
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
    return ChunkIngestionResult(
        chunk_hash=chunk_hash, entity_count=0, relationship_count=0
    )


async def _ingest_graph_chunk(
    text: str,
    collection: Collection,
    chunk_hash: str,
    report,
    domain: str | None = None,
) -> ChunkIngestionResult:
    """Full Graph RAG pipeline: extract → resolve → store."""
    embedding_provider = await _resolve_embedding_provider(collection)
    llm_provider = await _resolve_llm_provider(
        namespace_id=collection.namespace_id,
        llm_profile_id=collection.llm_profile_id,
    )

    cached = await _get_raw_extraction(chunk_hash, collection.id, domain=domain)
    if cached:
        extraction = cached
    else:
        extractor = LLMGraphExtractor(llm=llm_provider)
        extraction = await extractor.extract_with_gleaning(
            text=text,
            max_gleaning=max(0, int(collection.gleaning_passes or 0)),
            domain=domain,
        )

        await _save_raw_extraction(
            chunk_hash=chunk_hash,
            collection_id=collection.id,
            extraction=extraction,
            domain=domain,
        )

    chunk_embedding = await embedding_provider.embed_query(text)
    await _graph_rag_vectors.upsert_chunk_embedding(
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

    resolver = IncrementalEntityResolver(
        embedding_provider=embedding_provider,
        collection_id=collection.id,
        domain=domain,
    )
    name_cache = EntityNameCache(str(collection.id))

    resolved_entity_ids: dict[str, uuid.UUID] = {}
    pending_cache: list[tuple[list[str], uuid.UUID]] = []

    async with AsyncSessionLocal() as session:
        for entity in extraction.entities:
            cached_id = await name_cache.get(entity.name)
            if cached_id:
                resolved_entity_ids[entity.name] = cached_id
                resolved_entity_ids[entity.name.strip().title()] = cached_id
                continue

            result = await resolver.resolve_entity(
                session=session,
                name=entity.name,
                entity_type=entity.entity_type,
                description=entity.description,
                source_chunk_hash=chunk_hash,
            )
            resolved_entity_ids[entity.name] = result.entity_id
            resolved_entity_ids[entity.name.strip().title()] = result.entity_id

            if result.is_new:
                pending_cache.append(
                    (
                        [
                            entity.name,
                            entity.name.strip().title(),
                            result.canonical_name,
                        ],
                        result.entity_id,
                    )
                )

        await session.commit()
        for names, entity_id in pending_cache:
            await name_cache.set_many(names, entity_id)

        canonical_name_by_id: dict[uuid.UUID, str] = {}
        if resolved_entity_ids:
            entity_ids = list(dict.fromkeys(resolved_entity_ids.values()))
            entity_rows = await session.execute(
                select(GraphEntity).where(GraphEntity.id.in_(entity_ids))
            )
            canonical_name_by_id = {
                entity_row.id: entity_row.canonical_name
                for entity_row in entity_rows.scalars().all()
            }

        nodes_to_upsert = []
        edges_to_upsert = []

        for rel in extraction.relationships:
            source_id = (
                resolved_entity_ids.get(rel.source_name)
                or resolved_entity_ids.get(rel.source_name.strip().title())
                or await name_cache.get(rel.source_name)
            )
            target_id = (
                resolved_entity_ids.get(rel.target_name)
                or resolved_entity_ids.get(rel.target_name.strip().title())
                or await name_cache.get(rel.target_name)
            )

            for is_source, name in [
                (True, rel.source_name),
                (False, rel.target_name),
            ]:
                if (source_id if is_source else target_id) is None:
                    synthetic = await resolver.resolve_entity(
                        session=session,
                        name=name,
                        entity_type="",
                        description="",
                        source_chunk_hash=chunk_hash,
                    )
                    await session.commit()
                    await name_cache.set_many(
                        [name, name.strip().title(), synthetic.canonical_name],
                        synthetic.entity_id,
                    )
                    resolved_entity_ids[name] = synthetic.entity_id
                    resolved_entity_ids[name.strip().title()] = synthetic.entity_id
                    canonical_name_by_id[synthetic.entity_id] = synthetic.canonical_name
                    if is_source:
                        source_id = synthetic.entity_id
                    else:
                        target_id = synthetic.entity_id

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
                rel_type=rel.rel_type,
            )
            persisted_rel = await session.get(
                GraphRelationship,
                rel_result.relationship_id,
            )
            await session.commit()

            source_name = canonical_name_by_id.get(
                source_id, rel.source_name.strip().title()
            )
            target_name = canonical_name_by_id.get(
                target_id, rel.target_name.strip().title()
            )
            nodes_to_upsert.append({
                "id": str(source_id),
                "name": source_name,
                "collection_id": str(collection.id),
            })
            nodes_to_upsert.append({
                "id": str(target_id),
                "name": target_name,
                "collection_id": str(collection.id),
            })
            edges_to_upsert.append({
                "source_id": str(source_id),
                "target_id": str(target_id),
                "id": str(rel_result.relationship_id),
                "weight": int(
                    persisted_rel.weight if persisted_rel else int(rel.weight * 10)
                ),
                "keywords": (
                    persisted_rel.keywords if persisted_rel else rel.keywords
                ),
                "rel_type": (
                    persisted_rel.rel_type if persisted_rel else rel.rel_type
                ),
                "collection_id": str(collection.id),
            })

    unique_nodes = {n["id"]: n for n in nodes_to_upsert}.values()

    graph_storage = get_graph_storage(collection.id)
    if unique_nodes:
        await graph_storage.upsert_nodes(list(unique_nodes))
    if edges_to_upsert:
        await graph_storage.upsert_edges(edges_to_upsert)

    return ChunkIngestionResult(
        chunk_hash=chunk_hash,
        entity_count=len(extraction.entities),
        relationship_count=len(extraction.relationships),
    )


async def _ingest_lightrag_chunk(
    text: str,
    collection: Collection,
    chunk_hash: str,
    report,
    domain: str | None = None,
) -> ChunkIngestionResult:
    """LightRAG ingestion: extract → store in FalkorDB + pgvector.

    Unlike custom_graph_rag, LightRAG uses entity NAME as the node ID,
    skips incremental entity resolution, and stores full metadata on
    FalkorDB nodes/edges directly.
    """
    embedding_provider = await _resolve_embedding_provider(collection)
    llm_provider = await _resolve_llm_provider(
        namespace_id=collection.namespace_id,
        llm_profile_id=collection.llm_profile_id,
    )

    cached = await _get_raw_extraction(chunk_hash, collection.id, domain=domain)
    if cached:
        extraction = cached
    else:
        extractor = LLMGraphExtractor(llm=llm_provider)
        extraction = await extractor.extract_with_gleaning(
            text=text,
            max_gleaning=max(0, int(collection.gleaning_passes or 0)),
            domain=domain,
        )

        await _save_raw_extraction(
            chunk_hash=chunk_hash,
            collection_id=collection.id,
            extraction=extraction,
            domain=domain,
        )

    chunk_embedding = await embedding_provider.embed_query(text)
    await _graph_rag_vectors.upsert_chunk_embedding(
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
    graph_storage = get_graph_storage(collection.id)

    entity_ids_resolved: dict[str, str] = {}

    for entity in extraction.entities:
        name = entity.name
        entity_ids_resolved[name] = name
        entity_uuid = deterministic_uuid(collection.id, name)

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
            existing = await graph_storage.get_lightrag_node(
                name, collection_id_str
            )
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

        desc_embedding = await embedding_provider.embed_query(
            entity.description
        )
        desc_id = deterministic_uuid(
            collection.id, f"desc:{name}:{chunk_hash}"
        )
        await _graph_rag_vectors.upsert_entity_embedding(
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

        if (
            source_name not in entity_ids_resolved
            or target_name not in entity_ids_resolved
        ):
            continue

        rel_id_str = f"{source_name}__{target_name}"
        rel_uuid = deterministic_uuid(collection.id, rel_id_str)
        source_entity_uuid = deterministic_uuid(collection.id, source_name)
        target_entity_uuid = deterministic_uuid(collection.id, target_name)

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
        await _graph_rag_vectors.upsert_relationship_embedding(
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


# ── Raw extraction cache ──


async def _save_raw_extraction(
    chunk_hash: str,
    collection_id: uuid.UUID,
    extraction: ExtractionResult,
    domain: str | None = None,
) -> None:
    """Persist raw LLM extraction to the database for deduplication."""
    async with AsyncSessionLocal() as session:
        record = RawChunkExtraction(
            chunk_content_hash=chunk_hash,
            collection_id=collection_id,
            entities_json=[
                {
                    "name": e.name,
                    "type": e.entity_type,
                    "description": e.description,
                }
                for e in extraction.entities
            ],
            relationships_json=[
                {
                    "source_name": r.source_name,
                    "target_name": r.target_name,
                    "description": r.description,
                    "keywords": r.keywords,
                    "weight": r.weight,
                    "rel_type": r.rel_type,
                }
                for r in extraction.relationships
            ],
            extraction_model=(f"domain:{domain}" if domain else None),
        )
        session.add(record)
        try:
            await session.commit()
        except Exception:
            await session.rollback()


async def _get_raw_extraction(
    chunk_hash: str, collection_id: uuid.UUID, domain: str | None = None
) -> ExtractionResult | None:
    """Retrieve a cached extraction by chunk hash, or None if not found."""
    from graph_core.services.graph_rag.extractor import (
        ExtractedEntity,
        ExtractedRelationship,
    )

    async with AsyncSessionLocal() as session:
        result = await session.execute(
            select(RawChunkExtraction).where(
                RawChunkExtraction.chunk_content_hash == chunk_hash,
                RawChunkExtraction.collection_id == collection_id,
                RawChunkExtraction.extraction_model
                == (f"domain:{domain}" if domain else None),
            )
        )
        record = result.scalar_one_or_none()
        if not record:
            return None

        entities = [
            ExtractedEntity(
                name=e["name"],
                entity_type=e["type"],
                description=e["description"],
            )
            for e in (record.entities_json or [])
        ]
        relationships = [
            ExtractedRelationship(
                source_name=r["source_name"],
                target_name=r["target_name"],
                description=r["description"],
                keywords=r.get("keywords", []),
                weight=r.get("weight", 1.0),
                rel_type=r.get("rel_type", "RELATES_TO"),
            )
            for r in (record.relationships_json or [])
        ]
        return ExtractionResult(entities=entities, relationships=relationships)


# ── Ledger ──


async def _write_ledger(
    collection: Collection,
    chunk_hash: str,
    report,
    result: ChunkIngestionResult,
) -> None:
    """Append an ingestion record to the audit ledger."""
    async with AsyncSessionLocal() as session:
        record = IngestionRecord(
            collection_id=collection.id,
            chunk_hash=chunk_hash,
            strategy=collection.strategy,
            entity_count=result.entity_count,
            relationship_count=result.relationship_count,
            sanitization_flags=(
                {"severity": report.severity, "details": report.details}
                if report.severity != "none"
                else None
            ),
        )
        session.add(record)
        await session.commit()
