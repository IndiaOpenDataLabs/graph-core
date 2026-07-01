"""Chunk ingestion processing functions extracted from GraphService.

Module-level async functions that handle the per-chunk ingestion pipeline
for vector, custom_graph_rag, and light_rag strategies.
"""

import hashlib
import uuid
from dataclasses import dataclass

from sqlalchemy import func, select
from sqlalchemy.dialects.postgresql import insert as pg_insert

from graph_core.database import AsyncSessionLocal
from graph_core.embedding import get_embedding_provider
from graph_core.embedding.interface import EmbeddingProvider
from graph_core.llm import get_llm_provider
from graph_core.llm.interface import LLMProvider
from graph_core.models.collection import Collection
from graph_core.models.credential import Credential
from graph_core.models.graph_rag import (
    EntityDescription,
    GraphEntity,
    GraphRelationship,
    GraphRelationshipType,
    RawChunkExtraction,
    RelationshipDescription,
)
from graph_core.models.ingestion import IngestionRecord
from graph_core.models.profile import Profile
from graph_core.models.rel_types import normalize_rel_type, relationship_embedding_text
from graph_core.services.crypto import CredentialCrypto
from graph_core.services.document_identity import (
    document_id_for_chunk,
    document_id_for_path,
    normalize_document_path,
)
from graph_core.services.graph_rag.extractor import (
    ExtractedEntity,
    ExtractedRelationship,
    ExtractionResult,
    LLMGraphExtractor,
)
from graph_core.services.sanitizer import TextSanitizer
from graph_core.storage.graph_names import collection_graph_name
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
_CUSTOM_GRAPH_CONTEXT_EXTRACTION_VERSION = "custom_context_v1"


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


async def resolve_llm_provider(
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


async def resolve_llm_provider_from_collection(
    collection: Collection,
) -> LLMProvider:
    """Resolve the LLM provider using a collection's llm_profile_id."""
    return await resolve_llm_provider(
        namespace_id=collection.namespace_id,
        llm_profile_id=collection.llm_profile_id,
    )


# ── Utility functions ──


def deterministic_uuid(collection_id: uuid.UUID, name: str) -> uuid.UUID:
    """Generate a deterministic UUID scoped to a collection."""
    return uuid.UUID(
        hashlib.md5(f"{collection_id}:{name}".encode()).hexdigest()
    )


def _short_text(value: str, max_chars: int = 800) -> str:
    normalized = " ".join((value or "").split())
    if len(normalized) <= max_chars:
        return normalized
    return normalized[: max_chars - 3].rstrip() + "..."


def _entity_type_for_mention(
    name: str,
    entity_by_name: dict[str, ExtractedEntity],
) -> str:
    entity = entity_by_name.get(name) or entity_by_name.get(name.strip().title())
    return normalize_rel_type(entity.entity_type if entity else "CONCEPT")


def _entity_description_for_mention(
    name: str,
    entity_by_name: dict[str, ExtractedEntity],
) -> str:
    entity = entity_by_name.get(name) or entity_by_name.get(name.strip().title())
    return entity.description if entity else ""


async def _graph_relationship_type_id(
    session,
    *,
    collection_id: uuid.UUID,
    rel_type: str,
) -> tuple[uuid.UUID, str]:
    canonical_type = normalize_rel_type(rel_type)
    existing = await session.execute(
        select(GraphRelationshipType).where(
            GraphRelationshipType.collection_id == collection_id,
            GraphRelationshipType.canonical_type == canonical_type,
        )
    )
    row = existing.scalar_one_or_none()
    if row:
        return row.id, row.canonical_type

    rel_type_id = deterministic_uuid(collection_id, f"rel_type:{canonical_type}")
    await session.execute(
        pg_insert(GraphRelationshipType)
        .values(
            id=rel_type_id,
            collection_id=collection_id,
            canonical_type=canonical_type,
        )
        .on_conflict_do_nothing(
            constraint="uq_graph_relationship_types_collection_canonical_type"
        )
    )
    return rel_type_id, canonical_type


async def _upsert_context_entity(
    session,
    *,
    collection: Collection,
    entity_id: uuid.UUID,
    canonical_name: str,
    primary_type: str,
    description: str,
    chunk_hash: str,
    document_id: uuid.UUID | None,
    document_path: str | None,
    embedding_provider: EmbeddingProvider,
) -> None:
    await session.execute(
        pg_insert(GraphEntity)
        .values(
            id=entity_id,
            collection_id=collection.id,
            canonical_name=canonical_name[:256],
            primary_type=primary_type[:64],
            description_count=1,
        )
        .on_conflict_do_update(
            index_elements=[GraphEntity.id],
            set_={
                "canonical_name": canonical_name[:256],
                "primary_type": primary_type[:64],
                "description_count": 1,
            },
        )
    )
    description_id = deterministic_uuid(
        collection.id,
        f"desc:{entity_id}:{document_id or chunk_hash}",
    )
    await session.execute(
        pg_insert(EntityDescription)
        .values(
            id=description_id,
            entity_id=entity_id,
            description=description,
            weight=1,
            source_chunk_hashes=[chunk_hash],
            document_id=document_id,
            document_path=normalize_document_path(document_path)
            if document_path
            else None,
        )
        .on_conflict_do_update(
            index_elements=[EntityDescription.id],
            set_={
                "description": description,
                "source_chunk_hashes": [chunk_hash],
                "document_id": document_id,
                "document_path": normalize_document_path(document_path)
                if document_path
                else None,
            },
        )
    )
    embedding = await embedding_provider.embed_query(
        f"{canonical_name}: {description}"
    )
    await _graph_rag_vectors.upsert_entity_embedding(
        entity_id=entity_id,
        collection_id=collection.id,
        name=canonical_name[:256],
        description=description,
        description_id=description_id,
        embedding=embedding,
        document_id=document_id,
        document_path=document_path,
        session=session,
    )
    await _graph_rag_vectors.upsert_entity_centroid(
        entity_id=entity_id,
        collection_id=collection.id,
        canonical_name=canonical_name[:256],
        primary_type=primary_type[:64],
        description_count=1,
        embedding=embedding,
        session=session,
    )


async def _resolve_context_concept(
    session,
    *,
    collection: Collection,
    mention_name: str,
    mention_type: str,
    mention_description: str,
    chunk_hash: str,
    document_id: uuid.UUID | None,
    document_path: str | None,
    embedding_provider: EmbeddingProvider,
) -> tuple[uuid.UUID, str]:
    concept_type = normalize_rel_type(mention_type or "CONCEPT")
    primary_type = f"CONCEPT_{concept_type}"[:64]
    canonical_name = f"{concept_type}: {mention_name}"[:256]
    description = (
        f"Concept projection for {concept_type} mention {mention_name!r}. "
        f"{mention_description}"
    ).strip()
    embedding = await embedding_provider.embed_query(
        f"{canonical_name}: {description}"
    )

    exact = await session.execute(
        select(GraphEntity).where(
            GraphEntity.collection_id == collection.id,
            GraphEntity.primary_type == primary_type,
            func.lower(GraphEntity.canonical_name) == canonical_name.lower(),
        )
    )
    existing = exact.scalar_one_or_none()
    if existing:
        return existing.id, existing.canonical_name

    hits = await _graph_rag_vectors.search_entity_embeddings(
        collection_id=collection.id,
        query_embedding=embedding,
        top_k=12,
    )
    candidate_ids: list[uuid.UUID] = []
    score_by_id: dict[uuid.UUID, float] = {}
    for hit in hits:
        try:
            entity_id = uuid.UUID(str(hit.metadata.get("entity_id") or ""))
        except ValueError:
            continue
        candidate_ids.append(entity_id)
        score_by_id[entity_id] = 1.0 - float(hit.distance)
    if candidate_ids:
        rows = await session.execute(
            select(GraphEntity).where(
                GraphEntity.collection_id == collection.id,
                GraphEntity.id.in_(candidate_ids),
                GraphEntity.primary_type == primary_type,
            )
        )
        for candidate in rows.scalars().all():
            if score_by_id.get(candidate.id, 0.0) >= 0.93:
                return candidate.id, candidate.canonical_name

    concept_id = deterministic_uuid(
        collection.id,
        f"concept:{concept_type}:{mention_name.lower()}",
    )
    await _upsert_context_entity(
        session,
        collection=collection,
        entity_id=concept_id,
        canonical_name=canonical_name,
        primary_type=primary_type,
        description=description,
        chunk_hash=chunk_hash,
        document_id=document_id,
        document_path=document_path,
        embedding_provider=embedding_provider,
    )
    return concept_id, canonical_name


async def _upsert_context_relationship(
    session,
    *,
    collection: Collection,
    relationship_id: uuid.UUID,
    source_id: uuid.UUID,
    target_id: uuid.UUID,
    source_name: str,
    target_name: str,
    rel_type: str,
    description: str,
    keywords: list[str],
    weight: float,
    chunk_hash: str,
    document_id: uuid.UUID | None,
    document_path: str | None,
) -> dict[str, object]:
    rel_type_id, canonical_type = await _graph_relationship_type_id(
        session,
        collection_id=collection.id,
        rel_type=rel_type,
    )
    int_weight = max(1, int(float(weight or 1.0) * 10))
    await session.execute(
        pg_insert(GraphRelationship)
        .values(
            id=relationship_id,
            source_entity_id=source_id,
            target_entity_id=target_id,
            weight=int_weight,
            keywords=keywords,
            relationship_type_id=rel_type_id,
            rel_type=canonical_type,
            collection_id=collection.id,
        )
        .on_conflict_do_update(
            index_elements=[GraphRelationship.id],
            set_={
                "source_entity_id": source_id,
                "target_entity_id": target_id,
                "weight": int_weight,
                "keywords": keywords,
                "relationship_type_id": rel_type_id,
                "rel_type": canonical_type,
                "collection_id": collection.id,
            },
        )
    )
    description_id = deterministic_uuid(
        collection.id,
        f"rel_desc:{relationship_id}:{document_id or chunk_hash}",
    )
    await session.execute(
        pg_insert(RelationshipDescription)
        .values(
            id=description_id,
            relationship_id=relationship_id,
            description=description,
            keywords=keywords,
            weight=1,
            source_chunk_hashes=[chunk_hash],
            document_id=document_id,
            document_path=normalize_document_path(document_path)
            if document_path
            else None,
        )
        .on_conflict_do_update(
            index_elements=[RelationshipDescription.id],
            set_={
                "description": description,
                "keywords": keywords,
                "source_chunk_hashes": [chunk_hash],
                "document_id": document_id,
                "document_path": normalize_document_path(document_path)
                if document_path
                else None,
            },
        )
    )
    return {
        "source_id": str(source_id),
        "target_id": str(target_id),
        "id": str(relationship_id),
        "weight": int_weight,
        "keywords": keywords,
        "rel_type": canonical_type,
        "collection_id": str(collection.id),
        "document_id": str(document_id) if document_id else None,
        "document_path": document_path,
        "_source_name": source_name[:256],
        "_target_name": target_name[:256],
        "_description": description,
    }


def _falkor_node(
    *,
    entity_id: uuid.UUID,
    name: str,
    collection: Collection,
    document_id: uuid.UUID | None,
    document_path: str | None,
) -> dict[str, object]:
    return {
        "id": str(entity_id),
        "name": name[:256],
        "collection_id": str(collection.id),
        "document_id": str(document_id) if document_id else None,
        "document_path": document_path,
    }


def _enforce_namespace(collection: Collection, namespace_id: uuid.UUID) -> None:
    """Raise if the collection does not belong to the given namespace."""
    if collection.namespace_id != namespace_id:
        raise PermissionError(
            f"Collection {collection.id} does not belong to namespace {namespace_id}"
        )


def get_graph_storage(collection: Collection):
    """Return a FalkorDBGraphStorage scoped to the collection's own graph."""
    from graph_core.storage.graph_storage import FalkorDBGraphStorage

    graph_name = collection_graph_name(
        namespace_id=collection.namespace_id,
        collection_id=collection.id,
        collection_name=collection.name,
    )
    return FalkorDBGraphStorage(
        graph_name,
        namespace_id=collection.namespace_id,
    )


# ── Core ingestion entry point ──


async def ingest_collection_chunk(
    text: str,
    collection: Collection,
    namespace_id: uuid.UUID,
    chunk_index: int,
    domain: str | None = None,
    document_id: uuid.UUID | None = None,
    document_path: str | None = None,
) -> ChunkIngestionResult:
    """Route a sanitized text chunk to the appropriate strategy handler."""
    _enforce_namespace(collection, namespace_id)
    sanitized_text, report = _sanitizer.sanitize(text, str(namespace_id))
    chunk_hash = _sanitizer.chunk_hash(sanitized_text)
    normalized_document_path = (
        normalize_document_path(document_path) if document_path else None
    )
    effective_document_id = document_id
    if effective_document_id is None:
        if normalized_document_path:
            effective_document_id = document_id_for_path(
                collection.id, normalized_document_path
            )
        else:
            effective_document_id = document_id_for_chunk(collection.id, chunk_hash)

    if collection.strategy == "vector":
        result = await _ingest_vector_chunk(
            sanitized_text,
            collection,
            chunk_hash,
            report,
            chunk_index=chunk_index,
            document_id=effective_document_id,
            document_path=normalized_document_path,
        )
    elif collection.strategy == "light_rag":
        result = await _ingest_lightrag_chunk(
            sanitized_text,
            collection,
            chunk_hash,
            report,
            domain=domain,
            document_id=effective_document_id,
            document_path=normalized_document_path,
        )
    else:
        result = await _ingest_graph_chunk(
            sanitized_text,
            collection,
            chunk_hash,
            report,
            domain=domain,
            document_id=effective_document_id,
            document_path=normalized_document_path,
        )

    await _write_ledger(
        collection,
        chunk_hash,
        report,
        result,
        document_id=effective_document_id,
        document_path=normalized_document_path,
    )
    return result


# ── Strategy-specific ingestion ──


async def _ingest_vector_chunk(
    text: str,
    collection: Collection,
    chunk_hash: str,
    report,
    chunk_index: int,
    document_id: uuid.UUID | None = None,
    document_path: str | None = None,
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
                    "document_id": str(document_id) if document_id else None,
                    "document_path": document_path,
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
    document_id: uuid.UUID | None = None,
    document_path: str | None = None,
) -> ChunkIngestionResult:
    """Full Graph RAG pipeline: extract → resolve → store."""
    embedding_provider = await _resolve_embedding_provider(collection)
    llm_provider = await resolve_llm_provider(
        namespace_id=collection.namespace_id,
        llm_profile_id=collection.llm_profile_id,
    )

    cached = await _get_raw_extraction(
        chunk_hash,
        collection.id,
        domain=domain,
        cache_variant=_CUSTOM_GRAPH_CONTEXT_EXTRACTION_VERSION,
    )
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
            cache_variant=_CUSTOM_GRAPH_CONTEXT_EXTRACTION_VERSION,
            document_id=document_id,
            document_path=document_path,
        )

    chunk_embedding = await embedding_provider.embed_query(text)
    await _graph_rag_vectors.upsert_chunk_embedding(
        collection_id=collection.id,
        chunk_hash=chunk_hash,
        chunk_index=0,
        content=text,
        embedding=chunk_embedding,
        document_id=document_id,
        document_path=document_path,
    )

    if not extraction.entities and not extraction.relationships:
        return ChunkIngestionResult(
            chunk_hash=chunk_hash, entity_count=0, relationship_count=0,
        )

    context_seed = (
        normalize_document_path(document_path)
        if document_path
        else str(document_id or chunk_hash)
    )
    context_id = deterministic_uuid(
        collection.id,
        f"context:{context_seed}:{chunk_hash}",
    )
    context_name = f"context:{context_seed}:{chunk_hash[:12]}"[:256]
    context_description = (
        f"Source context for chunk {chunk_hash[:12]}"
        f"{' from ' + context_seed if context_seed else ''}. "
        f"Text excerpt: {_short_text(text, 1200)}"
    )
    entity_by_name = {entity.name: entity for entity in extraction.entities}
    nodes_to_upsert: list[dict[str, object]] = []
    edges_to_upsert: list[dict[str, object]] = []

    async with AsyncSessionLocal() as session:
        await _upsert_context_entity(
            session,
            collection=collection,
            entity_id=context_id,
            canonical_name=context_name,
            primary_type="CONTEXT",
            description=context_description,
            chunk_hash=chunk_hash,
            document_id=document_id,
            document_path=document_path,
            embedding_provider=embedding_provider,
        )
        nodes_to_upsert.append(
            _falkor_node(
                entity_id=context_id,
                name=context_name,
                collection=collection,
                document_id=document_id,
                document_path=document_path,
            )
        )

        for index, rel in enumerate(extraction.relationships):
            rel_type = normalize_rel_type(rel.rel_type)
            source_type = _entity_type_for_mention(rel.source_name, entity_by_name)
            target_type = _entity_type_for_mention(rel.target_name, entity_by_name)
            source_description = _entity_description_for_mention(
                rel.source_name,
                entity_by_name,
            )
            target_description = _entity_description_for_mention(
                rel.target_name,
                entity_by_name,
            )
            assertion_name = (
                f"{rel.source_name} {rel_type} {rel.target_name}"
            )[:256]
            assertion_id = deterministic_uuid(
                collection.id,
                f"assertion:{chunk_hash}:{index}:{assertion_name}",
            )
            source_mention_id = deterministic_uuid(
                collection.id,
                f"mention:{chunk_hash}:{index}:source:{rel.source_name}",
            )
            target_mention_id = deterministic_uuid(
                collection.id,
                f"mention:{chunk_hash}:{index}:target:{rel.target_name}",
            )
            assertion_description = (
                f"{context_name}: {assertion_name}. Evidence: {rel.description}"
            )
            source_mention_name = (
                f"mention:{chunk_hash[:12]}:{index}:source:{rel.source_name}"
            )[:256]
            target_mention_name = (
                f"mention:{chunk_hash[:12]}:{index}:target:{rel.target_name}"
            )[:256]

            await _upsert_context_entity(
                session,
                collection=collection,
                entity_id=assertion_id,
                canonical_name=assertion_name,
                primary_type="ASSERTION",
                description=assertion_description,
                chunk_hash=chunk_hash,
                document_id=document_id,
                document_path=document_path,
                embedding_provider=embedding_provider,
            )
            await _upsert_context_entity(
                session,
                collection=collection,
                entity_id=source_mention_id,
                canonical_name=source_mention_name,
                primary_type=f"MENTION_{source_type}"[:64],
                description=(
                    f"Context-local source mention {rel.source_name!r} "
                    f"of type {source_type}. {source_description}"
                ),
                chunk_hash=chunk_hash,
                document_id=document_id,
                document_path=document_path,
                embedding_provider=embedding_provider,
            )
            await _upsert_context_entity(
                session,
                collection=collection,
                entity_id=target_mention_id,
                canonical_name=target_mention_name,
                primary_type=f"MENTION_{target_type}"[:64],
                description=(
                    f"Context-local target mention {rel.target_name!r} "
                    f"of type {target_type}. {target_description}"
                ),
                chunk_hash=chunk_hash,
                document_id=document_id,
                document_path=document_path,
                embedding_provider=embedding_provider,
            )
            source_concept_id, source_concept_name = await _resolve_context_concept(
                session,
                collection=collection,
                mention_name=rel.source_name,
                mention_type=source_type,
                mention_description=source_description,
                chunk_hash=chunk_hash,
                document_id=document_id,
                document_path=document_path,
                embedding_provider=embedding_provider,
            )
            target_concept_id, target_concept_name = await _resolve_context_concept(
                session,
                collection=collection,
                mention_name=rel.target_name,
                mention_type=target_type,
                mention_description=target_description,
                chunk_hash=chunk_hash,
                document_id=document_id,
                document_path=document_path,
                embedding_provider=embedding_provider,
            )

            for node_id, name in (
                (assertion_id, assertion_name),
                (source_mention_id, source_mention_name),
                (target_mention_id, target_mention_name),
                (source_concept_id, source_concept_name),
                (target_concept_id, target_concept_name),
            ):
                nodes_to_upsert.append(
                    _falkor_node(
                        entity_id=node_id,
                        name=name,
                        collection=collection,
                        document_id=document_id,
                        document_path=document_path,
                    )
                )

            relationship_specs = [
                (
                    context_id,
                    assertion_id,
                    context_name,
                    assertion_name,
                    "HAS_ASSERTION",
                    assertion_description,
                    ["context", "assertion"],
                    1.0,
                ),
                (
                    assertion_id,
                    source_mention_id,
                    assertion_name,
                    source_mention_name,
                    "HAS_SUBJECT_MENTION",
                    f"{assertion_name} has source mention {rel.source_name}.",
                    ["assertion", "source", source_type.lower()],
                    1.0,
                ),
                (
                    assertion_id,
                    target_mention_id,
                    assertion_name,
                    target_mention_name,
                    "HAS_OBJECT_MENTION",
                    f"{assertion_name} has target mention {rel.target_name}.",
                    ["assertion", "target", target_type.lower()],
                    1.0,
                ),
                (
                    source_mention_id,
                    source_concept_id,
                    source_mention_name,
                    source_concept_name,
                    "DENOTES",
                    (
                        f"Source mention {rel.source_name!r} denotes "
                        f"{source_concept_name}."
                    ),
                    ["mention", "concept", source_type.lower()],
                    1.0,
                ),
                (
                    target_mention_id,
                    target_concept_id,
                    target_mention_name,
                    target_concept_name,
                    "DENOTES",
                    (
                        f"Target mention {rel.target_name!r} denotes "
                        f"{target_concept_name}."
                    ),
                    ["mention", "concept", target_type.lower()],
                    1.0,
                ),
                (
                    source_mention_id,
                    target_mention_id,
                    source_mention_name,
                    target_mention_name,
                    rel_type,
                    assertion_description,
                    rel.keywords,
                    rel.weight,
                ),
                (
                    context_id,
                    target_mention_id,
                    context_name,
                    target_mention_name,
                    rel_type,
                    assertion_description,
                    rel.keywords,
                    rel.weight,
                ),
            ]

            for spec_index, spec in enumerate(relationship_specs):
                (
                    source_id,
                    target_id,
                    source_name,
                    target_name,
                    spec_rel_type,
                    description,
                    keywords,
                    weight,
                ) = spec
                relationship_id = deterministic_uuid(
                    collection.id,
                    (
                        f"edge:{chunk_hash}:{index}:{spec_index}:"
                        f"{source_id}:{spec_rel_type}:{target_id}"
                    ),
                )
                edges_to_upsert.append(
                    await _upsert_context_relationship(
                        session,
                        collection=collection,
                        relationship_id=relationship_id,
                        source_id=source_id,
                        target_id=target_id,
                        source_name=source_name,
                        target_name=target_name,
                        rel_type=spec_rel_type,
                        description=description,
                        keywords=list(keywords),
                        weight=float(weight),
                        chunk_hash=chunk_hash,
                        document_id=document_id,
                        document_path=document_path,
                    )
                )

        await session.commit()

    for edge in edges_to_upsert:
        rel_embedding = await embedding_provider.embed_query(
            relationship_embedding_text(
                source_name=str(edge.get("_source_name") or ""),
                target_name=str(edge.get("_target_name") or ""),
                rel_type=str(edge.get("rel_type") or "RELATES_TO"),
                description=str(edge.get("_description") or ""),
                keywords=list(edge.get("keywords") or []),
            )
        )
        await _graph_rag_vectors.upsert_relationship_embedding(
            relationship_id=uuid.UUID(str(edge["id"])),
            collection_id=collection.id,
            source_name=str(edge.get("_source_name") or "")[:256],
            target_name=str(edge.get("_target_name") or "")[:256],
            description=str(edge.get("_description") or ""),
            embedding=rel_embedding,
            document_id=document_id,
            document_path=document_path,
        )

    graph_storage = get_graph_storage(collection)
    unique_nodes = list({str(n["id"]): n for n in nodes_to_upsert}.values())
    if unique_nodes:
        await graph_storage.upsert_nodes(unique_nodes)
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
    document_id: uuid.UUID | None = None,
    document_path: str | None = None,
) -> ChunkIngestionResult:
    """LightRAG ingestion: extract → store in FalkorDB + pgvector.

    Unlike custom_graph_rag, LightRAG uses entity NAME as the node ID,
    skips incremental entity resolution, and stores full metadata on
    FalkorDB nodes/edges directly.
    """
    embedding_provider = await _resolve_embedding_provider(collection)
    llm_provider = await resolve_llm_provider(
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
            document_id=document_id,
            document_path=document_path,
        )

    chunk_embedding = await embedding_provider.embed_query(text)
    await _graph_rag_vectors.upsert_chunk_embedding(
        collection_id=collection.id,
        chunk_hash=chunk_hash,
        chunk_index=0,
        content=text,
        embedding=chunk_embedding,
        document_id=document_id,
        document_path=document_path,
    )

    if not extraction.entities and not extraction.relationships:
        return ChunkIngestionResult(
            chunk_hash=chunk_hash, entity_count=0, relationship_count=0,
        )

    collection_id_str = str(collection.id)
    graph_storage = get_graph_storage(collection)

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
                    "document_id": str(document_id) if document_id else None,
                    "document_path": document_path,
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
                        "document_id": str(document_id) if document_id else None,
                        "document_path": document_path,
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
            document_id=document_id,
            document_path=document_path,
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
            document_id=document_id,
            document_path=document_path,
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
                "document_id": str(document_id) if document_id else None,
                "document_path": document_path,
            },
        )

    return ChunkIngestionResult(
        chunk_hash=chunk_hash,
        entity_count=len(extraction.entities),
        relationship_count=len(extraction.relationships),
    )


# ── Raw extraction cache ──


def _raw_extraction_model_key(
    domain: str | None,
    cache_variant: str | None = None,
) -> str | None:
    domain_key = f"domain:{domain}" if domain else None
    if not cache_variant:
        return domain_key
    if domain_key:
        return f"{cache_variant}:{domain_key}"
    return cache_variant


async def _save_raw_extraction(
    chunk_hash: str,
    collection_id: uuid.UUID,
    extraction: ExtractionResult,
    domain: str | None = None,
    cache_variant: str | None = None,
    document_id: uuid.UUID | None = None,
    document_path: str | None = None,
) -> None:
    """Persist raw LLM extraction to the database for deduplication."""
    async with AsyncSessionLocal() as session:
        record = RawChunkExtraction(
            chunk_content_hash=chunk_hash,
            collection_id=collection_id,
            document_id=document_id,
            document_path=(
                normalize_document_path(document_path) if document_path else None
            ),
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
            extraction_model=_raw_extraction_model_key(domain, cache_variant),
        )
        session.add(record)
        try:
            await session.commit()
        except Exception:
            await session.rollback()


async def _get_raw_extraction(
    chunk_hash: str,
    collection_id: uuid.UUID,
    domain: str | None = None,
    cache_variant: str | None = None,
) -> ExtractionResult | None:
    """Retrieve a cached extraction by chunk hash, or None if not found."""
    from graph_core.services.graph_rag.extractor import (
        ExtractedEntity,
    )

    async with AsyncSessionLocal() as session:
        result = await session.execute(
            select(RawChunkExtraction).where(
                RawChunkExtraction.chunk_content_hash == chunk_hash,
                RawChunkExtraction.collection_id == collection_id,
                RawChunkExtraction.extraction_model
                == _raw_extraction_model_key(domain, cache_variant),
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
    document_id: uuid.UUID | None = None,
    document_path: str | None = None,
) -> None:
    """Append an ingestion record to the audit ledger."""
    async with AsyncSessionLocal() as session:
        record = IngestionRecord(
            collection_id=collection.id,
            chunk_hash=chunk_hash,
            document_id=document_id,
            document_path=(
                normalize_document_path(document_path) if document_path else None
            ),
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
