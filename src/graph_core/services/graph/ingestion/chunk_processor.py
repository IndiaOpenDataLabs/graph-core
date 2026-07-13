"""Chunk ingestion processing functions extracted from GraphService.

Module-level async functions that handle the per-chunk ingestion pipeline
for vector, custom_graph_rag, and light_rag strategies.
"""

import hashlib
import uuid
from dataclasses import dataclass
from pathlib import PurePosixPath

from sqlalchemy import func, or_, select
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
from graph_core.services.graph.incremental_ingestion import (
    ContributionInput,
    EntityMappingInput,
    PredicateMappingInput,
    completed_segment,
    ingestion_contract,
    publish_chunk_delta,
)
from graph_core.services.graph.semantic_frames import (
    SemanticFrameInput,
    build_proposition_frame,
    persist_semantic_frames,
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


@dataclass(frozen=True)
class _SourceHierarchy:
    document_path: str | None
    folder_path: str | None
    headings: tuple[str, ...]


# ── Module-level singleton dependencies ──

_sanitizer = TextSanitizer()
_vector_store = VectorStore()
_graph_rag_vectors = GraphRAGVectorStore()
_crypto = CredentialCrypto()
_CUSTOM_GRAPH_CONTEXT_EXTRACTION_VERSION = "custom_context_v1"
_NON_SEMANTIC_EDGE_TYPES = {
    "ANTECEDENT_OF",
    "APPLIES_TO",
    "BLOCKS",
    "CONDITION",
    "CONCLUDES",
    "CONTAINS",
    "DENOTES",
    "EXCEPTION",
    "HAS_ASSERTION",
    "HAS_CONTEXT",
    "HAS_OBJECT_MENTION",
    "HAS_SECTION",
    "HAS_SUBJECT_MENTION",
    "OBJECT",
    "SUBJECT",
    "SUPPORTS",
}


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


async def _upsert_canonical_graph_entity(
    session,
    *,
    collection_id: uuid.UUID,
    entity_id: uuid.UUID,
    canonical_name: str,
    primary_type: str,
) -> uuid.UUID:
    entity_insert = pg_insert(GraphEntity).values(
        id=entity_id,
        collection_id=collection_id,
        canonical_name=canonical_name[:256],
        primary_type=primary_type[:64],
        description_count=1,
    )
    inserted_id = (
        await session.execute(
            entity_insert.on_conflict_do_nothing().returning(GraphEntity.id)
        )
    ).scalar_one_or_none()
    if inserted_id is not None:
        return inserted_id
    existing_id = (
        await session.execute(
            select(GraphEntity.id).where(
                or_(
                    GraphEntity.id == entity_id,
                    (
                        (GraphEntity.collection_id == collection_id)
                        & (GraphEntity.canonical_name == canonical_name[:256])
                    ),
                )
            )
        )
    ).scalar_one_or_none()
    if existing_id is None:
        raise RuntimeError(
            f"Graph entity conflict could not be resolved: {canonical_name[:256]}"
        )
    return existing_id


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
    precomputed_embedding: list[float] | None = None,
) -> uuid.UUID:
    canonical_entity_id = await _upsert_canonical_graph_entity(
        session,
        collection_id=collection.id,
        entity_id=entity_id,
        canonical_name=canonical_name,
        primary_type=primary_type,
    )
    description_id = deterministic_uuid(
        collection.id,
        f"desc:{canonical_entity_id}:{document_id or chunk_hash}",
    )
    await session.execute(
        pg_insert(EntityDescription)
        .values(
            id=description_id,
            entity_id=canonical_entity_id,
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
    embedding = precomputed_embedding or await embedding_provider.embed_query(
        f"{canonical_name}: {description}"
    )
    await _graph_rag_vectors.upsert_entity_embedding(
        entity_id=canonical_entity_id,
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
        entity_id=canonical_entity_id,
        collection_id=collection.id,
        canonical_name=canonical_name[:256],
        primary_type=primary_type[:64],
        description_count=1,
        embedding=embedding,
        session=session,
    )
    return canonical_entity_id


async def _upsert_reasoning_entity(
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
) -> uuid.UUID:
    """Persist an executable structural node without polluting vector seeds."""
    canonical_entity_id = await _upsert_canonical_graph_entity(
        session,
        collection_id=collection.id,
        entity_id=entity_id,
        canonical_name=canonical_name,
        primary_type=primary_type,
    )
    description_id = deterministic_uuid(
        collection.id,
        f"reasoning-desc:{canonical_entity_id}:{document_id or chunk_hash}",
    )
    await session.execute(
        pg_insert(EntityDescription)
        .values(
            id=description_id,
            entity_id=canonical_entity_id,
            description=description,
            weight=1,
            source_chunk_hashes=[chunk_hash],
            document_id=document_id,
            document_path=(
                normalize_document_path(document_path) if document_path else None
            ),
        )
        .on_conflict_do_update(
            index_elements=[EntityDescription.id],
            set_={"description": description},
        )
    )
    return canonical_entity_id


def _context_concept_values(
    mention_name: str,
    mention_type: str,
    mention_description: str,
) -> tuple[str, str, str]:
    concept_type = normalize_rel_type(mention_type or "CONCEPT")
    canonical_name = f"{concept_type}: {mention_name}"[:256]
    description = (
        f"Concept projection for {concept_type} mention {mention_name!r}. "
        f"{mention_description}"
    ).strip()
    return concept_type, canonical_name, description


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
    precomputed_embedding: list[float] | None = None,
) -> tuple[uuid.UUID, str]:
    concept_type, canonical_name, description = _context_concept_values(
        mention_name,
        mention_type,
        mention_description,
    )
    primary_type = f"CONCEPT_{concept_type}"[:64]
    embedding = precomputed_embedding or await embedding_provider.embed_query(
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
    concept_id = await _upsert_context_entity(
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
        precomputed_embedding=embedding,
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


def _parse_source_hierarchy(
    text: str,
    *,
    document_path: str | None,
) -> _SourceHierarchy:
    parsed_document_path: str | None = None
    folder_path: str | None = None
    headings: tuple[str, ...] = ()
    marker = "\n\nChunk text:\n"
    if text.startswith("Source hierarchy:\n") and marker in text:
        prefix = text.split(marker, 1)[0]
        for raw_line in prefix.splitlines()[1:]:
            label, _, value = raw_line.partition(":")
            value = value.strip()
            if not value:
                continue
            if label == "Document":
                parsed_document_path = normalize_document_path(value)
            elif label == "Folder":
                folder_path = normalize_document_path(value)
            elif label == "Section":
                headings = tuple(
                    part.strip()
                    for part in value.split(">")
                    if part.strip()
                )

    normalized_document_path = (
        parsed_document_path
        or (normalize_document_path(document_path) if document_path else None)
    )
    if folder_path is None and normalized_document_path:
        parent = PurePosixPath(normalized_document_path).parent.as_posix()
        if parent not in ("", "."):
            folder_path = parent
    return _SourceHierarchy(
        document_path=normalized_document_path,
        folder_path=folder_path,
        headings=headings,
    )


async def _upsert_source_hierarchy(
    session,
    *,
    collection: Collection,
    hierarchy: _SourceHierarchy,
    context_id: uuid.UUID,
    context_name: str,
    chunk_hash: str,
    document_id: uuid.UUID | None,
    document_path: str | None,
    embedding_provider: EmbeddingProvider,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    nodes: list[dict[str, object]] = []
    edges: list[dict[str, object]] = []
    previous_id: uuid.UUID | None = None
    previous_name: str | None = None

    async def add_node(
        *,
        stable_key: str,
        name: str,
        primary_type: str,
        description: str,
        rel_type: str | None = None,
    ) -> uuid.UUID:
        nonlocal previous_id, previous_name
        entity_id = deterministic_uuid(collection.id, stable_key)
        entity_id = await _upsert_context_entity(
            session,
            collection=collection,
            entity_id=entity_id,
            canonical_name=name[:256],
            primary_type=primary_type,
            description=description,
            chunk_hash=chunk_hash,
            document_id=document_id,
            document_path=document_path,
            embedding_provider=embedding_provider,
        )
        nodes.append(
            _falkor_node(
                entity_id=entity_id,
                name=name,
                collection=collection,
                document_id=document_id,
                document_path=document_path,
            )
        )
        if previous_id is not None and previous_name is not None and rel_type:
            relationship_id = deterministic_uuid(
                collection.id,
                f"source_hierarchy:{previous_id}:{rel_type}:{entity_id}",
            )
            edges.append(
                await _upsert_context_relationship(
                    session,
                    collection=collection,
                    relationship_id=relationship_id,
                    source_id=previous_id,
                    target_id=entity_id,
                    source_name=previous_name,
                    target_name=name,
                    rel_type=rel_type,
                    description=f"{previous_name} contains {name}.",
                    keywords=["source", "hierarchy"],
                    weight=1.0,
                    chunk_hash=chunk_hash,
                    document_id=document_id,
                    document_path=document_path,
                )
            )
        previous_id = entity_id
        previous_name = name
        return entity_id

    if hierarchy.folder_path:
        parts = [part for part in hierarchy.folder_path.split("/") if part]
        cumulative: list[str] = []
        for part in parts:
            cumulative.append(part)
            folder = "/".join(cumulative)
            await add_node(
                stable_key=f"source_folder:{folder}",
                name=f"folder:{folder}",
                primary_type="SOURCE_FOLDER",
                description=f"Source folder {folder}.",
                rel_type="CONTAINS",
            )

    if hierarchy.document_path:
        await add_node(
            stable_key=f"source_document:{hierarchy.document_path}",
            name=f"document:{hierarchy.document_path}",
            primary_type="SOURCE_DOCUMENT",
            description=f"Source document {hierarchy.document_path}.",
            rel_type="CONTAINS" if previous_id is not None else None,
        )

    section_path: list[str] = []
    for heading in hierarchy.headings:
        section_path.append(heading)
        section_key = " > ".join(section_path)
        document_scope = hierarchy.document_path or str(document_id or chunk_hash)
        await add_node(
            stable_key=f"source_section:{document_scope}:{section_key}",
            name=f"section:{document_scope}:{section_key}"[:256],
            primary_type="SOURCE_SECTION",
            description=(
                f"Source section {section_key}"
                f"{' in ' + document_scope if document_scope else ''}."
            ),
            rel_type="HAS_SECTION",
        )

    if previous_id is not None and previous_name is not None:
        relationship_id = deterministic_uuid(
            collection.id,
            f"source_hierarchy:{previous_id}:HAS_CONTEXT:{context_id}",
        )
        edges.append(
            await _upsert_context_relationship(
                session,
                collection=collection,
                relationship_id=relationship_id,
                source_id=previous_id,
                target_id=context_id,
                source_name=previous_name,
                target_name=context_name,
                rel_type="HAS_CONTEXT",
                description=f"{previous_name} contains context {context_name}.",
                keywords=["source", "context"],
                weight=1.0,
                chunk_hash=chunk_hash,
                document_id=document_id,
                document_path=document_path,
            )
        )

    return nodes, edges


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
    contract_version = ingestion_contract(
        domain,
        _CUSTOM_GRAPH_CONTEXT_EXTRACTION_VERSION,
    )
    existing_segment = await completed_segment(
        collection.id,
        chunk_hash,
        contract_version,
    )
    if existing_segment is not None:
        payload = dict(existing_segment.raw_extraction or {})
        return ChunkIngestionResult(
            chunk_hash=chunk_hash,
            entity_count=len(payload.get("entities", [])),
            relationship_count=len(payload.get("relationships", [])),
        )
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
        await publish_chunk_delta(
            collection=collection,
            chunk_hash=chunk_hash,
            chunk_index=0,
            contract_version=contract_version,
            raw_extraction=_extraction_payload(extraction),
            document_id=document_id,
            document_path=document_path,
            entity_mappings=[],
            predicate_mappings=[],
            contributions=[],
        )
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
    source_hierarchy = _parse_source_hierarchy(
        text,
        document_path=document_path,
    )
    entity_by_name = {entity.name: entity for entity in extraction.entities}
    nodes_to_upsert: list[dict[str, object]] = []
    edges_to_upsert: list[dict[str, object]] = []
    semantic_frames: list[SemanticFrameInput] = []
    entity_mappings: list[EntityMappingInput] = []
    predicate_mappings: list[PredicateMappingInput] = []

    async with AsyncSessionLocal() as session:
        context_id = await _upsert_context_entity(
            session,
            collection=collection,
            entity_id=context_id,
            canonical_name=context_name,
            primary_type="EVIDENCE_CHUNK",
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
        hierarchy_nodes, hierarchy_edges = await _upsert_source_hierarchy(
            session,
            collection=collection,
            hierarchy=source_hierarchy,
            context_id=context_id,
            context_name=context_name,
            chunk_hash=chunk_hash,
            document_id=document_id,
            document_path=document_path,
            embedding_provider=embedding_provider,
        )
        nodes_to_upsert.extend(hierarchy_nodes)
        edges_to_upsert.extend(hierarchy_edges)

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
            assertion_text = f"{rel.source_name} {rel_type} {rel.target_name}"
            assertion_fingerprint = hashlib.sha1(
                assertion_text.encode("utf-8")
            ).hexdigest()[:12]
            assertion_name = (
                f"assertion:{chunk_hash[:12]}:{index}:"
                f"{assertion_fingerprint}:{assertion_text}"
            )[:256]
            assertion_display_name = assertion_text[:256]
            assertion_id = deterministic_uuid(
                collection.id,
                f"assertion:{chunk_hash}:{index}:{assertion_name}",
            )
            assertion_description = (
                f"{context_name}: {assertion_text}. Evidence: {rel.description}"
            )
            _, source_concept_name_hint, source_concept_description = (
                _context_concept_values(
                    rel.source_name,
                    source_type,
                    source_description,
                )
            )
            _, target_concept_name_hint, target_concept_description = (
                _context_concept_values(
                    rel.target_name,
                    target_type,
                    target_description,
                )
            )
            entity_embeddings = await embedding_provider.embed_documents(
                [
                    f"{assertion_name}: {assertion_description}",
                    f"{source_concept_name_hint}: {source_concept_description}",
                    f"{target_concept_name_hint}: {target_concept_description}",
                ]
            )

            assertion_id = await _upsert_context_entity(
                session,
                collection=collection,
                entity_id=assertion_id,
                canonical_name=assertion_name,
                primary_type="PROPOSITION",
                description=assertion_description,
                chunk_hash=chunk_hash,
                document_id=document_id,
                document_path=document_path,
                embedding_provider=embedding_provider,
                precomputed_embedding=entity_embeddings[0],
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
                precomputed_embedding=entity_embeddings[1],
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
                precomputed_embedding=entity_embeddings[2],
            )
            entity_mappings.extend(
                [
                    EntityMappingInput(
                        local_key=f"relationship:{index}:subject",
                        raw_name=rel.source_name,
                        raw_type=source_type,
                        canonical_entity_id=source_concept_id,
                    ),
                    EntityMappingInput(
                        local_key=f"relationship:{index}:object",
                        raw_name=rel.target_name,
                        raw_type=target_type,
                        canonical_entity_id=target_concept_id,
                    ),
                ]
            )
            predicate_mappings.append(
                PredicateMappingInput(
                    raw_predicate=rel.rel_type,
                    canonical_predicate_id=deterministic_uuid(
                        collection.id,
                        f"rel_type:{rel_type}",
                    ),
                    confidence=float(
                        rel.predicate_properties.get("confidence", 0.0)
                    ),
                    inferred_properties=dict(rel.predicate_properties),
                )
            )

            for node_id, name in (
                (assertion_id, assertion_display_name),
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
                    source_concept_id,
                    target_concept_id,
                    source_concept_name,
                    target_concept_name,
                    rel_type,
                    assertion_description,
                    rel.keywords,
                    rel.weight,
                ),
                (
                    context_id,
                    assertion_id,
                    context_name,
                    assertion_display_name,
                    "SUPPORTS",
                    assertion_description,
                    ["evidence", "proposition"],
                    1.0,
                ),
                (
                    assertion_id,
                    source_concept_id,
                    assertion_display_name,
                    source_concept_name,
                    "SUBJECT",
                    f"{assertion_text} has canonical subject {source_concept_name}.",
                    ["proposition", "subject"],
                    1.0,
                ),
                (
                    assertion_id,
                    target_concept_id,
                    assertion_display_name,
                    target_concept_name,
                    "OBJECT",
                    f"{assertion_text} has canonical object {target_concept_name}.",
                    ["proposition", "object"],
                    1.0,
                ),
            ]
            if rel.conditions or rel.exceptions:
                rule_id = deterministic_uuid(
                    collection.id,
                    f"rule:{assertion_id}",
                )
                rule_name = f"rule:{assertion_text}"[:256]
                rule_id = await _upsert_reasoning_entity(
                    session,
                    collection=collection,
                    entity_id=rule_id,
                    canonical_name=rule_name,
                    primary_type="RULE",
                    description=f"Applicability rule for {assertion_text}.",
                    chunk_hash=chunk_hash,
                    document_id=document_id,
                    document_path=document_path,
                )
                nodes_to_upsert.append(
                    _falkor_node(
                        entity_id=rule_id,
                        name=rule_name,
                        collection=collection,
                        document_id=document_id,
                        document_path=document_path,
                    )
                )
                relationship_specs.append(
                    (
                        rule_id,
                        assertion_id,
                        rule_name,
                        assertion_display_name,
                        "CONCLUDES",
                        f"The rule concludes {assertion_text}.",
                        ["rule", "conclusion"],
                        1.0,
                    )
                )
                for condition_index, condition_text in enumerate(rel.conditions):
                    condition_id = deterministic_uuid(
                        collection.id,
                        f"condition:{assertion_id}:{condition_index}:{condition_text}",
                    )
                    condition_name = f"condition:{condition_text}"[:256]
                    condition_id = await _upsert_reasoning_entity(
                        session,
                        collection=collection,
                        entity_id=condition_id,
                        canonical_name=condition_name,
                        primary_type="CONDITION",
                        description=condition_text,
                        chunk_hash=chunk_hash,
                        document_id=document_id,
                        document_path=document_path,
                    )
                    nodes_to_upsert.append(
                        _falkor_node(
                            entity_id=condition_id,
                            name=condition_name,
                            collection=collection,
                            document_id=document_id,
                            document_path=document_path,
                        )
                    )
                    relationship_specs.append(
                        (
                            assertion_id,
                            condition_id,
                            assertion_display_name,
                            condition_name,
                            "CONDITION",
                            condition_text,
                            ["proposition", "condition"],
                            1.0,
                        )
                    )
                    relationship_specs.append(
                        (
                            condition_id,
                            rule_id,
                            condition_name,
                            rule_name,
                            "ANTECEDENT_OF",
                            condition_text,
                            ["condition", "rule"],
                            1.0,
                        )
                    )
                for exception_index, exception_text in enumerate(rel.exceptions):
                    exception_id = deterministic_uuid(
                        collection.id,
                        f"exception:{assertion_id}:{exception_index}:{exception_text}",
                    )
                    exception_name = f"exception:{exception_text}"[:256]
                    exception_id = await _upsert_reasoning_entity(
                        session,
                        collection=collection,
                        entity_id=exception_id,
                        canonical_name=exception_name,
                        primary_type="EXCEPTION",
                        description=exception_text,
                        chunk_hash=chunk_hash,
                        document_id=document_id,
                        document_path=document_path,
                    )
                    nodes_to_upsert.append(
                        _falkor_node(
                            entity_id=exception_id,
                            name=exception_name,
                            collection=collection,
                            document_id=document_id,
                            document_path=document_path,
                        )
                    )
                    relationship_specs.append(
                        (
                            assertion_id,
                            exception_id,
                            assertion_display_name,
                            exception_name,
                            "EXCEPTION",
                            exception_text,
                            ["proposition", "exception"],
                            1.0,
                        )
                    )
                    relationship_specs.append(
                        (
                            exception_id,
                            rule_id,
                            exception_name,
                            rule_name,
                            "BLOCKS",
                            exception_text,
                            ["exception", "blocker"],
                            1.0,
                        )
                    )

            for scope_index, scope_text in enumerate(rel.scopes):
                scope_id = deterministic_uuid(
                    collection.id,
                    f"scope:{assertion_id}:{scope_index}:{scope_text}",
                )
                scope_name = f"scope:{scope_text}"[:256]
                scope_id = await _upsert_reasoning_entity(
                    session,
                    collection=collection,
                    entity_id=scope_id,
                    canonical_name=scope_name,
                    primary_type="SCOPE",
                    description=scope_text,
                    chunk_hash=chunk_hash,
                    document_id=document_id,
                    document_path=document_path,
                )
                nodes_to_upsert.append(
                    _falkor_node(
                        entity_id=scope_id,
                        name=scope_name,
                        collection=collection,
                        document_id=document_id,
                        document_path=document_path,
                    )
                )
                relationship_specs.append(
                    (
                        assertion_id,
                        scope_id,
                        assertion_display_name,
                        scope_name,
                        "APPLIES_TO",
                        scope_text,
                        ["proposition", "scope"],
                        1.0,
                    )
                )

            semantic_relationship_id = deterministic_uuid(
                collection.id,
                (
                    f"edge:{chunk_hash}:{index}:0:"
                    f"{source_concept_id}:{rel_type}:{target_concept_id}"
                ),
            )
            semantic_frames.append(
                build_proposition_frame(
                    collection_id=collection.id,
                    chunk_hash=chunk_hash,
                    relationship_index=index,
                    proposition_entity_id=assertion_id,
                    source_relationship_id=semantic_relationship_id,
                    source_entity_id=source_concept_id,
                    source_name=rel.source_name,
                    target_entity_id=target_concept_id,
                    target_name=rel.target_name,
                    predicate=rel_type,
                    description=rel.description,
                    polarity=rel.polarity,
                    modality=rel.modality,
                    conditions=tuple(rel.conditions),
                    exceptions=tuple(rel.exceptions),
                    scopes=tuple(rel.scopes),
                    metadata={
                        "chunk_hash": chunk_hash,
                        "document_id": str(document_id) if document_id else None,
                        "document_path": document_path,
                        "domain": domain,
                    },
                )
            )

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

    contribution_by_object = {
        ("frame", frame.id): ContributionInput("frame", frame.id)
        for frame in semantic_frames
    }
    contribution_by_object.update(
        {
            ("entity", uuid.UUID(str(node["id"]))): ContributionInput(
                "entity", uuid.UUID(str(node["id"]))
            )
            for node in nodes_to_upsert
        }
    )
    contribution_by_object.update(
        {
            ("relationship", uuid.UUID(str(edge["id"]))): ContributionInput(
                "relationship", uuid.UUID(str(edge["id"]))
            )
            for edge in edges_to_upsert
        }
    )
    contributions = list(contribution_by_object.values())
    await persist_semantic_frames(
        collection,
        embedding_provider,
        semantic_frames,
    )

    semantic_edges = [
        edge
        for edge in edges_to_upsert
        if str(edge.get("rel_type") or "").upper() not in _NON_SEMANTIC_EDGE_TYPES
    ]
    for offset in range(0, len(semantic_edges), 64):
        edge_batch = semantic_edges[offset : offset + 64]
        embedding_texts = [
            relationship_embedding_text(
                source_name=str(edge.get("_source_name") or ""),
                target_name=str(edge.get("_target_name") or ""),
                rel_type=str(edge.get("rel_type") or "RELATES_TO"),
                description=str(edge.get("_description") or ""),
                keywords=list(edge.get("keywords") or []),
            )
            for edge in edge_batch
        ]
        embeddings = await embedding_provider.embed_documents(embedding_texts)
        await _graph_rag_vectors.upsert_relationship_embeddings(
            collection.id,
            [
                {
                    "relationship_id": uuid.UUID(str(edge["id"])),
                    "source_name": str(edge.get("_source_name") or "")[:256],
                    "target_name": str(edge.get("_target_name") or "")[:256],
                    "description": str(edge.get("_description") or ""),
                    "embedding": embedding,
                    "document_id": document_id,
                    "document_path": document_path,
                }
                for edge, embedding in zip(edge_batch, embeddings, strict=True)
            ],
        )

    graph_storage = get_graph_storage(collection)
    unique_nodes = list({str(n["id"]): n for n in nodes_to_upsert}.values())
    if unique_nodes:
        await graph_storage.upsert_nodes(unique_nodes)
    if edges_to_upsert:
        await graph_storage.upsert_edges(
            edges_to_upsert,
            merge_existing_keywords=False,
        )

    # Publish only after every SQL, vector, and graph write has succeeded. A
    # completed segment is the retry boundary and must never describe a partial
    # materialization.
    await publish_chunk_delta(
        collection=collection,
        chunk_hash=chunk_hash,
        chunk_index=0,
        contract_version=contract_version,
        raw_extraction=_extraction_payload(extraction),
        document_id=document_id,
        document_path=document_path,
        entity_mappings=entity_mappings,
        predicate_mappings=predicate_mappings,
        contributions=contributions,
    )

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


def _extraction_payload(extraction: ExtractionResult) -> dict[str, object]:
    return {
        "entities": [
            {
                "name": entity.name,
                "type": entity.entity_type,
                "description": entity.description,
            }
            for entity in extraction.entities
        ],
        "relationships": [
            {
                "source_name": relationship.source_name,
                "target_name": relationship.target_name,
                "description": relationship.description,
                "keywords": relationship.keywords,
                "weight": relationship.weight,
                "rel_type": relationship.rel_type,
                "conditions": list(relationship.conditions),
                "exceptions": list(relationship.exceptions),
                "scopes": list(relationship.scopes),
                "polarity": relationship.polarity,
                "modality": relationship.modality,
                "predicate_properties": relationship.predicate_properties,
            }
            for relationship in extraction.relationships
        ],
    }


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
    payload = _extraction_payload(extraction)
    async with AsyncSessionLocal() as session:
        record = RawChunkExtraction(
            chunk_content_hash=chunk_hash,
            collection_id=collection_id,
            document_id=document_id,
            document_path=(
                normalize_document_path(document_path) if document_path else None
            ),
            entities_json=payload["entities"],
            relationships_json=payload["relationships"],
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
                conditions=tuple(r.get("conditions", [])),
                exceptions=tuple(r.get("exceptions", [])),
                scopes=tuple(r.get("scopes", [])),
                polarity=r.get("polarity", "positive"),
                modality=r.get("modality", "asserted"),
                predicate_properties=dict(r.get("predicate_properties") or {}),
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
