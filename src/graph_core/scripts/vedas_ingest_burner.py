"""Prototype the incremental graph ingestion design against a local folder.

This is intentionally independent of ``ingest_collection_chunk``. New chunks get
one extraction call, one batched embedding call, bulk canonical graph writes, and
bulk graph storage writes. A completed chunk hash is skipped before any provider
or graph work. Raw extraction is immutable and reusable after a partial failure.

Usage:
    uv run python -m graph_core.scripts.vedas_ingest_burner
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sqlalchemy import delete, func, select, text
from sqlalchemy.dialects.postgresql import insert as pg_insert

from graph_core.config import settings
from graph_core.database import AsyncSessionLocal, _uuid_for_sql
from graph_core.embedding.interface import EmbeddingProvider
from graph_core.llm.interface import LLMProvider
from graph_core.models.collection import Collection
from graph_core.models.graph_rag import (
    EntityDescription,
    GraphEntity,
    GraphRelationship,
    GraphRelationshipType,
    RawChunkExtraction,
    RelationshipDescription,
)
from graph_core.models.incremental_graph import (
    GraphChunkContribution,
    GraphChunkSegment,
    GraphDerivedDependency,
    GraphEntityMapping,
    GraphPredicateMapping,
    GraphVersion,
)
from graph_core.models.ingestion import IngestionRecord
from graph_core.models.profile import Profile
from graph_core.models.rel_types import normalize_rel_type, relationship_embedding_text
from graph_core.services.chunking import DocumentChunker
from graph_core.services.document_identity import document_id_for_path
from graph_core.services.graph import GraphService
from graph_core.services.graph.ingestion.chunk_processor import (
    _resolve_embedding_provider,
    get_graph_storage,
    resolve_llm_provider_from_collection,
)
from graph_core.services.graph_rag.extractor import (
    ExtractedEntity,
    ExtractedRelationship,
    ExtractionResult,
    LLMGraphExtractor,
)
from graph_core.services.sanitizer import TextSanitizer
from graph_core.storage.vector_tables import get_collection_dimensions, table_name

DEFAULT_COLLECTION_ID = uuid.UUID("855f1950-ff65-4f47-9549-9a20f9a1332d")
DEFAULT_EMBEDDING_PROFILE_ID = uuid.UUID("7f4dbd8a-3ce7-4736-bcee-6a5e096a3b64")
DEFAULT_LLM_PROFILE_ID = uuid.UUID("5310e131-cfe7-4793-8590-db3c2deab8ff")
DEFAULT_SOURCE_DIR = Path.home() / "Downloads" / "Vedas_shorter"
EXTRACTION_CONTRACT = "incremental_raw_v1:domain:books"
SUPPORTED_SUFFIXES = {".md", ".markdown", ".txt"}

_sanitizer = TextSanitizer()


@dataclass(frozen=True)
class ChunkWork:
    document_path: str
    document_id: uuid.UUID
    chunk_index: int
    text: str


@dataclass(frozen=True)
class EntityRow:
    id: uuid.UUID
    name: str
    primary_type: str
    description: str
    description_id: uuid.UUID


@dataclass(frozen=True)
class RelationshipRow:
    id: uuid.UUID
    source_id: uuid.UUID
    source_name: str
    target_id: uuid.UUID
    target_name: str
    rel_type_id: uuid.UUID
    rel_type: str
    description: str
    description_id: uuid.UUID
    keywords: tuple[str, ...]
    weight: int


@dataclass(frozen=True)
class ChunkResult:
    status: str
    entity_count: int = 0
    relationship_count: int = 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Incremental-ingestion prototype for the vedas2 collection"
    )
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE_DIR)
    parser.add_argument(
        "--collection-id", type=uuid.UUID, default=DEFAULT_COLLECTION_ID
    )
    parser.add_argument("--domain", default="books")
    parser.add_argument("--concurrency", type=int, default=None)
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Clear this collection before the burner run.",
    )
    parser.add_argument(
        "--reset-only",
        action="store_true",
        help="Clear this collection and exit without ingesting.",
    )
    return parser.parse_args()


def deterministic_uuid(collection_id: uuid.UUID, key: str) -> uuid.UUID:
    return uuid.UUID(hashlib.md5(f"{collection_id}:{key}".encode()).hexdigest())


def vector_literal(embedding: list[float]) -> str:
    if not embedding or any(not math.isfinite(float(value)) for value in embedding):
        raise ValueError("Invalid embedding")
    return "[" + ",".join(str(float(value)) for value in embedding) + "]"


def canonical_name(value: str) -> str:
    return " ".join(value.strip().split())[:256]


def canonical_type(value: str) -> str:
    return normalize_rel_type(value or "CONCEPT")[:64]


async def load_runtime(
    collection_id: uuid.UUID,
    requested_concurrency: int | None,
) -> tuple[Collection, EmbeddingProvider, LLMProvider, int]:
    async with AsyncSessionLocal() as session:
        collection = await session.get(Collection, collection_id)
        if collection is None:
            raise ValueError(f"Collection {collection_id} not found")
        if collection.strategy != "custom_graph_rag":
            raise ValueError("Collection must use custom_graph_rag")
        if collection.embedding_profile_id != DEFAULT_EMBEDDING_PROFILE_ID:
            raise ValueError("Unexpected embedding profile")
        if collection.llm_profile_id != DEFAULT_LLM_PROFILE_ID:
            raise ValueError("Unexpected LLM profile")

        profiles = (
            (
                await session.execute(
                    select(Profile).where(
                        Profile.id.in_(
                            [DEFAULT_EMBEDDING_PROFILE_ID, DEFAULT_LLM_PROFILE_ID]
                        )
                    )
                )
            )
            .scalars()
            .all()
        )
        limits = [
            int(profile.max_concurrent_calls)
            for profile in profiles
            if profile.max_concurrent_calls and profile.max_concurrent_calls > 0
        ]
        profile_limit = min(limits) if limits else 1
        concurrency = min(requested_concurrency or profile_limit, profile_limit)
        if concurrency <= 0:
            raise ValueError("Concurrency must be positive")

    embedding_provider, llm_provider = await asyncio.gather(
        _resolve_embedding_provider(collection),
        resolve_llm_provider_from_collection(collection),
    )
    return collection, embedding_provider, llm_provider, concurrency


async def reset_collection(collection: Collection) -> None:
    async with AsyncSessionLocal() as session:
        await session.execute(
            delete(GraphVersion).where(GraphVersion.collection_id == collection.id)
        )
        await session.execute(
            delete(GraphChunkSegment).where(
                GraphChunkSegment.collection_id == collection.id
            )
        )
        await session.execute(
            delete(IngestionRecord).where(
                IngestionRecord.collection_id == collection.id
            )
        )
        await session.execute(
            delete(RawChunkExtraction).where(
                RawChunkExtraction.collection_id == collection.id
            )
        )
        await session.commit()
    await GraphService()._reset_collection_contents(collection)


def load_work(
    source_dir: Path,
    collection: Collection,
    domain: str,
) -> tuple[int, list[ChunkWork]]:
    if not source_dir.is_dir():
        raise ValueError(f"Source directory does not exist: {source_dir}")
    chunker = DocumentChunker(
        chunk_size_tokens=settings.chunk_size_tokens,
        chunk_overlap_tokens=settings.chunk_overlap_tokens,
    )
    document_count = 0
    work: list[ChunkWork] = []
    for path in sorted(source_dir.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in SUPPORTED_SUFFIXES:
            continue
        raw_text = path.read_text(encoding="utf-8", errors="replace")
        if not raw_text.strip():
            continue
        document_count += 1
        document_path = path.relative_to(source_dir).as_posix()
        document_id = document_id_for_path(collection.id, document_path)
        chunks = chunker.chunk_document(
            raw_text,
            domain=domain,
            document_path=document_path,
        )
        work.extend(
            ChunkWork(document_path, document_id, index, chunk)
            for index, chunk in enumerate(chunks)
        )
    if not work:
        raise ValueError(f"No ingestible text found under {source_dir}")
    return document_count, work


async def completed_chunk_exists(collection_id: uuid.UUID, chunk_hash: str) -> bool:
    async with AsyncSessionLocal() as session:
        return bool(
            await session.scalar(
                select(GraphChunkSegment.id)
                .where(
                    GraphChunkSegment.collection_id == collection_id,
                    GraphChunkSegment.chunk_hash == chunk_hash,
                    GraphChunkSegment.contract_version == EXTRACTION_CONTRACT,
                    GraphChunkSegment.status == "completed",
                    GraphChunkSegment.tombstoned_at.is_(None),
                )
                .limit(1)
            )
        )


async def load_raw_extraction(
    collection_id: uuid.UUID,
    chunk_hash: str,
) -> ExtractionResult | None:
    async with AsyncSessionLocal() as session:
        segment = await session.scalar(
            select(GraphChunkSegment)
            .where(
                GraphChunkSegment.collection_id == collection_id,
                GraphChunkSegment.chunk_hash == chunk_hash,
                GraphChunkSegment.contract_version == EXTRACTION_CONTRACT,
                GraphChunkSegment.tombstoned_at.is_(None),
            )
            .limit(1)
        )
        if segment is not None:
            payload = segment.raw_extraction or {}
            return extraction_from_payload(payload)
        record = await session.scalar(
            select(RawChunkExtraction)
            .where(
                RawChunkExtraction.collection_id == collection_id,
                RawChunkExtraction.chunk_content_hash == chunk_hash,
            )
            .limit(1)
        )
    if record is None:
        return None
    return extraction_from_payload(
        {
            "entities": record.entities_json or [],
            "relationships": record.relationships_json or [],
        }
    )


def extraction_from_payload(payload: dict[str, Any]) -> ExtractionResult:
    return ExtractionResult(
        entities=[
            ExtractedEntity(
                name=item["name"],
                entity_type=item.get("type", "CONCEPT"),
                description=item.get("description", ""),
            )
            for item in payload.get("entities", [])
        ],
        relationships=[
            ExtractedRelationship(
                source_name=item["source_name"],
                target_name=item["target_name"],
                description=item.get("description", ""),
                keywords=item.get("keywords", []),
                weight=float(item.get("weight", 1.0)),
                rel_type=item.get("rel_type", "RELATES_TO"),
                conditions=tuple(item.get("conditions", [])),
                exceptions=tuple(item.get("exceptions", [])),
                scopes=tuple(item.get("scopes", [])),
                polarity=item.get("polarity", "positive"),
                modality=item.get("modality", "asserted"),
            )
            for item in payload.get("relationships", [])
        ],
    )


def extraction_payload(extraction: ExtractionResult) -> dict[str, Any]:
    return {
        "entities": [
            {
                "name": item.name,
                "type": item.entity_type,
                "description": item.description,
            }
            for item in extraction.entities
        ],
        "relationships": [
            {
                "source_name": item.source_name,
                "target_name": item.target_name,
                "description": item.description,
                "keywords": item.keywords,
                "weight": item.weight,
                "rel_type": item.rel_type,
                "conditions": list(item.conditions),
                "exceptions": list(item.exceptions),
                "scopes": list(item.scopes),
                "polarity": item.polarity,
                "modality": item.modality,
            }
            for item in extraction.relationships
        ],
    }


async def save_extracted_segment(
    collection: Collection,
    work: ChunkWork,
    chunk_hash: str,
    extraction: ExtractionResult,
) -> None:
    segment_id = deterministic_uuid(
        collection.id, f"segment:{EXTRACTION_CONTRACT}:{chunk_hash}"
    )
    async with AsyncSessionLocal() as session:
        await session.execute(
            pg_insert(GraphChunkSegment)
            .values(
                id=segment_id,
                collection_id=collection.id,
                document_id=work.document_id,
                document_path=work.document_path,
                chunk_hash=chunk_hash,
                chunk_index=work.chunk_index,
                contract_version=EXTRACTION_CONTRACT,
                status="extracted",
                raw_extraction=extraction_payload(extraction),
            )
            .on_conflict_do_nothing(constraint="uq_graph_chunk_segment_contract")
        )
        await session.commit()


def compile_rows(
    collection: Collection,
    extraction: ExtractionResult,
    chunk_hash: str,
) -> tuple[list[EntityRow], list[RelationshipRow]]:
    extracted_by_name = {
        canonical_name(entity.name).casefold(): entity
        for entity in extraction.entities
        if canonical_name(entity.name)
    }
    for relationship in extraction.relationships:
        for name in (relationship.source_name, relationship.target_name):
            normalized = canonical_name(name)
            if normalized and normalized.casefold() not in extracted_by_name:
                extracted_by_name[normalized.casefold()] = ExtractedEntity(
                    name=normalized,
                    entity_type="CONCEPT",
                    description="",
                )

    entities: list[EntityRow] = []
    entity_by_name: dict[str, EntityRow] = {}
    entity_by_id: dict[uuid.UUID, EntityRow] = {}
    for key, extracted in sorted(extracted_by_name.items()):
        name = canonical_name(extracted.name)
        entity_id = deterministic_uuid(collection.id, f"entity:{key}")
        row = EntityRow(
            id=entity_id,
            name=name,
            primary_type=canonical_type(extracted.entity_type),
            description=extracted.description.strip() or f"Concept: {name}.",
            description_id=deterministic_uuid(
                collection.id, f"entity-description:{entity_id}:{chunk_hash}"
            ),
        )
        entities.append(row)
        entity_by_name[key] = row
        entity_by_id[row.id] = row

    def add_reasoning_entity(
        stable_key: str,
        name: str,
        primary_type: str,
        description: str,
    ) -> EntityRow:
        entity_id = deterministic_uuid(collection.id, stable_key)
        existing = entity_by_id.get(entity_id)
        if existing is not None:
            return existing
        row = EntityRow(
            id=entity_id,
            name=name[:256],
            primary_type=primary_type,
            description=description,
            description_id=deterministic_uuid(
                collection.id, f"entity-description:{entity_id}:{chunk_hash}"
            ),
        )
        entities.append(row)
        entity_by_id[row.id] = row
        return row

    relationships_by_key: dict[tuple[uuid.UUID, str, uuid.UUID], RelationshipRow] = {}

    def add_relationship(
        source: EntityRow,
        target: EntityRow,
        rel_type: str,
        description: str,
        keywords: tuple[str, ...] = (),
        weight: int = 10,
    ) -> None:
        normalized_type = normalize_rel_type(rel_type)
        key = (source.id, normalized_type, target.id)
        relationship_id = deterministic_uuid(
            collection.id,
            f"relationship:{source.id}:{normalized_type}:{target.id}",
        )
        candidate = RelationshipRow(
            id=relationship_id,
            source_id=source.id,
            source_name=source.name,
            target_id=target.id,
            target_name=target.name,
            rel_type_id=deterministic_uuid(
                collection.id, f"rel-type:{normalized_type}"
            ),
            rel_type=normalized_type,
            description=description,
            description_id=deterministic_uuid(
                collection.id,
                f"relationship-description:{relationship_id}:{chunk_hash}",
            ),
            keywords=keywords,
            weight=weight,
        )
        current = relationships_by_key.get(key)
        if current is None or len(candidate.description) > len(current.description):
            relationships_by_key[key] = candidate

    evidence = add_reasoning_entity(
        f"evidence-chunk:{chunk_hash}",
        f"evidence:{chunk_hash[:16]}",
        "EVIDENCE_CHUNK",
        f"Immutable evidence segment {chunk_hash}.",
    )
    for extracted in extraction.relationships:
        source = entity_by_name.get(canonical_name(extracted.source_name).casefold())
        target = entity_by_name.get(canonical_name(extracted.target_name).casefold())
        if source is None or target is None or source.id == target.id:
            continue
        rel_type = normalize_rel_type(extracted.rel_type or "RELATES_TO")
        description = extracted.description.strip() or (
            f"{source.name} {rel_type} {target.name}."
        )
        keywords = tuple(dict.fromkeys(extracted.keywords))
        weight = max(1, int(float(extracted.weight or 1.0) * 10))
        add_relationship(source, target, rel_type, description, keywords, weight)

        proposition_fingerprint = hashlib.sha256(
            json.dumps(
                {
                    "source": str(source.id),
                    "predicate": rel_type,
                    "target": str(target.id),
                    "conditions": sorted(extracted.conditions),
                    "exceptions": sorted(extracted.exceptions),
                    "scopes": sorted(extracted.scopes),
                    "polarity": extracted.polarity,
                    "modality": extracted.modality,
                },
                sort_keys=True,
            ).encode()
        ).hexdigest()
        proposition = add_reasoning_entity(
            f"proposition:{proposition_fingerprint}",
            f"proposition:{proposition_fingerprint[:16]}",
            "PROPOSITION",
            (
                f"{description} Polarity={extracted.polarity}; "
                f"modality={extracted.modality}."
            ),
        )
        add_relationship(proposition, source, "SUBJECT", description)
        add_relationship(proposition, target, "OBJECT", description)
        add_relationship(evidence, proposition, "SUPPORTS", description)

        conditions: list[EntityRow] = []
        for condition_text in extracted.conditions:
            digest = hashlib.sha256(condition_text.casefold().encode()).hexdigest()
            condition = add_reasoning_entity(
                f"condition:{digest}",
                f"condition:{digest[:12]}:{condition_text}"[:256],
                "CONDITION",
                condition_text,
            )
            conditions.append(condition)
            add_relationship(proposition, condition, "CONDITION", condition_text)
        for scope_text in extracted.scopes:
            digest = hashlib.sha256(scope_text.casefold().encode()).hexdigest()
            scope = add_reasoning_entity(
                f"scope:{digest}",
                f"scope:{digest[:12]}:{scope_text}"[:256],
                "SCOPE",
                scope_text,
            )
            add_relationship(proposition, scope, "APPLIES_TO", scope_text)

        exceptions: list[EntityRow] = []
        for exception_text in extracted.exceptions:
            digest = hashlib.sha256(exception_text.casefold().encode()).hexdigest()
            exception = add_reasoning_entity(
                f"exception:{digest}",
                f"exception:{digest[:12]}:{exception_text}"[:256],
                "EXCEPTION",
                exception_text,
            )
            exceptions.append(exception)
            add_relationship(proposition, exception, "EXCEPTION", exception_text)

        if conditions or exceptions:
            rule_fingerprint = hashlib.sha256(
                (
                    proposition_fingerprint
                    + "|"
                    + "|".join(sorted(str(item.id) for item in conditions))
                    + "|"
                    + "|".join(sorted(str(item.id) for item in exceptions))
                ).encode()
            ).hexdigest()
            rule = add_reasoning_entity(
                f"rule:{rule_fingerprint}",
                f"rule:{rule_fingerprint[:16]}",
                "RULE",
                f"Conditional rule concluding {description}",
            )
            for condition in conditions:
                add_relationship(
                    condition,
                    rule,
                    "ANTECEDENT_OF",
                    f"{condition.description} activates {rule.name}.",
                )
            for exception in exceptions:
                add_relationship(
                    exception,
                    rule,
                    "BLOCKS",
                    f"{exception.description} blocks {rule.name}.",
                )
            add_relationship(rule, proposition, "CONCLUDES", description)
    return entities, list(relationships_by_key.values())


async def embed_chunk_objects(
    provider: EmbeddingProvider,
    chunk_text: str,
    entities: list[EntityRow],
    relationships: list[RelationshipRow],
) -> tuple[list[float], list[list[float]], list[list[float]]]:
    texts = [chunk_text]
    texts.extend(f"{row.name}: {row.description}" for row in entities)
    texts.extend(
        relationship_embedding_text(
            row.source_name,
            row.target_name,
            row.rel_type,
            row.description,
            list(row.keywords),
        )
        for row in relationships
    )
    embeddings = await provider.embed_documents(texts)
    entity_end = 1 + len(entities)
    return embeddings[0], embeddings[1:entity_end], embeddings[entity_end:]


async def persist_chunk(
    collection: Collection,
    work: ChunkWork,
    chunk_hash: str,
    extraction: ExtractionResult,
    entities: list[EntityRow],
    relationships: list[RelationshipRow],
    chunk_embedding: list[float],
    entity_embeddings: list[list[float]],
    relationship_embeddings: list[list[float]],
) -> None:
    dimensions = await get_collection_dimensions(collection.id)
    if dimensions is None:
        raise ValueError("Collection vector dimensions are not initialized")
    cast = f"::vector({dimensions})"
    chunk_table = table_name(collection.id, "vector_chunks")
    entity_table = table_name(collection.id, "entity_embeddings")
    centroid_table = table_name(collection.id, "entity_centroids")
    relationship_table = table_name(collection.id, "relationship_embeddings")

    async with AsyncSessionLocal() as session:
        segment_id = deterministic_uuid(
            collection.id, f"segment:{EXTRACTION_CONTRACT}:{chunk_hash}"
        )
        await session.execute(
            pg_insert(RawChunkExtraction)
            .values(
                chunk_content_hash=chunk_hash,
                collection_id=collection.id,
                document_id=work.document_id,
                document_path=work.document_path,
                entities_json=[
                    {
                        "name": item.name,
                        "type": item.entity_type,
                        "description": item.description,
                    }
                    for item in extraction.entities
                ],
                relationships_json=[
                    {
                        "source_name": item.source_name,
                        "target_name": item.target_name,
                        "description": item.description,
                        "keywords": item.keywords,
                        "weight": item.weight,
                        "rel_type": item.rel_type,
                        "conditions": list(item.conditions),
                        "exceptions": list(item.exceptions),
                        "scopes": list(item.scopes),
                        "polarity": item.polarity,
                        "modality": item.modality,
                    }
                    for item in extraction.relationships
                ],
                extraction_model=EXTRACTION_CONTRACT,
                gleaning_passes=0,
            )
            .on_conflict_do_nothing(
                constraint="uq_raw_chunk_extractions_hash_collection"
            )
        )
        await session.execute(
            pg_insert(GraphChunkSegment)
            .values(
                id=segment_id,
                collection_id=collection.id,
                document_id=work.document_id,
                document_path=work.document_path,
                chunk_hash=chunk_hash,
                chunk_index=work.chunk_index,
                contract_version=EXTRACTION_CONTRACT,
                status="materializing",
                raw_extraction=extraction_payload(extraction),
            )
            .on_conflict_do_update(
                constraint="uq_graph_chunk_segment_contract",
                set_={
                    "status": "materializing",
                    "tombstoned_at": None,
                },
            )
        )

        if entities:
            await session.execute(
                pg_insert(GraphEntity)
                .values(
                    [
                        {
                            "id": row.id,
                            "collection_id": collection.id,
                            "canonical_name": row.name,
                            "primary_type": row.primary_type,
                            "description_count": 1,
                        }
                        for row in entities
                    ]
                )
                .on_conflict_do_nothing(index_elements=[GraphEntity.id])
            )
            await session.execute(
                pg_insert(EntityDescription)
                .values(
                    [
                        {
                            "id": row.description_id,
                            "entity_id": row.id,
                            "description": row.description,
                            "weight": 1,
                            "source_chunk_hashes": [chunk_hash],
                            "document_id": work.document_id,
                            "document_path": work.document_path,
                        }
                        for row in entities
                    ]
                )
                .on_conflict_do_nothing(index_elements=[EntityDescription.id])
            )

        rel_types = {row.rel_type: row.rel_type_id for row in relationships}
        canonical_rel_type_ids: dict[str, uuid.UUID] = {}
        if rel_types:
            await session.execute(
                pg_insert(GraphRelationshipType)
                .values(
                    [
                        {
                            "id": rel_type_id,
                            "collection_id": collection.id,
                            "canonical_type": rel_type,
                        }
                        for rel_type, rel_type_id in rel_types.items()
                    ]
                )
                .on_conflict_do_nothing(
                    constraint="uq_graph_relationship_types_collection_canonical_type"
                )
            )
            persisted_rel_types = await session.execute(
                select(GraphRelationshipType).where(
                    GraphRelationshipType.collection_id == collection.id,
                    GraphRelationshipType.canonical_type.in_(list(rel_types)),
                )
            )
            canonical_rel_type_ids = {
                row.canonical_type: row.id
                for row in persisted_rel_types.scalars().all()
            }
            missing_rel_types = set(rel_types) - set(canonical_rel_type_ids)
            if missing_rel_types:
                raise RuntimeError(
                    "Canonical predicate rows missing after upsert: "
                    + ", ".join(sorted(missing_rel_types))
                )

        if entities:
            await session.execute(
                pg_insert(GraphEntityMapping)
                .values(
                    [
                        {
                            "id": deterministic_uuid(
                                collection.id,
                                f"entity-map:{segment_id}:{row.name.casefold()}",
                            ),
                            "collection_id": collection.id,
                            "segment_id": segment_id,
                            "local_key": row.name.casefold(),
                            "raw_name": row.name,
                            "raw_type": row.primary_type,
                            "canonical_entity_id": row.id,
                            "resolution_method": "deterministic_exact",
                            "confidence": 1.0,
                            "mapping_version": 1,
                        }
                        for row in entities
                    ]
                )
                .on_conflict_do_nothing(constraint="uq_graph_entity_mapping_local")
            )
        if rel_types:
            await session.execute(
                pg_insert(GraphPredicateMapping)
                .values(
                    [
                        {
                            "id": deterministic_uuid(
                                collection.id,
                                f"predicate-map:{segment_id}:{rel_type}",
                            ),
                            "collection_id": collection.id,
                            "segment_id": segment_id,
                            "raw_predicate": rel_type,
                            "canonical_predicate_id": canonical_rel_type_ids[rel_type],
                            "resolution_method": "normalized_exact",
                            "confidence": 1.0,
                            "mapping_version": 1,
                            "inferred_properties": {"directed": True},
                        }
                        for rel_type in rel_types
                    ]
                )
                .on_conflict_do_nothing(constraint="uq_graph_predicate_mapping_local")
            )

        contribution_rows = [
            {
                "id": deterministic_uuid(
                    collection.id, f"contribution:{segment_id}:entity:{row.id}"
                ),
                "collection_id": collection.id,
                "segment_id": segment_id,
                "object_kind": "entity",
                "object_id": row.id,
                "contribution_kind": "description",
                "weight": 1.0,
                "metadata_json": {"description_id": str(row.description_id)},
            }
            for row in entities
        ]
        contribution_rows.extend(
            {
                "id": deterministic_uuid(
                    collection.id,
                    f"contribution:{segment_id}:relationship:{row.id}",
                ),
                "collection_id": collection.id,
                "segment_id": segment_id,
                "object_kind": "relationship",
                "object_id": row.id,
                "contribution_kind": "evidence",
                "weight": 1.0,
                "metadata_json": {"description_id": str(row.description_id)},
            }
            for row in relationships
        )
        if contribution_rows:
            await session.execute(
                pg_insert(GraphChunkContribution)
                .values(contribution_rows)
                .on_conflict_do_nothing(constraint="uq_graph_chunk_contribution_object")
            )
            dependency_rows = [
                {
                    "id": deterministic_uuid(
                        collection.id,
                        f"dependency:{segment_id}:{row['object_kind']}:{row['object_id']}",
                    ),
                    "collection_id": collection.id,
                    "source_kind": "segment",
                    "source_id": segment_id,
                    "target_kind": row["object_kind"],
                    "target_id": row["object_id"],
                    "dependency_type": "contributes_to",
                    "overlay_version": 1,
                }
                for row in contribution_rows
            ]
            await session.execute(
                pg_insert(GraphDerivedDependency)
                .values(dependency_rows)
                .on_conflict_do_nothing(constraint="uq_graph_derived_dependency")
            )
            await session.execute(
                pg_insert(GraphRelationship)
                .values(
                    [
                        {
                            "id": row.id,
                            "source_entity_id": row.source_id,
                            "target_entity_id": row.target_id,
                            "weight": row.weight,
                            "keywords": list(row.keywords),
                            "relationship_type_id": canonical_rel_type_ids[
                                row.rel_type
                            ],
                            "rel_type": row.rel_type,
                            "collection_id": collection.id,
                        }
                        for row in relationships
                    ]
                )
                .on_conflict_do_nothing(index_elements=[GraphRelationship.id])
            )
            await session.execute(
                pg_insert(RelationshipDescription)
                .values(
                    [
                        {
                            "id": row.description_id,
                            "relationship_id": row.id,
                            "description": row.description,
                            "keywords": list(row.keywords),
                            "weight": 1,
                            "source_chunk_hashes": [chunk_hash],
                            "document_id": work.document_id,
                            "document_path": work.document_path,
                        }
                        for row in relationships
                    ]
                )
                .on_conflict_do_nothing(index_elements=[RelationshipDescription.id])
            )

        await session.execute(
            text(
                f"INSERT INTO {chunk_table} "
                "(namespace_id, collection_id, chunk_hash, chunk_index, content, "
                "token_count, metadata_json, embedding) VALUES "
                f"(:nsid, :cid, :hash, :idx, :content, :tokens, :meta, (:emb){cast})"
            ),
            {
                "nsid": _uuid_for_sql(collection.namespace_id),
                "cid": _uuid_for_sql(collection.id),
                "hash": chunk_hash,
                "idx": work.chunk_index,
                "content": work.text,
                "tokens": len(work.text.split()),
                "meta": json.dumps(
                    {
                        "document_id": str(work.document_id),
                        "document_path": work.document_path,
                        "ingestion": "incremental_burner_v1",
                    }
                ),
                "emb": vector_literal(chunk_embedding),
            },
        )

        if entities:
            await session.execute(
                text(
                    f"INSERT INTO {entity_table} "
                    "(entity_id, collection_id, document_id, document_path, name, "
                    "description, description_id, embedding) VALUES "
                    "(:eid, :cid, :did, :path, :name, :description, "
                    f":desc_id, (:emb){cast})"
                ),
                [
                    {
                        "eid": _uuid_for_sql(row.id),
                        "cid": _uuid_for_sql(collection.id),
                        "did": _uuid_for_sql(work.document_id),
                        "path": work.document_path,
                        "name": row.name,
                        "description": row.description,
                        "desc_id": _uuid_for_sql(row.description_id),
                        "emb": vector_literal(embedding),
                    }
                    for row, embedding in zip(entities, entity_embeddings, strict=True)
                ],
            )
            await session.execute(
                text(
                    f"INSERT INTO {centroid_table} "
                    "(entity_id, collection_id, canonical_name, primary_type, "
                    "description_count, embedding) VALUES "
                    f"(:eid, :cid, :name, :type, 1, (:emb){cast}) "
                    "ON CONFLICT (entity_id) DO NOTHING"
                ),
                [
                    {
                        "eid": _uuid_for_sql(row.id),
                        "cid": _uuid_for_sql(collection.id),
                        "name": row.name,
                        "type": row.primary_type,
                        "emb": vector_literal(embedding),
                    }
                    for row, embedding in zip(entities, entity_embeddings, strict=True)
                ],
            )

        if relationships:
            await session.execute(
                text(
                    f"INSERT INTO {relationship_table} "
                    "(relationship_id, collection_id, document_id, document_path, "
                    "source_name, target_name, description, embedding) VALUES "
                    "(:rid, :cid, :did, :path, :source, :target, :description, "
                    f"(:emb){cast})"
                ),
                [
                    {
                        "rid": _uuid_for_sql(row.id),
                        "cid": _uuid_for_sql(collection.id),
                        "did": _uuid_for_sql(work.document_id),
                        "path": work.document_path,
                        "source": row.source_name,
                        "target": row.target_name,
                        "description": row.description,
                        "emb": vector_literal(embedding),
                    }
                    for row, embedding in zip(
                        relationships, relationship_embeddings, strict=True
                    )
                ],
            )

        session.add(
            IngestionRecord(
                collection_id=collection.id,
                chunk_hash=chunk_hash,
                document_id=work.document_id,
                document_path=work.document_path,
                strategy="custom_graph_rag",
                extraction_model=EXTRACTION_CONTRACT,
                embedding_model="profile-bound",
                entity_count=len(entities),
                relationship_count=len(relationships),
                sanitization_flags=None,
            )
        )
        await session.execute(
            text("SELECT pg_advisory_xact_lock(hashtext(:collection_id))"),
            {"collection_id": str(collection.id)},
        )
        latest_version = await session.scalar(
            select(GraphVersion)
            .where(GraphVersion.collection_id == collection.id)
            .order_by(GraphVersion.version.desc())
            .limit(1)
        )
        next_version = (latest_version.version if latest_version else 0) + 1
        session.add(
            GraphVersion(
                id=deterministic_uuid(collection.id, f"version:{segment_id}"),
                collection_id=collection.id,
                version=next_version,
                parent_version_id=latest_version.id if latest_version else None,
                delta_segment_id=segment_id,
                status="ready",
                manifest={"segment_id": str(segment_id)},
                published_at=func.now(),
            )
        )
        await session.execute(
            text(
                "UPDATE graph_chunk_segments SET status = 'completed', "
                "completed_at = now() WHERE id = :segment_id"
            ),
            {"segment_id": _uuid_for_sql(segment_id)},
        )
        await session.commit()


async def ingest_one(
    collection: Collection,
    embedding_provider: EmbeddingProvider,
    extractor: LLMGraphExtractor,
    graph_storage: Any,
    work: ChunkWork,
    domain: str,
) -> ChunkResult:
    sanitized, _ = _sanitizer.sanitize(work.text, str(collection.namespace_id))
    chunk_hash = _sanitizer.chunk_hash(sanitized)
    if await completed_chunk_exists(collection.id, chunk_hash):
        return ChunkResult(status="skipped")

    extraction = await load_raw_extraction(collection.id, chunk_hash)
    status = "resumed" if extraction is not None else "ingested"
    if extraction is None:
        extraction = await extractor.extract(sanitized, domain=domain)
        await save_extracted_segment(collection, work, chunk_hash, extraction)

    entities, relationships = compile_rows(collection, extraction, chunk_hash)
    (
        chunk_embedding,
        entity_embeddings,
        relationship_embeddings,
    ) = await embed_chunk_objects(
        embedding_provider,
        sanitized,
        entities,
        relationships,
    )
    await persist_chunk(
        collection,
        work,
        chunk_hash,
        extraction,
        entities,
        relationships,
        chunk_embedding,
        entity_embeddings,
        relationship_embeddings,
    )
    await graph_storage.upsert_nodes(
        [
            {
                "id": str(row.id),
                "name": row.name,
                "collection_id": str(collection.id),
                "document_id": str(work.document_id),
                "document_path": work.document_path,
            }
            for row in entities
        ]
    )
    await graph_storage.upsert_edges(
        [
            {
                "id": str(row.id),
                "source_id": str(row.source_id),
                "target_id": str(row.target_id),
                "weight": row.weight,
                "keywords": list(row.keywords),
                "rel_type": row.rel_type,
                "collection_id": str(collection.id),
                "document_id": str(work.document_id),
                "document_path": work.document_path,
            }
            for row in relationships
        ],
        merge_existing_keywords=False,
    )
    return ChunkResult(status, len(entities), len(relationships))


async def main() -> None:
    args = parse_args()
    collection, embedding_provider, llm_provider, concurrency = await load_runtime(
        args.collection_id,
        args.concurrency,
    )
    if args.reset or args.reset_only:
        await reset_collection(collection)
        print(f"reset collection={collection.id}", flush=True)
    if args.reset_only:
        return
    source_dir = args.source_dir.expanduser().resolve()
    document_count, work = load_work(source_dir, collection, args.domain)
    extractor = LLMGraphExtractor(llm=llm_provider)
    graph_storage = get_graph_storage(collection)
    print(
        f"collection={collection.name} ({collection.id}) documents={document_count} "
        f"chunks={len(work)} concurrency={concurrency} gleaning=0",
        flush=True,
    )

    queue: asyncio.Queue[ChunkWork | None] = asyncio.Queue()
    for item in work:
        queue.put_nowait(item)
    for _ in range(concurrency):
        queue.put_nowait(None)

    progress_lock = asyncio.Lock()
    totals = {"completed": 0, "ingested": 0, "resumed": 0, "skipped": 0}

    async def worker() -> None:
        while True:
            item = await queue.get()
            try:
                if item is None:
                    return
                result = await ingest_one(
                    collection,
                    embedding_provider,
                    extractor,
                    graph_storage,
                    item,
                    args.domain,
                )
                async with progress_lock:
                    totals["completed"] += 1
                    totals[result.status] += 1
                    print(
                        f"[{totals['completed']}/{len(work)}] "
                        f"{item.document_path}#{item.chunk_index} {result.status} "
                        f"entities={result.entity_count} "
                        f"relationships={result.relationship_count}",
                        flush=True,
                    )
            finally:
                queue.task_done()

    workers = [asyncio.create_task(worker()) for _ in range(concurrency)]
    try:
        await asyncio.gather(*workers)
    except BaseException:
        for task in workers:
            task.cancel()
        await asyncio.gather(*workers, return_exceptions=True)
        raise

    print(f"complete {totals}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
