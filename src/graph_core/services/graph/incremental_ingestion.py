"""Incremental manifests and graph-version publication for production ingestion."""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import select, text
from sqlalchemy.dialects.postgresql import insert as pg_insert

from graph_core.database import AsyncSessionLocal
from graph_core.models.collection import Collection
from graph_core.models.incremental_graph import (
    GraphChunkContribution,
    GraphChunkSegment,
    GraphDerivedDependency,
    GraphEntityMapping,
    GraphPredicateMapping,
    GraphVersion,
)


@dataclass(frozen=True, slots=True)
class EntityMappingInput:
    local_key: str
    raw_name: str
    raw_type: str
    canonical_entity_id: uuid.UUID
    resolution_method: str = "context_resolver"
    confidence: float = 1.0


@dataclass(frozen=True, slots=True)
class PredicateMappingInput:
    raw_predicate: str
    canonical_predicate_id: uuid.UUID
    resolution_method: str = "normalized"
    confidence: float = 1.0


@dataclass(frozen=True, slots=True)
class ContributionInput:
    object_kind: str
    object_id: uuid.UUID
    contribution_kind: str = "created_or_reused"
    weight: float = 1.0
    metadata: dict[str, Any] | None = None


def ingestion_contract(domain: str | None, extraction_version: str) -> str:
    return f"{extraction_version}:domain:{domain or 'general'}:compiled:v1"


async def completed_segment(
    collection_id: uuid.UUID,
    chunk_hash: str,
    contract_version: str,
) -> GraphChunkSegment | None:
    async with AsyncSessionLocal() as session:
        return (
            await session.execute(
                select(GraphChunkSegment).where(
                    GraphChunkSegment.collection_id == collection_id,
                    GraphChunkSegment.chunk_hash == chunk_hash,
                    GraphChunkSegment.contract_version == contract_version,
                    GraphChunkSegment.status == "completed",
                    GraphChunkSegment.tombstoned_at.is_(None),
                )
            )
        ).scalar_one_or_none()


async def publish_chunk_delta(
    *,
    collection: Collection,
    chunk_hash: str,
    chunk_index: int,
    contract_version: str,
    raw_extraction: dict[str, Any],
    document_id: uuid.UUID | None,
    document_path: str | None,
    entity_mappings: list[EntityMappingInput],
    predicate_mappings: list[PredicateMappingInput],
    contributions: list[ContributionInput],
) -> tuple[GraphChunkSegment, GraphVersion]:
    """Publish one chunk as a collection-scoped delta under an advisory lock."""
    segment_id = uuid.uuid5(
        collection.id,
        f"segment:{chunk_hash}:{contract_version}",
    )
    async with AsyncSessionLocal() as session:
        await session.execute(
            text("SELECT pg_advisory_xact_lock(hashtext(:lock_key))"),
            {"lock_key": f"graph-version:{collection.id}"},
        )
        segment_insert = pg_insert(GraphChunkSegment).values(
            id=segment_id,
            collection_id=collection.id,
            document_id=document_id,
            document_path=document_path,
            chunk_hash=chunk_hash,
            chunk_index=chunk_index,
            contract_version=contract_version,
            status="completed",
            raw_extraction=raw_extraction,
            completed_at=datetime.now(UTC),
        )
        await session.execute(
            segment_insert.on_conflict_do_update(
                constraint="uq_graph_chunk_segment_contract",
                set_={
                    "status": "completed",
                    "raw_extraction": segment_insert.excluded.raw_extraction,
                    "document_id": segment_insert.excluded.document_id,
                    "document_path": segment_insert.excluded.document_path,
                    "completed_at": segment_insert.excluded.completed_at,
                    "tombstoned_at": None,
                },
            )
        )
        if entity_mappings:
            entity_mappings = list(
                {item.local_key: item for item in entity_mappings}.values()
            )
            await session.execute(
                pg_insert(GraphEntityMapping)
                .values(
                    [
                        {
                            "id": uuid.uuid5(
                                segment_id,
                                f"entity-mapping:{item.local_key}",
                            ),
                            "collection_id": collection.id,
                            "segment_id": segment_id,
                            "local_key": item.local_key,
                            "raw_name": item.raw_name,
                            "raw_type": item.raw_type,
                            "canonical_entity_id": item.canonical_entity_id,
                            "resolution_method": item.resolution_method,
                            "confidence": item.confidence,
                        }
                        for item in entity_mappings
                    ]
                )
                .on_conflict_do_nothing(
                    constraint="uq_graph_entity_mapping_local"
                )
            )
        if predicate_mappings:
            predicate_mappings = list(
                {item.raw_predicate: item for item in predicate_mappings}.values()
            )
            await session.execute(
                pg_insert(GraphPredicateMapping)
                .values(
                    [
                        {
                            "id": uuid.uuid5(
                                segment_id,
                                f"predicate-mapping:{item.raw_predicate}",
                            ),
                            "collection_id": collection.id,
                            "segment_id": segment_id,
                            "raw_predicate": item.raw_predicate,
                            "canonical_predicate_id": item.canonical_predicate_id,
                            "resolution_method": item.resolution_method,
                            "confidence": item.confidence,
                        }
                        for item in predicate_mappings
                    ]
                )
                .on_conflict_do_nothing(
                    constraint="uq_graph_predicate_mapping_local"
                )
            )
        if contributions:
            contribution_values = [
                {
                    "id": uuid.uuid5(
                        segment_id,
                        f"contribution:{item.object_kind}:{item.object_id}",
                    ),
                    "collection_id": collection.id,
                    "segment_id": segment_id,
                    "object_kind": item.object_kind,
                    "object_id": item.object_id,
                    "contribution_kind": item.contribution_kind,
                    "weight": item.weight,
                    "metadata_json": item.metadata,
                }
                for item in contributions
            ]
            await session.execute(
                pg_insert(GraphChunkContribution)
                .values(contribution_values)
                .on_conflict_do_nothing(
                    constraint="uq_graph_chunk_contribution_object"
                )
            )
            await session.execute(
                pg_insert(GraphDerivedDependency)
                .values(
                    [
                        {
                            "id": uuid.uuid5(
                                segment_id,
                                f"dependency:{item.object_kind}:{item.object_id}",
                            ),
                            "collection_id": collection.id,
                            "source_kind": "segment",
                            "source_id": segment_id,
                            "target_kind": item.object_kind,
                            "target_id": item.object_id,
                            "dependency_type": "materializes",
                            "overlay_version": 1,
                        }
                        for item in contributions
                    ]
                )
                .on_conflict_do_nothing(
                    constraint="uq_graph_derived_dependency"
                )
            )
        latest = (
            await session.execute(
                select(GraphVersion)
                .where(GraphVersion.collection_id == collection.id)
                .order_by(GraphVersion.version.desc())
                .limit(1)
            )
        ).scalar_one_or_none()
        existing_version = (
            await session.execute(
                select(GraphVersion).where(
                    GraphVersion.collection_id == collection.id,
                    GraphVersion.delta_segment_id == segment_id,
                )
            )
        ).scalar_one_or_none()
        if existing_version is None:
            next_version = int(latest.version if latest else 0) + 1
            graph_version = GraphVersion(
                id=uuid.uuid5(segment_id, "graph-version"),
                collection_id=collection.id,
                version=next_version,
                parent_version_id=latest.id if latest else None,
                delta_segment_id=segment_id,
                status="ready",
                manifest={
                    "chunk_hash": chunk_hash,
                    "contract_version": contract_version,
                    "contribution_count": len(contributions),
                },
                published_at=datetime.now(UTC),
            )
            session.add(graph_version)
        else:
            graph_version = existing_version
        await session.commit()
        segment = await session.get(GraphChunkSegment, segment_id)
        if segment is None:
            raise RuntimeError("Published graph segment disappeared")
        return segment, graph_version
