"""Multi-resolution semantic frames for graph ingestion and retrieval."""

from __future__ import annotations

import hashlib
import math
import uuid
from dataclasses import dataclass, field
from typing import Any

from sqlalchemy import bindparam, delete, select, text
from sqlalchemy.dialects.postgresql import insert as pg_insert

from graph_core.database import AsyncSessionLocal, _uuid_for_sql
from graph_core.embedding.interface import EmbeddingProvider
from graph_core.models.collection import Collection
from graph_core.models.incremental_graph import GraphFrameArgument, GraphSemanticFrame
from graph_core.storage.vector_tables import (
    ensure_semantic_frame_table,
    get_collection_dimensions,
    table_name,
)


@dataclass(frozen=True, slots=True)
class FrameArgumentInput:
    role: str
    entity_id: uuid.UUID
    position: int
    argument_type: str = "entity"


@dataclass(frozen=True, slots=True)
class SemanticFrameInput:
    id: uuid.UUID
    collection_id: uuid.UUID
    frame_kind: str
    title: str
    frame_text: str
    content_hash: str
    predicate: str | None = None
    proposition_entity_id: uuid.UUID | None = None
    source_relationship_id: uuid.UUID | None = None
    polarity: str = "positive"
    modality: str = "asserted"
    conditions: tuple[str, ...] = ()
    exceptions: tuple[str, ...] = ()
    scopes: tuple[str, ...] = ()
    executable_status: str = "grounded_binary"
    metadata: dict[str, Any] = field(default_factory=dict)
    arguments: tuple[FrameArgumentInput, ...] = ()


def deterministic_frame_id(collection_id: uuid.UUID, key: str) -> uuid.UUID:
    return uuid.UUID(hashlib.md5(f"{collection_id}:{key}".encode()).hexdigest())


def build_proposition_frame(
    *,
    collection_id: uuid.UUID,
    chunk_hash: str,
    relationship_index: int,
    proposition_entity_id: uuid.UUID,
    source_relationship_id: uuid.UUID,
    source_entity_id: uuid.UUID,
    source_name: str,
    target_entity_id: uuid.UUID,
    target_name: str,
    predicate: str,
    description: str,
    polarity: str,
    modality: str,
    conditions: tuple[str, ...],
    exceptions: tuple[str, ...],
    scopes: tuple[str, ...],
    metadata: dict[str, Any] | None = None,
) -> SemanticFrameInput:
    title = f"{source_name} {predicate.replace('_', ' ').lower()} {target_name}"
    parts = [title + ".", description.strip()]
    if conditions:
        parts.append("Conditions: " + "; ".join(conditions) + ".")
    if exceptions:
        parts.append("Exceptions: " + "; ".join(exceptions) + ".")
    if scopes:
        parts.append("Scope: " + "; ".join(scopes) + ".")
    parts.append(f"Polarity: {polarity}. Modality: {modality}.")
    frame_text = " ".join(part for part in parts if part)
    frame_id = deterministic_frame_id(
        collection_id,
        f"semantic-frame:{chunk_hash}:{relationship_index}",
    )
    return SemanticFrameInput(
        id=frame_id,
        collection_id=collection_id,
        frame_kind="proposition",
        predicate=predicate,
        title=title,
        frame_text=frame_text,
        content_hash=hashlib.sha256(frame_text.encode()).hexdigest(),
        proposition_entity_id=proposition_entity_id,
        source_relationship_id=source_relationship_id,
        polarity=polarity or "positive",
        modality=modality or "asserted",
        conditions=conditions,
        exceptions=exceptions,
        scopes=scopes,
        metadata=metadata or {},
        arguments=(
            FrameArgumentInput("subject", source_entity_id, 0),
            FrameArgumentInput("object", target_entity_id, 1),
        ),
    )


def _vector_literal(embedding: list[float]) -> str:
    if not embedding or any(not math.isfinite(float(value)) for value in embedding):
        raise ValueError("Invalid semantic frame embedding")
    return "[" + ",".join(str(float(value)) for value in embedding) + "]"


async def persist_semantic_frames(
    collection: Collection,
    provider: EmbeddingProvider,
    frames: list[SemanticFrameInput],
    *,
    batch_size: int = 64,
) -> tuple[int, int]:
    """Upsert frame structure and embed only frames not already materialized."""
    if not frames:
        return 0, 0
    dimensions = await get_collection_dimensions(collection.id)
    if dimensions is None:
        raise ValueError("Collection vector dimensions are not initialized")
    await ensure_semantic_frame_table(collection.id, dimensions)
    vector_table = table_name(collection.id, "semantic_frame_embeddings")
    frame_values = [
        {
            "id": frame.id,
            "collection_id": frame.collection_id,
            "proposition_entity_id": frame.proposition_entity_id,
            "source_relationship_id": frame.source_relationship_id,
            "frame_kind": frame.frame_kind,
            "predicate": frame.predicate,
            "title": frame.title,
            "frame_text": frame.frame_text,
            "content_hash": frame.content_hash,
            "polarity": frame.polarity,
            "modality": frame.modality,
            "conditions_json": list(frame.conditions),
            "exceptions_json": list(frame.exceptions),
            "scopes_json": list(frame.scopes),
            "executable_status": frame.executable_status,
            "metadata_json": frame.metadata,
        }
        for frame in frames
    ]
    argument_values = [
        {
            "id": deterministic_frame_id(
                collection.id,
                f"frame-argument:{frame.id}:{argument.position}:{argument.role}",
            ),
            "frame_id": frame.id,
            "position": argument.position,
            "role": argument.role,
            "entity_id": argument.entity_id,
            "argument_type": argument.argument_type,
        }
        for frame in frames
        for argument in frame.arguments
    ]
    async with AsyncSessionLocal() as session:
        previous_hashes = dict(
            (
                await session.execute(
                    select(
                        GraphSemanticFrame.id,
                        GraphSemanticFrame.content_hash,
                    ).where(
                        GraphSemanticFrame.id.in_([frame.id for frame in frames])
                    )
                )
            ).all()
        )
        frame_insert = pg_insert(GraphSemanticFrame).values(frame_values)
        await session.execute(
            frame_insert.on_conflict_do_update(
                index_elements=[GraphSemanticFrame.id],
                set_={
                    "proposition_entity_id": frame_insert.excluded.proposition_entity_id,
                    "source_relationship_id": frame_insert.excluded.source_relationship_id,
                    "frame_kind": frame_insert.excluded.frame_kind,
                    "predicate": frame_insert.excluded.predicate,
                    "title": frame_insert.excluded.title,
                    "frame_text": frame_insert.excluded.frame_text,
                    "content_hash": frame_insert.excluded.content_hash,
                    "polarity": frame_insert.excluded.polarity,
                    "modality": frame_insert.excluded.modality,
                    "conditions_json": frame_insert.excluded.conditions_json,
                    "exceptions_json": frame_insert.excluded.exceptions_json,
                    "scopes_json": frame_insert.excluded.scopes_json,
                    "executable_status": frame_insert.excluded.executable_status,
                    "metadata_json": frame_insert.excluded.metadata_json,
                },
            )
        )
        if argument_values:
            argument_insert = pg_insert(GraphFrameArgument).values(argument_values)
            await session.execute(
                argument_insert.on_conflict_do_update(
                    index_elements=[GraphFrameArgument.id],
                    set_={
                        "entity_id": argument_insert.excluded.entity_id,
                        "argument_type": argument_insert.excluded.argument_type,
                    },
                )
            )
        await session.commit()

    changed_ids = [
        frame.id
        for frame in frames
        if frame.id in previous_hashes
        and previous_hashes[frame.id] != frame.content_hash
    ]
    if changed_ids:
        async with AsyncSessionLocal() as session:
            await session.execute(
                text(
                    f"DELETE FROM {vector_table} WHERE frame_id IN :frame_ids"
                ).bindparams(bindparam("frame_ids", expanding=True)),
                {"frame_ids": tuple(changed_ids)},
            )
            await session.commit()

    async with AsyncSessionLocal() as session:
        existing = set(
            (
                await session.execute(
                    text(
                        f"SELECT frame_id FROM {vector_table} "
                        "WHERE collection_id = :collection_id"
                    ),
                    {"collection_id": _uuid_for_sql(collection.id)},
                )
            ).scalars()
        )
    missing = [frame for frame in frames if frame.id not in existing]
    embedded = 0
    for offset in range(0, len(missing), batch_size):
        batch = missing[offset : offset + batch_size]
        embeddings = await provider.embed_documents(
            [frame.frame_text for frame in batch]
        )
        async with AsyncSessionLocal() as session:
            for frame, embedding in zip(batch, embeddings, strict=True):
                await session.execute(
                    text(
                        f"INSERT INTO {vector_table} "
                        "(frame_id, collection_id, graph_version_id, frame_kind, "
                        f"content, embedding) VALUES (:frame_id, :collection_id, "
                        ":graph_version_id, :frame_kind, :content, "
                        f"(:embedding)::vector({dimensions})) "
                        "ON CONFLICT (frame_id) DO NOTHING"
                    ),
                    {
                        "frame_id": _uuid_for_sql(frame.id),
                        "collection_id": _uuid_for_sql(collection.id),
                        "graph_version_id": None,
                        "frame_kind": frame.frame_kind,
                        "content": frame.frame_text,
                        "embedding": _vector_literal(embedding),
                    },
                )
                embedded += 1
            await session.commit()
    return len(frames), embedded


async def replace_community_frames(
    collection: Collection,
    provider: EmbeddingProvider,
    frames: list[SemanticFrameInput],
) -> tuple[int, int]:
    """Replace the collection's derived navigation layer after analytics."""
    dimensions = await get_collection_dimensions(collection.id)
    if dimensions is None:
        raise ValueError("Collection vector dimensions are not initialized")
    await ensure_semantic_frame_table(collection.id, dimensions)
    vector_table = table_name(collection.id, "semantic_frame_embeddings")
    async with AsyncSessionLocal() as session:
        old_ids = set(
            (
                await session.execute(
                    select(GraphSemanticFrame.id).where(
                        GraphSemanticFrame.collection_id == collection.id,
                        GraphSemanticFrame.frame_kind == "community_summary",
                    )
                )
            ).scalars()
        )
        stale_ids = old_ids - {frame.id for frame in frames}
        if stale_ids:
            await session.execute(
                text(
                    f"DELETE FROM {vector_table} WHERE frame_id IN :frame_ids"
                ).bindparams(bindparam("frame_ids", expanding=True)),
                {"frame_ids": tuple(stale_ids)},
            )
            await session.execute(
                delete(GraphSemanticFrame).where(
                    GraphSemanticFrame.id.in_(stale_ids)
                )
            )
            await session.commit()
    return await persist_semantic_frames(collection, provider, frames)
