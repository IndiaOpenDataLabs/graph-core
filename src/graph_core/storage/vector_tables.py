"""Dynamic per-collection vector table management.

Creates and drops vector tables at collection creation/deletion time,
using the embedding dimensions from the collection's embedding profile.
This avoids hardcoded dimension mismatches across collections.
"""

from __future__ import annotations

import uuid

from sqlalchemy import text

from graph_core.database import AsyncSessionLocal, _uuid_for_sql


def _safe_id(collection_id: uuid.UUID) -> str:
    """Sanitize collection UUID for use in table names.

    Uses first 16 hex chars (no dashes) to keep names short and unique.
    """
    return collection_id.hex.replace("-", "")[:16]


def table_name(collection_id: uuid.UUID, kind: str) -> str:
    """Return the per-collection table name for a given kind.

    Kinds:
        vector_chunks
        entity_embeddings
        relationship_embeddings
        entity_centroids
        chunk_embeddings
    """
    return f"vc_{_safe_id(collection_id)}_{kind}"


def create_vector_chunks_sql(collection_id: uuid.UUID, dimensions: int) -> str:
    tbl = table_name(collection_id, "vector_chunks")
    return f"""
        CREATE TABLE {tbl} (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            namespace_id UUID NOT NULL REFERENCES namespaces(id) ON DELETE CASCADE,
            collection_id UUID NOT NULL REFERENCES collections(id) ON DELETE CASCADE,
            chunk_hash VARCHAR(64) NOT NULL,
            chunk_index INTEGER NOT NULL DEFAULT 0,
            content TEXT NOT NULL,
            token_count INTEGER NOT NULL DEFAULT 0,
            metadata_json JSON,
            embedding vector({dimensions}) NOT NULL,
            created_at TIMESTAMPTZ DEFAULT now(),
            CONSTRAINT uq_{tbl}_identity UNIQUE (collection_id, chunk_hash, chunk_index)
        )
    """


def create_entity_embeddings_sql(collection_id: uuid.UUID, dimensions: int) -> str:
    tbl = table_name(collection_id, "entity_embeddings")
    return f"""
        CREATE TABLE {tbl} (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            entity_id UUID NOT NULL REFERENCES graph_entities(id) ON DELETE CASCADE,
            description_id UUID NOT NULL,
            collection_id UUID NOT NULL REFERENCES collections(id) ON DELETE CASCADE,
            name VARCHAR(256) NOT NULL,
            description TEXT NOT NULL,
            embedding vector({dimensions}) NOT NULL,
            created_at TIMESTAMPTZ DEFAULT now()
        )
    """


def create_relationship_embeddings_sql(collection_id: uuid.UUID, dimensions: int) -> str:
    tbl = table_name(collection_id, "relationship_embeddings")
    return f"""
        CREATE TABLE {tbl} (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            relationship_id UUID NOT NULL REFERENCES graph_relationships(id) ON DELETE CASCADE,
            collection_id UUID NOT NULL REFERENCES collections(id) ON DELETE CASCADE,
            source_name VARCHAR(256) NOT NULL,
            target_name VARCHAR(256) NOT NULL,
            description TEXT NOT NULL,
            embedding vector({dimensions}) NOT NULL,
            created_at TIMESTAMPTZ DEFAULT now()
        )
    """


def create_entity_centroids_sql(collection_id: uuid.UUID, dimensions: int) -> str:
    tbl = table_name(collection_id, "entity_centroids")
    return f"""
        CREATE TABLE {tbl} (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            entity_id UUID NOT NULL REFERENCES graph_entities(id) ON DELETE CASCADE,
            collection_id UUID NOT NULL REFERENCES collections(id) ON DELETE CASCADE,
            canonical_name VARCHAR(256) NOT NULL,
            primary_type VARCHAR(64),
            description_count INTEGER DEFAULT 1,
            embedding vector({dimensions}) NOT NULL,
            created_at TIMESTAMPTZ DEFAULT now(),
            CONSTRAINT uq_{tbl}_entity_id UNIQUE (entity_id)
        )
    """


def create_chunk_embeddings_sql(collection_id: uuid.UUID, dimensions: int) -> str:
    tbl = table_name(collection_id, "chunk_embeddings")
    return f"""
        CREATE TABLE {tbl} (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            collection_id UUID NOT NULL REFERENCES collections(id) ON DELETE CASCADE,
            chunk_hash VARCHAR(64) NOT NULL,
            chunk_index INTEGER NOT NULL DEFAULT 0,
            content TEXT NOT NULL,
            embedding vector({dimensions}) NOT NULL,
            created_at TIMESTAMPTZ DEFAULT now()
        )
    """


ALL_KINDS = [
    "vector_chunks",
    "entity_embeddings",
    "relationship_embeddings",
    "entity_centroids",
    "chunk_embeddings",
]

CREATE_SQL_MAP: dict[str, callable] = {
    "vector_chunks": create_vector_chunks_sql,
    "entity_embeddings": create_entity_embeddings_sql,
    "relationship_embeddings": create_relationship_embeddings_sql,
    "entity_centroids": create_entity_centroids_sql,
    "chunk_embeddings": create_chunk_embeddings_sql,
}


async def create_all_tables(collection_id: uuid.UUID, dimensions: int) -> None:
    """Create all per-collection vector tables."""

    async with AsyncSessionLocal() as session:
        for kind in ALL_KINDS:
            sql_fn = CREATE_SQL_MAP[kind]
            sql = sql_fn(collection_id, dimensions)
            await session.execute(text(sql))
        await session.commit()


async def drop_all_tables(collection_id: uuid.UUID) -> None:
    """Drop all per-collection vector tables."""

    async with AsyncSessionLocal() as session:
        for kind in ALL_KINDS:
            tbl = table_name(collection_id, kind)
            await session.execute(text(f"DROP TABLE IF EXISTS {tbl} CASCADE"))
        await session.commit()


async def get_collection_dimensions(collection_id: uuid.UUID) -> int | None:
    """Get embedding dimensions from the collections table.

    We store the resolved embedding dimensions on the collection row
    so storage layers don't need to join the profiles table.
    """

    async with AsyncSessionLocal() as session:
        result = await session.execute(
            text("SELECT embedding_dimensions FROM collections WHERE id = :cid"),
            {"cid": _uuid_for_sql(collection_id)},
        )
        row = result.one_or_none()
        if row is None:
            return None
        return row[0]


def vector_cast(literal: str, dimensions: int) -> str:
    """Return the SQL cast expression for a vector literal."""
    return f"{literal}::vector({dimensions})"
