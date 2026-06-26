"""Graph RAG pgvector storage — per-collection vector operations."""

from __future__ import annotations

import math
import uuid
from dataclasses import dataclass
from typing import Any

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from graph_core.database import AsyncSessionLocal, _uuid_for_sql
from graph_core.storage.vector_tables import (
    get_collection_dimensions,
    table_name,
)


def _embedding_literal(embedding: list[float]) -> str:
    if not embedding:
        raise ValueError("Embedding cannot be empty")
    if any(not math.isfinite(float(v)) for v in embedding):
        raise ValueError("Embedding contains non-finite values")
    return "[" + ",".join(str(float(v)) for v in embedding) + "]"


def _vector_cast_sql(dimensions: int) -> str:
    """Return the SQL cast suffix for a vector parameter."""
    return f"::vector({dimensions})"


def _expand_uuid_params(
    values: list[uuid.UUID],
    prefix: str,
) -> tuple[str, dict[str, Any]]:
    if not values:
        return "", {}
    params: dict[str, Any] = {}
    placeholders: list[str] = []
    for index, value in enumerate(values):
        key = f"{prefix}_{index}"
        placeholders.append(f":{key}")
        params[key] = _uuid_for_sql(value)
    return ", ".join(placeholders), params


@dataclass
class VectorSearchHit:
    id: str
    distance: float
    content: str
    metadata: dict[str, Any]


class GraphRAGVectorStore:
    """Pgvector-backed storage for graph RAG vectors.

    All tables are per-collection, created at collection creation time
    with the embedding dimensions from the collection's profile.
    """

    # ── Entity Embeddings ──

    async def upsert_entity_embedding(
        self,
        entity_id: uuid.UUID,
        collection_id: uuid.UUID,
        name: str,
        description: str,
        description_id: uuid.UUID,
        embedding: list[float],
        document_id: uuid.UUID | None = None,
        document_path: str | None = None,
        session: AsyncSession | None = None,
    ) -> None:
        tbl = table_name(collection_id, "entity_embeddings")
        dimensions = await get_collection_dimensions(collection_id)
        if dimensions is None:
            raise ValueError(f"Collection {collection_id} has no embedding dimensions")

        cast = _vector_cast_sql(dimensions)

        owns_session = session is None
        if session is None:
            session = AsyncSessionLocal()

        try:
            await session.execute(
                text(
                    f"INSERT INTO {tbl} "
                    f"(entity_id, collection_id, document_id, document_path, name, description, description_id, embedding) "
                    f"VALUES (:eid, :cid, :document_id, :document_path, :name, :desc, :did, (:emb){cast})"
                ),
                {
                    "eid": _uuid_for_sql(entity_id),
                    "cid": _uuid_for_sql(collection_id),
                    "document_id": _uuid_for_sql(document_id) if document_id else None,
                    "document_path": document_path,
                    "name": name,
                    "desc": description,
                    "did": _uuid_for_sql(description_id),
                    "emb": _embedding_literal(embedding),
                },
            )
            if owns_session:
                await session.commit()
        finally:
            if owns_session:
                await session.close()

    async def search_entity_embeddings(
        self,
        collection_id: uuid.UUID,
        query_embedding: list[float],
        top_k: int,
        entity_id: uuid.UUID | None = None,
        document_ids: list[uuid.UUID] | None = None,
    ) -> list[VectorSearchHit]:
        tbl = table_name(collection_id, "entity_embeddings")
        dimensions = await get_collection_dimensions(collection_id)
        if dimensions is None:
            raise ValueError(f"Collection {collection_id} has no embedding dimensions")

        cast = _vector_cast_sql(dimensions)

        where_extra = ""
        params: dict[str, Any] = {
            "cid": _uuid_for_sql(collection_id),
            "top_k": top_k,
            "qemb": _embedding_literal(query_embedding),
        }
        if entity_id:
            where_extra = "AND entity_id = :entity_id"
            params["entity_id"] = _uuid_for_sql(entity_id)
        if document_ids:
            document_clause, document_params = _expand_uuid_params(
                document_ids, "document_id"
            )
            where_extra += (
                f" AND document_id IN ({document_clause})" if document_clause else ""
            )
            params.update(document_params)

        async with AsyncSessionLocal() as session:
            result = await session.execute(
                text(
                    f"""
                    SELECT id::text, entity_id::text, description_id::text, name, description,
                           document_id::text, document_path,
                           1 - (embedding <=> (:qemb){cast}) as score,
                           embedding <=> (:qemb){cast} as distance
                    FROM {tbl}
                    WHERE collection_id = :cid {where_extra}
                    ORDER BY distance
                    LIMIT :top_k
                    """
                ),
                params,
            )
            return [
                VectorSearchHit(
                    id=row[0],
                    distance=float(row[8]),
                    content=row[4],
                    metadata={
                        "entity_id": row[1],
                        "description_id": row[2],
                        "name": row[3],
                        "document_id": row[5],
                        "document_path": row[6],
                        "collection_id": _uuid_for_sql(collection_id),
                    },
                )
                for row in result
            ]

    # ── Relationship Embeddings ──

    async def upsert_relationship_embedding(
        self,
        relationship_id: uuid.UUID,
        collection_id: uuid.UUID,
        source_name: str,
        target_name: str,
        description: str,
        embedding: list[float],
        document_id: uuid.UUID | None = None,
        document_path: str | None = None,
    ) -> None:
        tbl = table_name(collection_id, "relationship_embeddings")
        dimensions = await get_collection_dimensions(collection_id)
        if dimensions is None:
            raise ValueError(f"Collection {collection_id} has no embedding dimensions")

        cast = _vector_cast_sql(dimensions)

        async with AsyncSessionLocal() as session:
            await session.execute(
                text(
                    f"INSERT INTO {tbl} "
                    f"(relationship_id, collection_id, document_id, document_path, source_name, target_name, description, embedding) "
                    f"VALUES (:rid, :cid, :document_id, :document_path, :sn, :tn, :desc, (:emb){cast})"
                ),
                {
                    "rid": _uuid_for_sql(relationship_id),
                    "cid": _uuid_for_sql(collection_id),
                    "document_id": _uuid_for_sql(document_id) if document_id else None,
                    "document_path": document_path,
                    "sn": source_name,
                    "tn": target_name,
                    "desc": description,
                    "emb": _embedding_literal(embedding),
                },
            )
            await session.commit()

    async def search_relationship_embeddings(
        self,
        collection_id: uuid.UUID,
        query_embedding: list[float],
        top_k: int,
        relationship_id: uuid.UUID | None = None,
        document_ids: list[uuid.UUID] | None = None,
    ) -> list[VectorSearchHit]:
        tbl = table_name(collection_id, "relationship_embeddings")
        dimensions = await get_collection_dimensions(collection_id)
        if dimensions is None:
            raise ValueError(f"Collection {collection_id} has no embedding dimensions")

        cast = _vector_cast_sql(dimensions)

        where_extra = ""
        params: dict = {
            "cid": _uuid_for_sql(collection_id),
            "top_k": top_k,
            "qemb": _embedding_literal(query_embedding),
        }
        if relationship_id:
            where_extra = "AND relationship_id = :rel_id"
            params["rel_id"] = _uuid_for_sql(relationship_id)
        if document_ids:
            document_clause, document_params = _expand_uuid_params(
                document_ids, "document_id"
            )
            where_extra += (
                f" AND document_id IN ({document_clause})" if document_clause else ""
            )
            params.update(document_params)

        async with AsyncSessionLocal() as session:
            result = await session.execute(
                text(
                    f"""
                    SELECT id::text, relationship_id::text, source_name, target_name, description,
                           document_id::text, document_path,
                           1 - (embedding <=> (:qemb){cast}) as score,
                           embedding <=> (:qemb){cast} as distance
                    FROM {tbl}
                    WHERE collection_id = :cid {where_extra}
                    ORDER BY distance
                    LIMIT :top_k
                    """
                ),
                params,
            )
            return [
                VectorSearchHit(
                    id=row[0],
                    distance=float(row[8]),
                    content=row[4],
                    metadata={
                        "relationship_id": row[1],
                        "source_name": row[2],
                        "target_name": row[3],
                        "document_id": row[5],
                        "document_path": row[6],
                        "collection_id": _uuid_for_sql(collection_id),
                    },
                )
                for row in result
            ]

    # ── Entity Centroids ──

    async def upsert_entity_centroid(
        self,
        entity_id: uuid.UUID,
        collection_id: uuid.UUID,
        canonical_name: str,
        primary_type: str | None,
        description_count: int,
        embedding: list[float],
        session: AsyncSession | None = None,
    ) -> None:
        tbl = table_name(collection_id, "entity_centroids")
        dimensions = await get_collection_dimensions(collection_id)
        if dimensions is None:
            raise ValueError(f"Collection {collection_id} has no embedding dimensions")

        cast = _vector_cast_sql(dimensions)
        emb_str = _embedding_literal(embedding)

        owns_session = session is None
        if session is None:
            session = AsyncSessionLocal()

        try:
            await session.execute(
                text(
                    f"INSERT INTO {tbl} "
                    f"(entity_id, collection_id, canonical_name, primary_type, "
                    f"description_count, embedding) "
                    f"VALUES (:eid, :cid, :cn, :pt, :dc, (:emb){cast}) "
                    f"ON CONFLICT (entity_id) DO UPDATE SET "
                    f"embedding = (:emb){cast}, "
                    f"canonical_name = :cn, "
                    f"primary_type = :pt, "
                    f"description_count = :dc"
                ),
                {
                    "eid": _uuid_for_sql(entity_id),
                    "cid": _uuid_for_sql(collection_id),
                    "cn": canonical_name,
                    "pt": primary_type,
                    "dc": description_count,
                    "emb": emb_str,
                },
            )
            if owns_session:
                await session.commit()
        finally:
            if owns_session:
                await session.close()

    async def search_entity_centroids(
        self,
        collection_id: uuid.UUID,
        query_embedding: list[float],
        top_k: int,
    ) -> list[VectorSearchHit]:
        tbl = table_name(collection_id, "entity_centroids")
        dimensions = await get_collection_dimensions(collection_id)
        if dimensions is None:
            raise ValueError(f"Collection {collection_id} has no embedding dimensions")

        cast = _vector_cast_sql(dimensions)

        async with AsyncSessionLocal() as session:
            result = await session.execute(
                text(
                    f"""
                    SELECT id::text, entity_id::text, canonical_name, primary_type, description_count,
                           1 - (embedding <=> (:qemb){cast}) as score,
                           embedding <=> (:qemb){cast} as distance
                    FROM {tbl}
                    WHERE collection_id = :cid
                    ORDER BY distance
                    LIMIT :top_k
                    """
                ),
                {
                    "cid": _uuid_for_sql(collection_id),
                    "top_k": top_k,
                    "qemb": _embedding_literal(query_embedding),
                },
            )
            return [
                VectorSearchHit(
                    id=row[0],
                    distance=float(row[6]),
                    content=row[2],
                    metadata={
                        "entity_id": row[1],
                        "canonical_name": row[2],
                        "primary_type": row[3] or "",
                        "description_count": row[4],
                        "collection_id": _uuid_for_sql(collection_id),
                    },
                )
                for row in result
            ]

    async def get_entity_centroid(
        self, entity_id: uuid.UUID, collection_id: uuid.UUID
    ) -> list[float] | None:
        tbl = table_name(collection_id, "entity_centroids")
        async with AsyncSessionLocal() as session:
            result = await session.execute(
                text(
                    f"SELECT embedding::text FROM {tbl} "
                    f"WHERE entity_id = :eid"
                ),
                {"eid": _uuid_for_sql(entity_id)},
            )
            row = result.one_or_none()
            if row is None:
                return None
            raw = row[0].strip("[]")
            return [float(v) for v in raw.split(",")]

    # ── Chunk Embeddings ──

    async def upsert_chunk_embedding(
        self,
        collection_id: uuid.UUID,
        chunk_hash: str,
        chunk_index: int,
        content: str,
        embedding: list[float],
        document_id: uuid.UUID | None = None,
        document_path: str | None = None,
    ) -> None:
        tbl = table_name(collection_id, "chunk_embeddings")
        dimensions = await get_collection_dimensions(collection_id)
        if dimensions is None:
            raise ValueError(f"Collection {collection_id} has no embedding dimensions")

        cast = _vector_cast_sql(dimensions)

        async with AsyncSessionLocal() as session:
            existing = await session.execute(
                text(
                    f"SELECT id FROM {tbl} "
                    f"WHERE collection_id = :cid AND chunk_hash = :ch"
                ),
                {"cid": _uuid_for_sql(collection_id), "ch": chunk_hash},
            )
            if existing.scalar_one_or_none():
                return

            await session.execute(
                text(
                    f"INSERT INTO {tbl} "
                    f"(collection_id, document_id, document_path, chunk_hash, chunk_index, content, embedding) "
                    f"VALUES (:cid, :document_id, :document_path, :ch, :ci, :content, (:emb){cast})"
                ),
                {
                    "cid": _uuid_for_sql(collection_id),
                    "document_id": _uuid_for_sql(document_id) if document_id else None,
                    "document_path": document_path,
                    "ch": chunk_hash,
                    "ci": chunk_index,
                    "content": content,
                    "emb": _embedding_literal(embedding),
                },
            )
            await session.commit()

    async def search_chunk_embeddings(
        self,
        collection_id: uuid.UUID,
        query_embedding: list[float],
        top_k: int,
        document_ids: list[uuid.UUID] | None = None,
    ) -> list[VectorSearchHit]:
        tbl = table_name(collection_id, "chunk_embeddings")
        dimensions = await get_collection_dimensions(collection_id)
        if dimensions is None:
            raise ValueError(f"Collection {collection_id} has no embedding dimensions")

        cast = _vector_cast_sql(dimensions)

        async with AsyncSessionLocal() as session:
            where_extra = ""
            params: dict[str, Any] = {
                "cid": _uuid_for_sql(collection_id),
                "top_k": top_k,
                "qemb": _embedding_literal(query_embedding),
            }
            if document_ids:
                document_clause, document_params = _expand_uuid_params(
                    document_ids, "document_id"
                )
                where_extra = (
                    f" AND document_id IN ({document_clause})"
                    if document_clause
                    else ""
                )
                params.update(document_params)
            result = await session.execute(
                text(
                    f"""
                    SELECT id::text, chunk_hash, chunk_index, content,
                           document_id::text, document_path,
                           1 - (embedding <=> (:qemb){cast}) as score,
                           embedding <=> (:qemb){cast} as distance
                    FROM {tbl}
                    WHERE collection_id = :cid {where_extra}
                    ORDER BY distance
                    LIMIT :top_k
                    """
                ),
                params,
            )
            return [
                VectorSearchHit(
                    id=row[0],
                    distance=float(row[7]),
                    content=row[3],
                    metadata={
                        "chunk_hash": row[1],
                        "chunk_index": row[2],
                        "document_id": row[4],
                        "document_path": row[5],
                        "collection_id": _uuid_for_sql(collection_id),
                    },
                )
                for row in result
            ]

    # ── Prefix (rel_type) Embeddings ──

    async def ensure_prefix_embeddings_table(
        self,
        collection_id: uuid.UUID,
    ) -> None:
        """Create the prefix_embeddings table if it doesn't exist."""
        tbl = table_name(collection_id, "prefix_embeddings")
        dimensions = await get_collection_dimensions(collection_id)
        if dimensions is None:
            return
        async with AsyncSessionLocal() as session:
            await session.execute(
                text(
                    f"CREATE TABLE IF NOT EXISTS {tbl} ("
                    f"  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),"
                    f"  collection_id UUID NOT NULL REFERENCES collections(id) ON DELETE CASCADE,"
                    f"  rel_type VARCHAR(256) NOT NULL,"
                    f"  embedding vector({dimensions}) NOT NULL,"
                    f"  created_at TIMESTAMPTZ DEFAULT now(),"
                    f"  CONSTRAINT uq_{tbl}_rel_type UNIQUE (collection_id, rel_type)"
                    f")"
                )
            )
            await session.commit()

    async def upsert_prefix_embedding(
        self,
        collection_id: uuid.UUID,
        rel_type: str,
        embedding: list[float],
    ) -> None:
        tbl = table_name(collection_id, "prefix_embeddings")
        dimensions = await get_collection_dimensions(collection_id)
        if dimensions is None:
            return
        cast = _vector_cast_sql(dimensions)
        async with AsyncSessionLocal() as session:
            await session.execute(
                text(
                    f"INSERT INTO {tbl} (collection_id, rel_type, embedding) "
                    f"VALUES (:cid, :rt, (:emb){cast}) "
                    f"ON CONFLICT (collection_id, rel_type) "
                    f"DO UPDATE SET embedding = EXCLUDED.embedding"
                ),
                {
                    "cid": _uuid_for_sql(collection_id),
                    "rt": rel_type,
                    "emb": _embedding_literal(embedding),
                },
            )
            await session.commit()

    async def load_all_prefix_embeddings(
        self,
        collection_id: uuid.UUID,
    ) -> dict[str, list[float]]:
        """Return {rel_type: embedding} for all stored prefix embeddings."""
        tbl = table_name(collection_id, "prefix_embeddings")
        async with AsyncSessionLocal() as session:
            result = await session.execute(
                text(
                    f"SELECT rel_type, embedding FROM {tbl} WHERE collection_id = :cid"
                ),
                {"cid": _uuid_for_sql(collection_id)},
            )
            rows = result.fetchall()
        result_dict: dict[str, list[float]] = {}
        for row in rows:
            emb = row[1]
            if emb is not None:
                if isinstance(emb, list):
                    result_dict[row[0]] = [float(v) for v in emb]
                elif isinstance(emb, str):
                    # pgvector returns vectors as "[0.1,0.2,...]" strings
                    inner = emb.strip("[]")
                    result_dict[row[0]] = [float(v) for v in inner.split(",") if v.strip()]
        return result_dict
