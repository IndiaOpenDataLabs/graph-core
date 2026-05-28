"""Graph RAG pgvector storage — per-collection vector operations."""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import Any

from sqlalchemy import text

from graph_core.database import AsyncSessionLocal
from graph_core.storage.vector_tables import (
    get_collection_dimensions,
    table_name,
)


def _embedding_literal(embedding: list[float]) -> str:
    return "[" + ",".join(str(float(v)) for v in embedding) + "]"


def _vector_cast_sql(dimensions: int) -> str:
    """Return the SQL cast suffix for a vector parameter."""
    return f"::vector({dimensions})"


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
    ) -> None:
        tbl = table_name(collection_id, "entity_embeddings")
        dimensions = await get_collection_dimensions(collection_id)
        if dimensions is None:
            raise ValueError(f"Collection {collection_id} has no embedding dimensions")

        cast = _vector_cast_sql(dimensions)

        async with AsyncSessionLocal() as session:
            await session.execute(
                text(
                    f"INSERT INTO {tbl} "
                    f"(entity_id, collection_id, name, description, description_id, embedding) "
                    f"VALUES (:eid, :cid, :name, :desc, :did, (:emb){cast})"
                ),
                {
                    "eid": entity_id,
                    "cid": collection_id,
                    "name": name,
                    "desc": description,
                    "did": description_id,
                    "emb": _embedding_literal(embedding),
                },
            )
            await session.commit()

    async def search_entity_embeddings(
        self,
        collection_id: uuid.UUID,
        query_embedding: list[float],
        top_k: int,
    ) -> list[VectorSearchHit]:
        tbl = table_name(collection_id, "entity_embeddings")
        dimensions = await get_collection_dimensions(collection_id)
        if dimensions is None:
            raise ValueError(f"Collection {collection_id} has no embedding dimensions")

        cast = _vector_cast_sql(dimensions)

        async with AsyncSessionLocal() as session:
            result = await session.execute(
                text(
                    f"""
                    SELECT id::text, entity_id::text, description_id::text, name, description,
                           1 - (embedding <=> (:qemb){cast}) as score,
                           embedding <=> (:qemb){cast} as distance
                    FROM {tbl}
                    WHERE collection_id = :cid
                    ORDER BY distance
                    LIMIT :top_k
                    """
                ),
                {
                    "cid": collection_id,
                    "top_k": top_k,
                    "qemb": _embedding_literal(query_embedding),
                },
            )
            return [
                VectorSearchHit(
                    id=row[0],
                    distance=float(row[6]),
                    content=row[4],
                    metadata={
                        "entity_id": row[1],
                        "description_id": row[2],
                        "name": row[3],
                        "collection_id": str(collection_id),
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
                    f"(relationship_id, collection_id, source_name, target_name, description, embedding) "
                    f"VALUES (:rid, :cid, :sn, :tn, :desc, (:emb){cast})"
                ),
                {
                    "rid": relationship_id,
                    "cid": collection_id,
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
    ) -> list[VectorSearchHit]:
        tbl = table_name(collection_id, "relationship_embeddings")
        dimensions = await get_collection_dimensions(collection_id)
        if dimensions is None:
            raise ValueError(f"Collection {collection_id} has no embedding dimensions")

        cast = _vector_cast_sql(dimensions)

        where_extra = ""
        params: dict = {
            "cid": collection_id,
            "top_k": top_k,
            "qemb": _embedding_literal(query_embedding),
        }
        if relationship_id:
            where_extra = "AND relationship_id = :rel_id"
            params["rel_id"] = relationship_id

        async with AsyncSessionLocal() as session:
            result = await session.execute(
                text(
                    f"""
                    SELECT id::text, relationship_id::text, source_name, target_name, description,
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
                    distance=float(row[6]),
                    content=row[4],
                    metadata={
                        "relationship_id": row[1],
                        "source_name": row[2],
                        "target_name": row[3],
                        "collection_id": str(collection_id),
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
    ) -> None:
        tbl = table_name(collection_id, "entity_centroids")
        dimensions = await get_collection_dimensions(collection_id)
        if dimensions is None:
            raise ValueError(f"Collection {collection_id} has no embedding dimensions")

        cast = _vector_cast_sql(dimensions)
        emb_str = _embedding_literal(embedding)

        async with AsyncSessionLocal() as session:
            existing = await session.execute(
                text(f"SELECT id FROM {tbl} WHERE entity_id = :eid"),
                {"eid": entity_id},
            )
            if existing.scalar_one_or_none():
                await session.execute(
                    text(
                        f"UPDATE {tbl} SET "
                        f"embedding = (:emb){cast}, "
                        f"canonical_name = :cn, "
                        f"primary_type = :pt, "
                        f"description_count = :dc "
                        f"WHERE entity_id = :eid"
                    ),
                    {
                        "cn": canonical_name,
                        "pt": primary_type,
                        "dc": description_count,
                        "eid": entity_id,
                        "emb": emb_str,
                    },
                )
            else:
                await session.execute(
                    text(
                        f"INSERT INTO {tbl} "
                        f"(entity_id, collection_id, canonical_name, primary_type, "
                        f"description_count, embedding) "
                        f"VALUES (:eid, :cid, :cn, :pt, :dc, (:emb){cast})"
                    ),
                    {
                        "eid": entity_id,
                        "cid": collection_id,
                        "cn": canonical_name,
                        "pt": primary_type,
                        "dc": description_count,
                        "emb": emb_str,
                    },
                )
            await session.commit()

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
                    "cid": collection_id,
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
                        "collection_id": str(collection_id),
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
                {"eid": entity_id},
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
                {"cid": collection_id, "ch": chunk_hash},
            )
            if existing.scalar_one_or_none():
                return

            await session.execute(
                text(
                    f"INSERT INTO {tbl} "
                    f"(collection_id, chunk_hash, chunk_index, content, embedding) "
                    f"VALUES (:cid, :ch, :ci, :content, (:emb){cast})"
                ),
                {
                    "cid": collection_id,
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
    ) -> list[VectorSearchHit]:
        tbl = table_name(collection_id, "chunk_embeddings")
        dimensions = await get_collection_dimensions(collection_id)
        if dimensions is None:
            raise ValueError(f"Collection {collection_id} has no embedding dimensions")

        cast = _vector_cast_sql(dimensions)

        async with AsyncSessionLocal() as session:
            result = await session.execute(
                text(
                    f"""
                    SELECT id::text, chunk_hash, chunk_index, content,
                           1 - (embedding <=> (:qemb){cast}) as score,
                           embedding <=> (:qemb){cast} as distance
                    FROM {tbl}
                    WHERE collection_id = :cid
                    ORDER BY distance
                    LIMIT :top_k
                    """
                ),
                {
                    "cid": collection_id,
                    "top_k": top_k,
                    "qemb": _embedding_literal(query_embedding),
                },
            )
            return [
                VectorSearchHit(
                    id=row[0],
                    distance=float(row[5]),
                    content=row[3],
                    metadata={
                        "chunk_hash": row[1],
                        "chunk_index": row[2],
                        "collection_id": str(collection_id),
                    },
                )
                for row in result
            ]
