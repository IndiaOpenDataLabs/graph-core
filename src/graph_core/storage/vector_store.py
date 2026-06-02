"""Postgres-backed vector storage with per-collection dynamic tables."""

from __future__ import annotations

import json
import math
import uuid

from sqlalchemy import text

from graph_core.database import AsyncSessionLocal, _uuid_for_sql
from graph_core.storage.vector_tables import (
    get_collection_dimensions,
    table_name,
)


def _cosine_similarity(left: list[float], right: list[float]) -> float:
    numerator = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return numerator / (left_norm * right_norm)


def _embedding_literal(embedding: list[float]) -> str:
    return "[" + ",".join(str(float(v)) for v in embedding) + "]"


def _vector_cast_sql(dimensions: int) -> str:
    """Return the SQL cast suffix for a vector parameter."""
    return f"::vector({dimensions})"


class VectorStore:
    async def upsert_chunks(
        self,
        namespace_id: uuid.UUID,
        collection_id: uuid.UUID,
        chunks: list[dict],
    ) -> None:
        tbl = table_name(collection_id, "vector_chunks")
        dimensions = await get_collection_dimensions(collection_id)
        if dimensions is None:
            raise ValueError(f"Collection {collection_id} has no embedding dimensions")

        cast = _vector_cast_sql(dimensions)

        async with AsyncSessionLocal() as session:
            for chunk in chunks:
                existing = await session.execute(
                    text(
                        f"SELECT id FROM {tbl} "
                        f"WHERE collection_id = :cid "
                        f"AND chunk_hash = :ch "
                        f"AND chunk_index = :ci"
                    ),
                    {
                        "cid": _uuid_for_sql(collection_id),
                        "ch": chunk["chunk_hash"],
                        "ci": chunk["chunk_index"],
                    },
                )
                if existing.scalar_one_or_none() is not None:
                    continue

                await session.execute(
                    text(
                        f"INSERT INTO {tbl} "
                        f"(namespace_id, collection_id, chunk_hash, chunk_index, "
                        f"content, token_count, metadata_json, embedding) "
                        f"VALUES (:nsid, :cid, :ch, :ci, :content, :tc, :meta, (:emb){cast})"
                    ),
                    {
                        "nsid": _uuid_for_sql(namespace_id),
                        "cid": _uuid_for_sql(collection_id),
                        "ch": chunk["chunk_hash"],
                        "ci": chunk["chunk_index"],
                        "content": chunk["content"],
                        "tc": chunk["token_count"],
                        "meta": json.dumps(chunk.get("metadata")) if chunk.get("metadata") else None,
                        "emb": _embedding_literal(chunk["embedding"]),
                    },
                )
            await session.commit()

    async def query_chunks(
        self,
        collection_id: uuid.UUID,
        query_embedding: list[float],
        top_k: int,
        metadata_filters: dict[str, str] | None = None,
    ) -> list[dict]:
        tbl = table_name(collection_id, "vector_chunks")
        dimensions = await get_collection_dimensions(collection_id)
        if dimensions is None:
            raise ValueError(f"Collection {collection_id} has no embedding dimensions")

        cast = _vector_cast_sql(dimensions)

        filter_clauses: list[str] = ["collection_id = :cid"]
        params: dict[str, object] = {
            "cid": _uuid_for_sql(collection_id),
            "top_k": top_k,
            "qemb": _embedding_literal(query_embedding),
        }
        if metadata_filters:
            for idx, (key, value) in enumerate(metadata_filters.items()):
                key_param = f"meta_key_{idx}"
                value_param = f"meta_value_{idx}"
                filter_clauses.append(
                    f"metadata_json::jsonb ->> :{key_param} = :{value_param}"
                )
                params[key_param] = key
                params[value_param] = value
        where_sql = " AND ".join(filter_clauses)

        async with AsyncSessionLocal() as session:
            result = await session.execute(
                text(
                    f"""
                    SELECT id::text, content, metadata_json,
                           1 - (embedding <=> (:qemb){cast}) AS score
                    FROM {tbl}
                    WHERE {where_sql}
                    ORDER BY embedding <=> (:qemb){cast}
                    LIMIT :top_k
                    """
                ),
                params,
            )
            return [
                {
                    "chunk_id": row[0],
                    "content": row[1],
                    "score": float(row[3]),
                    "metadata": json.loads(row[2]) if isinstance(row[2], str) else row[2],
                }
                for row in result
            ]
