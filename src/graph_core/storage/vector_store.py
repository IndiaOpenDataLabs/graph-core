"""Postgres-backed vector storage with SQLite fallback for tests."""

from __future__ import annotations

import math
import uuid
from dataclasses import dataclass

from sqlalchemy import Select, select, text

from graph_core.database import AsyncSessionLocal
from graph_core.models.vector_chunk import VectorChunk


@dataclass
class VectorSearchResult:
    chunk_id: uuid.UUID
    content: str
    score: float
    metadata: dict | None


class VectorStore:
    async def upsert_chunks(
        self,
        namespace_id: uuid.UUID,
        collection_id: uuid.UUID,
        chunks: list[dict],
    ) -> None:
        async with AsyncSessionLocal() as session:
            for chunk in chunks:
                existing = await session.execute(
                    select(VectorChunk.id).where(
                        VectorChunk.collection_id == collection_id,
                        VectorChunk.chunk_hash == chunk["chunk_hash"],
                        VectorChunk.chunk_index == chunk["chunk_index"],
                    )
                )
                if existing.scalar_one_or_none() is not None:
                    continue

                row = VectorChunk(
                    namespace_id=namespace_id,
                    collection_id=collection_id,
                    chunk_hash=chunk["chunk_hash"],
                    chunk_index=chunk["chunk_index"],
                    content=chunk["content"],
                    token_count=chunk["token_count"],
                    metadata_json=chunk.get("metadata"),
                    embedding=chunk["embedding"],
                )
                session.add(row)
            await session.commit()

    async def query_chunks(
        self,
        collection_id: uuid.UUID,
        query_embedding: list[float],
        top_k: int,
    ) -> list[VectorSearchResult]:
        async with AsyncSessionLocal() as session:
            if session.bind and session.bind.dialect.name == "postgresql":
                query_literal = "[" + ",".join(str(float(value)) for value in query_embedding) + "]"
                result = await session.execute(
                    text(
                        """
                        SELECT id, content, metadata_json, 1 - (embedding <=> CAST(:query_embedding AS vector)) AS score
                        FROM vector_chunks
                        WHERE collection_id = :collection_id
                        ORDER BY embedding <=> CAST(:query_embedding AS vector)
                        LIMIT :top_k
                        """
                    ),
                    {
                        "collection_id": collection_id,
                        "query_embedding": query_literal,
                        "top_k": top_k,
                    },
                )
                return [
                    VectorSearchResult(
                        chunk_id=row.id,
                        content=row.content,
                        score=float(row.score),
                        metadata=row.metadata_json,
                    )
                    for row in result
                ]

            statement: Select[tuple[VectorChunk]] = (
                select(VectorChunk)
                .where(VectorChunk.collection_id == collection_id)
            )
            result = await session.execute(statement)
            rows = list(result.scalars().all())
            scored = [
                VectorSearchResult(
                    chunk_id=row.id,
                    content=row.content,
                    score=_cosine_similarity(query_embedding, row.embedding),
                    metadata=row.metadata_json,
                )
                for row in rows
            ]
            return sorted(scored, key=lambda item: item.score, reverse=True)[:top_k]


def _cosine_similarity(left: list[float], right: list[float]) -> float:
    numerator = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return numerator / (left_norm * right_norm)
