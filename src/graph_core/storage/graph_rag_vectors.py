"""Graph RAG pgvector storage — vector operations for entities, relationships, centroids, chunks."""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import Any

from sqlalchemy import select, text

from graph_core.database import AsyncSessionLocal
from graph_core.models.graph_rag_vectors import (
    GraphChunkEmbedding,
    GraphEntityCentroid,
    GraphEntityEmbedding,
    GraphRelationshipEmbedding,
)


@dataclass
class VectorSearchHit:
    id: str
    distance: float
    content: str
    metadata: dict[str, Any]


class GraphRAGVectorStore:
    """Pgvector-backed storage for graph RAG vectors.

    Manages 4 collections:
    - entity_embeddings: entity description embeddings (seed search)
    - relationship_embeddings: relationship description embeddings (edge scoring)
    - entity_centroids: centroid embeddings (incremental resolution)
    - chunk_embeddings: text chunk embeddings (naive retrieval)
    """

    async def _ensure_session(self):
        return AsyncSessionLocal()

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
        async with AsyncSessionLocal() as session:
            row = GraphEntityEmbedding(
                entity_id=entity_id,
                collection_id=collection_id,
                name=name,
                description=description,
                description_id=description_id,
                embedding=embedding,
            )
            session.add(row)
            await session.commit()

    async def search_entity_embeddings(
        self,
        collection_id: uuid.UUID,
        query_embedding: list[float],
        top_k: int,
    ) -> list[VectorSearchHit]:
        async with AsyncSessionLocal() as session:
            if session.bind and session.bind.dialect.name == "postgresql":
                qe = "[" + ",".join(str(float(v)) for v in query_embedding) + "]"
                result = await session.execute(
                    text(
                        """
                        SELECT id::text, entity_id::text, description_id::text, name, description,
                               1 - (embedding <=> CAST(:qe AS vector)) as score,
                               embedding <=> CAST(:qe AS vector) as distance
                        FROM graph_entity_embeddings
                        WHERE collection_id = :cid
                        ORDER BY distance
                        LIMIT :top_k
                        """
                    ),
                    {"qe": qe, "cid": collection_id, "top_k": top_k},
                )
                hits = []
                for row in result:
                    hits.append(VectorSearchHit(
                        id=row[0],
                        distance=float(row[6]),
                        content=row[4],
                        metadata={
                            "entity_id": row[1],
                            "description_id": row[2],
                            "name": row[3],
                            "collection_id": str(collection_id),
                        },
                    ))
                return hits
            return []

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
        async with AsyncSessionLocal() as session:
            row = GraphRelationshipEmbedding(
                relationship_id=relationship_id,
                collection_id=collection_id,
                source_name=source_name,
                target_name=target_name,
                description=description,
                embedding=embedding,
            )
            session.add(row)
            await session.commit()

    async def search_relationship_embeddings(
        self,
        collection_id: uuid.UUID,
        query_embedding: list[float],
        top_k: int,
        relationship_id: uuid.UUID | None = None,
    ) -> list[VectorSearchHit]:
        async with AsyncSessionLocal() as session:
            if session.bind and session.bind.dialect.name == "postgresql":
                qe = "[" + ",".join(str(float(v)) for v in query_embedding) + "]"
                where_extra = ""
                params = {"qe": qe, "cid": collection_id, "top_k": top_k}
                if relationship_id:
                    where_extra = "AND relationship_id = :rel_id"
                    params["rel_id"] = relationship_id

                result = await session.execute(
                    text(
                        f"""
                        SELECT id::text, relationship_id::text, source_name, target_name, description,
                               1 - (embedding <=> CAST(:qe AS vector)) as score,
                               embedding <=> CAST(:qe AS vector) as distance
                        FROM graph_relationship_embeddings
                        WHERE collection_id = :cid {where_extra}
                        ORDER BY distance
                        LIMIT :top_k
                        """
                    ),
                    params,
                )
                hits = []
                for row in result:
                    hits.append(VectorSearchHit(
                        id=row[0],
                        distance=float(row[6]),
                        content=row[4],
                        metadata={
                            "relationship_id": row[1],
                            "source_name": row[2],
                            "target_name": row[3],
                            "collection_id": str(collection_id),
                        },
                    ))
                return hits
            return []

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
        async with AsyncSessionLocal() as session:
            existing = await session.execute(
                select(GraphEntityCentroid).where(
                    GraphEntityCentroid.entity_id == entity_id
                )
            )
            existing_row = existing.scalar_one_or_none()
            if existing_row:
                existing_row.embedding = embedding
                existing_row.canonical_name = canonical_name
                existing_row.primary_type = primary_type
                existing_row.description_count = description_count
            else:
                row = GraphEntityCentroid(
                    entity_id=entity_id,
                    collection_id=collection_id,
                    canonical_name=canonical_name,
                    primary_type=primary_type,
                    description_count=description_count,
                    embedding=embedding,
                )
                session.add(row)
            await session.commit()

    async def search_entity_centroids(
        self,
        collection_id: uuid.UUID,
        query_embedding: list[float],
        top_k: int,
    ) -> list[VectorSearchHit]:
        async with AsyncSessionLocal() as session:
            if session.bind and session.bind.dialect.name == "postgresql":
                qe = "[" + ",".join(str(float(v)) for v in query_embedding) + "]"
                result = await session.execute(
                    text(
                        """
                        SELECT id::text, entity_id::text, canonical_name, primary_type, description_count,
                               1 - (embedding <=> CAST(:qe AS vector)) as score,
                               embedding <=> CAST(:qe AS vector) as distance
                        FROM graph_entity_centroids
                        WHERE collection_id = :cid
                        ORDER BY distance
                        LIMIT :top_k
                        """
                    ),
                    {"qe": qe, "cid": collection_id, "top_k": top_k},
                )
                hits = []
                for row in result:
                    hits.append(VectorSearchHit(
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
                    ))
                return hits
            return []

    async def get_entity_centroid(
        self, entity_id: uuid.UUID
    ) -> list[float] | None:
        async with AsyncSessionLocal() as session:
            result = await session.execute(
                select(GraphEntityCentroid.embedding).where(
                    GraphEntityCentroid.entity_id == entity_id
                )
            )
            row = result.one_or_none()
            return row.embedding if row else None

    # ── Chunk Embeddings ──

    async def upsert_chunk_embedding(
        self,
        collection_id: uuid.UUID,
        chunk_hash: str,
        chunk_index: int,
        content: str,
        embedding: list[float],
    ) -> None:
        async with AsyncSessionLocal() as session:
            existing = await session.execute(
                select(GraphChunkEmbedding).where(
                    GraphChunkEmbedding.collection_id == collection_id,
                    GraphChunkEmbedding.chunk_hash == chunk_hash,
                )
            )
            if existing.scalar_one_or_none():
                return
            row = GraphChunkEmbedding(
                collection_id=collection_id,
                chunk_hash=chunk_hash,
                chunk_index=chunk_index,
                content=content,
                embedding=embedding,
            )
            session.add(row)
            await session.commit()

    async def search_chunk_embeddings(
        self,
        collection_id: uuid.UUID,
        query_embedding: list[float],
        top_k: int,
    ) -> list[VectorSearchHit]:
        async with AsyncSessionLocal() as session:
            if session.bind and session.bind.dialect.name == "postgresql":
                qe = "[" + ",".join(str(float(v)) for v in query_embedding) + "]"
                result = await session.execute(
                    text(
                        """
                        SELECT id::text, chunk_hash, chunk_index, content,
                               1 - (embedding <=> CAST(:qe AS vector)) as score,
                               embedding <=> CAST(:qe AS vector) as distance
                        FROM graph_chunk_embeddings
                        WHERE collection_id = :cid
                        ORDER BY distance
                        LIMIT :top_k
                        """
                    ),
                    {"qe": qe, "cid": collection_id, "top_k": top_k},
                )
                hits = []
                for row in result:
                    hits.append(VectorSearchHit(
                        id=row[0],
                        distance=float(row[5]),
                        content=row[3],
                        metadata={
                            "chunk_hash": row[1],
                            "chunk_index": row[2],
                            "collection_id": str(collection_id),
                        },
                    ))
                return hits
            return []
