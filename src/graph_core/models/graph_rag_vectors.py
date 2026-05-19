"""Graph RAG vector storage — pgvector-backed tables for entity, relationship, centroid, and chunk embeddings."""

import uuid

from sqlalchemy import Column, DateTime, ForeignKey, Integer, String, Text, func
from sqlalchemy.dialects.postgresql import UUID as UUIDType

from graph_core.config import settings
from graph_core.database import Base
from graph_core.storage.vector_types import EmbeddingVector


class GraphEntityEmbedding(Base):
    """Entity description embeddings — used for seed entity search during query."""

    __tablename__ = "graph_entity_embeddings"

    id = Column(UUIDType(as_uuid=True), primary_key=True, default=uuid.uuid4)
    entity_id = Column(
        UUIDType(as_uuid=True),
        ForeignKey("graph_entities.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    description_id = Column(UUIDType(as_uuid=True), nullable=False)
    collection_id = Column(
        UUIDType(as_uuid=True),
        ForeignKey("collections.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    name = Column(String(256), nullable=False)
    description = Column(Text, nullable=False)
    embedding = Column(
        EmbeddingVector(settings.default_embedding_dimensions), nullable=False
    )
    created_at = Column(DateTime(timezone=True), server_default=func.now())


class GraphRelationshipEmbedding(Base):
    """Relationship description embeddings — used for edge scoring during traversal."""

    __tablename__ = "graph_relationship_embeddings"

    id = Column(UUIDType(as_uuid=True), primary_key=True, default=uuid.uuid4)
    relationship_id = Column(
        UUIDType(as_uuid=True),
        ForeignKey("graph_relationships.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    collection_id = Column(
        UUIDType(as_uuid=True),
        ForeignKey("collections.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    source_name = Column(String(256), nullable=False)
    target_name = Column(String(256), nullable=False)
    description = Column(Text, nullable=False)
    embedding = Column(
        EmbeddingVector(settings.default_embedding_dimensions), nullable=False
    )
    created_at = Column(DateTime(timezone=True), server_default=func.now())


class GraphEntityCentroid(Base):
    """Incremental entity centroid embeddings — used for entity resolution."""

    __tablename__ = "graph_entity_centroids"

    id = Column(UUIDType(as_uuid=True), primary_key=True, default=uuid.uuid4)
    entity_id = Column(
        UUIDType(as_uuid=True),
        ForeignKey("graph_entities.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
        unique=True,
    )
    collection_id = Column(
        UUIDType(as_uuid=True),
        ForeignKey("collections.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    canonical_name = Column(String(256), nullable=False)
    primary_type = Column(String(64), nullable=True)
    description_count = Column(Integer, default=1)
    embedding = Column(
        EmbeddingVector(settings.default_embedding_dimensions), nullable=False
    )
    created_at = Column(DateTime(timezone=True), server_default=func.now())


class GraphChunkEmbedding(Base):
    """Text chunk embeddings — used for naive vector retrieval fallback."""

    __tablename__ = "graph_chunk_embeddings"

    id = Column(UUIDType(as_uuid=True), primary_key=True, default=uuid.uuid4)
    collection_id = Column(
        UUIDType(as_uuid=True),
        ForeignKey("collections.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    chunk_hash = Column(String(64), nullable=False, index=True)
    chunk_index = Column(Integer, nullable=False, default=0)
    content = Column(Text, nullable=False)
    embedding = Column(
        EmbeddingVector(settings.default_embedding_dimensions), nullable=False
    )
    created_at = Column(DateTime(timezone=True), server_default=func.now())
