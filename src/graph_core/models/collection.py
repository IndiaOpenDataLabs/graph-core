"""Collection model — knowledge graph scoped to a namespace."""

import uuid
from datetime import datetime, timezone

from sqlalchemy import Column, DateTime, Enum, ForeignKey, String, func
from sqlalchemy.dialects.postgresql import UUID as UUIDType
from sqlalchemy.orm import relationship

from graph_core.database import Base


class Collection(Base):
    __tablename__ = "collections"

    id = Column(UUIDType(as_uuid=True), primary_key=True, default=uuid.uuid4)
    namespace_id = Column(UUIDType(as_uuid=True), ForeignKey("namespaces.id", ondelete="CASCADE"), nullable=False, index=True)
    name = Column(String(255), nullable=False, index=True)
    description = Column(String(1024), nullable=True)

    # Strategy: immutable after creation
    strategy = Column(Enum("vector", "light_rag", "custom_graph_rag", name="rag_strategy", create_type=True), nullable=False)
    default_query_mode = Column(String(64), nullable=True)

    # Embedding profile: immutable after creation
    embedding_profile_id = Column(UUIDType(as_uuid=True), ForeignKey("profiles.id", ondelete="SET NULL"), nullable=True, index=True)

    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    namespace = relationship("Namespace", back_populates="collections")
    embedding_profile = relationship("Profile", foreign_keys=[embedding_profile_id])
    ingestion_records = relationship("IngestionRecord", back_populates="collection", cascade="all, delete-orphan")
    jobs = relationship("Job", back_populates="collection", cascade="all, delete-orphan")
    graph_entities = relationship("GraphEntity", back_populates="collection", cascade="all, delete-orphan")

    __table_args__ = (
        # Unique name per namespace
        __import__("sqlalchemy").UniqueConstraint("namespace_id", "name", name="uq_namespace_collection_name"),
    )

    def __repr__(self) -> str:
        return f"<Collection {self.name} (ns={self.namespace_id})>"
