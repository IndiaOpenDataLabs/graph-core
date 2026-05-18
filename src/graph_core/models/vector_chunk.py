"""Vector chunk storage for the vector retrieval strategy."""

import uuid

from sqlalchemy import Column, DateTime, ForeignKey, Integer, JSON, String, Text, UniqueConstraint, func
from sqlalchemy.dialects.postgresql import UUID as UUIDType

from graph_core.config import settings
from graph_core.database import Base
from graph_core.storage.vector_types import EmbeddingVector


class VectorChunk(Base):
    __tablename__ = "vector_chunks"

    id = Column(UUIDType(as_uuid=True), primary_key=True, default=uuid.uuid4)
    namespace_id = Column(UUIDType(as_uuid=True), ForeignKey("namespaces.id", ondelete="CASCADE"), nullable=False, index=True)
    collection_id = Column(UUIDType(as_uuid=True), ForeignKey("collections.id", ondelete="CASCADE"), nullable=False, index=True)
    chunk_hash = Column(String(64), nullable=False, index=True)
    chunk_index = Column(Integer, nullable=False, default=0)
    content = Column(Text, nullable=False)
    token_count = Column(Integer, nullable=False, default=0)
    metadata_json = Column(JSON, nullable=True)
    embedding = Column(EmbeddingVector(settings.default_embedding_dimensions), nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    __table_args__ = (
        UniqueConstraint("collection_id", "chunk_hash", "chunk_index", name="uq_vector_chunk_identity"),
    )

    def __repr__(self) -> str:
        return f"<VectorChunk {self.collection_id}#{self.chunk_index}>"
