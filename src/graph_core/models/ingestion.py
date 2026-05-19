"""Ingestion record — ledger of what text was ingested and how."""

import uuid
from datetime import datetime, timezone

from sqlalchemy import Column, DateTime, Enum, ForeignKey, Integer, JSON, String, func
from sqlalchemy.dialects.postgresql import UUID as UUIDType
from sqlalchemy.orm import relationship

from graph_core.database import Base


class IngestionRecord(Base):
    __tablename__ = "ingestion_records"

    id = Column(UUIDType(as_uuid=True), primary_key=True, default=uuid.uuid4)
    collection_id = Column(UUIDType(as_uuid=True), ForeignKey("collections.id", ondelete="CASCADE"), nullable=False, index=True)
    chunk_hash = Column(String(64), nullable=False, index=True)  # SHA-256

    strategy = Column(Enum("vector", "custom_graph_rag", "light_rag", name="ingest_strategy", create_type=True), nullable=False)
    extraction_model = Column(String(128), nullable=True)
    embedding_model = Column(String(128), nullable=True)

    entity_count = Column(Integer, default=0)
    relationship_count = Column(Integer, default=0)
    sanitization_flags = Column(JSON, nullable=True)

    ingested_at = Column(DateTime(timezone=True), server_default=func.now())
    source_document_id = Column(UUIDType(as_uuid=True), nullable=True)  # Parent document if from doc ingestion

    collection = relationship("Collection", back_populates="ingestion_records")

    def __repr__(self) -> str:
        return f"<IngestionRecord {self.chunk_hash[:12]}>"
