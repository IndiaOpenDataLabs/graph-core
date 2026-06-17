"""Ingestion chunks — tracks individual chunk status within a document job."""

import uuid
from datetime import datetime, timezone

from sqlalchemy import (
    Column,
    DateTime,
    Enum,
    ForeignKey,
    Integer,
    String,
    Text,
    func,
)
from sqlalchemy.dialects.postgresql import UUID as UUIDType
from sqlalchemy.orm import relationship

from graph_core.database import Base


class IngestionChunk(Base):
    __tablename__ = "ingestion_chunks"

    id = Column(UUIDType(as_uuid=True), primary_key=True, default=uuid.uuid4)
    job_id = Column(
        UUIDType(as_uuid=True),
        ForeignKey("jobs.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    chunk_index = Column(Integer, nullable=False)
    text = Column(Text, nullable=False)
    document_id = Column(UUIDType(as_uuid=True), nullable=True, index=True)
    document_path = Column(String(1024), nullable=True)
    status = Column(
        Enum("pending", "processing", "completed", "failed", name="chunk_status", create_type=True),
        nullable=False,
        default="pending",
    )
    error = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    completed_at = Column(DateTime(timezone=True), nullable=True)

    job = relationship("Job", back_populates="chunks")

    __table_args__ = (
        __import__("sqlalchemy").UniqueConstraint(
            "job_id", "chunk_index", name="uq_ingestion_chunks_job_index"
        ),
    )

    def __repr__(self) -> str:
        return f"<IngestionChunk job={self.job_id} index={self.chunk_index} {self.status}>"
