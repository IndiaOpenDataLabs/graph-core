"""Job and JobEvent models — durable async execution state."""

import uuid
from datetime import datetime, timezone

from sqlalchemy import Column, DateTime, Enum, ForeignKey, Integer, JSON, String, Text, func
from sqlalchemy.dialects.postgresql import UUID as UUIDType
from sqlalchemy.orm import relationship

from graph_core.database import Base


class Job(Base):
    __tablename__ = "jobs"

    id = Column(UUIDType(as_uuid=True), primary_key=True, default=uuid.uuid4)
    namespace_id = Column(UUIDType(as_uuid=True), ForeignKey("namespaces.id", ondelete="CASCADE"), nullable=False, index=True)
    collection_id = Column(UUIDType(as_uuid=True), ForeignKey("collections.id", ondelete="CASCADE"), nullable=True, index=True)
    document_id = Column(UUIDType(as_uuid=True), nullable=True, index=True)
    document_path = Column(String(1024), nullable=True)

    job_type = Column(
        Enum(
            "ingest_chunk",
            "ingest_document",
            "delete_collection",
            "reindex",
            "query",
            "enhance",
            name="job_type",
            create_type=True,
        ),
        nullable=False,
    )
    status = Column(
        Enum("pending", "running", "completed", "failed", "cancelled", name="job_status", create_type=True),
        nullable=False, default="pending",
    )
    progress_percent = Column(Integer, default=0)
    error = Column(Text, nullable=True)
    payload = Column(JSON, nullable=True)
    chunks_total = Column(Integer, default=0)
    chunks_completed = Column(Integer, default=0)

    created_at = Column(DateTime(timezone=True), server_default=func.now())
    started_at = Column(DateTime(timezone=True), nullable=True)
    completed_at = Column(DateTime(timezone=True), nullable=True)

    namespace = relationship("Namespace", back_populates="jobs")
    collection = relationship("Collection", back_populates="jobs")
    events = relationship("JobEvent", back_populates="job", cascade="all, delete-orphan")
    chunks = relationship("IngestionChunk", back_populates="job", cascade="all, delete-orphan")

    def __repr__(self) -> str:
        return f"<Job {self.job_type} {self.status}>"


class JobEvent(Base):
    __tablename__ = "job_events"

    id = Column(UUIDType(as_uuid=True), primary_key=True, default=uuid.uuid4)
    job_id = Column(UUIDType(as_uuid=True), ForeignKey("jobs.id", ondelete="CASCADE"), nullable=False, index=True)
    timestamp = Column(DateTime(timezone=True), server_default=func.now())
    event_type = Column(String(64), nullable=False)
    payload = Column(JSON, nullable=True)

    job = relationship("Job", back_populates="events")

    def __repr__(self) -> str:
        return f"<JobEvent {self.event_type}>"
