"""Namespace model — top-level isolation boundary."""

import uuid
from datetime import datetime, timezone

from sqlalchemy import Column, DateTime, String, func
from sqlalchemy.dialects.postgresql import UUID as UUIDType
from sqlalchemy.orm import relationship

from graph_core.database import Base


class Namespace(Base):
    __tablename__ = "namespaces"

    id = Column(UUIDType(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String(128), unique=True, nullable=False, index=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    collections = relationship("Collection", back_populates="namespace", cascade="all, delete-orphan")
    credentials = relationship("Credential", back_populates="namespace", cascade="all, delete-orphan")
    profiles = relationship("Profile", back_populates="namespace", cascade="all, delete-orphan")
    jobs = relationship("Job", back_populates="namespace")

    def __repr__(self) -> str:
        return f"<Namespace {self.name}>"
