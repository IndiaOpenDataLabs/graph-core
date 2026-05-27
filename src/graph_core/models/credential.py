"""Credential model — encrypted secret references."""

import uuid
from datetime import datetime, timezone

from sqlalchemy import Column, DateTime, ForeignKey, String, func
from sqlalchemy.dialects.postgresql import UUID as UUIDType
from sqlalchemy.orm import relationship

from graph_core.database import Base


class Credential(Base):
    __tablename__ = "credentials"

    id = Column(UUIDType(as_uuid=True), primary_key=True, default=uuid.uuid4)
    namespace_id = Column(UUIDType(as_uuid=True), ForeignKey("namespaces.id", ondelete="CASCADE"), nullable=False, index=True)
    provider = Column(String(64), nullable=False, index=True)
    label = Column(String(128), nullable=True)

    # Encrypted secret — never stored plaintext
    encrypted_secret = Column(String(1024), nullable=False)

    # Optional custom API base URL (e.g. for local OpenAI-compatible servers)
    base_url = Column(String(512), nullable=True)

    created_at = Column(DateTime(timezone=True), server_default=func.now())

    namespace = relationship("Namespace", back_populates="credentials")
    profiles = relationship("Profile", back_populates="credential")

    __table_args__ = (
        __import__("sqlalchemy").UniqueConstraint("namespace_id", "label", name="uq_namespace_credential_label"),
    )

    def __repr__(self) -> str:
        return f"<Credential {self.label or self.id} ({self.provider})>"
