"""Profile model — reusable embedding/LLM configuration."""

import uuid
from datetime import datetime, timezone

from sqlalchemy import Column, DateTime, Enum, ForeignKey, Integer, String, func
from sqlalchemy.dialects.postgresql import UUID as UUIDType
from sqlalchemy.orm import relationship

from graph_core.database import Base


class Profile(Base):
    __tablename__ = "profiles"

    id = Column(UUIDType(as_uuid=True), primary_key=True, default=uuid.uuid4)
    namespace_id = Column(UUIDType(as_uuid=True), ForeignKey("namespaces.id", ondelete="CASCADE"), nullable=False, index=True)
    credential_id = Column(UUIDType(as_uuid=True), ForeignKey("credentials.id", ondelete="SET NULL"), nullable=True)

    kind = Column(Enum("embedding", "llm", name="profile_kind", create_type=True), nullable=False)
    provider = Column(String(64), nullable=False)
    model = Column(String(128), nullable=False)
    label = Column(String(128), nullable=True)

    # Optional custom API base URL (overrides credential base_url)
    base_url = Column(String(512), nullable=True)

    # Embedding-specific fields
    dimensions = Column(Integer, nullable=True)
    distance_metric = Column(String(32), nullable=True)

    created_at = Column(DateTime(timezone=True), server_default=func.now())

    namespace = relationship("Namespace", back_populates="profiles")
    credential = relationship("Credential", back_populates="profiles")

    def __repr__(self) -> str:
        return f"<Profile {self.kind}/{self.model}>"
