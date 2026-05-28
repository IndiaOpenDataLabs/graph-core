"""Registered app and user-link models — multi-tenant app registration."""

import uuid

from sqlalchemy import Boolean, Column, DateTime, ForeignKey, String, func
from sqlalchemy.dialects.postgresql import UUID as UUIDType
from sqlalchemy.orm import relationship

from graph_core.database import Base


class RegisteredApp(Base):
    """A registered consuming application in multi-tenant mode."""

    __tablename__ = "registered_apps"

    id = Column(UUIDType(as_uuid=True), primary_key=True, default=uuid.uuid4)
    client_id = Column(String(64), unique=True, nullable=False, index=True)
    client_secret_hash = Column(String(128), nullable=False)
    name = Column(String(128), nullable=False)
    owner_email = Column(String(256), nullable=True)
    active = Column(Boolean, nullable=False, default=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    user_links = relationship("AppUserLink", back_populates="app", cascade="all, delete-orphan")

    def __repr__(self) -> str:
        return f"<RegisteredApp {self.name} ({self.client_id})>"


class AppUserLink(Base):
    """Maps (app, user) → default namespace. Enables portable user workspaces."""

    __tablename__ = "app_user_links"

    app_id = Column(UUIDType(as_uuid=True), ForeignKey("registered_apps.id", ondelete="CASCADE"), primary_key=True)
    user_sub = Column(String(256), primary_key=True)
    namespace_id = Column(UUIDType(as_uuid=True), ForeignKey("namespaces.id", ondelete="CASCADE"), nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    app = relationship("RegisteredApp", back_populates="user_links")
    namespace = relationship("Namespace")

    def __repr__(self) -> str:
        return f"<AppUserLink {self.user_sub} → {self.namespace_id}>"
