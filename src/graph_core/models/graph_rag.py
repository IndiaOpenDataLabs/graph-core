"""Graph entity and related models — canonical records for Custom Graph RAG."""

import uuid
from datetime import datetime, timezone

from sqlalchemy import (
    Column,
    DateTime,
    Enum,
    ForeignKey,
    Index,
    Integer,
    JSON,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import UUID as UUIDType
from sqlalchemy.orm import relationship

from graph_core.database import Base


class GraphEntity(Base):
    __tablename__ = "graph_entities"

    id = Column(UUIDType(as_uuid=True), primary_key=True, default=uuid.uuid4)
    collection_id = Column(
        UUIDType(as_uuid=True),
        ForeignKey("collections.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    canonical_name = Column(String(256), nullable=False, index=True)
    primary_type = Column(String(64), nullable=True)
    description_count = Column(Integer, default=0)

    created_at = Column(DateTime(timezone=True), server_default=func.now())

    collection = relationship("Collection", back_populates="graph_entities")
    descriptions = relationship(
        "EntityDescription", back_populates="entity", cascade="all, delete-orphan"
    )
    aliases = relationship(
        "EntityAlias", back_populates="entity", cascade="all, delete-orphan"
    )
    types = relationship(
        "EntityType", back_populates="entity", cascade="all, delete-orphan"
    )
    source_relationships = relationship(
        "GraphRelationship", foreign_keys="GraphRelationship.source_entity_id", back_populates="source_entity"
    )
    target_relationships = relationship(
        "GraphRelationship", foreign_keys="GraphRelationship.target_entity_id", back_populates="target_entity"
    )

    __table_args__ = (
        UniqueConstraint(
            "canonical_name",
            "collection_id",
            name="uq_graph_entities_canonical_name_collection_id",
        ),
    )

    def __repr__(self) -> str:
        return f"<GraphEntity {self.canonical_name}>"


class EntityDescription(Base):
    __tablename__ = "entity_descriptions"

    id = Column(UUIDType(as_uuid=True), primary_key=True, default=uuid.uuid4)
    entity_id = Column(
        UUIDType(as_uuid=True),
        ForeignKey("graph_entities.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    description = Column(Text, nullable=False)
    weight = Column(Integer, default=1)
    source_chunk_hashes = Column(JSON, nullable=True)
    document_id = Column(UUIDType(as_uuid=True), nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    entity = relationship("GraphEntity", back_populates="descriptions")

    def __repr__(self) -> str:
        return f"<EntityDescription {self.id}>"


class EntityAlias(Base):
    __tablename__ = "entity_aliases"

    id = Column(UUIDType(as_uuid=True), primary_key=True, default=uuid.uuid4)
    collection_id = Column(
        UUIDType(as_uuid=True),
        ForeignKey("collections.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    entity_id = Column(
        UUIDType(as_uuid=True),
        ForeignKey("graph_entities.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    alias_name = Column(String(256), nullable=False, index=True)
    source_chunk_hash = Column(String(64), nullable=True)
    document_id = Column(UUIDType(as_uuid=True), nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    entity = relationship("GraphEntity", back_populates="aliases")

    __table_args__ = (
        UniqueConstraint(
            "collection_id",
            "alias_name",
            name="uq_entity_aliases_collection_alias_name",
        ),
    )

    def __repr__(self) -> str:
        return f"<EntityAlias {self.alias_name}>"


class RelationshipTypeAlias(Base):
    __tablename__ = "relationship_type_aliases"

    id = Column(UUIDType(as_uuid=True), primary_key=True, default=uuid.uuid4)
    collection_id = Column(
        UUIDType(as_uuid=True),
        ForeignKey("collections.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    canonical_type = Column(String(64), nullable=False, index=True)
    alias_type = Column(String(64), nullable=False, index=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    __table_args__ = (
        UniqueConstraint(
            "collection_id",
            "alias_type",
            name="uq_relationship_type_aliases_collection_alias_type",
        ),
        Index(
            "ix_relationship_type_aliases_canonical",
            "collection_id",
            "canonical_type",
        ),
    )

    def __repr__(self) -> str:
        return f"<RelationshipTypeAlias {self.alias_type} -> {self.canonical_type}>"


class EntityType(Base):
    __tablename__ = "entity_types"

    id = Column(UUIDType(as_uuid=True), primary_key=True, default=uuid.uuid4)
    entity_id = Column(
        UUIDType(as_uuid=True),
        ForeignKey("graph_entities.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    type_name = Column(String(64), nullable=False)
    frequency = Column(Integer, default=1)

    entity = relationship("GraphEntity", back_populates="types")

    __table_args__ = (
        UniqueConstraint(
            "entity_id",
            "type_name",
            name="uq_entity_types_entity_type",
        ),
    )

    def __repr__(self) -> str:
        return f"<EntityType {self.type_name}>"


class GraphRelationship(Base):
    __tablename__ = "graph_relationships"

    id = Column(UUIDType(as_uuid=True), primary_key=True, default=uuid.uuid4)
    source_entity_id = Column(
        UUIDType(as_uuid=True),
        ForeignKey("graph_entities.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    target_entity_id = Column(
        UUIDType(as_uuid=True),
        ForeignKey("graph_entities.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    weight = Column(Integer, default=1)
    keywords = Column(JSON, nullable=True)
    rel_type = Column(String(64), nullable=False, default="RELATES_TO", index=True)
    collection_id = Column(
        UUIDType(as_uuid=True),
        ForeignKey("collections.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    source_entity = relationship(
        "GraphEntity", foreign_keys=[source_entity_id], back_populates="source_relationships"
    )
    target_entity = relationship(
        "GraphEntity", foreign_keys=[target_entity_id], back_populates="target_relationships"
    )
    descriptions = relationship(
        "RelationshipDescription",
        back_populates="relationship",
        cascade="all, delete-orphan",
    )

    __table_args__ = (
        Index("ix_graph_relationships_source_target", "source_entity_id", "target_entity_id"),
        Index(
            "ix_graph_relationships_source_target_type",
            "source_entity_id",
            "target_entity_id",
            "rel_type",
        ),
    )

    def __repr__(self) -> str:
        return (
            f"<GraphRelationship {self.source_entity_id} "
            f"-[{self.rel_type}]-> {self.target_entity_id}>"
        )


class RelationshipDescription(Base):
    __tablename__ = "relationship_descriptions"

    id = Column(UUIDType(as_uuid=True), primary_key=True, default=uuid.uuid4)
    relationship_id = Column(
        UUIDType(as_uuid=True),
        ForeignKey("graph_relationships.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    description = Column(Text, nullable=False)
    keywords = Column(JSON, nullable=True)
    weight = Column(Integer, default=1)
    source_chunk_hashes = Column(JSON, nullable=True)
    document_id = Column(UUIDType(as_uuid=True), nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    relationship = relationship("GraphRelationship", back_populates="descriptions")

    def __repr__(self) -> str:
        return f"<RelationshipDescription {self.id}>"


class RawChunkExtraction(Base):
    __tablename__ = "raw_chunk_extractions"

    id = Column(UUIDType(as_uuid=True), primary_key=True, default=uuid.uuid4)
    chunk_content_hash = Column(String(64), nullable=False, index=True)
    collection_id = Column(
        UUIDType(as_uuid=True),
        ForeignKey("collections.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    document_id = Column(UUIDType(as_uuid=True), nullable=True)
    entities_json = Column(JSON, nullable=True)
    relationships_json = Column(JSON, nullable=True)
    extraction_model = Column(String(128), nullable=True)
    gleaning_passes = Column(Integer, default=0)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    __table_args__ = (
        UniqueConstraint(
            "chunk_content_hash",
            "collection_id",
            name="uq_raw_chunk_extractions_hash_collection",
        ),
    )

    def __repr__(self) -> str:
        return f"<RawChunkExtraction {self.chunk_content_hash[:12]}>"
