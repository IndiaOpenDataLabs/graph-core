"""Versioned incremental-ingestion and graph-analytics persistence models."""

import uuid

from sqlalchemy import (
    JSON,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import UUID as UUIDType  # noqa: N811

from graph_core.database import Base


class GraphChunkSegment(Base):
    __tablename__ = "graph_chunk_segments"

    id = Column(UUIDType(as_uuid=True), primary_key=True, default=uuid.uuid4)
    collection_id = Column(
        UUIDType(as_uuid=True),
        ForeignKey("collections.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    document_id = Column(UUIDType(as_uuid=True), nullable=True, index=True)
    document_path = Column(String(1024), nullable=True)
    chunk_hash = Column(String(64), nullable=False)
    chunk_index = Column(Integer, nullable=False)
    contract_version = Column(String(128), nullable=False)
    status = Column(String(32), nullable=False, default="extracted")
    raw_extraction = Column(JSON, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    completed_at = Column(DateTime(timezone=True), nullable=True)
    tombstoned_at = Column(DateTime(timezone=True), nullable=True)

    __table_args__ = (
        UniqueConstraint(
            "collection_id",
            "chunk_hash",
            "contract_version",
            name="uq_graph_chunk_segment_contract",
        ),
        Index(
            "ix_graph_chunk_segments_document_order",
            "collection_id",
            "document_id",
            "chunk_index",
        ),
    )


class GraphEntityMapping(Base):
    __tablename__ = "graph_entity_mappings"

    id = Column(UUIDType(as_uuid=True), primary_key=True, default=uuid.uuid4)
    collection_id = Column(
        UUIDType(as_uuid=True),
        ForeignKey("collections.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    segment_id = Column(
        UUIDType(as_uuid=True),
        ForeignKey("graph_chunk_segments.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    local_key = Column(String(256), nullable=False)
    raw_name = Column(String(256), nullable=False)
    raw_type = Column(String(64), nullable=True)
    canonical_entity_id = Column(
        UUIDType(as_uuid=True),
        ForeignKey("graph_entities.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    resolution_method = Column(String(32), nullable=False)
    confidence = Column(Float, nullable=False, default=1.0)
    mapping_version = Column(Integer, nullable=False, default=1)

    __table_args__ = (
        UniqueConstraint(
            "segment_id", "local_key", name="uq_graph_entity_mapping_local"
        ),
    )


class GraphPredicateMapping(Base):
    __tablename__ = "graph_predicate_mappings"

    id = Column(UUIDType(as_uuid=True), primary_key=True, default=uuid.uuid4)
    collection_id = Column(
        UUIDType(as_uuid=True),
        ForeignKey("collections.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    segment_id = Column(
        UUIDType(as_uuid=True),
        ForeignKey("graph_chunk_segments.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    raw_predicate = Column(String(128), nullable=False)
    canonical_predicate_id = Column(
        UUIDType(as_uuid=True),
        ForeignKey("graph_relationship_types.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    resolution_method = Column(String(32), nullable=False)
    confidence = Column(Float, nullable=False, default=1.0)
    mapping_version = Column(Integer, nullable=False, default=1)
    inferred_properties = Column(JSON, nullable=True)

    __table_args__ = (
        UniqueConstraint(
            "segment_id",
            "raw_predicate",
            name="uq_graph_predicate_mapping_local",
        ),
    )


class GraphEntityRedirect(Base):
    __tablename__ = "graph_entity_redirects"

    id = Column(UUIDType(as_uuid=True), primary_key=True, default=uuid.uuid4)
    collection_id = Column(
        UUIDType(as_uuid=True),
        ForeignKey("collections.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    source_entity_id = Column(
        UUIDType(as_uuid=True),
        ForeignKey("graph_entities.id", ondelete="CASCADE"),
        nullable=False,
        unique=True,
    )
    target_entity_id = Column(
        UUIDType(as_uuid=True),
        ForeignKey("graph_entities.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    reason = Column(String(128), nullable=True)
    mapping_version = Column(Integer, nullable=False, default=1)
    created_at = Column(DateTime(timezone=True), server_default=func.now())


class GraphChunkContribution(Base):
    __tablename__ = "graph_chunk_contributions"

    id = Column(UUIDType(as_uuid=True), primary_key=True, default=uuid.uuid4)
    collection_id = Column(
        UUIDType(as_uuid=True),
        ForeignKey("collections.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    segment_id = Column(
        UUIDType(as_uuid=True),
        ForeignKey("graph_chunk_segments.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    object_kind = Column(String(32), nullable=False)
    object_id = Column(UUIDType(as_uuid=True), nullable=False)
    contribution_kind = Column(String(32), nullable=False)
    weight = Column(Float, nullable=False, default=1.0)
    metadata_json = Column(JSON, nullable=True)

    __table_args__ = (
        UniqueConstraint(
            "segment_id",
            "object_kind",
            "object_id",
            "contribution_kind",
            name="uq_graph_chunk_contribution_object",
        ),
        Index(
            "ix_graph_chunk_contributions_reverse",
            "collection_id",
            "object_kind",
            "object_id",
        ),
    )


class GraphDerivedDependency(Base):
    __tablename__ = "graph_derived_dependencies"

    id = Column(UUIDType(as_uuid=True), primary_key=True, default=uuid.uuid4)
    collection_id = Column(
        UUIDType(as_uuid=True),
        ForeignKey("collections.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    source_kind = Column(String(32), nullable=False)
    source_id = Column(UUIDType(as_uuid=True), nullable=False)
    target_kind = Column(String(32), nullable=False)
    target_id = Column(UUIDType(as_uuid=True), nullable=False)
    dependency_type = Column(String(32), nullable=False)
    overlay_version = Column(Integer, nullable=False, default=1)

    __table_args__ = (
        UniqueConstraint(
            "collection_id",
            "source_kind",
            "source_id",
            "target_kind",
            "target_id",
            "dependency_type",
            "overlay_version",
            name="uq_graph_derived_dependency",
        ),
        Index(
            "ix_graph_derived_dependencies_target",
            "collection_id",
            "target_kind",
            "target_id",
        ),
    )


class GraphVersion(Base):
    __tablename__ = "graph_versions"

    id = Column(UUIDType(as_uuid=True), primary_key=True, default=uuid.uuid4)
    collection_id = Column(
        UUIDType(as_uuid=True),
        ForeignKey("collections.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    version = Column(Integer, nullable=False)
    parent_version_id = Column(
        UUIDType(as_uuid=True),
        ForeignKey("graph_versions.id", ondelete="SET NULL"),
        nullable=True,
    )
    delta_segment_id = Column(
        UUIDType(as_uuid=True),
        ForeignKey("graph_chunk_segments.id", ondelete="SET NULL"),
        nullable=True,
    )
    status = Column(String(32), nullable=False, default="ready")
    manifest = Column(JSON, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    published_at = Column(DateTime(timezone=True), nullable=True)

    __table_args__ = (
        UniqueConstraint(
            "collection_id", "version", name="uq_graph_version_collection"
        ),
    )


class GraphProjectionSnapshot(Base):
    __tablename__ = "graph_projection_snapshots"

    id = Column(UUIDType(as_uuid=True), primary_key=True, default=uuid.uuid4)
    graph_version_id = Column(
        UUIDType(as_uuid=True),
        ForeignKey("graph_versions.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    name = Column(String(128), nullable=False)
    spec_hash = Column(String(64), nullable=False)
    specification = Column(JSON, nullable=False)
    status = Column(String(32), nullable=False, default="building")
    node_count = Column(Integer, nullable=False, default=0)
    edge_count = Column(Integer, nullable=False, default=0)
    diagnostics = Column(JSON, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    completed_at = Column(DateTime(timezone=True), nullable=True)

    __table_args__ = (
        UniqueConstraint(
            "graph_version_id",
            "name",
            "spec_hash",
            name="uq_graph_projection_snapshot_spec",
        ),
    )


class GraphNodeMetric(Base):
    __tablename__ = "graph_node_metrics"

    id = Column(UUIDType(as_uuid=True), primary_key=True, default=uuid.uuid4)
    projection_id = Column(
        UUIDType(as_uuid=True),
        ForeignKey("graph_projection_snapshots.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    entity_id = Column(
        UUIDType(as_uuid=True),
        ForeignKey("graph_entities.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    metric = Column(String(64), nullable=False)
    value = Column(Float, nullable=False)
    rank = Column(Integer, nullable=True)
    metadata_json = Column(JSON, nullable=True)

    __table_args__ = (
        UniqueConstraint(
            "projection_id", "entity_id", "metric", name="uq_graph_node_metric"
        ),
    )


class GraphCommunity(Base):
    __tablename__ = "graph_communities"

    id = Column(UUIDType(as_uuid=True), primary_key=True, default=uuid.uuid4)
    projection_id = Column(
        UUIDType(as_uuid=True),
        ForeignKey("graph_projection_snapshots.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    algorithm = Column(String(64), nullable=False)
    community_key = Column(String(128), nullable=False)
    score = Column(Float, nullable=True)
    summary = Column(Text, nullable=True)
    metadata_json = Column(JSON, nullable=True)

    __table_args__ = (
        UniqueConstraint(
            "projection_id",
            "algorithm",
            "community_key",
            name="uq_graph_community_key",
        ),
    )


class GraphCommunityMembership(Base):
    __tablename__ = "graph_community_memberships"

    id = Column(UUIDType(as_uuid=True), primary_key=True, default=uuid.uuid4)
    community_id = Column(
        UUIDType(as_uuid=True),
        ForeignKey("graph_communities.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    entity_id = Column(
        UUIDType(as_uuid=True),
        ForeignKey("graph_entities.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    strength = Column(Float, nullable=False, default=1.0)
    metadata_json = Column(JSON, nullable=True)

    __table_args__ = (
        UniqueConstraint(
            "community_id",
            "entity_id",
            name="uq_graph_community_membership",
        ),
    )


class GraphComponentMetric(Base):
    __tablename__ = "graph_component_metrics"

    id = Column(UUIDType(as_uuid=True), primary_key=True, default=uuid.uuid4)
    projection_id = Column(
        UUIDType(as_uuid=True),
        ForeignKey("graph_projection_snapshots.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    component_key = Column(String(128), nullable=False)
    metric = Column(String(64), nullable=False)
    value = Column(Float, nullable=False)
    metadata_json = Column(JSON, nullable=True)

    __table_args__ = (
        UniqueConstraint(
            "projection_id",
            "component_key",
            "metric",
            name="uq_graph_component_metric",
        ),
    )
