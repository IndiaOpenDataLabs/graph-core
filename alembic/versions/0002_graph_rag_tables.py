"""graph rag tables

Revision ID: 0002_graph_rag_tables
Revises: 0001_initial_platform_schema
Create Date: 2026-05-19
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision = "0002_graph_rag_tables"
down_revision = "0001_initial_platform_schema"
branch_labels = None
depends_on = None


chunk_status = postgresql.ENUM(
    "pending",
    "processing",
    "completed",
    "failed",
    name="chunk_status",
    create_type=False,
)


def upgrade() -> None:
    # Create chunk_status enum
    chunk_status.create(op.get_bind(), checkfirst=True)

    # Ingestion chunks
    op.create_table(
        "ingestion_chunks",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True, server_default=sa.text("gen_random_uuid()")),
        sa.Column("job_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("jobs.id", ondelete="CASCADE"), nullable=False, index=True),
        sa.Column("chunk_index", sa.Integer, nullable=False),
        sa.Column("text", sa.Text, nullable=False),
        sa.Column("status", chunk_status, nullable=False, server_default="pending"),
        sa.Column("error", sa.Text, nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.UniqueConstraint("job_id", "chunk_index", name="uq_ingestion_chunks_job_index"),
    )

    # Job chunk counters
    with op.batch_alter_table("jobs") as batch_op:
        batch_op.add_column(sa.Column("chunks_total", sa.Integer, server_default="0", nullable=True))
        batch_op.add_column(sa.Column("chunks_completed", sa.Integer, server_default="0", nullable=True))

    # Graph entities
    op.create_table(
        "graph_entities",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True, server_default=sa.text("gen_random_uuid()")),
        sa.Column("collection_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("collections.id", ondelete="CASCADE"), nullable=False, index=True),
        sa.Column("canonical_name", sa.String(256), nullable=False, index=True),
        sa.Column("primary_type", sa.String(64), nullable=True),
        sa.Column("description_count", sa.Integer, server_default="0"),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.UniqueConstraint("canonical_name", "collection_id", name="uq_graph_entities_canonical_name_collection_id"),
    )

    # Entity descriptions
    op.create_table(
        "entity_descriptions",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True, server_default=sa.text("gen_random_uuid()")),
        sa.Column("entity_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("graph_entities.id", ondelete="CASCADE"), nullable=False, index=True),
        sa.Column("description", sa.Text, nullable=False),
        sa.Column("weight", sa.Integer, server_default="1"),
        sa.Column("source_chunk_hashes", postgresql.JSON, nullable=True),
        sa.Column("document_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )

    # Entity aliases
    op.create_table(
        "entity_aliases",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True, server_default=sa.text("gen_random_uuid()")),
        sa.Column("entity_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("graph_entities.id", ondelete="CASCADE"), nullable=False, index=True),
        sa.Column("alias_name", sa.String(256), nullable=False, index=True),
        sa.Column("source_chunk_hash", sa.String(64), nullable=True),
        sa.Column("document_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.UniqueConstraint("alias_name", name="uq_entity_aliases_alias_name"),
    )

    # Entity types
    op.create_table(
        "entity_types",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True, server_default=sa.text("gen_random_uuid()")),
        sa.Column("entity_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("graph_entities.id", ondelete="CASCADE"), nullable=False, index=True),
        sa.Column("type_name", sa.String(64), nullable=False),
        sa.Column("frequency", sa.Integer, server_default="1"),
        sa.UniqueConstraint("entity_id", "type_name", name="uq_entity_types_entity_type"),
    )

    # Graph relationships
    op.create_table(
        "graph_relationships",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True, server_default=sa.text("gen_random_uuid()")),
        sa.Column("source_entity_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("graph_entities.id", ondelete="CASCADE"), nullable=False, index=True),
        sa.Column("target_entity_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("graph_entities.id", ondelete="CASCADE"), nullable=False, index=True),
        sa.Column("weight", sa.Integer, server_default="1"),
        sa.Column("keywords", postgresql.JSON, nullable=True),
        sa.Column("collection_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("collections.id", ondelete="CASCADE"), nullable=False, index=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.create_index("ix_graph_relationships_source_target", "graph_relationships", ["source_entity_id", "target_entity_id"])

    # Relationship descriptions
    op.create_table(
        "relationship_descriptions",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True, server_default=sa.text("gen_random_uuid()")),
        sa.Column("relationship_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("graph_relationships.id", ondelete="CASCADE"), nullable=False, index=True),
        sa.Column("description", sa.Text, nullable=False),
        sa.Column("keywords", postgresql.JSON, nullable=True),
        sa.Column("weight", sa.Integer, server_default="1"),
        sa.Column("source_chunk_hashes", postgresql.JSON, nullable=True),
        sa.Column("document_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )

    # Raw chunk extractions
    op.create_table(
        "raw_chunk_extractions",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True, server_default=sa.text("gen_random_uuid()")),
        sa.Column("chunk_content_hash", sa.String(64), nullable=False, index=True),
        sa.Column("collection_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("collections.id", ondelete="CASCADE"), nullable=False, index=True),
        sa.Column("document_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("entities_json", postgresql.JSON, nullable=True),
        sa.Column("relationships_json", postgresql.JSON, nullable=True),
        sa.Column("extraction_model", sa.String(128), nullable=True),
        sa.Column("gleaning_passes", sa.Integer, server_default="0"),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.UniqueConstraint("chunk_content_hash", "collection_id", name="uq_raw_chunk_extractions_hash_collection"),
    )


def downgrade() -> None:
    op.drop_table("raw_chunk_extractions")
    op.drop_table("relationship_descriptions")
    op.drop_index("ix_graph_relationships_source_target", table_name="graph_relationships")
    op.drop_table("graph_relationships")
    op.drop_table("entity_types")
    op.drop_table("entity_aliases")
    op.drop_table("entity_descriptions")
    op.drop_table("graph_entities")

    with op.batch_alter_table("jobs") as batch_op:
        batch_op.drop_column("chunks_completed")
        batch_op.drop_column("chunks_total")

    op.drop_table("ingestion_chunks")
    chunk_status.drop(op.get_bind(), checkfirst=True)
