"""graph rag vector tables

Revision ID: 0003_graph_rag_vector_tables
Revises: 0002_graph_rag_tables
Create Date: 2026-05-19
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision = "0003_graph_rag_vector_tables"
down_revision = "0002_graph_rag_tables"
branch_labels = None
depends_on = None

# Read dimension from settings at migration time
from graph_core.config import settings

DIM = settings.default_embedding_dimensions


def upgrade() -> None:
    op.create_table(
        "graph_entity_embeddings",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True, server_default=sa.text("gen_random_uuid()")),
        sa.Column("entity_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("graph_entities.id", ondelete="CASCADE"), nullable=False, index=True),
        sa.Column("description_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("collection_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("collections.id", ondelete="CASCADE"), nullable=False, index=True),
        sa.Column("name", sa.String(256), nullable=False),
        sa.Column("description", sa.Text, nullable=False),
        sa.Column("embedding", sa.Text, nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    # Cast text column to vector type
    op.execute(f"ALTER TABLE graph_entity_embeddings ALTER COLUMN embedding TYPE vector({DIM}) USING embedding::vector({DIM})")

    op.create_table(
        "graph_relationship_embeddings",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True, server_default=sa.text("gen_random_uuid()")),
        sa.Column("relationship_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("graph_relationships.id", ondelete="CASCADE"), nullable=False, index=True),
        sa.Column("collection_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("collections.id", ondelete="CASCADE"), nullable=False, index=True),
        sa.Column("source_name", sa.String(256), nullable=False),
        sa.Column("target_name", sa.String(256), nullable=False),
        sa.Column("description", sa.Text, nullable=False),
        sa.Column("embedding", sa.Text, nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.execute(f"ALTER TABLE graph_relationship_embeddings ALTER COLUMN embedding TYPE vector({DIM}) USING embedding::vector({DIM})")

    op.create_table(
        "graph_entity_centroids",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True, server_default=sa.text("gen_random_uuid()")),
        sa.Column("entity_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("graph_entities.id", ondelete="CASCADE"), nullable=False, index=True, unique=True),
        sa.Column("collection_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("collections.id", ondelete="CASCADE"), nullable=False, index=True),
        sa.Column("canonical_name", sa.String(256), nullable=False),
        sa.Column("primary_type", sa.String(64), nullable=True),
        sa.Column("description_count", sa.Integer, server_default="1"),
        sa.Column("embedding", sa.Text, nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.execute(f"ALTER TABLE graph_entity_centroids ALTER COLUMN embedding TYPE vector({DIM}) USING embedding::vector({DIM})")

    op.create_table(
        "graph_chunk_embeddings",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True, server_default=sa.text("gen_random_uuid()")),
        sa.Column("collection_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("collections.id", ondelete="CASCADE"), nullable=False, index=True),
        sa.Column("chunk_hash", sa.String(64), nullable=False, index=True),
        sa.Column("chunk_index", sa.Integer, nullable=False, server_default="0"),
        sa.Column("content", sa.Text, nullable=False),
        sa.Column("embedding", sa.Text, nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.execute(f"ALTER TABLE graph_chunk_embeddings ALTER COLUMN embedding TYPE vector({DIM}) USING embedding::vector({DIM})")


def downgrade() -> None:
    op.drop_table("graph_chunk_embeddings")
    op.drop_table("graph_entity_centroids")
    op.drop_table("graph_relationship_embeddings")
    op.drop_table("graph_entity_embeddings")
