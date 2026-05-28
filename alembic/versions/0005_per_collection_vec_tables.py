"""drop global vector tables and add embedding_dimensions to collections.

Revision ID: 0005_per_collection_vector_tables
Revises: 0004
Create Date: 2026-05-27

Drops the old global vector tables (vector_chunks, graph_entity_embeddings,
graph_relationship_embeddings, graph_entity_centroids, graph_chunk_embeddings)
which used a fixed embedding dimension.  Replaces them with per-collection
dynamic tables created at collection creation time.

Adds embedding_dimensions column to collections to store the resolved
dimension from the embedding profile.
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa


revision = "0005_per_collection_vec_tables"
down_revision = "0004_add_base_url_to_profiles"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Drop old global vector tables
    op.drop_table("vector_chunks")
    op.drop_table("graph_entity_embeddings")
    op.drop_table("graph_relationship_embeddings")
    op.drop_table("graph_entity_centroids")
    op.drop_table("graph_chunk_embeddings")

    # Add embedding_dimensions to collections
    op.add_column(
        "collections",
        sa.Column("embedding_dimensions", sa.Integer(), nullable=True),
    )


def downgrade() -> None:
    from sqlalchemy.dialects import postgresql

    from graph_core.config import settings

    DIM = settings.default_embedding_dimensions

    op.drop_column("collections", "embedding_dimensions")

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
    op.execute(f"ALTER TABLE graph_entity_embeddings ALTER COLUMN embedding TYPE vector({DIM}) USING embedding::vector({DIM})")

    op.create_table(
        "vector_chunks",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("namespace_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("namespaces.id", ondelete="CASCADE"), nullable=False, index=True),
        sa.Column("collection_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("collections.id", ondelete="CASCADE"), nullable=False, index=True),
        sa.Column("chunk_hash", sa.String(64), nullable=False, index=True),
        sa.Column("chunk_index", sa.Integer, nullable=False, server_default="0"),
        sa.Column("content", sa.Text, nullable=False),
        sa.Column("token_count", sa.Integer, nullable=False, server_default="0"),
        sa.Column("metadata_json", postgresql.JSON, nullable=True),
        sa.Column("embedding", sa.Text, nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.UniqueConstraint("collection_id", "chunk_hash", "chunk_index", name="uq_vector_chunk_identity"),
    )
    op.execute(f"ALTER TABLE vector_chunks ALTER COLUMN embedding TYPE vector({DIM}) USING embedding::vector({DIM})")
