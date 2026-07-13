"""add semantic frames

Revision ID: f17a2c9d4e61
Revises: cbfd1a878428
Create Date: 2026-07-13
"""

from __future__ import annotations

import sqlalchemy as sa

from alembic import op

revision = "f17a2c9d4e61"
down_revision = "cbfd1a878428"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "graph_semantic_frames",
        sa.Column("id", sa.UUID(), nullable=False),
        sa.Column("collection_id", sa.UUID(), nullable=False),
        sa.Column("segment_id", sa.UUID(), nullable=True),
        sa.Column("graph_version_id", sa.UUID(), nullable=True),
        sa.Column("proposition_entity_id", sa.UUID(), nullable=True),
        sa.Column("source_relationship_id", sa.UUID(), nullable=True),
        sa.Column("frame_kind", sa.String(length=32), nullable=False),
        sa.Column("predicate", sa.String(length=128), nullable=True),
        sa.Column("title", sa.String(length=512), nullable=False),
        sa.Column("frame_text", sa.Text(), nullable=False),
        sa.Column("content_hash", sa.String(length=64), nullable=False),
        sa.Column("polarity", sa.String(length=32), nullable=False),
        sa.Column("modality", sa.String(length=32), nullable=False),
        sa.Column("conditions_json", sa.JSON(), nullable=True),
        sa.Column("exceptions_json", sa.JSON(), nullable=True),
        sa.Column("scopes_json", sa.JSON(), nullable=True),
        sa.Column("executable_status", sa.String(length=32), nullable=False),
        sa.Column("metadata_json", sa.JSON(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=True,
        ),
        sa.ForeignKeyConstraint(
            ["collection_id"], ["collections.id"], ondelete="CASCADE"
        ),
        sa.ForeignKeyConstraint(
            ["graph_version_id"], ["graph_versions.id"], ondelete="CASCADE"
        ),
        sa.ForeignKeyConstraint(
            ["proposition_entity_id"], ["graph_entities.id"], ondelete="CASCADE"
        ),
        sa.ForeignKeyConstraint(
            ["segment_id"], ["graph_chunk_segments.id"], ondelete="SET NULL"
        ),
        sa.ForeignKeyConstraint(
            ["source_relationship_id"],
            ["graph_relationships.id"],
            ondelete="SET NULL",
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    for column in (
        "collection_id",
        "segment_id",
        "graph_version_id",
        "proposition_entity_id",
        "source_relationship_id",
        "frame_kind",
        "predicate",
        "content_hash",
    ):
        op.create_index(
            op.f(f"ix_graph_semantic_frames_{column}"),
            "graph_semantic_frames",
            [column],
            unique=False,
        )
    op.create_index(
        "ix_graph_semantic_frames_lookup",
        "graph_semantic_frames",
        ["collection_id", "frame_kind", "predicate"],
        unique=False,
    )
    op.create_table(
        "graph_frame_arguments",
        sa.Column("id", sa.UUID(), nullable=False),
        sa.Column("frame_id", sa.UUID(), nullable=False),
        sa.Column("position", sa.Integer(), nullable=False),
        sa.Column("role", sa.String(length=64), nullable=False),
        sa.Column("entity_id", sa.UUID(), nullable=True),
        sa.Column("literal_value", sa.JSON(), nullable=True),
        sa.Column("variable_name", sa.String(length=128), nullable=True),
        sa.Column("argument_type", sa.String(length=64), nullable=True),
        sa.ForeignKeyConstraint(
            ["entity_id"], ["graph_entities.id"], ondelete="CASCADE"
        ),
        sa.ForeignKeyConstraint(
            ["frame_id"], ["graph_semantic_frames.id"], ondelete="CASCADE"
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "frame_id",
            "position",
            "role",
            name="uq_graph_frame_argument_position",
        ),
    )
    op.create_index(
        op.f("ix_graph_frame_arguments_entity_id"),
        "graph_frame_arguments",
        ["entity_id"],
        unique=False,
    )
    op.create_index(
        op.f("ix_graph_frame_arguments_frame_id"),
        "graph_frame_arguments",
        ["frame_id"],
        unique=False,
    )
    op.create_index(
        "ix_graph_frame_arguments_role_entity",
        "graph_frame_arguments",
        ["role", "entity_id"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_table("graph_frame_arguments")
    op.drop_table("graph_semantic_frames")
