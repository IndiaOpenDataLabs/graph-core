"""Add rel_type to graph_relationships for multi-dimensional graph retrieval.

Revision ID: 0014_add_relationship_rel_type
Revises: 0013_add_chat_messages
Create Date: 2026-06-03
"""

import sqlalchemy as sa

from alembic import op

revision = "0014_add_relationship_rel_type"
down_revision = "0013_add_chat_messages"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "graph_relationships",
        sa.Column("rel_type", sa.String(length=64), nullable=False, server_default="RELATES_TO"),
    )
    op.create_index(
        op.f("ix_graph_relationships_rel_type"),
        "graph_relationships",
        ["rel_type"],
    )
    op.create_index(
        op.f("ix_graph_relationships_source_target_type"),
        "graph_relationships",
        ["source_entity_id", "target_entity_id", "rel_type"],
    )


def downgrade() -> None:
    op.drop_index(
        op.f("ix_graph_relationships_source_target_type"), table_name="graph_relationships"
    )
    op.drop_index(op.f("ix_graph_relationships_rel_type"), table_name="graph_relationships")
    op.drop_column("graph_relationships", "rel_type")
