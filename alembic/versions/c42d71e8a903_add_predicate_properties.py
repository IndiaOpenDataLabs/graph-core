"""add predicate properties

Revision ID: c42d71e8a903
Revises: a61c4e8b932f
Create Date: 2026-07-13
"""

from __future__ import annotations

import sqlalchemy as sa

from alembic import op

revision = "c42d71e8a903"
down_revision = "a61c4e8b932f"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "graph_relationship_types",
        sa.Column("inferred_properties", sa.JSON(), nullable=True),
    )
    op.add_column(
        "graph_relationship_types",
        sa.Column("property_votes", sa.JSON(), nullable=True),
    )
    op.add_column(
        "graph_relationship_types",
        sa.Column(
            "property_observation_count",
            sa.Integer(),
            nullable=False,
            server_default="0",
        ),
    )


def downgrade() -> None:
    op.drop_column("graph_relationship_types", "property_observation_count")
    op.drop_column("graph_relationship_types", "property_votes")
    op.drop_column("graph_relationship_types", "inferred_properties")
