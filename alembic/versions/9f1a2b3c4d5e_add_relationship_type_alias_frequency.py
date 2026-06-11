"""add_relationship_type_alias_frequency

Revision ID: 9f1a2b3c4d5e
Revises: 7c8b2d1c3f4e
Create Date: 2026-06-11 18:30:00.000000
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa


revision = "9f1a2b3c4d5e"
down_revision = "7c8b2d1c3f4e"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "relationship_type_aliases",
        sa.Column("frequency", sa.Integer(), nullable=False, server_default="1"),
    )
    op.execute(
        sa.text(
            """
            INSERT INTO relationship_type_aliases
                (id, collection_id, relationship_type_id, canonical_type, alias_type, frequency)
            SELECT
                gen_random_uuid(),
                grt.collection_id,
                grt.id,
                grt.canonical_type,
                grt.canonical_type,
                COALESCE(rel_counts.edge_count, 1)
            FROM graph_relationship_types grt
            LEFT JOIN (
                SELECT relationship_type_id, COUNT(*) AS edge_count
                FROM graph_relationships
                GROUP BY relationship_type_id
            ) rel_counts
            ON rel_counts.relationship_type_id = grt.id
            WHERE NOT EXISTS (
                SELECT 1
                FROM relationship_type_aliases rta
                WHERE rta.relationship_type_id = grt.id
                  AND rta.alias_type = grt.canonical_type
            )
            """
        )
    )
    op.alter_column(
        "relationship_type_aliases",
        "frequency",
        server_default=None,
    )


def downgrade() -> None:
    op.drop_column("relationship_type_aliases", "frequency")
