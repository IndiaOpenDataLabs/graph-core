"""add_graph_relationship_types

Revision ID: 7c8b2d1c3f4e
Revises: d7088ddd7fea
Create Date: 2026-06-11 16:40:00.000000
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision = "7c8b2d1c3f4e"
down_revision = "d7088ddd7fea"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "graph_relationship_types",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            primary_key=True,
            server_default=sa.func.gen_random_uuid(),
        ),
        sa.Column("collection_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("canonical_type", sa.String(64), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
        ),
        sa.ForeignKeyConstraint(
            ["collection_id"],
            ["collections.id"],
            ondelete="CASCADE",
        ),
        sa.UniqueConstraint(
            "collection_id",
            "canonical_type",
            name="uq_graph_relationship_types_collection_canonical_type",
        ),
    )
    op.create_index(
        op.f("ix_graph_relationship_types_collection_id"),
        "graph_relationship_types",
        ["collection_id"],
        unique=False,
    )
    op.create_index(
        op.f("ix_graph_relationship_types_canonical_type"),
        "graph_relationship_types",
        ["canonical_type"],
        unique=False,
    )

    op.add_column(
        "relationship_type_aliases",
        sa.Column("relationship_type_id", postgresql.UUID(as_uuid=True), nullable=True),
    )
    op.create_index(
        op.f("ix_relationship_type_aliases_relationship_type_id"),
        "relationship_type_aliases",
        ["relationship_type_id"],
        unique=False,
    )
    op.create_foreign_key(
        "fk_relationship_type_aliases_relationship_type_id",
        "relationship_type_aliases",
        "graph_relationship_types",
        ["relationship_type_id"],
        ["id"],
        ondelete="CASCADE",
    )

    op.add_column(
        "graph_relationships",
        sa.Column("relationship_type_id", postgresql.UUID(as_uuid=True), nullable=True),
    )
    op.create_index(
        op.f("ix_graph_relationships_relationship_type_id"),
        "graph_relationships",
        ["relationship_type_id"],
        unique=False,
    )
    op.create_foreign_key(
        "fk_graph_relationships_relationship_type_id",
        "graph_relationships",
        "graph_relationship_types",
        ["relationship_type_id"],
        ["id"],
        ondelete="CASCADE",
    )

    op.execute(
        sa.text(
            """
            INSERT INTO graph_relationship_types (collection_id, canonical_type)
            SELECT src.collection_id, src.canonical_type
            FROM (
                SELECT DISTINCT collection_id, rel_type AS canonical_type
                FROM graph_relationships
                UNION
                SELECT DISTINCT collection_id, canonical_type
                FROM relationship_type_aliases
            ) AS src
            ON CONFLICT (collection_id, canonical_type) DO NOTHING
            """
        )
    )

    op.execute(
        sa.text(
            """
            UPDATE relationship_type_aliases rta
            SET relationship_type_id = grt.id
            FROM graph_relationship_types grt
            WHERE rta.collection_id = grt.collection_id
              AND rta.canonical_type = grt.canonical_type
            """
        )
    )

    op.execute(
        sa.text(
            """
            UPDATE graph_relationships gr
            SET relationship_type_id = rta.relationship_type_id,
                rel_type = rta.canonical_type
            FROM relationship_type_aliases rta
            WHERE gr.collection_id = rta.collection_id
              AND gr.rel_type = rta.alias_type
              AND gr.relationship_type_id IS DISTINCT FROM rta.relationship_type_id
            """
        )
    )

    op.execute(
        sa.text(
            """
            UPDATE graph_relationships gr
            SET relationship_type_id = grt.id
            FROM graph_relationship_types grt
            WHERE gr.relationship_type_id IS NULL
              AND gr.collection_id = grt.collection_id
              AND gr.rel_type = grt.canonical_type
            """
        )
    )

    op.alter_column(
        "relationship_type_aliases",
        "relationship_type_id",
        nullable=False,
    )
    op.alter_column(
        "graph_relationships",
        "relationship_type_id",
        nullable=False,
    )


def downgrade() -> None:
    op.drop_constraint(
        "fk_graph_relationships_relationship_type_id",
        "graph_relationships",
        type_="foreignkey",
    )
    op.drop_index(
        op.f("ix_graph_relationships_relationship_type_id"),
        table_name="graph_relationships",
    )
    op.drop_column("graph_relationships", "relationship_type_id")

    op.drop_constraint(
        "fk_relationship_type_aliases_relationship_type_id",
        "relationship_type_aliases",
        type_="foreignkey",
    )
    op.drop_index(
        op.f("ix_relationship_type_aliases_relationship_type_id"),
        table_name="relationship_type_aliases",
    )
    op.drop_column("relationship_type_aliases", "relationship_type_id")

    op.drop_index(
        op.f("ix_graph_relationship_types_canonical_type"),
        table_name="graph_relationship_types",
    )
    op.drop_index(
        op.f("ix_graph_relationship_types_collection_id"),
        table_name="graph_relationship_types",
    )
    op.drop_table("graph_relationship_types")
