"""Scope entity aliases to collection.

Revision ID: 0015_scope_entity_aliases_to_collection
Revises: 0014_add_relationship_rel_type
Create Date: 2026-06-04
"""

from __future__ import annotations

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op


revision = "0015_scope_entity_aliases_to_collection"
down_revision = "0014_add_relationship_rel_type"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "entity_aliases",
        sa.Column("collection_id", postgresql.UUID(as_uuid=True), nullable=True),
    )
    op.create_index(
        op.f("ix_entity_aliases_collection_id"),
        "entity_aliases",
        ["collection_id"],
        unique=False,
    )
    op.create_foreign_key(
        "fk_entity_aliases_collection_id_collections",
        "entity_aliases",
        "collections",
        ["collection_id"],
        ["id"],
        ondelete="CASCADE",
    )

    op.execute(
        sa.text(
            """
            UPDATE entity_aliases ea
            SET collection_id = ge.collection_id
            FROM graph_entities ge
            WHERE ge.id = ea.entity_id
            """
        )
    )

    op.alter_column("entity_aliases", "collection_id", nullable=False)

    op.drop_constraint("uq_entity_aliases_alias_name", "entity_aliases", type_="unique")
    op.create_unique_constraint(
        "uq_entity_aliases_collection_alias_name",
        "entity_aliases",
        ["collection_id", "alias_name"],
    )


def downgrade() -> None:
    op.drop_constraint(
        "uq_entity_aliases_collection_alias_name", "entity_aliases", type_="unique"
    )
    op.create_unique_constraint(
        "uq_entity_aliases_alias_name",
        "entity_aliases",
        ["alias_name"],
    )
    op.drop_constraint(
        "fk_entity_aliases_collection_id_collections",
        "entity_aliases",
        type_="foreignkey",
    )
    op.drop_index(op.f("ix_entity_aliases_collection_id"), table_name="entity_aliases")
    op.drop_column("entity_aliases", "collection_id")
