"""add_relationship_type_aliases

Revision ID: d7088ddd7fea
Revises: 0015_aliases_per_collection
Create Date: 2026-06-10 18:51:47.558710
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


# revision identifiers, used by Alembic.
revision = 'd7088ddd7fea'
down_revision = '0015_aliases_per_collection'
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "relationship_type_aliases",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True, server_default=sa.func.gen_random_uuid()),
        sa.Column("collection_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("canonical_type", sa.String(64), nullable=False),
        sa.Column("alias_type", sa.String(64), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.ForeignKeyConstraint(["collection_id"], ["collections.id"], ondelete="CASCADE"),
        sa.UniqueConstraint("collection_id", "alias_type", name="uq_relationship_type_aliases_collection_alias_type"),
        sa.Index("ix_relationship_type_aliases_collection_canonical", "collection_id", "canonical_type"),
        sa.Index("ix_relationship_type_aliases_alias_type", "alias_type"),
    )


def downgrade() -> None:
    op.drop_table("relationship_type_aliases")