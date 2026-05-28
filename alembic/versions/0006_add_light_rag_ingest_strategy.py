"""add light_rag to ingest_strategy enum.

Revision ID: 0006_add_light_rag_ingest_strategy
Revises: 0005_per_collection_vec_tables
Create Date: 2026-05-28
"""

from __future__ import annotations

from alembic import op

revision = "0006_light_rag_ingest_enum"
down_revision = "0005_per_collection_vec_tables"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("ALTER TYPE ingest_strategy ADD VALUE IF NOT EXISTS 'light_rag'")


def downgrade() -> None:
    pass
