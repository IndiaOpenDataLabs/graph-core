"""add cancelled chunk status

Revision ID: 0027_add_chunk_cancelled_status
Revises: 0026_add_ingestion_chunk_leases
Create Date: 2026-06-29
"""

from __future__ import annotations

from alembic import op


# revision identifiers, used by Alembic.
revision = "0027_add_chunk_cancelled_status"
down_revision = "0026_add_ingestion_chunk_leases"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("ALTER TYPE chunk_status ADD VALUE IF NOT EXISTS 'cancelled'")


def downgrade() -> None:
    pass
