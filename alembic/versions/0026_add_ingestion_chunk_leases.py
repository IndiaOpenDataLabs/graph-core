"""add ingestion chunk lease tracking

Revision ID: 0026_add_ingestion_chunk_leases
Revises: 0025_document_provenance_fields
Create Date: 2026-06-29
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = "0026_add_ingestion_chunk_leases"
down_revision = "0025_document_provenance_fields"
branch_labels = None
depends_on = None


def _add_column_if_missing(table_name_: str, column_name: str, column_ddl: str) -> None:
    op.execute(
        sa.text(
            f"ALTER TABLE {table_name_} "
            f"ADD COLUMN IF NOT EXISTS {column_name} {column_ddl}"
        )
    )


def upgrade() -> None:
    _add_column_if_missing(
        "ingestion_chunks", "processing_started_at", "TIMESTAMPTZ"
    )
    _add_column_if_missing("ingestion_chunks", "lease_expires_at", "TIMESTAMPTZ")
    _add_column_if_missing("ingestion_chunks", "completed_at", "TIMESTAMPTZ")
    op.create_index(
        "ix_ingestion_chunks_lease_expires_at",
        "ingestion_chunks",
        ["lease_expires_at"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index("ix_ingestion_chunks_lease_expires_at", table_name="ingestion_chunks")
    op.execute("ALTER TABLE ingestion_chunks DROP COLUMN IF EXISTS lease_expires_at")
    op.execute(
        "ALTER TABLE ingestion_chunks DROP COLUMN IF EXISTS processing_started_at"
    )
