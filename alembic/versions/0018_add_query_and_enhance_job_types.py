"""add query and enhance job types

Revision ID: 0018_query_enhance_jobs
Revises: 0017_merge_migration_heads
Create Date: 2026-06-14
"""

from __future__ import annotations

from alembic import op


# revision identifiers, used by Alembic.
revision = "0018_query_enhance_jobs"
down_revision = "0017_merge_migration_heads"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("ALTER TYPE job_type ADD VALUE IF NOT EXISTS 'query'")
    op.execute("ALTER TYPE job_type ADD VALUE IF NOT EXISTS 'enhance'")


def downgrade() -> None:
    pass
