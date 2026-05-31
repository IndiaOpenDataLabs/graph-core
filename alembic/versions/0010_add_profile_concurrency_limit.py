"""add profile concurrency limit

Revision ID: 0010_profile_concurrency
Revises: 0009_fix_prefix_len
Create Date: 2026-05-31
"""

from __future__ import annotations

import sqlalchemy as sa

from alembic import op

revision = "0010_profile_concurrency"
down_revision = "0009_fix_prefix_len"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "profiles",
        sa.Column("max_concurrent_calls", sa.Integer(), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("profiles", "max_concurrent_calls")
