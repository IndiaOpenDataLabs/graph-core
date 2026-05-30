"""fix api_key_prefix column length

Revision ID: 0009_fix_prefix_len
Revises: 0008_auth_tables
Create Date: 2026-05-30
"""

from __future__ import annotations

import sqlalchemy as sa

from alembic import op

# ruff: noqa: E501
revision = "0009_fix_prefix_len"
down_revision = "0008_auth_tables"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.alter_column(
        "namespaces",
        "api_key_prefix",
        type_=sa.String(length=16),
        existing_type=sa.String(length=8),
        existing_nullable=True,
    )


def downgrade() -> None:
    op.alter_column(
        "namespaces",
        "api_key_prefix",
        type_=sa.String(length=8),
        existing_type=sa.String(length=16),
        existing_nullable=True,
    )
