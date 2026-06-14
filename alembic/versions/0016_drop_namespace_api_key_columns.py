"""drop namespace api key columns

Revision ID: 0016_drop_namespace_api_key_columns
Revises: d7088ddd7fea
Create Date: 2026-06-14
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = "0016_drop_namespace_api_key_columns"
down_revision = "d7088ddd7fea"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.drop_column("namespaces", "api_key_prefix")
    op.drop_column("namespaces", "api_key_hash")


def downgrade() -> None:
    op.add_column("namespaces", sa.Column("api_key_hash", sa.String(length=128), nullable=True))
    op.add_column("namespaces", sa.Column("api_key_prefix", sa.String(length=8), nullable=True))
