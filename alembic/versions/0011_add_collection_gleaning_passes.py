"""Add gleaning_passes to collections.

Revision ID: 0011_collection_gleaning
Revises: 0010_profile_concurrency
Create Date: 2026-06-02
"""

import sqlalchemy as sa

from alembic import op

revision = "0011_collection_gleaning"
down_revision = "0010_profile_concurrency"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "collections",
        sa.Column(
            "gleaning_passes",
            sa.Integer(),
            nullable=False,
            server_default="1",
        ),
    )


def downgrade() -> None:
    op.drop_column("collections", "gleaning_passes")
