"""add base_url to credentials and profiles

Revision ID: 0004
Revises: 0003
Create Date: 2026-05-27
"""

from alembic import op
import sqlalchemy as sa

revision = "0004_add_base_url_to_profiles"
down_revision = "0003_graph_rag_vector_tables"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("credentials", sa.Column("base_url", sa.String(512), nullable=True))
    op.add_column("profiles", sa.Column("base_url", sa.String(512), nullable=True))


def downgrade() -> None:
    op.drop_column("profiles", "base_url")
    op.drop_column("credentials", "base_url")
