"""auth tables — namespace keys, registered apps, user links

Revision ID: 0008_auth_tables
Revises: 0007_namespace_rls_policies
Create Date: 2026-05-28
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# ruff: noqa: E501
revision = "0008_auth_tables"
down_revision = "0007_namespace_rls_policies"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Create registered_apps table first (before FK reference)
    op.create_table(
        "registered_apps",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("client_id", sa.String(length=64), nullable=False),
        sa.Column("client_secret_hash", sa.String(length=128), nullable=False),
        sa.Column("name", sa.String(length=128), nullable=False),
        sa.Column("owner_email", sa.String(length=256), nullable=True),
        sa.Column("active", sa.Boolean(), nullable=False, server_default=sa.text("true")),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()")),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(op.f("ix_registered_apps_client_id"), "registered_apps", ["client_id"], unique=True)

    # Add auth columns to namespaces
    op.add_column("namespaces", sa.Column("api_key_hash", sa.String(length=128), nullable=True))
    op.add_column("namespaces", sa.Column("api_key_prefix", sa.String(length=8), nullable=True))
    op.add_column("namespaces", sa.Column("owner_app_id", postgresql.UUID(as_uuid=True), nullable=True))
    op.add_column("namespaces", sa.Column("owner_user_sub", sa.String(length=256), nullable=True))
    op.add_column("namespaces", sa.Column("metadata_json", postgresql.JSONB(astext_type=sa.Text()), nullable=True))

    # FK from namespaces to registered_apps (may not exist yet in self-hosted, so nullable)
    op.create_foreign_key(
        "fk_namespaces_owner_app_id",
        "namespaces", "registered_apps",
        ["owner_app_id"], ["id"],
        ondelete="SET NULL",
    )

    # Create app_user_links table
    op.create_table(
        "app_user_links",
        sa.Column("app_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("user_sub", sa.String(length=256), nullable=False),
        sa.Column("namespace_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()")),
        sa.ForeignKeyConstraint(["app_id"], ["registered_apps.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["namespace_id"], ["namespaces.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("app_id", "user_sub"),
    )


def downgrade() -> None:
    op.drop_table("app_user_links")
    op.drop_table("registered_apps")

    op.drop_constraint("fk_namespaces_owner_app_id", "namespaces", type_="foreignkey")
    op.drop_column("namespaces", "metadata_json")
    op.drop_column("namespaces", "owner_user_sub")
    op.drop_column("namespaces", "owner_app_id")
    op.drop_column("namespaces", "api_key_prefix")
    op.drop_column("namespaces", "api_key_hash")
