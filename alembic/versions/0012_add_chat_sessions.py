"""Add chat session tables.

Revision ID: 0012_add_chat_sessions
Revises: 0011_collection_gleaning
Create Date: 2026-06-02
"""

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

revision = "0012_add_chat_sessions"
down_revision = "0011_collection_gleaning"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "chat_sessions",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("namespace_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("collection_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("title", sa.String(length=255), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=True,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=True,
        ),
        sa.ForeignKeyConstraint(
            ["collection_id"], ["collections.id"], ondelete="CASCADE"
        ),
        sa.ForeignKeyConstraint(
            ["namespace_id"], ["namespaces.id"], ondelete="CASCADE"
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        op.f("ix_chat_sessions_collection_id"),
        "chat_sessions",
        ["collection_id"],
        unique=False,
    )
    op.create_index(
        op.f("ix_chat_sessions_namespace_id"),
        "chat_sessions",
        ["namespace_id"],
        unique=False,
    )

    op.create_table(
        "chat_turns",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("chat_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("collection_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("turn_index", sa.Integer(), nullable=False),
        sa.Column("question", sa.Text(), nullable=False),
        sa.Column("response", sa.Text(), nullable=False),
        sa.Column("mode", sa.String(length=64), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=True,
        ),
        sa.ForeignKeyConstraint(["chat_id"], ["chat_sessions.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(
            ["collection_id"], ["collections.id"], ondelete="CASCADE"
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        op.f("ix_chat_turns_chat_id"),
        "chat_turns",
        ["chat_id"],
        unique=False,
    )
    op.create_index(
        op.f("ix_chat_turns_collection_id"),
        "chat_turns",
        ["collection_id"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index(op.f("ix_chat_turns_collection_id"), table_name="chat_turns")
    op.drop_index(op.f("ix_chat_turns_chat_id"), table_name="chat_turns")
    op.drop_table("chat_turns")
    op.drop_index(op.f("ix_chat_sessions_namespace_id"), table_name="chat_sessions")
    op.drop_index(op.f("ix_chat_sessions_collection_id"), table_name="chat_sessions")
    op.drop_table("chat_sessions")
