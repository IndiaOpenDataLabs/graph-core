"""Add role-aware chat messages.

Revision ID: 0013_add_chat_messages
Revises: 0012_add_chat_sessions
Create Date: 2026-06-02
"""

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

revision = "0013_add_chat_messages"
down_revision = "0012_add_chat_sessions"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "chat_messages",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("chat_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("collection_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("role", sa.String(length=32), nullable=False),
        sa.Column("turn_index", sa.Integer(), nullable=False),
        sa.Column("message_index", sa.Integer(), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("mode", sa.String(length=64), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=True,
        ),
        sa.ForeignKeyConstraint(["chat_id"], ["chat_sessions.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(
            ["collection_id"],
            ["collections.id"],
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        op.f("ix_chat_messages_chat_id"),
        "chat_messages",
        ["chat_id"],
        unique=False,
    )
    op.create_index(
        op.f("ix_chat_messages_collection_id"),
        "chat_messages",
        ["collection_id"],
        unique=False,
    )

    op.execute(
        """
        INSERT INTO chat_messages (
            id, chat_id, collection_id, role, turn_index, message_index,
            content, mode, created_at
        )
        SELECT
            (
                substr(md5(chat_id::text || ':user:' || turn_index::text), 1, 8)
                || '-' ||
                substr(md5(chat_id::text || ':user:' || turn_index::text), 9, 4)
                || '-' ||
                substr(md5(chat_id::text || ':user:' || turn_index::text), 13, 4)
                || '-' ||
                substr(md5(chat_id::text || ':user:' || turn_index::text), 17, 4)
                || '-' ||
                substr(md5(chat_id::text || ':user:' || turn_index::text), 21, 12)
            )::uuid,
            chat_id,
            collection_id,
            'user',
            turn_index,
            (turn_index * 2) - 1,
            question,
            NULL,
            created_at
        FROM chat_turns
        """
    )
    op.execute(
        """
        INSERT INTO chat_messages (
            id, chat_id, collection_id, role, turn_index, message_index,
            content, mode, created_at
        )
        SELECT
            (
                substr(md5(chat_id::text || ':assistant:' || turn_index::text), 1, 8)
                || '-' ||
                substr(md5(chat_id::text || ':assistant:' || turn_index::text), 9, 4)
                || '-' ||
                substr(md5(chat_id::text || ':assistant:' || turn_index::text), 13, 4)
                || '-' ||
                substr(md5(chat_id::text || ':assistant:' || turn_index::text), 17, 4)
                || '-' ||
                substr(md5(chat_id::text || ':assistant:' || turn_index::text), 21, 12)
            )::uuid,
            chat_id,
            collection_id,
            'assistant',
            turn_index,
            turn_index * 2,
            response,
            mode,
            created_at
        FROM chat_turns
        """
    )


def downgrade() -> None:
    op.drop_index(op.f("ix_chat_messages_collection_id"), table_name="chat_messages")
    op.drop_index(op.f("ix_chat_messages_chat_id"), table_name="chat_messages")
    op.drop_table("chat_messages")
