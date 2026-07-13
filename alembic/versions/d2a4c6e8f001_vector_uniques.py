"""deduplicate dynamic vector identities

Revision ID: d2a4c6e8f001
Revises: c42d71e8a903
Create Date: 2026-07-13
"""

from __future__ import annotations

from alembic import op

revision = "d2a4c6e8f001"
down_revision = "c42d71e8a903"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        r"""
        DO $$
        DECLARE
            target RECORD;
            index_name TEXT;
        BEGIN
            FOR target IN
                SELECT schemaname, tablename
                FROM pg_tables
                WHERE schemaname = current_schema()
                  AND tablename ~ '^vc_[0-9a-f]{16}_entity_embeddings$'
            LOOP
                EXECUTE format(
                    'LOCK TABLE %I.%I IN SHARE ROW EXCLUSIVE MODE',
                    target.schemaname, target.tablename
                );
                EXECUTE format(
                    'DELETE FROM %I.%I newer USING %I.%I older '
                    'WHERE newer.description_id = older.description_id '
                    'AND newer.id > older.id',
                    target.schemaname, target.tablename,
                    target.schemaname, target.tablename
                );
                index_name := 'uq_' || target.tablename || '_description_id';
                EXECUTE format(
                    'CREATE UNIQUE INDEX IF NOT EXISTS %I ON %I.%I (description_id)',
                    index_name, target.schemaname, target.tablename
                );
            END LOOP;

            FOR target IN
                SELECT schemaname, tablename
                FROM pg_tables
                WHERE schemaname = current_schema()
                  AND tablename ~ '^vc_[0-9a-f]{16}_relationship_embeddings$'
            LOOP
                EXECUTE format(
                    'LOCK TABLE %I.%I IN SHARE ROW EXCLUSIVE MODE',
                    target.schemaname, target.tablename
                );
                EXECUTE format(
                    'DELETE FROM %I.%I newer USING %I.%I older '
                    'WHERE newer.relationship_id = older.relationship_id '
                    'AND newer.id > older.id',
                    target.schemaname, target.tablename,
                    target.schemaname, target.tablename
                );
                index_name := 'uq_' || target.tablename || '_relationship_id';
                EXECUTE format(
                    'CREATE UNIQUE INDEX IF NOT EXISTS %I ON %I.%I (relationship_id)',
                    index_name, target.schemaname, target.tablename
                );
            END LOOP;
        END $$;
        """
    )


def downgrade() -> None:
    op.execute(
        r"""
        DO $$
        DECLARE
            target RECORD;
            index_name TEXT;
        BEGIN
            FOR target IN
                SELECT schemaname, tablename
                FROM pg_tables
                WHERE schemaname = current_schema()
                  AND tablename ~ '^vc_[0-9a-f]{16}_(entity|relationship)_embeddings$'
            LOOP
                index_name := 'uq_' || target.tablename || CASE
                    WHEN target.tablename LIKE '%entity_embeddings'
                    THEN '_description_id'
                    ELSE '_relationship_id'
                END;
                IF NOT EXISTS (
                    SELECT 1
                    FROM pg_constraint
                    WHERE conname = index_name
                      AND connamespace = target.schemaname::regnamespace
                ) THEN
                    EXECUTE format(
                        'DROP INDEX IF EXISTS %I.%I',
                        target.schemaname, index_name
                    );
                END IF;
            END LOOP;
        END $$;
        """
    )
