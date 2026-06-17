"""add document provenance fields

Revision ID: 0025_document_provenance_fields
Revises: 0024_falkordb_acl_getuser_fix
Create Date: 2026-06-17
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa

from graph_core.storage.vector_tables import table_name


# revision identifiers, used by Alembic.
revision = "0025_document_provenance_fields"
down_revision = "0024_falkordb_acl_getuser_fix"
branch_labels = None
depends_on = None


def _add_column_if_missing(table_name_: str, column_name: str, column_ddl: str) -> None:
    op.execute(
        sa.text(
            f"ALTER TABLE {table_name_} "
            f"ADD COLUMN IF NOT EXISTS {column_name} {column_ddl}"
        )
    )


def upgrade() -> None:
    bind = op.get_bind()

    _add_column_if_missing("jobs", "document_id", "UUID")
    _add_column_if_missing("jobs", "document_path", "VARCHAR(1024)")
    op.create_index("ix_jobs_document_id", "jobs", ["document_id"], unique=False)

    _add_column_if_missing("ingestion_chunks", "document_id", "UUID")
    _add_column_if_missing("ingestion_chunks", "document_path", "VARCHAR(1024)")
    op.create_index(
        "ix_ingestion_chunks_document_id",
        "ingestion_chunks",
        ["document_id"],
        unique=False,
    )

    _add_column_if_missing("ingestion_records", "document_id", "UUID")
    _add_column_if_missing("ingestion_records", "document_path", "VARCHAR(1024)")
    op.create_index(
        "ix_ingestion_records_document_id",
        "ingestion_records",
        ["document_id"],
        unique=False,
    )

    _add_column_if_missing("raw_chunk_extractions", "document_path", "VARCHAR(1024)")
    _add_column_if_missing("entity_descriptions", "document_path", "VARCHAR(1024)")
    _add_column_if_missing("entity_aliases", "document_path", "VARCHAR(1024)")
    _add_column_if_missing(
        "relationship_descriptions", "document_path", "VARCHAR(1024)"
    )

    collection_ids = bind.execute(sa.text("SELECT id FROM collections")).scalars().all()
    for collection_id in collection_ids:
        for kind in (
            "vector_chunks",
            "chunk_embeddings",
            "entity_embeddings",
            "relationship_embeddings",
        ):
            tbl = table_name(collection_id, kind)
            _add_column_if_missing(tbl, "document_id", "UUID")
            _add_column_if_missing(tbl, "document_path", "VARCHAR(1024)")

    # Backfill job/chunk provenance from payloads where available.
    op.execute(
        sa.text(
            """
            UPDATE jobs
            SET document_id = NULLIF(payload->>'document_id', '')::uuid,
                document_path = NULLIF(payload->>'document_path', '')
            WHERE payload IS NOT NULL
            """
        )
    )
    op.execute(
        sa.text(
            """
            UPDATE ingestion_chunks AS c
            SET document_id = j.document_id,
                document_path = j.document_path
            FROM jobs AS j
            WHERE c.job_id = j.id
              AND (c.document_id IS NULL AND c.document_path IS NULL)
            """
        )
    )


def downgrade() -> None:
    bind = op.get_bind()

    op.drop_index("ix_ingestion_records_document_id", table_name="ingestion_records")
    op.drop_index("ix_ingestion_chunks_document_id", table_name="ingestion_chunks")
    op.drop_index("ix_jobs_document_id", table_name="jobs")

    for collection_id in bind.execute(sa.text("SELECT id FROM collections")).scalars().all():
        for kind in (
            "vector_chunks",
            "chunk_embeddings",
            "entity_embeddings",
            "relationship_embeddings",
        ):
            tbl = table_name(collection_id, kind)
            op.execute(sa.text(f"ALTER TABLE {tbl} DROP COLUMN IF EXISTS document_path"))
            op.execute(sa.text(f"ALTER TABLE {tbl} DROP COLUMN IF EXISTS document_id"))

    op.drop_column("relationship_descriptions", "document_path")
    op.drop_column("entity_aliases", "document_path")
    op.drop_column("entity_descriptions", "document_path")
    op.drop_column("raw_chunk_extractions", "document_path")

    op.drop_column("ingestion_records", "document_path")
    op.drop_column("ingestion_records", "document_id")
    op.drop_column("ingestion_chunks", "document_path")
    op.drop_column("ingestion_chunks", "document_id")
    op.drop_column("jobs", "document_path")
    op.drop_column("jobs", "document_id")
