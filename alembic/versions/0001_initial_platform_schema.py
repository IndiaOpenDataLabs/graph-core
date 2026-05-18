"""initial platform schema

Revision ID: 0001_initial_platform_schema
Revises:
Create Date: 2026-05-19
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision = "0001_initial_platform_schema"
down_revision = None
branch_labels = None
depends_on = None


rag_strategy = postgresql.ENUM(
    "vector",
    "light_rag",
    "custom_graph_rag",
    name="rag_strategy",
    create_type=False,
)
profile_kind = postgresql.ENUM(
    "embedding",
    "llm",
    name="profile_kind",
    create_type=False,
)
job_type = postgresql.ENUM(
    "ingest_chunk",
    "ingest_document",
    "delete_collection",
    "reindex",
    name="job_type",
    create_type=False,
)
job_status = postgresql.ENUM(
    "pending",
    "running",
    "completed",
    "failed",
    "cancelled",
    name="job_status",
    create_type=False,
)
ingest_strategy = postgresql.ENUM(
    "vector",
    "custom_graph_rag",
    name="ingest_strategy",
    create_type=False,
)


def upgrade() -> None:
    bind = op.get_bind()

    op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    rag_strategy.create(bind, checkfirst=True)
    profile_kind.create(bind, checkfirst=True)
    job_type.create(bind, checkfirst=True)
    job_status.create(bind, checkfirst=True)
    ingest_strategy.create(bind, checkfirst=True)

    op.create_table(
        "namespaces",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("name", sa.String(length=128), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=True,
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(op.f("ix_namespaces_name"), "namespaces", ["name"], unique=True)

    op.create_table(
        "credentials",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column(
            "namespace_id",
            postgresql.UUID(as_uuid=True),
            nullable=False,
        ),
        sa.Column("provider", sa.String(length=64), nullable=False),
        sa.Column("label", sa.String(length=128), nullable=True),
        sa.Column("encrypted_secret", sa.String(length=1024), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=True,
        ),
        sa.ForeignKeyConstraint(
            ["namespace_id"],
            ["namespaces.id"],
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "namespace_id",
            "label",
            name="uq_namespace_credential_label",
        ),
    )
    op.create_index(
        op.f("ix_credentials_namespace_id"),
        "credentials",
        ["namespace_id"],
        unique=False,
    )
    op.create_index(
        op.f("ix_credentials_provider"),
        "credentials",
        ["provider"],
        unique=False,
    )

    op.create_table(
        "profiles",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column(
            "namespace_id",
            postgresql.UUID(as_uuid=True),
            nullable=False,
        ),
        sa.Column(
            "credential_id",
            postgresql.UUID(as_uuid=True),
            nullable=True,
        ),
        sa.Column("kind", profile_kind, nullable=False),
        sa.Column("provider", sa.String(length=64), nullable=False),
        sa.Column("model", sa.String(length=128), nullable=False),
        sa.Column("label", sa.String(length=128), nullable=True),
        sa.Column("dimensions", sa.Integer(), nullable=True),
        sa.Column("distance_metric", sa.String(length=32), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=True,
        ),
        sa.ForeignKeyConstraint(
            ["credential_id"],
            ["credentials.id"],
            ondelete="SET NULL",
        ),
        sa.ForeignKeyConstraint(
            ["namespace_id"],
            ["namespaces.id"],
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        op.f("ix_profiles_namespace_id"),
        "profiles",
        ["namespace_id"],
        unique=False,
    )

    op.create_table(
        "collections",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column(
            "namespace_id",
            postgresql.UUID(as_uuid=True),
            nullable=False,
        ),
        sa.Column("name", sa.String(length=255), nullable=False),
        sa.Column("description", sa.String(length=1024), nullable=True),
        sa.Column("strategy", rag_strategy, nullable=False),
        sa.Column("default_query_mode", sa.String(length=64), nullable=True),
        sa.Column(
            "embedding_profile_id",
            postgresql.UUID(as_uuid=True),
            nullable=True,
        ),
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
            ["embedding_profile_id"],
            ["profiles.id"],
            ondelete="SET NULL",
        ),
        sa.ForeignKeyConstraint(
            ["namespace_id"],
            ["namespaces.id"],
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "namespace_id",
            "name",
            name="uq_namespace_collection_name",
        ),
    )
    op.create_index(
        op.f("ix_collections_embedding_profile_id"),
        "collections",
        ["embedding_profile_id"],
        unique=False,
    )
    op.create_index(
        op.f("ix_collections_name"),
        "collections",
        ["name"],
        unique=False,
    )
    op.create_index(
        op.f("ix_collections_namespace_id"),
        "collections",
        ["namespace_id"],
        unique=False,
    )

    op.create_table(
        "jobs",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column(
            "namespace_id",
            postgresql.UUID(as_uuid=True),
            nullable=False,
        ),
        sa.Column(
            "collection_id",
            postgresql.UUID(as_uuid=True),
            nullable=True,
        ),
        sa.Column("job_type", job_type, nullable=False),
        sa.Column("status", job_status, nullable=False),
        sa.Column("progress_percent", sa.Integer(), nullable=True),
        sa.Column("error", sa.Text(), nullable=True),
        sa.Column("payload", postgresql.JSON(astext_type=sa.Text()), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=True,
        ),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(
            ["collection_id"],
            ["collections.id"],
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["namespace_id"],
            ["namespaces.id"],
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        op.f("ix_jobs_collection_id"),
        "jobs",
        ["collection_id"],
        unique=False,
    )
    op.create_index(
        op.f("ix_jobs_namespace_id"),
        "jobs",
        ["namespace_id"],
        unique=False,
    )

    op.create_table(
        "job_events",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("job_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column(
            "timestamp",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=True,
        ),
        sa.Column("event_type", sa.String(length=64), nullable=False),
        sa.Column("payload", postgresql.JSON(astext_type=sa.Text()), nullable=True),
        sa.ForeignKeyConstraint(["job_id"], ["jobs.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        op.f("ix_job_events_job_id"),
        "job_events",
        ["job_id"],
        unique=False,
    )

    op.create_table(
        "ingestion_records",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column(
            "collection_id",
            postgresql.UUID(as_uuid=True),
            nullable=False,
        ),
        sa.Column("chunk_hash", sa.String(length=64), nullable=False),
        sa.Column("strategy", ingest_strategy, nullable=False),
        sa.Column("extraction_model", sa.String(length=128), nullable=True),
        sa.Column("embedding_model", sa.String(length=128), nullable=True),
        sa.Column("entity_count", sa.Integer(), nullable=True),
        sa.Column("relationship_count", sa.Integer(), nullable=True),
        sa.Column("sanitization_flags", postgresql.JSON(astext_type=sa.Text()), nullable=True),
        sa.Column(
            "ingested_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=True,
        ),
        sa.Column("source_document_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.ForeignKeyConstraint(
            ["collection_id"],
            ["collections.id"],
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        op.f("ix_ingestion_records_chunk_hash"),
        "ingestion_records",
        ["chunk_hash"],
        unique=False,
    )
    op.create_index(
        op.f("ix_ingestion_records_collection_id"),
        "ingestion_records",
        ["collection_id"],
        unique=False,
    )

    op.execute(
        """
        CREATE TABLE vector_chunks (
            id UUID PRIMARY KEY,
            namespace_id UUID NOT NULL REFERENCES namespaces(id) ON DELETE CASCADE,
            collection_id UUID NOT NULL REFERENCES collections(id) ON DELETE CASCADE,
            chunk_hash VARCHAR(64) NOT NULL,
            chunk_index INTEGER NOT NULL DEFAULT 0,
            content TEXT NOT NULL,
            token_count INTEGER NOT NULL DEFAULT 0,
            metadata_json JSON,
            embedding vector(256) NOT NULL,
            created_at TIMESTAMPTZ DEFAULT now(),
            CONSTRAINT uq_vector_chunk_identity
                UNIQUE (collection_id, chunk_hash, chunk_index)
        )
        """
    )
    op.create_index(
        op.f("ix_vector_chunks_namespace_id"),
        "vector_chunks",
        ["namespace_id"],
        unique=False,
    )
    op.create_index(
        op.f("ix_vector_chunks_collection_id"),
        "vector_chunks",
        ["collection_id"],
        unique=False,
    )
    op.create_index(
        op.f("ix_vector_chunks_chunk_hash"),
        "vector_chunks",
        ["chunk_hash"],
        unique=False,
    )


def downgrade() -> None:
    bind = op.get_bind()

    op.drop_index(op.f("ix_vector_chunks_chunk_hash"), table_name="vector_chunks")
    op.drop_index(op.f("ix_vector_chunks_collection_id"), table_name="vector_chunks")
    op.drop_index(op.f("ix_vector_chunks_namespace_id"), table_name="vector_chunks")
    op.execute("DROP TABLE vector_chunks")

    op.drop_index(
        op.f("ix_ingestion_records_collection_id"),
        table_name="ingestion_records",
    )
    op.drop_index(
        op.f("ix_ingestion_records_chunk_hash"),
        table_name="ingestion_records",
    )
    op.drop_table("ingestion_records")

    op.drop_index(op.f("ix_job_events_job_id"), table_name="job_events")
    op.drop_table("job_events")

    op.drop_index(op.f("ix_jobs_namespace_id"), table_name="jobs")
    op.drop_index(op.f("ix_jobs_collection_id"), table_name="jobs")
    op.drop_table("jobs")

    op.drop_index(op.f("ix_collections_namespace_id"), table_name="collections")
    op.drop_index(
        op.f("ix_collections_name"),
        table_name="collections",
    )
    op.drop_index(
        op.f("ix_collections_embedding_profile_id"),
        table_name="collections",
    )
    op.drop_table("collections")

    op.drop_index(op.f("ix_profiles_namespace_id"), table_name="profiles")
    op.drop_table("profiles")

    op.drop_index(op.f("ix_credentials_provider"), table_name="credentials")
    op.drop_index(op.f("ix_credentials_namespace_id"), table_name="credentials")
    op.drop_table("credentials")

    op.drop_index(op.f("ix_namespaces_name"), table_name="namespaces")
    op.drop_table("namespaces")

    ingest_strategy.drop(bind, checkfirst=True)
    job_status.drop(bind, checkfirst=True)
    job_type.drop(bind, checkfirst=True)
    profile_kind.drop(bind, checkfirst=True)
    rag_strategy.drop(bind, checkfirst=True)
