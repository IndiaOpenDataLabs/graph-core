"""Custom SQLAlchemy types for vector storage."""

from __future__ import annotations

from sqlalchemy import JSON
from sqlalchemy.types import TypeDecorator, UserDefinedType


class _PostgresVector(UserDefinedType):
    cache_ok = True

    def __init__(self, dimensions: int):
        self.dimensions = dimensions

    def get_col_spec(self, **kw) -> str:
        return f"vector({self.dimensions})"


class EmbeddingVector(TypeDecorator):
    """Use pgvector on PostgreSQL and JSON elsewhere."""

    impl = JSON
    cache_ok = True

    def __init__(self, dimensions: int):
        super().__init__()
        self.dimensions = dimensions

    def load_dialect_impl(self, dialect):
        if dialect.name == "postgresql":
            return dialect.type_descriptor(_PostgresVector(self.dimensions))
        return dialect.type_descriptor(JSON())

    def process_bind_param(self, value, dialect):
        if value is None:
            return value
        if dialect.name == "postgresql":
            return "[" + ",".join(str(float(component)) for component in value) + "]"
        return [float(component) for component in value]

    def process_result_value(self, value, dialect):
        if value is None:
            return value
        if isinstance(value, str):
            stripped = value.strip("[]")
            if not stripped:
                return []
            return [float(component) for component in stripped.split(",")]
        return [float(component) for component in value]
