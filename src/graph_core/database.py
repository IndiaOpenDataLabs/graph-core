"""SQLAlchemy async engine, session factory, and base model."""

import uuid
from contextvars import ContextVar
from typing import Any

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase

from graph_core.config import settings


def _uuid_for_sql(val: uuid.UUID) -> str:
    """Convert UUID to the string format expected by the current DB dialect.

    SQLite stores UUIDs as 32-char hex (no dashes). Postgres UUID type
    accepts any canonical format. Using .hex works for both.
    """
    return val.hex


class Base(DeclarativeBase):
    pass


engine = create_async_engine(
    settings.database_url,
    echo=False,
    pool_size=settings.sqlalchemy_pool_size,
    max_overflow=settings.sqlalchemy_max_overflow,
    pool_timeout=settings.sqlalchemy_pool_timeout,
    pool_recycle=300,
)

# Request-scoped namespace context — set by API middleware or dependency.
# When populated, every new session automatically sets the Postgres
# session variable so RLS policies enforce namespace isolation.
current_namespace_id: ContextVar[uuid.UUID | None] = ContextVar(
    "current_namespace_id",
    default=None,
)


class NamespacedAsyncSession(AsyncSession):
    """AsyncSession that sets RLS namespace context on transaction begin."""

    async def begin(self) -> Any:
        await super().begin()
        ns_id = current_namespace_id.get()
        if ns_id is not None:
            await self.execute(
                text("SET LOCAL app.current_namespace_id = :nsid"),
                {"nsid": _uuid_for_sql(ns_id)},
            )


AsyncSessionLocal = async_sessionmaker(
    engine,
    class_=NamespacedAsyncSession,
    expire_on_commit=False,
)


async def set_namespace_context(session: AsyncSession, namespace_id: uuid.UUID) -> None:
    """Explicitly set the Postgres session variable for RLS.

    Use this when you need to override the request-scoped contextvar
    (e.g., in background workers operating on a specific namespace).

    Must be called inside an active transaction before any query runs.
    """
    await session.execute(
            text("SET LOCAL app.current_namespace_id = :nsid"),
            {"nsid": _uuid_for_sql(namespace_id)},
        )


async def get_session() -> AsyncSession:
    """FastAPI dependency: yields a session, commits, and closes.

    Automatically sets RLS namespace context when current_namespace_id
    contextvar is populated (e.g., by namespace middleware or dependency).
    """
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()
