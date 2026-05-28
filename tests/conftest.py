"""Shared test fixtures — uses in-memory SQLite, no Postgres required."""

import asyncio
import uuid

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

# Patch config BEFORE any graph_core modules import database.py
import graph_core.config

graph_core.config.settings.database_url = "sqlite+aiosqlite:///:memory:"
graph_core.config.settings.credential_encryption_key = "test-key"
graph_core.config.settings.platform_admin_key = "test-admin-key"

# Now import — engine will be created with SQLite
from graph_core.database import AsyncSessionLocal, Base, engine
from graph_core.main import app
from graph_core.models.collection import Collection
from graph_core.models.namespace import Namespace
from graph_core.services.graph import GraphService


@pytest.fixture(scope="session")
def event_loop():
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest_asyncio.fixture(scope="function", autouse=True)
async def _setup_tables():
    """Create/destroy tables before/after each test."""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)


@pytest_asyncio.fixture(scope="function")
async def db_session():
    """Yield a session wrapped in a transaction that rolls back after each test."""
    async with AsyncSessionLocal() as session:
        try:
            yield session
        finally:
            await session.rollback()
            await session.close()


@pytest_asyncio.fixture(scope="function")
async def test_namespace(db_session):
    ns = Namespace(name="test-ns", id=uuid.uuid4())
    db_session.add(ns)
    await db_session.commit()
    return ns


@pytest_asyncio.fixture(scope="function")
async def test_collection(db_session, test_namespace):
    coll = Collection(
        id=uuid.uuid4(),
        namespace_id=test_namespace.id,
        name="test-collection",
        strategy="vector",
        embedding_dimensions=256,
    )
    db_session.add(coll)
    await db_session.commit()
    return coll


@pytest_asyncio.fixture(scope="function")
async def test_graph_rag_collection(db_session, test_namespace):
    coll = Collection(
        id=uuid.uuid4(),
        namespace_id=test_namespace.id,
        name="graph-rag-collection",
        strategy="custom_graph_rag",
        embedding_dimensions=256,
    )
    db_session.add(coll)
    await db_session.commit()
    return coll


@pytest_asyncio.fixture(scope="function")
async def test_light_rag_collection(db_session, test_namespace):
    coll = Collection(
        id=uuid.uuid4(),
        namespace_id=test_namespace.id,
        name="light-rag-collection",
        strategy="light_rag",
        embedding_dimensions=256,
    )
    db_session.add(coll)
    await db_session.commit()
    return coll


@pytest_asyncio.fixture(scope="function")
async def async_client(test_namespace):
    transport = ASGITransport(app=app)
    headers = {"X-Namespace-ID": str(test_namespace.id)}
    async with AsyncClient(transport=transport, base_url="http://test", headers=headers) as client:
        yield client


@pytest.fixture
def service():
    return GraphService()
