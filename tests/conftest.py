"""Shared test fixtures."""

import asyncio
import uuid

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

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


@pytest_asyncio.fixture(scope="function")
async def db_session():
    """Create tables and yield a session wrapped in a transaction that rolls back."""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
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
    )
    db_session.add(coll)
    await db_session.commit()
    return coll


@pytest_asyncio.fixture(scope="function")
async def async_client(test_namespace):
    """HTTPX async client with namespace header."""
    transport = ASGITransport(app=app)
    headers = {"X-Namespace-ID": str(test_namespace.id)}
    async with AsyncClient(transport=transport, base_url="http://test", headers=headers) as client:
        yield client


@pytest.fixture
def service():
    return GraphService()
