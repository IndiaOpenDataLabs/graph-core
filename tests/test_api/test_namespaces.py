"""Namespace API — integration tests."""

import jwt
import pytest
from httpx import ASGITransport, AsyncClient

from graph_core.config import settings
from graph_core.main import app


def _admin_token() -> str:
    return jwt.encode(
        {
            "sub": "admin-1",
            "scope": "graph-core:admin",
            "token_type": "admin",
        },
        settings.jwt_secret,
        algorithm="HS256",
    )


@pytest.mark.asyncio
async def test_create_namespace_auto_provisions_falkordb_credential(db_session):
    transport = ASGITransport(app=app)
    headers = {"Authorization": f"Bearer {_admin_token()}"}
    async with AsyncClient(
        transport=transport, base_url="http://test", headers=headers
    ) as client:
        resp = await client.post(
            "/platform/namespaces/",
            json={"name": "tenant-alpha"},
        )

    assert resp.status_code == 200
    data = resp.json()
    assert data["name"] == "tenant-alpha"
    assert data["falkordb_db"] == 0
    assert data["scope"] == "graph-core:user"
    assert data["credential_id"]
    assert data["falkordb_username"]
    assert data["falkordb_secret"]
    assert data["falkordb_graph_pattern"].startswith("tenant:")
