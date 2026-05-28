"""Collections API — integration tests."""

import pytest


@pytest.mark.asyncio
async def test_create_collection(async_client, test_namespace):
    resp = await async_client.post(
        "/collections/",
        json={"name": "api-test-collection", "strategy": "vector"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["name"] == "api-test-collection"
    assert data["strategy"] == "vector"
    assert data["namespace_id"] == str(test_namespace.id)


@pytest.mark.asyncio
async def test_list_collections(async_client, test_namespace, test_collection):
    resp = await async_client.get("/collections/")
    assert resp.status_code == 200
    data = resp.json()
    assert any(c["name"] == "test-collection" for c in data)


@pytest.mark.asyncio
async def test_create_collection_missing_namespace(async_client):
    resp = await async_client.post(
        "/collections/",
        json={"name": "orphan"},
        headers={"X-Namespace-ID": ""},
    )
    assert resp.status_code == 401  # Missing auth
