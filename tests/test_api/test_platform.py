"""Platform API — integration tests."""

import pytest


@pytest.mark.asyncio
async def test_get_capabilities(async_client):
    resp = await async_client.get("/platform/capabilities")
    assert resp.status_code == 200
    data = resp.json()
    assert "embedding_profiles" in data
    assert "llm_profiles" in data
    assert "retrieval_strategies" in data
    assert "vector" in data["retrieval_strategies"]
    assert "custom_graph_rag" in data["retrieval_strategies"]


@pytest.mark.asyncio
async def test_register_credential(async_client):
    resp = await async_client.post(
        "/platform/credentials",
        json={"provider": "openai", "secret": "sk-fake", "label": "test"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["provider"] == "openai"
    assert data["label"] == "test"
    assert "credential_id" in data


@pytest.mark.asyncio
async def test_create_and_list_profiles(async_client):
    cred_resp = await async_client.post(
        "/platform/credentials",
        json={"provider": "openai", "secret": "sk-fake", "label": "test"},
    )
    credential_id = cred_resp.json()["credential_id"]

    profile_resp = await async_client.post(
        "/platform/profiles",
        json={
            "kind": "embedding",
            "provider": "local_hash",
            "model": "hash-256",
            "credential_id": credential_id,
            "label": "embed-default",
            "dimensions": 256,
            "distance_metric": "cosine",
        },
    )
    assert profile_resp.status_code == 200

    list_resp = await async_client.get("/platform/embedding-profiles")
    assert list_resp.status_code == 200
    profiles = list_resp.json()
    assert len(profiles) == 1
    assert profiles[0]["provider"] == "local_hash"
