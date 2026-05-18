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
async def test_register_credential_not_implemented(async_client):
    resp = await async_client.post(
        "/platform/credentials",
        json={"provider": "openai", "secret": "sk-fake", "label": "test"},
    )
    assert resp.status_code == 501  # Not Implemented
