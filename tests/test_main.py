"""Tests for the FastAPI application entry point."""

import pytest

from graph_core import main as main_module


class _FakeSession:
    async def run_sync(self, fn):
        assert fn is main_module.load_namespace_acl_payloads
        return [
            ("namespace-id", "tenant_user", "secret", "redis://localhost:6379"),
        ]


class _FakeSessionLocal:
    async def __aenter__(self):
        return _FakeSession()

    async def __aexit__(self, exc_type, exc, tb):  # noqa: ANN001
        return False


@pytest.mark.asyncio
async def test_replay_namespace_falkordb_acls_replays_payloads(monkeypatch):
    replayed: list[list[tuple[str, str, str, str | None]]] = []

    async def _fake_replay(payloads):
        replayed.append(payloads)

    monkeypatch.setattr(main_module, "AsyncSessionLocal", lambda: _FakeSessionLocal())
    monkeypatch.setattr(
        main_module,
        "replay_namespace_acl_payloads",
        _fake_replay,
    )

    await main_module._replay_namespace_falkordb_acls()

    assert replayed == [
        [("namespace-id", "tenant_user", "secret", "redis://localhost:6379")]
    ]
