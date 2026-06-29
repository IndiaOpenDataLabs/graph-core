from __future__ import annotations

import asyncio

import httpx
import pytest

from graph_core_cli.mcp_client import AuthenticatedMCPClient


class _FakeHttpClient:
    def __init__(self) -> None:
        self.closed = False

    async def __aenter__(self) -> "_FakeHttpClient":
        return self

    async def aclose(self) -> None:
        self.closed = True


class _FailingTransport:
    async def __aenter__(self):  # noqa: ANN201
        raise httpx.ConnectError(
            "All connection attempts failed",
            request=httpx.Request("POST", "http://localhost:18103/mcp/"),
        )

    async def __aexit__(self, exc_type, exc, tb) -> None:  # noqa: ANN001,ANN202
        return None


class _CancelledExit:
    async def __aexit__(self, exc_type, exc, tb) -> None:  # noqa: ANN001,ANN202
        raise asyncio.CancelledError()


@pytest.mark.asyncio
async def test_connect_failure_cleans_up_and_raises_runtime_error(monkeypatch):
    fake_http_client = _FakeHttpClient()

    monkeypatch.setattr(
        "graph_core_cli.mcp_client.httpx.AsyncClient",
        lambda **kwargs: fake_http_client,
    )
    monkeypatch.setattr(
        "graph_core_cli.mcp_client.streamable_http_client",
        lambda *args, **kwargs: _FailingTransport(),
    )

    client = AuthenticatedMCPClient("http://localhost:18103/mcp/", "token")

    with pytest.raises(RuntimeError, match="Unable to connect to MCP server"):
        await client.connect()

    assert fake_http_client.closed is True
    assert client._session is None
    assert client._transport_ctx is None
    assert client._http_client is None


@pytest.mark.asyncio
async def test_disconnect_swallows_cancelled_error():
    client = AuthenticatedMCPClient("http://localhost:18103/mcp/", "token")
    client._session = _CancelledExit()
    client._transport_ctx = _CancelledExit()
    client._http_client = _FakeHttpClient()

    await client.disconnect()

    assert client._session is None
    assert client._transport_ctx is None
    assert client._http_client is None
