"""MCP client with auth support.

Passes the API key through MCP protocol metadata on every tool call,
so the MCP server can extract it reliably regardless of transport.
"""

import asyncio
import contextlib

import httpx
from mcp import ClientSession, types
from mcp.client.streamable_http import streamable_http_client


class AuthenticatedMCPClient:
    """MCP client that passes API key via protocol metadata."""

    def __init__(self, mcp_url: str, api_key: str) -> None:
        self.mcp_url = mcp_url if mcp_url.endswith("/") else mcp_url + "/"
        self._api_key = api_key
        self._session: ClientSession | None = None
        self._transport_ctx = None
        self._http_client: httpx.AsyncClient | None = None

    async def connect(self) -> None:
        self._http_client = httpx.AsyncClient(
            headers={
                "Authorization": f"Bearer {self._api_key}",
            },
            follow_redirects=True,
            timeout=1800.0,
        )
        await self._http_client.__aenter__()

        transport = streamable_http_client(
            self.mcp_url,
            http_client=self._http_client,
        )
        self._transport_ctx = transport
        streams = await transport.__aenter__()
        if len(streams) == 2:
            read_stream, write_stream = streams
        elif len(streams) == 3:
            read_stream, write_stream, _ = streams
        else:
            raise RuntimeError(
                f"Unexpected stream count: {len(streams)}. "
                f"Check MCP library version compatibility."
            )
        self._session = ClientSession(read_stream, write_stream)
        await self._session.__aenter__()
        await self._session.initialize()

    async def disconnect(self) -> None:
        if self._session:
            with contextlib.suppress(Exception):
                await asyncio.wait_for(
                    self._session.__aexit__(None, None, None),
                    timeout=2.0,
                )
            self._session = None
        if self._transport_ctx:
            with contextlib.suppress(Exception):
                await asyncio.wait_for(
                    self._transport_ctx.__aexit__(None, None, None),
                    timeout=2.0,
                )
            self._transport_ctx = None
        if self._http_client:
            with contextlib.suppress(Exception):
                await asyncio.wait_for(self._http_client.aclose(), timeout=2.0)
            self._http_client = None

    async def call(self, tool_name: str, arguments: dict | None = None) -> str:
        if self._session is None:
            raise RuntimeError("Not connected; call connect() first")
        result = await self._session.call_tool(
            tool_name,
            arguments=arguments or {},
            meta={"api_key": self._api_key},
        )
        parts: list[str] = []
        for block in result.content:
            if isinstance(block, types.TextContent):
                parts.append(block.text)
        return "\n".join(parts) if parts else ""
