"""MCP client with auth header support.

Sends Authorization: Bearer <key> on every HTTP request through the
streamable HTTP transport, so the MCP server can extract credentials
from the incoming request instead of relying on env vars.
"""

import httpx
from mcp import ClientSession, types
from mcp.client.streamable_http import streamable_http_client


class AuthenticatedMCPClient:
    """MCP client that passes auth headers on every request."""

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
            await self._session.__aexit__(None, None, None)
            self._session = None
        if self._transport_ctx:
            await self._transport_ctx.__aexit__(None, None, None)
            self._transport_ctx = None
        if self._http_client:
            await self._http_client.aclose()
            self._http_client = None

    async def call(self, tool_name: str, arguments: dict | None = None) -> str:
        if self._session is None:
            raise RuntimeError("Not connected; call connect() first")
        result = await self._session.call_tool(tool_name, arguments=arguments or {})
        parts: list[str] = []
        for block in result.content:
            if isinstance(block, types.TextContent):
                parts.append(block.text)
        return "\n".join(parts) if parts else ""
