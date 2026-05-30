"""MCP client helper — connects to Graph Core via streamable HTTP transport.

Uses the SDK low-level ClientSession and streamable_http_client directly,
without importing anything from graph_core.client.
"""

from mcp import ClientSession, types
from mcp.client.streamable_http import streamable_http_client


class MCPClient:
    """Thin wrapper around ClientSession + streamable_http_client."""

    def __init__(self, mcp_url: str) -> None:
        self.mcp_url = mcp_url if mcp_url.endswith("/") else mcp_url + "/"
        self._session: ClientSession | None = None
        self._transport_ctx = None

    async def connect(self) -> None:
        try:
            transport = streamable_http_client(self.mcp_url)
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
        except Exception as e:
            raise RuntimeError(
                f"Failed to connect to MCP server at {self.mcp_url}: {e}"
            ) from e

    async def disconnect(self) -> None:
        if self._session:
            await self._session.__aexit__(None, None, None)
            self._session = None
        if self._transport_ctx:
            await self._transport_ctx.__aexit__(None, None, None)
            self._transport_ctx = None

    async def call(self, tool_name: str, arguments: dict | None = None) -> str:
        if self._session is None:
            raise RuntimeError("MCPClient not connected; call connect() first")
        result = await self._session.call_tool(tool_name, arguments=arguments or {})
        parts: list[str] = []
        for block in result.content:
            if isinstance(block, types.TextContent):
                parts.append(block.text)
        return "\n".join(parts) if parts else ""

    async def list_tools(self) -> list[str]:
        if self._session is None:
            raise RuntimeError("MCPClient not connected; call connect() first")
        resp = await self._session.list_tools()
        return [t.name for t in resp.tools]
