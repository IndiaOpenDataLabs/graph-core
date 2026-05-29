"""MCP server entry — run with: python -m graph_core.mcp"""

from graph_core.mcp.server import mcp

if __name__ == "__main__":
    import sys

    transport = sys.argv[1] if len(sys.argv) > 1 else "stdio"
    mcp.run(transport=transport)
