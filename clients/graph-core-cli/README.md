# Graph Core TUI

A terminal UI for managing the Graph Core platform, built with [Textual](https://textual.textualize.io/). Communicates with the backend exclusively via the MCP (Model Context Protocol) protocol.

## Quick Start

```bash
# Install CLI extras
uv sync

# Launch
make tui
# or
python -m graph_core_cli
# or
graph-core-tui
```

## First-Time Setup

On first launch, the TUI shows a setup screen asking for:

- **MCP URL** — defaults to `http://localhost:8001/mcp/`
- **Platform Admin Key** — the `PLATFORM_ADMIN_KEY` from your environment

Your configuration is persisted to `~/.config/graph-core/config.json`. Subsequent launches skip setup and go straight to the home dashboard. Reconfigure anytime from the home screen's "Reconfigure" button.

## Screens

### Home
Dashboard showing connection status, active namespace, and navigation to all other screens.

### Namespaces *(admin only)*
List and create namespaces. Press `a` to create, `r` to refresh.

### Collections
List and create collections within the current namespace. Choose strategy: `vector`, `light_rag`, or `custom_graph_rag`. Press `a` to create, `r` to refresh.

### Query
Select a collection, pick a query mode (`local`, `global`, `hybrid`, `naive`, `mix`), and type your question. Results appear in a scrollable log.

### Ingest
Paste text or provide a file path to ingest into a collection. Two methods:
- **Document (async)** — spawns a background job
- **Chunk (sync)** — processes immediately

### Jobs
Check the status of ingestion jobs by entering their UUID.

## Key Bindings

| Key | Action |
|-----|--------|
| `h` | Home |
| `c` | Config (Home) |
| `n` | Namespaces |
| `l` | Collections |
| `Shift+Q` | Query |
| `i` | Ingest |
| `j` | Jobs |
| `q` | Quit |
| `Esc` | Back (within a screen) |

## Architecture

```
graph-core-tui  →  MCP (streamable HTTP)  →  FastMCP server  →  FastAPI backend
   (Textual)       (mcp SDK)                (port 8001)         (port 8000)
```

The TUI never talks to the FastAPI REST API directly. Every action is an MCP tool call. See `PATTERNS.md` for how to add new screens and actions.
