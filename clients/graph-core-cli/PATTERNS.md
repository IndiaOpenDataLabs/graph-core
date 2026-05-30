# TUI Development Patterns

Patterns every developer should know when editing the Graph Core TUI client.

## Package Location

The TUI lives in `clients/graph-core-cli/` — a separate package from the backend. It communicates with the backend exclusively via the MCP (Model Context Protocol) protocol with proper auth headers.

```
graph-core/
├── src/graph_core/           # Backend (FastAPI + MCP server)
├── clients/graph-core-cli/   # TUI client (this package)
│   ├── src/graph_core_cli/
│   │   ├── app.py            # Textual App + global bindings
│   │   ├── screens.py        # All TUI screens
│   │   ├── config.py         # Persistent config (~/.config/graph-core/)
│   │   └── mcp_client.py     # Authenticated MCP client wrapper
│   ├── pyproject.toml        # Standalone package
│   ├── README.md
│   └── PATTERNS.md           # This file
```

## Auth Flow

```
TUI Client  →  Authorization: Bearer <key>  →  MCP Server  →  Bearer <key>  →  FastAPI
  (httpx)       (on every HTTP request)        (extracts       (GraphCoreClient   (validates
                via streamable_http              from request)      sends it)          token)
```

The client sends `Authorization: Bearer <key>` on every MCP HTTP request via `httpx.AsyncClient`. The MCP server extracts the key from `ctx.request_context.request.headers` and passes it to `GraphCoreClient`. This works across machines.

## Entry Points

| Command | File | Function |
|---------|------|----------|
| `python -m graph_core_cli` | `__main__.py` | `GraphCoreTUI().run()` |
| `graph-core-tui` | `app.py` | `main()` |
| `make tui` | `Makefile` | runs from `clients/graph-core-cli/` |

## Startup Flow

1. `GraphCoreTUI.on_mount()` loads persisted config from `~/.config/graph-core/config.json`
2. If `api_key` is set → pushes `HomeScreen`
3. If no key → pushes `SetupScreen` (first-time only)
4. Config setter (`app.config = ...`) auto-saves to disk

## Config Persistence

- **Module**: `config.py`
- **File**: `~/.config/graph-core/config.json`
- **Keys**: `mcp_url`, `api_key`, `is_admin`, `namespace_id`, `namespace_name`

## MCP Client

The `AuthenticatedMCPClient` in `mcp_client.py` wraps the MCP SDK's `streamable_http_client` with an `httpx.AsyncClient` that injects the `Authorization: Bearer <key>` header on every request.

```python
client = self.app.mcp_client  # AuthenticatedMCPClient
await client.connect()
try:
    text = await client.call("tool_name", args)
finally:
    await client.disconnect()
```

## Screen Anatomy

Every screen in `screens.py` follows the same pattern:

```python
class MyScreen(Screen):
    CSS = """..."""          # Textual CSS (inline string)
    BINDINGS = [...]         # screen-local key bindings

    def compose(self) -> None:
        yield Label(...)
        yield DataTable(...)

    async def on_mount(self) -> None:
        self.run_worker(self._load_data(), exclusive=True, group="load")

    async def _load_data(self) -> None:
        client = self.app.mcp_client
        await client.connect()
        try:
            text = await client.call("tool_name", args)
        finally:
            await client.disconnect()
        # update widgets
```

### Key conventions:
- **Styling**: Inline `CSS` string on the class
- **Async work**: Always wrap MCP calls in `self.run_worker(coro, exclusive=True, group="load"|"action")`
- **MCP client**: Access via `self.app.mcp_client`. Connect → call → disconnect.
- **Navigation**: `self.app.push_screen(ScreenClass())` to push, `esc` to pop

## Adding a New Screen

1. Define the `Screen` subclass in `screens.py`
2. Add a `BINDING` in `GraphCoreTUI.BINDINGS` in `app.py`
3. Add a corresponding `action_show_*` method in `app.py`
4. (Optional) Add a button on `HomeScreen`'s nav section

## Parsing MCP Responses

MCP tools return formatted text strings, not JSON. The TUI parses them with regex helpers in `screens.py`:

| Helper | Purpose |
|--------|---------|
| `parse_namespaces(text)` | List of namespaces |
| `parse_collections(text)` | List of collections |
| `parse_key_value(text)` | Generic key: value pairs |
| `extract_id(text)` | First `id: <uuid>` match |
| `extract_name(text)` | First `name: <string>` match |
| `extract_job_id(text)` | First `job_id: <uuid>` match |
| `extract_status(text)` | First `status: <value>` match |

## Worker Groups

- **`group="load"`** — data loading. Only one load runs at a time.
- **`group="action"`** — user actions. Only one action runs at a time.

## Lazy Imports

Screen classes are imported inside action methods, not at module level. Avoids circular imports and reduces startup cost.

## Testing

```bash
# From the project root
cd clients/graph-core-cli && uv sync && uv run python -m graph_core_cli
```

## Common Pitfalls

- **Forgetting `await client.disconnect()`** — always use `try/finally`
- **Blocking the TUI** — never `await` an MCP call directly in an event handler
- **Hardcoding URLs** — use `self.app.mcp_client` which reads from config
- **Not handling empty responses** — MCP tools can return empty strings
- **Widget IDs** — every widget you interact with programmatically needs a unique `id`
