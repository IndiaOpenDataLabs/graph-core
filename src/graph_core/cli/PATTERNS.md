# TUI Development Patterns

Patterns every developer should know when editing the Graph Core TUI.

## Entry Points

| Command | File | Function |
|---------|------|----------|
| `python -m graph_core.cli` | `cli/__main__.py` | `GraphCoreTUI().run()` |
| `graph-core-tui` | `cli/app.py` | `main()` |
| `make tui` | `Makefile` | delegates to module entry |

## Startup Flow

1. `GraphCoreTUI.on_mount()` loads persisted config from `~/.config/graph-core/config.json` via `cli/config.py`
2. If `api_key` is set → pushes `HomeScreen`
3. If no key → pushes `SetupScreen` (first-time only)
4. Config setter (`app.config = ...`) auto-saves to disk

## Config Persistence

- **Module**: `cli/config.py`
- **File**: `~/.config/graph-core/config.json`
- **Keys**: `mcp_url`, `api_key`, `is_admin`, `namespace_id`, `namespace_name`
- `save_config()` also populates `os.environ` so the running process picks up values immediately
- New config keys are merged with defaults on load, so adding a key never breaks older config files

## Screen Anatomy

Every screen in `cli/screens.py` follows the same pattern:

```python
class MyScreen(Screen):
    CSS = """..."""          # Textual CSS (inline string)
    BINDINGS = [...]         # screen-local key bindings

    def compose(self) -> None:
        yield Label(...)
        yield DataTable(...)
        # ...

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
- **Styling**: Inline `CSS` string on the class, not external files
- **Async work**: Always wrap MCP calls in `self.run_worker(coro, exclusive=True, group="load"|"action")`
- **MCP client**: Access via `self.app.mcp_client`. Connect → call → disconnect. Never hold a session open.
- **Navigation**: `self.app.push_screen(ScreenClass())` to push, `esc` binding (`"app.pop_screen"`) to go back

## Adding a New Screen

1. Define the `Screen` subclass in `cli/screens.py`
2. Add a `BINDING` in `GraphCoreTUI.BINDINGS` in `cli/app.py`
3. Add a corresponding `action_show_*` method in `app.py`
4. (Optional) Add a button on `HomeScreen`'s nav section

## MCP Communication

### The MCP Client (`cli/mcp_client.py`)

Thin wrapper around the `mcp` SDK's `ClientSession` + `streamable_http_client`. Every tool call is:

```python
client = self.app.mcp_client
await client.connect()
try:
    text = await client.call("tool_name", {"arg": "value"})
finally:
    await client.disconnect()
```

### Parsing Responses

MCP tools return formatted text strings, not JSON. The TUI parses them with regex helpers in `cli/screens.py`:

| Helper | Purpose |
|--------|---------|
| `parse_namespaces(text)` | List of namespaces from `list_namespaces` output |
| `parse_collections(text)` | List of collections from `list_collections` output |
| `parse_key_value(text)` | Generic key: value pairs |
| `extract_id(text)` | First `id: <uuid>` match |
| `extract_name(text)` | First `name: <string>` match |
| `extract_job_id(text)` | First `job_id: <uuid>` match |
| `extract_status(text)` | First `status: <value>` match |

If you add a new MCP tool, you may need to add a corresponding parser.

## Worker Groups

Two worker groups are used to manage concurrency:

- **`group="load"`** — data loading (list namespaces, list collections, etc.). Exclusive within group, so only one load runs at a time.
- **`group="action"`** — user actions (create, query, ingest). Exclusive within group.

This prevents multiple simultaneous MCP connections.

## Lazy Imports

Screen classes are imported inside action methods, not at module level:

```python
async def action_show_namespaces(self) -> None:
    from graph_core.cli.screens import NamespacesScreen
    self.push_screen(NamespacesScreen())
```

This avoids circular imports and reduces startup cost.

## Testing

Run the full test suite (includes TUI-specific tests):

```bash
make test
# or
uv run pytest tests/
```

For manual testing of a specific screen:

```bash
make tui
# Navigate with key bindings
```

## Common Pitfalls

- **Forgetting `await client.disconnect()`** — always use `try/finally` to clean up the session
- **Blocking the TUI** — never `await` an MCP call directly in an event handler; use `self.run_worker()` or `self.app.push_screen()` for navigation
- **Hardcoding URLs** — use `self.app.mcp_client` which reads from config
- **Not handling empty responses** — MCP tools can return empty strings; parsers should handle this gracefully
- **Widget IDs** — every widget you interact with programmatically needs a unique `id` for `self.query_one()`
