# CLI Development Patterns

Patterns every developer should know when editing the Graph Core CLI client.

## Package Location

The CLI lives in `clients/graph-core-cli/` as a separate package from the backend.
It talks to the backend only through MCP over HTTP with bearer auth.

```text
graph-core/
├── src/graph_core/           # Backend (FastAPI + MCP server + workers)
├── clients/graph-core-cli/   # Textual CLI client
│   ├── src/graph_core_cli/
│   │   ├── app.py            # Textual App bootstrap
│   │   ├── screens.py        # Console + guided modal flows
│   │   ├── config.py         # Persistent config (~/.config/graph-core/)
│   │   └── mcp_client.py     # Authenticated MCP client wrapper
│   ├── pyproject.toml
│   ├── README.md
│   └── PATTERNS.md
```

## Current UX Model

This is no longer a multi-screen CRUD TUI.

The current client is a keyboard-first slash-command console:
- `SetupScreen` handles first-run MCP URL and API key setup
- `ConsoleScreen` is the main surface
- guided modal flows are used only for structured operations such as:
  - profile creation
  - collection creation/edit
  - delete confirmation

The console is intentionally closer to `opencode` than to a traditional form-heavy TUI:
- slash commands for navigation and actions
- autocomplete and history
- terminal-native text selection
- minimal mouse dependence

## Entry Points

| Command | File | Function |
|---------|------|----------|
| `python -m graph_core_cli` | `__main__.py` | `main()` |
| `graph-core-tui` | `app.py` | `main()` |
| `make tui` | root `Makefile` | runs from `clients/graph-core-cli/` |

## Startup Flow

1. `GraphCoreTUI.on_mount()` loads persisted config from `~/.config/graph-core/config.json`
2. If an active API key exists, it pushes `ConsoleScreen`
3. If not, it pushes `SetupScreen`
4. Config writes go through `self.app.config = ...`, which auto-saves

## Config Persistence

- Module: `config.py`
- File: `~/.config/graph-core/config.json`
- Current keys:
  - `mcp_url`
  - `api_key`
  - `admin_api_key`
  - `namespace_api_key`
  - `active_api_key_kind`
  - `is_admin`
  - `namespace_id`
  - `namespace_name`

Important behavior:
- admin and namespace keys are persisted separately
- namespace context shown in the UI is local client state until refreshed/verified
- admin mode clears displayed namespace context

## Auth Flow

```text
CLI  →  Authorization: Bearer <key>  →  MCP Server  →  GraphCoreClient  →  FastAPI
```

The client sends `Authorization: Bearer <key>` on every MCP HTTP request via
`httpx.AsyncClient`. The MCP server extracts the key from
`ctx.request_context.request.headers` and uses it when calling the backend API.

## MCP Client Pattern

`AuthenticatedMCPClient` in `mcp_client.py` wraps the MCP SDK streamable HTTP client.

Use this pattern for all direct MCP calls:

```python
client = self.app.mcp_client_for_key(self.app.active_api_key)
await client.connect()
try:
    text = await client.call("tool_name", args)
finally:
    await client.disconnect()
```

Rules:
- always `connect()` / `disconnect()` in `try/finally`
- never keep long-lived MCP sessions inside the screen
- prefer `self._call(...)` helpers inside `ConsoleScreen` where possible

## Console Anatomy

`ConsoleScreen` in `screens.py` is the primary app surface.

Main widgets:
- `#context` shows MCP URL, active key kind, and namespace context
- `#output` is a read-only `TextArea`
- `#command` is the prompt input
- `#suggestions` renders autocomplete results

Key interaction patterns:
- `/` starts command discovery
- `Tab` accepts the highlighted suggestion
- `Up` / `Down`:
  - navigate suggestions while typing and suggestions are visible
  - otherwise navigate command history
- `Ctrl+C` clears the current command input
- `/quit` or `q` exits

## Copy and Selection

The app runs with `mouse=False` in `app.py`.

That is deliberate:
- terminal-native selection works more reliably this way
- it matches the general `opencode` pattern of letting the terminal own selection

In-app copy fallbacks:
- `/copy`
- `Ctrl+Y`
- `y` when the output pane is focused

On macOS, the app writes to the real clipboard with `pbcopy`, not just Textual's
clipboard hook.

## Command Handling

All slash commands are dispatched from `ConsoleScreen._execute_command()`.

Current top-level commands:
- `/help`
- `/status`
- `/copy`
- `/clear`
- `/quit`
- `/config ...`
- `/auth ...`
- `/namespace ...`
- `/profile ...`
- `/collection ...`
- `/enhance ...`
- `/ingest ...`
- `/query ...`
- `/jobs ...`

Guideline:
- keep command dispatch centralized
- prefer small `_command_*` methods for each domain
- keep parsing logic local to the command handler unless reused broadly

## Modal Flow Pattern

Structured create/edit flows live as separate `Screen` classes in `screens.py`:
- `ProfileCreateScreen`
- `CollectionFormScreen`
- `ConfirmScreen`

Use modal flows when:
- the command has too many fields for a good slash-only experience
- there are enum or profile selection choices
- the flow benefits from validation before the MCP call

Use slash-only when:
- the command is short and repeatable
- it is power-user oriented

Pattern:

```python
self.app.push_screen(
    SomeFormScreen(...),
    self._handle_modal_result,
)
```

## Parsing MCP Responses

MCP tools still return formatted text, not JSON, so the CLI parses them with regex helpers.

Important helpers in `screens.py`:
- `parse_namespaces(text)`
- `parse_collections(text)`
- `parse_profiles(text, kind)`
- `parse_jobs(text)`
- `extract_id(text)`
- `extract_name(text)`
- `extract_api_key(text)`
- `extract_job_id(text)`

If you change MCP response formatting in the backend, update these parsers immediately.

## File Ingest Pattern

`/ingest file ...` reads the file locally in the CLI and sends the file contents to MCP.

This is intentional:
- the backend runs in Docker
- host file paths are not meaningful inside the container

Do not revert this to passing raw file paths through to the backend.

`/ingest dir ...` is also intentionally CLI-local:
- directory walking happens on the host
- `.gitignore` / `.dockerignore` are read from the provided directory root
- matching files and directories are excluded before enqueueing
- remaining files are sent one by one as normal `ingest_document` requests

This keeps host-path semantics and ignore behavior in the client, where the
paths actually exist.

## Job Tracking Pattern

Async file/document ingestion returns a `job_id`.

The CLI supports:
- `/jobs list`
- `/jobs show <job_id>`
- `/jobs watch <job_id>`
- `last` alias after `/ingest file ...`

`ConsoleScreen` keeps `_last_job_id` so recent ingestion flows can do:
- `/jobs show last`
- `/jobs watch last`

For `/ingest dir ...`, multiple jobs may be started, one per file.
The CLI currently stores the most recent returned `job_id` in `_last_job_id`.

## Enhance Pattern

`/enhance <collection> [--levels N]` is a collection-scoped operation that
rebuilds one or more higher-level meta collections from the selected
collection's current canonical graph.

Design intent:
- the CLI should not know how derived graphs are built
- it should call a first-class collection operation through MCP
- the backend owns:
  - graph analysis
  - derived summary generation
  - derived Falkor graph persistence
  - derived vector-summary persistence

So `/enhance` belongs next to collection operations, not under ingest or jobs.

## Profile and Collection Patterns

Profiles now support concurrency tuning for OpenAI-compatible providers:
- `max_concurrent_calls`

Important backend behavior:
- concurrency is profile-scoped when set
- fallback comes from global env defaults when unset
- Redis semaphore keys are per profile, not just per provider type

CLI implications:
- guided profile creation must expose `max_concurrent_calls`
- profile list parsing must tolerate additional formatted fields

Collections:
- must use an embedding profile
- may optionally use an LLM profile

## Provider Concurrency Patterns

The backend throttles OpenAI-compatible provider calls with Redis-backed semaphores.

Design rules:
- semaphore scope must be keyed by profile ID when a profile is known
- global env defaults are only fallbacks
- local hash embedding and local echo LLM should remain unaffected

Do not implement concurrency caps only at the worker level. The throttle belongs at
the provider-call layer so it works across processes and containers.

## Worker and Time-Limit Patterns

Current ingestion worker behavior:
- `run_ingestion` has no time limit
- `run_chunk` uses `INGEST_CHUNK_TIME_LIMIT_MS`

Reason:
- local LLM-backed chunk processing can legitimately take minutes
- the parent document-ingestion actor should not be killed by a blanket limit

If you touch worker execution:
- preserve the no-limit parent actor behavior
- keep chunk timeouts configurable
- remember that timeout failures should not leave chunk rows stuck forever in
  `processing`

## Testing

Useful checks from the repo root:

```bash
uv run ruff check clients/graph-core-cli/src/graph_core_cli
uv run python -m compileall clients/graph-core-cli/src/graph_core_cli
cd clients/graph-core-cli && uv run python -m graph_core_cli
```

When backend profile or worker behavior changes, also test from the repo root:

```bash
uv run ruff check src
uv run python -m compileall src
```

## Common Pitfalls

- Forgetting `await client.disconnect()` after MCP calls
- Reintroducing mouse-heavy UI in a keyboard-first console
- Assuming right-click copy can be owned by the app
- Changing MCP response text without updating CLI parsers
- Passing host file paths to Docker instead of reading files locally in the CLI
- Using one global provider semaphore for all profiles
- Confusing namespace key state in config with verified backend namespace state
- Adding background MCP workers on mount/resume that make shutdown brittle
