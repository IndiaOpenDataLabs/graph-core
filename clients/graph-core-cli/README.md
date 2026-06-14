# Graph Core CLI

Terminal client for Graph Core, built with [Textual](https://textual.textualize.io/). The CLI talks only to the MCP server exposed by the Docker stack.

## Quick Start

```bash
make docker-up

cd clients/graph-core-cli
uv sync
uv run python -m graph_core_cli
```

The default MCP endpoint is `http://localhost:8001/mcp/`.

## First Run

On first launch, the CLI asks for:

- `MCP URL`
- `API Key`

Use:
- an admin JWT for namespace management
- a namespace token for namespace-scoped operations

Config is stored in `~/.config/graph-core/config.json`.

## Interaction Model

The current UI is a hybrid:

- a slash-command console for navigation and power-user flows
- guided modal forms for structured create/edit/delete tasks
- autocomplete for slash commands
- `@` file completion for file ingestion

Keyboard behavior:

- `Tab`: accept current autocomplete suggestion
- `Up` / `Down`: cycle suggestions, otherwise command history
- `q`, `/quit`, `/exit`: quit

## Main Commands

### Status and Config

- `/help`
- `/status`
- `/config show`
- `/config set-url <url>`
- `/auth set-key <token> [--kind admin|namespace|auto]`
- `/auth use admin`
- `/auth use namespace`

### Namespaces

- `/namespace list`
- `/namespace create <name>`
- `/namespace current`
- `/namespace rotate-key <id-or-name>`

### Profiles

- `/profile list`
- `/profile list embedding`
- `/profile list llm`
- `/profile create`

`/profile create` opens a guided form for:
- kind
- provider
- model
- secret
- base URL
- dimensions
- distance metric
- label

### Collections

- `/collection list`
- `/collection create`
- `/collection edit <collection>`
- `/collection delete <collection>`
- `/enhance <collection> [--levels N]`

The create/edit flows open guided forms with:
- name
- strategy
- embedding profile
- optional LLM profile
- optional default query mode

Supported strategies:
- `vector`
- `light_rag`
- `custom_graph_rag`

Supported default query modes:
- `local`
- `global`
- `hybrid`
- `naive`
- `mix`

`/enhance <collection> [--levels N]` rebuilds one or more higher-level meta
collections. Running it on a base collection builds `__meta__l1`, `__meta__l2`,
and so on. Running it on an existing meta collection starts from that level and
builds the next levels above it.

### Ingestion

- `/ingest chunk <collection> "<text>"`
- `/ingest file <collection> @<path>`
- `/ingest dir <collection> @<path>`

Notes:
- `chunk` is synchronous
- `file` is asynchronous and returns a `job_id`
- `dir` walks the directory recursively and enqueues one document-ingest job per file
- `@` triggers file autocomplete in the repo/workspace

Directory ingest behavior:

- uses the provided directory as the walk root
- reads `.gitignore` and `.dockerignore` from that directory if present
- excludes matching files and directories before enqueueing
- also skips common local/cache directories such as:
  - `.git`
  - `.venv`
  - `node_modules`
  - `__pycache__`

The ignore handling is practical rather than a complete Git-spec parser, but it
covers the common project patterns.

### Query

- `/query <collection> "<question>"`
- `/query <collection> "<question>" --mode local`
- `/query <collection> "<question>" --chat-id <chat_id>`

### Chat Sessions

- `/chat create <collection> [--title <title>]`
- `/chat list <collection> [--limit <n>]`

Chat sessions are explicit. The CLI does not keep an implicit active chat yet.

Use `/chat create` first, then pass the returned `chat_id` to `/query`.

Behind the scenes:

- chronology is stored as ordered role-based messages
- semantic follow-up memory is stored separately and retrieved by provenance from those messages
- short follow-ups are rewritten into standalone queries from that recovered context

### Jobs

- `/jobs list`
- `/jobs list --limit 50`
- `/jobs show <job_id>`
- `/jobs show last`
- `/jobs watch <job_id>`
- `/jobs watch last`

`/jobs watch` polls once per second and stops when the job reaches `completed`, `failed`, or `cancelled`.

## Autocomplete

Typing `/` suggests commands.

Typing `@` suggests files from the workspace. This is especially useful for:

```text
/ingest file coll1 @docs/large-document.txt
/ingest dir coll1 @src/
```

Autocomplete is intentionally lightweight today:

- command suggestions after `/`
- strategy suggestions after `--strategy`
- query mode suggestions after `--default-query-mode`
- chat mode suggestions after `--mode`
- file suggestions after `@`

## Architecture

```text
graph-core-cli  ->  MCP (streamable HTTP)  ->  FastMCP server  ->  FastAPI backend
   (Textual)          (mcp SDK)                (port 8001)         (port 8000)
```

The CLI does not call the FastAPI API directly. All operations go through MCP tools.
