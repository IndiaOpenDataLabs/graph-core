# Graph Core Platform

AI-native knowledge infrastructure platform.

## Architecture

```
Applications / Clients
    └── Graph Platform (this repo)
            ├── Transport (REST, MCP, SSE)
            ├── Control Plane (collections, jobs, profiles, credentials)
            ├── Execution (Dramatiq workers, durable Postgres state)
            ├── Graph Runtime (ingestion, query, extraction, retrieval)
            └── Storage (Postgres, Vector DB, Graph DB, Redis)
```

## Starting the Stack

The entire stack (infrastructure + app + worker) runs in Docker. One command starts everything:

```bash
make docker-up
```

This builds and starts:
- **Postgres** with `pgvector` (port 5432)
- **FalkorDB** (graph DB, port 6379)
- **Redis** (dramatiq + SSE, port 6380)
- **App** (FastAPI + MCP, port 8001)
- **Worker** (Dramatiq background jobs)

### Recommended local flow

1. Install dependencies (for migrations, linting, tests):

```bash
uv sync --all-groups
```

2. Start the full stack:

```bash
make docker-up
```

Migrations run automatically on app start. Verify infrastructure if needed:

```bash
make infra-check
```

### Useful commands

```bash
make docker-logs        # Follow all logs
make docker-logs-app    # Follow app logs
make docker-logs-worker # Follow worker logs
make docker-ps          # List running containers
make docker-down        # Stop all services
make docker-clean       # Stop and remove all containers, volumes, networks
```

### Schema note

This repo uses Alembic migrations. They run automatically when the app container starts.

If your local Postgres volume was created before recent schema changes, recreate
it and rerun migrations:

```bash
make docker-clean
make docker-up
```

## Concepts

- **Namespace** — top-level isolation boundary (e.g., `scripture-assistant-prod`)
- **Collection** — knowledge graph scoped to a namespace, binds strategy + embedding profile
- **Chat Session** — follow-up query scope attached to a collection
- **Job** — durable async unit of work, tracked in Postgres
- **Profile** — reusable configuration for embeddings or LLMs
- **Credential** — encrypted secret reference, bound to namespace

## Chat Sessions

Chat follow-ups are managed with two layers:

- **Chronology**: ordered `chat_messages` in Postgres with `role`, `message_index`, and `turn_index`
- **Semantic memory**: a chat-specific Falkor graph plus vectorized semantic memory extracted from chat messages

Each extracted semantic node/edge carries provenance:

- `source_message_ids`
- `source_roles`

This supports two different follow-up behaviors:

- **Ordinary follow-up**: retrieve relevant semantic memory and rewrite the short follow-up into a standalone query
- **Correction / disagreement**: use recent chronological messages to anchor the relevant message IDs, then pull the semantic region sourced from those IDs

Current implementation note:

- chat semantic extraction runs synchronously in the query path today
- it is not backgrounded yet like document ingestion

## Clients

The platform supports three ways to connect:

### Terminal UI (TUI)

Interactive terminal application for managing namespaces, collections, queries, ingestion, and jobs. Lives in `clients/graph-core-cli/`.

```bash
make docker-up

cd clients/graph-core-cli
uv sync
uv run python -m graph_core_cli
```

The TUI talks to the MCP server exposed by the Docker stack at `http://localhost:8001/mcp/`. On first launch, it prompts for your MCP URL and API key. Configuration is persisted to `~/.config/graph-core/config.json`. Once connected, use key bindings to navigate:

| Key     | Screen        | Description                          |
|---------|---------------|--------------------------------------|
| `h`     | Home          | Dashboard overview                   |
| `n`     | Namespaces    | List/create namespaces (admin JWT)   |
| `p`     | Profiles      | Create/list embedding and LLM profiles |
| `l`     | Collections   | List/create collections              |
| `Shift+Q` | Query       | Query a collection with NL           |
| `i`     | Ingest        | Ingest text or files into a collection |
| `j`     | Jobs          | Track async ingestion jobs           |
| `c`     | Config        | Reconfigure connection               |
| `q`     | Quit          | Exit                                 |

### MCP Server

Exposes all platform operations as MCP tools, compatible with Claude Desktop,
Claude Code, and any MCP client.

```bash
make docker-up
```

The MCP server is part of the full Docker stack and is exposed over streamable HTTP at `http://localhost:8001/mcp/`. The same app process also serves the REST API on `http://localhost:8001/`. You do not need to run a separate `make server` target.

Configure via environment variables:

| Env Var                  | Description                          |
|--------------------------|--------------------------------------|
| `GRAPH_CORE_URL`         | Platform base URL (default: localhost:8001) |
| `GRAPH_CORE_API_KEY`     | Namespace API key                    |
| `GRAPH_CORE_ADMIN_JWT`   | Admin JWT for namespace management   |

JWT bearer tokens are also supported for external MCP/API clients:

- `graph-core:admin` scope exposes namespace-management tools only
- `graph-core:user` scope exposes namespace-scoped tools only
- user tokens must include a `namespace_id` claim
- set `JWT_SECRET` to enable JWT verification

Generate an admin token:

```bash
uv run graph-core-admin-jwt
```

Or, if you want to use a custom subject:

```bash
uv run graph-core-admin-jwt --subject my-mcp-client
```

**Available tools:**

| Tool                   | Description                        |
|------------------------|------------------------------------|
| `create_namespace`     | Create a new namespace (admin)     |
| `list_namespaces`      | List all namespaces (admin)        |
| `get_current_namespace`| Get current namespace info         |
| `rotate_namespace_key` | Rotate a namespace API key (admin) |
| `create_embedding_profile` | Create an embedding profile     |
| `create_llm_profile`   | Create an LLM profile              |
| `list_embedding_profiles` | List embedding profiles         |
| `list_llm_profiles`    | List LLM profiles                  |
| `create_collection`    | Create a collection                |
| `list_collections`     | List collections in namespace      |
| `ingest_chunk`         | Ingest a text chunk (sync)         |
| `ingest_document`      | Ingest a document (async, returns job_id) |
| `ingest_file`          | Ingest from a local file path      |
| `query_collection`     | Query a collection with a question |
| `create_chat_session`  | Create a chat session for follow-up memory |
| `list_chat_sessions`   | List chat sessions for a collection |
| `get_job_status`       | Check async job status             |
| `get_capabilities`     | List platform capabilities         |

### Shared HTTP Client (Python)

The `GraphCoreClient` class provides a typed async client for all REST endpoints.

```python
from graph_core.client import GraphCoreClient

async with GraphCoreClient(
    base_url="http://localhost:8000",
    api_key="ns_key_...",
) as client:
    collections = await client.list_collections()
    result = await client.query_collection(collection_id, "What is dharma?")
```

## Platform Setup

All control-plane endpoints are namespace-scoped through `X-Namespace-ID`.

### 1. Register a credential

```bash
curl -X POST http://localhost:8000/platform/credentials \
  -H "Content-Type: application/json" \
  -H "X-Namespace-ID: <namespace-uuid>" \
  -d '{
    "provider": "openai",
    "secret": "sk-...",
    "label": "openai-prod"
  }'
```

Response:

```json
{
  "credential_id": "cred-uuid",
  "provider": "openai",
  "label": "openai-prod"
}
```

### 2. Create an embedding profile

For local draft work with no external API calls:

```bash
curl -X POST http://localhost:8000/platform/profiles \
  -H "Content-Type: application/json" \
  -H "X-Namespace-ID: <namespace-uuid>" \
  -d '{
    "kind": "embedding",
    "provider": "local_hash",
    "model": "hash-256",
    "label": "local-embed",
    "dimensions": 256,
    "distance_metric": "cosine"
  }'
```

For real OpenAI embeddings:

```bash
curl -X POST http://localhost:8000/platform/profiles \
  -H "Content-Type: application/json" \
  -H "X-Namespace-ID: <namespace-uuid>" \
  -d '{
    "kind": "embedding",
    "provider": "openai",
    "model": "text-embedding-3-large",
    "credential_id": "<credential-uuid>",
    "label": "openai-large-v1",
    "dimensions": 3072,
    "distance_metric": "cosine"
  }'
```

### 3. Create an LLM profile

Offline draft mode:

```bash
curl -X POST http://localhost:8000/platform/profiles \
  -H "Content-Type: application/json" \
  -H "X-Namespace-ID: <namespace-uuid>" \
  -d '{
    "kind": "llm",
    "provider": "local_echo",
    "model": "echo-v1",
    "label": "local-echo"
  }'
```

OpenAI-backed answering:

```bash
curl -X POST http://localhost:8000/platform/profiles \
  -H "Content-Type: application/json" \
  -H "X-Namespace-ID: <namespace-uuid>" \
  -d '{
    "kind": "llm",
    "provider": "openai",
    "model": "gpt-4o",
    "credential_id": "<credential-uuid>",
    "label": "openai-gpt4o"
  }'
```

### 4. Create a vector collection bound to the embedding profile

```bash
curl -X POST http://localhost:8000/collections/ \
  -H "Content-Type: application/json" \
  -H "X-Namespace-ID: <namespace-uuid>" \
  -d '{
    "name": "bhagavad-gita",
    "strategy": "vector",
    "embedding_profile_id": "<embedding-profile-uuid>",
    "default_query_mode": "local"
  }'
```

### 5. Ingest text

Synchronous chunk ingest:

```bash
curl -X POST http://localhost:8000/collections/<collection-uuid>/ingest/chunk \
  -H "Content-Type: application/json" \
  -H "X-Namespace-ID: <namespace-uuid>" \
  -d '{
    "text": "Krishna teaches Arjuna about duty and devotion."
  }'
```

Async document ingest:

```bash
curl -X POST http://localhost:8000/collections/<collection-uuid>/ingest/doc \
  -H "Content-Type: application/json" \
  -H "X-Namespace-ID: <namespace-uuid>" \
  -d '{
    "text": "Long document text goes here"
  }'
```

### 6. Query the collection

With the collection default LLM path:

```bash
curl -X POST http://localhost:8000/collections/<collection-uuid>/query \
  -H "Content-Type: application/json" \
  -H "X-Namespace-ID: <namespace-uuid>" \
  -d '{
    "question": "What does Krishna teach Arjuna?"
  }'
```

With an explicit LLM profile:

```bash
curl -X POST http://localhost:8000/collections/<collection-uuid>/query \
  -H "Content-Type: application/json" \
  -H "X-Namespace-ID: <namespace-uuid>" \
  -d '{
    "question": "What does Krishna teach Arjuna?",
    "llm_profile_id": "<llm-profile-uuid>"
  }'
```

## Current Behavior

- `local_hash` embeddings are deterministic and offline-safe. They are useful for
  draft development and tests, not production-quality semantic retrieval.
- `local_echo` LLM mode returns the top retrieved chunk directly. It is useful
  for validating retrieval without external API calls.
- OpenAI-backed profiles become active when a profile uses `provider: "openai"`
  and its credential is present.
