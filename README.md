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
- **App** (FastAPI, port 8000)
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
- **Job** — durable async unit of work, tracked in Postgres
- **Profile** — reusable configuration for embeddings or LLMs
- **Credential** — encrypted secret reference, bound to namespace

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
