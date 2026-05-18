# Architecture Redesign: Platform-First Graph Infrastructure

> **Status:** Design — approved, awaiting implementation plan
>
> **Replaces:** Dagster-based ingestion, app-centric backend, env-var-only strategy selection, no ingestion ledger

---

## 1. Vision: From Application Backend to Reusable Platform

The system is no longer being treated as:
```
"Scripture Assistant backend with GraphRAG features"
```

It is evolving toward:
```
Reusable AI-native knowledge infrastructure platform
```

Scripture Assistant becomes **one consumer** of the platform, not the owner of the graph runtime. This shifts architectural priorities:
- Platform boundaries matter more than framework choices
- Transport independence matters more than application convenience
- Reusability and namespace isolation matter more than app-local coupling
- Infrastructure abstractions (profiles, capabilities, jobs) matter more than product semantics

---

## 2. Core Invariants (Non-Negotiable)

**I1 — Strategy is per collection, set at creation, never changed.**
A collection binds to a strategy (`vector` | `custom_graph_rag`) and embedding profile at creation. Mixed-strategy collections are forbidden.

**I2 — User-supplied text is sanitized before LLM processing.**
Inbound text passes through `TextSanitizer` (Unicode normalization, size limits, encoding validation). Prompt injection defense relies primarily on structured outputs/function-calling, with sanitization as a defense-in-depth layer.

**I3 — Namespace isolation is enforced at the platform layer.**
Collections belong to namespaces. The platform enforces isolation at the namespace, collection, and credential boundaries. Application-layer user/org semantics are external.

**I4 — The service layer has no transport dependencies.**
Internal orchestration (`GraphService`) imports nothing from FastAPI, MCP, Dramatiq, or Dagster. It is a pure Python class callable from any context. The public API is resource-oriented, not a direct export of `GraphService`.

**I5 — Durable state lives in Postgres; events are transient.**
Job lifecycle, progress, and results are stored in Postgres tables (`jobs`, `job_events`, `ingestion_records`). SSE/Redis pubsub is only for real-time UX — never the source of truth.

**I6 — Embedding profiles are infrastructure, not runtime preferences.**
Embedding configuration determines vector dimensions, semantic space, and index compatibility. It is immutable for a collection. Changing embeddings requires a reindex operation.

---

## 3. Layered Architecture

```
┌──────────────────────────────────────────────────────────┐
│                  Application Layer                        │
│  Scripture Assistant  ·  MCP Clients  ·  Agents  ·  CLI  │
│  (Owns: users, RBAC, billing, UI, business workflows)    │
└────────────────────────┬─────────────────────────────────┘
                         │
┌────────────────────────▼─────────────────────────────────┐
│                   Transport Layer                         │
│  REST (HTTP/JSON/SSE)  ·  MCP Adapter  ·  SDK Wrappers   │
└────────────────────────┬─────────────────────────────────┘
                         │
┌────────────────────────▼─────────────────────────────────┐
│               Platform API / Control Plane                │
│  Collections  ·  Jobs  ·  Profiles  ·  Capabilities      │
│  Credentials  ·  Namespace Auth  ·  Resource Endpoints   │
└────────────────────────┬─────────────────────────────────┘
                         │
┌────────────────────────▼─────────────────────────────────┐
│                   Execution Layer                         │
│  Dramatiq Workers (thin wrappers)                         │
│  Postgres Durable State  ·  Redis Transient Events        │
└────────────────────────┬─────────────────────────────────┘
                         │
┌────────────────────────▼─────────────────────────────────┐
│                  Graph Runtime                            │
│  GraphService (internal) · Ingestion · Query · Extraction │
│  Embedding Execution · Retrieval · Graph Traversal        │
└────────────────────────┬─────────────────────────────────┘
                         │
┌────────────────────────▼─────────────────────────────────┐
│                   Storage Layer                           │
│  PostgreSQL · Vector DB · Graph DB · Redis/Cache          │
└──────────────────────────────────────────────────────────┘
```

---

## 4. Platform Concepts

### 4.1 Namespaces & Isolation
The platform does not understand "users", "teams", or "orgs". It understands **namespaces**.

```json
{
  "collection_id": "col_abc123",
  "namespace": "scripture-assistant-prod"
}
```

Authorization is namespace-bound. Applications resolve users to namespaces and pass the namespace identifier. Cross-namespace access is forbidden unless explicitly proxied by the application layer.

### 4.2 Embedding & LLM Profiles
Embeddings are stable, collection-bound infrastructure. LLMs are dynamic, query-time executables.

**Embedding Profile** (immutable per collection):
```json
{
  "profile_id": "openai-large-v1",
  "provider": "openai",
  "model": "text-embedding-3-large",
  "dimensions": 3072,
  "distance_metric": "cosine",
  "credential_id": "cred_abc123"
}
```

**LLM Profile** (dynamic, overridable per query/job):
```json
{
  "profile_id": "qwen3-70b-prod",
  "provider": "openai_compat",
  "model": "qwen3-70b",
  "credential_id": "cred_xyz789"
}
```

### 4.3 Credential Architecture
Clients never pass raw secrets on every request. Credentials are registered once, encrypted, and referenced by ID.

```http
POST /credentials
{ "provider": "openai", "secret": "sk-...", "label": "prod-key" }
→ { "credential_id": "cred_abc123" }
```

Profiles bind to `credential_id`. The runtime resolves secrets internally. Credentials are namespace-scoped.

### 4.4 Capability Discovery
The platform exposes its capabilities for machine-readable negotiation:

```http
GET /capabilities
→ {
    "embedding_profiles": [...],
    "llm_profiles": [...],
    "retrieval_strategies": ["vector", "custom_graph_rag"],
    "max_chunk_size": 8000
  }
```

Enables dynamic clients, MCP tool generation, and SDK auto-configuration.

---

## 5. Durable Job State

### Postgres = Source of Truth

Two new tables provide durable, queryable, recoverable job state:

#### `jobs` table
| Column | Type | Purpose |
|---|---|---|
| `id` | UUID | Primary key, returned to caller immediately |
| `type` | enum | `ingest_chunk`, `ingest_document`, `delete_collection`, `reindex` |
| `status` | enum | `pending`, `running`, `completed`, `failed`, `cancelled` |
| `created_at` / `started_at` / `completed_at` | timestamptz | Lifecycle timestamps |
| `error` | Text | Error message if failed |
| `progress_percent` | Integer | 0-100 progress indicator |
| `collection_id` / `namespace_id` | UUID FK | Scoping |

This is what `GET /jobs/{id}` reads from.

#### `job_events` table
| Column | Type | Purpose |
|---|---|---|
| `id` | UUID | Primary key |
| `job_id` | UUID FK | Parent job |
| `timestamp` | timestamptz | Event time |
| `event_type` | str | `chunk_started`, `llm_called`, `error`, etc. |
| `payload` | JSONB | Event-specific data |

Provides debugging, observability, and audit trail. TTL cleanup policy (e.g., vacuum >30 days) prevents unbounded growth.

---

## 6. Execution Runtime

### Dramatiq Workers (Background Execution)
Replaces Dagster. Chosen because it's already in use for MinIO operations, Redis-backed, and ergonomically simple for medium-scale async pipelines.

**Workers are thin — they never contain business logic:**

```python
@dramatiq.actor
async def run_ingestion(job_id: str):
    """Thin wrapper — all logic lives in GraphService."""
    await graph_service.ingest_document_pipeline(UUID(job_id))
```

The worker pulls `job_id`, calls the internal service, writes progress to Postgres, and emits transient SSE events via Redis pubsub. The runtime is an implementation detail — domain logic lives in `GraphService`.

### Transient Event Streaming
SSE/WebSocket endpoints (`GET /jobs/{id}/stream`) subscribe to Redis channels for live progress. If a client disconnects, state persists in Postgres and jobs continue. Events are UX only, never the source of truth.

---

## 7. Public API Design

Resource-oriented, cloud-style control plane. `GraphService` remains internal.

```
POST /namespaces/{ns}/collections          → Create collection (binds strategy + embedding profile)
POST /namespaces/{ns}/collections/{id}/ingest/chunk  → Sync chunk (small text)
POST /namespaces/{ns}/collections/{id}/ingest/doc    → Async doc (returns job_id)
POST /namespaces/{ns}/collections/{id}/query         → Query (binds LLM profile)

GET  /jobs/{job_id}                         → Durable status (Postgres)
GET  /jobs/{job_id}/stream                  → Transient progress (SSE)

POST /credentials                           → Register encrypted credential
GET  /capabilities                          → Platform capability discovery
GET  /embedding-profiles                    → List available embedding configs
GET  /llm-profiles                          → List available LLM configs
```

**MCP Integration:** MCP is an adapter layer, not the core protocol. MCP tools wrap the resource-oriented endpoints. Tool descriptions drive LLM client behavior.

---

## 8. Security & Isolation

### Namespace Boundaries
Infrastructure isolation enforced at:
- Namespace boundary
- Collection boundary
- Credential boundary
- Profile resolution boundary

Application-layer RBAC maps users to namespaces. The platform trusts the namespace identifier.

### Prompt Injection Defense
Defense-in-depth approach:
1. **Structured outputs** (function calling / JSON mode) — primary defense
2. **TextSanitizer** — Unicode normalization, size limits, encoding validation, collection-aware pattern detection
3. **Provenance logging** — `sanitization_flags` written to ingestion ledger for traceability
4. **Domain awareness** — Indic scripture naturally contains imperative phrases; aggressive regex stripping is avoided in favor of structured parsing

---

## 9. Migration Plan

| Current component | Fate |
|---|---|
| Dagster jobs (`*ingest_jobs.py`) | Deleted. Logic moves to `GraphService` + Dramatiq worker. |
| `process_chunk_async()` | Becomes `GraphService._process_chunk()` private method. |
| `GraphRAGIngestor` | Deleted. Incremental logic absorbed into `GraphService`. |
| `app/api/ingest.py`, `common_utils.py` | Replaced with namespace-scoped, resource-oriented endpoints. |
| `app/core/config.py` | `RAG_STRATEGY` becomes default only. Strategy/profiles move to collection records. |
| Query dispatcher/plugins | Unchanged internally. Router reads `collection.strategy` instead of global env var. |
| Storage backends | Unchanged. |
| Dramatiq worker service | Extended to handle ingestion jobs. |
| Dagster services (4 containers) | Removed from `docker-compose.yml`. |
| `TextSanitizer` | New. `app/services/graph_rag/sanitizer.py`. |
| `jobs`, `job_events`, `IngestionRecord` | New. Alembic migration. |
| `Collection.strategy`, `Collection.embedding_profile_id` | New columns. Alembic migration. |
| Credential store | New. Encrypted Postgres table. |
| MCP server | New. `app/mcp/server.py` (adapter layer). |

### File Layout (New)
```
app/
  services/
    graph_rag/
      service.py           # GraphService — internal orchestration
      sanitizer.py         # TextSanitizer
      ingestion/
        chunk_processor.py # _process_chunk() logic
        document_pipeline.py # ingest_document_pipeline()
      query/               # existing query plugins
      storage/             # existing storage backends
  api/
    collections.py         # collection CRUD
    ingest.py              # ingest endpoints → enqueue or sync call
    query.py               # query endpoint
    jobs.py                # job status + SSE streaming
    platform.py            # credentials, profiles, capabilities
  mcp/
    server.py              # MCP adapter wrapping platform APIs
  workers/
    ingestion_worker.py    # Dramatiq actors — thin wrappers
```

---

## 10. Out of Scope

- Application-layer auth (users, tokens, scopes, billing) — owned by consuming apps
- LightRAG strategy — can be ported to `GraphService` in a follow-up
- Sleep cycle / meta-entity consolidation — unchanged, scheduled task against storage
- Community detection / GraphRAG global search — future phase
- Polyglot rewrite (Go/Rust) — architecture supports it, Python remains practical for now
- External secret managers (Vault/AWS SM) — encrypted Postgres initially, pluggable later
