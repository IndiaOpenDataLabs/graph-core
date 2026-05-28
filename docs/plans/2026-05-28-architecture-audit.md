# Architecture Redesign Audit Report

> **Date:** 2026-05-28
> **Reference Plan:** [docs/plans/2026-05-18-architecture-redesign.md](./2026-05-18-architecture-redesign.md)
> **Base Commit:** `11d5d4c` — refactor: use per-collection FalkorDB graphs instead of single knowledge_graph

---

## Summary

| Result | Count |
|---|---|
| PASS | 23 |
| PARTIAL | 17 |
| FAIL | 7 |

---

## 1. File Layout (Section 9)

| Expected File | Status | Evidence |
|---|---|---|
| `services/graph_rag/service.py` | PARTIAL | `GraphService` exists at `src/graph_core/services/graph.py:76` (1771 lines), not under `services/graph_rag/` |
| `services/graph_rag/sanitizer.py` | PARTIAL | `TextSanitizer` exists at `src/graph_core/services/sanitizer.py:28`, not under `services/graph_rag/` |
| `services/graph_rag/ingestion/chunk_processor.py` | FAIL | Does not exist. Chunk processing logic is inline in `services/graph.py:182-207` (`_ingest_collection_chunk`) |
| `services/graph_rag/ingestion/document_pipeline.py` | FAIL | Does not exist. Document pipeline is inline in `services/graph.py:236-276` (`ingest_document_pipeline`) |
| `services/graph_rag/query/` | FAIL | Directory does not exist. All query logic (`_query_graph_rag`, `_query_vector`, `_query_lightrag`) is inline in `services/graph.py:593-1320` |
| `services/graph_rag/storage/` | FAIL | Directory does not exist. Storage backends live at `src/graph_core/storage/` (sibling to services) |
| `api/collections.py` | PASS | `src/graph_core/api/collections.py` — resource-oriented router |
| `api/ingest.py` | PASS | `src/graph_core/api/ingest.py` — namespace-scoped endpoints |
| `api/query.py` | PASS | `src/graph_core/api/query.py` — namespace-scoped endpoint |
| `api/jobs.py` | PASS | `src/graph_core/api/jobs.py` — job status + stream |
| `api/platform.py` | PASS | `src/graph_core/api/platform.py` — credentials, profiles, capabilities |
| `mcp/server.py` | FAIL | `src/graph_core/mcp/` directory does not exist. No MCP server implemented. |
| `workers/ingestion_worker.py` | PARTIAL | Actual file is `src/graph_core/workers/ingestion.py` (name differs from plan) |

**Result:** 4 PASS, 3 PARTIAL, 6 FAIL. Key gaps: monolithic `graph.py` (1771 lines) instead of decomposed submodules; MCP server absent.

---

## 2. Core Invariants

### I1: Strategy is per collection, set at creation, never changed
**PARTIAL**

- `Collection.strategy` column exists (`models/collection.py:22`), set at creation
- Model comment says "immutable after creation" (`models/collection.py:21`)
- `rag_strategy` enum values: `vector`, `light_rag`, `custom_graph_rag`
- **Gap:** No DB-level constraint (trigger/check) preventing post-creation updates. Immutability is enforced only by application logic, not at the schema level.

### I2: TextSanitizer exists at `app/services/graph_rag/sanitizer.py`
**PASS** (with path variance)

- `TextSanitizer` class at `services/sanitizer.py:28`
- Implements: Unicode NFC normalization (`:40`), zero-width char removal (`:46`), null byte rejection (`:52`), size limit enforcement (`:57`, MAX=16000), pattern detection (`:63-76`)
- Returns `SanitizationReport` with severity levels
- `sanitization_flags` written to ingestion ledger (`graph.py:1671-1673`)
- Path is `services/sanitizer.py` rather than `services/graph_rag/sanitizer.py`

### I3: Namespace isolation enforced at platform layer
**PASS**

- `_enforce_namespace` in `services/graph.py:1445-1449` raises `PermissionError` on mismatch
- API dependency `get_namespace_id` in `api/dependencies.py:8-14` extracts from `X-Namespace-ID` header
- All API routes use namespace dependency injection
- PlatformService validates namespace on credential/profile operations (`services/platform.py:55`)

### I4: GraphService has NO transport dependencies
**PARTIAL**

- No FastAPI, MCP, or Dagster imports in `services/graph.py`
- **Gap:** `services/graph.py:230` contains `from graph_core.workers.ingestion import run_ingestion` (Dramatiq import). Similarly `:298` imports `run_chunk`. The service layer directly imports Dramatiq actors. The plan states: "imports nothing from FastAPI, MCP, Dramatiq, or Dagster."

### I5: Durable state in Postgres
**PASS**

- `jobs` table: `models/job.py:13-44` with all specified columns
- `job_events` table: `models/job.py:47-59` with all specified columns
- `ingestion_records` table: `models/ingestion.py:13-34`
- Job status reads from Postgres (`api/jobs.py:16-21` → `graph.py:1397-1411`)
- All writes go to Postgres, SSE is transient only

### I6: Embedding profiles immutable per collection
**PARTIAL**

- `Collection.embedding_profile_id` exists (`models/collection.py:26`)
- Set at creation, dimensions resolved at creation time (`models/collection.py:33`)
- **Gap:** Same as I1 — no DB-level constraint preventing post-creation updates. No reindex job type is wired up (the `reindex` enum value exists but no handler).

---

## 3. Platform Concepts (Section 4)

### 4.1 Namespaces & Isolation
**PASS**

- `Namespace` model: `models/namespace.py:13-26`
- All resources scoped to namespace: collections, credentials, profiles, jobs
- Isolation at namespace, collection, and credential boundaries

### 4.2 Embedding/LLM Profiles
**PASS**

- `Profile` model: `models/profile.py:13-38`
- `kind` enum: `embedding` / `llm`
- Embedding profiles have `dimensions`, `distance_metric`, `credential_id`
- LLM profiles are dynamic, overridable per query (`api/query.py:15-16`)
- Profiles bound to `credential_id`

### 4.3 Credential Architecture
**PASS**

- `Credential` model: `models/credential.py:13-37`
- `encrypted_secret` column, never stored plaintext
- Encryption via `services/crypto.py:11-21` (Fernet-based)
- Namespace-scoped with unique constraint on `(namespace_id, label)`
- Profiles reference credentials by ID

### 4.4 Capability Discovery
**PARTIAL**

- Endpoint at `GET /platform/capabilities` (`api/platform.py:54-78`)
- Returns `embedding_profiles`, `llm_profiles`, `retrieval_strategies`, `max_chunk_size`
- **Gap 1:** Plan specifies `GET /capabilities` at root level. Actual is `GET /platform/capabilities`.
- **Gap 2:** `retrieval_strategies` is hardcoded as `["vector", "custom_graph_rag", "light_rag"]` (`api/platform.py:76`) rather than dynamically derived from configuration.
- **Gap 3:** The plan lists `GET /embedding-profiles` and `GET /llm-profiles` as separate root endpoints. Actual: `GET /platform/embedding-profiles` and `GET /platform/llm-profiles`.

---

## 4. Durable Job State (Section 5)

### `jobs` table
**PASS** (with extras)

| Column | Spec | Actual | Status |
|---|---|---|---|
| `id` (UUID PK) | Required | `models/job.py:16` | PASS |
| `type` (enum) | `ingest_chunk`, `ingest_document`, `delete_collection`, `reindex` | Matches (`models/job.py:20-22`) | PASS |
| `status` (enum) | `pending`, `running`, `completed`, `failed`, `cancelled` | Matches exactly (`models/job.py:23-26`) | PASS |
| `created_at`/`started_at`/`completed_at` | timestamptz | `models/job.py:35-37` | PASS |
| `error` (Text) | Required | `models/job.py:29` | PASS |
| `progress_percent` (Integer) | 0-100 | `models/job.py:28` | PASS |
| `collection_id`/`namespace_id` | UUID FK | `models/job.py:17-18` | PASS |
| — | — | Extras: `payload` (JSON), `chunks_total`, `chunks_completed` | — |

### `job_events` table
**PASS**

| Column | Spec | Actual | Status |
|---|---|---|---|
| `id` (UUID PK) | Required | `models/job.py:50` | PASS |
| `job_id` (UUID FK) | Required | `models/job.py:51` | PASS |
| `timestamp` (timestamptz) | Required | `models/job.py:52` | PASS |
| `event_type` (str) | Required | `models/job.py:53` | PASS |
| `payload` (JSONB) | Required | `models/job.py:54` (JSON) | PASS |

### `ingestion_records` table
**PASS**

- `models/ingestion.py:13-34`: All fields present, including `sanitization_flags` (JSON), `entity_count`, `relationship_count`, `strategy`, `source_document_id`

---

## 5. Migration Items (Section 9 Table)

| Migration Item | Status | Evidence |
|---|---|---|
| Dagster jobs (`*ingest_jobs.py`) deleted | PASS | No files matching `*ingest_jobs*` exist. No `dagster` references in code. |
| `process_chunk_async()` → `GraphService._process_chunk()` | PASS | No `process_chunk_async` exists in codebase. Equivalent is `_ingest_collection_chunk` at `graph.py:182`. |
| `GraphRAGIngestor` deleted | PASS | No references found anywhere in codebase. |
| `app/api/ingest.py`, `common_utils.py` replaced | PASS | No `common_utils.py` exists. `api/ingest.py` has been rewritten with namespace-scoped, resource-oriented endpoints. |
| `RAG_STRATEGY` env var → default only | PASS | No `RAG_STRATEGY` in `config.py`. `config.py` has `default_embedding_*` and `default_llm_*` for defaults. Strategy is per-collection. |
| `TextSanitizer` new | PASS | `services/sanitizer.py:28` |
| `jobs`, `job_events`, `IngestionRecord` new | PASS | All in `models/job.py` and `models/ingestion.py`, with Alembic migration `0001` |
| `Collection.strategy`, `Collection.embedding_profile_id` new columns | PASS | `models/collection.py:22,26`, migration `0001_initial_platform_schema.py:180,188` |
| Credential store new | PASS | `models/credential.py`, `services/crypto.py`, `api/platform.py:81-99` |
| MCP server new | FAIL | No `mcp/` directory exists. No MCP server implementation. |
| Dramatiq worker extended | PASS | `workers/ingestion.py:13-43` — two thin Dramatiq actors (`run_ingestion`, `run_chunk`) that delegate to GraphService |

---

## 6. Public API Design (Section 7)

| Endpoint | Spec Path | Actual Path | Status |
|---|---|---|---|
| Create collection | `POST /namespaces/{ns}/collections` | `POST /collections/` | PARTIAL — no `/namespaces/{ns}/` prefix; namespace from header |
| Ingest chunk | `POST /namespaces/{ns}/collections/{id}/ingest/chunk` | `POST /collections/{collection_id}/ingest/chunk` | PARTIAL — no namespace prefix |
| Ingest doc | `POST /namespaces/{ns}/collections/{id}/ingest/doc` | `POST /collections/{collection_id}/ingest/doc` | PARTIAL — no namespace prefix |
| Query | `POST /namespaces/{ns}/collections/{id}/query` | `POST /collections/{collection_id}/query` | PARTIAL — no namespace prefix |
| Job status | `GET /jobs/{job_id}` | `GET /jobs/{job_id}` | PASS |
| Job stream | `GET /jobs/{job_id}/stream` | `GET /jobs/{job_id}/stream` | PARTIAL — endpoint exists but is a stub (`api/jobs.py:24-33`, TODO comment, no actual Redis pubsub) |
| Credentials | `POST /credentials` | `POST /platform/credentials` | PARTIAL — nested under `/platform/` |
| Capabilities | `GET /capabilities` | `GET /platform/capabilities` | PARTIAL — nested under `/platform/` |
| Embedding profiles | `GET /embedding-profiles` | `GET /platform/embedding-profiles` | PARTIAL — nested under `/platform/` |
| LLM profiles | `GET /llm-profiles` | `GET /platform/llm-profiles` | PARTIAL — nested under `/platform/` |

**Note:** The implementation uses header-based namespace identification (`X-Namespace-ID`) rather than URL-path namespace scoping. This is a valid architectural choice but differs from the plan's explicit URL pattern.

---

## Critical Gaps Requiring Attention

1. **MCP server missing entirely** — The plan designates `app/mcp/server.py` as a new adapter layer wrapping platform APIs. No implementation exists.
2. **`graph.py` is a 1771-line monolith** — The plan calls for decomposition into `ingestion/chunk_processor.py`, `ingestion/document_pipeline.py`, and `query/`. All logic is currently in one file.
3. **GraphService imports Dramatiq actors** — Violates I4 (transport independence). The Dramatiq import at `graph.py:230,298` should be inverted (workers should know about GraphService, not vice versa).
4. **Job SSE streaming is a stub** — `api/jobs.py:27` has a TODO; no Redis pubsub subscription is implemented.
5. **No DB-level immutability constraints** — I1 and I6 rely on application-layer enforcement only.
6. **API routes lack namespace URL prefixes** — Plan specifies `/namespaces/{ns}/...` routes; implementation uses header-based namespace identification instead.
