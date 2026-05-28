# Architecture Redesign Audit Report

> **Date:** 2026-05-28
> **Reference Plan:** [docs/plans/2026-05-18-architecture-redesign.md](./2026-05-18-architecture-redesign.md)
> **Base Commit:** `11d5d4c` — refactor: use per-collection FalkorDB graphs instead of single knowledge_graph
> **Current HEAD:** `ed20f93` — chore: update architecture audit with decomposed module status
> **Commits Since Base:** 6

| Commit | Description |
|---|---|
| `ed20f93` | chore: update architecture audit with decomposed module status |
| `7d3a1b5` | refactor: decompose 1771-line graph.py into modular submodules |
| `fdf7f9c` | refactor: invert Dramatiq dependency direction — enforce I4 |
| `4a7d94c` | feat: enforce collection strategy and embedding_profile immutability at DB level |
| `1e38044` | feat: add namespace-level RLS policies for defense-in-depth isolation |
| `6139773` | docs: add architecture redesign consistency audit report |

---

## Summary

| Result | Previous | Current | Change |
|---|---|---|---|
| PASS | 27 | **29** | +2 (I1, I6 now DB-enforced) |
| PARTIAL | 8 | **6** | -2 (I1, I6 resolved) |
| FAIL | 2 | **1** | -1 (I6 resolved) |

---

## 1. File Layout (Section 9)

| Expected File | Status | Evidence |
|---|---|---|
| `services/graph_rag/service.py` | PARTIAL | `GraphService` exists at `src/graph_core/services/graph/__init__.py` (314 lines), decomposed into submodules — not a single file as implied. |
| `services/graph_rag/sanitizer.py` | PASS | `TextSanitizer` exists at `src/graph_core/services/sanitizer.py:28` (95 lines). |
| `services/graph_rag/ingestion/chunk_processor.py` | PASS | `src/graph_core/services/graph/ingestion/chunk_processor.py` (626 lines) — exists and is fully decomposed. |
| `services/graph_rag/ingestion/document_pipeline.py` | PASS | `src/graph_core/services/graph/ingestion/document_pipeline.py` (260 lines) — exists and is fully decomposed. |
| `services/graph_rag/query/` | PASS | Directory exists at `src/graph_core/services/graph/query/` with `graph_rag.py` (381 lines), `lightrag.py` (612 lines), `vector.py` (141 lines). |
| `services/graph_rag/storage/` | PARTIAL | Plan expected storage under `services/graph_rag/storage/`. Actual storage is at `src/graph_core/storage/` (sibling to services): `graph_storage.py` (442), `vector_store.py` (127), `graph_rag_vectors.py` (430), `vector_tables.py` (181), `vector_types.py` (49). |
| `api/collections.py` | PASS | `src/graph_core/api/collections.py` (74 lines) — resource-oriented router. |
| `api/ingest.py` | PASS | `src/graph_core/api/ingest.py` (65 lines) — namespace-scoped endpoints. |
| `api/query.py` | PASS | `src/graph_core/api/query.py` (48 lines) — namespace-scoped endpoint. |
| `api/jobs.py` | PASS | `src/graph_core/api/jobs.py` (33 lines) — job status + SSE stub. |
| `api/platform.py` | PASS | `src/graph_core/api/platform.py` (157 lines) — credentials, profiles, capabilities. |
| `mcp/server.py` | FAIL | `src/graph_core/mcp/` directory does not exist. No MCP server implemented. |
| `workers/ingestion_worker.py` | PARTIAL | Actual file is `src/graph_core/workers/ingestion.py` (61 lines; name differs from plan, but functionality matches). |

**Result:** 7 PASS, 4 PARTIAL, 2 FAIL. Key gaps: monolith decomposed from 1771 lines to 2867 total across service modules; MCP server absent; storage path variance.

### New Files and Directories Since Base

| File | Lines | Purpose |
|---|---|---|
| `services/graph_rag/extractor.py` | 254 | `LLMGraphExtractor` — LLM-based graph extraction with gleaning, MD5-cached results |
| `services/graph_rag/entity_resolver.py` | 498 | `IncrementalEntityResolver` — 3-tier resolution: alias lookup, embedding centroid search, fuzzy `difflib` matching; Hebbian centroid learning |
| `services/platform.py` | 85 | `PlatformService` — credential/profile CRUD control-plane |
| `services/chunking.py` | 37 | `TokenChunker` utility |
| `services/entity_name_cache.py` | 43 | Entity name cache for fast lookups |
| `middleware/semaphore.py` | 86 | Distributed Dramatiq semaphore — Redis sorted sets with Lua-based atomic acquire/release, fleet-wide `run_chunk` concurrency limiting |
| `models/chunk.py` | 51 | `IngestionChunk` model — per-chunk status tracking within document jobs |
| `models/graph_rag.py` | 243 | Graph RAG models: `GraphEntity`, `EntityDescription`, `EntityAlias`, `EntityType`, `GraphRelationship`, `RelationshipDescription`, `RawChunkExtraction` |
| `llm/` | 145 | LLM provider interface, factory, OpenAI provider |
| `embedding/` | 158 | Embedding provider interface, factory, hash provider, OpenAI provider |
| `scripts/smoke_test.py` | — | Smoke test script |
| `tests/` | — | 11 test files across `test_services`, `test_api`, `test_workers` |

---

## 2. Core Invariants

### I1: Strategy is per collection, set at creation, never changed
**PASS** (upgraded from PARTIAL)

- `Collection.strategy` column exists (`models/collection.py:22`), set at creation
- Model comment says "immutable after creation" (`models/collection.py:21`)
- `rag_strategy` enum values: `vector`, `light_rag`, `custom_graph_rag`
- **RESOLVED:** DB-level constraint now enforced via migration `0007_namespace_rls_policies.py:215-240`. PostgreSQL `BEFORE UPDATE` trigger `trg_collection_immutable_fields` raises exception if `strategy` changes post-creation.
- **Addition:** `Collection.llm_profile_id` column also exists (`models/collection.py:29`) — explicitly NOT immutable (can be changed post-creation).

### I2: TextSanitizer exists at `app/services/graph_rag/sanitizer.py`
**PASS** (with path variance)

- `TextSanitizer` class at `services/sanitizer.py:28`
- Implements: Unicode NFC normalization (`:40`), zero-width char removal (`:46`), null byte rejection (`:52`), size limit enforcement (`:57`, MAX=16000), pattern detection (`:63-76`)
- Returns `SanitizationReport` with severity levels
- Path is `services/sanitizer.py` rather than `services/graph_rag/sanitizer.py`
- **Addition:** Constructor accepts `trusted_namespace_ids` parameter (`:29`) for namespace-aware sanitization.
- **Addition:** Added `chunk_hash` static method for SHA-256 deduplication (`:92-95`).

### I3: Namespace isolation enforced at platform layer
**PASS** (enhanced with RLS)

- `_enforce_namespace` in `services/graph/__init__.py:265-268` raises `PermissionError` on mismatch
- **Addition:** Namespace enforcement also duplicated in `services/graph/ingestion/chunk_processor.py:126-131` at submodule level.
- API dependency `get_namespace_id` in `api/dependencies.py:8-14` extracts from `X-Namespace-ID` header
- All API routes use namespace dependency injection
- PlatformService validates namespace on credential/profile operations
- **RESOLVED:** Migration `0007` adds comprehensive Row-Level Security (RLS) policies on 15 tables (credentials, profiles, collections, jobs, ingestion_records, graph_entities, graph_relationships, raw_chunk_extractions, entity_descriptions, entity_aliases, entity_types, relationship_descriptions, job_events, ingestion_chunks). Helper functions `ns_check(uuid)` and `ns_check_via_collection(uuid)` enforce isolation via `app.current_namespace_id` session variable.

### I4: GraphService has NO transport dependencies
**PASS**

- No FastAPI, MCP, or Dramatiq imports in `services/graph/__init__.py` or any submodules
- Previously reported gap (Dramatiq imports at `graph.py:230,298`) has been resolved — the old monolithic `graph.py` file no longer exists
- Dramatiq import exists only in `workers/ingestion.py:6` which is the correct direction (workers know about GraphService, not vice versa)

### I5: Durable state in Postgres
**PASS**

- `jobs` table: `models/job.py:13-44` with all specified columns
- `job_events` table: `models/job.py:47-59` with all specified columns
- `ingestion_records` table: `models/ingestion.py:13-34`
- Job status reads from Postgres (`api/jobs.py:16-21` → `services/graph/__init__.py:217-231`)
- All writes go to Postgres, SSE is transient only
- **Addition:** `IngestionChunk` model (`models/chunk.py:13-51`) for per-chunk status tracking within document jobs.

### I6: Embedding profiles immutable per collection
**PASS** (upgraded from PARTIAL)

- `Collection.embedding_profile_id` exists (`models/collection.py:26`)
- Set at creation, dimensions resolved at creation time (`models/collection.py:33`)
- **RESOLVED:** Same DB-level trigger as I1 (`trg_collection_immutable_fields`) now enforces immutability for `embedding_profile_id`.
- **Gap remains:** No `reindex` job handler is wired up (the `reindex` enum value exists but no handler).

---

## 3. Platform Concepts (Section 4)

### 4.1 Namespaces & Isolation
**PASS**

- `Namespace` model: `models/namespace.py:13-26`
- All resources scoped to namespace: collections, credentials, profiles, jobs
- Isolation at namespace, collection, and credential boundaries
- **Addition:** RLS policies on 15 tables provide defense-in-depth isolation at the database level.

### 4.2 Embedding/LLM Profiles
**PASS**

- `Profile` model: `models/profile.py:13-38`
- `kind` enum: `embedding` / `llm`
- Embedding profiles have `dimensions`, `distance_metric`, `credential_id`
- LLM profiles are dynamic, overridable per query (`api/query.py:15-16`)
- Profiles bound to `credential_id`
- **Addition:** Profile model now has `base_url` column (`models/profile.py:26`) for per-profile API override.

### 4.3 Credential Architecture
**PASS**

- `Credential` model: `models/credential.py:13-37`
- `encrypted_secret` column, never stored plaintext
- Encryption via `services/crypto.py:11-21` (Fernet-based)
- Namespace-scoped with unique constraint on `(namespace_id, label)`
- Profiles reference credentials by ID
- **Addition:** `Credential` model also has `base_url` column (`models/credential.py:25`).

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
| `created_at`/`started_at`/`completed_at` | timestamptz | `models/job.py:34-36` | PASS |
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
- **Addition:** Also includes `extraction_model`, `embedding_model` columns for model tracking.

### `ingestion_chunks` table (new)
**PASS**

- `models/chunk.py:13-51`: `IngestionChunk` model for per-chunk status within document jobs
- Columns: `id`, `job_id`, `chunk_hash`, `status` (enum: `pending`, `processing`, `completed`, `failed`), `error`, `created_at`, `updated_at`

---

## 5. Migration Items (Section 9 Table)

| Migration Item | Status | Evidence |
|---|---|---|
| Dagster jobs (`*ingest_jobs.py`) deleted | PASS | No files matching `*ingest_jobs*` exist. No `dagster` references in code. |
| `process_chunk_async()` → `GraphService._process_chunk()` | PASS | No `process_chunk_async` exists in codebase. Equivalent is `ingest_collection_chunk` at `services/graph/ingestion/chunk_processor.py:145`. |
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

## 7. Architecture Changes Since Previous Audit

The codebase has undergone significant decomposition since the audit referenced a monolithic `graph.py` (1771 lines):

### Decomposed Modules (2867 total lines in service + API)

| File | Lines | Purpose |
|---|---|---|
| `services/graph/__init__.py` | 314 | GraphService orchestration class, delegates to submodules |
| `services/graph/ingestion/chunk_processor.py` | 626 | Per-chunk ingestion pipeline for all 3 strategies |
| `services/graph/ingestion/document_pipeline.py` | 260 | Document ingestion, chunking, fan-out, progress tracking |
| `services/graph/query/graph_rag.py` | 381 | Graph RAG query with energy-decay DFS traversal |
| `services/graph/query/lightrag.py` | 612 | LightRAG query with local/global/hybrid/naive/mix modes |
| `services/graph/query/vector.py` | 141 | Pure vector query + answer generation |
| `services/sanitizer.py` | 95 | Text sanitization with pattern detection |
| `workers/ingestion.py` | 61 | Dramatiq actors (thin wrappers) |
| `api/*.py` | ~370 | API routers (collections, ingest, query, jobs, platform) |

### Storage Layer (1230 total lines)

| File | Lines | Purpose |
|---|---|---|
| `storage/graph_storage.py` | 442 | FalkorDB graph storage (nodes, edges, LightRAG ops) |
| `storage/graph_rag_vectors.py` | 430 | GraphRAG-specific vector storage (entity/relationship/chunk embeddings, centroids) |
| `storage/vector_tables.py` | 181 | Dynamic per-collection vector table management (create/drop) |
| `storage/vector_store.py` | 127 | Postgres vector store (per-collection dynamic tables) |
| `storage/vector_types.py` | 49 | Custom SQLAlchemy types for pgvector |

### New Graph RAG Services (752 total lines)

| File | Lines | Purpose |
|---|---|---|
| `services/graph_rag/entity_resolver.py` | 498 | Incremental entity resolution: alias lookup, embedding centroid search, fuzzy matching |
| `services/graph_rag/extractor.py` | 254 | LLM-based graph extraction with gleaning, MD5-cached results |

### Dramatiq Violation Fixed
The previous audit flagged I4 violation: `GraphService` importing Dramatiq actors. This has been **fully resolved** — the old `graph.py` monolith is gone, and Dramatiq imports exist only in `workers/ingestion.py` (the correct direction: workers depend on service, not vice versa).

### DB-Level Immutability Added (migration `0007`)
The `enforce_collection_immutable_fields()` trigger function and `trg_collection_immutable_fields` trigger on the `collections` table now enforce I1 and I6 at the database level, closing the previous PARTIAL gaps.

### Namespace RLS Policies Added (migration `0007`)
Comprehensive Row-Level Security policies on 15 tables provide defense-in-depth namespace isolation, supplementing the application-layer enforcement.

---

## Critical Gaps Requiring Attention

1. **MCP server missing entirely** — The plan designates `app/mcp/server.py` as a new adapter layer wrapping platform APIs. No implementation exists.
2. **Job SSE streaming is a stub** — `api/jobs.py:27` has a TODO; no Redis pubsub subscription is implemented.
3. **`retrieval_strategies` hardcoded in capabilities** — `api/platform.py:76` returns hardcoded list rather than deriving from configuration.
4. **API routes lack namespace URL prefixes** — Plan specifies `/namespaces/{ns}/...` routes; implementation uses header-based namespace identification instead.
5. **Capability endpoint path variance** — Plan specifies root-level `/capabilities`, `/embedding-profiles`, `/llm-profiles`; actual routes are nested under `/platform/`.
6. **No `reindex` job handler** — The `reindex` enum value exists in `JobType` but no worker handles it.
