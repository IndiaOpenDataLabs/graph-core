# Qualified Statement Ingestion Remediation Plan

> **Status:** Proposed
>
> **Date:** 2026-07-26
>
> **Scope:** `custom_graph_rag` extraction, canonicalization, persistence,
> vector indexing, graph projection, reasoning, and incremental provenance
>
> **Supersedes:** The ingestion/materialization details of
> `2026-07-12-incremental-compiled-graph-reasoning.md`; its bounded-complexity,
> immutable-extraction, versioning, and auditable-reasoning goals remain valid.

## 1. Problem Statement

The current graph ingestion path materializes one extracted relationship as
several independently authored representations:

```text
raw extracted relationship
  -> GraphRelationship
  -> PROPOSITION GraphEntity
  -> SUBJECT and OBJECT edges
  -> condition/exception/scope GraphEntities and edges
  -> GraphSemanticFrame plus arguments
  -> FalkorDB nodes and edges
  -> vector rows
  -> incremental mappings and dependencies
```

This has produced correctness, identity, retrieval, and maintainability defects:

- ingestion writes `PROPOSITION` and `SUPPORTS`, while query paths still read
  `ASSERTION` and `HAS_ASSERTION`;
- source hierarchy and evidence objects are forced through the canonical entity
  abstraction;
- evidence chunks and hierarchy nodes receive entity embeddings and centroids
  despite already having chunk/source representations;
- statement identity depends on extraction list order;
- a SHA-1 display fingerprint duplicates information already present in the
  assertion name and ID;
- concept type is encoded in both `canonical_name` and `primary_type`;
- the same subject-predicate-object statement is stored as a direct edge,
  proposition node, structural edges, and semantic-frame arguments;
- plain-text qualifiers are promoted into apparently executable reasoning nodes;
- several provenance tables are written but have no production read/retraction
  path;
- `_ingest_graph_chunk` owns extraction, resolution, SQL persistence, vector
  indexing, graph projection, reasoning compilation, and publication.

The remediation must establish one authoritative statement model, migrate reads
and writes without losing existing collections, and then delete redundant
representations.

## 2. Safety Rules

This work must not be implemented as one rewrite.

1. Add characterization tests before changing persistence.
2. Fix ingestion/query vocabulary compatibility before schema redesign.
3. Preserve raw extraction records; historical content must not be sent to an
   LLM again.
4. Introduce the new statement model with dual-read/dual-write controls.
5. Backfill deterministically from raw extraction and existing mappings.
6. Compare old and new retrieval results before changing the default read path.
7. Remove legacy writes only after repository-wide reader searches and production
   metrics show no use.
8. Every cutover must have a feature flag and rollback path.
9. Foreground ingestion must remain bounded by the new chunk plus fixed-size
   indexed lookups.

## 3. Target Domain Model

### 3.1 Immutable source and extraction layer

```text
SourceDocument
SourceSegment
RawExtraction
ExtractedStatement
EndpointMention
```

The extractor emits statements. An endpoint mention is nested statement data,
not a canonical entity:

```text
ExtractedStatement
  local_key
  subject: EndpointMention(name, type_hint, description)
  predicate
  object: EndpointMention(name, type_hint, description)
  description
  keywords
  weight/confidence
  polarity
  modality
  conditions
  exceptions
  scopes
  predicate_properties
```

`ExtractionResult.entities` will be removed. During compatibility it may remain
as a computed property derived from statement endpoints.

### 3.2 Canonical semantic layer

```text
CanonicalConcept
CanonicalPredicate
QualifiedStatement
StatementEvidence
```

`QualifiedStatement` is the authoritative semantic fact:

```text
id
collection_id
subject_concept_id
predicate_id
object_concept_id
description
polarity
modality
conditions_json
exceptions_json
scopes_json
keywords
confidence
content_hash
```

`StatementEvidence` associates a statement with one or more source segments and
stores occurrence-specific extraction metadata. Identical statements may merge
canonically without losing independent source occurrences.

Canonical concept fields are separate:

```text
canonical_name = "Alice"
concept_type = "PERSON"
object_kind = "CONCEPT"
```

The uniqueness contract becomes:

```text
(collection_id, normalized_name, concept_type)
```

No type prefix is stored in the display name.

### 3.3 Derived representations

The following become projections of `QualifiedStatement`, not independent
sources of truth:

- FalkorDB direct navigation edge;
- statement/semantic-frame embedding;
- graph reasoning rule structures;
- query context rendering;
- analytics projection edges.

The relational statement ID is carried on every derived row or graph edge so a
projection can be rebuilt and audited.

### 3.4 Source hierarchy

Documents, folders, sections, and chunks belong to a source hierarchy, not the
canonical concept table. Reuse existing document/segment models where possible;
otherwise add a small typed source-node table.

Source hierarchy objects:

- have deterministic source IDs;
- use `document_id` before mutable `document_path`;
- retain normalized paths as metadata;
- do not receive entity centroids;
- receive embeddings only when there is meaningful title/summary content and a
  demonstrated retrieval use case.

## 4. Identity Contracts

Identity must not depend on extraction ordering.

### Source occurrence

```text
segment_id = UUID(collection_id, document_id, stable chunk occurrence)
```

When no document ID exists, use normalized path plus a stable occurrence key.
Content hash remains a separate attribute and deduplication key.

### Raw statement occurrence

```text
raw_statement_id = UUID(
  segment_id,
  normalized subject mention,
  normalized predicate,
  normalized object mention,
  qualifier hash,
  duplicate ordinal only when exact duplicates exist
)
```

The duplicate ordinal is assigned after sorting by canonical serialized content,
not by LLM response position.

### Canonical statement

```text
statement_id = UUID(
  collection_id,
  canonical subject ID,
  canonical predicate ID,
  canonical object ID,
  semantic qualifier hash
)
```

Evidence identity is separate, so the same claim from two documents shares a
statement but retains two evidence records.

## 5. Immediate Correctness Work

Before the architectural migration:

1. Add an end-to-end test that ingests one custom graph chunk and retrieves it
   through every supported custom graph query mode.
2. Define shared constants/enums for proposition type and evidence edge type.
3. Make query readers temporarily accept both:

   ```text
   ASSERTION | PROPOSITION
   HAS_ASSERTION | SUPPORTS
   ```

4. Add counters for which legacy/new vocabulary each query path encounters.
5. Add a test proving extraction entities are derived from relationship
   endpoints and cannot silently diverge from accepted relationships.
6. Add deterministic-ID tests that reorder extracted statements and expect
   unchanged IDs.
7. Stop adding new dependencies on `SUBJECT`, `OBJECT`, `CONDITION`,
   `EXCEPTION`, and `APPLIES_TO` edges.

These changes stop active data invisibility while preserving rollback
compatibility.

## 6. Service Boundaries

Reduce `_ingest_graph_chunk` to orchestration:

```text
extractor
  -> extraction cache
  -> statement resolver
  -> canonical delta writer
  -> vector materializer
  -> graph projector
  -> segment publisher
```

Proposed modules:

```text
services/graph/ingestion/contracts.py
services/graph/ingestion/extraction_cache.py
services/graph/ingestion/statement_resolver.py
services/graph/ingestion/canonical_writer.py
services/graph/ingestion/vector_materializer.py
services/graph/ingestion/graph_projector.py
services/graph/ingestion/segment_publisher.py
services/graph/reasoning/rule_compiler.py
```

Core typed boundaries:

```text
ExtractionResult
ResolvedChunk
CanonicalStatementDelta
MaterializationResult
PublicationResult
```

Persistence helpers accept typed batches and do not call the next persistence
layer.

## 7. Vector Index Policy

Keep:

- one embedding per source segment/chunk;
- canonical concept embeddings and centroids;
- one embedding per qualified statement;
- optional document/section summary embeddings when useful.

Remove:

- `EVIDENCE_CHUNK` entity embeddings;
- evidence entity centroids;
- folder/path entity embeddings and centroids;
- proposition entity embeddings when statement embeddings are enabled;
- separate qualifier-node embeddings.

Statement embedding text contains:

```text
subject + predicate + object + description +
polarity/modality + conditions/exceptions/scopes
```

Concept resolution searches only concept vectors. Statement retrieval searches
only statement vectors. Source retrieval searches only segment/document vectors.
No untyped all-entity vector search is allowed in a new query path.

## 8. Reasoning Policy

Plain-text qualifiers remain fields on `QualifiedStatement`. They must not be
presented as executable rules.

A separate rule compiler may promote a statement only when it can produce a
validated structure:

```text
CompiledRule
  antecedent statement/expression IDs
  blocker statement/expression IDs
  conclusion statement ID
  scope
  confidence
  compiler version
  source statement/evidence IDs
```

Until then:

- conditions and exceptions affect retrieval/context rendering;
- polarity and modality affect ranking and answer qualification;
- no condition becomes active solely because query tokens overlap its text;
- no `RULE`, `CONDITION`, `EXCEPTION`, or `SCOPE` `GraphEntity` is created.

For compiled rules, `ANTECEDENT_OF`, `BLOCKS`, and `CONCLUDES` may exist as a
derived reasoning projection. Redundant proposition-to-qualifier edges are not
created.

## 9. Incremental Provenance and Retraction

Classify current incremental tables by demonstrated responsibility:

| Object | Required action |
|---|---|
| `GraphChunkSegment` | Keep as idempotency and source-segment publication record |
| `GraphVersion` | Keep for compatible snapshots and reasoning versions |
| `GraphPredicateMapping` | Keep while predicate consensus reads it or its effects |
| `GraphEntityMapping` | Migrate to endpoint/raw-mention resolution mapping |
| `GraphChunkContribution` | Implement retraction consumer or remove |
| `GraphDerivedDependency` | Implement invalidation consumer or remove |

Retraction must be implemented before the contribution/dependency design is
declared complete:

1. Tombstone a source segment.
2. Remove its `StatementEvidence` rows.
3. Delete a canonical statement only when it has no evidence or derived support.
4. Rebuild/remove statement vector and Falkor projection.
5. Garbage-collect concepts only under an explicit collection policy.
6. Publish a new graph version.
7. Invalidate only snapshots depending on affected statement IDs.

Add integration tests for shared statements supported by multiple chunks. Removing
one chunk must retain the statement and its remaining evidence.

## 10. Migration and Cutover

### Phase A - Characterize and stabilize

- add ingestion-to-query contract tests;
- add vocabulary compatibility reads;
- inventory all readers of legacy node/edge types and vector tables;
- add write/read metrics and structured materialization logs;
- freeze new uses of legacy structural edges.

**Exit:** Newly ingested propositions are visible to every supported query mode.

### Phase B - Clean extraction contracts

- introduce `EndpointMention` and `ExtractedStatement`;
- derive compatibility `entities` from accepted statements only;
- eliminate unused independent entity extraction code;
- canonicalize statement serialization and order-independent local keys.

**Exit:** Extraction has one statement-first schema and stable IDs under reorder.

### Phase C - Add canonical statement schema

- add `QualifiedStatement` and `StatementEvidence` models;
- add concept type/normalized-name columns and indexes;
- add statement vector table/index;
- add foreign keys and collection-scoped uniqueness constraints;
- do not remove existing tables yet.

**Exit:** Schema supports authoritative concepts, statements, and evidence.

### Phase D - Split ingestion and dual-write

- extract resolver/writer/materializer/projector services;
- write new canonical statements and evidence;
- continue legacy writes behind `legacy_graph_materialization_enabled`;
- record old/new object counts and identity mappings;
- fail publication if the authoritative statement write fails.

**Exit:** New ingestion produces equivalent new and legacy materializations.

### Phase E - Backfill without LLM calls

- read `RawChunkExtraction`/`GraphChunkSegment.raw_extraction`;
- deterministically resolve using stored mappings where available;
- create statements/evidence/vectors in bounded batches;
- checkpoint by collection and segment;
- make reruns idempotent;
- produce a reconciliation report for missing endpoints, predicates, and evidence.

**Exit:** Existing active segments have canonical statement coverage.

### Phase F - Query cutover

- implement statement-vector retrieval;
- render query context directly from statements and evidence;
- update reasoning seeds to use statement IDs and argument columns;
- shadow old/new retrieval and compare recall, citations, latency, and result size;
- switch default reads by collection feature flag.

**Exit:** Production queries no longer require proposition entities or
`SUBJECT`/`OBJECT` edges.

### Phase G - Reasoning compiler

- retain qualifiers as data;
- define formal expression and compiled-rule schemas;
- compile only validated rules;
- replace token-overlap condition activation;
- persist proof dependencies and compiler version.

**Exit:** Executed rules have typed antecedents, blockers, conclusions, and
auditable evidence.

### Phase H - Retraction and cleanup

- implement contribution-driven segment retraction;
- verify shared-evidence behavior;
- disable legacy writes;
- remove legacy readers;
- drop redundant vectors and graph projections after a deprecation window;
- remove write-only tables if no lifecycle consumer remains.

**Exit:** One authoritative statement model remains and deletion works without a
collection scan.

## 11. Pull Request Sequence

Keep each change reviewable and reversible:

1. **Contract tests and vocabulary compatibility**
2. **Statement-first extraction DTOs and stable serialization**
3. **Canonical statement/evidence schema migration**
4. **Ingestion service boundaries and dual-write**
5. **Backfill command plus reconciliation report**
6. **Statement retrieval and shadow-read comparison**
7. **Query cutover and legacy-write disablement**
8. **Formal rule compiler**
9. **Segment retraction and dependency consumers**
10. **Legacy schema/vector/edge cleanup**

No PR should combine a schema introduction, read cutover, and legacy deletion.

## 12. Verification Matrix

### Extraction

- generic and code extraction derive endpoints from accepted statements;
- gleaning merge does not leave orphan endpoint metadata;
- reordered LLM output produces identical occurrence and canonical IDs;
- extraction cache version changes do not collide.

### Canonicalization

- same name and type resolves to the same concept;
- same name with different types remains distinct without name prefixes;
- ANN resolution is type-scoped and bounded;
- identical statements from different chunks share canonical identity;
- qualifiers that differ semantically produce distinct statements.

### Retrieval

- a freshly ingested statement is immediately retrievable;
- old and new vocabulary both work during migration;
- concept, statement, and source vector searches are type-isolated;
- citations resolve through `StatementEvidence` to the correct source segment;
- hierarchy paths do not appear as semantic concepts.

### Reasoning

- uncompiled text conditions are never treated as satisfied facts;
- compiled rules fire only with active typed antecedents;
- blockers prevent conclusions;
- every conclusion includes rule, premise, blocker, statement, and evidence IDs;
- bounded working-graph limits remain enforced.

### Incrementality

- unchanged segments cause no provider/materialization calls;
- backfill makes zero LLM calls;
- adding a fixed-size chunk has bounded lookup/write counts;
- retracting one of two supporting chunks retains the shared statement;
- retracting final support removes derived projections without scanning the
  collection.

### Migration

- dual-write reconciliation has zero unexplained statement-count differences;
- shadow retrieval meets agreed recall and citation parity;
- rollback to legacy reads works while dual-write is enabled;
- all legacy readers are absent before legacy writes are disabled;
- all production readers are absent before old tables/vectors are dropped.

## 13. Observability

Add metrics by collection and ingestion contract:

- extracted/accepted/rejected statement count;
- endpoint mention and canonical concept count;
- exact/ANN/new concept resolution count;
- canonical statement created/reused count;
- evidence occurrence count;
- legacy/new materialization divergence count;
- vector writes by object kind;
- query hits by old/new proposition vocabulary;
- shadow-read overlap and citation parity;
- retraction affected-object count and duration;
- backfill checkpoint, failures, and unresolved mappings.

Log IDs and counts, never full sensitive chunk or statement text by default.

## 14. Rollback Strategy

- Keep raw extraction immutable throughout.
- Dual-write before any read cutover.
- Feature flags are collection-scoped:

  ```text
  canonical_statement_write_enabled
  canonical_statement_read_enabled
  legacy_graph_materialization_enabled
  formal_rule_compilation_enabled
  ```

- Turning off new reads restores legacy query behavior without data rollback.
- Turning off formal compilation preserves statements and qualifiers.
- Do not drop old tables, vector tables, or graph edge types until the rollback
  window closes.
- Backfills are idempotent and resumable; rollback never deletes raw extraction.

## 15. Definition of Done

- `_ingest_graph_chunk` is a small orchestrator with typed stage boundaries.
- extraction is statement-first and has no independently drifting entity list;
- concept names do not encode type;
- source hierarchy and evidence are not canonical graph entities;
- one chunk/source embedding replaces evidence entity embedding and centroid;
- `QualifiedStatement` is the authoritative subject-predicate-object record;
- semantic-frame retrieval uses the authoritative statement representation;
- FalkorDB edges are rebuildable projections carrying statement IDs;
- `SUBJECT`, `OBJECT`, qualifier, and duplicate proposition structures are no
  longer required by production reads;
- plain-text qualifiers are not executed as formal rules;
- ingestion and query use one shared semantic vocabulary;
- provenance supports tested segment retraction, or unused provenance structures
  have been removed;
- existing collections are migrated without historical LLM extraction;
- append, query, reasoning, citation, retraction, and rollback tests pass.

## 16. Explicit Non-Goals

- redesigning all graph analytics in the same migration;
- changing public API response shapes unless required for correctness;
- re-extracting historical chunks with an LLM;
- eagerly rewriting every existing FalkorDB object in one transaction;
- treating every natural-language condition as formal logic;
- deleting compatibility storage before observed read parity and rollback
  readiness.
