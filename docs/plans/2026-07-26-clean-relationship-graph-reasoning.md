# Clean Relationship-Centric Graph Reasoning Plan

> **Status:** Proposed; directed relationship preservation landed in PR #8
>
> **Date:** 2026-07-26
>
> **Base:** `main` at `f921e52`
>
> **Scope:** Clean implementation of incremental graph ingestion, typed
> relationships, evidence, retrieval, reasoning, retraction, and versioned derived
> projections
>
> **Branch policy:** The `incremental-compiled-graph-reasoning` branch is a
> requirements and test reference only. Do not merge or broadly cherry-pick its
> ingestion, semantic-frame, reasoning-node, or incremental-schema implementation.

## 1. Objective

Extend the graph from `main` without introducing parallel representations of the
same fact.

The authoritative semantic object is one directed, typed relationship:

```text
source entity -- relationship type --> target entity
```

Conditions, exceptions, scopes, and other meaningful connections use the same
relationship abstraction and entity-resolution path as every other extracted
relationship. They are distinguished by `GraphRelationshipType`, not by nested
qualifier fields or special entity classes. `RelationshipDescription` records
how the endpoints are related in a source occurrence.

```text
source segment
  -> immutable extraction
  -> endpoint resolution
  -> canonical relationships + descriptions/evidence
  -> derived indexes/projections
```

The design must remain bounded for incremental ingestion and query. Adding one
chunk may inspect only the chunk and fixed-size indexed candidate sets; querying
may activate only a bounded working graph.

## 2. What Exists on `main`

`main` already provides useful primitives:

- `RawChunkExtraction` caches LLM extraction;
- the extractor emits relationships with nested source and target endpoints;
- `ExtractionResult.entities` is derived from relationship endpoints;
- `IncrementalEntityResolver` performs exact, alias, and bounded vector
  resolution;
- `GraphEntity` stores canonical entities;
- `GraphRelationship` stores source, relationship type, and target;
- `RelationshipDescription` stores source-specific descriptions and keywords;
- chunk, entity, relationship, and centroid vector indexes exist;
- FalkorDB is a graph projection of canonical SQL entities and relationships;
- document ingestion has durable job/chunk work records.

The clean implementation should evolve these objects instead of adding
`PROPOSITION` entities, `SUBJECT`/`OBJECT` edges, evidence entities, semantic-frame
truth tables, condition/rule entities, or a parallel statement model.

## 3. Baseline Defects and Completed Prerequisites

These concerns are independent of the discarded feature implementation.

### 3.1 Direction loss is fixed

PR #8 changed `resolve_relationship()` to reuse a relationship only when its
type and ordered `(source, target)` endpoints match. A regression test
confirms that ingesting `(A, R, B)` and `(B, R, A)` creates two relationships.

Current behavior:

```text
(A, R, B) != (B, R, A)
```

Explicit symmetric-type normalization is not implemented yet. It belongs
with the relationship-type metadata and canonical relationship identity work
below; until
then, all relationship types preserve extracted direction.

### 3.2 Extraction has two drifting views

The LLM schema emits relationships with nested endpoints, but the service creates
a separate `entities` list and later loops over both lists.

The endpoint list is derived data. It must not become a second mutable extraction
contract.

### 3.3 Raw extraction mixes cache identity and source occurrence

`RawChunkExtraction` is unique by `(collection_id, chunk_content_hash)` but also
stores one `document_id` and `document_path`. The same content appearing in two
documents therefore has one cache record but multiple real source occurrences.

Content-level extraction and document-level provenance need separate identities.

### 3.4 Extraction cache lacks an explicit contract key

Changing prompts, schema, domain rules, normalization, or gleaning behavior can
reuse stale extraction because the uniqueness key does not include an extraction
contract version.

### 3.5 Canonical entity uniqueness ignores type

`GraphEntity` is unique by `(collection_id, canonical_name)`. Two legitimate
entities with the same name and different semantic types cannot coexist without
encoding type into the name or incorrectly merging them.

### 3.6 IDs and writes are not suitable for deterministic replay

Entity, relationship, and relationship-description IDs are generally random.
Relationship ingestion commits inside the per-relationship loop. A retry can
re-resolve and re-aggregate rather than deterministically replay one chunk delta.

## 4. Non-Negotiable Invariants

### I1 - One relationship abstraction

`GraphRelationship` is the authoritative semantic edge. Conditions, exceptions,
and general relationships use the same record, resolver, evidence model, vector
index, and graph projection. No proposition entity, semantic frame, qualifier
record, or parallel statement table independently owns relationship semantics.

### I2 - Source records are not semantic entities

Documents, folders, sections, chunks, evidence, and arbitrary condition/rule text
do not enter `GraphEntity` merely because the graph store supports nodes. Entity
mentions used as endpoints of `CONDITION`, `EXCEPTION`, or any other relationship
are canonicalized normally.

### I3 - Directed semantics

Direction survives extraction, resolution, persistence, embedding, projection,
analytics, and query unless the relationship-type metadata explicitly declares
symmetry.

### I4 - Immutable extraction

The exact structured extractor output is immutable under a versioned extraction
contract. Historical chunks are never sent back to an LLM merely because
canonicalization or reasoning changes.

### I5 - Occurrence and content are distinct

Content hashing enables extraction reuse. Source-segment identity preserves
document, path, chunk position, and independent evidence.

### I6 - Derived projections are rebuildable

Every FalkorDB edge, vector row, compiled rule, analytics edge, and snapshot can
be rebuilt from canonical SQL records and carries the authoritative object ID.

### I7 - Relationship descriptions are not executable logic

`CONDITION` and `EXCEPTION` relationships are queryable semantic edges. Their
descriptions preserve how the source relates their endpoints, but prose alone
does not become executable logic. Formal execution requires a typed compiler.

### I8 - Bounded work

Foreground ingestion does not scan historical chunks, entities, relationships,
vectors, or projections. Query reasoning never loads the complete collection.

### I9 - Retraction is part of ingestion

Any provenance/dependency structure added during ingestion must have a tested
reader that retracts or invalidates the affected materialization.

## 5. Target Model

### 5.1 Source occurrence

Add a durable `SourceSegment` distinct from `IngestionChunk`.

`IngestionChunk` remains a job work item. `SourceSegment` represents a stable
source occurrence across jobs:

```text
SourceSegment
  id
  collection_id
  document_id
  normalized_document_path
  chunk_index
  content_hash
  content
  active/tombstoned status
  created_at
```

Identity:

```text
UUID(collection_id, document_id-or-path, stable chunk index, content hash)
```

Prefer immutable `document_id`; use normalized path only as a fallback. A path is
metadata and may change without changing a document-backed segment identity.

### 5.2 Content-level extraction

Replace the mixed responsibility of `RawChunkExtraction` with:

```text
RawExtraction
  id
  collection_id
  content_hash
  contract_version
  domain
  payload_json
  created_at

SegmentExtraction
  segment_id
  raw_extraction_id
```

Unique cache key:

```text
(collection_id, content_hash, contract_version)
```

`payload_json` preserves the extractor response in a relationship-first schema. It
does not store document/path provenance.

During migration, existing `RawChunkExtraction` may remain the physical table
while new columns and a segment link are introduced. The responsibility split is
more important than an immediate table rename.

### 5.3 Extraction contract

Use one DTO:

```text
ExtractedRelationship
  local_key
  source: EndpointMention
  relationship_type
  target: EndpointMention
  description
  keywords
  weight
  confidence
  relationship_type_properties

EndpointMention
  name
  type_hint
  description
```

`ExtractionResult` contains `relationships`. Conditions and exceptions are
emitted as relationships whose types are `CONDITION` and `EXCEPTION`; they are
not nested string arrays. A temporary `entities` property may derive unique
endpoint mentions from every accepted relationship for compatibility, but it is
never serialized as an independent source of truth.

The extractor assigns `local_key` from canonical serialized content. Response
array order is not identity.

### 5.4 Canonical entities

Evolve `GraphEntity` as the canonical entity table:

```text
canonical_name
normalized_name
primary_type
```

New uniqueness contract:

```text
(collection_id, normalized_name, primary_type)
```

Do not prefix display names with type. Aliases remain separate and type-aware.
Unknown type is explicit rather than encoded as an empty string.

Entity resolution remains:

1. type-scoped exact canonical match;
2. type-scoped exact alias match;
3. bounded type-compatible ANN candidates;
4. create a new entity.

All paths have fixed top-k, score threshold, timeout, and retry limits.

### 5.5 Canonical relationships

Evolve `GraphRelationship`; do not add a parallel proposition or statement table:

```text
GraphRelationship
  id
  collection_id
  source_entity_id
  relationship_type_id
  target_entity_id
  confidence
  aggregate_weight
```

Canonical identity:

```text
UUID(
  collection_id,
  source entity ID,
  canonical relationship type ID,
  target entity ID
)
```

Database uniqueness enforces the same tuple. Opposite directions remain distinct.
For a symmetric relationship type, normalize endpoint order before computing
identity.

Every meaningful extracted connection uses this identity, including `CONDITION`,
`EXCEPTION`, and `SCOPE` relationships. The relationship description supplies
source-specific semantics without changing the canonical endpoint/type identity.

### 5.6 Relationship evidence

Evolve `RelationshipDescription` into the evidence-occurrence record:

```text
RelationshipEvidence
  id
  relationship_id
  source_segment_id
  raw_relationship_local_key
  description
  keywords
  extracted_weight
  extraction_confidence
  metadata_json
```

Identity:

```text
UUID(relationship_id, source_segment_id, raw_relationship_local_key)
```

One relationship may have many evidence rows. Aggregate weight and consensus fields
are derived from active evidence rows, never incremented blindly on retry.

During compatibility the physical table may remain
`relationship_descriptions`, with new deterministic keys and a
`source_segment_id` foreign key.

### 5.7 Relationship-type metadata

Evolve `GraphRelationshipType` with explicit, observed properties:

```text
directionality
symmetry
transitivity
causal
temporal
property_votes_json
observation_count
```

Raw labels map to a canonical relationship type through a versioned mapping.
Property consensus never changes historical extraction and never silently flips
existing edge direction.

## 6. Derived Storage Policy

### 6.1 PostgreSQL

PostgreSQL owns:

- source segments;
- immutable extraction payloads;
- endpoint-to-entity resolution records;
- canonical entities and relationship types;
- canonical relationships;
- relationship evidence;
- optional compiled rules;
- publication/materialization status.

### 6.2 Vector indexes

Keep separate typed indexes:

| Index | Content | Purpose |
|---|---|---|
| source segment | original chunk text | source retrieval and citation |
| entity description | mention-specific entity description | entity resolution/retrieval |
| entity centroid | aggregate entity semantics | bounded canonical resolution |
| relationship | source, type, target, and description | relationship retrieval |

Do not create vectors for:

- evidence entities;
- folders or raw paths;
- proposition entities;
- subject/object structural edges;
- condition/exception/scope text nodes.

The relationship vector covers the semantic-frame retrieval capability without
another persisted semantic object. `CONDITION` and `EXCEPTION` relationships are
embedded and retrieved through this same index.

### 6.3 FalkorDB

Project each canonical relationship as one direct edge:

```text
(source)-[relationship_type {
  relationship_id,
  confidence
}]->(target)
```

FalkorDB entity nodes contain entity identity/display metadata only. Source
segments and evidence do not become ordinary semantic entity nodes.

If a query needs evidence traversal, fetch evidence from PostgreSQL by
`relationship_id`. Add a separate named provenance projection only after a measured
graph-traversal requirement.

### 6.4 Source hierarchy

Represent document/folder/section hierarchy in source metadata tables.

- Path filters and section lookup use SQL indexes.
- Source retrieval embeds meaningful document/section title plus summary only.
- Source hierarchy is not included in semantic graph analytics.
- A named source-containment projection may be built separately if required.

## 7. Ingestion Service Boundaries

Reduce `_ingest_graph_chunk` to an orchestrator:

```text
segment = source_segments.get_or_create(...)
extraction = extraction_cache.get_or_extract(segment.content, contract)
resolved = relationship_resolver.resolve(extraction.relationships)
delta = canonical_writer.upsert(segment, resolved)
materializers.apply(delta)
publisher.complete(segment, delta)
```

Proposed modules:

```text
services/graph/ingestion/contracts.py
services/graph/ingestion/source_segments.py
services/graph/ingestion/extraction_cache.py
services/graph/ingestion/relationship_resolver.py
services/graph/ingestion/canonical_writer.py
services/graph/ingestion/materializer.py
services/graph/ingestion/publisher.py
```

Typed boundaries:

```text
ExtractionContract
ExtractedRelationship
ResolvedEndpoint
ResolvedRelationship
CanonicalChunkDelta
MaterializationResult
```

Rules:

- extraction code performs no canonical writes;
- resolution code performs no FalkorDB writes;
- canonical writer performs no vector or graph writes;
- materializers consume committed canonical IDs;
- publication succeeds only after required materializers succeed;
- one chunk uses bounded batch operations, not per-relationship commits.

## 8. Publication, Idempotency, and Versioning

Add a minimal `SegmentMaterialization`:

```text
segment_id
contract_version
materializer_version
status
delta_manifest_json
completed_at
```

This is the retry boundary. A completed matching materialization returns without
LLM, embedding, resolution, SQL semantic, or FalkorDB work.

`delta_manifest_json` lists relationship and evidence IDs created/reused for the
segment. Do not add a general dependency graph until retraction or a compiler has
a concrete reader for it.

Introduce collection graph versions only when a consumer requires snapshot
identity:

- analytics snapshot;
- compiled rule set;
- cached reasoning trace;
- external version-pinned query.

A version references a completed set of segment materializations. It does not
duplicate canonical objects.

## 9. Retraction

Retraction is evidence-first:

1. Tombstone the source segment.
2. Mark/remove its active relationship-evidence rows.
3. Recompute affected relationship aggregate fields from remaining evidence.
4. Delete an unsupported relationship when no active evidence or explicit derived
   support remains.
5. Remove/rebuild its relationship vector and FalkorDB edge.
6. Invalidate only snapshots/compiled rules that list the relationship ID.
7. Garbage-collect unsupported entities only under an explicit policy.

Required fixture:

```text
segment A ─┐
           ├─ supports relationship R
segment B ─┘
```

Retracting A retains R and B's evidence. Retracting B then removes R and its
derived projections. Neither operation scans the collection.

## 10. Query and Retrieval

Build a bounded query plan:

```text
question
  -> entity seeds
  -> relationship-vector seeds
  -> typed bounded neighborhoods
  -> evidence hydration
  -> bounded working graph
  -> answer plus citations/trace
```

Query paths use authoritative IDs:

- entities from `GraphEntity`;
- relationships from `GraphRelationship`;
- condition/exception/general semantics from `GraphRelationshipType`;
- evidence/citations from `RelationshipDescription`/`RelationshipEvidence`;
- source text from `SourceSegment`;
- traversal from FalkorDB edges carrying `relationship_id`.

There is no `ASSERTION` versus `PROPOSITION` vocabulary because neither is a
entity-node type. All semantic edges use the relationship record.

Hard limits:

- entity ANN top-k;
- relationship ANN top-k;
- graph hops;
- nodes and edges;
- evidence rows per relationship;
- runtime and memory;
- optional community/projection expansion count.

## 11. Reasoning

### 11.1 Typed-relationship reasoning

Before formal rules exist, reasoning may:

- retrieve relationships by semantic similarity;
- traverse and filter by relationship type, including `CONDITION` and
  `EXCEPTION`;
- rank by confidence and source evidence;
- surface supporting and conflicting evidence;
- explain paths without claiming formal proof.

Condition and exception endpoints are normal canonical entities, so queries about
those entities retrieve the relationships directly. Relationship descriptions
are retrieval/evidence text; lexical overlap with a description does not execute
a formal rule.

### 11.2 Formal rule compiler

Add `CompiledRule` only after the extractor or deterministic parser can emit typed
logic:

```text
CompiledRule
  id
  collection_id
  antecedent_expression_json
  blocker_expression_json
  conclusion_relationship_id
  scope_json
  confidence
  compiler_version
  source_relationship_ids
```

An expression references canonical entities/relationships or typed variables. The
compiler rejects unbound or unresolved natural-language clauses.

Rule indexes support:

- antecedent-to-rule lookup;
- conclusion-to-rule lookup;
- blocker lookup;
- proof dependency tracing.

A rule graph (`ANTECEDENT_OF`, `BLOCKS`, `CONCLUDES`) is a derived reasoning
projection. It does not require `RULE`, `CONDITION`, or `EXCEPTION`
`GraphEntity` records.

### 11.3 Working-graph execution

1. Resolve query entities and candidate relationships.
2. Activate relevant relationships.
3. Expand backward from desired conclusions and forward from active facts.
4. Evaluate typed antecedents and blockers.
5. Record proof/counterproof dependencies.
6. Stop at fixed point, answer, conflict, or resource limit.
7. Return relationship, rule, evidence, and source-segment IDs in the trace.

## 12. Analytics

Analytics runs on named projections of canonical relationships:

```text
semantic_directed
causal_directed
code_dependency_directed
rule_dependency_directed
semantic_affinity_undirected
```

Each projection declares:

- included relationship types;
- direction/symmetry behavior;
- confidence/evidence threshold;
- parallel relationship aggregation;
- self-loop policy;
- weight interpretation.

Source hierarchy, evidence, rule scaffolding, and query-time nodes are excluded
unless explicitly named.

Global analytics is asynchronous and versioned. It never blocks chunk ingestion.
Cheap relationship/evidence counters may update incrementally.

## 13. Delivery Sequence

### Completed prerequisite - Preserve extracted direction

- add directed relationship fixtures;
- remove unconditional reverse-edge matching;
- merge PR #8 into `main`.

**Result:** Opposite directed facts no longer merge accidentally.

### PR 1 - Characterize `main` and add explicit symmetry

- add relationship-type metadata before allowing symmetric normalization;
- add symmetric-type fixtures;
- add ingestion-to-query contract tests;
- record current provider calls and SQL/graph writes per chunk.

**Exit:** Direction remains the default, and only relationship types explicitly marked
symmetric normalize endpoint order.

### PR 2 - Relationship-first extraction contract

- introduce `EndpointMention` and `ExtractedRelationship`;
- emit conditions, exceptions, and scopes as typed relationships;
- derive the endpoint entity list from all accepted relationships;
- version the extraction contract;
- add canonical serialization and order-independent local keys.

**Exit:** One immutable, versioned relationship payload drives ingestion, and
every relationship endpoint uses the same entity-resolution path.

### PR 3 - Source segment and extraction split

- add durable `SourceSegment`;
- link job chunks to source segments;
- separate content-level extraction cache from source occurrence;
- update chunk embedding to use real segment/chunk position;
- migrate existing cache rows without LLM calls.

**Exit:** Identical content in multiple documents has one extraction and multiple
source/evidence occurrences.

### PR 4 - Type-aware entities

- add normalized entity name;
- add type-aware uniqueness and aliases;
- bound/type-scope every resolution lookup;
- provide deterministic entity creation under concurrency;
- backfill normalized names and types.

**Exit:** Same-name entities of different types do not collide.

### PR 5 - Deterministic canonical relationships and evidence

- add deterministic directed relationship identity;
- add source-segment/local-key identity to `RelationshipDescription`;
- replace blind weight increments with evidence-derived aggregates;
- batch the canonical writes in one transaction.

**Exit:** One `GraphRelationship` record authoritatively represents each typed
edge, including conditions and exceptions.

### PR 6 - Typed materializers

- add relationship vector index;
- project relationship IDs into FalkorDB edges;
- make chunk, entity, and relationship materializers independent consumers;
- add materialization status/retry boundary;
- add reconciliation commands for SQL/vector/Falkor parity.

**Exit:** Derived stores are rebuildable and idempotent.

### PR 7 - Query cutover to canonical relationships

- retrieve relationship embeddings directly;
- hydrate relationship types, descriptions, and evidence from SQL;
- use source segments for citations;
- enforce typed search and working-graph budgets;
- remove any need for proposition/assertion nodes.

**Exit:** End-to-end graph RAG answers use entities, relationships, evidence, and
source segments only.

### PR 8 - Retraction

- implement evidence-first source-segment tombstoning;
- recompute only affected relationship aggregates;
- remove/rebuild affected vectors and graph edges;
- add shared-evidence and retry fixtures.

**Exit:** A changed/deleted source retracts only its affected materialization.

### PR 9 - Typed relationship reasoning

- retrieve and traverse `CONDITION`, `EXCEPTION`, `SCOPE`, and general
  relationship types uniformly;
- add relationship-type/confidence/evidence-aware ranking;
- expose support/conflict/citation traces;
- clearly label non-formal reasoning;
- persist no rule nodes.

**Exit:** Useful bounded reasoning works without pretending text clauses are
formal logic.

### PR 10 - Formal rule compiler

- add typed expression and compiled-rule schema;
- compile only validated antecedents, blockers, and conclusions;
- add bounded forward/backward execution;
- persist proof dependencies and compiler version.

**Exit:** Formal conclusions are reproducible and auditable.

### PR 11 - Named analytics projections

- define projection contracts;
- compute versioned snapshots asynchronously;
- add affected-set invalidation;
- keep global metrics out of foreground ingestion.

**Exit:** Analytics cannot accidentally mix semantic facts with provenance or
reasoning scaffolding.

## 14. Feature-Branch Salvage Policy

Do not port feature-branch modules wholesale.

For each desired behavior:

1. Write or copy the smallest behavior-level test onto the clean branch.
2. Make the test describe the target model in this document.
3. Implement against `main` abstractions.
4. Compare behavior and performance with the prototype.
5. Port isolated algorithms only when they have no dependency on proposition
   entities, semantic-frame truth tables, source-as-entity modeling, or write-only
   provenance schemas.

Potentially reusable as behavior/tests:

- bounded query budgets and timings;
- Unicode-aware chunking fixes;
- provider concurrency hardening;
- goal-directed search heuristics;
- named projection analytics algorithms;
- job cancellation/progress behavior.

Must be redesigned, not cherry-picked:

- custom `_ingest_graph_chunk` proposition/context materialization;
- source hierarchy as canonical entities;
- evidence entity vectors/centroids;
- semantic frames as a separate authoritative table;
- `SUBJECT`/`OBJECT` duplication;
- plain-text condition/rule entities instead of typed relationships;
- contribution/dependency tables without retraction consumers.

## 15. Verification Matrix

### Identity and direction

- `(A, R, B)` and `(B, R, A)` are distinct for directed R;
- symmetric R normalizes endpoint order;
- relationship IDs survive extraction reorder and retry;
- identical typed relationships from different segments share a relationship ID;
- condition and exception relationships use the same identity rules as all
  other relationships.

### Extraction and provenance

- endpoint mentions derive from all accepted relationships only;
- entities mentioned by condition and exception relationships are resolved and
  queryable like every other endpoint;
- changing contract version creates a new extraction cache entry;
- identical content in two documents reuses extraction but retains two segments;
- backfill and retry invoke no historical LLM calls;
- citations point to the correct segment and document.

### Entities

- same normalized name and type resolves together;
- same normalized name with different types remains separate;
- ANN candidates are type-compatible and bounded;
- concurrent creation returns one canonical entity.

### Persistence and materialization

- one retry does not double relationship weight/evidence;
- one chunk commits canonical writes atomically;
- SQL is authoritative when a vector/graph materializer fails;
- retry repairs only incomplete projections;
- reconciliation detects missing/stale vector and Falkor objects.

### Query and reasoning

- fresh ingestion is immediately queryable after publication;
- relationship retrieval returns its type, descriptions, and evidence;
- querying an entity returns its condition and exception relationships;
- source paths never appear as canonical semantic entities;
- relationship descriptions are never executed as formal rules;
- compiled rules require typed active antecedents and inactive blockers;
- every answer trace includes authoritative IDs and citations;
- working graph respects node, edge, hop, runtime, and memory limits.

### Retraction and scale

- retracting one of two evidence occurrences retains the relationship;
- retracting final support removes affected derived objects;
- retraction does not scan unrelated relationships;
- fixed-size append lookup/write counts remain bounded as collection size grows;
- global analytics failure never blocks ingestion or base query.

## 16. Observability

Record by collection and contract version:

- cache hit/miss and extractor calls;
- accepted/rejected relationships by type;
- exact/alias/ANN/new entity resolutions;
- created/reused relationships;
- evidence occurrences;
- canonical transaction duration;
- vector/Falkor materialization status;
- retry and reconciliation counts;
- retraction affected-object counts;
- query seed/working-graph sizes;
- reasoning stop reason;
- analytics version/staleness.

Log IDs and counts, not full source or extracted text by default.

## 17. Definition of Done

- `main`'s reverse-edge merge bug remains fixed (landed in PR #8);
- extraction is relationship-first and contract-versioned;
- source occurrence and content-level extraction cache are distinct;
- canonical entity identity is type-aware without type-prefixed display names;
- `GraphRelationship` is the sole authoritative semantic-edge record;
- `CONDITION`, `EXCEPTION`, and general relationship types share one ingestion,
  resolution, persistence, embedding, and retrieval path;
- `RelationshipDescription`/evidence links relationships to stable source
  segments;
- retries are deterministic and do not double-count evidence;
- `_ingest_graph_chunk` is a small typed orchestrator;
- chunk, entity, and relationship vector indexes have distinct responsibilities;
- FalkorDB contains rebuildable entity/relationship projections;
- source hierarchy is modeled outside the canonical entity graph;
- no proposition/assertion, subject/object, evidence, condition, exception, or
  rule entities
  are required;
- relationship descriptions are not executed as formal rules;
- formal rules, when enabled, use typed expressions and auditable dependencies;
- retraction is implemented and bounded;
- query and reasoning operate on bounded working graphs;
- analytics runs on named, versioned projections;
- historical migration and backfill require zero LLM re-extraction.

## 18. Non-Goals

- merging the prototype feature branch;
- preserving prototype internal schemas merely for code reuse;
- re-running LLM extraction over historical chunks;
- graphifying every domain object;
- implementing formal logic from arbitrary relationship descriptions;
- adding dependency/provenance tables without lifecycle readers;
- running global compilation or analytics in foreground ingestion;
- renaming every existing SQL table/class before behavior is correct.
