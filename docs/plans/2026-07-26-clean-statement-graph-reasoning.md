# Clean Statement-Centric Graph Reasoning Plan

> **Status:** Proposed; directed relationship preservation landed in PR #8
>
> **Date:** 2026-07-26
>
> **Base:** `main` at `f921e52`
>
> **Scope:** Clean implementation of incremental graph ingestion, qualified
> statements, evidence, retrieval, reasoning, retraction, and versioned derived
> projections
>
> **Branch policy:** The `incremental-compiled-graph-reasoning` branch is a
> requirements and test reference only. Do not merge or broadly cherry-pick its
> ingestion, semantic-frame, reasoning-node, or incremental-schema implementation.

## 1. Objective

Extend the graph from `main` without introducing parallel representations of the
same fact.

The authoritative semantic object is one directed, qualified statement:

```text
subject concept -- canonical predicate --> object concept
```

The statement owns semantic qualifiers such as polarity, modality, conditions,
exceptions, scopes, confidence, and predicate properties. Source occurrences are
linked as evidence. FalkorDB edges, vector rows, query context, analytics edges,
and compiled reasoning structures are derived projections.

```text
source segment
  -> immutable extraction
  -> endpoint resolution
  -> canonical statement + evidence
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
- `GraphEntity` stores canonical concepts;
- `GraphRelationship` stores source, predicate, and target;
- `RelationshipDescription` stores source-specific descriptions and keywords;
- chunk, entity, relationship, and centroid vector indexes exist;
- FalkorDB is a graph projection of canonical SQL entities and relationships;
- document ingestion has durable job/chunk work records.

The clean implementation should evolve these objects instead of adding
`PROPOSITION` entities, `SUBJECT`/`OBJECT` edges, evidence entities, semantic-frame
truth tables, or condition/rule entities.

## 3. Baseline Defects and Completed Prerequisites

These concerns are independent of the discarded feature implementation.

### 3.1 Direction loss is fixed

PR #8 changed `resolve_relationship()` to reuse a relationship only when its
predicate and ordered `(source, target)` endpoints match. A regression test
confirms that ingesting `(A, R, B)` and `(B, R, A)` creates two relationships.

Current behavior:

```text
(A, R, B) != (B, R, A)
```

Explicit symmetric-predicate normalization is not implemented yet. It belongs
with the predicate metadata and canonical statement identity work below; until
then, all predicates preserve extracted direction.

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

### 3.5 Canonical concept uniqueness ignores type

`GraphEntity` is unique by `(collection_id, canonical_name)`. Two legitimate
concepts with the same name and different semantic types cannot coexist without
encoding type into the name or incorrectly merging them.

### 3.6 IDs and writes are not suitable for deterministic replay

Entity, relationship, and relationship-description IDs are generally random.
Relationship ingestion commits inside the per-relationship loop. A retry can
re-resolve and re-aggregate rather than deterministically replay one chunk delta.

## 4. Non-Negotiable Invariants

### I1 - One semantic source of truth

One SQL statement record is authoritative. No proposition entity, semantic frame,
or FalkorDB edge independently owns statement semantics.

### I2 - Source objects are not concepts

Documents, folders, sections, chunks, evidence, conditions, rules, and scopes do
not enter `GraphEntity` merely because the graph store supports nodes.

### I3 - Directed semantics

Direction survives extraction, resolution, persistence, embedding, projection,
analytics, and query unless the predicate metadata explicitly declares symmetry.

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

### I7 - Natural language is not executable logic

Plain-text conditions and exceptions remain qualifiers. They become executable
only after a typed rule compiler produces validated antecedents, blockers, and
conclusions.

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

`payload_json` preserves the extractor response in a statement-first schema. It
does not store document/path provenance.

During migration, existing `RawChunkExtraction` may remain the physical table
while new columns and a segment link are introduced. The responsibility split is
more important than an immediate table rename.

### 5.3 Extraction contract

Use one DTO:

```text
ExtractedStatement
  local_key
  subject: EndpointMention
  predicate
  object: EndpointMention
  description
  keywords
  weight
  polarity
  modality
  conditions
  exceptions
  scopes
  predicate_properties

EndpointMention
  name
  type_hint
  description
```

`ExtractionResult` contains `statements`. A temporary `entities` property may
derive unique endpoint mentions for compatibility, but it is never serialized as
an independent source of truth.

The extractor assigns `local_key` from canonical serialized content. Response
array order is not identity.

### 5.4 Canonical concepts

Evolve `GraphEntity` as the canonical concept table:

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

Concept resolution remains:

1. type-scoped exact canonical match;
2. type-scoped exact alias match;
3. bounded type-compatible ANN candidates;
4. create a new concept.

All paths have fixed top-k, score threshold, timeout, and retry limits.

### 5.5 Canonical qualified statements

Evolve `GraphRelationship`; do not add a parallel proposition table.

Conceptually rename it `CanonicalStatement`, while retaining the current class and
table names during compatibility:

```text
GraphRelationship / CanonicalStatement
  id
  collection_id
  source_entity_id
  relationship_type_id
  target_entity_id
  polarity
  modality
  conditions_json
  exceptions_json
  scopes_json
  predicate_properties_json
  qualifier_hash
  confidence
  aggregate_weight
```

Canonical identity:

```text
UUID(
  collection_id,
  source concept ID,
  canonical predicate ID,
  target concept ID,
  polarity,
  modality,
  semantic qualifier hash
)
```

Database uniqueness enforces the same tuple. Opposite directions remain distinct.
For a symmetric predicate, normalize endpoint order before computing identity.

### 5.6 Statement evidence

Evolve `RelationshipDescription` into the evidence-occurrence record:

```text
StatementEvidence
  id
  statement_id
  source_segment_id
  raw_statement_local_key
  description
  keywords
  extracted_weight
  extraction_confidence
  metadata_json
```

Identity:

```text
UUID(statement_id, source_segment_id, raw_statement_local_key)
```

One statement may have many evidence rows. Aggregate weight and consensus fields
are derived from active evidence rows, never incremented blindly on retry.

During compatibility the physical table may remain
`relationship_descriptions`, with new deterministic keys and a
`source_segment_id` foreign key.

### 5.7 Predicate metadata

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

Raw predicate labels map to a canonical predicate through a versioned mapping.
Property consensus never changes historical extraction and never silently flips
existing edge direction.

## 6. Derived Storage Policy

### 6.1 PostgreSQL

PostgreSQL owns:

- source segments;
- immutable extraction payloads;
- endpoint-to-concept resolution records;
- canonical concepts and predicates;
- canonical qualified statements;
- statement evidence;
- optional compiled rules;
- publication/materialization status.

### 6.2 Vector indexes

Keep separate typed indexes:

| Index | Content | Purpose |
|---|---|---|
| source segment | original chunk text | source retrieval and citation |
| concept description | mention-specific concept description | concept resolution/retrieval |
| concept centroid | aggregate concept semantics | bounded canonical resolution |
| statement | rendered qualified statement | claim/reasoning retrieval |

Do not create vectors for:

- evidence entities;
- folders or raw paths;
- proposition entities;
- subject/object structural edges;
- condition/exception/scope nodes.

The statement vector is the semantic-frame capability. A semantic frame is a
rendered DTO over a canonical statement, not another persisted semantic object.

### 6.3 FalkorDB

Project each canonical statement as one direct edge:

```text
(subject)-[predicate {
  statement_id,
  polarity,
  modality,
  confidence,
  qualifier_hash
}]->(object)
```

FalkorDB concept nodes contain concept identity/display metadata only. Source
segments and evidence do not become ordinary concept nodes.

If a query needs evidence traversal, fetch evidence from PostgreSQL by
`statement_id`. Add a separate named provenance projection only after a measured
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
resolved = statement_resolver.resolve(extraction.statements)
delta = canonical_writer.upsert(segment, resolved)
materializers.apply(delta)
publisher.complete(segment, delta)
```

Proposed modules:

```text
services/graph/ingestion/contracts.py
services/graph/ingestion/source_segments.py
services/graph/ingestion/extraction_cache.py
services/graph/ingestion/statement_resolver.py
services/graph/ingestion/canonical_writer.py
services/graph/ingestion/materializer.py
services/graph/ingestion/publisher.py
```

Typed boundaries:

```text
ExtractionContract
ExtractedStatement
ResolvedEndpoint
ResolvedStatement
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

`delta_manifest_json` lists statement and evidence IDs created/reused for the
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
2. Mark/remove its active statement-evidence rows.
3. Recompute affected statement aggregate fields from remaining evidence.
4. Delete an unsupported statement when no active evidence or explicit derived
   support remains.
5. Remove/rebuild its statement vector and FalkorDB edge.
6. Invalidate only snapshots/compiled rules that list the statement ID.
7. Garbage-collect unsupported concepts only under an explicit policy.

Required fixture:

```text
segment A ─┐
           ├─ supports statement S
segment B ─┘
```

Retracting A retains S and B's evidence. Retracting B then removes S and its
derived projections. Neither operation scans the collection.

## 10. Query and Retrieval

Build a bounded query plan:

```text
question
  -> concept seeds
  -> statement-vector seeds
  -> typed bounded neighborhoods
  -> evidence hydration
  -> bounded working graph
  -> answer plus citations/trace
```

Query paths use authoritative IDs:

- concepts from `GraphEntity`;
- statements from `GraphRelationship`;
- qualifiers from statement columns;
- evidence/citations from `RelationshipDescription`/`StatementEvidence`;
- source text from `SourceSegment`;
- traversal from FalkorDB edges carrying `statement_id`.

There is no `ASSERTION` versus `PROPOSITION` vocabulary because neither is a
concept-node type. The statement is identified by its statement record.

Hard limits:

- concept ANN top-k;
- statement ANN top-k;
- graph hops;
- nodes and edges;
- evidence rows per statement;
- runtime and memory;
- optional community/projection expansion count.

## 11. Reasoning

### 11.1 Qualified-statement reasoning

Before formal rules exist, reasoning may:

- retrieve statements by semantic similarity;
- filter/rank by polarity, modality, confidence, scope, and exceptions;
- traverse typed statement edges;
- surface supporting and conflicting evidence;
- explain paths without claiming formal proof.

Plain-text conditions are rendered as applicability qualifications. They are not
activated by lexical overlap and do not become graph nodes.

### 11.2 Formal rule compiler

Add `CompiledRule` only after the extractor or deterministic parser can emit typed
logic:

```text
CompiledRule
  id
  collection_id
  antecedent_expression_json
  blocker_expression_json
  conclusion_statement_id
  scope_json
  confidence
  compiler_version
  source_statement_ids
```

An expression references canonical concepts/statements or typed variables. The
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

1. Resolve query concepts and candidate statements.
2. Activate asserted, applicable statements.
3. Expand backward from desired conclusions and forward from active facts.
4. Evaluate typed antecedents and blockers.
5. Record proof/counterproof dependencies.
6. Stop at fixed point, answer, conflict, or resource limit.
7. Return statement, rule, evidence, and source-segment IDs in the trace.

## 12. Analytics

Analytics runs on named projections of canonical statements:

```text
semantic_directed
causal_directed
code_dependency_directed
rule_dependency_directed
semantic_affinity_undirected
```

Each projection declares:

- included predicates and qualifiers;
- direction/symmetry behavior;
- confidence/evidence threshold;
- parallel statement aggregation;
- self-loop policy;
- weight interpretation.

Source hierarchy, evidence, rule scaffolding, and query-time nodes are excluded
unless explicitly named.

Global analytics is asynchronous and versioned. It never blocks chunk ingestion.
Cheap statement/evidence counters may update incrementally.

## 13. Delivery Sequence

### Completed prerequisite - Preserve extracted direction

- add directed relationship fixtures;
- remove unconditional reverse-edge matching;
- merge PR #8 into `main`.

**Result:** Opposite directed facts no longer merge accidentally.

### PR 1 - Characterize `main` and add explicit symmetry

- add predicate metadata before allowing symmetric normalization;
- add symmetric-predicate fixtures;
- add ingestion-to-query contract tests;
- record current provider calls and SQL/graph writes per chunk.

**Exit:** Direction remains the default, and only predicates explicitly marked
symmetric normalize endpoint order.

### PR 2 - Statement-first extraction contract

- introduce `EndpointMention` and `ExtractedStatement`;
- make endpoint entity lists computed compatibility data;
- add polarity, modality, qualifiers, and predicate properties to schema;
- version the extraction contract;
- add canonical serialization and order-independent local keys.

**Exit:** One immutable, versioned statement payload drives ingestion.

### PR 3 - Source segment and extraction split

- add durable `SourceSegment`;
- link job chunks to source segments;
- separate content-level extraction cache from source occurrence;
- update chunk embedding to use real segment/chunk position;
- migrate existing cache rows without LLM calls.

**Exit:** Identical content in multiple documents has one extraction and multiple
source/evidence occurrences.

### PR 4 - Type-aware concepts

- add normalized concept name;
- add type-aware uniqueness and aliases;
- bound/type-scope every resolution lookup;
- provide deterministic concept creation under concurrency;
- backfill normalized names and types.

**Exit:** Same-name concepts of different types do not collide.

### PR 5 - Qualified canonical statements and evidence

- add qualifier columns and qualifier hash to `GraphRelationship`;
- add deterministic directed statement identity;
- add source-segment/local-key identity to `RelationshipDescription`;
- replace blind weight increments with evidence-derived aggregates;
- batch the canonical writes in one transaction.

**Exit:** One SQL record authoritatively represents each qualified statement.

### PR 6 - Typed materializers

- add statement vector index;
- project statement IDs into FalkorDB edges;
- make chunk, concept, and statement materializers independent consumers;
- add materialization status/retry boundary;
- add reconciliation commands for SQL/vector/Falkor parity.

**Exit:** Derived stores are rebuildable and idempotent.

### PR 7 - Query cutover to statements

- retrieve statement embeddings directly;
- hydrate qualifiers and evidence from SQL;
- use source segments for citations;
- enforce typed search and working-graph budgets;
- remove any need for proposition/assertion nodes.

**Exit:** End-to-end graph RAG answers use concepts, statements, evidence, and
source segments only.

### PR 8 - Retraction

- implement evidence-first source-segment tombstoning;
- recompute only affected statement aggregates;
- remove/rebuild affected vectors and graph edges;
- add shared-evidence and retry fixtures.

**Exit:** A changed/deleted source retracts only its affected materialization.

### PR 9 - Qualified reasoning

- add scope/modality/polarity/confidence-aware retrieval and ranking;
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
- plain-text condition/rule entities;
- contribution/dependency tables without retraction consumers.

## 15. Verification Matrix

### Identity and direction

- `(A, R, B)` and `(B, R, A)` are distinct for directed R;
- symmetric R normalizes endpoint order;
- statement IDs survive extraction reorder and retry;
- same triple with materially different qualifiers remains distinct;
- identical qualified statements from different segments share a statement ID.

### Extraction and provenance

- endpoint mentions derive from accepted statements only;
- changing contract version creates a new extraction cache entry;
- identical content in two documents reuses extraction but retains two segments;
- backfill and retry invoke no historical LLM calls;
- citations point to the correct segment and document.

### Concepts

- same normalized name and type resolves together;
- same normalized name with different types remains separate;
- ANN candidates are type-compatible and bounded;
- concurrent creation returns one canonical concept.

### Persistence and materialization

- one retry does not double weight/evidence;
- one chunk commits canonical writes atomically;
- SQL is authoritative when a vector/graph materializer fails;
- retry repairs only incomplete projections;
- reconciliation detects missing/stale vector and Falkor objects.

### Query and reasoning

- fresh ingestion is immediately queryable after publication;
- statement retrieval returns qualifiers and evidence;
- source paths never appear as canonical concepts;
- uncompiled text conditions never fire as facts;
- compiled rules require typed active antecedents and inactive blockers;
- every answer trace includes authoritative IDs and citations;
- working graph respects node, edge, hop, runtime, and memory limits.

### Retraction and scale

- retracting one of two evidence occurrences retains the statement;
- retracting final support removes affected derived objects;
- retraction does not scan unrelated statements;
- fixed-size append lookup/write counts remain bounded as collection size grows;
- global analytics failure never blocks ingestion or base query.

## 16. Observability

Record by collection and contract version:

- cache hit/miss and extractor calls;
- accepted/rejected statements;
- exact/alias/ANN/new concept resolutions;
- created/reused statements;
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
- extraction is statement-first and contract-versioned;
- source occurrence and content-level extraction cache are distinct;
- canonical concept identity is type-aware without type-prefixed display names;
- `GraphRelationship` is the sole authoritative qualified statement record;
- `RelationshipDescription`/evidence links statements to stable source segments;
- retries are deterministic and do not double-count evidence;
- `_ingest_graph_chunk` is a small typed orchestrator;
- chunk, concept, and statement vector indexes have distinct responsibilities;
- FalkorDB contains rebuildable concept/statement projections;
- source hierarchy is modeled outside the canonical concept graph;
- no proposition/assertion, subject/object, evidence, qualifier, or rule entities
  are required;
- natural-language qualifiers are not executed as formal rules;
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
- implementing formal logic from arbitrary text qualifiers;
- adding dependency/provenance tables without lifecycle readers;
- running global compilation or analytics in foreground ingestion;
- renaming every existing SQL table/class before behavior is correct.
