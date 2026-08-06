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
same fact, while making graph structure the primary executable input to reasoning.
Ordinary inference must be determined by typed relationships, compiled rule
bindings, active query state, and graph dependencies. Source text and an answer
LLM may seed, explain, or verbalize a trace; they must not decide whether a rule
fires.

The authoritative semantic fact is one directed, typed relationship:

```text
source entity -- relationship type --> target entity
```

Conditions, exceptions, scopes, and other meaningful connections that are
independently asserted facts use the same relationship abstraction and
entity-resolution path as every other extracted relationship. Their role as an
antecedent, blocker, or scope for a particular conclusion is preserved separately
inside a typed, rebuildable rule binding; it is never inferred from proximity,
free-form qualifier text, or special canonical entity classes.
`RelationshipDescription` records how the endpoints are related in a source
occurrence.

```text
source segment
  -> immutable relationship and explicit-rule extraction
  -> endpoint resolution
  -> canonical relationships + descriptions/evidence
  -> delta-triggered rule compilation
  -> derived indexes/projections

query known state + goal
  -> activate a bounded relationship/rule subgraph
  -> fire typed rules and blockers to a fixed point
  -> proof/counterproof trace
```

The design must remain bounded for incremental ingestion and query. Adding one
chunk may inspect only the chunk and fixed-size indexed candidate sets; querying
may activate only a bounded working graph. The first executable vertical slice
must prove that a graph delta can compile a rule and that query-time graph
activation can derive and retract a conclusion without an answer LLM deciding the
result.

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

`CONDITION` and `EXCEPTION` relationships may be queryable semantic facts, but a
relationship-type label alone does not bind a condition or exception to a
particular conclusion. Descriptions preserve what a source said; prose and
lexical overlap never decide rule activation. Executable applicability requires
an explicit typed rule binding with antecedents, blockers, conclusion, variables,
and scope.

### I8 - Bounded work

Foreground ingestion does not scan historical chunks, entities, relationships,
vectors, or projections. Query reasoning never loads the complete collection.

### I9 - Retraction is part of ingestion

Any provenance/dependency structure added during ingestion must have a tested
reader that retracts or invalidates the affected materialization.

### I10 - Graph-triggered execution

A canonical graph delta triggers bounded affected-rule compilation. At query time,
known-state relationships and goals activate indexed rules; typed graph structure,
not answer-generation prose, determines unification, firing, blocking, conflict,
and derived conclusions. Retrieval/path adjacency alone never establishes a new
fact.

### I11 - Applicability remains bound

Polarity, modality, typed scope, temporal validity, and rule-clause membership
must remain attached to the exact relationship occurrence or rule that supplied
them. Two occurrences of the same `(source, type, target)` with materially
different executable context must not be merged into one unconditional fact.

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

`payload_json` preserves the extractor response in a relationship-and-explicit-rule
schema. It does not store document/path provenance.

During migration, existing `RawChunkExtraction` may remain the physical table
while new columns and a segment link are introduced. The responsibility split is
more important than an immediate table rename.

### 5.3 Extraction contract

Use one relationship shape for facts and rule clauses, plus an explicit grouping
shape for executable rules:

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
  polarity
  modality
  typed_scope
  temporal_validity
  relationship_type_properties

ExtractedRule
  local_key
  antecedent_expression: RuleExpression
  blocker_expression: RuleExpression
  conclusion: RelationshipPattern
  scope_expression: RuleExpression
  priority
  confidence

RelationshipPattern
  source: EndpointTerm
  relationship_type
  target: EndpointTerm
  polarity
  modality

EndpointTerm
  mention: EndpointMention | null
  variable_name: string | null
  type_constraint: string | null

EndpointMention
  name
  type_hint
  description
```

`ExtractionResult` contains `relationships` and `rules`. Conditions, exceptions,
and scopes that are themselves semantic facts are emitted as ordinary
relationships and use the normal endpoint-resolution path. When they govern a
specific conclusion, their applicability role is also preserved in an
`ExtractedRule`; an independent `CONDITION` or `EXCEPTION` edge is never assumed
to qualify a nearby relationship.

Rule clauses reuse the relationship source/type/target vocabulary rather than
introducing a parallel proposition fact model. A conclusion pattern is not
published as an asserted canonical relationship merely because it was extracted
from a conditional rule. Only independently asserted ground facts enter the
active canonical fact graph; rule conclusions become query-local derived
relationships after successful activation.

A temporary `entities` property may derive unique endpoint mentions from every
accepted relationship and grounded rule term for compatibility, but it is never
serialized as an independent source of truth. The extractor assigns `local_key`
from canonical serialized content. Response array order is not identity. Rules
with unresolved prose clauses, unbound variables, or ambiguous clause grouping
remain retrieval evidence and are not executable.

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
  polarity
  modality
  semantic_context_hash
  confidence
  aggregate_weight
```

Canonical identity:

```text
UUID(
  collection_id,
  source entity ID,
  canonical relationship type ID,
  target entity ID,
  polarity,
  modality,
  semantic context hash
)
```

`semantic_context_hash` covers only typed, executable scope and temporal context;
it never hashes free-form description prose. Database uniqueness enforces the
same tuple. Opposite directions remain distinct. For a symmetric relationship
type, normalize endpoint order before computing identity.

Every independently asserted connection uses this identity, including
`CONDITION`, `EXCEPTION`, and `SCOPE` facts. Materially different polarity,
modality, scope, or temporal validity remains distinct instead of being aggregated
into an unconditional edge. A conditional conclusion template belongs to a
compiled rule and does not become an active `GraphRelationship` until query-time
activation derives it. Relationship descriptions preserve source wording but do
not supply missing executable context.

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
  polarity
  modality
  scope_json
  valid_time_json
  support_kind
  metadata_json
```

Identity:

```text
UUID(relationship_id, source_segment_id, raw_relationship_local_key)
```

One relationship may have many evidence rows. Aggregate weight and consensus fields
are derived from active evidence rows, never incremented blindly on retry.
`support_kind` distinguishes assertion, support, attack, mention, and rule source.
Conditional, hypothetical, scoped, temporal, or negative evidence never silently
increases support for an unconditional positive relationship.

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
reasoning_operators_json
property_votes_json
observation_count
```

Raw labels map to a canonical relationship type through a versioned mapping.
Property consensus never changes historical extraction and never silently flips
existing edge direction. Observed properties are descriptive until an explicitly
configured, versioned reasoning operator enables executable behavior. A label
such as `CAUSES` or a high transitivity vote never licenses inference by itself.

## 6. Derived Storage Policy

### 6.1 PostgreSQL

PostgreSQL owns:

- source segments;
- immutable extraction payloads;
- endpoint-to-entity resolution records;
- canonical entities and relationship types;
- canonical relationships;
- relationship evidence;
- typed raw-rule bindings and versioned compiled rules;
- rule reverse dependencies and compiler publication status;
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
  polarity,
  modality,
  semantic_context_hash,
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
delta = canonical_writer.upsert(segment, resolved, extraction.rules)
materializers.apply(delta)
reasoning_compiler.enqueue(delta)
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
services/graph/reasoning/compiler.py
services/graph/reasoning/dependencies.py
services/graph/reasoning/executor.py
```

Typed boundaries:

```text
ExtractionContract
ExtractedRelationship
ResolvedEndpoint
ResolvedRelationship
ResolvedRule
CanonicalChunkDelta
MaterializationResult
CompilerMaterializationResult
```

Rules:

- extraction code performs no canonical writes;
- resolution code performs no FalkorDB writes;
- canonical writer performs no vector or graph writes;
- materializers consume committed canonical IDs;
- compiler jobs consume committed delta IDs and never mutable graph payloads;
- publication reports base and compiler versions independently;
- formal reasoning is available only from a compatible published compiler version;
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

`delta_manifest_json` lists relationship, evidence, raw-rule, and compiler-job IDs
created or reused for the segment. Do not add a general dependency graph beyond
the indexed reverse dependencies that retraction and compilation concretely read.

Introduce collection graph versions only when a consumer requires snapshot
identity:

- analytics snapshot;
- compiled rule set;
- cached reasoning trace;
- external version-pinned query.

A version references a completed set of segment materializations. It does not
duplicate canonical objects.

### 8.1 Delta-triggered compiler lifecycle

1. The canonical writer commits a `CanonicalChunkDelta` containing changed
   relationship, evidence, and raw-rule local keys.
2. The publisher transactionally enqueues `(collection_id, delta_id,
   compiler_version)`.
3. The compiler resolves typed rule terms, calculates an indexed bounded affected
   closure, writes rules and reverse dependencies privately, and atomically
   publishes the compiler materialization.
4. Over-budget closures continue asynchronously; they never cause historical LLM
   extraction.
5. Retry is idempotent by `(delta_id, compiler_version)`.
6. Retraction invalidates or recompiles only rules reached through reverse
   indexes.
7. Queries pin and report canonical and compiler versions. When no compatible
   compiler version exists they may explicitly fall back to non-formal retrieval,
   never silently claim formal reasoning.

Required reverse indexes include relationship/predicate-to-antecedent rules,
relationship/predicate-to-blocker rules, conclusion-pattern-to-rules,
rule-to-source segments, and rule-to-proof dependencies. Compilation is a real
reader of these dependencies; no write-only general dependency graph is added.

## 9. Retraction

Retraction is evidence-first:

1. Tombstone the source segment.
2. Mark/remove its active relationship-evidence rows.
3. Recompute affected relationship aggregate fields from remaining evidence.
4. Delete an unsupported relationship when no active evidence or explicit derived
   support remains.
5. Remove/rebuild its relationship vector and FalkorDB edge.
6. Follow reverse indexes to invalidate or recompile dependent rules, cached
   proofs, and snapshots under their recorded compiler/materializer versions.
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

Build a bounded query plan that separates candidate retrieval from execution:

```text
question
  -> intent + goals + known facts + constraints + typed scope
  -> entity and relationship-vector seeds       # candidate selection only
  -> rule-index backward/forward expansion
  -> typed bounded neighborhoods
  -> evidence hydration
  -> bounded working graph activation
  -> proof/counterproof plus citations
  -> optional answer verbalization
```

Query compilation must distinguish an asserted query fact from a desired goal,
constraint, hypothetical assumption, negation, and mere lexical mention. ANN
similarity, adjacency, and path existence may seed the working graph but never
activate a fact or fire a rule.

Query paths use authoritative IDs:

- entities from `GraphEntity`;
- active facts from `GraphRelationship`;
- compiled rules and reverse indexes from the compatible compiler publication;
- condition/exception/general semantics from typed rule roles and
  `GraphRelationshipType`;
- evidence/citations from `RelationshipDescription`/`RelationshipEvidence`;
- source text from `SourceSegment`;
- traversal from FalkorDB edges carrying `relationship_id`.

There is no `ASSERTION` versus `PROPOSITION` entity vocabulary. Canonical facts
remain relationship records; rule nodes and relationship-pattern references exist
only in rebuildable reasoning projections.

Hard limits:

- entity and relationship ANN top-k;
- rule candidates per predicate/goal;
- graph hops, nodes, edges, and rule bindings;
- evidence rows per relationship;
- fixed-point iterations;
- runtime and memory;
- optional community/projection expansion count.

## 11. Reasoning

### 11.1 Retrieval is not inference

Before a compatible compiled rule set exists, the system may retrieve typed
relationships, traverse bounded paths, rank evidence, and explain what the graph
contains. It must label this result non-formal. It may not convert adjacency,
lexical overlap, a relationship label, or an LLM interpretation into a derived
fact.

Conditions and exceptions that are ordinary domain facts remain directly
queryable. Their `CONDITION` or `EXCEPTION` type does not bind them to a conclusion
or activate a blocker; only an explicit compiled rule supplies that role.

### 11.2 Typed rule compiler

Compile each valid `ExtractedRule` deterministically into:

```text
CompiledRule
  id
  collection_id
  antecedent_expression_json
  blocker_expression_json
  conclusion_pattern_json
  scope_expression_json
  priority
  confidence
  compiler_version
  source_segment_ids
  source_relationship_ids
  active
```

Expressions contain typed relationship patterns, variables, and `atom`, `all`,
`any`, and `not` operators. `conclusion_pattern_json` is variable-bearing; it does
not require a conclusion relationship to pre-exist. The compiler rejects unbound
variables, unresolved endpoints or relationship types, ambiguous prose clauses,
and unsupported operators.

Rule indexes support antecedent, blocker, and conclusion lookup plus proof and
retraction dependencies. A derived rule graph (`ANTECEDENT_OF`, `BLOCKS`,
`CONCLUDES`) may contain rule and relationship-pattern reference nodes, but those
nodes are rebuildable executable scaffolding, not `GraphEntity` domain objects or
a second authoritative fact model.

### 11.3 Query-time graph activation

1. Compile the question into goals, known-state facts, assumptions, constraints,
   and typed scope.
2. Resolve and activate only facts explicitly asserted by the canonical graph or
   query known state; retrieval similarity alone does not activate them.
3. Expand backward from goals and forward from active facts through rule indexes.
4. Unify typed variables against active relationships and evaluate `all`, `any`,
   `not`, polarity, modality, temporal validity, and scope.
5. Fire a rule only when its antecedent expression is satisfied and no applicable
   higher-priority blocker is active.
6. Instantiate a query-local `DerivedRelationship` with its rule ID, binding,
   premises, evidence, confidence calculation, and compiler version.
7. Preserve competing support and attack branches. Apply declared
   priority/specificity rules; otherwise report conflict rather than inventing a
   winner.
8. Memoize `(rule_id, binding)` and continue until a fixed point, proved/disproved
   goal, unresolved conflict, or resource limit.
9. Return `proved`, `disproved`, `supported_hypothesis`, `conflicted`, `unknown`,
   or `resource_limited` with relationship, rule, evidence, source-segment, and
   version IDs.

Derived relationships are query-local by default. Persisting a derived conclusion
requires a separate, versioned materialization policy and truth-maintenance
contract; it must never be confused with independently asserted evidence.

### 11.4 Graph-delta activation

Committed graph deltas trigger compilation and invalidation, not automatic truth
from arbitrary topology. A compiler consumes only the bounded affected closure,
publishes a versioned rule set, and records reverse dependencies. Retraction of a
premise or rule source invalidates dependent compiled rules and cached proofs
without scanning unrelated relationships. Global motif discovery may propose rule
candidates asynchronously, but candidates remain non-executable until validated
under a versioned typed compiler contract.

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

**Exit:** Direction remains the default, and only relationship types explicitly
marked symmetric normalize endpoint order.

### PR 2 - Relationship and typed-rule extraction contract

- introduce `EndpointMention`, `ExtractedRelationship`, `RelationshipPattern`,
  and `ExtractedRule`;
- preserve explicit antecedent, blocker, conclusion, variable, and scope binding;
- emit independently asserted conditions, exceptions, and scopes as ordinary
  relationships without inferring rule attachment from their labels;
- derive endpoint mentions from accepted relationships and grounded rule terms;
- version the extraction contract and add order-independent local keys.

**Exit:** One immutable contract preserves atomic relationships and explicit rule
structure without a proposition fact model or executable prose.

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

### PR 5 - Deterministic qualified relationships and evidence

- add deterministic directed relationship identity including typed executable
  context;
- add source-segment/local-key identity and epistemic fields to evidence;
- prevent conditional, negative, or scoped evidence from increasing unconditional
  support;
- replace blind weight increments with evidence-derived aggregates;
- batch canonical writes in one transaction.

**Exit:** One `GraphRelationship` authoritatively represents each compatible typed
fact without merging materially different activation contexts.

### PR 6 - Minimal graph-triggered reasoning vertical slice

- extract one typed `all` rule with variables and one blocker;
- compile it from `CanonicalChunkDelta` through an idempotent queued materializer;
- activate it with bounded forward/backward rule-index lookup;
- instantiate a query-local derived conclusion with proof and compiler IDs;
- retract its source evidence and invalidate the rule and cached proof.

**Exit:** One conclusion is reproducibly derived from graph structure without an
answer LLM choosing whether the rule fires.

### PR 7 - Typed vector/Falkor materializers

- add relationship vector index;
- project relationship IDs into FalkorDB edges;
- publish the minimal rule projection separately from canonical facts;
- make materializers independent, idempotent consumers;
- add status, retry, and SQL/vector/Falkor/compiler reconciliation.

**Exit:** Fact, retrieval, graph, and reasoning projections are rebuildable and
versioned.

### PR 8 - Query cutover and explicit retrieval fallback

- retrieve relationship embeddings directly for candidate selection;
- compile goals, known state, constraints, and scope;
- hydrate relationship types, evidence, rules, and source citations;
- enforce retrieval and working-graph budgets;
- label answers non-formal when no compatible compiler publication exists.

**Exit:** Query cleanly separates GraphRAG retrieval from graph-executed inference.

### PR 9 - Retraction across facts, rules, and projections

- implement evidence-first source-segment tombstoning;
- recompute only affected relationship aggregates;
- follow reverse dependencies through rules and cached proofs;
- remove or rebuild affected vector, Falkor, and compiler objects;
- add shared-evidence, retry, and proof-invalidation fixtures.

**Exit:** A changed or deleted source retracts only its dependent materialization.

### PR 10 - Full typed expression and conflict runtime

- implement `atom`, `all`, `any`, and `not`, variable unification, and conclusion
  templates;
- add modality, scope, temporal, priority, specificity, support, and attack
  semantics;
- add bounded forward/backward fixed-point execution;
- persist proof dependencies and compiler version;
- port behavior-level reasoning fixtures without proposition entities.

**Exit:** Formal conclusions, conflicts, hypotheses, and resource-limited results
are reproducible and auditable.

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
- condition and exception facts use the same identity rules as all other
  relationships;
- materially different polarity, modality, typed scope, or temporal validity does
  not collapse into one unconditional relationship.

### Extraction and provenance

- endpoint mentions derive from accepted relationships and grounded rule terms;
- “A causes B if C except D” preserves C as antecedent and D as blocker of that
  exact conclusion pattern;
- a conditional conclusion is not published as an asserted active fact;
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

- fresh canonical facts are immediately queryable after base publication;
- relationship retrieval returns its type, executable context, descriptions, and
  evidence;
- query compilation separates goals, known facts, assumptions, constraints, and
  mentions;
- source paths never appear as canonical semantic entities;
- relationship descriptions, ANN similarity, adjacency, causal labels, and
  uncompiled `CONDITION` edges never fire a rule;
- a variable-bearing rule derives a conclusion that did not pre-exist;
- forward and backward proof produce the same valid dependency trace;
- multi-antecedent conjunction, disjunction, negation, and exception blocking work;
- conditional/scoped evidence is not treated as an unconditional fact;
- positive and negative independent support yields conflict rather than a silent
  merge;
- priority and specificity resolve only declared conflicts;
- compiler delta retry is idempotent and over-budget work continues asynchronously;
- retracting a premise invalidates only its dependent rule and proof;
- cyclic rules terminate at a fixed point and budgets return `resource_limited`;
- every formal trace includes relationship, premise, rule, evidence,
  source-segment, canonical-version, and compiler-version IDs;
- working graph respects node, edge, rule-binding, hop, iteration, runtime, and
  memory limits.

Port behavior rather than schema from the prototype fixtures for goal unification,
backward proof, conditional hypothesis, exception blocking, compiled-property
path policy, and rule-projection separation.

### Retraction and scale

- retracting one of two evidence occurrences retains the relationship;
- retracting final support removes affected derived objects;
- retraction does not scan unrelated relationships;
- fixed-size append lookup/write counts remain bounded as collection size grows;
- global analytics failure never blocks ingestion or base query.

## 16. Observability

Record by collection and contract version:

- cache hit/miss and extractor calls;
- accepted/rejected relationships and explicit rules by type/operator;
- exact/alias/ANN/new entity resolutions;
- created/reused relationships;
- evidence occurrences;
- canonical transaction duration;
- vector/Falkor materialization status;
- compiler delta queue, affected-closure, publication, and staleness status;
- active rule candidates, bindings, firings, blockers, and derived relationships;
- retry and reconciliation counts;
- retraction affected-object counts;
- query seed/working-graph sizes;
- reasoning result and stop reason;
- canonical, compiler, and analytics version/staleness.

Log IDs and counts, not full source or extracted text by default.

## 17. Definition of Done

- `main`'s reverse-edge merge bug remains fixed (landed in PR #8);
- extraction is relationship-first, explicit-rule aware, and contract-versioned;
- source occurrence and content-level extraction cache are distinct;
- canonical entity identity is type-aware without type-prefixed display names;
- `GraphRelationship` is the sole authoritative atomic-fact record;
- materially different polarity, modality, scope, and temporal validity do not
  collapse into an unconditional fact;
- explicit antecedent, blocker, conclusion, and variable binding survives
  extraction and deterministic compilation;
- `CONDITION`, `EXCEPTION`, and general relationship facts share one ingestion,
  resolution, persistence, embedding, and retrieval path;
- `RelationshipDescription`/evidence links relationships and rule sources to
  stable source segments;
- retries are deterministic and do not double-count evidence or compilation;
- `_ingest_graph_chunk` is a small typed orchestrator;
- graph deltas trigger bounded, idempotent compiler materialization and indexed
  invalidation;
- at least the minimal typed-rule vertical slice is enabled end to end: extract,
  compile, activate, derive, prove, and retract;
- an answer LLM does not decide rule firing, blocking, proof, or conflict status;
- GraphRAG fallback is labeled non-formal and cannot satisfy reasoning acceptance;
- chunk, entity, relationship, rule, and proof projections have distinct,
  rebuildable responsibilities and published versions;
- FalkorDB contains rebuildable entity/relationship and named reasoning
  projections;
- source hierarchy is modeled outside the canonical entity graph;
- no proposition/assertion, subject/object, evidence, condition, exception, or
  rule objects are required as canonical `GraphEntity` domain records;
- relationship descriptions are not executed as formal rules;
- typed rules use auditable expressions, reverse dependencies, and compiler
  versions;
- retraction across facts, rules, proofs, and derived projections is bounded;
- query and reasoning operate on bounded working graphs and fixed-point budgets;
- every formal result reports authoritative IDs and canonical/compiler versions;
- analytics runs on named, versioned projections;
- historical migration and backfill require zero LLM re-extraction.

## 18. Non-Goals

- merging the prototype feature branch;
- preserving prototype internal schemas merely for code reuse;
- re-running LLM extraction over historical chunks;
- graphifying every domain object;
- implementing formal logic from arbitrary relationship descriptions;
- treating retrieval similarity, adjacency, path existence, relationship labels,
  or analytics metrics as inference;
- allowing an answer LLM to choose whether a typed rule fires;
- materializing every query-local derived relationship as a canonical asserted
  fact;
- adding dependency/provenance tables without lifecycle readers;
- running global compilation or analytics in foreground ingestion;
- renaming every existing SQL table/class before behavior is correct.

Typed rules emitted under a versioned extraction contract, deterministic rule
compilation, bounded graph activation, and auditable proof/retraction are
explicitly in scope.
