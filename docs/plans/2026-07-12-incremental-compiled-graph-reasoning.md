# Incremental Compiled Graph Reasoning Plan

> **Status:** Design - awaiting implementation
>
> **Date:** 2026-07-12
>
> **Scope:** `custom_graph_rag` ingestion, knowledge compilation, structural
> analytics, and graph-native query reasoning for large evolving collections

## 1. Objective

Build a graph that is the primary executable knowledge representation, not an
index that requires returning to source text for ordinary reasoning.

The system must support collections with millions of nodes and documents or
chunks added at any time. Foreground ingestion work must be bounded by the new
input and configured local lookup limits; it must never scan, rewrite, re-embed,
or re-extract the existing collection.

```text
new chunk
  -> immutable raw extraction segment
  -> bounded indexed attachment
  -> immediately queryable overlay
  -> asynchronous local compilation
  -> asynchronous global analytics snapshot
```

Source locations remain terminal provenance for audit and citation. Normal query
reasoning executes over compiled graph structures: propositions, rules,
mechanisms, constraints, alternatives, contradictions, and derived structural
properties.

## 2. Non-Negotiable Invariants

### I1 - Incremental foreground complexity

Adding a chunk costs:

```text
O(size(new_chunk) + bounded_matches + emitted_graph_delta)
```

Strict `O(1)` is impossible with respect to input size, but ingestion must be
independent of existing collection size. No foreground operation may iterate over
all existing chunks, entities, predicates, edges, communities, or embeddings.

### I2 - Immutable extraction

Every chunk is content-addressed. Its raw LLM extraction is immutable and stored
under an extraction contract version. Unchanged content causes no LLM call,
embedding call, resolution pass, or graph write.

### I3 - No historical LLM rewrites

Adding a chunk, discovering a new predicate family, changing an ontology, or
merging entities must not invoke an LLM on previous chunks. Historical raw
extractions are interpreted through versioned mappings and overlays.

### I4 - Bounded resolution

Entity and predicate resolution use indexed exact lookup followed by bounded ANN
or local structural lookup. Resolution has fixed top-k, timeout, and candidate
limits and never falls back to a collection scan.

### I5 - Directed semantics

`(A, R, B)` and `(B, R, A)` are distinct unless the canonical predicate is
explicitly declared symmetric. Direction must survive ingestion, compilation,
analytics, and query.

### I6 - Layer separation

Raw observations, canonical knowledge, compiled reasoning structures, and
analytics snapshots are independently versioned layers. Publishing a new derived
layer never mutates historical extraction segments.

### I7 - Auditable reasoning

Every answer is produced from an activated subgraph. Derived conclusions retain
the rules, premises, exceptions, conflicts, and confidence calculation that led
to them.

## 3. Layered Graph Model

### 3.1 Immutable observation layer

One chunk extraction emits chunk-local objects:

```text
Chunk
Mention
RawEntity
RawPredicate
RawProposition
RawRule
RawCondition
RawConstraint
RawEvidenceLink
```

IDs are deterministic from collection, chunk hash, extraction contract version,
and local object index. The layer preserves exactly what the extractor emitted.

The LLM should emit all available structure in one call per new chunk:

- entities and entity types
- propositions and predicate arguments
- direction and polarity
- conditions and exceptions
- modality, confidence, time, population, and scope
- causal/mechanistic steps
- rules with antecedents and consequents
- goals, constraints, alternatives, evaluations, and outcomes
- support, contradiction, and provenance links

Missing fields stay unknown. The system must not invent executable semantics from
a generic relationship label.

### 3.2 Canonical overlay

Canonical objects are stable representatives referenced by mappings:

```text
RawEntity       -[MAPS_TO]-> CanonicalEntity
RawPredicate    -[MAPS_TO]-> CanonicalPredicate
RawProposition  -[COMPILES_TO]-> Proposition
CanonicalEntity -[SAME_AS/ALIAS_OF]-> CanonicalEntity
```

Entity merges use redirects or union-find-style representatives. Existing raw
edges are not eagerly rewired. Reads resolve the current representative through
a bounded path or flattened lookup table maintained asynchronously.

For code, identity must include repository, path/module, language, symbol kind,
enclosing scope, qualified name, and signature where available. A short name is
display metadata, never collection-wide identity.

### 3.3 Proposition and rule layer

A proposition is a reified typed hyperedge:

```text
Proposition P
  -[PREDICATE]-> Increases
  -[SUBJECT]-> Pranayama A
  -[OBJECT]-> Activation
  -[CONDITION]-> Rapid breathing
  -[POLARITY]-> Positive
  -[MODALITY]-> Probable
  -[SCOPE]-> Population/Time/Context
```

A rule is executable graph structure:

```text
Condition C1 -[ANTECEDENT_OF]-> Rule R
Condition C2 -[ANTECEDENT_OF]-> Rule R
Exception E  -[BLOCKS]-> Rule R
Rule R       -[CONCLUDES]-> Proposition P
Rule R       -[PRIORITY]-> PriorityValue
```

Rules may be extracted explicitly or compiled from recurring, high-confidence
motifs. Arbitrary edges are not rules.

### 3.4 Evidence and epistemic layer

Claims retain graph-native epistemic structure:

```text
Evidence -[SUPPORTS]-> Proposition
Evidence -[ATTACKS]-> Proposition
Proposition -[CONTRADICTS]-> Proposition
Proposition -[REFINES]-> Proposition
Proposition -[EXCEPTION_TO]-> Rule
Evidence -[APPLIES_TO]-> Scope
```

Evidence independence, authority, observation type, applicability, and extraction
confidence are properties or connected nodes. Repetition from one document does
not count as independent evidence.

### 3.5 Containment and provenance layer

Repository/file/scope and book/part/chapter/section hierarchies remain available
for scope, provenance, and reasoning about containment. They are excluded from
ordinary semantic centrality unless a named projection explicitly includes them.

## 4. Dynamic Domain and Predicate Model

Collections do not require a complete relationship taxonomy before ingestion.

### 4.1 Domain bootstrap

For a new document or domain:

1. Sample structurally diverse chunks using headings, file types, or AST scopes.
2. Infer an initial entity, predicate, proposition, and rule schema.
3. Freeze a versioned extraction contract.
4. Extract subsequent chunks against that contract.

Bootstrap sampling must be stratified; a book introduction or repository root
alone is not representative.

### 4.2 Unknown predicates

Unknown raw predicate types are accepted immediately. A background compiler maps
them using:

- normalized label and aliases
- argument domain/range signatures
- direction and reciprocal behavior
- neighboring predicate motifs
- description embeddings
- observed conditions, polarity, and temporal behavior

The compiler infers candidate properties such as directed, symmetric, transitive,
causal, temporal, compositional, or evaluative. Low-confidence mappings remain
raw and non-executable.

Mapping changes add a new predicate-map version. They do not rewrite or re-run
LLM extraction on historical chunks.

## 5. Incremental Ingestion Pipeline

### 5.1 Idempotency and change detection

- Normalize content and compute a stable chunk hash.
- Look up `(collection_id, chunk_hash, extraction_contract_version)` by index.
- Return the existing segment immediately when present.
- For changed documents, diff chunk hashes and process only additions/removals.
- Keep a document manifest mapping document version to ordered chunk hashes.

### 5.2 New chunk path

1. Sanitize and hash the chunk.
2. Resolve the versioned extraction contract.
3. Perform one structured LLM extraction if the raw segment is absent.
4. Batch embeddings for the chunk and selected raw semantic objects.
5. Bulk insert the immutable observation segment.
6. Run bounded exact and ANN candidate lookups.
7. Add canonical mapping and aggregate-evidence deltas.
8. Publish the overlay as immediately queryable.
9. Enqueue affected-set compilation and analytics work.

### 5.3 Deletion path

A contribution index maps each chunk to every raw object, canonical aggregate,
compiled rule, and derived metric it influences. Deleting a chunk:

- tombstones its immutable segment
- retracts only its evidence contributions
- decrements aggregate counters atomically
- invalidates only objects in its dependency closure
- schedules garbage collection for unsupported derived objects

Deletion never scans the collection.

### 5.4 Provider and persistence efficiency

- Batch embeddings while respecting embedding-profile concurrency.
- Enforce LLM concurrency from the LLM profile.
- Use one transaction and bulk upserts per chunk batch, not per object.
- Bulk FalkorDB and vector writes.
- Do not embed scaffolding edges by default.
- Do not perform per-edge FalkorDB read-before-merge operations.
- Cache immutable provider/profile resolution for the job lifetime.

## 6. Dependency and Delta Indexes

Incrementality depends on explicit reverse dependencies:

```text
chunk -> raw objects
raw entity -> canonical representative
raw predicate -> canonical predicate
raw proposition -> compiled proposition/rules
canonical object -> projections
projection edge/node -> communities/metrics
rule -> premises/conclusions
derived conclusion -> proof dependencies
```

Each derived artifact stores its dependency IDs and version. A new delta computes
an affected closure through indexed reverse edges with configurable work limits.
If the closure exceeds the foreground limit, ingestion completes and continuation
runs asynchronously.

## 7. Knowledge Compilation

Compilation converts observations into structures that query can execute.

### 7.1 Local compilation

Triggered by graph deltas and bounded to the affected closure:

- resolve aliases and canonical representatives
- canonicalize predicate mappings
- aggregate equivalent propositions
- attach support and contradiction
- compile explicit conditions and exceptions into rules
- detect recurring local motifs eligible for rule candidates
- update rule antecedent and conclusion indexes
- update transitive reductions or closures only for declared predicates

No LLM is required to revisit old chunks. Optional LLM review may inspect only a
new ambiguous object and its bounded neighborhood, and its result is a new mapping
overlay rather than a historical rewrite.

### 7.2 Global compilation

Runs asynchronously on versioned snapshots:

- ontology/type hierarchy consolidation
- large predicate-family clustering
- abstraction and hierarchical summaries
- contradiction and argument-network consolidation
- reusable reasoning motifs
- landmark nodes and community summaries

Global compilation may be `O(n)` but is never part of foreground ingestion and
does not block immediate querying of the new overlay.

## 8. Structural Analytics

Analytics operates on named projections, never the raw mixed storage graph.

Every projection records included types, excluded scaffolding, direction policy,
parallel-edge aggregation, self-loop policy, confidence threshold, and weight
interpretation.

Initial projections:

- `semantic_all_directed`
- `causal_mechanism_directed`
- `rule_dependency_directed`
- `semantic_affinity_undirected`
- `code_dependency_directed`
- `book_concept_directed`
- `book_practice_decision`
- `role_similarity_undirected`

Post-snapshot metrics include:

- typed in/out degree and components
- SCCs and condensation DAGs
- PageRank and harmonic centrality
- k-core, triangles, and clustering coefficient
- Leiden communities with connectivity checks
- articulation points, graph bridges, and bounded approximate betweenness
- sparse normalized-Laplacian and Fiedler diagnostics
- bounded clique, biclique, and selected motif analysis

Global results are versioned and may be stale while a new delta is pending. Cheap
degree/evidence counters update incrementally. Expensive global metrics never
block ingestion.

## 9. Million-Node Query Reasoning

Reasoning never loads or traverses the complete collection.

### 9.1 Query compilation

Convert the question into:

- intent operators
- seed entities and known state
- desired goals or target predicates
- constraints and exceptions
- relevant predicate families and projection names

Core operators:

```text
evaluate(subject, rubric, severity)
choose(alternatives, state, goal, constraints)
explain(cause, outcome, scope)
redesign(subgraph, goals, constraints)
prove_or_disprove(proposition, scope)
```

### 9.2 Working graph construction

Build a bounded query-local graph from:

- exact/ANN seed resolution
- rule antecedent and conclusion indexes
- seed communities and landmarks
- typed bounded neighborhoods
- relevant contradiction/support subgraphs
- applicable conditions, scopes, and exceptions

Limits include maximum nodes, edges, hops, communities, relation families,
runtime, and memory.

### 9.3 Reasoning algorithm

1. Activate the query's known state and constraints.
2. Expand forward from active facts and backward from goals.
3. Meet in the middle using indexed rule premises and conclusions.
4. Fire a rule when required antecedents are active and no stronger exception
   blocks it.
5. Materialize conclusions in the query-local activation graph.
6. Maintain support and attack paths for competing conclusions.
7. Rank the frontier by semantic fit, specificity, confidence, applicability,
   structural proximity, and contradiction risk.
8. Widen to another community or predicate family only when the current graph is
   insufficient.
9. Stop at a fixed point, proved/disproved goal, bounded resource limit, or
   unresolved conflict.
10. Return the activated proof, counterexample, decision, or transformation
    subgraph as the reasoning trace.

Communities and centrality prioritize exploration; they never establish truth.

### 9.4 Intent-specific execution

#### Evaluation

Combine explicit goals, constraints, quality attributes, consequences, tests,
failure modes, and structural metrics. `good`, `bad`, and `ugly` are rubric-based
conclusion classes backed by argument paths.

#### Decision and comparison

Traverse:

```text
candidate -> action -> mechanism -> effect -> goal
candidate -> precondition/contraindication/exception
```

Compare candidates under the same state and goals, resolving competing rules by
specificity, scope, priority, confidence, and attack strength.

#### Redesign

Apply bounded graph rewrite operators such as split responsibility, invert
dependency, introduce interface, move ownership, or replace a pattern. Evaluate
candidate graphs against hard constraints, soft goals, and structural metrics.

#### Health and safety

Use scoped intervention-mechanism-outcome rules plus contraindications and
epistemic support/attack structure. Evidence sufficiency is itself a graph result
based on applicability, independence, unresolved contradiction, and risk.

## 10. Persistence and Versioning

Add durable models for:

- immutable chunk extraction segments
- document-to-chunk manifests
- raw-to-canonical entity mappings
- raw-to-canonical predicate mappings
- canonical propositions and executable rules
- proposition support/attack and rule exceptions
- chunk contribution and reverse-dependency indexes
- graph overlay versions
- named projection specifications
- analytics snapshots, node metrics, communities, and component metrics
- cached reasoning traces keyed by query plan and graph version

Derived writes are idempotent. A new overlay is built privately and atomically
published. Readers use the newest compatible published versions and can tolerate
temporarily stale analytics.

## 11. Implementation Boundaries

Keep current public service entry points compatible while separating concerns:

```text
services/graph/ingestion/observation_writer.py
services/graph/resolution/entities.py
services/graph/resolution/predicates.py
services/graph/compilation/propositions.py
services/graph/compilation/rules.py
services/graph/compilation/dependencies.py
services/graph/reasoning/planner.py
services/graph/reasoning/working_graph.py
services/graph/reasoning/executor.py
services/graph/analytics/projections.py
services/graph/analytics/metrics.py
services/graph/analytics/persistence.py
```

`analytics.py` remains a facade if changing it into a package creates import
churn. Workers stay thin and accept IDs/versions rather than graph payloads.

## 12. Delivery Sequence

### Milestone 1 - Complexity contract and immutable segments

- instrument current ingestion by operation count and collection size
- add stable chunk/extraction contract keys
- skip all work for unchanged chunks
- add document manifests and chunk contribution indexes
- add regression tests proving no collection scan occurs on append

### Milestone 2 - Bounded attachment

- batch embeddings and persistence
- add fixed-budget exact/ANN entity and predicate resolution
- preserve directed relationships
- remove duplicate semantic paths and per-edge read/commit loops
- make new overlays immediately queryable

### Milestone 3 - Canonical and epistemic overlays

- add redirect-based entity representatives
- add versioned predicate mappings
- add structured propositions, conditions, exceptions, and scopes
- aggregate evidence without rewriting raw observations
- add support/attack/contradiction structures

### Milestone 4 - Incremental compilation

- add reverse-dependency indexes
- compile rules and proposition aggregates for affected closures
- add bounded local continuation jobs
- add overlay publication and stale-version handling

### Milestone 5 - Named projections and analytics

- implement projection contracts
- add foundational, community, bridge, and spectral metrics
- persist versioned snapshots
- add incremental cheap counters and asynchronous global recomputation

### Milestone 6 - Graph reasoning runtime

- compile queries into operator plans
- construct bounded working graphs
- implement bidirectional rule activation and conflict resolution
- return auditable activation traces
- add adaptive widening and strict resource budgets

### Milestone 7 - Evaluation and graph rewrites

- implement rubric-based evaluation
- implement constrained candidate comparison
- add bounded redesign transformations and before/after metric comparison
- add high-risk applicability and sufficiency policies

## 13. Tests and Benchmarks

### Incrementality invariants

- appending one fixed-size chunk performs the same bounded number of existing-data
  lookups at 1,000 and 1,000,000 existing nodes
- unchanged chunks cause zero provider calls and zero semantic writes
- adding a predicate mapping causes zero historical LLM calls
- entity merges add redirects and do not rewrite historical observation edges
- deletion touches only objects reachable from the chunk contribution index
- ingestion completes even when global analytics is stale or unavailable

### Reasoning fixtures

- forward and backward chaining
- conjunction, exceptions, and rule priorities
- contradictory conclusions with independent support
- causal mechanism paths
- decision among alternatives under constraints
- graph rewrite satisfying one goal while violating another
- fixed-point termination and resource-limit termination
- adaptive widening across a community boundary

### Structural fixtures

- directed cycles and SCCs
- disconnected and star graphs
- barbell graph with known articulation/betweenness
- known Leiden community structure
- complete graph and role-similarity clique distinction
- weighted path with explicit strength-to-distance transformation
- provenance expansion that leaves semantic metrics unchanged

### Scale benchmarks

Measure foreground append latency, provider calls, SQL statements, graph writes,
ANN candidates, affected-closure size, background compilation latency, analytics
memory/runtime, working-graph size, reasoning expansions, and query latency.

## 14. Acceptance Criteria

- foreground append cost is independent of existing collection size
- no append or mapping update triggers historical LLM extraction
- unchanged chunks perform no provider or persistence work
- changed documents process only changed chunk hashes
- all resolution paths have hard candidate and time bounds
- raw observations are immutable and all derived layers are versioned overlays
- directed edges are never merged in reverse unless explicitly symmetric
- propositions and rules encode conditions, exceptions, scope, polarity, and
  epistemic conflict as executable graph structure
- query reasoning operates on bounded working graphs and returns activation traces
- global analytics and compilation are asynchronous and never block ingestion
- structural metrics are tied to named projections and graph versions
- million-node tests demonstrate bounded foreground operations and bounded query
  expansion

## 15. Non-Goals

- re-running LLM extraction over historical chunks during ordinary evolution
- eagerly rewiring the entire graph after entity or predicate merges
- executing inference over the full million-node graph per query
- treating arbitrary relationships as executable rules
- using centrality or community membership as evidence of truth
- running unbounded clique, motif, path, or graph-rewrite searches
- requiring the MCP server to access a client's filesystem

