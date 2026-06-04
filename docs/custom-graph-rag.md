# Custom Graph RAG

This document explains how `custom_graph_rag` works today:

- what gets extracted and stored during ingestion
- how the base graph is queried
- what the multi-dimensional relationship model changes
- how the derived graph layer is built and used

It focuses on the current implementation in this repo, not an abstract design.

## Overview

`custom_graph_rag` has two graph layers:

1. **Base graph**
   - canonical entities
   - canonical relationships
   - relationship descriptions
   - per-collection relationship/entity embeddings
   - FalkorDB graph for traversal

2. **Derived graph**
   - higher-level summaries built from the base graph
   - typed community summaries
   - typed bridge-node summaries
   - directed typed connector-path summaries
   - separate FalkorDB graph plus derived summary vectors

The base graph is the source of truth.
The derived graph is a synthesized routing and summarization layer on top.

## Base Graph Ingestion

### 1. Chunk extraction

Each document is split into chunks.
Each chunk goes through LLM-based extraction with optional gleaning passes.

Extraction returns:

- entities
- relationships

Relationships include:

- `source_name`
- `target_name`
- `description`
- `keywords`
- `weight`
- `rel_type`

### 2. Canonicalization and merge

Chunk-local entities and relationships are merged into canonical collection-level records in Postgres:

- `graph_entities`
- `entity_descriptions`
- `entity_aliases`
- `graph_relationships`
- `relationship_descriptions`

Important details:

- entities are resolved across chunks into canonical entities
- relationships are canonicalized by:
  - source entity
  - target entity
  - `rel_type`
- repeated relationship evidence increases canonical relationship weight
- descriptions accumulate as supporting evidence

### 3. Base graph storage

The base graph is stored in two places:

- **Postgres**
  - canonical entities and relationships
  - relationship weights and descriptions
- **FalkorDB**
  - one graph per collection
  - graph name:
    - `collection_<collection_id_without_dashes>`

In Falkor:

- nodes are `:Entity`
- edges are typed by `rel_type`
- edge properties also store:
  - `id`
  - `weight`
  - `keywords`
  - `collection_id`
  - `rel_type`

### 4. Embeddings

Per collection, we store:

- entity description embeddings
- relationship description embeddings
- entity centroid embeddings
- raw chunk embeddings

Relationship embedding text is rel-type-aware. It includes:

- source
- target
- `rel_type`
- description
- keywords

That matters because retrieval should distinguish:

- `CAUSES`
- `EXPLAINS`
- `SUPPORTS`
- `IS_AN_EXAMPLE_OF`

instead of treating every edge as generic prose.

## Multi-Dimensional Relationships

The important change in the base graph is that relationships are not just:

- `A -> B`

They are:

- `A -[REL_TYPE]-> B`

The same pair can exist with multiple meanings.

Example:

- `Pranayama -[EXPLAINS]-> Ojas`
- `Pranayama -[AFFECTS]-> Ojas`
- `Pranayama -[REQUIRES]-> Ojas`

This matters in three places:

1. **Storage**
   - separate canonical relationships per `rel_type`

2. **Embedding retrieval**
   - relationship embedding text includes `rel_type`

3. **Final context**
   - relationship evidence is grouped under `Relationships By Type`

## Base Query Flow

The main query implementation is in:

- `src/graph_core/services/graph/query/graph_rag.py`

### Supported modes

`custom_graph_rag` supports:

- `entity-first`
- `relationship-first`
- `hybrid`
- `mix`

Current default is `mix`.

### Query-side embeddings

Query embeddings are instruction-formatted before retrieval:

- entity retrieval prompt
- relationship retrieval prompt
- derived graph retrieval prompt

This improves embedding behavior for models like `Qwen3-Embedding-8B`.

### Base retrieval strategies

#### Entity-first

- retrieve entity seeds from entity embeddings and aliases
- traverse outward in Falkor
- score edges using:
  - relationship similarity
  - canonical edge weight
  - keyword overlap
  - dimension weight

#### Relationship-first

- retrieve relationship seeds first
- take endpoint entities
- connect endpoints using bounded path search
- use those paths as the main evidence

#### Hybrid

- merge entity-first and relationship-first states

#### Mix

- first-pass entity candidate read
- optional rewrite into longer retrieval subqueries
- relationship-first retrieval on those subqueries
- merge resulting states

The rewrite is gated by first-pass entity confidence.

## Relationship Scoring

Relationship relevance is not just cosine similarity.

The combined score uses:

- raw relationship embedding similarity
- canonical relationship weight
- keyword overlap with query
- dimension weight for the active `rel_type`

Important implementation detail:

- raw relationship similarity and combined relationship score are stored separately
- path expansion uses raw similarity as input, then applies weighting cleanly
- final relation ordering uses combined score

This avoids double-applying weight effects during path search.

## Final Prompt for Base Graph

The answer LLM does not see retrieval internals.

It sees:

- `Derived Understanding` if available
- `Entities`
- `Relationships By Type`
- original user question

The relationship section is formatted as:

- `SRC -[REL_TYPE]-> TGT: description`

This lets the model see multiple meanings between the same endpoints.

## Derived Graph

The derived graph is a second-pass structural summary over the canonical base graph.

It is not extracted directly from chunks.
It is built from the merged collection graph.

### Why it exists

The base graph is good at local facts.
The derived graph is meant to capture higher-order structure, such as:

- subsystem-like regions
- bridge entities
- small connector flows between important anchors

This is the layer that should eventually support more genuine “understanding” instead of only chunk-local extraction.

## Derived Graph Build: First Pass

The first pass is structural analysis over the base graph.

Current implementation is **rel-type-aware** and partly **direction-aware**:

1. Split canonical relationships by `rel_type`
2. For each `rel_type`:
   - build a weighted undirected projection for community detection
   - build a directed adjacency view for path and metric analysis
3. Compute strong-edge communities using `min_edge_strength`
4. Merge very small communities into stronger neighbors
5. Compute per-node graph metrics for each `rel_type`:
   - `PageRank`
   - `HITS` authority / hub
   - `eigenvector centrality`
   - directed `betweenness`
   - directed harmonic `closeness`
   - articulation behavior
   - cross-community connectivity
   - inbound / outbound strength
6. Select typed anchors and typed bridge nodes
7. Build bounded **directed** connector paths between anchors for that `rel_type`
8. Aggregate the per-type analyses into a collection-level summary

This runs over canonical collection-level relationships, not chunk-local raw extractions.

### Output of first pass

The analysis returns:

- `rel_type`-specific communities
- `rel_type`-specific node metrics
- aggregated top anchors
- aggregated bridge nodes
- directed connector paths

This is a structural view, not yet a natural-language summary layer.

## Derived Graph Build: Second Pass

The second pass turns the first-pass structures into stored derived knowledge.

Current derived node types:

- `derived_community`
- `derived_bridge`
- `derived_connector`
- `base_entity_ref`

Current derived edge types:

- `SUMMARIZES`
- `FOCUSES_ON`
- `USES`
- `CONNECTS`

### Storage

The derived layer is persisted in two places:

1. **Derived Falkor graph**
   - one graph per collection
   - graph name:
     - `collection_<collection_id_without_dashes>_derived`

2. **Derived vector summaries**
   - stored in the collection vector table
   - tagged with:
     - `memory_type=derived_graph`

### Rebuild behavior

When derived understanding is rebuilt:

- the derived Falkor graph is dropped and recreated
- previous `memory_type=derived_graph` vector chunks are deleted
- new summaries are embedded and inserted

So the derived layer is treated as canonical generated state, not append-only memory.

## What the Derived Graph Currently Stores

### Community summaries

Each strong community becomes a summary node scoped to a specific `rel_type`.

The summary includes:

- `rel_type`
- size
- strong edge count
- anchor preview
- representative entity names

### Bridge summaries

Each important bridge node becomes a derived summary node describing:

- its community
- the `rel_type`s in which it is important
- which other communities it connects to
- external connection strength
- weighted degree
- inbound / outbound strength
- graph metrics such as:
  - betweenness
  - closeness
  - hub score
  - authority score

### Connector summaries

Each bounded connector path becomes a derived summary node describing:

- `rel_type`
- start anchor
- end anchor
- hop count
- path score
- directed flow of entities and edges

## How Queries Use the Derived Graph

Derived graph usage is now integrated into `custom_graph_rag` query flow.

### 1. Initial base-graph retrieval

At query time, the normal `custom_graph_rag` retrieval runs first:

- entity-first, relationship-first, hybrid, or mix
- rel-type-aware graph traversal
- combined relationship scoring using similarity, weight, keywords, and dimension weight

This builds the initial matched graph footprint.

### 2. Graph-grounded route profiling

The matched footprint is then compared against offline graph metrics from the enhanced analysis.

From the matched nodes, the query layer derives a route profile across:

- `hub`
- `authority`
- `bridge`
- `central`
- `importance`

This routing is **graph-grounded**, not keyword-routed:

- it uses the metric profile of the actually matched nodes
- it also tracks dominant matched `rel_type`s

### 3. Route-aware derived summary retrieval

The user question is embedded with a derived-retrieval instruction and vector search is run against collection chunks filtered by:

- `memory_type=derived_graph`

But the resulting derived summaries are no longer treated equally.
They are reranked by:

- route kind
  - e.g. hub-like questions slightly prefer bridge/connector derived nodes
  - authority-like questions slightly prefer community summaries
  - bridge-like questions strongly prefer bridge/connector summaries
  - central questions prefer connector/community summaries
- matched `rel_type` profile

So the derived layer is used differently depending on the graph shape of the query.

### 4. Derived graph expansion

For each matched derived summary:

- load the derived Falkor node
- expand one hop in the derived graph
- collect:
  - linked derived edges
  - linked target nodes
  - base graph provenance from `source_ids`

### 5. Seeding base graph retrieval

Base entity IDs recovered from derived provenance are injected into the base graph query state as additional relevant entities.

So the derived layer does not replace base retrieval.
It nudges the base retrieval toward the right part of the graph.

### 6. Final answer context

The final prompt now includes:

- `Derived Understanding`
- `Entities`
- `Relationships By Type`

The LLM is explicitly instructed:

- use derived understanding as high-level guidance
- ground specific claims in base entity and relationship evidence

## Operational Commands

### Analyze first-pass structure

```bash
PYTHONPATH=src .venv/bin/python docs/HELPER.py graph-analysis <collection_id>
```

or

```bash
uv run python -m graph_core.scripts.graph_analysis <collection_id>
```

### Build and persist derived understanding

```bash
PYTHONPATH=src .venv/bin/python docs/HELPER.py graph-understanding <collection_id>
```

or

```bash
uv run python -m graph_core.scripts.graph_understanding <collection_id> <namespace_id>
```

## Current Limitations

### 1. Derived summaries are heuristic, not LLM-authored

The derived layer is currently built from graph structure with deterministic summaries.

That is useful, but still limited.
Later it can grow into:

- richer subgraph summaries
- subsystem naming
- invariants
- failure modes
- code-flow understanding

### 2. Query routing is metric-aware, but still lightweight

Current query usage is stronger than before:

- retrieve base graph footprint first
- derive a metric-based route profile from matched nodes
- rerank derived summaries by route kind and `rel_type`
- expand one hop
- seed base entity IDs
- prepend derived understanding to the final prompt

This is better than uniform derived retrieval, but it is still lightweight.

It does **not** yet:

- run full metric-specific traversal policies
- construct different detailed subgraphs for hub vs authority vs bridge questions
- use the derived graph as the primary planner for the entire retrieval path

A later version can make routing more aggressive and subgraph-aware.

### 3. No background build pipeline yet

Derived understanding is built explicitly today.
It is not yet an automatic background pass on ingest completion.

### 4. Cross-chunk logic is still limited

The base graph already merges entities and relationships across chunks, but true cross-chunk understanding is still mostly indirect.

The derived graph is the beginning of that layer, not the end state.

### 5. Sparse semantic analyses are still missing

The current enhanced graph is best at structural architecture questions.

It is still weaker on sparse semantic questions that need special edge families, such as:

- exception redundancy
- field write/read usage
- auth gating
- config propagation

Those cases will require richer base graph edges such as:

- `RAISES`
- `CATCHES`
- `READS`
- `WRITES`
- `GUARDS`

## Mental Model

The simplest way to think about `custom_graph_rag` now is:

- **base graph** = grounded extracted facts
- **derived graph** = typed structural summaries and routing hints over important regions and flows

Querying now uses both:

- derived graph for metric-aware guidance
- base graph for evidence

That is the current architecture. It is already more expressive than chunk-only retrieval, but it is still an intermediate step toward richer graph reasoning.
