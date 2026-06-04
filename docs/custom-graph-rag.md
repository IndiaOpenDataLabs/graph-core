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
   - community summaries
   - bridge-node summaries
   - connector-path summaries
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

Current implementation:

1. Build a weighted undirected projection of canonical relationships
2. Compute strong-edge communities using `min_edge_strength`
3. Merge very small communities into stronger neighbors
4. Compute bridge nodes using:
   - articulation behavior
   - cross-community connectivity
   - weighted degree
5. Select top anchors
6. Build bounded connector paths between anchors

This runs over canonical collection-level relationships, not chunk-local raw extractions.

### Output of first pass

The analysis returns:

- communities
- top anchors
- bridge nodes
- connector paths

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

Each strong community becomes a summary node with text like:

- size
- strong edge count
- anchor preview
- representative entity names

### Bridge summaries

Each important bridge node becomes a derived summary node describing:

- its community
- which other communities it connects to
- external connection strength
- weighted degree

### Connector summaries

Each bounded connector path becomes a derived summary node describing:

- start anchor
- end anchor
- hop count
- path score
- ordered flow of entities

## How Queries Use the Derived Graph

Derived graph usage is now integrated into `custom_graph_rag` query flow.

### 1. Derived summary retrieval

At query time:

- the user question is embedded with a derived-retrieval instruction
- vector search is run against collection chunks filtered by:
  - `memory_type=derived_graph`

This returns the most relevant derived summaries.

### 2. Derived graph expansion

For each matched derived summary:

- load the derived Falkor node
- expand one hop in the derived graph
- collect:
  - linked derived edges
  - linked target nodes
  - base graph provenance from `source_ids`

### 3. Seeding base graph retrieval

Base entity IDs recovered from derived provenance are injected into the base graph query state as additional relevant entities.

So the derived layer does not replace base retrieval.
It nudges the base retrieval toward the right part of the graph.

### 4. Final answer context

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

### 2. Query integration is conservative

Current query usage is:

- retrieve derived summaries
- expand one hop
- seed base entity IDs
- prepend derived understanding to the final prompt

That is a safe first integration, but not the final form.

A later version can query the derived graph as a real routing layer first, then descend into selected base subgraphs more aggressively.

### 3. No background build pipeline yet

Derived understanding is built explicitly today.
It is not yet an automatic background pass on ingest completion.

### 4. Cross-chunk logic is still limited

The base graph already merges entities and relationships across chunks, but true cross-chunk understanding is still mostly indirect.

The derived graph is the beginning of that layer, not the end state.

## Mental Model

The simplest way to think about `custom_graph_rag` now is:

- **base graph** = grounded extracted facts
- **derived graph** = structural summaries of important regions and flows

Querying now uses both:

- derived graph for guidance
- base graph for evidence

That is the current architecture. It is already more expressive than chunk-only retrieval, but it is still an intermediate step toward richer graph reasoning.
