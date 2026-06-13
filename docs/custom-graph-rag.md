# Custom Graph RAG

This document describes the current `custom_graph_rag` implementation in this repo:

- how the base graph is ingested and queried
- how the meta graph is built today
- how base and meta graphs are stored
- how query uses both layers

It reflects the code after the move away from community-based derived graphs.

## Overview

`custom_graph_rag` now works with one base collection plus zero or more
higher-level meta collections:

1. **Base collection**
   - the canonical extracted graph from source documents

2. **Meta collections**
   - sibling collections named:
     - `<base_name>__meta__l1`
     - `<base_name>__meta__l2`
     - ...
   - `__meta` is still recognized as a legacy level-1 name
   - each level is built from the graph at the level immediately below it
   - each stores higher-level concepts and concept-to-concept relationships

Both are real collections. Both use the same persistence/query machinery:

- Postgres canonical graph tables
- vector storage
- FalkorDB graph storage

The difference is only in how the meta collection is generated.

## Base Graph

### Ingestion

Documents are chunked, extracted, canonicalized, and merged into:

- `graph_entities`
- `entity_descriptions`
- `entity_aliases`
- `graph_relationships`
- `relationship_descriptions`

Base relationships are canonicalized by:

- source entity
- target entity
- `rel_type`

Relationship descriptions, keywords, and weights are retained as evidence.

### Storage

The base collection is stored in:

- **Postgres**
  - canonical entities, aliases, descriptions, relationships
- **FalkorDB**
  - one graph per collection:
    - `collection_<collection_id_without_dashes>`

### Embeddings

Per collection, we store embeddings for:

- raw chunks
- entity descriptions / centroids
- relationship descriptions

Relationship embedding text is `rel_type`-aware, so:

- `CAUSES`
- `SUPPORTS`
- `IS_AN_EXAMPLE_OF`
- `DEPENDS_ON`

remain meaningfully distinct during retrieval.

## Query Modes

The main query implementation is:

- `src/graph_core/services/graph/query/graph_rag.py`

Supported modes:

- `entity-first`
- `relationship-first`
- `hybrid`
- `mix`

`mix` is still the most capable mode.

Base query flow remains:

1. retrieve entity and/or relationship seeds
2. traverse the Falkor graph
3. score and merge graph evidence
4. answer from grounded entity/relationship context

The final answer prompt is grounded in:

- `Entities`
- `Relationships By Type`

and uses the base graph as the evidence layer.

## Meta Graph

### What it is

The old special `_derived` graph is gone.

The meta layer is now one or more normal collections:

- `<base_name>__meta__l1`
- `<base_name>__meta__l2`
- ...

It is materialized using the same models as base:

- `GraphEntity`
- `EntityAlias`
- `EntityDescription`
- `GraphRelationship`
- `RelationshipDescription`

So meta concepts are first-class graph objects, not special-case Falkor-only nodes.

### How it is built

Each meta collection is built from the graph below it in two stages:

1. **Concept candidate mining**
2. **Concept materialization + deterministic concept linking**

## Concept Candidate Mining

The current candidate generator is **role similarity**, not Louvain communities.

### Role-similarity construction

For each base entity, we build a typed signature from its neighborhood:

- outgoing tokens:
  - `(rel_type, target)`
- incoming tokens:
  - `(source, rel_type)`

This gives each node a typed in/out structural signature.

### Pair similarity

We compare entity pairs using standard measures:

- **overlap count**
- **cosine similarity**
- **Jaccard similarity**

Current thresholds in analytics:

- overlap `>= 3`
- cosine `>= 0.2`
- Jaccard `>= 0.1`

If a pair clears those thresholds, it becomes an edge in a **role-similarity graph**.

### Groups

From that similarity graph we extract **maximal cliques**.

These cliques can be:

- pairs
- triplets
- larger groups

Those groups are the current concept candidates.

This is the key change:

- we no longer treat dense graph fragments as concepts
- we treat **entities with similar typed graph roles** as concept candidates

That makes the meta layer much closer to:

- `Vata`, `Pitta`, `Kapha` -> `Dosha`
- `Heaviness`, `Inertia`, `Stagnation` -> `Tamasic Qualities`

instead of producing one concept per `rel_type` fragment.

## Concept Induction

Each role-similarity clique is sent to the collection LLM as one candidate region.

The prompt gives the LLM:

- member entities
- dominant relation types
- pairwise role-similarity evidence
- representative neighborhood edges

The LLM returns one concept object:

- `label`
- `concept_type`
- `description`
- `aliases`
- `importance_reason`
- `member_entity_names`

The `label` becomes the canonical name of the meta entity.

Example:

```json
{
  "anchor": "Heaviness",
  "label": "Tamasic Qualities",
  "concept_type": "Guna/Energetic State"
}
```

In that case:

- `Tamasic Qualities` is the meta entity name
- `Heaviness` is evidence grounding, not the meta entity name

## Meta Relationship Construction

Concept-to-concept edges are **not** produced by a second LLM pass.

They are built deterministically from the **base graph**.

After concepts are grounded to base `source_ids`, concept pairs are linked using:

- direct cross-concept base edges
- dominant cross-concept `rel_type`s
- short directed paths between grounded base node sets
- boundary entities
- bridge/intermediate entities

The current persisted concept-to-concept edge type is:

- `CONNECTS_TO`

Its description and keywords summarize:

- dominant base `rel_type`s
- path evidence
- boundary / bridge entities

So:

- concept nodes are LLM-induced
- concept-to-concept edges are base-graph-derived

## Meta Collection Materialization

The meta collection is materialized through the same resolver/update path as base ingestion.

That means:

- canonical-name matching
- alias matching
- normal entity/relationship persistence

There is no separate special-case meta storage model anymore.

Meta entities also carry:

- aliases
- type
- base evidence IDs via metadata/source IDs

## Query Behavior

Query now treats base and meta collections the same way operationally.

When querying a collection:

1. run `custom_graph_rag` on the selected collection using the user-selected mode
2. if higher meta levels exist:
   - run the same `custom_graph_rag` logic on every higher level with the same mode
3. combine all higher-level contexts into the final answer prompt

So there is no special shallow meta retrieval path anymore.

### Important constraint

The final answer is instructed to:

- use meta concepts only as internal higher-level context
- ground the actual answer in base entities and base relationships
- avoid framing the answer around meta concepts unless the same idea is directly supported by base evidence

So:

- meta graph helps with abstraction
- base graph remains the grounding/evidence layer

## Chunking

Chunking is now structure-aware.

### Text

Non-code text uses recursive character splitting rather than fixed token windows.

### Code

Code uses code-aware chunking first, with fallback to language-aware recursive splitting.

This is meant to reduce:

- mid-function splits
- mid-class splits
- broken code context

## Current Shape of the Meta Graph

For a typical collection, the meta graph now looks like:

- a relatively small number of concept candidates from role cliques
- grounded `base_entity_ref` nodes as evidence
- deterministic `CONNECTS_TO` edges between concepts
- `EVIDENCED_BY` edges from concepts to grounded base entities

This is much smaller and cleaner than the old community/bridge/connector derived graph.

## Current Limitations

### 1. Role similarity still needs tuning

Current thresholds are fixed:

- overlap `>= 3`
- cosine `>= 0.2`
- Jaccard `>= 0.1`

These are reasonable, but not final.

The important part is the method:

- role similarity over typed signatures

not the exact numeric thresholds.

### 2. Clique induction can still over-fragment

Some collections produce many small exact-match pairs/triplets.

This is still better than the old community pipeline, but there is more work to do on:

- larger group induction
- merging nearby concepts
- concept deduplication beyond simple resolver behavior

### 3. Sparse semantic edges still matter

For some questions, the base graph still needs richer edge families such as:

- `RAISES`
- `CATCHES`
- `READS`
- `WRITES`
- `GUARDS`

Without those, neither base nor meta graph can fully answer some semantic questions.

## Mental Model

The current mental model is:

- **base graph** = grounded facts and evidence
- **meta graph** = higher-level concepts induced from role-similar entities, linked deterministically through base-graph structure

And query uses both:

- meta graph for abstraction
- base graph for evidence

That is the current architecture.
