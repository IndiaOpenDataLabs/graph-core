# Independent Entity and Relationship Extraction Plan

> **Status:** Proposed
>
> **Date:** 2026-08-09
>
> **Base:** `main` at `f921e52`
>
> **Scope:** Restore independent entity and relationship extraction for generic
> Custom Graph RAG in one structured LLM call. Keep code-domain extraction
> unchanged.

## 1. Problem

Custom Graph RAG currently asks the LLM to emit only relationships. The service
then derives `ExtractionResult.entities` from relationship endpoints.

This means a meaningful concept survives extraction only when the LLM chooses it
as a relationship endpoint. Given:

```text
Krishna teaches Arjuna to perform duty without attachment to results.
```

the current contract can preserve `Krishna` and `Arjuna` while losing `Duty`,
`Attachment`, or `Results` as independently retrievable concepts.

The behavior was introduced by commit `1320886` (`relate chat and graph
extraction prompts`). Before that commit, generic extraction returned independent
`entities` and `relationships` arrays in one response. The commit:

- removed `entities` from `_GENERIC_EXTRACTION_SCHEMA`;
- removed entity extraction instructions from `_EXTRACTION_SYSTEM_PROMPT`;
- changed generic relationship endpoints into nested endpoint objects;
- added `_collect_generic_entities()` to derive entities only from endpoints;
- changed generic gleaning to operate on relationships only.

The endpoint-based contract was useful for code extraction, where concrete code
objects are naturally exposed by typed operations. Applying it to all generic
content made the graph's node vocabulary depend on the edge extractor's choices.

## 2. Objective

In one structured LLM call, independently extract:

1. salient entities and concepts present in the text;
2. selective, explicitly supported relationships between those entities.

The change must not:

- add a second entity-to-relationship LLM pass;
- connect every possible entity pair;
- turn every verb, modifier, or preposition into a new relationship type;
- hide meaningful concepts inside relationship descriptions or nested argument
  objects;
- change the code-domain taxonomy contract in the first implementation.

## 3. Design Principle

Entity extraction and relationship extraction answer different questions:

```text
Entity extraction:       What meaningful things are active in this passage?
Relationship extraction: Which reusable semantic connections are explicitly
                         supported between those things?
```

A node does not require an edge in the same chunk to be useful. An independently
extracted concept can:

- receive descriptions and embeddings;
- become a query seed;
- resolve to an entity found in another chunk;
- later participate in a bounded associative projection.

Likewise, graph density must not be produced by inventing pairwise facts. The
semantic relationship graph should remain selective. A separate rebuildable
co-activation/affinity projection may later provide dense associative traversal
without presenting co-occurrence as truth.

## 4. Single-Call Extraction Contract

Restore a generic response with two independent arrays:

```json
{
  "entities": [
    {
      "name": "Krishna",
      "type": "Person",
      "description": "The teacher addressing Arjuna."
    },
    {
      "name": "Arjuna",
      "type": "Person",
      "description": "The recipient of Krishna's teaching."
    },
    {
      "name": "Duty",
      "type": "Concept",
      "description": "Action or obligation that should be performed."
    },
    {
      "name": "Non-Attachment To Results",
      "type": "Concept",
      "description": "Acting without attachment to the outcome."
    }
  ],
  "relationships": [
    {
      "source": "Krishna",
      "target": "Arjuna",
      "description": "Krishna teaches Arjuna how duty should be performed.",
      "keywords": ["teaching", "duty", "guidance"],
      "weight": 0.95,
      "rel_type": [
        {
          "name": "TEACHES",
          "description": "Krishna gives instruction to Arjuna.",
          "keywords": ["teaches", "instruction"],
          "weight": 0.95
        }
      ]
    }
  ]
}
```

Relationship `source` and `target` refer to exact names in the `entities` array.
The parser validates this invariant. Response array order is not semantic
identity.

A relationship may introduce an endpoint missing from `entities` only as a
compatibility fallback. The parser creates an `UNKNOWN` entity and records a
validation metric; accepted extractor output should normally have zero such
repairs.

### 4.1 Entity granularity

The prompt must distinguish semantic concepts from grammatical tokens.

Prefer:

```text
Non-Attachment To Results
```

when the phrase functions as one meaningful concept. Do not mechanically split
it into `Non-Attachment` and `Results` unless the passage treats both as
independently meaningful and describes useful connections involving them.

Extract an entity when it is:

- a concrete participant, object, place, organization, work, or system;
- a named or clearly meaningful concept;
- a process, state, event, or teaching that is discussed as a thing;
- useful as a potential retrieval anchor beyond the current sentence.

Do not extract:

- generic filler nouns;
- pronouns when their referent is known;
- every adjective or adverb;
- arbitrary fragments created only to satisfy a relationship schema.

### 4.2 Relationship granularity

Relationships should use a compact, reusable semantic vocabulary. The precise
meaning remains in the relationship description and embedding.

Prefer:

```text
TEACHES
QUALIFIES
CONTRASTS_WITH
PART_OF
CAUSES
ABOUT
```

over one-off grammatical labels such as:

```text
PERFORMED_WITHOUT
DIRECTED_TOWARD_IN_THIS_SENTENCE
FREE_FROM_WHILE_DOING
```

A relationship is emitted only when:

- both endpoints are meaningful entities from the inventory;
- the text explicitly supports the connection;
- the edge helps recover or traverse the passage's meaning;
- its type is reusable beyond this exact sentence;
- its description can state the precise local meaning without changing the
  relationship type into a sentence-specific phrase.

The extractor must not evaluate or fill every possible pair. With `N` entities,
it returns only the supported subset of the `N * (N - 1)` directed candidates.

## 5. Prompt Shape

The generic system prompt should use this structure:

```text
1. Entity extraction
   - First identify the meaningful entities and concepts explicitly present.
   - Include salient concepts even when no explicit relationship is emitted for
     them in this chunk.
   - Prefer coherent compound concepts over blindly splitting every noun.
   - Return name, type, and a source-grounded description.

2. Relationship extraction
   - Identify selective, explicitly supported relationships between the
     extracted entities.
   - Source and target must exactly match names from the entities array.
   - Do not connect every pair.
   - Prefer stable, reusable relationship types over sentence-specific labels.
   - Preserve direction when the text expresses direction.
   - Put local precision in descriptions and keywords, not in proliferating
     predicate names.

3. Output
   - Return one object containing entities and relationships.
   - Output both arrays in the same structured response.
   - Do not invent entities or relationships unsupported by the text.
```

The user prompt remains a single request:

```text
Extract the meaningful entities and explicitly supported relationships from the
following text. Return the structured entities and relationships object.
```

## 6. Parser and Ingestion Changes

For generic, chat, and dynamically classified prose domains:

1. Restore `entities` to `_GENERIC_EXTRACTION_SCHEMA`.
2. Parse entities directly with `_extract_entities()`.
3. Parse relationships using source/target entity names.
4. Validate that relationship endpoints exist in the entity inventory.
5. Retain a compatibility repair for missing endpoint entities, with metrics.
6. Stop using `_collect_generic_entities()` as the normal source of entities.
7. Keep the current ingestion loops: independently extracted entities already
   flow through `IncrementalEntityResolver` and receive descriptions, aliases,
   types, embeddings, and centroids.
8. Preserve directed relationship resolution from PR #8.

The code-domain schema and fixed taxonomy remain endpoint-derived for now. Code
can adopt the independent contract later only if code-specific fixtures show a
real retrieval benefit.

## 7. Gleaning and Call Count

The restored contract itself uses one LLM call. However, collections currently
default to `gleaning_passes=1`, which produces another call for non-empty
extraction.

To make one-call ingestion the normal behavior:

- change the default for newly created collections to `gleaning_passes=0`;
- preserve explicit nonzero configuration for users who choose the cost;
- if gleaning is enabled, use the same independent `entities` and
  `relationships` schema and request only missed/corrected items;
- report extractor and gleaning call counts separately.

This avoids silently adding a second pass while retaining gleaning as an opt-in
quality mechanism.

## 8. Extraction Contract Version

The prompt/schema change must not silently reuse relationship-only cached
extractions.

Add a contract key such as:

```text
generic-independent-entities-v1
```

and include it in raw extraction cache identity. Existing cached payloads remain
readable under their old contract; no historical LLM re-extraction occurs unless
that content is explicitly ingested under the new contract.

## 9. Associative Density

Independent entity extraction intentionally permits entities without asserted
relationships. Do not compensate by asking the LLM to invent a dense semantic
clique.

A later derived association projection may add bounded links based on:

- co-activation in the same sentence, frame, or chunk;
- repeated co-occurrence across chunks;
- entity-description similarity;
- a fixed top-neighbor and weight threshold.

These links must be marked as derived association rather than source-asserted
fact. Query traversal may use them to reach relevant regions, but answer context
must not verbalize them as claims.

This projection is a follow-up, not required to restore independent extraction.

## 10. Verification

### Extraction contract

- one structured call returns both arrays;
- `Duty` survives even when it is not a relationship endpoint;
- compound concepts are not blindly decomposed;
- every accepted relationship endpoint resolves to an extracted entity;
- unsupported entity pairs do not receive invented edges;
- relationship labels remain reusable and descriptions preserve local precision;
- directed source and target survive parsing.

### Ingestion and retrieval

- independent entities are persisted and embedded;
- an independent entity with no same-chunk relationship is retrievable as an
  entity-vector seed;
- relationship ingestion behavior remains unchanged;
- retry uses the contract-versioned cache and invokes no duplicate LLM call;
- a new default collection performs one extraction call when gleaning is not
  explicitly enabled.

### Regression fixture

For:

```text
Krishna teaches Arjuna to perform duty without attachment to results.
```

assert that:

- `Krishna`, `Arjuna`, `Duty`, and a coherent non-attachment concept are
  extracted as entities;
- `Krishna -[TEACHES]-> Arjuna` is present and directed;
- no exhaustive pairwise clique is emitted;
- no sentence-specific predicate proliferation is required;
- querying for duty or non-attachment can seed the independently stored entity
  even before an associative projection is implemented.

## 11. Delivery

Implement as one focused change set:

1. characterize the pre-`1320886` generic schema and add regression fixtures;
2. restore independent generic entity extraction in the structured schema and
   prompt;
3. add endpoint-inventory validation and compatibility repair metrics;
4. update generic gleaning to the same contract;
5. version the extraction cache contract;
6. make gleaning opt-in for new collections;
7. add ingestion-to-entity-query coverage for an entity that is not a
   relationship endpoint.

## 12. Non-Goals

- a second entity or relationship LLM pass;
- formal rule extraction or compilation;
- temporal relationship identity;
- reifying every relationship as an entity;
- connecting every extracted entity pair;
- using derived co-activation links as asserted facts;
- changing code-domain extraction without dedicated evidence.
