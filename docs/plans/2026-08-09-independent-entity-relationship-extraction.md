# Independent Entity and Relationship Extraction Plan

> **Status:** Proposed, reviewed against `main` at `32a3976`
>
> **Date:** 2026-08-09, revised 2026-09-15
>
> **Base:** `main` at `f921e52`
>
> **Scope:** Restore independent entity and relationship extraction for generic
> Custom Graph RAG in one structured LLM call. Keep code-domain extraction
> unchanged.
>
> **Review note:** the 2026-09-15 revision records verification against the
> current code, including one correction of an earlier revision of this document
> (section 1.1). Sections 1.1, 4, 4.3, 5, 6, 6.1, 7, 8, 10, 11 and 12 contain the
> corrections and preconditions. The section 8 cache contract work, and the
> section 1.1 repair-path description fix, are landing first on
> `fix/extraction-contract-and-endpoint-descriptions`, independently of this
> plan.

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

### 1.1 What is and is not actually lost today

An earlier revision of this section claimed the generic path asks for endpoint
descriptions and throws them away. That was wrong, and correcting it matters for
how this plan's benefit should be estimated. Verified against `main`
(`32a3976`):

- `_collect_generic_entities()` does use the endpoint description. It keeps the
  longest description per normalized name and stores it as
  `ExtractedEntity.description`.
- `_ingest_graph_chunk()` passes that description into
  `IncrementalEntityResolver.resolve_entity()`. An endpoint that survives
  extraction therefore reaches the resolver with a description and becomes an
  embeddable, searchable entity.
- `ExtractedEntity` has exactly `name`, `entity_type`, and `description`. There
  is no endpoint-provenance field, and `_collect_code_entities()` does not
  populate one either. The two collectors differ only in the hardcoded type.

Two real losses remain, both smaller than the missing-node problem:

- every generic entity is created with `entity_type="UNKNOWN"`. The resolver's
  type check short-circuits only on an empty type and otherwise rejects just three
  incompatible pairs (`person/place`, `person/object`, `place/concept`). `UNKNOWN`
  is in none of them, so every pair involving it passes and the guard does no work
  for generic content. A typed independent array restores a check that currently
  never rejects anything.
- endpoint descriptions are discarded at relationship-parse time, and the
  ingestion repair path that creates an entity for a relationship endpoint
  missing from the entity inventory therefore creates it with an empty
  description. `_add_description_and_update_centroid()` returns early on an empty
  description, so such an entity gets no `EntityDescription` row, no embedding,
  and no centroid contribution: it exists in the graph and is invisible to every
  retrieval path. That path is effectively unreachable on `main`, because entities
  are endpoint-derived and the gleaning pass merges gleaned entities collected
  from gleaned endpoints, so the two name sets always coincide. It becomes a live
  path exactly when this plan makes the entity array independent, and the
  exact-name binding in section 4 then decides how often it fires.

Both are fixed on `fix/extraction-contract-and-endpoint-descriptions`, with a
warning log added at the repair site. That site is also where the exact-name
binding in section 4 will land, so read it before implementing: it is the one
place where a name mismatch becomes a permanently invisible node rather than a
dropped edge.

The core claim of section 1 stands. A concept the model did not choose as a
relationship endpoint does not exist in the graph at all, under any name, with or
without a description.

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

That validator does not exist today and must be written. `_parse_relationship_
endpoint()` accepts an endpoint object or string and `_extract_generic_
relationships()` only drops items whose endpoints fail to parse. Enforcing the
invariant requires a new step after both arrays are parsed.

A relationship may introduce an endpoint missing from `entities` only as a
compatibility fallback. The parser creates an `UNKNOWN` entity and records a
validation metric; accepted extractor output should normally have zero such
repairs.

Two constraints on that repair, verified against `main`:

- Matching must run on the same normalization the resolver uses, not on raw
  strings. `LLMGraphExtractor._normalize_entity_name()` and
  `entity_resolver._normalize_name()` collapse whitespace and cap length but
  preserve case, and `_merge_entities()` dedupes on raw `name.strip()`. If the
  validator compares raw strings, ordinary model casing drift between the two
  arrays becomes a spurious repair. Normalize with the existing helper and match
  case-insensitively.
- There is no metrics framework in this repository. `entity_count` and
  `relationship_count` are Postgres columns on `ingestion_records`, not counters.
  Before implementing, choose a sink: a structured log line per chunk carrying
  `repairs`, `entities`, `relationships`, and the contract, so a bad prompt
  shows up as a repair-rate shift. A counter column alone cannot distinguish a
  prompt regression from an ordinary chunk.

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

### 4.3 Wire format decision

The generic response must use its own relationship item schema.
`_RELATIONSHIP_ITEM_SCHEMA` is spread into `_GENERIC_EXTRACTION_SCHEMA` and
referenced directly by `_build_code_taxonomy_schema()`. Editing it in place
mutates the code-domain contract this plan promises not to change. Create a
generic-local copy. The generic schema also sets `additionalProperties: false` at
the top level, so the new `entities` property must be declared explicitly, and
`_extract_entities()` — the parser for an entities array — already exists and is
currently unused.

Decide explicitly whether generic endpoints stay nested objects or become plain
strings, because the answer changes retrieval:

- If `source`/`target` become strings, generic extraction loses the structured
  endpoint description, and schema validation can no longer enforce entity
  granularity for endpoints (`additionalProperties: false` on the endpoint
  object is the only mechanical enforcement that exists; `minItems: 1` on
  `rel_type` is the other). Entity granularity then lives only in the prompt and
  the `entities` array. Note the parser tolerates plain strings but
  `_extract_generic_relationships()` drops any item whose endpoints do not parse,
  so a string endpoint must be handled there too.
- If endpoints stay nested objects, name the winner when the endpoint
  description and the `entities` entry description differ for the same name.
  Today there is only one source, so the question does not arise:
  `_collect_generic_entities()` takes the longest endpoint description and that
  becomes the entity description. Adding an independent `entities` array creates
  a second source, and without a stated rule the two coexist and the ingestion
  loop's choice decides retrieval. The fix branch makes the entities array the
  winner and the endpoint description a repair fallback only (section 1.1);
  section 4 already states that endpoints refer to names in the `entities` array,
  which is the same rule.

Whatever is chosen, the gleaning prompts must change with the schema. They
currently say "Output requirements: 1. Return structured JSON with one object:
'relationships'." and "Do not emit an entities section." Leaving them as-is
contradicts the new contract in the same request.

## 5. Prompt Shape

`_EXTRACTION_SYSTEM_PROMPT` opens "You are a Knowledge Graph Specialist
responsible for extracting relationships from input text." and still instructs
"Treat relationships as undirected unless the text clearly indicates direction."
The directed behavior merged in PR #8 lives in the code-domain prompt
(`_CODE_TAXONOMY_SYSTEM_PROMPT`) and in parsing, not here. Section 10 asserts
directed survival, so this line is part of the change: rewrite the role line and
replace the undirected default with the direction-preservation rule below.

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
7. Keep the current ingestion loops. `chunk_processor.py` already runs
   `IncrementalEntityResolver` over `extraction.entities` before the relationship
   loop, so a non-empty independent array reaches the resolver without new
   wiring. Two caveats: the set is non-empty today only because of endpoint
   derivation, and every member has `entity_type="UNKNOWN"`, which disables the
   resolver's type guard for generic content (see section 1.1). Descriptions do
   survive today, so switching the source of the array must not regress them.
8. Preserve directed relationship resolution from PR #8.
9. Persist and re-hydrate the independent array. `_save_raw_extraction()` writes
   `entities_json` and `_get_raw_extraction()` rebuilds `ExtractedEntity` from it
   while collapsing `entity_type` to `UNKNOWN`. Both must carry the new fields,
   or a cache hit and a fresh extraction produce different graphs for the same
   chunk.
10. Require a description on every extracted entity. An entity whose description
    is empty gets no `EntityDescription` row and no embedding
    (`entity_resolver.resolve_entity` returns early on `not description`), and
    entities without descriptions are dropped when the answer context is
    assembled (`graph_rag.py` keeps only described entities, capped at 10
    entities and 4 descriptions each). A standalone entity without a description
    is invisible to retrieval, which is the exact outcome this plan is trying to
    avoid. The prompt must make descriptions mandatory, and the parser should
    reject or fill empty ones rather than persist silent placeholders.
11. Keep the repair path described in section 1.1 correct under the new contract.
    Endpoint descriptions now survive parsing, the cache round-trip, and the
    gleaning merge, and the ingestion repair path uses them instead of an empty
    string. That fixes today's invisible-node case; it does not remove the need
    for item 4, because a description cannot repair a name that should have
    matched an existing inventory entry.

The code-domain schema and fixed taxonomy remain endpoint-derived for now. Code
can adopt the independent contract later only if code-specific fixtures show a
real retrieval benefit. Section 11 records that no such measurement capability
exists yet.

### 6.1 Resolver consequences, which this plan currently assumes away

Raising entity recall moves more abstract compound concepts through
`IncrementalEntityResolver`, a deliberately zero-LLM resolver:

- name and alias lookup are exact string matches against a raw-string unique
  constraint on `canonical_name`;
- centroid similarity merges above 0.8 with no name check, merges between 0.65
  and 0.8 only when a `difflib` name ratio clears 0.8, and otherwise creates a
  new entity;
- the entity-type guard does no work for generic content, because the generic
  extractor types everything `UNKNOWN` (see section 1.1).

Two failure modes follow, and both get worse with the change as planned:

- casing, hyphenation, and article or preposition variants of a long compound
  concept ("Non-Attachment To Results" against "non attachment to results") miss
  the exact-match steps, and the long-name ratio dilution misses the fuzzy gate,
  so each variant becomes a canonical node;
- the similarity gate keys on `"{name}: {description}"`, so the same concept
  written with different descriptions can fall below the threshold and duplicate,
  while near-antonyms such as `Attachment` and `Non-Attachment` can clear it and
  merge into one node without any name check.

Recall without resolution quality converts lost concepts into a fragmented
vocabulary and diluted centroids. The plan should either bound this (a
before/after measure of entities per chunk and duplicate-name rate on a fixed
corpus, as part of the fixtures in section 11) or state explicitly that a
resolution pass is a prerequisite for accepting higher recall.

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

Three corrections:

- "one LLM call" is only the extractor. Per chunk today: first-pass extraction,
  plus one gleaning pass when the first pass is non-empty, each with one retry on
  failure. Per document, domain classification adds one call when the payload has
  no domain. And per relationship, `run_enhance_relationships` is dispatched as a
  separate job that makes an LLM call and an embedding call per relationship.
  Report call counts against that whole picture, not just the extractor.
- Changing the default is not a one-line edit. `Collection.gleaning_passes` is
  `default=1, server_default="1"` and the API request model also defaults to 1.
  If only the Python default changes, rows inserted without an explicit value
  still receive 1 from the server default, so collections would keep two calls
  while the code claims otherwise. Change the column default and the server
  default together.
- Ship this as a separate change from the schema work. It changes cost and
  retrieval quality at the same time, and bundling it makes any quality movement
  after the contract change unattributable.

## 8. Extraction Contract Version

The prompt/schema change must not silently reuse relationship-only cached
extractions.

Add a contract key such as:

```text
generic-independent-entities-v1
```

and include it in raw extraction cache identity.

This is not implementable as a cache-key edit, and it is load-bearing: without
it, this plan's new behavior is silently bypassed for every chunk that has ever
been ingested.

Verified against `main`:

- `RawChunkExtraction` has no version field. `gleaning_passes` exists on the
  model and is never written or read. `extraction_model` is already occupied
  with the string `domain:{domain}` and is part of the lookup predicate, which is
  why the code path and generic path get separate rows.
- The table's unique constraint is `(chunk_content_hash, collection_id)`. There
  can only ever be one cached row per chunk and collection, so old and new
  contracts cannot coexist.
- `_get_raw_extraction()` selects on hash, collection, and domain only. After the
  contract change, re-ingesting an already-seen chunk returns the
  relationship-only row with `entities_json = []`. The plan would then create
  `UNKNOWN` entities from those stale rows instead of the restored ones, on
  exactly the retry path section 10 claims to verify.
- `_save_raw_extraction()` commits inside `except Exception: rollback`. Writing a
  second row for the same key raises the unique violation and is swallowed with no
  log, so the failure is invisible.
- The lookup uses `scalar_one_or_none()`, which raises `MultipleResultsFound` if
  duplicate rows ever appear for the key it reads.

Shape of the fix, implemented on `fix/extraction-contract-and-endpoint-descriptions`
(migration `0028_add_raw_extraction_contract`) so this plan inherits a working
cache:

1. Add a non-null `extraction_contract` column, backfilled from `extraction_model`
   so existing rows resolve under the contract that produced them
   (`code-taxonomy-v0` for `domain:code` rows, `generic-endpoints-v0` otherwise).
   Backfilling from `extraction_model` matters: the alternative, a single blanket
   default, would make every cached code-domain chunk look like a miss and force a
   full LLM re-extraction on the next ingest.
2. Replace `uq_raw_chunk_extractions_hash_collection` with
   `(chunk_content_hash, collection_id, extraction_contract)`.
3. Treat "row exists under a different contract" as a miss, not an error, and log
   the contracts that are cached when it happens. The values live in
   `services/graph_rag/contracts.py`, whose docstring states the bump procedure.
4. Read all cached payloads for the chunk in one query and select the matching
   contract in Python, rather than `scalar_one_or_none()` plus a second
   contract-mismatch query. The result set is bounded by the number of contracts
   ever shipped, and the first-ingestion path stays a single read. A failed cache
   write is logged instead of silently rolled back.

Existing cached payloads then stay readable under their own contract, and no
historical LLM re-extraction occurs unless that content is explicitly ingested
under the new contract. Bumping the contract is now a one-line change in
`contracts.py`; this plan needs to do exactly that.

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

### Retrieval facts to verify against, not assume

Query-time seeds in `custom_graph_rag` are exactly: an exact n-gram mention index
over `canonical_name` and `EntityAlias`, entity description embeddings
(embedded as `"{name}: {description}"`), relationship description embeddings, and
an alias `ILIKE` keyword pass. Entity centroids and chunk embeddings are not
searched by this query path at all. Consequences worth encoding as tests:

- degree-0 entities are valid seeds; nothing in the path filters on degree, and
  traversal tolerates zero edges. This is the property section 3 depends on.
- an entity with an empty description is neither searchable nor renderable in
  context, so section 6 item 10 is a retrieval requirement, not hygiene.
- `entity-first` (the alias for `local`) cannot enter the graph through a
  relationship description; relationship vectors only re-rank entity seeds there.
  The default `mix` mode can, through its relationship-first leg. So "a concept in
  a relationship description is reachable" is mode-dependent, and the fixture in
  section 10 must state the mode.
- relationship-first can never surface a degree-0 entity, since its discovered set
  starts from relationship endpoints. An independent concept is therefore
  unreachable in that mode no matter how good extraction gets; only the entity
  seed paths can surface it.

## 11. Measurement

This plan is the seventeenth consecutive tuning change to extraction or retrieval
heuristics in the history since 2026-06-21, and there is no way to measure any of
them. There is no evaluation harness, no golden question set, and no recall
measurement in the repository; `tests/test_services/test_graph_rag_aliases.py`
drives the extractor with a fake LLM and asserts output shapes, not outcomes. The
plan's own standard for adopting the contract in the code domain, "only if
code-specific fixtures show a real retrieval benefit", is a standard the
repository cannot currently apply.

Before or alongside the first implementation commit, add a small evaluation
harness:

- one pinned corpus (a set of prose chapters plus, separately, a small code
  repository), checked in;
- 25 to 40 questions typed by intent: entity fact, thematic or concept question,
  multi-hop, negation;
- a runner reporting recall at k of gold entities, relationships, and chunks per
  query mode, plus a cheap model-judged answer score;
- recorded LLM responses replayed by default, so the run is deterministic,
  network-free, and cheap enough to run per change. Note that the current suite
  needs the Docker stack (Redis on 6380, FalkorDB), so a recorded replay is also
  the only form that runs in an environment without it.

Baseline `main` before changing extraction, then treat the plan's acceptance
criteria as measured deltas rather than shape assertions.

## 12. Delivery

Implement as two change sets, in this order, with the measurement harness landing
before or with the first commit:

1. add the pinned corpus, typed question set, and replay-based retrieval runner,
   and record the `main` baseline;
2. characterize the pre-`1320886` generic schema and add regression fixtures;
3. restore independent generic entity extraction in the structured schema and the
   prompt, including the role line, the undirected default in section 5, the
   generic-local item schema in section 4.3, and the gleaning prompt lines that
   currently forbid an entities section;
4. persist descriptions and types end to end, including `entities_json`
   round-tripping and the empty-description rule in section 6;
5. add endpoint-inventory validation on normalized names and a repair-rate log
   line;
6. add ingestion-to-entity-query coverage for an entity that is not a relationship
   endpoint, per query mode, and the duplicate-name and entities-per-chunk measures
   from section 6.1;
7. compare against the baseline from step 1 before merging.

Separate change set, landed on its own so quality movement is attributable:

8. make gleaning opt-in for new collections, changing the Python default, the
   server default, and the API default together.

Already on `fix/extraction-contract-and-endpoint-descriptions`, as preconditions:

- the contract-versioned raw extraction cache in section 8. Required whether or not
  this plan proceeds: without it, every already-ingested chunk silently keeps
  serving the payload written under the previous contract, so this plan's new code
  path never runs for existing content.
- endpoint descriptions carried through parsing, the cache round-trip, and the
  gleaning merge, plus the repair path using one instead of an empty string, and a
  warning log there. Only meaningful once this plan lands; harmless before it,
  since the path is unreachable while entities are endpoint-derived.

## 13. Non-Goals

- a second entity or relationship LLM pass;
- formal rule extraction or compilation;
- temporal relationship identity;
- reifying every relationship as an entity;
- connecting every extracted entity pair;
- using derived co-activation links as asserted facts;
- changing code-domain extraction without dedicated evidence.
