# Decision model

Text-generating LLMs extract entities, descriptions, predicates and keywords, generate query variants, and write answers. They do not generate weights or confidence scores. Entity identity and custom Graph RAG query relevance use the decision model's native `POST /v1/systemone` API—not chat completions. Ingestion does **not** revalidate extracted descriptions or relationships against their source passages.

## Run locally

Use a SystemOne-capable llama.cpp build and a compatible decision-model GGUF:

```sh
llama-server --model /path/to/decision-model.gguf --host 0.0.0.0 --port 8081
make docker-up
```

The existing provider URL normalizer selects `http://host.docker.internal:8081/v1/systemone` inside Docker and `http://localhost:8081/v1/systemone` outside Docker. Compose already defines the host gateway. No forwarder or decision-model credential/profile is needed. Restrict host port 8081 to your trusted development network.

The app entrypoint applies migration `0029_decision_scores` when rebuilt/restarted. For a non-Docker deployment, run `uv run alembic upgrade head` against the configured database. If this migration was already applied before the rename, its recorded revision must be updated to `0029_decision_scores` without reapplying the schema changes.

## Concurrency

SystemOne calls use the existing Redis semaphore infrastructure with a separate global decision-model pool shared by API and workers. `DECISION_MODEL_MAX_CONCURRENT_CALLS` defaults to `1` and must be positive. The existing semaphore lease, wait timeout and cancellation-safe release apply. This pool is independent of LLM and embedding limits; no Compose changes are required.

## Batched ingestion

Custom Graph RAG ingestion plans identity decisions per chunk, not per entity:

1. Propose identity pairs from aliases, bounded embedding candidates and fuzzy names. Deduplicate pairs across these paths; include earlier entities from the same chunk. Score multiple pairs jointly with the source passage supplied once per request.
2. Apply identity decisions and persist canonical entity/relationship identities in short transactions.
3. Persist extracted descriptions and their original passages, document references, chunk hashes and relationship endpoint names. New descriptions are embedded without a source-support acceptance gate.
4. Check the description snapshot under a write lock. If another chunk added evidence while embeddings were prepared, merge against a fresh snapshot rather than overwriting it. Deadlock retries repeat only the rolled-back write, not identity inference or previously committed entities.

LightRAG relationship ingestion also persists descriptions and provenance without source-support assessments. Single-entity/relationship resolver APIs remain available for other callers, but chunk ingestion uses the bulk APIs. Query-time relevance remains a separate task.

`DECISION_MODEL_BATCH_TOKEN_BUDGET` defaults to **6000** estimated input tokens. The estimate uses `cl100k_base`, not Clef's tokenizer: tune it conservatively for your model, source language, server context and physical batch capacity. `DECISION_MODEL_BATCH_MAX_QUESTIONS` defaults to **64**. Both limits split identity and query-relevance requests without dropping questions or truncating source passages/descriptions. If even one item exceeds the budget, the job fails explicitly; reduce chunk size or increase the budget only when the server allows it.

The global concurrency limit remains **1** by default: each slot processes many decisions. `decision_batch` logs report task, question count, estimated tokens and request duration without logging source contents.

## Staged query selection

1. **Node gate:** native `noul`; keep P(relevant) **> 0.50**. Mix navigation candidates are hydrated from actual stored entity descriptions (including exact-name hits), rather than labels or embedding snippets. Accepted nodes are navigation starting points, not necessarily complete answer passages. All passing nodes survive; there is no additional top-eight acceptance cap.
2. **Edge ranking:** gather the complete incident-edge pool around accepted mix anchors, plus existing retrieved relationship candidates. Native `score` uses four ordered levels (0 unrelated, 1 tangential, 2 useful context, 3 directly useful). Rank by the returned expected score and shortlist **40** distinct edges. Stored structural weights are neither relevance scores nor overwritten.
3. **Edge gate:** native `noul` on just those shortlisted edges; keep P(relevant) **> 0.50**. Include each edge's source, predicate, target, actual extracted description and directly connected accepted anchors. Do not backfill rejected edges to force 40 survivors.
4. **Context assembly:** format accepted nodes and surviving edges without another description-level relevance call or the old 10-node/20-edge context caps. One highest-observation-weight description per node/edge is used, with deterministic description-ID tie-breaking and document scope respected. Provenance accompanies the decision traces. Derived understanding, if supplied, is assembled without an additional classifier.

Mix node gating runs before anchor expansion; other retrieval modes gate their retrieved node pool during context assembly before edge ranking. Document routing, vector retrieval and subquery generation remain in place. Unassessed descriptions remain eligible; historical ingestion support does not gate query candidates.

`DECISION_MODEL_QUERY_BATCH_TOKEN_BUDGET` defaults to **2500**, the conservative budget used in the edge experiment. Staged node/edge requests use the smaller of this and the global token budget, plus the existing question-count limit. This remains a `cl100k_base` estimate, not a guarantee about the server's physical batch capacity. Increase only after checking your server limits. Batch composition can affect Clef probabilities and cross-batch ranking; the 0.50 gates are initial policy, not an accuracy guarantee. Oversized singleton evidence still fails explicitly rather than being truncated.

The native client supports typed `choice`, `noul` and `score` answers and request-level `instructions`. It validates boolean probabilities and ordinal score distributions/expected values. A `score` is not a joint distribution over edges; no sum-to-one normalization across edges is needed.

## Score semantics

- **Identity:** embeddings propose candidates; cross-name reuse requires the decision model's `same` probability ≥ 0.95. Cache hits and title-cased names cannot authorize merges.
- **Source confidence and verified support count:** not computed during ingestion. New/updated description assessments and relationship confidence/support-count fields are left unset, rather than fabricating probabilities or declaring every extraction `supported`.
- **Observations:** description weight counts distinct `(document_id, chunk_hash)` source observations. Relationship metadata records their union as `source_count`; reprocessing the same observation does not increment it. This is provenance, not verified support. Relationship graph projection weight is a neutral structural **1**, not a confidence score.
- **Relevance:** query-specific staged node/edge decisions against the original question, as described above. Legacy `direct`/`contextual`/`irrelevant` classification remains available for document routing, but is no longer the final context filter. Decision traces (`query_nodes`, `query_edge_scores`, `query_edge_gate`) include native schemas, probabilities, expected scores, provenance, shortlisting and inclusion status in query results as `relevance_scores`.

An unavailable decision model or invalid native probabilities fail identity/relevance operations explicitly; there is no generative scoring fallback. Description persistence does not call the decision model.

Existing graph rows are not automatically assigned confidence, re-embedded, or revalidated, and incorrect aliases/endpoints are not automatically rewritten. Existing descriptions are eligible for query relevance selection regardless of any historical support assessment. No data migration or ingestion restart is required for the code change itself.
