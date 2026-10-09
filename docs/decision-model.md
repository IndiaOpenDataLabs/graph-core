# Decision model

Text-generating LLMs extract entities, descriptions, predicates and keywords, generate query variants, and write answers. They do not generate weights or confidence scores. Entity identity, source support and custom Graph RAG query relevance use the decision model's native `POST /v1/systemone` API—not chat completions.

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

Custom Graph RAG ingestion plans decisions per chunk, not per entity:

1. Propose identity pairs from aliases, bounded embedding candidates and fuzzy names. Deduplicate pairs across these paths; include earlier entities from the same chunk. Score multiple pairs jointly with the source passage supplied once per request.
2. Apply identity decisions and persist canonical entity/relationship identities in short transactions.
3. Jointly assess entity descriptions, individual relationship passages, accumulated descriptions and relationship-level support. Shared original passages are stored once in each request and referenced by claim IDs. Passage support and aggregate confidence remain separate questions, not separate forward passes.
4. Persist scores after checking the evidence snapshot under a write lock. If another chunk added evidence during inference, re-score the changed group outside the lock. Deadlock retries repeat only the rolled-back write, not decision inference or previously committed entities.

LightRAG batches its relationship support assessments too. Single-entity/relationship resolver APIs remain available for other callers, but chunk ingestion uses the bulk APIs. Query-time relevance remains a separate task.

`DECISION_MODEL_BATCH_TOKEN_BUDGET` defaults to **6000** estimated input tokens, leaving head/template overhead for an 8K server context. The estimate uses `cl100k_base`, not Clef's tokenizer: tune it conservatively for your model and source language. `DECISION_MODEL_BATCH_MAX_QUESTIONS` defaults to **64**. Both limits split requests without dropping questions or truncating source evidence. If even one item exceeds the budget, the job fails explicitly; reduce chunk size or increase the budget only when the server context allows it. Long accumulated evidence may also require a larger context.

The global concurrency limit remains **1** by default: each slot now processes many decisions, rather than repeatedly evaluating the same source for one decision. `decision_batch` logs report task, question count, estimated tokens and request duration without logging source contents.

## Score semantics

- **Identity:** embeddings propose candidates; cross-name reuse requires the decision model's `same` probability ≥ 0.95. Cache hits and title-cased names cannot authorize merges.
- **Confidence:** native probability that the source supports the claim, in 0–1 units. Source acceptance uses SystemOne's `supported` decision, not a probability cutoff. `contradicted` and `uncertain` claims are excluded. Original passages and endpoint names are retained.
- **Support count:** distinct `(document_id, chunk_hash)` passages independently classified as `supported` by SystemOne. Reprocessing does not increment this count. Graph projection weight is confidence × 100, never a passage count.
- **Relevance:** assessed only at query time against the question, passage by passage. SystemOne's `direct` and `contextual` decisions are accepted; `irrelevant` is excluded. Probabilities rank accepted passages, with no relevance cutoff. Embeddings propose bounded candidates without similarity or endpoint-score rejection. Unrelated descriptions and unscored induced edges cannot enter final custom Graph RAG context. Decision traces are stored in query job results as `relevance_scores`.

An unavailable decision model or invalid native probabilities fail the operation explicitly; there is no generative scoring fallback. Long accumulated evidence can require a larger llama.cpp context window.

Existing graph rows are not automatically assigned confidence, and incorrect aliases/endpoints are not automatically rewritten. Unassessed descriptions are excluded from custom Graph RAG context. For validation, start with a fresh collection and ingest the original sources; copying old weights into confidence would preserve the original problem.
