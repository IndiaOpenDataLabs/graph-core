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

## Score semantics

- **Identity:** embeddings propose candidates; cross-name reuse requires the decision model's `same` probability ≥ 0.95. Cache hits and title-cased names cannot authorize merges.
- **Confidence:** native probability that the source supports the claim, in 0–1 units. Source acceptance uses SystemOne's `supported` decision, not a probability cutoff. `contradicted` and `uncertain` claims are excluded. Original passages and endpoint names are retained.
- **Support count:** distinct `(document_id, chunk_hash)` passages independently classified as `supported` by SystemOne. Reprocessing does not increment this count. Graph projection weight is confidence × 100, never a passage count.
- **Relevance:** assessed only at query time against the question, passage by passage. SystemOne's `direct` and `contextual` decisions are accepted; `irrelevant` is excluded. Probabilities rank accepted passages, with no relevance cutoff. Embeddings propose bounded candidates without similarity or endpoint-score rejection. Unrelated descriptions and unscored induced edges cannot enter final custom Graph RAG context. Decision traces are stored in query job results as `relevance_scores`.

An unavailable decision model or invalid native probabilities fail the operation explicitly; there is no generative scoring fallback. Long accumulated evidence can require a larger llama.cpp context window.

Existing graph rows are not automatically assigned confidence, and incorrect aliases/endpoints are not automatically rewritten. Unassessed descriptions are excluded from custom Graph RAG context. For validation, start with a fresh collection and ingest the original sources; copying old weights into confidence would preserve the original problem.
