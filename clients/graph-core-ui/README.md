# Graph Core web workspace

Vite + React + TypeScript client for the existing REST API. The TUI remains available.

## Start locally

Use Node 22.12+ (22 LTS) or Node 24 LTS.

From the repository root:

```bash
make docker-up          # API, worker, Postgres, Redis, FalkorDB
make ui-install         # npm ci from the checked-in lockfile
make ui                 # http://localhost:5173
uv run graph-core-admin-jwt
```

Select Admin and paste the admin JWT into the connection form. Admin login does
not require a namespace ID. After signing in, select an existing namespace or
create one. Opening an existing namespace mints a one-day scoped user JWT.

For a namespace user JWT, both the token and namespace ID are required. The UI
verifies the ID against the namespace returned by the API; a token cannot connect
to another namespace. User tokens cannot list other namespaces.

The UI talks to `/api` via Vite's development proxy. By default this forwards to
`http://127.0.0.1:8001`. For another backend:

```bash
GRAPH_CORE_API_URL=http://127.0.0.1:9001 make ui
```

Or place `GRAPH_CORE_API_URL` in `.env.local` here. This variable is server-side
proxy configuration, not a client-side secret. No CORS changes are needed.

## Workflow

1. **Model profiles**: register a provider credential (optional for local providers),
   then create an embedding or LLM profile. Embedding profiles require dimensions.
   You can paste an existing credential ID, or use the ID filled in by registration.
   Provider/model fields accept backend-supported names such as `openai`,
   `local_hash` (embedding), and `local_echo` (LLM).
2. **Collections**: bind profiles and choose a retrieval strategy. Leaving profile
   selections blank uses server defaults. Strategies are discovered from the API.
3. **Add data**: paste text or load a UTF-8 text/Markdown file (up to 10 MB locally).
   PDF/office parsing is not supported. Single chunks ingest synchronously;
   documents queue background jobs with progress polling.
4. **Query explorer**: select a collection, ask a question, optionally override the
   retrieval mode or answer-model profile. The UI polls the durable job until it
   completes; worker/provider failures are shown without fabricating an answer.
5. **Activity**: see the original question first for each query job, followed by
   collection details and status, and reopen completed query results.

Switching namespace clears the workspace. Switching tabs or collections stops
local polling but does **not** cancel an already-enqueued backend job. Activity
can reopen completed results.

## Retrieval inspection

New query jobs store an additive `retrieval_context` field in `payload.result` and
return it through `GET /jobs/{id}/result`:

- Vector and LightRAG naive: retrieved chunk text.
- Custom graph RAG: assembled graph context, including derived understanding
  when selected by retrieval.
- LightRAG local/global: budgeted entities, relationships, and source text.
- LightRAG hybrid/mix: labelled contexts from each retrieval path, not a claim
  that all paths were used in one final generation prompt.

Answers remain separate from context. Entity/relationship identifiers and raw
result JSON are also inspectable. Historical jobs without `retrieval_context`
show an explanatory empty state; no backfill or schema migration is needed.
Retrieval content is stored with the job, so it may increase payload size and
contains the same sensitive source material as the collection.

## Checks

```bash
make ui-test
make ui-build
uv run pytest tests/test_services/test_query_retrieval_context.py -q
```

## Security and deployment

- JWTs and the active namespace are saved in this tab's `sessionStorage`, so a
  page refresh restores the connection. They are not saved in `localStorage` or
  URLs. Sign out clears saved credentials; disconnecting clears the namespace
  connection while retaining admin login for namespace switching. Closing the
  tab ends the browser session (browser session-restore behavior may vary).
  Treat same-origin scripts as trusted: session storage is readable by them.
  JWT expiry still applies. If browser storage is blocked, login works in memory
  and the sidebar warns that refresh will require signing in.
- Provider secrets are sent to the API for encrypted registration; they are
  cleared from the form after success and are not stored by the browser client.
- Text, answers, and raw results are rendered as escaped text, not HTML.
- The dev/preview servers bind loopback only. They are local-development tools,
  not production authentication gateways.
- `npm run build` writes `dist/`. For production, serve those static files with
  SPA fallback and configure a same-origin reverse proxy from `/api/` to Graph
  Core (strip `/api`). `GRAPH_CORE_API_URL` only configures Vite dev/preview.
- Use TLS and enforce backend authorization before exposing a deployment.
  Existing job detail/result REST routes currently do not enforce namespace
  authentication; do not expose them publicly without addressing this backend
  limitation. The UI always sends a scoped bearer token, but cannot enforce
  server-side isolation itself.
