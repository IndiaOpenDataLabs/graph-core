"""MCP server for Graph Core — exposes platform operations as tools.

The admin and user surfaces run as separate MCP servers on different ports.
"""

import json
import os
from contextlib import asynccontextmanager

from fastapi import HTTPException
from mcp.server.fastmcp import Context, FastMCP
from mcp.types import CallToolResult, TextContent
from starlette.applications import Starlette
from starlette.routing import Mount

from graph_core.api.auth import resolve_bearer_identity
from graph_core.client import GraphCoreAPIError, GraphCoreClient


def _get_base_url() -> str:
    return os.getenv("GRAPH_CORE_URL", "http://localhost:8001").rstrip("/")


def _extract_api_key(ctx: Context) -> str:
    """Extract the API key from the incoming MCP request.

    Checks in order:
    1. MCP protocol meta ({"api_key": "..."}) — most reliable
    2. Authorization: Bearer <key> header
    3. X-API-Key header
    """
    # 1. MCP protocol metadata (passed through call_tool meta param)
    try:
        meta = ctx.request_context.meta
        if meta and getattr(meta, "api_key", None):
            return meta.api_key
    except AttributeError:
        pass

    # 2. HTTP request headers
    try:
        request = ctx.request_context.request
        if request is not None:
            auth_header = request.headers.get("authorization", "")
            if auth_header.startswith("Bearer "):
                return auth_header[7:]
            api_key = request.headers.get("x-api-key", "")
            if api_key:
                return api_key
    except AttributeError:
        pass

    raise GraphCoreAPIError(
        "No bearer token found in request metadata or headers."
    )


@asynccontextmanager
async def _client(api_key: str, admin: bool = False):
    """Create a scoped HTTP client, closing it on exit."""
    c = GraphCoreClient(
        base_url=_get_base_url(),
        api_key=api_key,
        is_admin=admin,
    )
    try:
        yield c
    finally:
        await c.close()


admin_mcp = FastMCP(
    name="graph-core-admin",
    instructions="MCP admin tools for namespace management",
    streamable_http_path="/",
)

user_mcp = FastMCP(
    name="graph-core-user",
    instructions="MCP user tools scoped to a namespace",
    streamable_http_path="/",
)

admin_tool = admin_mcp.tool
user_tool = user_mcp.tool


def _header_lookup(scope: dict, name: bytes) -> str:
    for key, value in scope.get("headers", []):
        if key.lower() == name:
            return value.decode("latin-1")
    return ""


def _result(text: str, structured: dict) -> CallToolResult:
    return CallToolResult(
        content=[TextContent(type="text", text=text)],
        structuredContent=structured,
    )


async def _send_http_error(send, status_code: int, detail: str) -> None:
    body = detail.encode("utf-8")
    await send(
        {
            "type": "http.response.start",
            "status": status_code,
            "headers": [
                (b"content-type", b"text/plain; charset=utf-8"),
                (b"content-length", str(len(body)).encode()),
            ],
        }
    )
    await send({"type": "http.response.body", "body": body})


async def _send_jsonrpc_error(send, status_code: int, code: int, message: str) -> None:
    payload = {
        "jsonrpc": "2.0",
        "error": {
            "code": code,
            "message": message,
        },
        "id": None,
    }
    body = json.dumps(payload).encode("utf-8")
    await send(
        {
            "type": "http.response.start",
            "status": status_code,
            "headers": [
                (b"content-type", b"application/json"),
                (b"content-length", str(len(body)).encode()),
            ],
        }
    )
    await send({"type": "http.response.body", "body": body})


class TokenScopedMCPApp:
    """Wrap a FastMCP app and require a specific bearer-token kind."""

    def __init__(self, *, app, required_kind: str) -> None:
        self._app = app
        self._required_kind = required_kind

    async def __call__(self, scope, receive, send) -> None:
        if scope["type"] != "http":
            await self._app(scope, receive, send)
            return

        authorization = _header_lookup(scope, b"authorization")
        try:
            if not authorization:
                raise HTTPException(status_code=401, detail="Authorization header required")
            identity = resolve_bearer_identity(authorization)
            if identity.kind != self._required_kind:
                raise HTTPException(
                    status_code=403,
                    detail=f"{self._required_kind.title()} token required",
                )
        except HTTPException as exc:
            await _send_jsonrpc_error(
                send,
                exc.status_code,
                code=-32001 if exc.status_code == 401 else -32003,
                message=str(exc.detail),
            )
            return

        await self._app(scope, receive, send)


class MCPPathAliasApp:
    """Rewrite `/mcp` and `/mcp/` to the MCP app root."""

    def __init__(self, app) -> None:
        self._app = app

    async def __call__(self, scope, receive, send) -> None:
        path = scope.get("path", "")
        if path in {"", "/", "/mcp", "/mcp/"}:
            aliased_scope = dict(scope)
            aliased_scope["path"] = "/"
            await self._app(aliased_scope, receive, send)
            return
        await _send_http_error(send, 404, "Not Found")


_admin_server_app = TokenScopedMCPApp(
    app=admin_mcp.streamable_http_app(),
    required_kind="admin",
)

_user_server_app = TokenScopedMCPApp(
    app=user_mcp.streamable_http_app(),
    required_kind="user",
)


@asynccontextmanager
async def _admin_lifespan(app: Starlette):
    del app
    async with admin_mcp.session_manager.run():
        yield


@asynccontextmanager
async def _user_lifespan(app: Starlette):
    del app
    async with user_mcp.session_manager.run():
        yield


# -- Namespace tools --------------------------------------------------------


@admin_tool()
async def create_namespace(name: str, ctx: Context) -> CallToolResult:
    """Create a new namespace. Requires admin JWT.

    Args:
        name: Human-readable namespace name (must be unique).
    """
    api_key = _extract_api_key(ctx)
    async with _client(api_key, admin=True) as client:
        result = await client.create_namespace(name)
        text = (
            f"Created namespace:\n"
            f"  id: {result['id']}\n"
            f"  name: {result['name']}\n"
            f"  token_type: {result['token_type']}\n"
            f"  scope: {result['scope']}\n"
            f"  token: {result['token']}\n"
            f"  expires_at: {result['expires_at']}"
        )
    return CallToolResult(
        content=[TextContent(type="text", text=text)],
        structuredContent={
            "namespace": {
                "id": result["id"],
                "name": result["name"],
                "token_type": result["token_type"],
                "scope": result["scope"],
                "token": result["token"],
                "expires_at": result["expires_at"],
            }
        },
    )


@admin_tool()
async def list_namespaces(ctx: Context) -> CallToolResult:
    """List all namespaces. Requires admin JWT."""
    api_key = _extract_api_key(ctx)
    async with _client(api_key, admin=True) as client:
        namespaces = await client.list_namespaces()
        if not namespaces:
            text = "No namespaces found."
            return CallToolResult(
                content=[TextContent(type="text", text=text)],
                structuredContent={"namespaces": []},
            )

        lines = ["Namespaces:"]
        items: list[dict[str, str]] = []
        for ns in namespaces:
            text_line = f"  - {ns['id']} | {ns['name']}"
            lines.append(text_line)
            items.append(
                {
                    "id": ns["id"],
                    "name": ns["name"],
                }
            )
        text = "\n".join(lines)
    return CallToolResult(
        content=[TextContent(type="text", text=text)],
        structuredContent={"namespaces": items},
    )


@user_tool()
async def get_current_namespace(ctx: Context) -> CallToolResult:
    """Get info about the current authenticated namespace."""
    api_key = _extract_api_key(ctx)
    async with _client(api_key) as client:
        ns = await client.get_namespace_me()
        text = f"Namespace: {ns['id']} | {ns['name']}"
    return CallToolResult(
        content=[TextContent(type="text", text=text)],
        structuredContent={
            "namespace": {
                "id": ns["id"],
                "name": ns["name"],
            }
        },
    )


@admin_tool()
async def issue_user_token(
    namespace_id: str,
    ctx: Context,
    subject: str | None = None,
    expires_in_days: int = 365,
) -> CallToolResult:
    """Issue a long-lived user JWT for an existing namespace. Requires admin JWT."""
    api_key = _extract_api_key(ctx)
    async with _client(api_key, admin=True) as client:
        result = await client.issue_user_token(
            namespace_id,
            subject=subject,
            expires_in_days=expires_in_days,
        )
        text = (
            f"Issued user token:\n"
            f"  namespace_id: {result['namespace_id']}\n"
            f"  namespace_name: {result['namespace_name']}\n"
            f"  token_type: {result['token_type']}\n"
            f"  scope: {result['scope']}\n"
            f"  token: {result['token']}\n"
            f"  expires_at: {result['expires_at']}"
        )
    return CallToolResult(
        content=[TextContent(type="text", text=text)],
        structuredContent={
            "namespace_id": result["namespace_id"],
            "namespace_name": result["namespace_name"],
            "token_type": result["token_type"],
            "scope": result["scope"],
            "token": result["token"],
            "expires_at": result["expires_at"],
        },
    )


# -- Collection tools -------------------------------------------------------


@user_tool()
async def create_collection(
    name: str,
    strategy: str,
    embedding_profile_id: str,
    ctx: Context,
    llm_profile_id: str | None = None,
    default_query_mode: str | None = None,
    gleaning_passes: int | None = None,
) -> CallToolResult:
    """Create a new collection in the current namespace.

    Args:
        name: Collection name (unique within namespace).
        strategy: Retrieval strategy: 'vector', 'light_rag', or 'custom_graph_rag'.
        embedding_profile_id: Required embedding profile UUID.
        llm_profile_id: Optional LLM profile UUID.
        default_query_mode: Optional default query mode.
        gleaning_passes: Optional number of extra gleaning passes per chunk.
    """
    api_key = _extract_api_key(ctx)
    async with _client(api_key) as client:
        result = await client.create_collection(
            name=name,
            strategy=strategy,
            embedding_profile_id=embedding_profile_id,
            llm_profile_id=llm_profile_id,
            default_query_mode=default_query_mode,
            gleaning_passes=gleaning_passes,
        )
    return _result(
        (
            f"Created collection:\n"
            f"  id: {result['id']}\n"
            f"  name: {result['name']}\n"
            f"  strategy: {result['strategy']}\n"
            f"  embedding_profile_id: {result.get('embedding_profile_id') or 'N/A'}\n"
            f"  llm_profile_id: {result.get('llm_profile_id') or 'N/A'}\n"
            f"  gleaning_passes: {result.get('gleaning_passes', 1)}"
        ),
        {
            "collection": {
                "id": result["id"],
                "name": result["name"],
                "strategy": result["strategy"],
                "embedding_profile_id": result.get("embedding_profile_id"),
                "llm_profile_id": result.get("llm_profile_id"),
                "gleaning_passes": result.get("gleaning_passes", 1),
            }
        },
    )


@user_tool()
async def list_collections(ctx: Context) -> CallToolResult:
    """List all collections in the current namespace."""
    api_key = _extract_api_key(ctx)
    async with _client(api_key) as client:
        collections = await client.list_collections()
        if not collections:
            return CallToolResult(
                content=[TextContent(type="text", text="No collections found.")],
                structuredContent={"collections": []},
            )
        lines = ["Collections:"]
        items: list[dict[str, str | None]] = []
        for col in collections:
            llm_profile_id = col.get("llm_profile_id")
            llm_suffix = f" | llm={llm_profile_id}" if llm_profile_id else ""
            lines.append(
                f"  - {col['id']} | {col['name']} ({col['strategy']}){llm_suffix}"
            )
            items.append(
                {
                    "id": col["id"],
                    "name": col["name"],
                    "strategy": col["strategy"],
                    "llm_profile_id": llm_profile_id,
                }
            )
        text = "\n".join(lines)
    return CallToolResult(
        content=[TextContent(type="text", text=text)],
        structuredContent={"collections": items},
    )


@user_tool()
async def update_collection(
    collection_id: str,
    ctx: Context,
    name: str | None = None,
    strategy: str | None = None,
    embedding_profile_id: str | None = None,
    llm_profile_id: str | None = None,
    default_query_mode: str | None = None,
    gleaning_passes: int | None = None,
    clear_llm_profile: bool = False,
    clear_default_query_mode: bool = False,
) -> CallToolResult:
    """Update a collection in the current namespace."""
    api_key = _extract_api_key(ctx)
    async with _client(api_key) as client:
        result = await client.update_collection(
            collection_id=collection_id,
            name=name,
            strategy=strategy,
            embedding_profile_id=embedding_profile_id,
            llm_profile_id=llm_profile_id,
            default_query_mode=default_query_mode,
            gleaning_passes=gleaning_passes,
            clear_llm_profile=clear_llm_profile,
            clear_default_query_mode=clear_default_query_mode,
        )
        text = (
            f"Updated collection:\n"
            f"  id: {result['id']}\n"
            f"  name: {result['name']}\n"
            f"  strategy: {result['strategy']}\n"
            f"  embedding_profile_id: {result.get('embedding_profile_id') or 'N/A'}\n"
            f"  llm_profile_id: {result.get('llm_profile_id') or 'N/A'}\n"
            f"  gleaning_passes: {result.get('gleaning_passes', 1)}"
        )
    return _result(
        text,
        {
            "collection": {
                "id": result["id"],
                "name": result["name"],
                "strategy": result["strategy"],
                "embedding_profile_id": result.get("embedding_profile_id"),
                "llm_profile_id": result.get("llm_profile_id"),
                "gleaning_passes": result.get("gleaning_passes", 1),
            }
        },
    )


@user_tool()
async def delete_collection(collection_id: str, ctx: Context) -> CallToolResult:
    """Delete a collection in the current namespace."""
    api_key = _extract_api_key(ctx)
    async with _client(api_key) as client:
        result = await client.delete_collection(collection_id)
        deleted_id = result.get("id", collection_id)
    return _result(f"Deleted collection {deleted_id}", {"collection_id": deleted_id})


@user_tool()
async def enhance_collection(
    collection_id: str,
    levels: int = 1,
    ctx: Context | None = None,
) -> CallToolResult:
    """Queue a derived-understanding build for a collection."""
    if ctx is None:
        raise ValueError("Context is required")
    api_key = _extract_api_key(ctx)
    async with _client(api_key) as client:
        result = await client.enhance_collection(collection_id, levels=levels)
        text = (
            f"Enhance queued:\n"
            f"  job_id: {result['job_id']}\n"
            f"  collection_id: {result['collection_id']}\n"
            f"  namespace_id: {result['namespace_id']}\n"
            f"  status: {result['status']}\n"
            f"  type: {result['type']}\n\n"
            f"Poll with get_job_status('{result['job_id']}')"
        )
    return _result(
        text,
        {
            "job_id": result["job_id"],
            "collection_id": result["collection_id"],
            "namespace_id": result["namespace_id"],
            "status": result["status"],
            "type": result["type"],
        },
    )


# -- Ingestion tools --------------------------------------------------------


@user_tool()
async def ingest_chunk(
    collection_id: str,
    text: str,
    ctx: Context,
    domain: str | None = None,
    document_path: str | None = None,
) -> CallToolResult:
    """Ingest a text chunk directly into a collection.

    For large documents, use ingest_document instead (it runs async with a job).

    Args:
        collection_id: The UUID of the target collection.
        text: The text content to ingest.
    """
    api_key = _extract_api_key(ctx)
    async with _client(api_key) as client:
        result = await client.ingest_chunk(
            collection_id,
            text,
            domain=domain,
            document_path=document_path,
        )
    return _result(
        (
            f"Ingested chunk:\n"
            f"  hash: {result.get('chunk_hash', 'N/A')}\n"
            f"  entities: {result.get('entity_count', 0)}\n"
            f"  relationships: {result.get('relationship_count', 0)}"
        ),
        {
            "chunk_hash": result.get("chunk_hash"),
            "entity_count": result.get("entity_count", 0),
            "relationship_count": result.get("relationship_count", 0),
        },
    )


@user_tool()
async def ingest_document(
    collection_id: str,
    text: str,
    ctx: Context,
    domain: str | None = None,
    document_path: str | None = None,
) -> CallToolResult:
    """Ingest a full document into a collection (async, returns job_id).

    For large documents, the platform will chunk and process in the background.

    Args:
        collection_id: The UUID of the target collection.
        text: The full document text.
        document_path: Optional path for stable document identity (enables idempotent re-ingestion).
    """
    api_key = _extract_api_key(ctx)
    async with _client(api_key) as client:
        result = await client.ingest_document(collection_id, text, domain=domain, document_path=document_path)
    return _result(
        (
            f"Document ingestion started:\n"
            f"  job_id: {result['job_id']}\n"
            f"  status: {result['status']}\n\n"
            f"Track with get_job_status('{result['job_id']}')"
        ),
        {"job_id": result["job_id"], "status": result["status"]},
    )


# -- Query tools ------------------------------------------------------------


@user_tool()
async def query_collection(
    collection_id: str,
    question: str,
    ctx: Context,
    mode: str | None = None,
    chat_id: str | None = None,
) -> CallToolResult:
    """Query a collection with a natural language question.

    Args:
        collection_id: The UUID of the collection to query.
        question: The natural language question.
        mode: Query mode for light_rag/custom graph retrieval. Leave empty for default.
        chat_id: Optional chat session UUID for follow-up memory.
    """
    api_key = _extract_api_key(ctx)
    async with _client(api_key) as client:
        result = await client.query_collection(
            collection_id,
            question,
            mode=mode,
            chat_id=chat_id,
        )
        text = (
            f"Query queued:\n"
            f"  job_id: {result['job_id']}\n"
            f"  collection_id: {result['collection_id']}\n"
            f"  namespace_id: {result['namespace_id']}\n"
            f"  status: {result['status']}\n"
            f"  type: {result['type']}\n\n"
            f"Poll with get_job_status('{result['job_id']}')"
        )
    return _result(
        text,
        {
            "job_id": result["job_id"],
            "collection_id": result["collection_id"],
            "namespace_id": result["namespace_id"],
            "status": result["status"],
            "type": result["type"],
        },
    )


@user_tool()
async def create_chat_session(
    collection_id: str,
    ctx: Context,
    title: str | None = None,
) -> CallToolResult:
    """Create a chat session for follow-up query context."""
    api_key = _extract_api_key(ctx)
    async with _client(api_key) as client:
        result = await client.create_chat_session(collection_id, title=title)
    return _result(
        (
            f"Created chat session:\n"
            f"  id: {result['id']}\n"
            f"  collection_id: {result['collection_id']}\n"
            f"  title: {result.get('title') or '-'}\n"
            f"  turn_count: {result.get('turn_count', 0)}"
        ),
        {
            "chat_session": {
                "id": result["id"],
                "collection_id": result["collection_id"],
                "title": result.get("title"),
                "turn_count": result.get("turn_count", 0),
            }
        },
    )


@user_tool()
async def list_chat_sessions(
    collection_id: str,
    ctx: Context,
    limit: int = 20,
) -> CallToolResult:
    """List chat sessions for a collection."""
    api_key = _extract_api_key(ctx)
    async with _client(api_key) as client:
        rows = await client.list_chat_sessions(collection_id, limit=limit)
        if not rows:
            return _result("No chat sessions found.", {"chat_sessions": []})
        lines = ["Chat sessions:"]
        items: list[dict[str, object]] = []
        for row in rows:
            lines.append(
                f"  - {row['id']} | turns={row.get('turn_count', 0)}"
                f" | title={row.get('title') or '-'}"
            )
            items.append(
                {
                    "id": row["id"],
                    "turn_count": row.get("turn_count", 0),
                    "title": row.get("title"),
                }
            )
    return _result("\n".join(lines), {"chat_sessions": items})


# -- Job tools --------------------------------------------------------------


@user_tool()
async def get_job_status(job_id: str, ctx: Context) -> CallToolResult:
    """Check the status of an async ingestion job.

    Args:
        job_id: The UUID of the job.
    """
    api_key = _extract_api_key(ctx)
    async with _client(api_key) as client:
        job = await client.get_job(job_id)
        lines = [
            f"Job: {job.get('id', job_id)}",
            f"  type: {job.get('type', job.get('job_type', 'N/A'))}",
            f"  status: {job.get('status', 'unknown')}",
            f"  progress: {job.get('progress_percent', 0)}%",
        ]
        if job.get("error"):
            lines.append(f"  error: {job['error']}")
        if job.get("chunks_total"):
            lines.append(
                f"  chunks: {job.get('chunks_completed', 0)}/{job['chunks_total']}"
            )
        payload = job.get("payload") or {}
        result = payload.get("result") if isinstance(payload, dict) else None
        if isinstance(result, dict):
            if "response" in result:
                response = str(result.get("response") or "").strip()
                if response:
                    lines.append(f"  response: {response[:240]}")
            if "generated_levels" in result:
                lines.append(
                    f"  generated_levels: {len(result.get('generated_levels') or [])}"
                )
    return _result(
        "\n".join(lines),
        {
            "job": {
                "id": job.get("id", job_id),
                "type": job.get("type", job.get("job_type")),
                "status": job.get("status", "unknown"),
                "progress_percent": job.get("progress_percent", 0),
                "error": job.get("error"),
                "chunks_total": job.get("chunks_total"),
                "chunks_completed": job.get("chunks_completed"),
                "payload": job.get("payload"),
            }
        },
    )


@user_tool()
async def get_job_result(job_id: str, ctx: Context) -> CallToolResult:
    """Get the final result payload for a completed query or enhance job."""
    api_key = _extract_api_key(ctx)
    async with _client(api_key) as client:
        result = await client.get_job_result(job_id)
        payload = result.get("result") or {}
        if result.get("type") == "query":
            text = (
                f"Query result:\n"
                f"  job_id: {result['id']}\n"
                f"  status: {result['status']}\n\n"
                f"{payload.get('response', '')}"
            )
            return _result(
                text,
                {
                    "job_id": result["id"],
                    "status": result["status"],
                    "type": result["type"],
                    "result": payload,
                },
            )
        if result.get("type") == "enhance":
            summary = payload
            generated_levels = summary.get("generated_levels") or []
            text = (
                f"Enhance result:\n"
                f"  job_id: {result['id']}\n"
                f"  status: {result['status']}\n"
                f"  requested_levels: {summary.get('requested_levels', 1)}\n"
                f"  generated_levels: {len(generated_levels)}"
            )
            return _result(
                text,
                {
                    "job_id": result["id"],
                    "status": result["status"],
                    "type": result["type"],
                    "result": summary,
                },
            )
    return _result(
        f"Job result:\n  job_id: {result['id']}\n  status: {result['status']}",
        {
            "job_id": result["id"],
            "status": result["status"],
            "type": result["type"],
            "result": payload,
        },
    )


@user_tool()
async def list_jobs(
    limit: int = 20,
    collection_id: str | None = None,
    ctx: Context = None,
) -> CallToolResult:
    """List recent jobs in the current namespace."""
    api_key = _extract_api_key(ctx)
    async with _client(api_key) as client:
        jobs = await client.list_jobs(limit=limit, collection_id=collection_id)
        if not jobs:
            return _result("No jobs found.", {"jobs": []})
        lines = ["Jobs:"]
        items: list[dict[str, object]] = []
        for job in jobs:
            chunks = ""
            if job.get("chunks_total"):
                chunks = (
                    f" | chunks {job.get('chunks_completed', 0)}/"
                    f"{job['chunks_total']}"
                )
            lines.append(
                f"  - {job['id']} | {job.get('type', 'N/A')} | "
                f"{job.get('status', 'unknown')} | "
                f"{job.get('progress_percent', 0)}%{chunks}"
            )
            items.append(
                {
                    "id": job["id"],
                    "type": job.get("type"),
                    "status": job.get("status"),
                    "progress_percent": job.get("progress_percent", 0),
                    "chunks_total": job.get("chunks_total"),
                    "chunks_completed": job.get("chunks_completed"),
                    "payload": job.get("payload"),
                }
            )
    return _result("\n".join(lines), {"jobs": items})


# -- Platform tools ---------------------------------------------------------


@user_tool()
async def create_embedding_profile(
    provider: str,
    model: str,
    secret: str,
    ctx: Context,
    label: str | None = None,
    base_url: str | None = None,
    dimensions: int | None = None,
    distance_metric: str | None = None,
    max_concurrent_calls: int | None = None,
) -> CallToolResult:
    """Create an embedding profile in the current namespace.

    Args:
        provider: Provider name, e.g. 'openai'.
        model: Embedding model identifier.
        secret: Provider API key or token.
        label: Optional human-readable label.
        base_url: Optional custom API base URL.
        dimensions: Optional embedding dimensions.
        distance_metric: Optional distance metric.
    """
    api_key = _extract_api_key(ctx)
    async with _client(api_key) as client:
        credential = await client.register_credential(
            provider=provider,
            secret=secret,
            label=label,
            base_url=base_url,
        )
        profile = await client.create_profile(
            kind="embedding",
            provider=provider,
            model=model,
            credential_id=credential["credential_id"],
            label=label,
            base_url=base_url,
            dimensions=dimensions,
            distance_metric=distance_metric,
            max_concurrent_calls=max_concurrent_calls,
        )
    return _result(
        (
            f"Created embedding profile:\n"
            f"  profile_id: {profile['profile_id']}\n"
            f"  label: {profile.get('label') or '-'}\n"
            f"  provider: {profile['provider']}\n"
            f"  model: {profile['model']}\n"
            f"  dimensions: {profile.get('dimensions') or '-'}\n"
            f"  max_concurrent_calls: {profile.get('max_concurrent_calls') or '-'}"
        ),
        {
            "profile": {
                "profile_id": profile["profile_id"],
                "label": profile.get("label"),
                "provider": profile["provider"],
                "model": profile["model"],
                "dimensions": profile.get("dimensions"),
                "max_concurrent_calls": profile.get("max_concurrent_calls"),
            }
        },
    )


@user_tool()
async def create_llm_profile(
    provider: str,
    model: str,
    secret: str,
    ctx: Context,
    label: str | None = None,
    base_url: str | None = None,
    max_concurrent_calls: int | None = None,
) -> CallToolResult:
    """Create an LLM profile in the current namespace.

    Args:
        provider: Provider name, e.g. 'openai'.
        model: LLM model identifier.
        secret: Provider API key or token.
        label: Optional human-readable label.
        base_url: Optional custom API base URL.
    """
    api_key = _extract_api_key(ctx)
    async with _client(api_key) as client:
        credential = await client.register_credential(
            provider=provider,
            secret=secret,
            label=label,
            base_url=base_url,
        )
        profile = await client.create_profile(
            kind="llm",
            provider=provider,
            model=model,
            credential_id=credential["credential_id"],
            label=label,
            base_url=base_url,
            max_concurrent_calls=max_concurrent_calls,
        )
    return _result(
        (
            f"Created llm profile:\n"
            f"  profile_id: {profile['profile_id']}\n"
            f"  label: {profile.get('label') or '-'}\n"
            f"  provider: {profile['provider']}\n"
            f"  model: {profile['model']}\n"
            f"  max_concurrent_calls: {profile.get('max_concurrent_calls') or '-'}"
        ),
        {
            "profile": {
                "profile_id": profile["profile_id"],
                "label": profile.get("label"),
                "provider": profile["provider"],
                "model": profile["model"],
                "max_concurrent_calls": profile.get("max_concurrent_calls"),
            }
        },
    )


def _format_profile_list(title: str, profiles: list[dict]) -> tuple[str, list[dict[str, object]]]:
    if not profiles:
        return f"No {title.lower()} found.", []
    lines = [f"{title}:"]
    items: list[dict[str, object]] = []
    for profile in profiles:
        label = profile.get("label") or "-"
        model = profile.get("model") or "-"
        provider = profile.get("provider") or "-"
        limit = profile.get("max_concurrent_calls")
        lines.append(
            f"  - {profile['profile_id']} | {label} | {provider} | {model} | "
            f"max_concurrent_calls={limit if limit is not None else '-'}"
        )
        items.append(
            {
                "profile_id": profile["profile_id"],
                "label": profile.get("label"),
                "provider": profile.get("provider"),
                "model": profile.get("model"),
                "max_concurrent_calls": profile.get("max_concurrent_calls"),
            }
        )
    return "\n".join(lines), items


@user_tool()
async def list_embedding_profiles(ctx: Context) -> CallToolResult:
    """List embedding profiles in the current namespace."""
    api_key = _extract_api_key(ctx)
    async with _client(api_key) as client:
        profiles = await client.list_embedding_profiles()
        text, items = _format_profile_list("Embedding Profiles", profiles)
    return _result(text, {"profiles": items})


@user_tool()
async def list_llm_profiles(ctx: Context) -> CallToolResult:
    """List LLM profiles in the current namespace."""
    api_key = _extract_api_key(ctx)
    async with _client(api_key) as client:
        profiles = await client.list_llm_profiles()
        text, items = _format_profile_list("LLM Profiles", profiles)
    return _result(text, {"profiles": items})


@user_tool()
async def get_capabilities(ctx: Context) -> CallToolResult:
    """Get available capabilities: embedding profiles, LLM profiles, strategies."""
    api_key = _extract_api_key(ctx)
    async with _client(api_key) as client:
        caps = await client.get_capabilities()
        lines = ["Platform Capabilities:"]
        for key, value in caps.items():
            lines.append(f"  {key}: {value}")
    return _result("\n".join(lines), {"capabilities": caps})


def admin_mcp_server_app() -> Starlette:
    """Create a standalone Starlette app for the admin MCP server."""
    return Starlette(
        routes=[Mount("/", app=MCPPathAliasApp(_admin_server_app))],
        lifespan=_admin_lifespan,
    )


def user_mcp_server_app() -> Starlette:
    """Create a standalone Starlette app for the user MCP server."""
    return Starlette(
        routes=[Mount("/", app=MCPPathAliasApp(_user_server_app))],
        lifespan=_user_lifespan,
    )


def admin_main() -> None:
    """CLI entry point for the admin MCP server."""
    import uvicorn

    port = int(os.getenv("GRAPH_CORE_ADMIN_MCP_PORT", "18102"))
    uvicorn.run(admin_mcp_server_app(), host="0.0.0.0", port=port)


def user_main() -> None:
    """CLI entry point for the user MCP server."""
    import uvicorn

    port = int(os.getenv("GRAPH_CORE_USER_MCP_PORT", "18103"))
    uvicorn.run(user_mcp_server_app(), host="0.0.0.0", port=port)
