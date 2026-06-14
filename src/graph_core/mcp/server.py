"""MCP server for Graph Core — exposes platform operations as tools.

The admin and user surfaces run as separate MCP servers on different ports.
"""

import asyncio
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


_client_cache: dict[str, GraphCoreClient] = {}


async def get_client(api_key: str, admin: bool = False) -> GraphCoreClient:
    """Get or create a cached client for the given api key."""
    cache_key = f"{'admin' if admin else 'ns'}:{api_key[:10]}"
    if cache_key not in _client_cache:
        _client_cache[cache_key] = GraphCoreClient(
            base_url=_get_base_url(),
            api_key=api_key,
            is_admin=admin,
        )
    return _client_cache[cache_key]


@asynccontextmanager
async def server_lifespan(server: FastMCP):
    try:
        yield
    finally:
        for client in _client_cache.values():
            await client.close()
        _client_cache.clear()


admin_mcp = FastMCP(
    name="graph-core-admin",
    instructions="MCP admin tools for namespace management",
    lifespan=server_lifespan,
    streamable_http_path="/",
)

user_mcp = FastMCP(
    name="graph-core-user",
    instructions="MCP user tools scoped to a namespace",
    lifespan=server_lifespan,
    streamable_http_path="/",
)

admin_tool = admin_mcp.tool
user_tool = user_mcp.tool


def _header_lookup(scope: dict, name: bytes) -> str:
    for key, value in scope.get("headers", []):
        if key.lower() == name:
            return value.decode("latin-1")
    return ""


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
            await _send_http_error(send, exc.status_code, str(exc.detail))
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
    client = await get_client(api_key, admin=True)
    result = await client.create_namespace(name)
    text = (
        f"Created namespace:\n"
        f"  id: {result['id']}\n"
        f"  name: {result['name']}"
    )
    return CallToolResult(
        content=[TextContent(type="text", text=text)],
        structuredContent={
            "namespace": {
                "id": result["id"],
                "name": result["name"],
            }
        },
    )


@admin_tool()
async def list_namespaces(ctx: Context) -> CallToolResult:
    """List all namespaces. Requires admin JWT."""
    api_key = _extract_api_key(ctx)
    client = await get_client(api_key, admin=True)
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
async def get_current_namespace(ctx: Context) -> str:
    """Get info about the current authenticated namespace."""
    api_key = _extract_api_key(ctx)
    client = await get_client(api_key)
    ns = await client.get_namespace_me()
    return f"Namespace: {ns['id']} | {ns['name']}"


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
) -> str:
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
    client = await get_client(api_key)
    result = await client.create_collection(
        name=name,
        strategy=strategy,
        embedding_profile_id=embedding_profile_id,
        llm_profile_id=llm_profile_id,
        default_query_mode=default_query_mode,
        gleaning_passes=gleaning_passes,
    )
    return (
        f"Created collection:\n"
        f"  id: {result['id']}\n"
        f"  name: {result['name']}\n"
        f"  strategy: {result['strategy']}\n"
        f"  embedding_profile_id: {result.get('embedding_profile_id') or 'N/A'}\n"
        f"  llm_profile_id: {result.get('llm_profile_id') or 'N/A'}\n"
        f"  gleaning_passes: {result.get('gleaning_passes', 1)}"
    )


@user_tool()
async def list_collections(ctx: Context) -> str:
    """List all collections in the current namespace."""
    api_key = _extract_api_key(ctx)
    client = await get_client(api_key)
    collections = await client.list_collections()
    if not collections:
        return "No collections found."
    lines = ["Collections:"]
    for col in collections:
        lines.append(f"  - {col['id']} | {col['name']} ({col['strategy']})")
    return "\n".join(lines)


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
) -> str:
    """Update a collection in the current namespace."""
    api_key = _extract_api_key(ctx)
    client = await get_client(api_key)
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
    return (
        f"Updated collection:\n"
        f"  id: {result['id']}\n"
        f"  name: {result['name']}\n"
        f"  strategy: {result['strategy']}\n"
        f"  embedding_profile_id: {result.get('embedding_profile_id') or 'N/A'}\n"
        f"  llm_profile_id: {result.get('llm_profile_id') or 'N/A'}\n"
        f"  gleaning_passes: {result.get('gleaning_passes', 1)}"
    )


@user_tool()
async def delete_collection(collection_id: str, ctx: Context) -> str:
    """Delete a collection in the current namespace."""
    api_key = _extract_api_key(ctx)
    client = await get_client(api_key)
    result = await client.delete_collection(collection_id)
    return f"Deleted collection {result.get('id', collection_id)}"


@user_tool()
async def enhance_collection(
    collection_id: str,
    levels: int = 1,
    ctx: Context | None = None,
) -> str:
    """Build or rebuild the derived understanding graph for a collection."""
    if ctx is None:
        raise ValueError("Context is required")
    api_key = _extract_api_key(ctx)
    client = await get_client(api_key)
    result = await client.enhance_collection(collection_id, levels=levels)
    type_counts = result.get("node_type_counts", {})
    type_lines = "\n".join(
        f"  {key}: {value}" for key, value in sorted(type_counts.items())
    )
    generated_levels = result.get("generated_levels", [])
    level_lines = "\n".join(
        (
            f"  l{level['level']}: {level['collection_name']} "
            f"(nodes={level['node_count']}, edges={level['edge_count']}, chunks={level['chunk_count']})"
        )
        for level in generated_levels
    )
    return (
        f"Enhanced collection:\n"
        f"  collection_id: {result['collection_id']}\n"
        f"  requested_levels: {result.get('requested_levels', levels)}\n"
        f"  graph_name: {result['graph_name']}\n"
        f"  node_count: {result['node_count']}\n"
        f"  edge_count: {result['edge_count']}\n"
        f"  chunk_count: {result['chunk_count']}\n"
        f"  rel_type_count: {result.get('rel_type_count', 0)}\n"
        f"  community_count: {result.get('community_count', 0)}\n"
        f"  anchor_count: {result.get('anchor_count', 0)}\n"
        f"  bridge_count: {result.get('bridge_count', 0)}\n"
        f"  connector_count: {result.get('connector_count', 0)}\n"
        f"  generated_levels:\n{level_lines or '  (none)'}\n"
        f"  node_type_counts:\n{type_lines or '  (none)'}"
    )


# -- Ingestion tools --------------------------------------------------------


@user_tool()
async def ingest_chunk(
    collection_id: str,
    text: str,
    ctx: Context,
    domain: str | None = None,
) -> str:
    """Ingest a text chunk directly into a collection.

    For large documents, use ingest_document instead (it runs async with a job).

    Args:
        collection_id: The UUID of the target collection.
        text: The text content to ingest.
    """
    api_key = _extract_api_key(ctx)
    client = await get_client(api_key)
    result = await client.ingest_chunk(collection_id, text, domain=domain)
    return (
        f"Ingested chunk:\n"
        f"  hash: {result.get('chunk_hash', 'N/A')}\n"
        f"  entities: {result.get('entity_count', 0)}\n"
        f"  relationships: {result.get('relationship_count', 0)}"
    )


@user_tool()
async def ingest_document(
    collection_id: str,
    text: str,
    ctx: Context,
    domain: str | None = None,
) -> str:
    """Ingest a full document into a collection (async, returns job_id).

    For large documents, the platform will chunk and process in the background.

    Args:
        collection_id: The UUID of the target collection.
        text: The full document text.
    """
    api_key = _extract_api_key(ctx)
    client = await get_client(api_key)
    result = await client.ingest_document(collection_id, text, domain=domain)
    return (
        f"Document ingestion started:\n"
        f"  job_id: {result['job_id']}\n"
        f"  status: {result['status']}\n\n"
        f"Track with get_job_status('{result['job_id']}')"
    )


@user_tool()
async def ingest_file(collection_id: str, file_path: str, ctx: Context) -> str:
    """Read a local file and ingest its contents into a collection.

    Args:
        collection_id: The UUID of the target collection.
        file_path: Absolute path to the text file.
    """
    loop = asyncio.get_event_loop()

    def _read(path: str) -> str:
        with open(path, encoding="utf-8") as f:
            return f.read()

    content = await loop.run_in_executor(None, _read, file_path)
    api_key = _extract_api_key(ctx)
    client = await get_client(api_key)
    result = await client.ingest_document(collection_id, content)
    return (
        f"Document ingestion started:\n"
        f"  job_id: {result['job_id']}\n"
        f"  status: {result['status']}\n\n"
        f"Track with get_job_status('{result['job_id']}')"
    )


# -- Query tools ------------------------------------------------------------


@user_tool()
async def query_collection(
    collection_id: str,
    question: str,
    ctx: Context,
    mode: str | None = None,
    chat_id: str | None = None,
) -> str:
    """Query a collection with a natural language question.

    Args:
        collection_id: The UUID of the collection to query.
        question: The natural language question.
        mode: Query mode for light_rag/custom graph retrieval. Leave empty for default.
        chat_id: Optional chat session UUID for follow-up memory.
    """
    api_key = _extract_api_key(ctx)
    client = await get_client(api_key)
    result = await client.query_collection(
        collection_id,
        question,
        mode=mode,
        chat_id=chat_id,
    )
    lines = [result["response"]]
    if result.get("entities_used"):
        lines.append(f"\nEntities used: {', '.join(result['entities_used'])}")
    if result.get("relationships_used"):
        lines.append(f"Relationships: {', '.join(result['relationships_used'])}")
    if result.get("mode"):
        lines.append(f"Mode: {result['mode']}")
    if result.get("chat_id"):
        lines.append(f"Chat ID: {result['chat_id']}")
    return "\n".join(lines)


@user_tool()
async def create_chat_session(
    collection_id: str,
    ctx: Context,
    title: str | None = None,
) -> str:
    """Create a chat session for follow-up query context."""
    api_key = _extract_api_key(ctx)
    client = await get_client(api_key)
    result = await client.create_chat_session(collection_id, title=title)
    return (
        f"Created chat session:\n"
        f"  id: {result['id']}\n"
        f"  collection_id: {result['collection_id']}\n"
        f"  title: {result.get('title') or '-'}\n"
        f"  turn_count: {result.get('turn_count', 0)}"
    )


@user_tool()
async def list_chat_sessions(
    collection_id: str,
    ctx: Context,
    limit: int = 20,
) -> str:
    """List chat sessions for a collection."""
    api_key = _extract_api_key(ctx)
    client = await get_client(api_key)
    rows = await client.list_chat_sessions(collection_id, limit=limit)
    if not rows:
        return "No chat sessions found."
    lines = ["Chat sessions:"]
    for row in rows:
        lines.append(
            f"  - {row['id']} | turns={row.get('turn_count', 0)}"
            f" | title={row.get('title') or '-'}"
        )
    return "\n".join(lines)


# -- Job tools --------------------------------------------------------------


@user_tool()
async def get_job_status(job_id: str, ctx: Context) -> str:
    """Check the status of an async ingestion job.

    Args:
        job_id: The UUID of the job.
    """
    api_key = _extract_api_key(ctx)
    client = await get_client(api_key)
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
    return "\n".join(lines)


@user_tool()
async def list_jobs(
    limit: int = 20,
    collection_id: str | None = None,
    ctx: Context = None,
) -> str:
    """List recent jobs in the current namespace."""
    api_key = _extract_api_key(ctx)
    client = await get_client(api_key)
    jobs = await client.list_jobs(limit=limit, collection_id=collection_id)
    if not jobs:
        return "No jobs found."
    lines = ["Jobs:"]
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
    return "\n".join(lines)


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
) -> str:
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
    client = await get_client(api_key)
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
    return (
        f"Created embedding profile:\n"
        f"  profile_id: {profile['profile_id']}\n"
        f"  label: {profile.get('label') or '-'}\n"
        f"  provider: {profile['provider']}\n"
        f"  model: {profile['model']}\n"
        f"  dimensions: {profile.get('dimensions') or '-'}\n"
        f"  max_concurrent_calls: {profile.get('max_concurrent_calls') or '-'}"
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
) -> str:
    """Create an LLM profile in the current namespace.

    Args:
        provider: Provider name, e.g. 'openai'.
        model: LLM model identifier.
        secret: Provider API key or token.
        label: Optional human-readable label.
        base_url: Optional custom API base URL.
    """
    api_key = _extract_api_key(ctx)
    client = await get_client(api_key)
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
    return (
        f"Created llm profile:\n"
        f"  profile_id: {profile['profile_id']}\n"
        f"  label: {profile.get('label') or '-'}\n"
        f"  provider: {profile['provider']}\n"
        f"  model: {profile['model']}\n"
        f"  max_concurrent_calls: {profile.get('max_concurrent_calls') or '-'}"
    )


def _format_profile_list(title: str, profiles: list[dict]) -> str:
    if not profiles:
        return f"No {title.lower()} found."
    lines = [f"{title}:"]
    for profile in profiles:
        label = profile.get("label") or "-"
        model = profile.get("model") or "-"
        provider = profile.get("provider") or "-"
        limit = profile.get("max_concurrent_calls")
        lines.append(
            f"  - {profile['profile_id']} | {label} | {provider} | {model} | "
            f"max_concurrent_calls={limit if limit is not None else '-'}"
        )
    return "\n".join(lines)


@user_tool()
async def list_embedding_profiles(ctx: Context) -> str:
    """List embedding profiles in the current namespace."""
    api_key = _extract_api_key(ctx)
    client = await get_client(api_key)
    profiles = await client.list_embedding_profiles()
    return _format_profile_list("Embedding Profiles", profiles)


@user_tool()
async def list_llm_profiles(ctx: Context) -> str:
    """List LLM profiles in the current namespace."""
    api_key = _extract_api_key(ctx)
    client = await get_client(api_key)
    profiles = await client.list_llm_profiles()
    return _format_profile_list("LLM Profiles", profiles)


@user_tool()
async def get_capabilities(ctx: Context) -> str:
    """Get available capabilities: embedding profiles, LLM profiles, strategies."""
    api_key = _extract_api_key(ctx)
    client = await get_client(api_key)
    caps = await client.get_capabilities()
    lines = ["Platform Capabilities:"]
    for key, value in caps.items():
        lines.append(f"  {key}: {value}")
    return "\n".join(lines)


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

    port = int(os.getenv("GRAPH_CORE_ADMIN_MCP_PORT", "8002"))
    uvicorn.run(admin_mcp_server_app(), host="0.0.0.0", port=port)


def user_main() -> None:
    """CLI entry point for the user MCP server."""
    import uvicorn

    port = int(os.getenv("GRAPH_CORE_USER_MCP_PORT", "8003"))
    uvicorn.run(user_mcp_server_app(), host="0.0.0.0", port=port)
