"""MCP server for Graph Core — exposes platform operations as tools.

Auth flow:
1. Client sends Authorization: Bearer <key> on every MCP HTTP request
2. Server extracts the key from ctx.request_context.request.headers
3. Server passes the key to GraphCoreClient which forwards it to FastAPI
"""

import asyncio
import os
from contextlib import asynccontextmanager

from mcp.server.fastmcp import Context, FastMCP

from graph_core.client import GraphCoreAPIError, GraphCoreClient


def _get_base_url() -> str:
    return os.getenv("GRAPH_CORE_URL", "http://localhost:8001").rstrip("/")


def _extract_api_key(ctx: Context) -> str:
    """Extract the API key from the incoming MCP request.

    Checks in order:
    1. MCP protocol meta ({"api_key": "..."}) — most reliable
    2. Authorization: Bearer <key> header
    3. X-API-Key header
    4. Environment variables (fallback for testing)
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

    # 3. Fallback for non-HTTP transports or testing
    env_key = os.getenv("PLATFORM_ADMIN_KEY") or os.getenv("GRAPH_CORE_API_KEY")
    if env_key:
        return env_key
    raise GraphCoreAPIError(
        "No API key found in request metadata or headers."
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


mcp = FastMCP(
    name="graph-core",
    instructions="MCP server for the Graph Core knowledge platform",
    lifespan=server_lifespan,
    streamable_http_path="/",
)


# -- Namespace tools --------------------------------------------------------


@mcp.tool()
async def create_namespace(name: str, ctx: Context) -> str:
    """Create a new namespace. Requires admin key.

    Args:
        name: Human-readable namespace name (must be unique).
    """
    api_key = _extract_api_key(ctx)
    client = await get_client(api_key, admin=True)
    result = await client.create_namespace(name)
    return (
        f"Created namespace:\n"
        f"  id: {result['id']}\n"
        f"  name: {result['name']}\n"
        f"  api_key: {result['api_key']}\n\n"
        f"Save the api_key — it won't be shown again."
    )


@mcp.tool()
async def list_namespaces(ctx: Context) -> str:
    """List all namespaces. Requires admin key."""
    api_key = _extract_api_key(ctx)
    client = await get_client(api_key, admin=True)
    namespaces = await client.list_namespaces()
    if not namespaces:
        return "No namespaces found."
    lines = ["Namespaces:"]
    for ns in namespaces:
        prefix = ns.get("api_key_prefix", "") or ""
        lines.append(f"  - {ns['id']} | {ns['name']} {prefix}")
    return "\n".join(lines)


@mcp.tool()
async def get_current_namespace(ctx: Context) -> str:
    """Get info about the current authenticated namespace."""
    api_key = _extract_api_key(ctx)
    client = await get_client(api_key)
    ns = await client.get_namespace_me()
    return f"Namespace: {ns['id']} | {ns['name']}"


@mcp.tool()
async def rotate_namespace_key(namespace_id: str, ctx: Context) -> str:
    """Rotate a namespace's API key. Requires admin key.

    Args:
        namespace_id: The UUID of the namespace.
    """
    api_key = _extract_api_key(ctx)
    client = await get_client(api_key, admin=True)
    result = await client.rotate_namespace_key(namespace_id)
    return f"New api_key: {result['api_key']}\nSave it — it won't be shown again."


# -- Collection tools -------------------------------------------------------


@mcp.tool()
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


@mcp.tool()
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


@mcp.tool()
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


@mcp.tool()
async def delete_collection(collection_id: str, ctx: Context) -> str:
    """Delete a collection in the current namespace."""
    api_key = _extract_api_key(ctx)
    client = await get_client(api_key)
    result = await client.delete_collection(collection_id)
    return f"Deleted collection {result.get('id', collection_id)}"


@mcp.tool()
async def enhance_collection(collection_id: str, ctx: Context) -> str:
    """Build or rebuild the derived understanding graph for a collection."""
    api_key = _extract_api_key(ctx)
    client = await get_client(api_key)
    result = await client.enhance_collection(collection_id)
    return (
        f"Enhanced collection:\n"
        f"  collection_id: {result['collection_id']}\n"
        f"  graph_name: {result['graph_name']}\n"
        f"  node_count: {result['node_count']}\n"
        f"  edge_count: {result['edge_count']}\n"
        f"  chunk_count: {result['chunk_count']}"
    )


# -- Ingestion tools --------------------------------------------------------


@mcp.tool()
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


@mcp.tool()
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


@mcp.tool()
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


@mcp.tool()
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


@mcp.tool()
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


@mcp.tool()
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


@mcp.tool()
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


@mcp.tool()
async def list_jobs(ctx: Context, limit: int = 20) -> str:
    """List recent jobs in the current namespace."""
    api_key = _extract_api_key(ctx)
    client = await get_client(api_key)
    jobs = await client.list_jobs(limit=limit)
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


@mcp.tool()
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


@mcp.tool()
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


@mcp.tool()
async def list_embedding_profiles(ctx: Context) -> str:
    """List embedding profiles in the current namespace."""
    api_key = _extract_api_key(ctx)
    client = await get_client(api_key)
    profiles = await client.list_embedding_profiles()
    return _format_profile_list("Embedding Profiles", profiles)


@mcp.tool()
async def list_llm_profiles(ctx: Context) -> str:
    """List LLM profiles in the current namespace."""
    api_key = _extract_api_key(ctx)
    client = await get_client(api_key)
    profiles = await client.list_llm_profiles()
    return _format_profile_list("LLM Profiles", profiles)


@mcp.tool()
async def get_capabilities(ctx: Context) -> str:
    """Get available capabilities: embedding profiles, LLM profiles, strategies."""
    api_key = _extract_api_key(ctx)
    client = await get_client(api_key)
    caps = await client.get_capabilities()
    lines = ["Platform Capabilities:"]
    for key, value in caps.items():
        lines.append(f"  {key}: {value}")
    return "\n".join(lines)


def mcp_server_app() -> object:
    """Create the StreamableHTTP ASGI app for mounting in FastAPI."""
    return mcp.streamable_http_app()


def main() -> None:
    """CLI entry point for the MCP server."""
    import sys

    transport = sys.argv[1] if len(sys.argv) > 1 else "streamable-http"
    mcp.run(transport=transport)
