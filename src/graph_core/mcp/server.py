"""MCP server for Graph Core — exposes platform operations as tools."""

import asyncio
import os
from contextlib import asynccontextmanager

from mcp.server.fastmcp import FastMCP

from graph_core.client import GraphCoreAPIError, GraphCoreClient


def _get_base_url() -> str:
    return os.getenv("GRAPH_CORE_URL", "http://localhost:8000").rstrip("/")


def _get_admin_key() -> str | None:
    return os.getenv("PLATFORM_ADMIN_KEY")


def _get_api_key() -> str | None:
    return os.getenv("GRAPH_CORE_API_KEY")


# Module-level client caches keyed by auth mode
_clients: dict[str, GraphCoreClient] = {}


async def get_client(admin: bool = False) -> GraphCoreClient:
    """Get or create a cached client for the given auth mode."""
    key = "admin" if admin else "namespace"
    if key not in _clients:
        if admin:
            api_key = _get_admin_key()
            if not api_key:
                raise GraphCoreAPIError(
                    "PLATFORM_ADMIN_KEY env var required for admin tools"
                )
            _clients[key] = GraphCoreClient(
                base_url=_get_base_url(), api_key=api_key, is_admin=True
            )
        else:
            api_key = _get_api_key() or _get_admin_key()
            if not api_key:
                raise GraphCoreAPIError(
                    "GRAPH_CORE_API_KEY or GRAPH_CORE_ADMIN_KEY env var required"
                )
            _clients[key] = GraphCoreClient(
                base_url=_get_base_url(), api_key=api_key, is_admin=admin
            )
    return _clients[key]


@asynccontextmanager
async def server_lifespan(server: FastMCP):
    try:
        yield
    finally:
        for client in _clients.values():
            await client.close()
        _clients.clear()


mcp = FastMCP(
    name="graph-core",
    instructions="MCP server for the Graph Core knowledge platform",
    lifespan=server_lifespan,
    streamable_http_path="/",
)


# -- Namespace tools --------------------------------------------------------


@mcp.tool()
async def create_namespace(name: str) -> str:
    """Create a new namespace. Requires GRAPH_CORE_ADMIN_KEY.

    Args:
        name: Human-readable namespace name (must be unique).
    """
    client = await get_client(admin=True)
    result = await client.create_namespace(name)
    return (
        f"Created namespace:\n"
        f"  id: {result['id']}\n"
        f"  name: {result['name']}\n"
        f"  api_key: {result['api_key']}\n\n"
        f"Save the api_key — it won't be shown again."
    )


@mcp.tool()
async def list_namespaces() -> str:
    """List all namespaces. Requires GRAPH_CORE_ADMIN_KEY."""
    client = await get_client(admin=True)
    namespaces = await client.list_namespaces()
    if not namespaces:
        return "No namespaces found."
    lines = ["Namespaces:"]
    for ns in namespaces:
        prefix = ns.get("api_key_prefix", "") or ""
        lines.append(f"  - {ns['id']} | {ns['name']} {prefix}")
    return "\n".join(lines)


@mcp.tool()
async def get_current_namespace() -> str:
    """Get info about the current authenticated namespace."""
    client = await get_client()
    ns = await client.get_namespace_me()
    return f"Namespace: {ns['id']} | {ns['name']}"


@mcp.tool()
async def rotate_namespace_key(namespace_id: str) -> str:
    """Rotate a namespace's API key. Requires GRAPH_CORE_ADMIN_KEY.

    Args:
        namespace_id: The UUID of the namespace.
    """
    client = await get_client(admin=True)
    result = await client.rotate_namespace_key(namespace_id)
    return f"New api_key: {result['api_key']}\nSave it — it won't be shown again."


# -- Collection tools -------------------------------------------------------


@mcp.tool()
async def create_collection(
    name: str,
    strategy: str = "vector",
) -> str:
    """Create a new collection in the current namespace.

    Args:
        name: Collection name (unique within namespace).
        strategy: Retrieval strategy: 'vector', 'light_rag', or 'custom_graph_rag'.
    """
    client = await get_client()
    result = await client.create_collection(name=name, strategy=strategy)
    return (
        f"Created collection:\n"
        f"  id: {result['id']}\n"
        f"  name: {result['name']}\n"
        f"  strategy: {result['strategy']}"
    )


@mcp.tool()
async def list_collections() -> str:
    """List all collections in the current namespace."""
    client = await get_client()
    collections = await client.list_collections()
    if not collections:
        return "No collections found."
    lines = ["Collections:"]
    for col in collections:
        lines.append(f"  - {col['id']} | {col['name']} ({col['strategy']})")
    return "\n".join(lines)


# -- Ingestion tools --------------------------------------------------------


@mcp.tool()
async def ingest_chunk(collection_id: str, text: str) -> str:
    """Ingest a text chunk directly into a collection.

    For large documents, use ingest_document instead (it runs async with a job).

    Args:
        collection_id: The UUID of the target collection.
        text: The text content to ingest.
    """
    client = await get_client()
    result = await client.ingest_chunk(collection_id, text)
    return (
        f"Ingested chunk:\n"
        f"  hash: {result.get('chunk_hash', 'N/A')}\n"
        f"  entities: {result.get('entity_count', 0)}\n"
        f"  relationships: {result.get('relationship_count', 0)}"
    )


@mcp.tool()
async def ingest_document(collection_id: str, text: str) -> str:
    """Ingest a full document into a collection (async, returns job_id).

    For large documents, the platform will chunk and process in the background.

    Args:
        collection_id: The UUID of the target collection.
        text: The full document text.
    """
    client = await get_client()
    result = await client.ingest_document(collection_id, text)
    return (
        f"Document ingestion started:\n"
        f"  job_id: {result['job_id']}\n"
        f"  status: {result['status']}\n\n"
        f"Track with get_job_status('{result['job_id']}')"
    )


@mcp.tool()
async def ingest_file(collection_id: str, file_path: str) -> str:
    """Read a local file and ingest its contents into a collection.

    Args:
        collection_id: The UUID of the target collection.
        file_path: Absolute path to the text file.
    """
    loop = asyncio.get_event_loop()
    content = await loop.run_in_executor(None, _read_file, file_path)
    return await ingest_document(collection_id, content)


def _read_file(path: str) -> str:
    with open(path, encoding="utf-8") as f:
        return f.read()


# -- Query tools ------------------------------------------------------------


@mcp.tool()
async def query_collection(
    collection_id: str,
    question: str,
    mode: str | None = None,
) -> str:
    """Query a collection with a natural language question.

    Args:
        collection_id: The UUID of the collection to query.
        question: The natural language question.
        mode: Query mode for light_rag: 'local', 'global', 'hybrid', 'naive', 'mix'.
               Leave empty for default.
    """
    client = await get_client()
    result = await client.query_collection(collection_id, question, mode=mode)
    lines = [result["response"]]
    if result.get("entities_used"):
        lines.append(f"\nEntities used: {', '.join(result['entities_used'])}")
    if result.get("relationships_used"):
        lines.append(f"Relationships: {', '.join(result['relationships_used'])}")
    if result.get("mode"):
        lines.append(f"Mode: {result['mode']}")
    return "\n".join(lines)


# -- Job tools --------------------------------------------------------------


@mcp.tool()
async def get_job_status(job_id: str) -> str:
    """Check the status of an async ingestion job.

    Args:
        job_id: The UUID of the job.
    """
    client = await get_client()
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


# -- Platform tools ---------------------------------------------------------


@mcp.tool()
async def get_capabilities() -> str:
    """Get available capabilities: embedding profiles, LLM profiles, strategies."""
    client = await get_client()
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
