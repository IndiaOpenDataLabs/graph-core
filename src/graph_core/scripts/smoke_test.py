"""Smoke test — verifies the full stack is functional end-to-end.

Run after `make docker-up` to confirm Postgres, FalkorDB, Redis,
the app, and the worker are all working together.

Usage (offline — vector strategy only, local providers):
    uv run python -m graph_core.scripts.smoke_test
    make smoke-test

Usage (all strategies + modes, with remote LLM/embeddings):
    uv run python -m graph_core.scripts.smoke_test \
        --llm-key sk-... --embed-key sk-...
    uv run python -m graph_core.scripts.smoke_test \
        --llm-key sk-... --llm-url https://custom.ai/v1 \
        --embed-key sk-... --embed-url https://custom.ai/v1
"""

import argparse
import asyncio
import os
import sys
import uuid

import httpx

BASE_URL = "http://localhost:8001"
DB_URL = "postgresql://graphcore:graphcore@localhost:5432/graphcore"

# Text used for ingestion — contains entities/relationships for graph strategies
INGEST_TEXT = (
    "Krishna teaches Arjuna about dharma and duty on the battlefield of Kurukshetra. "
    "Arjuna is a great warrior prince of the Pandavas. "
    "The Bhagavad Gita is a sacred Hindu scripture that records this conversation. "
    "Dharma represents cosmic order and moral duty in Hindu philosophy."
)

# Query to test retrieval
QUERY_TEXT = "What does Krishna teach Arjuna?"

# Keywords expected in a correct response
EXPECTED_KEYWORDS = ["krishna", "arjuna", "dharma", "duty", "teach"]

# Strategies and their query modes
STRATEGIES = [
    {"name": "vector", "modes": ["local"]},
    {"name": "custom_graph_rag", "modes": []},
    {"name": "light_rag", "modes": ["local", "global", "hybrid", "naive", "mix"]},
]

GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
RESET = "\033[0m"
BOLD = "\033[1m"


def _pass(msg: str) -> None:
    print(f"  {GREEN}✓{RESET} {msg}")


def _fail(msg: str) -> None:
    print(f"  {RED}✗{RESET} {msg}")


def _skip(msg: str) -> None:
    print(f"  {YELLOW}⊘{RESET} {msg}")


def _info(msg: str) -> None:
    print(f"  {CYAN}ℹ{RESET} {msg}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Graph Core smoke test")
    p.add_argument("--admin-jwt", default=os.environ.get("GRAPH_CORE_ADMIN_JWT", ""), help="Admin JWT for namespace creation")
    p.add_argument("--llm-key", help="OpenAI (or compatible) API key for LLM")
    p.add_argument("--llm-url", help="Custom LLM base URL (e.g. https://api.openai.com/v1)")
    p.add_argument("--llm-model", default="gpt-4o", help="LLM model name (default: gpt-4o)")
    p.add_argument("--embed-key", help="OpenAI (or compatible) API key for embeddings")
    p.add_argument("--embed-url", help="Custom embedding base URL")
    p.add_argument("--embed-model", default="text-embedding-3-large", help="Embedding model (default: text-embedding-3-large)")
    p.add_argument("--embed-dimensions", type=int, default=3072, help="Embedding dimensions (default: 3072)")
    return p.parse_args()


async def create_namespace_via_api(client, admin_jwt: str) -> tuple[uuid.UUID, str, str]:
    """Create a namespace via API. Returns the namespace ID, name, and token."""
    ns_name = f"smoke-test-{uuid.uuid4().hex[:8]}"
    r = await client.post(
        "/platform/namespaces/",
        json={"name": ns_name},
        headers={"Authorization": f"Bearer {admin_jwt}"},
    )
    if r.status_code != 200:
        raise ValueError(f"Namespace creation failed: {r.text}")
    data = r.json()
    return uuid.UUID(data["id"]), data["name"], data["token"]


async def create_namespace_direct_db() -> uuid.UUID:
    """Legacy: create a namespace directly in Postgres (no API key)."""
    import asyncpg

    conn = await asyncpg.connect(DB_URL)
    ns_id = uuid.uuid4()
    ns_name = f"smoke-test-{ns_id.hex[:8]}"
    await conn.execute(
        "INSERT INTO namespaces (id, name) VALUES ($1, $2)",
        ns_id,
        ns_name,
    )
    await conn.close()
    return ns_id


async def register_credential(client, headers, api_key: str, base_url: str | None = None) -> str:
    """Register an OpenAI credential. Returns credential_id."""
    body = {"provider": "openai", "secret": api_key, "label": "smoke-test"}
    if base_url:
        body["base_url"] = base_url
    r = await client.post(
        "/platform/credentials",
        json=body,
        headers=headers,
    )
    if r.status_code != 200:
        raise ValueError(f"Credential registration failed: {r.text}")
    return r.json()["credential_id"]


async def create_profile(client, headers, kind: str, provider: str, model: str,
                         credential_id: str | None, label: str,
                         base_url: str | None = None,
                         dimensions: int | None = None) -> str:
    """Create an embedding or LLM profile. Returns profile_id."""
    body = {
        "kind": kind,
        "provider": provider,
        "model": model,
        "label": label,
    }
    if credential_id:
        body["credential_id"] = credential_id
    if base_url:
        body["base_url"] = base_url
    if dimensions:
        body["dimensions"] = dimensions
        body["distance_metric"] = "cosine"

    r = await client.post("/platform/profiles", json=body, headers=headers)
    if r.status_code != 200:
        raise ValueError(f"Profile creation failed ({kind}): {r.text}")
    return r.json()["profile_id"]


async def create_collection(client, headers, name: str, strategy: str,
                            embed_profile_id: str, llm_profile_id: str | None = None) -> str:
    """Create a collection. Returns collection_id."""
    body = {
        "name": name,
        "strategy": strategy,
        "embedding_profile_id": embed_profile_id,
    }
    if llm_profile_id:
        body["llm_profile_id"] = llm_profile_id
    r = await client.post(
        "/collections/",
        json=body,
        headers=headers,
    )
    if r.status_code != 200:
        raise ValueError(f"Collection creation failed: {r.text}")
    return r.json()["id"]


async def ingest_chunk(client, headers, collection_id: str, text: str) -> dict:
    """Ingest a text chunk. Returns response JSON."""
    r = await client.post(
        f"/collections/{collection_id}/ingest/chunk",
        json={"text": text},
        headers=headers,
    )
    if r.status_code != 200:
        raise ValueError(f"Ingest failed: {r.text}")
    return r.json()


async def query_collection(client, headers, collection_id: str,
                           question: str, mode: str | None = None) -> dict:
    """Query a collection. Returns response JSON."""
    body = {"question": question}
    if mode:
        body["mode"] = mode
    r = await client.post(
        f"/collections/{collection_id}/query",
        json=body,
        headers=headers,
    )
    if r.status_code != 200:
        raise ValueError(f"Query failed: {r.text}")
    return r.json()


def check_response(data: dict, keywords: list[str]) -> bool:
    """Check if the query response contains any of the expected keywords."""
    response_text = data.get("response", "").lower()
    if not response_text:
        return False
    return any(kw.lower() in response_text for kw in keywords)


async def run_smoke_test(args: argparse.Namespace) -> bool:
    passed = 0
    failed = 0
    skipped = 0

    def record(ok: bool, msg: str):
        nonlocal passed, failed
        if ok:
            passed += 1
            _pass(msg)
        else:
            failed += 1
            _fail(msg)

    def skip(msg: str):
        nonlocal skipped
        skipped += 1
        _skip(msg)

    has_api = bool(args.llm_key and args.embed_key)
    provider_label = "remote" if has_api else "local"

    async with httpx.AsyncClient(base_url=BASE_URL, timeout=120.0) as client:

        # ── 1. Health check ──────────────────────────────────────────
        print(f"\n{BOLD}1. Health check{RESET}")
        try:
            r = await client.get("/health")
            record(r.status_code == 200 and r.json().get("status") == "ok", "App is healthy")
        except Exception as e:
            record(False, f"Health check failed: {e}")
            return False

        # ── 2. Create namespace ──────────────────────────────────────
        print(f"\n{BOLD}2. Create namespace{RESET}")
        ns_id = None
        user_token = None
        use_legacy_auth = not args.admin_jwt

        if use_legacy_auth:
            # Legacy: raw DB creation + X-Namespace-ID header
            try:
                ns_id = await create_namespace_direct_db()
                record(True, f"Namespace created (legacy DB): {ns_id}")
            except Exception as e:
                record(False, f"Failed to create namespace: {e}")
                return False
            headers = {"X-Namespace-ID": str(ns_id)}
            _info("Using legacy X-Namespace-ID auth — pass --admin-jwt for API-based namespace creation")
        else:
            # New: API-based creation + user JWT
            try:
                ns_id, ns_name, user_token = await create_namespace_via_api(client, args.admin_jwt)
                record(True, f"Namespace created: {ns_id} ({ns_name})")
            except Exception as e:
                record(False, f"Failed to create namespace: {e}")
                return False
            headers = {"Authorization": f"Bearer {user_token}"}

        # ── 3. Register credential (OpenAI only) ─────────────────────
        cred_id = None
        if has_api:
            print(f"\n{BOLD}3. Register credential (OpenAI){RESET}")
            try:
                cred_id = await register_credential(client, headers, args.llm_key, args.llm_url)
                record(True, f"Credential registered: {cred_id}")
            except Exception as e:
                record(False, f"Failed to register credential: {e}")
                return False

        # ── 4. Create embedding profile ──────────────────────────────
        print(f"\n{BOLD}4. Create embedding profile ({provider_label}){RESET}")
        try:
            if has_api:
                embed_profile_id = await create_profile(
                    client, headers,
                    kind="embedding", provider="openai",
                    model=args.embed_model,
                    credential_id=cred_id,
                    label="smoke-embed-remote",
                    base_url=args.embed_url,
                    dimensions=args.embed_dimensions,
                )
            else:
                embed_profile_id = await create_profile(
                    client, headers,
                    kind="embedding", provider="local_hash",
                    model="hash-256",
                    credential_id=None,
                    label="smoke-embed-local",
                    dimensions=256,
                )
            record(True, f"Embedding profile created: {embed_profile_id}")
        except Exception as e:
            record(False, f"Failed to create embedding profile: {e}")
            return False

        # ── 5. Create LLM profile ────────────────────────────────────
        print(f"\n{BOLD}5. Create LLM profile ({provider_label}){RESET}")
        try:
            if has_api:
                llm_profile_id = await create_profile(
                    client, headers,
                    kind="llm", provider="openai",
                    model=args.llm_model,
                    credential_id=cred_id,
                    label="smoke-llm-remote",
                    base_url=args.llm_url,
                )
            else:
                llm_profile_id = await create_profile(
                    client, headers,
                    kind="llm", provider="local_echo",
                    model="echo-v1",
                    credential_id=None,
                    label="smoke-llm-local",
                )
            record(True, f"LLM profile created: {llm_profile_id}")
        except Exception as e:
            record(False, f"Failed to create LLM profile: {e}")
            return False

        # ── 6. Check capabilities ────────────────────────────────────
        print(f"\n{BOLD}6. Platform capabilities{RESET}")
        try:
            r = await client.get("/platform/capabilities", headers=headers)
            if r.status_code == 200:
                data = r.json()
                strategies = data.get("retrieval_strategies", [])
                record(
                    "vector" in strategies and "custom_graph_rag" in strategies,
                    f"Strategies: {', '.join(strategies)}",
                )
            else:
                record(False, f"Unexpected status {r.status_code}: {r.text}")
        except Exception as e:
            record(False, f"Failed to get capabilities: {e}")

        # ── 7. Test each strategy × mode ─────────────────────────────
        step = 7
        for strategy_info in STRATEGIES:
            strategy = strategy_info["name"]
            modes = strategy_info["modes"]

            print(f"\n{BOLD}{step}. Strategy: {strategy}{RESET}")

            # Create collection
            try:
                coll_name = f"smoke-{strategy}"
                coll_id = await create_collection(client, headers, coll_name, strategy, embed_profile_id, llm_profile_id)
                record(True, f"Collection created: {coll_id}")
            except Exception as e:
                record(False, f"Failed to create {strategy} collection: {e}")
                step += 1
                continue

            # Ingest
            try:
                result = await ingest_chunk(client, headers, coll_id, INGEST_TEXT)
                entity_count = result.get("entity_count", 0)
                record(
                    "chunk_hash" in result,
                    f"Chunk ingested (entities={entity_count}, hash={result.get('chunk_hash', '?')[:12]}...)",
                )
            except Exception as e:
                record(False, f"Failed to ingest into {strategy}: {e}")
                step += 1
                continue

            # Query each mode (or once with no mode if the strategy doesn't use modes)
            query_modes = modes if modes else [None]
            for mode in query_modes:
                mode_label = f"{strategy}/{mode}" if mode else strategy
                try:
                    data = await query_collection(client, headers, coll_id, QUERY_TEXT, mode=mode)
                    response_text = data.get("response", "")
                    ok = check_response(data, EXPECTED_KEYWORDS)
                    record(ok, f"Query {mode_label} returned relevant context")
                    if response_text:
                        _info(f"Response: {response_text[:150]}...")
                except Exception as e:
                    record(False, f"Query {mode_label} failed: {e}")

            step += 1

        # ── Final: List collections ──────────────────────────────────
        print(f"\n{BOLD}{step}. List collections{RESET}")
        try:
            r = await client.get("/collections/", headers=headers)
            if r.status_code == 200:
                collections = r.json()
                record(True, f"Found {len(collections)} collection(s)")
            else:
                record(False, f"Unexpected status {r.status_code}: {r.text}")
        except Exception as e:
            record(False, f"Failed to list collections: {e}")

    # ── Summary ──────────────────────────────────────────────────────
    total = passed + failed + skipped
    print(f"\n{'='*50}")
    print(f"  Provider: {BOLD}{provider_label}{RESET}")
    if has_api:
        print(f"  LLM: {args.llm_model}{' (' + args.llm_url + ')' if args.llm_url else ''}")
        print(f"  Embed: {args.embed_model}{' (' + args.embed_url + ')' if args.embed_url else ''}")
    if failed == 0 and skipped == 0:
        print(f"  {GREEN}{BOLD}All {passed} checks passed ✓{RESET}")
    elif failed == 0:
        print(f"  {GREEN}{BOLD}{passed} passed, {skipped} skipped{RESET}")
    else:
        print(f"  {RED}{BOLD}{passed} passed, {failed} failed, {skipped} skipped{RESET}")
    print(f"{'='*50}\n")

    return failed == 0


def main():
    args = parse_args()

    # Validate: graph strategies need both keys
    if args.llm_key != args.embed_key and (args.llm_key or args.embed_key):
        # Different keys provided — that's fine, but we need both
        if not (args.llm_key and args.embed_key):
            print(f"{RED}Both --llm-key and --embed-key are required for graph strategies.{RESET}")
            print(f"Pass --help for usage.\n")
            sys.exit(1)

    # Check if asyncpg is available (only needed for legacy auth)
    if not args.admin_jwt:
        try:
            import asyncpg  # noqa: F401
        except ImportError:
            print(f"{RED}asyncpg is not installed. Run: uv sync --all-groups or pass --admin-jwt{RESET}")
            sys.exit(1)

    has_api = bool(args.llm_key and args.embed_key)
    provider_label = "remote" if has_api else "local"

    print(f"\n{BOLD}Graph Core Smoke Test{RESET}")
    print(f"Target: {BASE_URL}")
    print(f"Database: {DB_URL}")
    print(f"Provider: {provider_label}")
    if has_api:
        print(f"LLM model: {args.llm_model}")
        print(f"Embed model: {args.embed_model}")
    print()

    ok = asyncio.run(run_smoke_test(args))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
