"""Smoke test — verifies the full stack is functional end-to-end.

Run after `make docker-up` to confirm Postgres, FalkorDB, Redis,
the app, and the worker are all working together.

Usage:
    uv run python -m graph_core.scripts.smoke_test
    # or
    make smoke-test
"""

import asyncio
import sys
import uuid

import httpx

BASE_URL = "http://localhost:8000"
DB_URL = "postgresql://graphcore:graphcore@localhost:5432/graphcore"

GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
RESET = "\033[0m"
BOLD = "\033[1m"


def _pass(msg: str) -> None:
    print(f"  {GREEN}✓{RESET} {msg}")


def _fail(msg: str) -> None:
    print(f"  {RED}✗{RESET} {msg}")


def _info(msg: str) -> None:
    print(f"  {YELLOW}ℹ{RESET} {msg}")


async def create_namespace() -> uuid.UUID:
    """Create a namespace directly in Postgres (no API endpoint for this)."""
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


async def run_smoke_test() -> bool:
    passed = 0
    failed = 0

    def record(ok: bool, msg: str):
        nonlocal passed, failed
        if ok:
            passed += 1
            _pass(msg)
        else:
            failed += 1
            _fail(msg)

    async with httpx.AsyncClient(base_url=BASE_URL, timeout=30.0) as client:

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
        try:
            ns_id = await create_namespace()
            record(True, f"Namespace created: {ns_id}")
        except Exception as e:
            record(False, f"Failed to create namespace: {e}")
            return False

        headers = {"X-Namespace-ID": str(ns_id)}

        # ── 3. Register credential ───────────────────────────────────
        print(f"\n{BOLD}3. Register credential{RESET}")
        try:
            r = await client.post(
                "/platform/credentials",
                json={"provider": "openai", "secret": "sk-smoke-test-fake", "label": "smoke-test"},
                headers=headers,
            )
            if r.status_code == 200:
                cred_id = r.json()["credential_id"]
                record(True, f"Credential registered: {cred_id}")
            else:
                record(False, f"Unexpected status {r.status_code}: {r.text}")
        except Exception as e:
            record(False, f"Failed to register credential: {e}")

        # ── 4. Create embedding profile (local_hash) ─────────────────
        print(f"\n{BOLD}4. Create embedding profile{RESET}")
        try:
            r = await client.post(
                "/platform/profiles",
                json={
                    "kind": "embedding",
                    "provider": "local_hash",
                    "model": "hash-256",
                    "label": "smoke-embed",
                    "dimensions": 256,
                    "distance_metric": "cosine",
                },
                headers=headers,
            )
            if r.status_code == 200:
                embed_profile_id = r.json()["profile_id"]
                record(True, f"Embedding profile created: {embed_profile_id}")
            else:
                record(False, f"Unexpected status {r.status_code}: {r.text}")
        except Exception as e:
            record(False, f"Failed to create embedding profile: {e}")

        # ── 5. Create LLM profile (local_echo) ───────────────────────
        print(f"\n{BOLD}5. Create LLM profile{RESET}")
        try:
            r = await client.post(
                "/platform/profiles",
                json={
                    "kind": "llm",
                    "provider": "local_echo",
                    "model": "echo-v1",
                    "label": "smoke-llm",
                },
                headers=headers,
            )
            if r.status_code == 200:
                llm_profile_id = r.json()["profile_id"]
                record(True, f"LLM profile created: {llm_profile_id}")
            else:
                record(False, f"Unexpected status {r.status_code}: {r.text}")
        except Exception as e:
            record(False, f"Failed to create LLM profile: {e}")

        # ── 6. Create collection ─────────────────────────────────────
        print(f"\n{BOLD}6. Create collection{RESET}")
        try:
            r = await client.post(
                "/collections/",
                json={
                    "name": "smoke-test-collection",
                    "strategy": "vector",
                    "embedding_profile_id": embed_profile_id,
                },
                headers=headers,
            )
            if r.status_code == 200:
                coll_id = r.json()["id"]
                record(True, f"Collection created: {coll_id}")
            else:
                record(False, f"Unexpected status {r.status_code}: {r.text}")
        except Exception as e:
            record(False, f"Failed to create collection: {e}")

        # ── 7. Ingest a chunk ────────────────────────────────────────
        print(f"\n{BOLD}7. Ingest chunk{RESET}")
        ingest_text = "The quick brown fox jumps over the lazy dog near the riverbank."
        try:
            r = await client.post(
                f"/collections/{coll_id}/ingest/chunk",
                json={"text": ingest_text},
                headers=headers,
            )
            if r.status_code == 200:
                data = r.json()
                record("chunk_hash" in data, f"Chunk ingested (hash: {data.get('chunk_hash', '?')[:12]}...)")
            else:
                record(False, f"Unexpected status {r.status_code}: {r.text}")
        except Exception as e:
            record(False, f"Failed to ingest chunk: {e}")

        # ── 8. Query the collection ──────────────────────────────────
        print(f"\n{BOLD}8. Query collection{RESET}")
        try:
            r = await client.post(
                f"/collections/{coll_id}/query",
                json={"question": "What animal jumps over the dog?"},
                headers=headers,
            )
            if r.status_code == 200:
                data = r.json()
                response_text = data.get("response", "")
                # local_echo returns the retrieved chunk directly
                has_context = "fox" in response_text.lower() or "quick brown" in response_text.lower()
                record(has_context, f"Query returned context (mode={data.get('mode', '?')})")
                _info(f"Response preview: {response_text[:120]}...")
            else:
                record(False, f"Unexpected status {r.status_code}: {r.text}")
        except Exception as e:
            record(False, f"Failed to query collection: {e}")

        # ── 9. List collections ──────────────────────────────────────
        print(f"\n{BOLD}9. List collections{RESET}")
        try:
            r = await client.get("/collections/", headers=headers)
            if r.status_code == 200:
                collections = r.json()
                record(
                    any(c["name"] == "smoke-test-collection" for c in collections),
                    f"Found {len(collections)} collection(s)",
                )
            else:
                record(False, f"Unexpected status {r.status_code}: {r.text}")
        except Exception as e:
            record(False, f"Failed to list collections: {e}")

        # ── 10. Check capabilities ───────────────────────────────────
        print(f"\n{BOLD}10. Platform capabilities{RESET}")
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

    # ── Summary ──────────────────────────────────────────────────────
    total = passed + failed
    print(f"\n{'='*50}")
    if failed == 0:
        print(f"  {GREEN}{BOLD}All {total} checks passed ✓{RESET}")
    else:
        print(f"  {RED}{BOLD}{passed}/{total} passed, {failed} failed{RESET}")
    print(f"{'='*50}\n")

    return failed == 0


async def main():
    # Check if asyncpg is available
    try:
        import asyncpg  # noqa: F401
    except ImportError:
        print(f"{RED}asyncpg is not installed. Run: uv sync --all-groups{RESET}")
        sys.exit(1)

    print(f"\n{BOLD}Graph Core Smoke Test{RESET}")
    print(f"Target: {BASE_URL}")
    print(f"Database: {DB_URL}\n")

    ok = await run_smoke_test()
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    asyncio.run(main())
