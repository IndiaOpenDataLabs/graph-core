"""Async HTTP client for the Graph Core REST API."""

import os
from typing import Any

import httpx


class GraphCoreClient:
    """Async HTTP client for the Graph Core REST API.

    Supports both admin-key and namespace-key authentication.
    """

    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        is_admin: bool = False,
    ) -> None:
        self.base_url = (
            base_url or os.getenv("GRAPH_CORE_URL") or "http://localhost:8000"
        ).rstrip("/")
        key = api_key or (
            os.getenv("PLATFORM_ADMIN_KEY")
            if is_admin
            else os.getenv("GRAPH_CORE_API_KEY")
        )
        if not key:
            raise ValueError(
                "api_key is required (set GRAPH_CORE_API_KEY env var)"
            )
        self._key = key
        self._is_admin = is_admin
        self._client = httpx.AsyncClient(
            base_url=self.base_url,
            headers={
                "Authorization": f"Bearer {key}",
                "Content-Type": "application/json",
            },
            timeout=120.0,
        )

    @property
    def api_key(self) -> str:
        return self._key

    async def close(self) -> None:
        await self._client.aclose()

    async def _request(
        self,
        method: str,
        path: str,
        *,
        json: dict | None = None,
        params: dict | None = None,
    ) -> dict[str, Any]:
        resp = await self._client.request(method, path, json=json, params=params)
        try:
            resp.raise_for_status()
            return resp.json()
        except httpx.HTTPStatusError as exc:
            body = ""
            try:
                body = resp.json()
            except Exception:
                body = resp.text
            raise GraphCoreAPIError(
                f"{exc.request.method} {exc.request.url} -> {resp.status_code}: {body}"
            ) from exc

    # -- Namespaces ---------------------------------------------------------

    async def create_namespace(self, name: str) -> dict[str, Any]:
        return await self._request("POST", "/platform/namespaces/", json={"name": name})

    async def list_namespaces(self) -> list[dict[str, Any]]:
        return await self._request("GET", "/platform/namespaces/")

    async def get_namespace_me(self) -> dict[str, Any]:
        return await self._request("GET", "/platform/namespaces/me")

    async def rotate_namespace_key(self, namespace_id: str) -> dict[str, Any]:
        return await self._request(
            "POST", f"/platform/namespaces/{namespace_id}/rotate-key"
        )

    # -- Collections --------------------------------------------------------

    async def create_collection(
        self,
        name: str,
        strategy: str = "vector",
        embedding_profile_id: str | None = None,
        llm_profile_id: str | None = None,
        default_query_mode: str | None = None,
    ) -> dict[str, Any]:
        body: dict[str, Any] = {"name": name, "strategy": strategy}
        if embedding_profile_id:
            body["embedding_profile_id"] = embedding_profile_id
        if llm_profile_id:
            body["llm_profile_id"] = llm_profile_id
        if default_query_mode:
            body["default_query_mode"] = default_query_mode
        return await self._request("POST", "/collections/", json=body)

    async def list_collections(self) -> list[dict[str, Any]]:
        return await self._request("GET", "/collections/")

    async def update_collection(
        self,
        collection_id: str,
        *,
        name: str | None = None,
        strategy: str | None = None,
        embedding_profile_id: str | None = None,
        llm_profile_id: str | None = None,
        default_query_mode: str | None = None,
        clear_llm_profile: bool = False,
        clear_default_query_mode: bool = False,
    ) -> dict[str, Any]:
        body: dict[str, Any] = {}
        if name is not None:
            body["name"] = name
        if strategy is not None:
            body["strategy"] = strategy
        if embedding_profile_id is not None:
            body["embedding_profile_id"] = embedding_profile_id
        if llm_profile_id is not None:
            body["llm_profile_id"] = llm_profile_id
        if default_query_mode is not None:
            body["default_query_mode"] = default_query_mode
        if clear_llm_profile:
            body["clear_llm_profile"] = True
        if clear_default_query_mode:
            body["clear_default_query_mode"] = True
        return await self._request(
            "PATCH",
            f"/collections/{collection_id}",
            json=body,
        )

    async def delete_collection(self, collection_id: str) -> dict[str, Any]:
        return await self._request("DELETE", f"/collections/{collection_id}")

    # -- Ingestion ----------------------------------------------------------

    async def ingest_chunk(self, collection_id: str, text: str) -> dict[str, Any]:
        return await self._request(
            "POST", f"/collections/{collection_id}/ingest/chunk", json={"text": text}
        )

    async def ingest_document(self, collection_id: str, text: str) -> dict[str, Any]:
        return await self._request(
            "POST", f"/collections/{collection_id}/ingest/doc", json={"text": text}
        )

    # -- Query --------------------------------------------------------------

    async def query_collection(
        self,
        collection_id: str,
        question: str,
        mode: str | None = None,
        llm_profile_id: str | None = None,
    ) -> dict[str, Any]:
        body: dict[str, Any] = {"question": question}
        if mode:
            body["mode"] = mode
        if llm_profile_id:
            body["llm_profile_id"] = llm_profile_id
        return await self._request(
            "POST", f"/collections/{collection_id}/query", json=body
        )

    # -- Jobs ---------------------------------------------------------------

    async def get_job(self, job_id: str) -> dict[str, Any]:
        return await self._request("GET", f"/jobs/{job_id}")

    async def list_jobs(self, limit: int = 20) -> list[dict[str, Any]]:
        return await self._request("GET", "/jobs/", params={"limit": limit})

    # -- Platform -----------------------------------------------------------

    async def get_capabilities(self) -> dict[str, Any]:
        return await self._request("GET", "/platform/capabilities")

    async def register_credential(
        self,
        provider: str,
        secret: str,
        label: str | None = None,
        base_url: str | None = None,
    ) -> dict[str, Any]:
        body: dict[str, Any] = {"provider": provider, "secret": secret}
        if label:
            body["label"] = label
        if base_url:
            body["base_url"] = base_url
        return await self._request("POST", "/platform/credentials", json=body)

    async def create_profile(
        self,
        kind: str,
        provider: str,
        model: str,
        credential_id: str | None = None,
        label: str | None = None,
        base_url: str | None = None,
        dimensions: int | None = None,
        distance_metric: str | None = None,
        max_concurrent_calls: int | None = None,
    ) -> dict[str, Any]:
        body: dict[str, Any] = {"kind": kind, "provider": provider, "model": model}
        if credential_id:
            body["credential_id"] = credential_id
        if label:
            body["label"] = label
        if base_url:
            body["base_url"] = base_url
        if dimensions is not None:
            body["dimensions"] = dimensions
        if distance_metric:
            body["distance_metric"] = distance_metric
        if max_concurrent_calls is not None:
            body["max_concurrent_calls"] = max_concurrent_calls
        return await self._request("POST", "/platform/profiles", json=body)

    async def list_embedding_profiles(self) -> list[dict[str, Any]]:
        return await self._request("GET", "/platform/embedding-profiles")

    async def list_llm_profiles(self) -> list[dict[str, Any]]:
        return await self._request("GET", "/platform/llm-profiles")


class GraphCoreAPIError(Exception):
    pass
