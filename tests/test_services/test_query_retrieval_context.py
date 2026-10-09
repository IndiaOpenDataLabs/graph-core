"""Retrieved content stays separate from answers and survives async job storage."""

import importlib
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from graph_core.database import AsyncSessionLocal
from graph_core.models.job import Job
from graph_core.services.graph.query.vector import QueryResult

vector = importlib.import_module("graph_core.services.graph.query.vector")
lightrag = importlib.import_module("graph_core.services.graph.query.lightrag")
graph_rag = importlib.import_module("graph_core.services.graph.query.graph_rag")


@pytest.mark.asyncio
@pytest.mark.parametrize("include_meta", [False, True])
async def test_graph_context_matches_generation_input(
    monkeypatch, test_graph_rag_collection, include_meta
):
    base = SimpleNamespace(
        context="Context:\nEntities:\nAda: mathematician",
        entities_used=["Ada"],
        relationships_used=["Ada -> wrote -> Notes"],
        rel_context="Ada wrote Notes",
        route_profile=SimpleNamespace(primary_route="entities"),
    )
    meta = SimpleNamespace(
        context="Context:\nDerived understanding: history of mathematics",
        entities_used=["Mathematics"],
        relationships_used=[],
    )
    monkeypatch.setattr(
        graph_rag,
        "_resolve_document_routing",
        AsyncMock(return_value=SimpleNamespace(use_all_documents=True)),
    )
    monkeypatch.setattr(
        graph_rag, "_build_graph_query_artifacts", AsyncMock(side_effect=[base, meta])
    )
    monkeypatch.setattr(
        graph_rag,
        "_load_meta_collections",
        AsyncMock(
            return_value=(
                [SimpleNamespace(name="research__meta__l1")] if include_meta else []
            )
        ),
    )
    answer = AsyncMock(return_value="Generated answer")
    monkeypatch.setattr(graph_rag, "_answer_from_context", answer)
    result = await graph_rag.graph_rag_query(
        "Question",
        test_graph_rag_collection,
        test_graph_rag_collection.namespace_id,
        "local",
    )
    assert result.retrieval_context == answer.call_args.args[3]
    assert "Ada: mathematician" in result.retrieval_context
    if include_meta:
        assert (
            "Derived understanding: history of mathematics" in result.retrieval_context
        )
    assert result.response == "Generated answer"


@pytest.mark.asyncio
async def test_vector_result_includes_retrieved_chunks(monkeypatch, test_collection):
    provider = SimpleNamespace(embed_query=AsyncMock(return_value=[0.1]))
    monkeypatch.setattr(
        vector, "_resolve_embedding_provider", AsyncMock(return_value=provider)
    )
    monkeypatch.setattr(
        vector._vector_store,
        "query_chunks",
        AsyncMock(
            return_value=[{"content": "First source"}, {"content": "Second source"}]
        ),
    )
    answer = AsyncMock(return_value="Generated answer")
    monkeypatch.setattr(vector, "generate_vector_answer", answer)
    result = await vector.vector_query(
        "Question", test_collection, test_collection.namespace_id, "local"
    )
    assert result.response == "Generated answer"
    assert (
        result.retrieval_context == "Chunk 1:\nFirst source\n\nChunk 2:\nSecond source"
    )
    assert answer.call_args.kwargs["chunks"] == ["First source", "Second source"]


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["hybrid", "mix"])
async def test_combined_lightrag_exposes_each_retrieval_path(
    monkeypatch, test_collection, mode
):
    for name, context in [
        ("local", "Local source"),
        ("global", "Global source"),
        ("naive", "Vector source"),
    ]:
        monkeypatch.setattr(
            lightrag,
            f"_lightrag_query_{name}",
            AsyncMock(
                return_value=QueryResult(
                    "Answer", [name], [], retrieval_context=context
                )
            ),
        )
    fn = getattr(lightrag, f"_lightrag_query_{mode}")
    result = await fn("Question", test_collection, ([], []), object(), object())
    assert "Local source" in result.retrieval_context
    assert "Global source" in result.retrieval_context
    if mode == "mix":
        assert "Vector source" in result.retrieval_context


@pytest.mark.asyncio
async def test_query_job_persists_context_and_exposes_it_via_rest(
    service, test_collection, async_client, monkeypatch
):
    result = QueryResult(
        "Answer",
        ["Ada"],
        ["Ada -> wrote -> Notes"],
        mode="mix",
        retrieval_context="Entities:\nAda: mathematician",
    )
    monkeypatch.setattr(service, "query", AsyncMock(return_value=result))
    async with AsyncSessionLocal() as session:
        job = Job(
            namespace_id=test_collection.namespace_id,
            collection_id=test_collection.id,
            job_type="query",
            status="pending",
            payload={"question": "Who is Ada?"},
        )
        session.add(job)
        await session.commit()
        job_id = job.id
    await service.run_query_job(job_id)
    response = await async_client.get(f"/jobs/{job_id}/result")
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "completed"
    assert payload["result"]["response"] == "Answer"
    assert payload["result"]["retrieval_context"] == result.retrieval_context
    assert payload["payload"]["question"] == "Who is Ada?"
