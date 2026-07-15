import pytest

from graph_core.services.graph.query import graph_rag


class _CollectionScopePlanner:
    async def structured_extract(self, prompt, schema):
        return {
            "operation": "describe",
            "scope": "collection",
            "anchors": ["yoga"],
            "requested_fields": ["definition"],
            "output_shape": "prose",
        }


class _CapturingProvider:
    def __init__(self):
        self.messages = None

    async def chat(self, messages):
        self.messages = messages
        return "bounded"


@pytest.mark.asyncio
async def test_query_plan_cannot_expand_definition_to_collection_scope(monkeypatch):
    async def resolve_provider(*args, **kwargs):
        return _CollectionScopePlanner()

    monkeypatch.setattr(graph_rag, "_resolve_llm_provider", resolve_provider)

    plan = await graph_rag._plan_graph_query("What is yoga?", None, None)

    assert plan.scope == "anchored"


@pytest.mark.asyncio
async def test_query_plan_preserves_explicit_collection_scope(monkeypatch):
    async def resolve_provider(*args, **kwargs):
        return _CollectionScopePlanner()

    monkeypatch.setattr(graph_rag, "_resolve_llm_provider", resolve_provider)

    plan = await graph_rag._plan_graph_query("List all yoga definitions", None, None)

    assert plan.scope == "collection"


def test_fallback_query_plan_treats_counts_as_collection_aggregates():
    plan = graph_rag._fallback_graph_query_plan("How many yoga practices are there?")

    assert plan.operation == "aggregate"
    assert plan.scope == "collection"


def test_context_budget_preserves_goal_contract():
    context = (
        ("Evidence statement with source provenance.\n" * 500)
        + "Goal-Directed Answer Contract:\n"
        + ("proved_claims: yoga is a disciplined practice\n" * 200)
    )

    bounded, original_tokens = graph_rag._budget_graph_context(context, 500)

    assert original_tokens > 500
    assert len(graph_rag._CONTEXT_TOKEN_ENCODING.encode(bounded)) <= 500
    assert "Context truncated" in bounded
    assert "Goal-Directed Answer Contract:" in bounded


def test_context_budget_prefers_later_direct_evidence_over_early_navigation():
    context = (
        "Supporting Graph Activation (navigation context, not proof):\n"
        + ("navigation-only material\n" * 300)
        + "\n\nSemantic Proposition Evidence:\n"
        + "- Yoga is disciplined practice [predicate=DEFINES; score=0.9900]\n"
        + "  Statement: Yoga joins disciplined methods toward integration.\n"
        + "\n\nGoal-Directed Answer Contract:\n"
        + '{"proved_claims":["Yoga is disciplined practice"]}'
    )

    bounded, _ = graph_rag._budget_graph_context(context, 250)

    assert "Yoga joins disciplined methods" in bounded
    assert "Goal-Directed Answer Contract:" in bounded
    assert "navigation-only material" not in bounded


def test_context_budget_uses_routing_score_between_direct_context_blocks():
    context = (
        "Context 1: weak\n"
        "Routing score: 0.1000\n"
        "Assertions:\n"
        + ("- weak peripheral evidence\n" * 120)
        + "Context 2: strong\n"
        "Routing score: 0.9500\n"
        "Assertions:\n"
        "- direct definition of yoga\n"
    )

    bounded, _ = graph_rag._budget_graph_context(context, 180)

    assert "Context 2: strong" in bounded
    assert "direct definition of yoga" in bounded
    assert "Context 1: weak" not in bounded


@pytest.mark.asyncio
async def test_answer_never_sends_unbounded_context(monkeypatch):
    provider = _CapturingProvider()

    async def resolve_provider(*args, **kwargs):
        return provider

    monkeypatch.setattr(graph_rag, "_resolve_llm_provider", resolve_provider)
    monkeypatch.setattr(graph_rag.settings, "graph_rag_max_context_tokens", 300)
    context = "Evidence with provenance.\n" * 1_000

    result = await graph_rag._answer_from_context(
        "What is yoga?",
        None,
        None,
        context,
        "fallback",
    )

    assert result == "bounded"
    assert provider.messages is not None
    sent_context = provider.messages[1]["content"].rsplit("\n\nQuestion:", 1)[0]
    assert len(graph_rag._CONTEXT_TOKEN_ENCODING.encode(sent_context)) <= 300
