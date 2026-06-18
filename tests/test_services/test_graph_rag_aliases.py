import uuid
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from sqlalchemy import select

from graph_core.llm.interface import LLMProvider
from graph_core.models.collection import Collection
from graph_core.models.graph_rag import (
    GraphEntity,
    GraphRelationship,
    GraphRelationshipType,
    RelationshipTypeAlias,
)
from graph_core.models.rel_types import DEFAULT_REL_TYPE
from graph_core.services.graph import GraphService
from graph_core.services.graph.analytics import (
    analyze_collection_graph,
    build_collection_understanding,
)
from graph_core.services.graph.query import graph_rag
from graph_core.services.graph.query.graph_rag import (
    DerivedRouteProfile,
    DocumentRoutingDecision,
    GraphQueryArtifacts,
    GraphQueryState,
)
from graph_core.services.graph_rag.entity_resolver import IncrementalEntityResolver
from graph_core.services.graph_rag.extractor import LLMGraphExtractor


class _FakeEmbeddingProvider:
    dimensions = 256

    async def embed_query(self, text: str) -> list[float]:
        return [0.1, 0.2, 0.3]


class _FakeLLMProvider(LLMProvider):
    async def chat(self, messages: list[dict]) -> str:
        return "ok"

    async def chat_stream(self, messages: list[dict]):
        if False:
            yield ""

    async def structured_extract(self, prompt: str, schema: dict) -> dict:
        return {
            "label": "Shared role",
            "concept_type": "role",
            "description": "Entities with similar graph roles.",
            "aliases": ["role family"],
            "importance_reason": "Useful for higher-level navigation.",
            "member_entity_names": ["Source", "Target"],
        }


class _SessionFactory:
    def __init__(self, session):
        self._session = session

    def __call__(self):
        return self

    async def __aenter__(self):
        return self._session

    async def __aexit__(self, exc_type, exc, tb):
        return False


class _FakeHit:
    def __init__(self, distance: float, metadata: dict[str, object] | None = None):
        self.distance = distance
        self.metadata = metadata or {}


@pytest.mark.asyncio
async def test_resolve_entity_reuses_case_insensitive_canonical_match(
    db_session,
    test_graph_rag_collection,
):
    entity = GraphEntity(
        id=uuid.uuid4(),
        collection_id=test_graph_rag_collection.id,
        canonical_name="OAuth",
        primary_type="Protocol",
        description_count=0,
    )
    db_session.add(entity)
    await db_session.commit()

    resolver = IncrementalEntityResolver(
        _FakeEmbeddingProvider(),
        test_graph_rag_collection.id,
    )
    resolver._add_description_and_update_centroid = AsyncMock()
    resolver._add_or_increment_type = AsyncMock()

    result = await resolver.resolve_entity(
        db_session,
        name="OAuth",
        entity_type="Protocol",
        description="Authorization framework",
        source_chunk_hash="chunk-1",
    )

    assert result.is_new is False
    assert result.entity_id == entity.id
    assert result.canonical_name == "OAuth"
    resolver._add_or_increment_type.assert_awaited_once()


@pytest.mark.asyncio
async def test_add_or_increment_type_awaits_async_session_execute(
    test_graph_rag_collection,
):
    resolver = IncrementalEntityResolver(
        _FakeEmbeddingProvider(),
        test_graph_rag_collection.id,
    )
    session = AsyncMock()

    await resolver._add_or_increment_type(session, uuid.uuid4(), "Protocol")

    session.execute.assert_awaited_once()


@pytest.mark.asyncio
async def test_acquire_entity_lock_uses_advisory_lock(
    test_graph_rag_collection,
):
    resolver = IncrementalEntityResolver(
        _FakeEmbeddingProvider(),
        test_graph_rag_collection.id,
    )
    session = AsyncMock()
    session.bind = SimpleNamespace(dialect=SimpleNamespace(name="postgresql"))
    entity_id = uuid.uuid4()

    await resolver._acquire_entity_lock(session, entity_id)

    session.execute.assert_awaited_once()
    sql_text, params = session.execute.await_args.args
    assert "pg_advisory_xact_lock" in str(sql_text)
    assert isinstance(params["lock_key"], int)
    assert params["lock_key"] >= 0


@pytest.mark.asyncio
async def test_code_domain_resolver_keeps_distinct_symbol_names_separate(
    db_session,
    test_graph_rag_collection,
):
    existing = GraphEntity(
        id=uuid.uuid4(),
        collection_id=test_graph_rag_collection.id,
        canonical_name="llm_query",
        primary_type="FUNCTION",
        description_count=0,
    )
    db_session.add(existing)
    await db_session.commit()

    resolver = IncrementalEntityResolver(
        _FakeEmbeddingProvider(),
        test_graph_rag_collection.id,
        domain="code",
    )
    resolver._add_description_and_update_centroid = AsyncMock()
    resolver._add_or_increment_type = AsyncMock()
    resolver._vstore.search_entity_centroids = AsyncMock(
        return_value=[
            _FakeHit(
                distance=0.01,
                metadata={
                    "entity_id": str(existing.id),
                    "canonical_name": "llm_query",
                },
            )
        ]
    )

    result = await resolver.resolve_entity(
        db_session,
        name="rlm_query",
        entity_type="FUNCTION",
        description="Recursive child RLM call for deeper reasoning.",
        source_chunk_hash="chunk-1",
    )

    assert result.is_new is True
    assert result.canonical_name == "rlm_query"
    assert result.entity_id != existing.id
    resolver._vstore.search_entity_centroids.assert_not_awaited()


def test_code_domain_fuzzy_match_requires_exact_symbol_match():
    resolver = IncrementalEntityResolver(
        _FakeEmbeddingProvider(),
        uuid.uuid4(),
        domain="code",
    )

    assert resolver._fuzzy_match("llm_query", "llm_query") is True
    assert resolver._fuzzy_match("rlm_query", "llm_query") is False
    assert resolver._fuzzy_match("self._rlm_query", "self._llm_query") is False


def test_extractor_normalize_entity_name_preserves_casing():
    assert LLMGraphExtractor._normalize_entity_name("  OAuth SDK  ") == "OAuth SDK"
    assert LLMGraphExtractor._normalize_entity_name("iOS") == "iOS"
    assert LLMGraphExtractor._normalize_entity_name("JSON-LD") == "JSON-LD"


@pytest.mark.asyncio
async def test_active_dimensions_consolidates_aliases(
    db_session,
    test_graph_rag_collection,
    monkeypatch,
):
    rel_type = GraphRelationshipType(
        id=uuid.uuid4(),
        collection_id=test_graph_rag_collection.id,
        canonical_type="CALLS",
    )
    db_session.add_all(
        [
            rel_type,
            RelationshipTypeAlias(
                id=uuid.uuid4(),
                collection_id=test_graph_rag_collection.id,
                relationship_type_id=rel_type.id,
                canonical_type="CALLS",
                alias_type="INVOKES",
            ),
            GraphRelationship(
                id=uuid.uuid4(),
                collection_id=test_graph_rag_collection.id,
                source_entity_id=uuid.uuid4(),
                target_entity_id=uuid.uuid4(),
                relationship_type_id=rel_type.id,
                rel_type="CALLS",
                weight=1,
                keywords=[],
            ),
            GraphRelationship(
                id=uuid.uuid4(),
                collection_id=test_graph_rag_collection.id,
                source_entity_id=uuid.uuid4(),
                target_entity_id=uuid.uuid4(),
                relationship_type_id=rel_type.id,
                rel_type="CALLS",
                weight=1,
                keywords=[],
            ),
        ]
    )
    await db_session.commit()
    monkeypatch.setattr(graph_rag, "AsyncSessionLocal", _SessionFactory(db_session))

    dimensions = await graph_rag._active_dimensions(test_graph_rag_collection)

    assert dimensions[0] == DEFAULT_REL_TYPE
    assert dimensions.count("CALLS") == 1
    assert "INVOKES" not in dimensions


@pytest.mark.asyncio
async def test_rank_dimensions_canonicalizes_graph_edge_aliases(
    db_session,
    test_graph_rag_collection,
    monkeypatch,
):
    rel_type = GraphRelationshipType(
        id=uuid.uuid4(),
        collection_id=test_graph_rag_collection.id,
        canonical_type="CALLS",
    )
    relationship = GraphRelationship(
        id=uuid.uuid4(),
        collection_id=test_graph_rag_collection.id,
        source_entity_id=uuid.uuid4(),
        target_entity_id=uuid.uuid4(),
        relationship_type_id=rel_type.id,
        rel_type="CALLS",
        weight=1,
        keywords=[],
    )
    db_session.add_all(
        [
            rel_type,
            relationship,
            RelationshipTypeAlias(
                id=uuid.uuid4(),
                collection_id=test_graph_rag_collection.id,
                relationship_type_id=rel_type.id,
                canonical_type="CALLS",
                alias_type="INVOKES",
            ),
        ]
    )
    await db_session.commit()

    class _GraphStorage:
        async def get_node_edges_with_types(self, entity_id):
            if entity_id == "seed-2":
                return []
            return [("seed-1", "other", "INVOKES")]

    monkeypatch.setattr(graph_rag, "AsyncSessionLocal", _SessionFactory(db_session))
    monkeypatch.setattr(graph_rag, "get_graph_storage", lambda _cid: _GraphStorage())
    monkeypatch.setattr(
        graph_rag,
        "_embed_entity_query",
        AsyncMock(return_value=[0.1, 0.2, 0.3]),
    )
    monkeypatch.setattr(
        graph_rag,
        "_search_entity_seeds",
        AsyncMock(return_value=(["seed-1", "seed-2"], {"seed-1": 0.9, "seed-2": 0.8})),
    )
    monkeypatch.setattr(
        graph_rag,
        "_embed_relationship_query",
        AsyncMock(return_value=[0.1, 0.2, 0.3]),
    )
    monkeypatch.setattr(
        graph_rag,
        "_find_relevant_path_for_ranking",
        AsyncMock(return_value=(["seed-1", "seed-2"], [str(relationship.id)])),
    )
    monkeypatch.setattr(
        graph_rag,
        "_embed_relationship_queries_batch",
        AsyncMock(return_value=[[0.1, 0.2, 0.3]]),
    )
    monkeypatch.setattr(
        graph_rag._graph_rag_vectors,
        "search_relationship_embeddings",
        AsyncMock(return_value=[_FakeHit(distance=0.1)]),
    )

    ranked = await graph_rag._rank_dimensions(
        test_graph_rag_collection,
        _FakeEmbeddingProvider(),
        "Which calls matter?",
        ["CALLS", "RELATES_TO"],
        top_k=1,
    )

    assert ranked == ["CALLS"]


@pytest.mark.asyncio
async def test_derive_route_profile_uses_canonical_rel_type_alias(
    db_session,
    test_graph_rag_collection,
    monkeypatch,
):
    rel_type = GraphRelationshipType(
        id=uuid.uuid4(),
        collection_id=test_graph_rag_collection.id,
        canonical_type="CALLS",
    )
    relationship = GraphRelationship(
        id=uuid.uuid4(),
        collection_id=test_graph_rag_collection.id,
        source_entity_id=uuid.uuid4(),
        target_entity_id=uuid.uuid4(),
        relationship_type_id=rel_type.id,
        rel_type="CALLS",
        weight=1,
        keywords=[],
    )
    db_session.add_all(
        [
            rel_type,
            RelationshipTypeAlias(
                id=uuid.uuid4(),
                collection_id=test_graph_rag_collection.id,
                relationship_type_id=rel_type.id,
                canonical_type="CALLS",
                alias_type="INVOKES",
            ),
            relationship,
        ]
    )
    await db_session.commit()
    monkeypatch.setattr(graph_rag, "AsyncSessionLocal", _SessionFactory(db_session))

    profile = await graph_rag._derive_route_profile(
        test_graph_rag_collection,
        GraphQueryState(
            discovered_entity_ids=set(),
            entity_relevance={},
            traversed_rel_ids=[str(relationship.id)],
            rel_score_cache={},
            rel_combined_score_cache={str(relationship.id): 2.0},
        ),
    )

    assert profile.primary_route == "hub"
    assert "CALLS" in profile.rel_type_scores
    assert "INVOKES" not in profile.rel_type_scores


@pytest.mark.asyncio
async def test_find_relevant_path_records_combined_scores_for_bridge_edges(
    monkeypatch,
):
    first_rel_id = str(uuid.uuid4())
    bridge_rel_id = str(uuid.uuid4())

    class _GraphStorage:
        async def get_node_edges(self, node_id, rel_types=None):
            if node_id == "source":
                return [("source", "mid")]
            if node_id == "mid":
                return [("source", "mid"), ("mid", "target")]
            return []

        async def get_edge(self, src, tgt, rel_types=None):
            edge_ids = {
                ("source", "mid"): first_rel_id,
                ("mid", "target"): bridge_rel_id,
            }
            rel_id = edge_ids.get((src, tgt))
            if rel_id is None:
                return None
            return {"id": rel_id, "weight": 1, "keywords": []}

    monkeypatch.setattr(
        graph_rag,
        "_score_relationship",
        AsyncMock(side_effect=lambda *args, **kwargs: 0.9),
    )

    rel_combined_score_cache: dict[str, float] = {}
    path = await graph_rag._find_relevant_path(
        _GraphStorage(),
        collection=type("_Collection", (), {"id": uuid.uuid4()})(),
        relationship_query_embedding=[0.1, 0.2, 0.3],
        source_id="source",
        target_id="target",
        rel_score_cache={},
        rel_combined_score_cache=rel_combined_score_cache,
    )

    assert path == (["source", "mid", "target"], [first_rel_id, bridge_rel_id])
    assert rel_combined_score_cache[bridge_rel_id] > 0.0


@pytest.mark.asyncio
async def test_entity_first_state_prioritizes_high_scoring_edges(
    db_session,
    test_graph_rag_collection,
    monkeypatch,
):
    entity_ids = {"seed": uuid.uuid4(), "high": uuid.uuid4()}
    for idx in range(6):
        entity_ids[f"low-{idx}"] = uuid.uuid4()
    db_session.add_all(
        [
            GraphEntity(
                id=entity_id,
                collection_id=test_graph_rag_collection.id,
                canonical_name=name,
                primary_type="concept",
                description_count=0,
            )
            for name, entity_id in entity_ids.items()
        ]
    )
    await db_session.commit()
    monkeypatch.setattr(graph_rag, "AsyncSessionLocal", _SessionFactory(db_session))
    monkeypatch.setattr(
        graph_rag,
        "_search_entity_seeds",
        AsyncMock(return_value=([str(entity_ids["seed"])], {str(entity_ids["seed"]): 0.1})),
    )
    monkeypatch.setattr(
        graph_rag._graph_rag_vectors,
        "search_relationship_embeddings",
        AsyncMock(return_value=[]),
    )

    high_rel_id = str(uuid.uuid4())
    low_rel_ids = [str(uuid.uuid4()) for _ in range(6)]

    class _GraphStorage:
        async def get_node_edges(self, node_id, rel_types=None):
            if node_id != str(entity_ids["seed"]):
                return []
            edges = [(str(entity_ids["seed"]), str(entity_ids["high"]))]
            edges.extend(
                (str(entity_ids["seed"]), str(entity_ids[f"low-{idx}"]))
                for idx in range(6)
            )
            return edges

        async def get_edge(self, src, tgt, rel_types=None):
            if tgt == str(entity_ids["high"]):
                return {"id": high_rel_id, "weight": 1, "keywords": []}
            for idx in range(6):
                if tgt == str(entity_ids[f"low-{idx}"]):
                    return {"id": low_rel_ids[idx], "weight": 1, "keywords": []}
            return None

    async def _score_relationship(_collection, _embedding, rel_id, _cache, *, top_k=4):
        if rel_id == high_rel_id:
            return 0.95
        return 0.51

    monkeypatch.setattr(graph_rag, "get_graph_storage", lambda _cid: _GraphStorage())
    monkeypatch.setattr(graph_rag, "_score_relationship", AsyncMock(side_effect=_score_relationship))
    monkeypatch.setattr(graph_rag, "_combined_edge_score", lambda sim, _edge_props, _query_tokens: sim)

    state = await graph_rag._entity_first_state(
        question="Find the most important link",
        collection=test_graph_rag_collection,
        entity_query_embedding=[0.1, 0.2, 0.3],
        relationship_query_embedding=[0.1, 0.2, 0.3],
    )

    assert high_rel_id in state.traversed_rel_ids


@pytest.mark.asyncio
async def test_resolve_rel_type_creates_canonical_relationship_type(
    db_session,
    test_graph_rag_collection,
):
    resolver = IncrementalEntityResolver(
        _FakeEmbeddingProvider(),
        test_graph_rag_collection.id,
    )
    resolver._vstore.ensure_prefix_embeddings_table = AsyncMock()
    resolver._vstore.load_all_prefix_embeddings = AsyncMock(return_value={})
    resolver._vstore.upsert_prefix_embedding = AsyncMock()

    result = await resolver._resolve_rel_type(db_session, "invokes")
    await db_session.commit()

    assert result.canonical_type == "INVOKES"
    rows = (
        await db_session.execute(
            select(GraphRelationshipType).where(
                GraphRelationshipType.id == result.relationship_type_id
            )
        )
    ).scalars().all()
    assert len(rows) == 1
    alias_rows = (
        await db_session.execute(
            select(RelationshipTypeAlias).where(
                RelationshipTypeAlias.relationship_type_id == result.relationship_type_id
            )
        )
    ).scalars().all()
    assert {row.alias_type for row in alias_rows} == {"INVOKES"}


@pytest.mark.asyncio
async def test_relationship_type_canonical_is_reelected_by_frequency(
    db_session,
    test_graph_rag_collection,
):
    resolver = IncrementalEntityResolver(
        _FakeEmbeddingProvider(),
        test_graph_rag_collection.id,
    )
    resolver._graph_storage.relabel_edges = AsyncMock()

    rel_type = GraphRelationshipType(
        id=uuid.uuid4(),
        collection_id=test_graph_rag_collection.id,
        canonical_type="EXPLAINS",
    )
    source = GraphEntity(
        id=uuid.uuid4(),
        collection_id=test_graph_rag_collection.id,
        canonical_name="Source",
        primary_type="concept",
        description_count=0,
    )
    target = GraphEntity(
        id=uuid.uuid4(),
        collection_id=test_graph_rag_collection.id,
        canonical_name="Target",
        primary_type="concept",
        description_count=0,
    )
    relationship = GraphRelationship(
        id=uuid.uuid4(),
        collection_id=test_graph_rag_collection.id,
        source_entity_id=source.id,
        target_entity_id=target.id,
        relationship_type_id=rel_type.id,
        rel_type="EXPLAINS",
        weight=1,
        keywords=[],
    )
    db_session.add_all([rel_type, source, target, relationship])
    await db_session.commit()

    await resolver._add_relationship_type_alias(
        db_session, rel_type.id, "EXPLAINS", "EXPLAINS"
    )
    await resolver._add_relationship_type_alias(
        db_session, rel_type.id, "EXPLAINS", "CAUSES"
    )
    await resolver._add_relationship_type_alias(
        db_session, rel_type.id, "EXPLAINS", "CAUSES"
    )

    updated = await resolver._reelect_relationship_type_canonical(db_session, rel_type)
    await db_session.commit()

    assert updated.canonical_type == "CAUSES"
    refreshed = await db_session.get(GraphRelationship, relationship.id)
    assert refreshed is not None
    assert refreshed.rel_type == "CAUSES"
    resolver._graph_storage.relabel_edges.assert_awaited_once()


@pytest.mark.asyncio
async def test_analyze_collection_graph_uses_canonical_relationship_types(
    db_session,
    test_graph_rag_collection,
):
    source = GraphEntity(
        id=uuid.uuid4(),
        collection_id=test_graph_rag_collection.id,
        canonical_name="Source",
        primary_type="concept",
        description_count=0,
    )
    target = GraphEntity(
        id=uuid.uuid4(),
        collection_id=test_graph_rag_collection.id,
        canonical_name="Target",
        primary_type="concept",
        description_count=0,
    )
    rel_type = GraphRelationshipType(
        id=uuid.uuid4(),
        collection_id=test_graph_rag_collection.id,
        canonical_type="CALLS",
    )
    relationship = GraphRelationship(
        id=uuid.uuid4(),
        collection_id=test_graph_rag_collection.id,
        source_entity_id=source.id,
        target_entity_id=target.id,
        relationship_type_id=rel_type.id,
        rel_type="CALLS",
        weight=1,
        keywords=[],
    )
    db_session.add_all([source, target, rel_type, relationship])
    await db_session.commit()

    analysis = await analyze_collection_graph(test_graph_rag_collection.id)

    assert analysis["relationship_records"][0]["rel_type"] == "CALLS"


@pytest.mark.asyncio
async def test_build_collection_understanding_tolerates_missing_relationship_count():
    source_id = str(uuid.uuid4())
    target_id = str(uuid.uuid4())
    analysis = {
        "collection": {
            "id": str(uuid.uuid4()),
            "name": "Test Collection",
            "namespace_id": str(uuid.uuid4()),
            "strategy": "graph_rag",
        },
        "relationship_records": [
            {
                "id": str(uuid.uuid4()),
                "source_id": source_id,
                "source_name": "Source",
                "target_id": target_id,
                "target_name": "Target",
                "rel_type": "CALLS",
                "weight": 2,
            }
        ],
        "role_groups": [
            {
                "group_id": "role:0",
                "size": 2,
                "node_ids": [source_id, target_id],
                "node_names": ["Source", "Target"],
                "avg_cosine": 0.5,
                "avg_jaccard": 0.5,
                "total_overlap": 1,
                "pair_metrics": [
                    {
                        "a": source_id,
                        "b": target_id,
                        "overlap": 1,
                        "cosine": 0.5,
                        "jaccard": 0.5,
                    }
                ],
                "top_rel_types": ["CALLS"],
                "representative_edges": [
                    {
                        "source_name": "Source",
                        "target_name": "Target",
                        "weight": 2,
                    }
                ],
            }
        ],
        "entity_aliases_by_id": {},
    }

    understanding = await build_collection_understanding(
        analysis,
        llm_provider=_FakeLLMProvider(),
    )

    assert understanding["nodes"]


@pytest.mark.asyncio
async def test_materialize_meta_collection_persists_derived_chunks(
    test_graph_rag_collection,
):
    service = GraphService()
    service._resolve_collection_embedding_provider = AsyncMock(
        return_value=_FakeEmbeddingProvider()
    )
    graph_storage = type(
        "_GraphStorage",
        (),
        {
            "upsert_nodes": AsyncMock(),
            "upsert_edges": AsyncMock(),
        },
    )()
    service._graph_storage = lambda _collection_id: graph_storage
    service._vector_store.upsert_chunks = AsyncMock()
    service._graph_rag_vectors.upsert_chunk_embedding = AsyncMock()

    understanding = {
        "nodes": [],
        "edges": [],
        "chunks": [
            {
                "chunk_hash": "derived-1",
                "chunk_index": 0,
                "content": "Derived concept summary",
                "metadata": {"memory_type": "derived_graph", "derived_kind": "concept"},
            }
        ],
    }

    await service._materialize_meta_collection(test_graph_rag_collection, understanding)

    service._vector_store.upsert_chunks.assert_awaited_once()
    upsert_kwargs = service._vector_store.upsert_chunks.await_args.kwargs
    assert upsert_kwargs["collection_id"] == test_graph_rag_collection.id
    assert upsert_kwargs["namespace_id"] == test_graph_rag_collection.namespace_id
    assert upsert_kwargs["chunks"][0]["chunk_hash"] == "derived-1"
    assert upsert_kwargs["chunks"][0]["metadata"]["memory_type"] == "derived_graph"
    service._graph_rag_vectors.upsert_chunk_embedding.assert_awaited_once()


@pytest.mark.asyncio
async def test_load_meta_collections_returns_all_levels(
    db_session,
    test_graph_rag_collection,
):
    l1 = Collection(
        id=uuid.uuid4(),
        namespace_id=test_graph_rag_collection.namespace_id,
        name="graph-rag-collection__meta__l1",
        strategy="custom_graph_rag",
        embedding_dimensions=256,
    )
    l2 = Collection(
        id=uuid.uuid4(),
        namespace_id=test_graph_rag_collection.namespace_id,
        name="graph-rag-collection__meta__l2",
        strategy="custom_graph_rag",
        embedding_dimensions=256,
    )
    db_session.add_all([l1, l2])
    await db_session.commit()

    collections = await graph_rag._load_meta_collections(test_graph_rag_collection)

    assert [collection.name for collection in collections] == [
        "graph-rag-collection__meta__l1",
        "graph-rag-collection__meta__l2",
    ]


@pytest.mark.asyncio
async def test_load_meta_collections_prefers_leveled_l1_over_legacy(
    db_session,
    test_graph_rag_collection,
):
    legacy_l1 = Collection(
        id=uuid.uuid4(),
        namespace_id=test_graph_rag_collection.namespace_id,
        name="graph-rag-collection__meta",
        strategy="custom_graph_rag",
        embedding_dimensions=256,
    )
    leveled_l1 = Collection(
        id=uuid.uuid4(),
        namespace_id=test_graph_rag_collection.namespace_id,
        name="graph-rag-collection__meta__l1",
        strategy="custom_graph_rag",
        embedding_dimensions=256,
    )
    db_session.add_all([legacy_l1, leveled_l1])
    await db_session.commit()

    collections = await graph_rag._load_meta_collections(test_graph_rag_collection)

    assert [collection.name for collection in collections] == [
        "graph-rag-collection__meta__l1",
    ]


@pytest.mark.asyncio
async def test_graph_rag_query_does_not_apply_document_routing_to_meta_collections(
    monkeypatch,
    test_graph_rag_collection,
):
    meta_collection = Collection(
        id=uuid.uuid4(),
        namespace_id=test_graph_rag_collection.namespace_id,
        name="graph-rag-collection__meta__l1",
        strategy="custom_graph_rag",
        embedding_dimensions=256,
    )
    routed_document_id = uuid.uuid4()
    build_calls: list[dict[str, object]] = []

    async def _fake_resolve_document_routing(*args, **kwargs):
        return DocumentRoutingDecision(
            use_all_documents=False,
            document_ids=[routed_document_id],
        )

    async def _fake_build_graph_query_artifacts(
        question,
        collection,
        namespace_id,
        mode,
        llm_profile_id,
        document_ids=None,
    ):
        build_calls.append(
            {
                "collection_name": collection.name,
                "document_ids": document_ids,
            }
        )
        return GraphQueryArtifacts(
            context=f"context for {collection.name}",
            entities_used=[],
            relationships_used=[],
            rel_context="",
            route_profile=DerivedRouteProfile(
                primary_route="entity",
                route_scores={},
                rel_type_scores={},
            ),
        )

    async def _fake_load_meta_collections(collection):
        return [meta_collection]

    async def _fake_answer_from_context(*args, **kwargs):
        return "ok"

    monkeypatch.setattr(
        graph_rag,
        "_resolve_document_routing",
        _fake_resolve_document_routing,
    )
    monkeypatch.setattr(
        graph_rag,
        "_build_graph_query_artifacts",
        _fake_build_graph_query_artifacts,
    )
    monkeypatch.setattr(graph_rag, "_load_meta_collections", _fake_load_meta_collections)
    monkeypatch.setattr(
        graph_rag,
        "_answer_from_context",
        _fake_answer_from_context,
    )

    await graph_rag.graph_rag_query(
        "Which document mentions OAuth?",
        test_graph_rag_collection,
        test_graph_rag_collection.namespace_id,
    )

    assert build_calls == [
        {
            "collection_name": test_graph_rag_collection.name,
            "document_ids": [routed_document_id],
        },
        {
            "collection_name": meta_collection.name,
            "document_ids": None,
        },
    ]
