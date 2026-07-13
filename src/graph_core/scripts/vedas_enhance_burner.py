"""Build versioned structural analytics for the incrementally ingested graph.

This burner is deliberately independent of the existing LLM-based enhancement
job. It computes deterministic named projections and persists their metrics for
the latest published graph version.

Usage:
    uv run python -m graph_core.scripts.vedas_enhance_burner
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import uuid
from collections import Counter
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

import networkx as nx
from scipy.sparse.linalg import eigsh
from sqlalchemy import delete, select, text
from sqlalchemy.dialects.postgresql import insert as pg_insert

from graph_core.database import AsyncSessionLocal, _uuid_for_sql
from graph_core.embedding.interface import EmbeddingProvider
from graph_core.models.collection import Collection
from graph_core.models.graph_rag import GraphEntity, GraphRelationship
from graph_core.models.incremental_graph import (
    GraphChunkSegment,
    GraphCommunity,
    GraphCommunityMembership,
    GraphComponentMetric,
    GraphFrameArgument,
    GraphNodeMetric,
    GraphProjectionSnapshot,
    GraphSemanticFrame,
    GraphVersion,
)
from graph_core.scripts.vedas_ingest_burner import (
    SemanticFrameRow,
    compile_semantic_frames,
    deterministic_uuid,
    extraction_from_payload,
    vector_literal,
)
from graph_core.services.graph.ingestion.chunk_processor import (
    _resolve_embedding_provider,
)
from graph_core.storage.vector_tables import (
    ensure_semantic_frame_table,
    get_collection_dimensions,
    table_name,
)

DEFAULT_COLLECTION_ID = uuid.UUID("855f1950-ff65-4f47-9549-9a20f9a1332d")

PROVENANCE_NODE_TYPES = {
    "ASSERTION",
    "CONTEXT",
    "EVIDENCE_CHUNK",
    "SOURCE_DOCUMENT",
    "SOURCE_FOLDER",
    "SOURCE_SECTION",
}
REASONING_NODE_TYPES = {"CONDITION", "EXCEPTION", "PROPOSITION", "RULE", "SCOPE"}
REASONING_EDGE_TYPES = {
    "ANTECEDENT_OF",
    "APPLIES_TO",
    "BLOCKS",
    "CONCLUDES",
    "CONDITION",
    "EXCEPTION",
    "OBJECT",
    "SUBJECT",
    "SUPPORTS",
}


@dataclass(frozen=True)
class Node:
    id: uuid.UUID
    node_type: str
    name: str = ""


@dataclass(frozen=True)
class Edge:
    source_id: uuid.UUID
    target_id: uuid.UUID
    rel_type: str
    weight: int


PROJECTION_SPECS: tuple[dict[str, Any], ...] = (
    {
        "name": "semantic_entities_directed",
        "directed": True,
        "node_policy": "semantic_entities",
        "edge_policy": "non_reasoning",
        "parallel_edges": "sum_weight",
        "self_loops": "exclude",
    },
    {
        "name": "semantic_affinity_undirected",
        "directed": False,
        "node_policy": "semantic_entities",
        "edge_policy": "non_reasoning",
        "parallel_edges": "sum_weight",
        "self_loops": "exclude",
    },
    {
        "name": "rule_dependency_directed",
        "directed": True,
        "node_policy": "reasoning_and_arguments",
        "edge_policy": "reasoning_only",
        "parallel_edges": "sum_weight",
        "self_loops": "exclude",
    },
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute versioned structural analytics for vedas2"
    )
    parser.add_argument(
        "--collection-id", type=uuid.UUID, default=DEFAULT_COLLECTION_ID
    )
    parser.add_argument(
        "--force", action="store_true", help="Rebuild completed snapshots"
    )
    return parser.parse_args()


def spec_hash(spec: dict[str, Any]) -> str:
    encoded = json.dumps(spec, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def build_projection(
    nodes: list[Node], edges: list[Edge], spec: dict[str, Any]
) -> nx.Graph | nx.DiGraph:
    graph: nx.Graph | nx.DiGraph
    graph = nx.DiGraph() if spec["directed"] else nx.Graph()
    node_by_id = {node.id: node for node in nodes}

    def include_node(node: Node) -> bool:
        node_type = node.node_type.upper()
        if node_type in PROVENANCE_NODE_TYPES or node_type.startswith("MENTION_"):
            return False
        if spec["node_policy"] == "semantic_entities":
            return node_type not in REASONING_NODE_TYPES
        return True

    included = {node.id for node in nodes if include_node(node)}
    graph.add_nodes_from(included)
    for edge in edges:
        if edge.source_id == edge.target_id:
            continue
        if edge.source_id not in included or edge.target_id not in included:
            continue
        is_reasoning = edge.rel_type.upper() in REASONING_EDGE_TYPES
        if spec["edge_policy"] == "reasoning_only" and not is_reasoning:
            continue
        if spec["edge_policy"] == "non_reasoning" and is_reasoning:
            continue
        if edge.source_id not in node_by_id or edge.target_id not in node_by_id:
            continue
        weight = max(1, int(edge.weight or 1))
        if graph.has_edge(edge.source_id, edge.target_id):
            graph[edge.source_id][edge.target_id]["weight"] += weight
        else:
            graph.add_edge(edge.source_id, edge.target_id, weight=weight)
    return graph


def _ranked(values: dict[uuid.UUID, float]) -> list[tuple[uuid.UUID, float, int]]:
    ordered = sorted(values.items(), key=lambda item: (-item[1], str(item[0])))
    return [
        (node_id, float(value), rank)
        for rank, (node_id, value) in enumerate(ordered, 1)
    ]


def _spectral_diagnostics(graph: nx.Graph) -> dict[str, float | str]:
    if graph.number_of_nodes() < 2 or graph.number_of_edges() == 0:
        return {"status": "insufficient_graph"}
    largest_nodes = max(nx.connected_components(graph), key=len)
    component = graph.subgraph(largest_nodes)
    if component.number_of_nodes() < 3:
        return {"status": "insufficient_component"}
    laplacian = nx.normalized_laplacian_matrix(component, weight="weight")
    eigenvalues = sorted(
        float(value)
        for value in eigsh(laplacian, k=2, which="SM", return_eigenvectors=False)
    )
    return {
        "status": "computed",
        "largest_component_nodes": float(component.number_of_nodes()),
        "fiedler_value": eigenvalues[1],
        "spectral_gap": eigenvalues[1] - eigenvalues[0],
    }


def compute_analytics(graph: nx.Graph | nx.DiGraph) -> dict[str, Any]:
    undirected = graph.to_undirected()
    node_metrics: dict[str, dict[uuid.UUID, float]] = {}
    if graph.is_directed():
        directed = graph
        node_metrics["in_degree"] = dict(directed.in_degree(weight="weight"))
        node_metrics["out_degree"] = dict(directed.out_degree(weight="weight"))
        strongly_connected = list(nx.strongly_connected_components(directed))
    else:
        node_metrics["degree"] = dict(graph.degree(weight="weight"))
        strongly_connected = []

    if graph.number_of_nodes():
        node_metrics["pagerank"] = nx.pagerank(graph, weight="weight")
        node_metrics["harmonic_centrality"] = nx.harmonic_centrality(undirected)
        node_metrics["clustering"] = nx.clustering(undirected, weight="weight")
        if undirected.number_of_edges():
            node_metrics["core_number"] = {
                node_id: float(value)
                for node_id, value in nx.core_number(undirected).items()
            }
            sample = min(
                graph.number_of_nodes(),
                max(10, int(math.sqrt(graph.number_of_nodes()))),
            )
            node_metrics["betweenness_approx"] = nx.betweenness_centrality(
                graph, k=sample, normalized=True, weight=None, seed=0
            )
    articulation = (
        set(nx.articulation_points(undirected))
        if undirected.number_of_edges()
        else set()
    )
    node_metrics["is_articulation"] = {
        node_id: float(node_id in articulation) for node_id in graph.nodes
    }

    communities = (
        list(nx.community.louvain_communities(undirected, weight="weight", seed=0))
        if undirected.number_of_edges()
        else [{node_id} for node_id in undirected.nodes]
    )
    components = list(nx.connected_components(undirected))
    component_metrics: list[dict[str, Any]] = [
        {
            "key": "__graph__",
            "metric": "component_count",
            "value": float(len(components)),
            "metadata": {"scc_count": len(strongly_connected)},
        },
        {
            "key": "__graph__",
            "metric": "bridge_count",
            "value": float(len(list(nx.bridges(undirected))))
            if undirected.number_of_edges()
            else 0.0,
            "metadata": None,
        },
    ]
    spectral = _spectral_diagnostics(undirected)
    if spectral.get("status") == "computed":
        for metric in ("fiedler_value", "spectral_gap"):
            component_metrics.append(
                {
                    "key": "largest_component",
                    "metric": metric,
                    "value": float(spectral[metric]),
                    "metadata": None,
                }
            )
    return {
        "node_metrics": node_metrics,
        "communities": communities,
        "component_metrics": component_metrics,
        "diagnostics": {
            "component_count": len(components),
            "scc_count": len(strongly_connected),
            "community_count": len(communities),
            "spectral": spectral,
        },
    }


async def load_graph(
    collection_id: uuid.UUID,
) -> tuple[Collection, GraphVersion, list[Node], list[Edge]]:
    async with AsyncSessionLocal() as session:
        collection = await session.get(Collection, collection_id)
        if collection is None:
            raise ValueError(f"Collection {collection_id} not found")
        version = (
            await session.execute(
                select(GraphVersion)
                .where(GraphVersion.collection_id == collection_id)
                .order_by(GraphVersion.version.desc())
                .limit(1)
            )
        ).scalar_one_or_none()
        if version is None:
            raise ValueError("Collection has no published graph version")
        node_rows = (
            await session.execute(
                select(
                    GraphEntity.id,
                    GraphEntity.canonical_name,
                    GraphEntity.primary_type,
                ).where(GraphEntity.collection_id == collection_id)
            )
        ).all()
        edge_rows = (
            await session.execute(
                select(
                    GraphRelationship.source_entity_id,
                    GraphRelationship.target_entity_id,
                    GraphRelationship.rel_type,
                    GraphRelationship.weight,
                ).where(GraphRelationship.collection_id == collection_id)
            )
        ).all()
    nodes = [
        Node(row.id, str(row.primary_type or ""), str(row.canonical_name))
        for row in node_rows
    ]
    edges = [
        Edge(row.source_entity_id, row.target_entity_id, row.rel_type, row.weight)
        for row in edge_rows
    ]
    return collection, version, nodes, edges


async def persist_projection(
    version: GraphVersion,
    spec: dict[str, Any],
    graph: nx.Graph | nx.DiGraph,
    analytics: dict[str, Any],
    *,
    force: bool,
) -> str:
    digest = spec_hash(spec)
    async with AsyncSessionLocal() as session:
        existing = (
            await session.execute(
                select(GraphProjectionSnapshot).where(
                    GraphProjectionSnapshot.graph_version_id == version.id,
                    GraphProjectionSnapshot.name == spec["name"],
                    GraphProjectionSnapshot.spec_hash == digest,
                )
            )
        ).scalar_one_or_none()
        if existing is not None and existing.status == "completed" and not force:
            return "skipped"
        if existing is not None:
            await session.execute(
                delete(GraphProjectionSnapshot).where(
                    GraphProjectionSnapshot.id == existing.id
                )
            )
            await session.flush()

        snapshot = GraphProjectionSnapshot(
            graph_version_id=version.id,
            name=spec["name"],
            spec_hash=digest,
            specification=spec,
            status="building",
            node_count=graph.number_of_nodes(),
            edge_count=graph.number_of_edges(),
        )
        session.add(snapshot)
        await session.flush()

        for metric, values in analytics["node_metrics"].items():
            session.add_all(
                [
                    GraphNodeMetric(
                        projection_id=snapshot.id,
                        entity_id=node_id,
                        metric=metric,
                        value=value,
                        rank=rank,
                    )
                    for node_id, value, rank in _ranked(values)
                ]
            )
        for index, members in enumerate(analytics["communities"], 1):
            community = GraphCommunity(
                projection_id=snapshot.id,
                algorithm="louvain",
                community_key=f"community-{index}",
                metadata_json={"size": len(members)},
            )
            session.add(community)
            await session.flush()
            session.add_all(
                [
                    GraphCommunityMembership(
                        community_id=community.id, entity_id=node_id, strength=1.0
                    )
                    for node_id in members
                ]
            )
        session.add_all(
            [
                GraphComponentMetric(
                    projection_id=snapshot.id,
                    component_key=item["key"],
                    metric=item["metric"],
                    value=item["value"],
                    metadata_json=item["metadata"],
                )
                for item in analytics["component_metrics"]
            ]
        )
        snapshot.status = "completed"
        snapshot.diagnostics = analytics["diagnostics"]
        snapshot.completed_at = datetime.now(UTC)
        await session.commit()
    return "completed"


async def compile_ingestion_frame_values(
    collection: Collection,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Recompile stored extraction JSON into retrieval-rich frames without an LLM."""
    async with AsyncSessionLocal() as session:
        segments = (
            (
                await session.execute(
                    select(GraphChunkSegment).where(
                        GraphChunkSegment.collection_id == collection.id,
                        GraphChunkSegment.status == "completed",
                        GraphChunkSegment.tombstoned_at.is_(None),
                    )
                )
            )
            .scalars()
            .all()
        )

    frames_by_id: dict[uuid.UUID, tuple[SemanticFrameRow, uuid.UUID]] = {}
    for segment in segments:
        extraction = extraction_from_payload(segment.raw_extraction or {})
        for frame in compile_semantic_frames(collection, extraction):
            frames_by_id.setdefault(frame.id, (frame, segment.id))

    frame_values: list[dict[str, Any]] = []
    argument_values: list[dict[str, Any]] = []
    for frame, segment_id in frames_by_id.values():
        frame_values.append(
            {
                "id": frame.id,
                "collection_id": collection.id,
                "segment_id": segment_id,
                "proposition_entity_id": frame.proposition_id,
                "source_relationship_id": frame.relationship_id,
                "frame_kind": "proposition",
                "predicate": frame.predicate,
                "title": frame.title,
                "frame_text": frame.frame_text,
                "content_hash": frame.content_hash,
                "polarity": frame.polarity,
                "modality": frame.modality,
                "conditions_json": list(frame.conditions),
                "exceptions_json": list(frame.exceptions),
                "scopes_json": list(frame.scopes),
                "executable_status": "grounded_binary",
                "metadata_json": {"backfilled_without_llm": True},
            }
        )
        argument_values.extend(
            {
                "id": argument.id,
                "frame_id": frame.id,
                "position": argument.position,
                "role": argument.role,
                "entity_id": argument.entity_id,
                "argument_type": argument.argument_type,
            }
            for argument in frame.arguments
        )
    return frame_values, argument_values


def compile_community_frame_values(
    collection: Collection,
    version: GraphVersion,
    nodes: list[Node],
    edges: list[Edge],
    analytics: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Create deterministic, navigation-only summaries at community resolution."""
    node_by_id = {node.id: node for node in nodes}
    pagerank = analytics["node_metrics"].get("pagerank", {})
    frame_values: list[dict[str, Any]] = []
    argument_values: list[dict[str, Any]] = []
    for members in analytics["communities"]:
        semantic_members = [member for member in members if member in node_by_id]
        if len(semantic_members) < 2:
            continue
        ordered = sorted(
            semantic_members,
            key=lambda node_id: (-pagerank.get(node_id, 0.0), str(node_id)),
        )
        top_members = ordered[:10]
        member_set = set(semantic_members)
        predicate_counts = Counter(
            edge.rel_type
            for edge in edges
            if edge.source_id in member_set and edge.target_id in member_set
        )
        predicates = [name for name, _ in predicate_counts.most_common(8)]
        names = [node_by_id[node_id].name for node_id in top_members]
        membership_key = hashlib.sha256(
            "|".join(sorted(str(node_id) for node_id in member_set)).encode()
        ).hexdigest()
        frame_id = deterministic_uuid(
            collection.id,
            f"community-summary:{version.id}:{membership_key}",
        )
        title = "Community: " + ", ".join(names[:4])
        frame_text = (
            f"A semantic community connecting {', '.join(names)}. "
            f"Prominent relationship families: {', '.join(predicates) or 'none'}. "
            "Use this frame to locate relevant propositions; it is not evidence."
        )
        frame_values.append(
            {
                "id": frame_id,
                "collection_id": collection.id,
                "graph_version_id": version.id,
                "frame_kind": "community_summary",
                "title": title,
                "frame_text": frame_text,
                "content_hash": hashlib.sha256(frame_text.encode()).hexdigest(),
                "polarity": "unknown",
                "modality": "navigation",
                "executable_status": "navigation_only",
                "metadata_json": {
                    "member_count": len(member_set),
                    "predicate_families": predicates,
                },
            }
        )
        argument_values.extend(
            {
                "id": deterministic_uuid(
                    collection.id,
                    f"frame-argument:{frame_id}:{position}:member",
                ),
                "frame_id": frame_id,
                "position": position,
                "role": "member",
                "entity_id": node_id,
                "argument_type": "entity",
            }
            for position, node_id in enumerate(top_members)
        )
    return frame_values, argument_values


async def persist_frame_values(
    collection: Collection,
    provider: EmbeddingProvider,
    frame_values: list[dict[str, Any]],
    argument_values: list[dict[str, Any]],
    *,
    batch_size: int = 64,
) -> tuple[int, int]:
    if not frame_values:
        return 0, 0
    dimensions = await get_collection_dimensions(collection.id)
    if dimensions is None:
        raise ValueError("Collection vector dimensions are not initialized")
    await ensure_semantic_frame_table(collection.id, dimensions)
    frame_table = table_name(collection.id, "semantic_frame_embeddings")

    async with AsyncSessionLocal() as session:
        await session.execute(
            pg_insert(GraphSemanticFrame)
            .values(frame_values)
            .on_conflict_do_nothing(index_elements=[GraphSemanticFrame.id])
        )
        if argument_values:
            await session.execute(
                pg_insert(GraphFrameArgument)
                .values(argument_values)
                .on_conflict_do_nothing(index_elements=[GraphFrameArgument.id])
            )
        await session.commit()

    async with AsyncSessionLocal() as session:
        existing_ids = set(
            (
                await session.execute(text(f"SELECT frame_id FROM {frame_table}"))
            ).scalars()
        )
    missing = [row for row in frame_values if row["id"] not in existing_ids]
    embedded = 0
    cast = f"::vector({dimensions})"
    for offset in range(0, len(missing), batch_size):
        batch = missing[offset : offset + batch_size]
        embeddings = await provider.embed_documents(
            [str(row["frame_text"]) for row in batch]
        )
        async with AsyncSessionLocal() as session:
            await session.execute(
                text(
                    f"INSERT INTO {frame_table} "
                    "(frame_id, collection_id, graph_version_id, frame_kind, "
                    "content, embedding) VALUES "
                    f"(:fid, :cid, :vid, :kind, :content, (:emb){cast}) "
                    "ON CONFLICT (frame_id) DO NOTHING"
                ),
                [
                    {
                        "fid": _uuid_for_sql(row["id"]),
                        "cid": _uuid_for_sql(collection.id),
                        "vid": (
                            _uuid_for_sql(row["graph_version_id"])
                            if row.get("graph_version_id")
                            else None
                        ),
                        "kind": row["frame_kind"],
                        "content": row["frame_text"],
                        "emb": vector_literal(embedding),
                    }
                    for row, embedding in zip(batch, embeddings, strict=True)
                ],
            )
            await session.commit()
        embedded += len(batch)
    return len(frame_values), embedded


async def main() -> None:
    args = parse_args()
    collection, version, nodes, edges = await load_graph(args.collection_id)
    print(
        f"collection={collection.name} version={version.version} "
        f"nodes={len(nodes)} edges={len(edges)}"
    )
    embedding_provider = await _resolve_embedding_provider(collection)
    ingestion_frames, ingestion_arguments = await compile_ingestion_frame_values(
        collection
    )
    frame_count, embedded_count = await persist_frame_values(
        collection,
        embedding_provider,
        ingestion_frames,
        ingestion_arguments,
    )
    print(f"proposition_frames: materialized={frame_count} embedded={embedded_count}")

    affinity_analytics: dict[str, Any] | None = None
    for spec in PROJECTION_SPECS:
        graph = build_projection(nodes, edges, spec)
        analytics = compute_analytics(graph)
        if spec["name"] == "semantic_affinity_undirected":
            affinity_analytics = analytics
        status = await persist_projection(
            version, spec, graph, analytics, force=args.force
        )
        print(
            f"{spec['name']}: {status} nodes={graph.number_of_nodes()} "
            f"edges={graph.number_of_edges()}"
        )
    if affinity_analytics is not None:
        summary_frames, summary_arguments = compile_community_frame_values(
            collection,
            version,
            nodes,
            edges,
            affinity_analytics,
        )
        frame_count, embedded_count = await persist_frame_values(
            collection,
            embedding_provider,
            summary_frames,
            summary_arguments,
        )
        print(f"community_frames: materialized={frame_count} embedded={embedded_count}")


if __name__ == "__main__":
    asyncio.run(main())
