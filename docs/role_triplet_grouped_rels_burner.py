"""Burner for clique concept induction with grouped relationship types.

Embed relationship-type labels, group very similar rel types under a dominant
representative label, then build a typed-signature similarity graph over base
entities, enumerate maximal cliques of size >= 2, and optionally ask an
OpenAI-compatible LLM to induce one concept per clique.

Usage:
  PYTHONPATH=src .venv/bin/python docs/role_triplet_grouped_rels_burner.py \
    --collection-id dac68659-3283-4de2-9095-5e6cc3bfcfc1 \
    --embedding-base-url http://localhost:1234/v1 \
    --embedding-model qwen3-embedding-8b-q4_k_m \
    --base-url http://localhost:8080/v1 \
    --model unsloth/gemma-4-26B-A4B-it-GGUF:UD-Q6_K_XL
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import uuid
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path
from typing import Any

import httpx
import networkx as nx

from graph_core.services.graph.analytics import _load_graph_records


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collection-id", required=True, help="Collection UUID")
    parser.add_argument(
        "--embedding-base-url",
        default="",
        help="OpenAI-compatible embedding base URL; required for rel-type grouping",
    )
    parser.add_argument(
        "--embedding-model",
        default="",
        help="Embedding model identifier for rel-type grouping",
    )
    parser.add_argument(
        "--rel-cosine-min",
        type=float,
        default=0.85,
        help="Minimum cosine similarity between rel-type label embeddings to group them",
    )
    parser.add_argument(
        "--overlap-min",
        type=int,
        default=2,
        help="Minimum typed-signature overlap required for a pair edge",
    )
    parser.add_argument(
        "--cosine-min",
        type=float,
        default=0.2,
        help="Minimum cosine similarity required for a pair edge",
    )
    parser.add_argument(
        "--jaccard-min",
        type=float,
        default=0.1,
        help="Minimum Jaccard similarity required for a pair edge",
    )
    parser.add_argument(
        "--min-signature",
        type=int,
        default=1,
        help="Minimum typed-signature size to keep a node",
    )
    parser.add_argument(
        "--max-cliques",
        type=int,
        default=0,
        help="Maximum cliques to keep after ranking; 0 means no limit",
    )
    parser.add_argument(
        "--base-url",
        default="",
        help="OpenAI-compatible base URL; omit to skip LLM induction",
    )
    parser.add_argument("--model", default="", help="Model identifier for LLM induction")
    parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=600.0,
        help="HTTP timeout for the chat completion call",
    )
    parser.add_argument(
        "--output-dir",
        default="docs/triplet_grouped_rels_burner_output",
        help="Directory to write clique concept JSON files into",
    )
    return parser.parse_args()


def _jaccard(a: set[Any], b: set[Any]) -> float:
    union = a | b
    return len(a & b) / len(union) if union else 0.0


def _cosine_binary(a: set[Any], b: set[Any]) -> float:
    return len(a & b) / math.sqrt(len(a) * len(b)) if a and b else 0.0


async def _build_similarity_structures(
    collection_id: uuid.UUID,
    *,
    embedding_base_url: str,
    embedding_model: str,
    rel_cosine_min: float,
    min_signature: int,
    overlap_min: int,
    cosine_min: float,
    jaccard_min: float,
) -> tuple[str, dict[str, set[tuple[str, str]]], dict[str, dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    collection, _nodes, relationships, _aliases = await _load_graph_records(collection_id)

    raw_rel_type_counts = Counter(
        str(rel.rel_type or "RELATES_TO").upper() for rel in relationships
    )
    rel_type_mapping = await _group_rel_types(
        raw_rel_type_counts,
        base_url=embedding_base_url,
        model=embedding_model,
        cosine_min=rel_cosine_min,
    )

    out_pairs: dict[str, set[tuple[str, str]]] = defaultdict(set)
    in_pairs: dict[str, set[tuple[str, str]]] = defaultdict(set)
    all_nodes: set[str] = set()

    for rel in relationships:
        source_name = str(rel.source_name)
        target_name = str(rel.target_name)
        raw_rel_type = str(rel.rel_type or "RELATES_TO").upper()
        rel_type = rel_type_mapping.get(raw_rel_type, raw_rel_type)
        weight = float(rel.weight or 0.0)
        all_nodes.add(source_name)
        all_nodes.add(target_name)
        out_pairs[source_name].add((rel_type, target_name))
        in_pairs[target_name].add((source_name, rel_type))

    signatures: dict[str, set[tuple[str, str]]] = {}
    token_index: dict[tuple[str, str], set[str]] = defaultdict(set)
    for node in all_nodes:
        signature = out_pairs[node] | in_pairs[node]
        if len(signature) < min_signature:
            continue
        signatures[node] = signature
        for token in signature:
            token_index[token].add(node)

    overlap_counts: Counter[tuple[str, str]] = Counter()
    for nodes_with_token in token_index.values():
        members = sorted(nodes_with_token)
        for a, b in combinations(members, 2):
            overlap_counts[(a, b)] += 1

    similarity_edges: list[dict[str, Any]] = []
    pair_metrics: dict[tuple[str, str], dict[str, Any]] = {}
    for (a, b), overlap in overlap_counts.items():
        if overlap < overlap_min:
            continue
        signature_a = signatures[a]
        signature_b = signatures[b]
        cosine = _cosine_binary(signature_a, signature_b)
        jaccard = _jaccard(signature_a, signature_b)
        if cosine < cosine_min or jaccard < jaccard_min:
            continue
        metrics = {
            "a": a,
            "b": b,
            "overlap": overlap,
            "cosine": cosine,
            "jaccard": jaccard,
            "size_a": len(signature_a),
            "size_b": len(signature_b),
        }
        pair_metrics[(a, b)] = metrics
        similarity_edges.append(metrics)

    rel_type_groups = _summarize_rel_type_groups(raw_rel_type_counts, rel_type_mapping)
    return collection.name, signatures, pair_metrics, similarity_edges, rel_type_groups


async def _embed_texts(
    texts: list[str],
    *,
    base_url: str,
    model: str,
) -> list[list[float]]:
    async with httpx.AsyncClient(timeout=300.0) as client:
        response = await client.post(
            f"{base_url.rstrip('/')}/embeddings",
            json={"model": model, "input": texts},
        )
        response.raise_for_status()
        data = response.json()["data"]
        return [list(item["embedding"]) for item in data]


def _vector_cosine(left: list[float], right: list[float]) -> float:
    dot = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return dot / (left_norm * right_norm)


async def _group_rel_types(
    raw_rel_type_counts: Counter[str],
    *,
    base_url: str,
    model: str,
    cosine_min: float,
) -> dict[str, str]:
    rel_types = sorted(raw_rel_type_counts)
    if not rel_types:
        return {}
    embeddings = await _embed_texts(
        [rel_type.replace("_", " ") for rel_type in rel_types],
        base_url=base_url,
        model=model,
    )
    graph = nx.Graph()
    graph.add_nodes_from(rel_types)
    for i, left in enumerate(rel_types):
        for j in range(i + 1, len(rel_types)):
            right = rel_types[j]
            cosine = _vector_cosine(embeddings[i], embeddings[j])
            if cosine >= cosine_min:
                graph.add_edge(left, right, cosine=cosine)
    mapping: dict[str, str] = {}
    for component in nx.connected_components(graph):
        members = sorted(component)
        representative = max(
            members,
            key=lambda rel_type: (raw_rel_type_counts[rel_type], rel_type),
        )
        for rel_type in members:
            mapping[rel_type] = representative
    return mapping


def _summarize_rel_type_groups(
    raw_rel_type_counts: Counter[str],
    rel_type_mapping: dict[str, str],
) -> list[dict[str, Any]]:
    grouped: dict[str, list[str]] = defaultdict(list)
    for rel_type, representative in rel_type_mapping.items():
        grouped[representative].append(rel_type)
    summary: list[dict[str, Any]] = []
    for representative, members in grouped.items():
        summary.append(
            {
                "grouped_rel_type": representative,
                "members": sorted(members),
                "member_counts": {
                    member: raw_rel_type_counts[member]
                    for member in sorted(members)
                },
                "total_count": sum(raw_rel_type_counts[member] for member in members),
            }
        )
    summary.sort(
        key=lambda item: (item["total_count"], item["grouped_rel_type"]),
        reverse=True,
    )
    return summary


def _enumerate_cliques(
    pair_metrics: dict[tuple[str, str], dict[str, Any]],
    max_cliques: int,
) -> list[dict[str, Any]]:
    graph = nx.Graph()
    for a, b in pair_metrics:
        graph.add_edge(a, b)

    cliques: list[dict[str, Any]] = []
    for clique in nx.find_cliques(graph):
        if len(clique) < 2:
            continue
        clique_nodes = sorted(clique)
        metrics = [
            pair_metrics[tuple(sorted((left, right)))]
            for left, right in combinations(clique_nodes, 2)
        ]
        avg_cosine = sum(metric["cosine"] for metric in metrics) / len(metrics)
        avg_jaccard = sum(metric["jaccard"] for metric in metrics) / len(metrics)
        total_overlap = sum(metric["overlap"] for metric in metrics)
        cliques.append(
            {
                "clique_id": f"clique_{len(cliques) + 1}",
                "nodes": clique_nodes,
                "size": len(clique_nodes),
                "avg_cosine": avg_cosine,
                "avg_jaccard": avg_jaccard,
                "total_overlap": total_overlap,
                "pair_metrics": metrics,
            }
        )
    cliques.sort(
        key=lambda item: (
            item["size"],
            item["avg_cosine"],
            item["avg_jaccard"],
            item["total_overlap"],
        ),
        reverse=True,
    )
    if max_cliques > 0:
        return cliques[:max_cliques]
    return cliques


async def _build_clique_payloads(
    collection_id: uuid.UUID,
    cliques: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    _collection, _nodes, relationships, _aliases = await _load_graph_records(collection_id)
    memberships = {node for clique in cliques for node in clique["nodes"]}
    ego_rels: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for rel in relationships:
        source_name = str(rel.source_name)
        target_name = str(rel.target_name)
        rel_type = str(rel.rel_type or "RELATES_TO").upper()
        if source_name in memberships or target_name in memberships:
            if source_name in memberships:
                ego_rels[source_name].append(
                    {
                        "source": source_name,
                        "target": target_name,
                        "rel_type": rel_type,
                        "weight": float(rel.weight or 0.0),
                        "direction": "out",
                    }
                )
            if target_name in memberships and target_name != source_name:
                ego_rels[target_name].append(
                    {
                        "source": source_name,
                        "target": target_name,
                        "rel_type": rel_type,
                        "weight": float(rel.weight or 0.0),
                        "direction": "in",
                    }
                )

    payloads: list[dict[str, Any]] = []
    for idx, clique in enumerate(cliques, start=1):
        representative: list[dict[str, Any]] = []
        for node in clique["nodes"]:
            by_type: dict[str, list[dict[str, Any]]] = defaultdict(list)
            for rel in ego_rels[node]:
                by_type[rel["rel_type"]].append(rel)
            for rel_type, items in sorted(by_type.items(), key=lambda kv: (-len(kv[1]), kv[0])):
                items = sorted(items, key=lambda item: (-item["weight"], item["source"], item["target"]))
                representative.extend(items[:1])
        deduped: list[dict[str, Any]] = []
        seen: set[tuple[str, str, str]] = set()
        for rel in representative:
            key = (rel["source"], rel["rel_type"], rel["target"])
            if key in seen:
                continue
            seen.add(key)
            deduped.append(rel)
        payloads.append(
            {
                "clique_id": clique.get("clique_id") or f"clique_{idx}",
                "nodes": clique["nodes"],
                "size": clique["size"],
                "avg_cosine": clique["avg_cosine"],
                "avg_jaccard": clique["avg_jaccard"],
                "total_overlap": clique["total_overlap"],
                "pair_metrics": clique["pair_metrics"],
                "representative_relationships": deduped,
            }
        )
    return payloads


async def _induce_clique_concepts(
    collection_name: str,
    payloads: list[dict[str, Any]],
    *,
    base_url: str,
    model: str,
    timeout_seconds: float,
) -> list[dict[str, Any]]:
    system = (
        "You are inducing reusable higher-level concepts from cliques of role-similar entities in a knowledge graph. "
        "Each clique is a candidate shared concept because its entities occupy similar typed graph positions. "
        "Return only valid JSON: an array of objects with keys clique_id, nodes, label, concept_type, description, aliases, rationale."
    )
    user = (
        f"Collection: {collection_name}\n"
        "For each clique below, produce one shared concept object.\n\n"
        + json.dumps(payloads, ensure_ascii=True)
    )
    async with httpx.AsyncClient(timeout=timeout_seconds) as client:
        response = await client.post(
            f"{base_url.rstrip('/')}/chat/completions",
            json={
                "model": model,
                "temperature": 0.1,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
            },
        )
        response.raise_for_status()
        data = response.json()
        content = data["choices"][0]["message"]["content"].strip()
        if content.startswith("```"):
            lines = content.splitlines()
            if lines and lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].startswith("```"):
                lines = lines[:-1]
            content = "\n".join(lines).strip()
        return json.loads(content)


async def main() -> None:
    args = _parse_args()
    if not args.embedding_base_url or not args.embedding_model:
        raise SystemExit(
            "--embedding-base-url and --embedding-model are required for grouped rel-type mode"
        )
    collection_id = uuid.UUID(args.collection_id)
    (
        collection_name,
        signatures,
        pair_metrics,
        similarity_edges,
        rel_type_groups,
    ) = await _build_similarity_structures(
        collection_id,
        embedding_base_url=args.embedding_base_url,
        embedding_model=args.embedding_model,
        rel_cosine_min=args.rel_cosine_min,
        min_signature=args.min_signature,
        overlap_min=args.overlap_min,
        cosine_min=args.cosine_min,
        jaccard_min=args.jaccard_min,
    )
    cliques = _enumerate_cliques(pair_metrics, args.max_cliques)

    print(
        json.dumps(
            {
                "collection": collection_name,
                "rel_type_groups": rel_type_groups,
                "nodes_with_signatures": len(signatures),
                "similarity_edges": len(similarity_edges),
                "cliques": len(cliques),
                "cliques_detail": cliques,
            },
            ensure_ascii=True,
            indent=2,
        )
    )

    if not args.base_url or not args.model or not cliques:
        return

    payloads = await _build_clique_payloads(collection_id, cliques)
    concepts = await _induce_clique_concepts(
        collection_name,
        payloads,
        base_url=args.base_url,
        model=args.model,
        timeout_seconds=args.timeout_seconds,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for concept in concepts:
        clique_id = str(concept["clique_id"])
        out_path = output_dir / f"{clique_id}.json"
        out_path.write_text(
            json.dumps(concept, ensure_ascii=True, indent=2) + "\n",
            encoding="utf-8",
        )
        print(out_path)


if __name__ == "__main__":
    asyncio.run(main())
