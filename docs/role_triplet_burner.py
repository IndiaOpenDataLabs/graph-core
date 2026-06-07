"""Burner for triplet concept induction from role-similarity structure.

Build a typed-signature similarity graph over base entities, keep pair edges
that satisfy overlap/similarity thresholds, enumerate triangles (3-cliques),
and optionally ask an OpenAI-compatible LLM to induce one concept per triplet.

Usage:
  PYTHONPATH=src .venv/bin/python docs/role_triplet_burner.py \
    --collection-id dac68659-3283-4de2-9095-5e6cc3bfcfc1 \
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

from graph_core.services.graph.analytics import _load_graph_records


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collection-id", required=True, help="Collection UUID")
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
        "--max-triplets",
        type=int,
        default=20,
        help="Maximum triplets to keep after ranking",
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
        default="docs/triplet_burner_output",
        help="Directory to write triplet concept JSON files into",
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
    min_signature: int,
    overlap_min: int,
    cosine_min: float,
    jaccard_min: float,
) -> tuple[str, dict[str, set[tuple[str, str]]], dict[str, dict[str, Any]], list[dict[str, Any]]]:
    collection, _nodes, relationships, _aliases = await _load_graph_records(collection_id)

    out_pairs: dict[str, set[tuple[str, str]]] = defaultdict(set)
    in_pairs: dict[str, set[tuple[str, str]]] = defaultdict(set)
    pair_relationships: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    all_nodes: set[str] = set()

    for rel in relationships:
        source_name = str(rel.source_name)
        target_name = str(rel.target_name)
        rel_type = str(rel.rel_type or "RELATES_TO").upper()
        weight = float(rel.weight or 0.0)
        all_nodes.add(source_name)
        all_nodes.add(target_name)
        out_pairs[source_name].add((rel_type, target_name))
        in_pairs[target_name].add((source_name, rel_type))
        pair_relationships[(source_name, target_name)].append(
            {
                "source": source_name,
                "target": target_name,
                "rel_type": rel_type,
                "weight": weight,
            }
        )

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
    adjacency: dict[str, set[str]] = defaultdict(set)
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
        adjacency[a].add(b)
        adjacency[b].add(a)
        similarity_edges.append(metrics)

    return collection.name, signatures, pair_metrics, similarity_edges


def _enumerate_triplets(
    pair_metrics: dict[tuple[str, str], dict[str, Any]],
    max_triplets: int,
) -> list[dict[str, Any]]:
    adjacency: dict[str, set[str]] = defaultdict(set)
    for a, b in pair_metrics:
        adjacency[a].add(b)
        adjacency[b].add(a)

    triplets: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    for a in sorted(adjacency):
        for b, c in combinations(sorted(adjacency[a]), 2):
            if b not in adjacency[c]:
                continue
            triplet = tuple(sorted((a, b, c)))
            if triplet in seen:
                continue
            seen.add(triplet)
            ab = pair_metrics[tuple(sorted((triplet[0], triplet[1])))]
            ac = pair_metrics[tuple(sorted((triplet[0], triplet[2])))]
            bc = pair_metrics[tuple(sorted((triplet[1], triplet[2])))]
            avg_cosine = (ab["cosine"] + ac["cosine"] + bc["cosine"]) / 3.0
            avg_jaccard = (ab["jaccard"] + ac["jaccard"] + bc["jaccard"]) / 3.0
            total_overlap = ab["overlap"] + ac["overlap"] + bc["overlap"]
            triplets.append(
                {
                    "nodes": list(triplet),
                    "avg_cosine": avg_cosine,
                    "avg_jaccard": avg_jaccard,
                    "total_overlap": total_overlap,
                    "pair_metrics": [ab, ac, bc],
                }
            )
    triplets.sort(
        key=lambda item: (
            item["avg_cosine"],
            item["avg_jaccard"],
            item["total_overlap"],
        ),
        reverse=True,
    )
    return triplets[:max_triplets]


async def _build_triplet_payloads(
    collection_id: uuid.UUID,
    triplets: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    _collection, _nodes, relationships, _aliases = await _load_graph_records(collection_id)
    memberships = {node for triplet in triplets for node in triplet["nodes"]}
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
    for idx, triplet in enumerate(triplets, start=1):
        representative: list[dict[str, Any]] = []
        for node in triplet["nodes"]:
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
                "triplet_id": f"triplet_{idx}",
                "nodes": triplet["nodes"],
                "avg_cosine": triplet["avg_cosine"],
                "avg_jaccard": triplet["avg_jaccard"],
                "total_overlap": triplet["total_overlap"],
                "pair_metrics": triplet["pair_metrics"],
                "representative_relationships": deduped,
            }
        )
    return payloads


async def _induce_triplet_concepts(
    collection_name: str,
    payloads: list[dict[str, Any]],
    *,
    base_url: str,
    model: str,
    timeout_seconds: float,
) -> list[dict[str, Any]]:
    system = (
        "You are inducing reusable higher-level concepts from triplets of role-similar entities in a knowledge graph. "
        "Each triplet is a candidate shared concept because the three entities occupy similar typed graph positions. "
        "Return only valid JSON: an array of objects with keys triplet_id, nodes, label, concept_type, description, aliases, rationale."
    )
    user = (
        f"Collection: {collection_name}\n"
        "For each triplet below, produce one shared concept object.\n\n"
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
    collection_id = uuid.UUID(args.collection_id)
    collection_name, signatures, pair_metrics, similarity_edges = await _build_similarity_structures(
        collection_id,
        min_signature=args.min_signature,
        overlap_min=args.overlap_min,
        cosine_min=args.cosine_min,
        jaccard_min=args.jaccard_min,
    )
    triplets = _enumerate_triplets(pair_metrics, args.max_triplets)

    print(
        json.dumps(
            {
                "collection": collection_name,
                "nodes_with_signatures": len(signatures),
                "similarity_edges": len(similarity_edges),
                "triplets": len(triplets),
                "top_triplets": triplets[:10],
            },
            ensure_ascii=True,
            indent=2,
        )
    )

    if not args.base_url or not args.model or not triplets:
        return

    payloads = await _build_triplet_payloads(collection_id, triplets)
    concepts = await _induce_triplet_concepts(
        collection_name,
        payloads,
        base_url=args.base_url,
        model=args.model,
        timeout_seconds=args.timeout_seconds,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for concept in concepts:
        triplet_id = str(concept["triplet_id"])
        out_path = output_dir / f"{triplet_id}.json"
        out_path.write_text(
            json.dumps(concept, ensure_ascii=True, indent=2) + "\n",
            encoding="utf-8",
        )
        print(out_path)


if __name__ == "__main__":
    asyncio.run(main())
