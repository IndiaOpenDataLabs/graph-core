"""Burner for anchor-centered concept induction over a base collection.

Builds a multi-rel-type ego neighborhood for one or more anchor entities,
then calls an OpenAI-compatible chat endpoint to induce one concept per
anchor.

Usage:
  PYTHONPATH=src .venv/bin/python docs/anchor_concept_burner.py \
    --collection-id dac68659-3283-4de2-9095-5e6cc3bfcfc1 \
    --anchors Vata Pitta Kapha \
    --base-url http://localhost:8080/v1 \
    --model unsloth/gemma-4-26B-A4B-it-GGUF:UD-Q6_K_XL
"""

from __future__ import annotations

import argparse
import asyncio
import json
import uuid
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import httpx

from graph_core.services.graph.analytics import _load_graph_records


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collection-id", required=True, help="Collection UUID")
    parser.add_argument(
        "--anchors",
        nargs="+",
        required=True,
        help="Anchor entity names to induce concepts for",
    )
    parser.add_argument(
        "--base-url",
        default="http://localhost:8080/v1",
        help="OpenAI-compatible base URL",
    )
    parser.add_argument("--model", required=True, help="Model identifier")
    parser.add_argument(
        "--per-type-limit",
        type=int,
        default=2,
        help="Max representative relationships to keep per rel_type",
    )
    parser.add_argument(
        "--top-rel-types",
        type=int,
        default=20,
        help="How many rel-type counts to include in the prompt summary",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=600.0,
        help="HTTP timeout for the chat completion call",
    )
    parser.add_argument(
        "--output-dir",
        default="docs",
        help="Directory to write per-anchor JSON files into",
    )
    return parser.parse_args()


async def _build_anchor_payload(
    collection_id: uuid.UUID,
    anchor: str,
    *,
    per_type_limit: int,
    top_rel_types: int,
) -> dict[str, Any]:
    collection, _nodes, relationships, _aliases = await _load_graph_records(
        collection_id
    )
    rels: list[dict[str, Any]] = []
    for rel in relationships:
        source_name = str(rel.source_name)
        target_name = str(rel.target_name)
        if source_name == anchor or target_name == anchor:
            rels.append(
                {
                    "source": source_name,
                    "target": target_name,
                    "rel_type": str(rel.rel_type or "RELATES_TO").upper(),
                    "weight": float(rel.weight or 0.0),
                    "direction": "out" if source_name == anchor else "in",
                }
            )

    rel_type_counts = Counter(rel["rel_type"] for rel in rels)
    by_type: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for rel in rels:
        by_type[rel["rel_type"]].append(rel)

    selected: list[dict[str, Any]] = []
    for rel_type, items in sorted(
        by_type.items(),
        key=lambda kv: (-len(kv[1]), kv[0]),
    ):
        items = sorted(
            items,
            key=lambda item: (-item["weight"], item["source"], item["target"]),
        )
        selected.extend(items[:per_type_limit])

    seen: set[tuple[str, str, str]] = set()
    representative_relationships: list[dict[str, Any]] = []
    for rel in selected:
        key = (rel["source"], rel["rel_type"], rel["target"])
        if key in seen:
            continue
        seen.add(key)
        representative_relationships.append(rel)

    return {
        "collection": collection.name,
        "anchor": anchor,
        "relationship_count": len(rels),
        "distinct_rel_types": len(rel_type_counts),
        "rel_type_counts": rel_type_counts.most_common(top_rel_types),
        "representative_relationships": representative_relationships,
    }


async def _induce_anchor_concepts(
    payloads: list[dict[str, Any]],
    *,
    base_url: str,
    model: str,
    timeout_seconds: float,
) -> list[dict[str, Any]]:
    collection_name = payloads[0]["collection"] if payloads else "unknown"
    system = (
        "You are inducing reusable concepts from multi-rel-type entity neighborhoods "
        "in a knowledge graph. For each anchor entity, infer a broader concept if "
        "warranted from its mixed relationships. Return only valid JSON: an array "
        "of objects with keys anchor, label, concept_type, description, aliases, "
        "rationale."
    )
    user = (
        f"Collection: {collection_name}\n"
        "For each anchor below, produce one concept object.\n\n"
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
    payloads = [
        await _build_anchor_payload(
            collection_id,
            anchor,
            per_type_limit=args.per_type_limit,
            top_rel_types=args.top_rel_types,
        )
        for anchor in args.anchors
    ]
    concepts = await _induce_anchor_concepts(
        payloads,
        base_url=args.base_url,
        model=args.model,
        timeout_seconds=args.timeout_seconds,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for concept in concepts:
        anchor = str(concept["anchor"])
        out_path = output_dir / f"{anchor.strip().lower()}_concept.json"
        out_path.write_text(
            json.dumps(concept, ensure_ascii=True, indent=2) + "\n",
            encoding="utf-8",
        )
        print(out_path)


if __name__ == "__main__":
    asyncio.run(main())
