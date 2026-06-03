"""FalkorDB graph storage for Custom Graph RAG.

Stores knowledge graph nodes and edges in FalkorDB.
Schema:
  - Nodes: (:Entity {id, name, collection_id})
  - Edges: [:REL_TYPE {id, weight, keywords, collection_id, rel_type}]
    where REL_TYPE is the per-edge rel_type (e.g. EXPLAINS, RELATES_TO).
    The rel_type is also stored as a property for filtering.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Optional

from graph_core.config import settings
from graph_core.models.rel_types import (
    DEFAULT_REL_TYPE,
    normalize_rel_type,
)

# FalkorDB is optional — only import when available
try:
    from falkordb.asyncio import FalkorDB
    from redis.asyncio import BlockingConnectionPool
except ImportError:
    FalkorDB = None  # type: ignore[misc,assignment]
    BlockingConnectionPool = None  # type: ignore[misc,assignment]

logger = logging.getLogger(__name__)


_LABEL_SAFE_RE = re.compile(r"[^A-Z0-9_]")


def _safe_label(rel_type: str) -> str:
    """Coerce a rel_type into a Cypher-safe label token."""
    cleaned = normalize_rel_type(rel_type)
    cleaned = _LABEL_SAFE_RE.sub("_", cleaned)
    if not cleaned or not cleaned[0].isalpha():
        cleaned = "R_" + cleaned
    return cleaned or "RELATES_TO"


def _edge_label_pattern(rel_types: list[str] | None) -> str:
    """Build a Cypher pattern for one or more edge labels, e.g. `[:EXPLAINS|RELATES_TO]`."""
    if not rel_types:
        return ""
    labels = [_safe_label(rt) for rt in rel_types if rt]
    if not labels:
        return ""
    if len(set(labels)) == 1:
        return f":{labels[0]}"
    return ":" + "|".join(labels)


class FalkorDBGraphStorage:
    """FalkorDB-backed knowledge graph storage.

    Uses the async falkordb Python client with connection pooling.
    """

    _connection_pool: Optional[BlockingConnectionPool] = None

    def __init__(self, namespace: str, **kwargs: Any) -> None:
        self._graph_name = namespace or settings.falkordb_graph_name
        self._host = "localhost"
        self._port = 6379
        self._client: Optional[Any] = None
        self._graph: Optional[Any] = None
        # Parse URL if provided
        if settings.falkordb_url:
            url = settings.falkordb_url
            if url.startswith("falkordb://"):
                url = "redis://" + url[len("falkordb://"):]
            from urllib.parse import urlparse
            parsed = urlparse(url)
            self._host = parsed.hostname or "localhost"
            self._port = parsed.port or 6379
        self._test_client: Optional[Any] = kwargs.get("_client")

    @classmethod
    def _get_connection_pool(cls) -> BlockingConnectionPool:
        if cls._connection_pool is None:
            url = settings.falkordb_url
            host = "localhost"
            port = 6379
            if url:
                if url.startswith("falkordb://"):
                    url = "redis://" + url[len("falkordb://"):]
                from urllib.parse import urlparse
                parsed = urlparse(url)
                host = parsed.hostname or "localhost"
                port = parsed.port or 6379

            cls._connection_pool = BlockingConnectionPool(
                max_connections=16,
                timeout=None,
                decode_responses=True,
                host=host,
                port=port,
            )
        return cls._connection_pool

    async def _get_graph(self):
        if FalkorDB is None:
            raise RuntimeError(
                "FalkorDB client not installed. Install with: pip install graph-core[graph]"
            )
        if self._graph is not None:
            return self._graph
        if self._test_client is not None:
            self._graph = self._test_client.select_graph(self._graph_name)
            return self._graph

        pool = self._get_connection_pool()
        if self._client is None:
            self._client = FalkorDB(connection_pool=pool)
        self._graph = self._client.select_graph(self._graph_name)
        return self._graph

    async def has_node(self, node_id: str) -> bool:
        graph = await self._get_graph()
        result = await graph.query(
            "MATCH (n:Entity {id: $id}) RETURN count(n) as count",
            {"id": node_id},
        )
        if result and result.result_set:
            count = result.result_set[0][0] or 0
            return count > 0
        return False

    async def get_node(self, node_id: str) -> Optional[dict[str, Any]]:
        graph = await self._get_graph()
        result = await graph.query(
            "MATCH (n:Entity {id: $id}) RETURN n",
            {"id": node_id},
        )
        if result and result.result_set:
            node = result.result_set[0][0]
            if node:
                return dict(node.properties)
        return None

    async def upsert_nodes(self, nodes: list[dict[str, Any]]) -> None:
        if not nodes:
            return
        graph = await self._get_graph()
        payload = [
            {
                "id": n["id"],
                "name": n.get("name", ""),
                "collection_id": str(n.get("collection_id", "")),
            }
            for n in nodes
        ]
        await graph.query(
            "UNWIND $nodes AS node"
            " MERGE (n:Entity {id: node.id})"
            " SET n.name = node.name, n.collection_id = node.collection_id",
            {"nodes": payload},
        )

    async def upsert_edges(self, edges: list[dict[str, Any]]) -> None:
        if not edges:
            return
        graph = await self._get_graph()
        # Group by rel_type so we can target the Cypher label precisely
        # (FalkorDB does not support dynamic rel-type set in a single
        # query without APOC). Each rel_type gets one MERGE pass.
        groups: dict[str, list[dict[str, Any]]] = {}
        for e in edges:
            rt = normalize_rel_type(e.get("rel_type"))
            groups.setdefault(rt, []).append(
                {
                    "source_id": e["source_id"],
                    "target_id": e["target_id"],
                    "id": e.get("id", ""),
                    "weight": e.get("weight", 1),
                    "keywords": (
                        json.dumps(e["keywords"])
                        if isinstance(e.get("keywords"), list)
                        else (e.get("keywords") or "[]")
                    ),
                    "collection_id": str(e.get("collection_id", "")),
                    "rel_type": rt,
                }
            )
        for rel_type, payload in groups.items():
            label = _safe_label(rel_type)
            await graph.query(
                "UNWIND $edges AS edge"
                " MERGE (a:Entity {id: edge.source_id})"
                " MERGE (b:Entity {id: edge.target_id})"
                f" MERGE (a)-[r:{label}]->(b)"
                " SET r.id = edge.id,"
                "     r.weight = edge.weight,"
                "     r.keywords = edge.keywords,"
                "     r.collection_id = edge.collection_id,"
                "     r.rel_type = edge.rel_type",
                {"edges": payload},
            )
        await self._merge_keywords_for_edges(
            [
                (
                    e["source_id"],
                    e["target_id"],
                    e.get("keywords") or [],
                    e.get("rel_type"),
                )
                for e in edges
            ]
        )

    async def _merge_keywords_for_edges(
        self,
        edge_keyword_inputs: list[tuple[str, str, list[str], str | None]],
    ) -> None:
        if not edge_keyword_inputs:
            return
        graph = await self._get_graph()
        updates: list[dict[str, Any]] = []
        for source_id, target_id, incoming, rel_type in edge_keyword_inputs:
            rts = [rel_type] if rel_type else None
            existing = await self.get_edge(source_id, target_id, rel_types=rts)
            if existing is None:
                existing = await self.get_edge(target_id, source_id, rel_types=rts)
            if not existing:
                continue
            existing_kws = existing.get("keywords") or []
            if isinstance(existing_kws, str):
                try:
                    existing_kws = json.loads(existing_kws)
                except (json.JSONDecodeError, TypeError):
                    existing_kws = []
            incoming_clean = [
                str(k).strip() for k in (incoming or []) if str(k).strip()
            ]
            merged: list[str] = []
            seen: set[str] = set()
            for k in list(existing_kws) + incoming_clean:
                key = k.lower()
                if key in seen:
                    continue
                seen.add(key)
                merged.append(k)
            if merged != list(existing_kws or []):
                updates.append(
                    {
                        "source_id": source_id,
                        "target_id": target_id,
                        "rel_type": rel_type or DEFAULT_REL_TYPE,
                        "keywords": json.dumps(merged),
                    }
                )
        if not updates:
            return
        for u in updates:
            label = _safe_label(u["rel_type"])
            await graph.query(
                f"MATCH (a:Entity {{id: $source_id}})-[r:{label}]->(b:Entity {{id: $target_id}})"
                " SET r.keywords = $keywords",
                {
                    "source_id": u["source_id"],
                    "target_id": u["target_id"],
                    "keywords": u["keywords"],
                },
            )
            await graph.query(
                f"MATCH (a:Entity {{id: $target_id}})-[r:{label}]->(b:Entity {{id: $source_id}})"
                " SET r.keywords = $keywords",
                {
                    "source_id": u["source_id"],
                    "target_id": u["target_id"],
                    "keywords": u["keywords"],
                },
            )

    async def upsert_node(self, node_id: str, properties: dict[str, Any]) -> None:
        graph = await self._get_graph()
        allowed_keys = {
            "id",
            "name",
            "collection_id",
            "type",
            "description",
            "source_ids",
            "source_message_ids",
            "source_roles",
            "role",
            "content",
            "question",
            "response",
            "chat_id",
            "turn_index",
            "message_index",
        }
        set_clauses = []
        params: dict[str, object] = {"id": node_id}
        for key, value in properties.items():
            if key != "id" and key in allowed_keys:
                set_clauses.append(f"n.{key} = ${key}")
                params[key] = value

        if set_clauses:
            set_str = ", ".join(set_clauses)
            query = f"MERGE (n:Entity {{id: $id}}) SET {set_str}"
        else:
            query = "MERGE (n:Entity {id: $id})"

        await graph.query(query, params)

    async def upsert_edge(
        self, source_id: str, target_id: str, properties: dict[str, Any]
    ) -> None:
        graph = await self._get_graph()
        rel_type = normalize_rel_type(properties.get("rel_type"))
        label = _safe_label(rel_type)
        allowed_keys = {
            "id",
            "weight",
            "keywords",
            "collection_id",
            "description",
            "source_ids",
            "source_message_ids",
            "source_roles",
        }
        set_clauses = []
        params: dict[str, object] = {
            "source_id": source_id,
            "target_id": target_id,
            "rel_type": rel_type,
        }
        for key, value in properties.items():
            if key not in ("source", "target", "rel_type") and key in allowed_keys:
                stored_value = json.dumps(value) if isinstance(value, list) else value
                set_clauses.append(f"r.{key} = $r_{key}")
                params[f"r_{key}"] = stored_value
        set_clauses.append("r.rel_type = $rel_type")

        if set_clauses:
            set_str = ", ".join(set_clauses)
            query = (
                "MERGE (a:Entity {id: $source_id})"
                " MERGE (b:Entity {id: $target_id})"
                f" MERGE (a)-[r:{label}]->(b) SET {set_str}"
            )
        else:
            query = (
                "MERGE (a:Entity {id: $source_id})"
                " MERGE (b:Entity {id: $target_id})"
                f" MERGE (a)-[r:{label}]->(b)"
            )

        await graph.query(query, params)

        incoming_keywords = properties.get("keywords")
        if incoming_keywords is not None:
            await self._merge_keywords_for_edges(
                [
                    (
                        source_id,
                        target_id,
                        list(incoming_keywords)
                        if not isinstance(incoming_keywords, list)
                        else incoming_keywords,
                        rel_type,
                    )
                ]
            )

    async def get_nodes_by_source_message_id(
        self,
        message_id: str,
    ) -> list[dict[str, Any]]:
        graph = await self._get_graph()
        result = await graph.query(
            """
            MATCH (n:Entity)
            WHERE n.source_message_ids CONTAINS $message_id
            RETURN n
            """,
            {"message_id": message_id},
        )
        rows: list[dict[str, Any]] = []
        if result and result.result_set:
            for row in result.result_set:
                node = row[0]
                if node:
                    rows.append(dict(node.properties))
        return rows

    async def get_edges_by_source_message_id(
        self,
        message_id: str,
    ) -> list[dict[str, Any]]:
        graph = await self._get_graph()
        result = await graph.query(
            """
            MATCH (a:Entity)-[r:RELATES_TO]->(b:Entity)
            WHERE r.source_message_ids CONTAINS $message_id
            RETURN a.id, b.id, r
            """,
            {"message_id": message_id},
        )
        edges: list[dict[str, Any]] = []
        if result and result.result_set:
            for row in result.result_set:
                rel = row[2]
                if rel:
                    props = dict(rel.properties)
                    props["source_id"] = row[0]
                    props["target_id"] = row[1]
                    edges.append(props)
        return edges

    # ── LightRAG name-based operations ──

    async def has_lightrag_node(self, node_name: str, collection_id: str) -> bool:
        """Check if a LightRAG entity node exists by name."""
        graph = await self._get_graph()
        result = await graph.query(
            "MATCH (n:Entity {id: $name, collection_id: $cid}) RETURN count(n) as count",
            {"name": node_name, "cid": collection_id},
        )
        if result and result.result_set:
            count = result.result_set[0][0] or 0
            return count > 0
        return False

    async def get_lightrag_node(self, node_name: str, collection_id: str) -> Optional[dict[str, Any]]:
        """Get LightRAG entity node properties by name."""
        graph = await self._get_graph()
        result = await graph.query(
            "MATCH (n:Entity {id: $name, collection_id: $cid}) RETURN n",
            {"name": node_name, "cid": collection_id},
        )
        if result and result.result_set:
            node = result.result_set[0][0]
            if node:
                props = dict(node.properties)
                if isinstance(props.get("source_ids"), str):
                    try:
                        props["source_ids"] = json.loads(props["source_ids"])
                    except (json.JSONDecodeError, TypeError):
                        props["source_ids"] = []
                return props
        return None

    async def upsert_lightrag_node(
        self, node_name: str, collection_id: str, properties: dict[str, Any]
    ) -> None:
        """Upsert a LightRAG entity node with name as ID."""
        graph = await self._get_graph()
        props = {
            "id": node_name,
            "name": node_name,
            "collection_id": collection_id,
            "type": properties.get("type", "UNKNOWN"),
            "description": properties.get("description", ""),
        }
        source_ids = properties.get("source_ids", [])
        if source_ids:
            props["source_ids"] = json.dumps(source_ids)

        await graph.query(
            "MERGE (n:Entity {id: $id, collection_id: $collection_id})"
            " SET n.name = $name, n.type = $type, n.description = $description"
            " SET n.source_ids = $source_ids",
            props,
        )

    async def upsert_lightrag_edge(
        self,
        source_name: str,
        target_name: str,
        collection_id: str,
        properties: dict[str, Any],
    ) -> None:
        """Upsert a LightRAG relationship edge with name-based nodes."""
        graph = await self._get_graph()
        rel_id = properties.get("id", f"{source_name}__{target_name}")
        rel_type = normalize_rel_type(properties.get("rel_type"))
        label = _safe_label(rel_type)
        keywords = properties.get("keywords", [])
        source_ids = properties.get("source_ids", [])

        await graph.query(
            "MERGE (a:Entity {id: $source_name, collection_id: $collection_id})"
            " MERGE (b:Entity {id: $target_name, collection_id: $collection_id})"
            f" MERGE (a)-[r:{label}]->(b)"
            " SET r.id = $rel_id,"
            " r.description = $description,"
            " r.weight = $weight,"
            " r.keywords = $keywords,"
            " r.source_ids = $source_ids,"
            " r.collection_id = $collection_id,"
            " r.rel_type = $rel_type",
            {
                "source_name": source_name,
                "target_name": target_name,
                "collection_id": collection_id,
                "rel_id": rel_id,
                "rel_type": rel_type,
                "description": properties.get("description", ""),
                "weight": properties.get("weight", 1),
                "keywords": json.dumps(keywords) if isinstance(keywords, list) else (keywords or "[]"),
                "source_ids": json.dumps(source_ids) if isinstance(source_ids, list) else (source_ids or "[]"),
            },
        )

        existing = await self.get_lightrag_edge(source_name, target_name, collection_id)
        if existing is None:
            existing = await self.get_lightrag_edge(target_name, source_name, collection_id)
        if existing:
            existing_kws = existing.get("keywords") or []
            if isinstance(existing_kws, str):
                try:
                    existing_kws = json.loads(existing_kws)
                except (json.JSONDecodeError, TypeError):
                    existing_kws = []
            incoming_clean = [
                str(k).strip() for k in (keywords or []) if str(k).strip()
            ]
            merged: list[str] = []
            seen: set[str] = set()
            for k in list(existing_kws) + incoming_clean:
                key = k.lower()
                if key in seen:
                    continue
                seen.add(key)
                merged.append(k)
            if merged != list(existing_kws or []):
                await graph.query(
                    f"MATCH (a:Entity {{id: $source_name, collection_id: $collection_id}})"
                    f"-[r:{label}]->"
                    f"(b:Entity {{id: $target_name, collection_id: $collection_id}})"
                    " SET r.keywords = $keywords",
                    {
                        "source_name": source_name,
                        "target_name": target_name,
                        "collection_id": collection_id,
                        "keywords": json.dumps(merged),
                    },
                )

    async def get_lightrag_node_edges(
        self,
        node_name: str,
        collection_id: str,
        rel_types: list[str] | None = None,
    ) -> list[tuple[str, str]]:
        """Get all edges connected to a LightRAG entity node."""
        graph = await self._get_graph()
        label_pat = _edge_label_pattern(rel_types)
        if rel_types and not label_pat:
            return []
        if label_pat:
            cypher = (
                f"MATCH (n:Entity {{id: $node_name, collection_id: $cid}})-[r{label_pat}]-(m:Entity)"
                " RETURN n.id as source, m.id as target"
            )
        else:
            cypher = (
                "MATCH (n:Entity {id: $node_name, collection_id: $cid})-[r]-(m:Entity)"
                " RETURN n.id as source, m.id as target"
            )
        result = await graph.query(cypher, {"node_name": node_name, "cid": collection_id})
        edges: list[tuple[str, str]] = []
        if result and result.result_set:
            for row in result.result_set:
                source, target = row[0], row[1]
                if source and target:
                    edges.append((str(source), str(target)))
        return edges

    async def get_lightrag_edge(
        self,
        source_name: str,
        target_name: str,
        collection_id: str,
        rel_types: list[str] | None = None,
    ) -> Optional[dict[str, Any]]:
        """Get LightRAG edge properties between two named nodes."""
        graph = await self._get_graph()
        label_pat = _edge_label_pattern(rel_types)
        if rel_types and not label_pat:
            return None
        if label_pat:
            cypher = (
                f"MATCH (a:Entity {{id: $source, collection_id: $cid}})-[r{label_pat}]->"
                f"(b:Entity {{id: $target, collection_id: $cid}}) RETURN r"
            )
        else:
            cypher = (
                "MATCH (a:Entity {id: $source, collection_id: $cid})-[r]->"
                "(b:Entity {id: $target, collection_id: $cid}) RETURN r"
            )
        result = await graph.query(
            cypher,
            {"source": source_name, "target": target_name, "cid": collection_id},
        )
        if result and result.result_set:
            edge = result.result_set[0][0]
            if edge:
                props = dict(edge.properties)
                for list_field in ("keywords", "source_ids"):
                    val = props.get(list_field)
                    if isinstance(val, str):
                        try:
                            props[list_field] = json.loads(val)
                        except (json.JSONDecodeError, TypeError):
                            props[list_field] = []
                return props
        return None

    async def delete_nodes_by_collection_lightrag(self, collection_id: str) -> int:
        """Delete all LightRAG nodes for a collection."""
        return await self.delete_nodes_by_collection(collection_id)

    async def get_node_edges(
        self,
        node_id: str,
        rel_types: list[str] | None = None,
    ) -> list[tuple[str, str]]:
        graph = await self._get_graph()
        label_pat = _edge_label_pattern(rel_types)
        if rel_types and not label_pat:
            return []
        if label_pat:
            cypher = (
                f"MATCH (n:Entity {{id: $node_id}})-[r{label_pat}]-(m:Entity)"
                " RETURN n.id as source, m.id as target, r.rel_type as rel_type"
            )
        else:
            cypher = (
                "MATCH (n:Entity {id: $node_id})-[r]-(m:Entity)"
                " RETURN n.id as source, m.id as target, r.rel_type as rel_type"
            )
        result = await graph.query(cypher, {"node_id": node_id})
        edges: list[tuple[str, str]] = []
        if result and result.result_set:
            for row in result.result_set:
                source, target = row[0], row[1]
                if source and target:
                    edges.append((str(source), str(target)))
        return edges

    async def get_edge(
        self,
        source_id: str,
        target_id: str,
        rel_types: list[str] | None = None,
    ) -> Optional[dict[str, Any]]:
        """Get edge properties between two entity IDs.

        When ``rel_types`` is provided, restricts the match to those
        relationship labels; when None, matches any label.
        """
        graph = await self._get_graph()
        label_pat = _edge_label_pattern(rel_types)
        if rel_types and not label_pat:
            return None
        if label_pat:
            cypher = (
                f"MATCH (a:Entity {{id: $source_id}})-[r{label_pat}]->(b:Entity {{id: $target_id}})"
                " RETURN r"
            )
        else:
            cypher = (
                "MATCH (a:Entity {id: $source_id})-[r]->(b:Entity {id: $target_id})"
                " RETURN r"
            )
        result = await graph.query(
            cypher, {"source_id": source_id, "target_id": target_id}
        )
        if result and result.result_set:
            edge = result.result_set[0][0]
            if edge:
                props = dict(edge.properties)
                for list_field in ("keywords", "source_ids"):
                    val = props.get(list_field)
                    if isinstance(val, str):
                        try:
                            props[list_field] = json.loads(val)
                        except (json.JSONDecodeError, TypeError):
                            props[list_field] = []
                return props
        return None

    async def get_knowledge_graph(
        self,
        node_id: str,
        max_depth: int = 3,
        max_nodes: int = 100,
        rel_types: list[str] | None = None,
    ) -> dict[str, Any]:
        graph = await self._get_graph()
        label_pat = _edge_label_pattern(rel_types) if rel_types else ""
        if rel_types and not label_pat:
            return {"nodes": [], "edges": []}
        if label_pat:
            cypher = (
                f"MATCH path = (n:Entity {{id: $node_id}})-[{label_pat.strip(':') or '*'}"
                f"*1..$max_depth]-(m:Entity)"
                " WITH path, m LIMIT $max_nodes RETURN path"
            )
        else:
            cypher = (
                "MATCH path = (n:Entity {id: $node_id})-[*1..$max_depth]-(m:Entity)"
                " WITH path, m LIMIT $max_nodes RETURN path"
            )
        result = await graph.query(
            cypher,
            {"node_id": node_id, "max_depth": max_depth, "max_nodes": max_nodes},
        )
        nodes: list[dict[str, Any]] = []
        edges: list[dict[str, Any]] = []
        node_ids_seen: set[str] = set()
        edge_ids_seen: set[str] = set()

        if result and result.result_set:
            for row in result.result_set:
                path = row[0]
                if path:
                    for node in path.nodes:
                        node_props = dict(node.properties)
                        nid = node_props.get("id")
                        if nid and nid not in node_ids_seen:
                            nodes.append(node_props)
                            node_ids_seen.add(nid)
                    for rel in path.relationships:
                        edge_props = dict(rel.properties)
                        edge_id = edge_props.get("id")
                        if not edge_id:
                            start_id = rel.src_node
                            end_id = rel.dest_node
                            if start_id and end_id:
                                edge_id = f"{start_id}__{end_id}"
                                edge_props["id"] = edge_id
                        if edge_id and edge_id not in edge_ids_seen:
                            edges.append(edge_props)
                            edge_ids_seen.add(edge_id)

        return {"nodes": nodes, "edges": edges}

    async def delete_nodes_by_collection(self, collection_id: str) -> int:
        graph = await self._get_graph()
        result = await graph.query(
            "MATCH (n:Entity {collection_id: $collection_id}) DETACH DELETE n",
            {"collection_id": str(collection_id)},
        )
        count = int(result.nodes_deleted) if result.nodes_deleted else 0
        logger.info("FalkorDB deleted %d nodes for collection_id=%s", count, collection_id)
        return count

    async def drop(self) -> None:
        graph = await self._get_graph()
        await graph.query("MATCH (n:Entity) DETACH DELETE n")

    async def close(self) -> None:
        pass
