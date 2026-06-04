#!/usr/bin/env python3
"""Burner for typed graph metrics over a collection's base graph.

Examples:
  PYTHONPATH=src .venv/bin/python docs/graph_metric_burner.py rlm exception
  PYTHONPATH=src .venv/bin/python docs/graph_metric_burner.py rlm architecture
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from collections import Counter, defaultdict, deque
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from sqlalchemy import select  # noqa: E402

from graph_core.database import AsyncSessionLocal  # noqa: E402
from graph_core.models.collection import Collection  # noqa: E402
from graph_core.models.graph_rag import GraphEntity, GraphRelationship  # noqa: E402

QUESTION_TERMS: dict[str, tuple[str, ...]] = {
    "exception": ("error", "exception"),
    "architecture": (
        "api",
        "client",
        "manager",
        "handler",
        "service",
        "gateway",
        "server",
        "repl",
        "environment",
        "config",
        "query",
        "auth",
        "tool",
        "broker",
        "cache",
        "logger",
        "provider",
        "core",
        "module",
    ),
}

TYPED_REL_SLICES = (
    "CALLS",
    "IMPORTS",
    "DEPENDS_ON",
    "DEFINES",
    "IMPLEMENTS",
    "EXTENDS",
    "RELATES_TO",
    "REFERENCES",
)


def power_pagerank(nodes, out_edges, in_edges, alpha=0.85, max_iter=100, tol=1e-8):
    n = len(nodes)
    if n == 0:
        return {}
    pr = {u: 1.0 / n for u in nodes}
    out_w = {u: sum(w for _, w in out_edges.get(u, [])) for u in nodes}
    for _ in range(max_iter):
        new = {u: (1 - alpha) / n for u in nodes}
        sink = sum(pr[u] for u in nodes if out_w.get(u, 0.0) == 0.0)
        sink_share = alpha * sink / n
        for u in nodes:
            new[u] += sink_share
        for v in nodes:
            for u, w in in_edges.get(v, []):
                denom = out_w.get(u, 0.0)
                if denom > 0:
                    new[v] += alpha * pr[u] * (w / denom)
        err = sum(abs(new[u] - pr[u]) for u in nodes)
        pr = new
        if err < tol:
            break
    return pr


def power_hits(nodes, out_edges, in_edges, max_iter=100, tol=1e-8):
    auth = {u: 1.0 for u in nodes}
    hub = {u: 1.0 for u in nodes}
    for _ in range(max_iter):
        new_auth = {u: sum(hub[v] * w for v, w in in_edges.get(u, [])) for u in nodes}
        norm = sum(v * v for v in new_auth.values()) ** 0.5 or 1.0
        for u in nodes:
            new_auth[u] /= norm
        new_hub = {u: sum(new_auth[v] * w for v, w in out_edges.get(u, [])) for u in nodes}
        norm = sum(v * v for v in new_hub.values()) ** 0.5 or 1.0
        for u in nodes:
            new_hub[u] /= norm
        err = sum(abs(new_auth[u] - auth[u]) + abs(new_hub[u] - hub[u]) for u in nodes)
        auth, hub = new_auth, new_hub
        if err < tol:
            break
    return auth, hub


def power_eigenvector_undirected(nodes, nbrs, max_iter=100, tol=1e-8):
    x = {u: 1.0 for u in nodes}
    for _ in range(max_iter):
        new = {u: sum(x[v] * w for v, w in nbrs.get(u, [])) for u in nodes}
        norm = sum(v * v for v in new.values()) ** 0.5 or 1.0
        for u in nodes:
            new[u] /= norm
        err = sum(abs(new[u] - x[u]) for u in nodes)
        x = new
        if err < tol:
            break
    return x


def shortest_paths_unweighted(adj, src):
    dist = {src: 0}
    sigma = defaultdict(float)
    sigma[src] = 1.0
    pred = defaultdict(list)
    q = deque([src])
    order = []
    while q:
        v = q.popleft()
        order.append(v)
        for w in adj.get(v, []):
            if w not in dist:
                dist[w] = dist[v] + 1
                q.append(w)
            if dist[w] == dist[v] + 1:
                sigma[w] += sigma[v]
                pred[w].append(v)
    return order, pred, sigma, dist


def betweenness_directed(nodes, adj):
    cb = {v: 0.0 for v in nodes}
    for s in nodes:
        order, pred, sigma, _dist = shortest_paths_unweighted(adj, s)
        delta = {v: 0.0 for v in nodes}
        for w in reversed(order):
            for v in pred[w]:
                if sigma[w]:
                    delta[v] += (sigma[v] / sigma[w]) * (1 + delta[w])
            if w != s:
                cb[w] += delta[w]
    n = len(nodes)
    if n > 2:
        scale = 1 / ((n - 1) * (n - 2))
        for v in cb:
            cb[v] *= scale
    return cb


def harmonic_closeness(nodes, adj):
    out = {}
    for s in nodes:
        _order, _pred, _sigma, dist = shortest_paths_unweighted(adj, s)
        out[s] = sum(1 / d for v, d in dist.items() if v != s and d > 0)
    return out


def articulation_points(nodes, undirected_adj):
    disc, low, parent = {}, {}, {}
    out = set()
    t = 0

    def dfs(u):
        nonlocal t
        t += 1
        disc[u] = low[u] = t
        children = 0
        for v in undirected_adj.get(u, []):
            if v not in disc:
                parent[v] = u
                children += 1
                dfs(v)
                low[u] = min(low[u], low[v])
                if parent.get(u) is None and children > 1:
                    out.add(u)
                if parent.get(u) is not None and low[v] >= disc[u]:
                    out.add(u)
            elif v != parent.get(u):
                low[u] = min(low[u], disc[v])

    for u in nodes:
        if u not in disc:
            parent[u] = None
            dfs(u)
    return out


def top_named(score_map, name_map, subset=None, topn=12):
    items = score_map.items()
    if subset is not None:
        items = ((k, v) for k, v in items if k in subset)
    return sorted(((name_map[k], v) for k, v in items), key=lambda x: x[1], reverse=True)[:topn]


def normalize(scores):
    if not scores:
        return {}
    max_value = max(scores.values()) or 0.0
    if max_value <= 0.0:
        return {k: 0.0 for k in scores}
    return {k: float(v) / float(max_value) for k, v in scores.items()}


def mean_score(scores, nodes):
    if not nodes:
        return 0.0
    return sum(scores.get(node_id, 0.0) for node_id in nodes) / float(len(nodes))


async def load_collection_graph(collection_name: str):
    async with AsyncSessionLocal() as session:
        coll = await session.scalar(select(Collection).where(Collection.name == collection_name))
        if not coll:
            raise ValueError(f"Collection {collection_name!r} not found")
        entities = (
            await session.execute(select(GraphEntity).where(GraphEntity.collection_id == coll.id))
        ).scalars().all()
        rels = (
            await session.execute(
                select(GraphRelationship).where(GraphRelationship.collection_id == coll.id)
            )
        ).scalars().all()
    return coll, entities, rels


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("collection")
    parser.add_argument("question_type", choices=sorted(QUESTION_TERMS))
    parser.add_argument("--topn", type=int, default=12)
    args = parser.parse_args()

    coll, entities, rels = await load_collection_graph(args.collection)
    terms = QUESTION_TERMS[args.question_type]
    name = {e.id: e.canonical_name for e in entities}
    focus_nodes = {e.id for e in entities if any(t in e.canonical_name.lower() for t in terms)}

    region_nodes = set(focus_nodes)
    region_rels = []
    reltype_counts = Counter()
    for r in rels:
        if r.source_entity_id in focus_nodes or r.target_entity_id in focus_nodes:
            region_nodes.add(r.source_entity_id)
            region_nodes.add(r.target_entity_id)
            region_rels.append(r)
            reltype_counts[r.rel_type] += 1

    out_edges = defaultdict(list)
    in_edges = defaultdict(list)
    dir_adj = defaultdict(list)
    undir_nbrs_w = defaultdict(list)
    undir_adj = defaultdict(set)
    for r in region_rels:
        w = max(float(r.weight), 1.0)
        s, t = r.source_entity_id, r.target_entity_id
        out_edges[s].append((t, w))
        in_edges[t].append((s, w))
        dir_adj[s].append(t)
        if s != t:
            undir_nbrs_w[s].append((t, w))
            undir_nbrs_w[t].append((s, w))
            undir_adj[s].add(t)
            undir_adj[t].add(s)

    print(f"collection={coll.name} id={coll.id}")
    print(f"question_type={args.question_type}")
    print(f"focus_nodes={len(focus_nodes)} region_nodes={len(region_nodes)} region_rels={len(region_rels)}")
    print("top_rel_types=", reltype_counts.most_common(12))

    pr = power_pagerank(region_nodes, out_edges, in_edges)
    auth, hub = power_hits(region_nodes, out_edges, in_edges)
    eig = power_eigenvector_undirected(region_nodes, undir_nbrs_w)
    btw = betweenness_directed(region_nodes, dir_adj)
    close = harmonic_closeness(region_nodes, dir_adj)
    arts = articulation_points(region_nodes, undir_adj)

    print("\n[global metrics]")
    for label, scores in [
        ("pagerank", pr),
        ("authority", auth),
        ("hub", hub),
        ("eigenvector", eig),
        ("betweenness", btw),
        ("harmonic_closeness", close),
    ]:
        print(f"\n{label}")
        for nm, val in top_named(scores, name, subset=focus_nodes, topn=args.topn):
            print(f"  {nm}: {val:.6f}")

    print("\narticulation_nodes")
    for eid in sorted(focus_nodes & arts, key=lambda x: name[x])[: args.topn * 2]:
        print(" ", name[eid])

    pr_n = normalize(pr)
    auth_n = normalize(auth)
    hub_n = normalize(hub)
    eig_n = normalize(eig)
    btw_n = normalize(btw)
    close_n = normalize(close)
    route_profile = {
        "hub": mean_score(hub_n, focus_nodes),
        "authority": mean_score(auth_n, focus_nodes),
        "bridge": mean_score(btw_n, focus_nodes),
        "central": mean_score(close_n, focus_nodes),
        "importance": (
            mean_score(pr_n, focus_nodes) + mean_score(eig_n, focus_nodes)
        )
        / 2.0,
    }
    routed = sorted(route_profile.items(), key=lambda item: item[1], reverse=True)
    print("\n[routing profile]")
    for label, value in routed:
        print(f"  {label}: {value:.6f}")
    top_route = routed[0][0] if routed else "importance"
    route_scores = {
        "hub": hub,
        "authority": auth,
        "bridge": btw,
        "central": close,
        "importance": pr,
    }
    print(f"route={top_route}")
    print("route_seed_nodes")
    for nm, val in top_named(route_scores[top_route], name, subset=focus_nodes, topn=min(args.topn, 8)):
        print(f"  {nm}: {val:.6f}")

    print("\n[typed slices]")
    for rel_type in TYPED_REL_SLICES:
        rs = [r for r in region_rels if r.rel_type == rel_type]
        if not rs:
            continue
        nodes = set()
        oe = defaultdict(list)
        ie = defaultdict(list)
        da = defaultdict(list)
        for r in rs:
            s, t = r.source_entity_id, r.target_entity_id
            w = max(float(r.weight), 1.0)
            nodes.add(s)
            nodes.add(t)
            oe[s].append((t, w))
            ie[t].append((s, w))
            da[s].append(t)
        present = focus_nodes & nodes
        pr_t = power_pagerank(nodes, oe, ie)
        btw_t = betweenness_directed(nodes, da)
        print(f"\n[{rel_type}] edges={len(rs)} focus_nodes={len(present)}")
        for nm, val in top_named(pr_t, name, subset=present, topn=min(args.topn, 10)):
            print(f"  PR {nm}: {val:.6f}")
        for nm, val in top_named(btw_t, name, subset=present, topn=min(args.topn, 6)):
            print(f"  BW {nm}: {val:.6f}")


if __name__ == "__main__":
    asyncio.run(main())
