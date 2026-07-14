"""NetworkX pure-Python motif backend.

Detects 5 simplified motif types (fan_out, fan_in, cycle, chain, reciprocal)
over a δ-temporal window. Per-node fingerprints are 5-dimensional.

This is a coarser catalogue than Paranjape et al. 2017, but it runs anywhere
with no external deps and serves as a baseline against the full Raphtory and
SNAP implementations.
"""

from __future__ import annotations

from collections import Counter, defaultdict

import pandas as pd

from src.graph_construction import build_transaction_graph


MOTIF_TYPES = ["fan_out", "fan_in", "cycle", "chain", "reciprocal"]
_COLUMNS = [f"motif_{t}" for t in MOTIF_TYPES]


def count_per_node_motifs(
    df: pd.DataFrame,
    delta: int,
    node_filter: set[str] | None = None,
) -> pd.DataFrame:
    """Count per-node motif participation using the NetworkX backend."""
    G = build_transaction_graph(df)
    motifs = extract_temporal_motifs_nx(G, delta=delta)

    per_node = defaultdict(Counter)
    for m in motifs:
        mtype = m["motif_type"]
        if mtype == "other":
            continue
        for node in m["nodes"]:
            per_node[node][mtype] += 1

    records = []
    nodes_iter = node_filter if node_filter is not None else per_node.keys()
    for addr in nodes_iter:
        counts = per_node.get(addr, Counter())
        record = {"address": addr}
        for t in MOTIF_TYPES:
            record[f"motif_{t}"] = counts.get(t, 0)
        records.append(record)

    df_out = pd.DataFrame(records)
    if df_out.empty:
        return pd.DataFrame(columns=["address"] + _COLUMNS).set_index("address")
    return df_out.set_index("address")


def extract_temporal_motifs_nx(G, delta: int) -> list[dict]:
    """Extract δ-temporal motifs from a NetworkX MultiDiGraph.

    Finds 2-node and 3-node temporal patterns within a sliding time window.
    Returns a list of motif dicts with keys: nodes, edges, motif_type.
    """
    motifs = []
    edges_with_time = _get_sorted_edges(G)

    for i, (src1, dst1, t1, _) in enumerate(edges_with_time):
        for j in range(i + 1, len(edges_with_time)):
            src2, dst2, t2, _ = edges_with_time[j]

            if t2 - t1 > delta:
                break

            nodes_1 = {src1, dst1}
            nodes_2 = {src2, dst2}
            if not nodes_1 & nodes_2:
                continue

            all_nodes = nodes_1 | nodes_2

            if len(all_nodes) == 3:
                for k in range(i, len(edges_with_time)):
                    if k == i or k == j:
                        continue
                    src3, dst3, t3, _ = edges_with_time[k]
                    if t3 - t1 > delta:
                        break
                    if t3 < t1:
                        continue

                    edge3_nodes = {src3, dst3}
                    if edge3_nodes <= all_nodes and len(edge3_nodes) == 2:
                        motif_edges = [
                            (src1, dst1, t1),
                            (src2, dst2, t2),
                            (src3, dst3, t3),
                        ]
                        motifs.append({
                            "nodes": tuple(sorted(all_nodes)),
                            "edges": motif_edges,
                            "motif_type": _classify_motif_simple(all_nodes, motif_edges),
                        })
                        break

            elif len(all_nodes) == 2:
                motif_edges = [(src1, dst1, t1), (src2, dst2, t2)]
                motifs.append({
                    "nodes": tuple(sorted(all_nodes)),
                    "edges": motif_edges,
                    "motif_type": _classify_motif_simple(all_nodes, motif_edges),
                })

    return motifs


def _get_sorted_edges(G) -> list[tuple]:
    edges = []
    for src, dst, data in G.edges(data=True):
        edges.append((src, dst, data["timestamp"], data.get("value", 0)))
    edges.sort(key=lambda x: x[2])
    return edges


def _classify_motif_simple(nodes: set, edges: list[tuple]) -> str:
    sources = [e[0] for e in edges]
    targets = [e[1] for e in edges]

    source_counts = Counter(sources)
    target_counts = Counter(targets)

    if len(source_counts) == 1 and len(set(targets)) > 1:
        return "fan_out"
    if len(target_counts) == 1 and len(set(sources)) > 1:
        return "fan_in"

    sorted_edges = sorted(edges, key=lambda e: e[2])
    if sorted_edges[-1][1] == sorted_edges[0][0]:
        return "cycle"
    for i in range(len(sorted_edges) - 1):
        if sorted_edges[i][1] == sorted_edges[i + 1][0]:
            return "chain"

    if len(nodes) == 2:
        dirs = {(e[0], e[1]) for e in edges}
        if len(dirs) == 2:
            return "reciprocal"

    return "other"
