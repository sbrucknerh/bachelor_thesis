"""Temporal motif detection algorithms for transaction networks.

A delta-temporal motif is a small subgraph pattern where all edges occur within
a time window of delta seconds. These motifs capture coordinated temporal
behaviour patterns in the network.
"""

from collections import Counter
from itertools import combinations

import networkx as nx


def extract_temporal_motifs(G: nx.MultiDiGraph, delta: int) -> list[dict]:
    """Extract delta-temporal motifs from the transaction graph.

    Finds all 3-node temporal motifs where all edges occur within a time window
    of delta seconds.

    Args:
        G: Transaction multigraph with timestamp edge attributes.
        delta: Maximum time window in seconds for edges to be considered
            part of the same temporal motif.

    Returns:
        List of motif dictionaries, each containing:
        - nodes: tuple of participating addresses
        - edges: list of (src, dst, timestamp) tuples
        - motif_type: classified pattern type
    """
    motifs = []
    edges_with_time = _get_sorted_edges(G)

    # Sliding window approach over time-sorted edges
    for i, (src1, dst1, t1, _) in enumerate(edges_with_time):
        for j in range(i + 1, len(edges_with_time)):
            src2, dst2, t2, _ = edges_with_time[j]

            # Break if outside time window
            if t2 - t1 > delta:
                break

            # Check if edges share at least one node (connected motif)
            nodes_1 = {src1, dst1}
            nodes_2 = {src2, dst2}
            shared = nodes_1 & nodes_2

            if not shared:
                continue

            all_nodes = nodes_1 | nodes_2

            # For 3-node motifs, look for a third edge connecting the remaining node
            if len(all_nodes) == 3:
                # Search for a third edge within the time window
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
                        motif_type = classify_motif(all_nodes, motif_edges)
                        motifs.append({
                            "nodes": tuple(sorted(all_nodes)),
                            "edges": motif_edges,
                            "motif_type": motif_type,
                        })
                        break  # One motif per edge pair

            # 2-node motifs (repeated transactions between same pair)
            elif len(all_nodes) == 2:
                motif_edges = [(src1, dst1, t1), (src2, dst2, t2)]
                motif_type = classify_motif(all_nodes, motif_edges)
                motifs.append({
                    "nodes": tuple(sorted(all_nodes)),
                    "edges": motif_edges,
                    "motif_type": motif_type,
                })

    return motifs


def _get_sorted_edges(G: nx.MultiDiGraph) -> list[tuple]:
    """Extract edges sorted by timestamp."""
    edges = []
    for src, dst, data in G.edges(data=True):
        edges.append((src, dst, data["timestamp"], data.get("value", 0)))
    edges.sort(key=lambda x: x[2])
    return edges


def classify_motif(nodes: set, edges: list[tuple]) -> str:
    """Classify a temporal motif by its structural pattern.

    Args:
        nodes: Set of node addresses in the motif.
        edges: List of (src, dst, timestamp) tuples.

    Returns:
        String label for the motif type.
    """
    sources = [e[0] for e in edges]
    targets = [e[1] for e in edges]

    source_counts = Counter(sources)
    target_counts = Counter(targets)

    # Fan-out: one node sends to multiple others
    if len(source_counts) == 1 and len(set(targets)) > 1:
        return "fan_out"

    # Fan-in: multiple nodes send to one target
    if len(target_counts) == 1 and len(set(sources)) > 1:
        return "fan_in"

    # Cycle: A -> B -> C -> A (or A -> B -> A for 2-node)
    if _is_cycle(edges):
        return "cycle"

    # Chain: A -> B -> C (sequential forwarding)
    if _is_chain(edges):
        return "chain"

    # Reciprocal: A -> B and B -> A
    if len(nodes) == 2:
        dirs = {(e[0], e[1]) for e in edges}
        if len(dirs) == 2:
            return "reciprocal"

    return "other"


def _is_cycle(edges: list[tuple]) -> bool:
    """Check if edges form a cycle."""
    if len(edges) < 2:
        return False
    sorted_edges = sorted(edges, key=lambda e: e[2])
    # Check if last target equals first source
    return sorted_edges[-1][1] == sorted_edges[0][0]


def _is_chain(edges: list[tuple]) -> bool:
    """Check if edges form a sequential chain A -> B -> C."""
    if len(edges) < 2:
        return False
    sorted_edges = sorted(edges, key=lambda e: e[2])
    for i in range(len(sorted_edges) - 1):
        if sorted_edges[i][1] != sorted_edges[i + 1][0]:
            return False
    return True


def compute_motif_distribution(motifs: list[dict]) -> dict[str, int]:
    """Count the frequency of each motif type.

    Args:
        motifs: List of motif dictionaries from extract_temporal_motifs.

    Returns:
        Dictionary mapping motif type names to their counts.
    """
    return dict(Counter(m["motif_type"] for m in motifs))
