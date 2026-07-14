"""Build transaction graphs from Ethereum transaction data.

Provides two graph backends:
- NetworkX MultiDiGraph for visualization, static features, and small-scale analysis
- SNAP TNEANet for scalable temporal motif counting (when snap-stanford is installed)
"""

from __future__ import annotations

import networkx as nx
import numpy as np
import pandas as pd


# --- NetworkX graph construction (visualization + static features) ---

def build_transaction_graph(df: pd.DataFrame) -> nx.MultiDiGraph:
    """Build a directed multigraph from a transaction DataFrame.

    Each node represents an Ethereum address. Each edge represents a transaction
    with attributes value, timestamp, and tx_hash when those columns exist.

    Uses `nx.from_pandas_edgelist` for vectorized construction — ~20-50x faster
    than row-by-row iteration on large frames.

    Args:
        df: DataFrame with columns from_address, to_address, value, timestamp, tx_hash.

    Returns:
        A NetworkX MultiDiGraph representing the transaction network.
    """
    candidate_attrs = ["value", "timestamp", "tx_hash"]
    edge_attrs = [c for c in candidate_attrs if c in df.columns]
    G = nx.from_pandas_edgelist(
        df,
        source="from_address",
        target="to_address",
        edge_attr=edge_attrs if edge_attrs else None,
        create_using=nx.MultiDiGraph,
    )
    _compute_node_attributes(G, df)
    return G


def _compute_node_attributes(G: nx.MultiDiGraph, df: pd.DataFrame) -> None:
    """Compute and assign summary attributes to each node (vectorized)."""
    sent = df.groupby("from_address")["value"].sum().to_dict()
    received = df.groupby("to_address")["value"].sum().to_dict()

    # Fill defaults for nodes that never sent / received
    for node in G.nodes():
        sent.setdefault(node, 0.0)
        received.setdefault(node, 0.0)

    nx.set_node_attributes(G, sent, "total_sent")
    nx.set_node_attributes(G, received, "total_received")
    nx.set_node_attributes(G, dict(G.in_degree()), "in_degree")
    nx.set_node_attributes(G, dict(G.out_degree()), "out_degree")


def get_graph_stats(G: nx.MultiDiGraph) -> dict:
    """Compute basic network statistics."""
    stats = {
        "num_nodes": G.number_of_nodes(),
        "num_edges": G.number_of_edges(),
        "density": nx.density(G),
        "num_weakly_connected_components": nx.number_weakly_connected_components(G),
        "num_strongly_connected_components": nx.number_strongly_connected_components(G),
    }

    if G.number_of_nodes() > 0:
        degrees = [d for _, d in G.degree()]
        stats["avg_degree"] = sum(degrees) / len(degrees)
        stats["max_degree"] = max(degrees)

    return stats


def get_node_features(G: nx.MultiDiGraph) -> pd.DataFrame:
    """Extract per-node structural features using NetworkX.

    Computes features useful for ML-based detection:
    - in_degree, out_degree, total_degree
    - in/out degree ratio
    - total_sent, total_received, net_flow
    - clustering coefficient (on simplified graph)
    - PageRank

    Args:
        G: The transaction multigraph.

    Returns:
        DataFrame indexed by address with feature columns.
    """
    nodes = list(G.nodes())
    features = {}

    # Degree features
    features["in_degree"] = {n: G.in_degree(n) for n in nodes}
    features["out_degree"] = {n: G.out_degree(n) for n in nodes}
    features["total_degree"] = {n: G.degree(n) for n in nodes}
    features["degree_ratio"] = {
        n: G.in_degree(n) / max(G.out_degree(n), 1) for n in nodes
    }

    # Value features
    features["total_sent"] = {n: G.nodes[n].get("total_sent", 0.0) for n in nodes}
    features["total_received"] = {n: G.nodes[n].get("total_received", 0.0) for n in nodes}
    features["net_flow"] = {
        n: G.nodes[n].get("total_received", 0.0) - G.nodes[n].get("total_sent", 0.0)
        for n in nodes
    }

    # Clustering coefficient (requires simple DiGraph)
    simple_G = nx.DiGraph(G)
    clustering = nx.clustering(simple_G)
    features["clustering_coeff"] = clustering

    # PageRank
    try:
        pagerank = nx.pagerank(simple_G)
    except nx.PowerIterationFailedConvergence:
        pagerank = {n: 1.0 / len(nodes) for n in nodes}
    features["pagerank"] = pagerank

    df = pd.DataFrame(features)
    df.index.name = "address"
    return df


# The SNAP graph builder used to live here. It has been moved into
# src/motifs/snap_backend.py because it's only relevant to the SNAP motif
# path and is currently broken against SNAP.py 6.0.
