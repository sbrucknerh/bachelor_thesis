"""Build NetworkX transaction graphs from Ethereum transaction data."""

import networkx as nx
import pandas as pd


def build_transaction_graph(df: pd.DataFrame) -> nx.MultiDiGraph:
    """Build a directed multigraph from a transaction DataFrame.

    Each node represents an Ethereum address. Each edge represents a transaction
    with attributes: value, timestamp, and tx_hash. A MultiDiGraph is used because
    multiple transactions can occur between the same pair of addresses.

    Args:
        df: DataFrame with columns from_address, to_address, value, timestamp, tx_hash.

    Returns:
        A NetworkX MultiDiGraph representing the transaction network.
    """
    G = nx.MultiDiGraph()

    for _, row in df.iterrows():
        src = row["from_address"]
        dst = row["to_address"]

        # Add nodes if they don't exist yet
        if not G.has_node(src):
            G.add_node(src)
        if not G.has_node(dst):
            G.add_node(dst)

        # Add edge with transaction attributes
        G.add_edge(
            src,
            dst,
            value=row["value"],
            timestamp=row["timestamp"],
            tx_hash=row["tx_hash"],
        )

    # Compute node attributes
    _compute_node_attributes(G, df)

    return G


def _compute_node_attributes(G: nx.MultiDiGraph, df: pd.DataFrame) -> None:
    """Compute and assign summary attributes to each node."""
    # Total value sent and received per address
    sent = df.groupby("from_address")["value"].sum()
    received = df.groupby("to_address")["value"].sum()

    for node in G.nodes():
        G.nodes[node]["total_sent"] = sent.get(node, 0.0)
        G.nodes[node]["total_received"] = received.get(node, 0.0)
        G.nodes[node]["in_degree"] = G.in_degree(node)
        G.nodes[node]["out_degree"] = G.out_degree(node)


def get_graph_stats(G: nx.MultiDiGraph) -> dict:
    """Compute basic network statistics.

    Args:
        G: The transaction multigraph.

    Returns:
        Dictionary with network statistics.
    """
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
