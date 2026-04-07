"""Visualization utilities for transaction networks and motif analysis."""

import matplotlib.pyplot as plt
import networkx as nx

from config import EDGE_ALPHA, FIGURE_SIZE, NODE_SIZE


def plot_transaction_graph(
    G: nx.MultiDiGraph,
    title: str = "Ethereum Transaction Network",
    highlight_nodes: list[str] | None = None,
) -> None:
    """Draw the transaction network graph.

    Args:
        G: The transaction multigraph.
        title: Plot title.
        highlight_nodes: Optional list of node addresses to highlight in red
            (e.g., flagged suspicious addresses).
    """
    fig, ax = plt.subplots(figsize=FIGURE_SIZE)

    pos = nx.spring_layout(G, seed=42, k=0.3)

    # Default node colors
    node_colors = []
    for node in G.nodes():
        if highlight_nodes and node in highlight_nodes:
            node_colors.append("red")
        else:
            node_colors.append("skyblue")

    nx.draw_networkx_nodes(
        G, pos, ax=ax,
        node_size=NODE_SIZE,
        node_color=node_colors,
        alpha=0.8,
    )

    nx.draw_networkx_edges(
        G, pos, ax=ax,
        alpha=EDGE_ALPHA,
        arrows=True,
        arrowsize=8,
        edge_color="gray",
    )

    ax.set_title(title, fontsize=14)
    ax.axis("off")
    plt.tight_layout()
    plt.show()


def plot_motif_distribution(distribution: dict[str, int]) -> None:
    """Plot a bar chart of temporal motif type frequencies.

    Args:
        distribution: Dictionary mapping motif type names to counts.
    """
    if not distribution:
        print("No motifs to plot.")
        return

    fig, ax = plt.subplots(figsize=(8, 5))

    types = list(distribution.keys())
    counts = list(distribution.values())

    bars = ax.bar(types, counts, color="steelblue", edgecolor="black")

    ax.set_xlabel("Motif Type")
    ax.set_ylabel("Frequency")
    ax.set_title("Temporal Motif Distribution")

    # Add count labels on bars
    for bar, count in zip(bars, counts):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.5,
            str(count),
            ha="center",
            va="bottom",
            fontsize=10,
        )

    plt.tight_layout()
    plt.show()


def plot_detection_results(flagged: list[dict], top_n: int = 10) -> None:
    """Plot risk scores for the top-N flagged addresses.

    Args:
        flagged: List of detection result dicts from detection.detect_coordinated_behaviour.
        top_n: Number of top results to show.
    """
    if not flagged:
        print("No suspicious addresses detected.")
        return

    results = flagged[:top_n]
    labels = [r["address"][:10] + "..." for r in results]
    scores = [r["risk_score"] for r in results]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.barh(labels, scores, color="crimson", edgecolor="black")
    ax.set_xlabel("Risk Score")
    ax.set_title(f"Top {len(results)} Suspicious Addresses by Risk Score")
    ax.invert_yaxis()

    plt.tight_layout()
    plt.show()
