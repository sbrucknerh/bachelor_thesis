"""Visualization utilities for transaction networks and motif analysis."""

from __future__ import annotations

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd

from config import EDGE_ALPHA, FIGURE_SIZE, NODE_SIZE


def plot_transaction_graph(
    G: nx.MultiDiGraph,
    title: str = "Ethereum Transaction Network",
    highlight_nodes: list[str] | None = None,
    save_path: str | None = None,
) -> None:
    """Draw the transaction network graph.

    Args:
        G: The transaction multigraph.
        title: Plot title.
        highlight_nodes: Optional list of node addresses to highlight in red.
        save_path: If provided, save figure instead of showing.
    """
    fig, ax = plt.subplots(figsize=FIGURE_SIZE)

    pos = nx.spring_layout(G, seed=42, k=0.3)

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

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def plot_motif_distribution(
    distribution: dict[str, int],
    save_path: str | None = None,
) -> None:
    """Plot a bar chart of temporal motif type frequencies."""
    if not distribution:
        print("No motifs to plot.")
        return

    fig, ax = plt.subplots(figsize=(10, 5))

    types = list(distribution.keys())
    counts = list(distribution.values())

    bars = ax.bar(types, counts, color="steelblue", edgecolor="black")

    ax.set_xlabel("Motif Type")
    ax.set_ylabel("Frequency")
    ax.set_title("Temporal Motif Distribution (36-Type Catalogue)")
    ax.tick_params(axis="x", rotation=45)

    for bar, count in zip(bars, counts):
        if count > 0:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.5,
                str(count),
                ha="center", va="bottom", fontsize=8,
            )

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def plot_detection_results(
    flagged: list[dict],
    top_n: int = 10,
    save_path: str | None = None,
) -> None:
    """Plot risk scores for the top-N flagged addresses."""
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

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


# --- New thesis-quality visualizations ---

def plot_motif_fingerprint_heatmap(
    fingerprints: pd.DataFrame,
    labels: pd.Series | None = None,
    save_path: str | None = None,
) -> None:
    """Plot a heatmap of motif fingerprints, optionally colored by label.

    Args:
        fingerprints: DataFrame of motif fingerprints (addresses x motif types).
        labels: Optional Series of labels (1=malicious, 0=benign) indexed by address.
        save_path: If provided, save figure.
    """
    data = fingerprints.copy()

    # Sort by label if available
    if labels is not None:
        data["_label"] = labels
        data = data.sort_values("_label", ascending=False)
        data = data.drop("_label", axis=1)

    fig, ax = plt.subplots(figsize=(14, max(6, len(data) * 0.15)))

    im = ax.imshow(data.values, aspect="auto", cmap="YlOrRd", interpolation="nearest")

    ax.set_xlabel("Motif Type")
    ax.set_ylabel("Address")
    ax.set_title("Motif Fingerprint Heatmap")
    ax.set_xticks(range(data.shape[1]))
    ax.set_xticklabels(data.columns, rotation=90, fontsize=7)

    # Truncate y-axis labels
    if len(data) <= 50:
        ax.set_yticks(range(len(data)))
        ax.set_yticklabels([addr[:8] + "..." for addr in data.index], fontsize=6)
    else:
        ax.set_yticks([])

    plt.colorbar(im, ax=ax, label="Motif Count")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def plot_cluster_tsne(
    X: pd.DataFrame,
    cluster_labels: np.ndarray,
    save_path: str | None = None,
) -> None:
    """Plot a 2D t-SNE projection of feature vectors colored by cluster.

    Args:
        X: Feature matrix (addresses x features).
        cluster_labels: Cluster assignments from detection.cluster_addresses.
        save_path: If provided, save figure.
    """
    from sklearn.manifold import TSNE

    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(X) - 1))
    embedding = tsne.fit_transform(X.values)

    fig, ax = plt.subplots(figsize=(10, 8))

    unique_labels = sorted(set(cluster_labels))
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(unique_labels), 1)))

    for label, color in zip(unique_labels, colors):
        mask = cluster_labels == label
        marker = "x" if label == -1 else "o"
        alpha = 0.3 if label == -1 else 0.7
        name = "Noise" if label == -1 else f"Cluster {label}"
        ax.scatter(
            embedding[mask, 0], embedding[mask, 1],
            c=[color], marker=marker, alpha=alpha, s=30, label=name,
        )

    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.set_title("Address Clusters by Motif Fingerprint (t-SNE)")
    ax.legend(fontsize=8, loc="best")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def plot_delta_sensitivity(
    results_by_delta: dict[int, dict],
    metric: str = "f1",
    save_path: str | None = None,
) -> None:
    """Plot how detection performance varies across delta values.

    Args:
        results_by_delta: Dict mapping delta (seconds) to evaluation metrics dict.
        metric: Which metric to plot (e.g. "f1", "precision", "recall", "roc_auc").
        save_path: If provided, save figure.
    """
    deltas = sorted(results_by_delta.keys())
    values = [results_by_delta[d].get(metric, 0) for d in deltas]
    delta_labels = [_format_delta(d) for d in deltas]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(range(len(deltas)), values, "o-", color="steelblue", linewidth=2, markersize=8)
    ax.set_xticks(range(len(deltas)))
    ax.set_xticklabels(delta_labels)
    ax.set_xlabel("Time Window (δ)")
    ax.set_ylabel(metric.upper())
    ax.set_title(f"Detection {metric.upper()} vs. Temporal Window δ")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def _format_delta(seconds: int) -> str:
    """Format delta seconds into a human-readable label."""
    if seconds < 3600:
        return f"{seconds // 60}min"
    elif seconds < 86400:
        return f"{seconds // 3600}h"
    elif seconds < 604800:
        return f"{seconds // 86400}d"
    else:
        return f"{seconds // 604800}w"
