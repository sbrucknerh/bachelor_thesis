"""Re-exports for the motif pipeline.

The actual implementations live in `src/motifs/` as one module per backend:
- src/motifs/networkx_backend.py — pure-Python 5-type counter
- src/motifs/snap_backend.py     — SNAP.py Paranjape 36-type (currently broken)
- src/motifs/raphtory_backend.py — Raphtory Paranjape 40-type (default)

This file keeps the old imports working for `run_networkx` and any code that
expects the NetworkX primitives.
"""

from __future__ import annotations

from collections import Counter

from src.motifs.networkx_backend import (
    _classify_motif_simple,
    _get_sorted_edges,
    count_per_node_motifs as count_per_node_motifs_nx,
    extract_temporal_motifs_nx,
)


def compute_motif_distribution(motifs_or_counts) -> dict:
    """Convert a list of motif dicts (NetworkX path) or a 1D count vector
    (SNAP/Raphtory paths) into a labeled distribution dict."""
    # Sequence of ints/floats -> motif_i labels
    if hasattr(motifs_or_counts, "__len__") and len(motifs_or_counts) > 0 and isinstance(motifs_or_counts[0], (int, float)):
        return {f"motif_{i}": int(c) for i, c in enumerate(motifs_or_counts) if c > 0}

    # List of motif dicts from extract_temporal_motifs_nx
    if isinstance(motifs_or_counts, list) and motifs_or_counts and isinstance(motifs_or_counts[0], dict):
        return dict(Counter(m["motif_type"] for m in motifs_or_counts))

    return {}


__all__ = [
    "extract_temporal_motifs_nx",
    "count_per_node_motifs_nx",
    "compute_motif_distribution",
    "_classify_motif_simple",
    "_get_sorted_edges",
]
