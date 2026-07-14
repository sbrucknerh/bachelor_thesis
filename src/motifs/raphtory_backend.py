"""Raphtory motif backend — Paranjape et al. (2017) algorithm, per-node counts.

Uses `raphtory.algorithms.local_temporal_three_node_motifs`, which returns a
40-dimensional vector per node (36 three-node motifs + 4 two-node motifs),
computed in parallel in Rust.

Install: `pip install raphtory`. Works on Python 3.10+, including 3.13.
Reference:
  Paranjape, Benson & Leskovec. "Motifs in Temporal Networks" (WSDM 2017).
  Raphtory docs: https://docs.raphtory.com/en/v0.9.0/reference/algorithms/motifs.html
"""

from __future__ import annotations

import pandas as pd


NUM_MOTIF_TYPES = 40
_COLUMNS = [f"motif_{i}" for i in range(NUM_MOTIF_TYPES)]


def count_per_node_motifs(
    df: pd.DataFrame,
    delta: int,
    node_filter: set[str] | None = None,
) -> pd.DataFrame:
    """Per-node motif fingerprints via Raphtory.

    Args:
        df: transactions with from_address, to_address, timestamp, value.
        delta: temporal window in seconds.
        node_filter: if given, drop result rows not in this set.

    Returns:
        DataFrame indexed by address with columns motif_0..motif_39.
    """
    try:
        from raphtory import Graph
        from raphtory.algorithms import local_temporal_three_node_motifs
    except ImportError as e:
        raise ImportError(
            "The `raphtory` package is required for this backend. "
            "Install it with: pip install raphtory"
        ) from e

    g = Graph()
    for row in df.itertuples(index=False):
        g.add_edge(int(row.timestamp), str(row.from_address), str(row.to_address))

    result = local_temporal_three_node_motifs(g, int(delta))

    records = []
    for node, payload in result.items():
        addr = node.name if hasattr(node, "name") else str(node)
        if node_filter is not None and addr not in node_filter:
            continue
        counts = payload["motif_counter"] if isinstance(payload, dict) else payload
        record = {"address": addr}
        for i, c in enumerate(counts):
            record[f"motif_{i}"] = int(c)
        records.append(record)

    if not records:
        return pd.DataFrame(columns=["address"] + _COLUMNS).set_index("address")

    out = pd.DataFrame(records).set_index("address")
    # Ensure consistent column order even if Raphtory returns fewer types
    for col in _COLUMNS:
        if col not in out.columns:
            out[col] = 0
    return out[_COLUMNS]
