"""SNAP.py motif backend — Paranjape et al. (2017) 36-type catalogue.

Status: BROKEN. The code here targets an API that does not exist in the
installed snap-stanford 6.0:

- `snap.TempMotifCounter` in the public Python bindings takes a filename
  string, not a graph object.
- `Count3TEdge3NodeStarDynamic` does not exist; there is no per-node variant
  of SNAP's temporal motif counter.

Kept in the codebase so the backend lineup stays visible for the thesis
writeup and so someone can fix it later (probably by writing the transaction
DataFrame to a temp TSV file and running the file-based counter per
ego-subgraph). Until then, `count_per_node_motifs()` raises RuntimeError.

SNAP.py also has no PyPI wheels for Python >= 3.10. Install with:
    pip install snap-stanford   # Python 3.9 only
"""

from __future__ import annotations

import numpy as np
import pandas as pd


NUM_MOTIF_TYPES = 36
_COLUMNS = [f"motif_{i}" for i in range(NUM_MOTIF_TYPES)]


def count_per_node_motifs(
    df: pd.DataFrame,
    delta: int,
    node_filter: set[str] | None = None,
) -> pd.DataFrame:
    """Per-node motif fingerprints via SNAP — currently non-functional.

    Raises RuntimeError until the SNAP 6.0 API mismatch is resolved. See the
    module docstring for details.
    """
    raise RuntimeError(
        "SNAP backend is currently broken: snap.TempMotifCounter does not accept "
        "a graph object in the installed version, and no per-node motif API is "
        "exposed. Use --motif-backend raphtory (default) or --motif-backend "
        "networkx instead."
    )


# --- Original (broken) code kept for reference / future repair ---

def _build_snap_temporal_graph_broken(df: pd.DataFrame) -> tuple:
    """Original builder targeting snap.TNEANet.New() — AttributeError on SNAP 6.0."""
    import snap  # noqa: F401

    all_addresses = sorted(set(df["from_address"]) | set(df["to_address"]))
    address_to_id = {addr: i for i, addr in enumerate(all_addresses)}
    id_to_address = {i: addr for addr, i in address_to_id.items()}

    G = snap.TNEANet.New()  # <-- AttributeError in SNAP 6.0

    for node_id in range(len(all_addresses)):
        G.AddNode(node_id)
    G.AddIntAttrE("timestamp")
    G.AddFltAttrE("value")

    for _, row in df.iterrows():
        src_id = address_to_id[row["from_address"]]
        dst_id = address_to_id[row["to_address"]]
        edge_id = G.AddEdge(src_id, dst_id)
        G.AddIntAttrDatE(edge_id, int(row["timestamp"]), "timestamp")
        G.AddFltAttrDatE(edge_id, float(row["value"]), "value")

    return G, address_to_id, id_to_address


def _count_temporal_motifs_global_broken(snap_graph, delta: int) -> np.ndarray:
    """Original global counter — TempMotifCounter(graph, delta) is wrong API."""
    import snap

    counter = snap.TempMotifCounter(snap_graph, delta)  # <-- wrong: expects filename
    motif_counts = counter.Count3TEdge3Node()

    result = np.zeros(NUM_MOTIF_TYPES, dtype=np.int64)
    idx = 0
    for i in range(6):
        for j in range(6):
            result[idx] = motif_counts[i][j]
            idx += 1
    return result
