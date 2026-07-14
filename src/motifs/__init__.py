"""Motif-counting backends.

Three implementations behind one interface — see `base.py` for the contract.

Usage:
    from src.motifs import get_backend
    count_motifs = get_backend("raphtory")
    fingerprints = count_motifs(df, delta=3600, node_filter={"0xabc..."})
"""

from __future__ import annotations

from typing import Callable

import pandas as pd


_BACKENDS = {"networkx", "snap", "raphtory"}


def get_backend(name: str) -> Callable[..., pd.DataFrame]:
    """Return the `count_per_node_motifs` function for the named backend."""
    if name not in _BACKENDS:
        raise ValueError(
            f"Unknown motif backend: {name!r}. "
            f"Available: {sorted(_BACKENDS)}"
        )

    if name == "networkx":
        from src.motifs.networkx_backend import count_per_node_motifs
    elif name == "snap":
        from src.motifs.snap_backend import count_per_node_motifs
    elif name == "raphtory":
        from src.motifs.raphtory_backend import count_per_node_motifs

    return count_per_node_motifs


__all__ = ["get_backend"]
