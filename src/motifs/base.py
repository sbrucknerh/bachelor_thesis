"""Shared contract for motif-counting backends.

Every backend in this package exposes a single function with the signature:

    count_per_node_motifs(
        df: pd.DataFrame,
        delta: int,
        node_filter: set[str] | None = None,
    ) -> pd.DataFrame

Inputs:
- df: transactions with columns `from_address`, `to_address`, `timestamp`, `value`.
- delta: temporal window in seconds for δ-temporal motifs.
- node_filter: optional set of addresses. If given, only return rows for these
  addresses (skips wasted work on unlabeled nodes during supervised training).

Output:
- DataFrame indexed by address, with one column per motif type. Column names and
  count vary by backend:
  - NetworkX backend: 5 columns (motif_fan_out, motif_fan_in, motif_cycle,
    motif_chain, motif_reciprocal).
  - Raphtory backend: 40 columns (motif_0..motif_39) — full Paranjape catalogue
    + 2-node motifs.
  - SNAP backend: 36 columns (motif_0..motif_35) — Paranjape 3-node catalogue.
"""

from __future__ import annotations

import pandas as pd


MotifFingerprint = pd.DataFrame
