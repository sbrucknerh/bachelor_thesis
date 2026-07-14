"""Feature engineering: combine motif fingerprints with graph-structural features.

Builds per-address feature matrices for ML-based detection of coordinated
malicious behaviour.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def build_feature_matrix(
    motif_fingerprints: pd.DataFrame,
    node_features: pd.DataFrame,
    df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Combine motif fingerprints with structural and temporal features.

    Args:
        motif_fingerprints: DataFrame indexed by address with motif_0..motif_35 columns.
        node_features: DataFrame indexed by address from graph_construction.get_node_features.
        df: Optional transaction DataFrame for computing temporal features.

    Returns:
        Unified feature DataFrame indexed by address.
    """
    # Start with motif fingerprints
    features = motif_fingerprints.copy()

    # Join graph-structural features
    features = features.join(node_features, how="left")

    # Add temporal features if transaction data is provided
    if df is not None:
        temporal = _compute_temporal_features(df)
        features = features.join(temporal, how="left")

    features = features.fillna(0)
    return features


def _compute_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute per-address temporal behavioural features.

    Features:
    - tx_count_sent / tx_count_received: number of transactions
    - avg_inter_arrival_sent: mean time between outgoing txs
    - std_inter_arrival_sent: std of inter-arrival times (burstiness)
    - active_time_span: time between first and last transaction
    - avg_value_sent / avg_value_received: mean transaction value
    """
    records = {}

    all_addresses = set(df["from_address"]) | set(df["to_address"])

    for addr in all_addresses:
        sent = df[df["from_address"] == addr].sort_values("timestamp")
        received = df[df["to_address"] == addr].sort_values("timestamp")

        all_txs = pd.concat([sent, received]).sort_values("timestamp")

        # Inter-arrival times for sent transactions
        if len(sent) > 1:
            diffs = sent["timestamp"].diff().dropna()
            avg_iat = diffs.mean()
            std_iat = diffs.std()
        else:
            avg_iat = 0.0
            std_iat = 0.0

        # Active time span
        if len(all_txs) > 1:
            active_span = all_txs["timestamp"].max() - all_txs["timestamp"].min()
        else:
            active_span = 0

        records[addr] = {
            "tx_count_sent": len(sent),
            "tx_count_received": len(received),
            "avg_inter_arrival_sent": avg_iat,
            "std_inter_arrival_sent": std_iat,
            "active_time_span": active_span,
            "avg_value_sent": sent["value"].mean() if len(sent) > 0 else 0.0,
            "avg_value_received": received["value"].mean() if len(received) > 0 else 0.0,
        }

    result = pd.DataFrame.from_dict(records, orient="index")
    result.index.name = "address"
    return result


def normalize_features(X: pd.DataFrame) -> tuple[pd.DataFrame, StandardScaler]:
    """Normalize features using StandardScaler.

    Args:
        X: Feature DataFrame.

    Returns:
        Tuple of (normalized DataFrame, fitted scaler).
    """
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(
        scaler.fit_transform(X),
        index=X.index,
        columns=X.columns,
    )
    return X_scaled, scaler
