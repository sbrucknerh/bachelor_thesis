"""Coordinated malicious behaviour detection using ML models.

Provides both supervised (Random Forest, XGBoost) and unsupervised
(Isolation Forest, DBSCAN) detection approaches based on temporal
motif fingerprints and graph features.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.model_selection import cross_val_score, train_test_split

from config import DEFAULT_MOTIF_THRESHOLD, RANDOM_STATE, TEST_SIZE


# --- Supervised detection ---

def train_supervised_model(
    X: pd.DataFrame,
    y: pd.Series,
    model_type: str = "random_forest",
) -> tuple:
    """Train a supervised classifier on labeled address data.

    Args:
        X: Feature matrix (addresses x features).
        y: Binary labels (1 = malicious/illicit, 0 = benign/licit).
        model_type: "random_forest" or "xgboost".

    Returns:
        Tuple of (trained_model, X_test, y_test, cv_scores).
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    # Compensate for class imbalance — scam addresses are usually rare in the
    # curated training sets, so we re-weight the minority class.
    pos = int((y_train == 1).sum())
    neg = int((y_train == 0).sum())
    scale_pos_weight = (neg / pos) if pos > 0 else 1.0

    if model_type == "random_forest":
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            class_weight="balanced",
        )
    elif model_type == "xgboost":
        from xgboost import XGBClassifier
        model = XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            eval_metric="logloss",
            scale_pos_weight=scale_pos_weight,
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    # Cross-validation scores
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring="f1")

    # Train on full training set
    model.fit(X_train, y_train)

    return model, X_test, y_test, cv_scores


def predict_supervised(
    X: pd.DataFrame,
    model,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate predictions and probabilities from a trained supervised model.

    Args:
        X: Feature matrix.
        model: Trained sklearn/xgboost classifier.

    Returns:
        Tuple of (predictions, probabilities).
    """
    predictions = model.predict(X)
    probabilities = model.predict_proba(X)[:, 1]
    return predictions, probabilities


# --- Unsupervised detection ---

def detect_anomalies(
    X: pd.DataFrame,
    method: str = "isolation_forest",
    contamination: float = 0.1,
) -> tuple[np.ndarray, np.ndarray]:
    """Detect anomalous addresses using unsupervised methods.

    Args:
        X: Feature matrix (addresses x features).
        method: "isolation_forest" or "dbscan".
        contamination: Expected proportion of anomalies (for Isolation Forest).

    Returns:
        Tuple of (labels, scores) where:
        - labels: -1 for anomalies, 1 for normal (Isolation Forest)
                  or cluster labels with -1 for noise (DBSCAN)
        - scores: anomaly scores (lower = more anomalous for IF,
                  zeros for DBSCAN)
    """
    if method == "isolation_forest":
        model = IsolationForest(
            contamination=contamination,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
        labels = model.fit_predict(X)
        scores = model.decision_function(X)
    elif method == "dbscan":
        model = DBSCAN(eps=0.5, min_samples=5)
        labels = model.fit_predict(X)
        scores = np.zeros(len(labels))
    else:
        raise ValueError(f"Unknown method: {method}")

    return labels, scores


def cluster_addresses(
    X: pd.DataFrame,
    method: str = "dbscan",
    **kwargs,
) -> np.ndarray:
    """Cluster addresses by motif fingerprint similarity.

    Addresses in the same cluster exhibit similar temporal motif patterns,
    suggesting potential coordination.

    Args:
        X: Feature matrix.
        method: Clustering method ("dbscan").
        **kwargs: Additional parameters passed to the clustering algorithm.

    Returns:
        Array of cluster labels (-1 = noise/unclustered).
    """
    if method == "dbscan":
        params = {"eps": 0.5, "min_samples": 3}
        params.update(kwargs)
        model = DBSCAN(**params)
    else:
        raise ValueError(f"Unknown clustering method: {method}")

    return model.fit_predict(X)


def find_coordinated_clusters(
    X: pd.DataFrame,
    cluster_labels: np.ndarray,
    min_cluster_size: int = 3,
) -> list[dict]:
    """Identify clusters of addresses with coordinated behaviour.

    Flags clusters where addresses share similar rare motif distributions,
    suggesting coordinated wallets.

    Args:
        X: Feature matrix indexed by address.
        cluster_labels: Cluster assignments from cluster_addresses.
        min_cluster_size: Minimum cluster size to report.

    Returns:
        List of cluster dicts with keys: cluster_id, addresses, size,
        mean_fingerprint, coordination_score.
    """
    results = []
    addresses = X.index.tolist()

    unique_labels = set(cluster_labels)
    unique_labels.discard(-1)  # Exclude noise

    for label in sorted(unique_labels):
        mask = cluster_labels == label
        cluster_addresses = [addresses[i] for i in range(len(addresses)) if mask[i]]

        if len(cluster_addresses) < min_cluster_size:
            continue

        cluster_features = X[mask]

        # Coordination score: how similar are the members?
        # Lower intra-cluster variance = higher coordination
        feature_std = cluster_features.std().mean()
        coordination_score = 1.0 / (1.0 + feature_std)

        results.append({
            "cluster_id": int(label),
            "addresses": cluster_addresses,
            "size": len(cluster_addresses),
            "mean_fingerprint": cluster_features.mean().to_dict(),
            "coordination_score": coordination_score,
        })

    results.sort(key=lambda x: x["coordination_score"], reverse=True)
    return results


# --- Threshold-based baseline (original approach) ---

def detect_threshold_baseline(
    motifs: list[dict],
    threshold: int = DEFAULT_MOTIF_THRESHOLD,
) -> list[dict]:
    """Simple threshold-based detection as a baseline for comparison.

    Flags addresses participating in many suspicious motif types.
    This is the original detection approach — kept for baseline comparison.

    Args:
        motifs: List of motif dicts from temporal_motifs.extract_temporal_motifs_nx.
        threshold: Minimum suspicious motif count to flag.

    Returns:
        List of flagged address dicts.
    """
    suspicious_types = {"fan_out", "fan_in", "cycle"}
    address_motifs = defaultdict(lambda: defaultdict(int))

    for motif in motifs:
        if motif["motif_type"] in suspicious_types:
            for node in motif["nodes"]:
                address_motifs[node][motif["motif_type"]] += 1

    flagged = []
    for address, type_counts in address_motifs.items():
        total = sum(type_counts.values())
        if total >= threshold:
            weights = {"fan_out": 1.0, "fan_in": 1.0, "cycle": 1.5}
            risk_score = sum(
                count * weights.get(mtype, 0.5)
                for mtype, count in type_counts.items()
            )
            flagged.append({
                "address": address,
                "motif_count": total,
                "suspicious_types": dict(type_counts),
                "risk_score": risk_score,
            })

    flagged.sort(key=lambda x: x["risk_score"], reverse=True)
    return flagged
