"""Main pipeline for coordinated malicious behaviour detection in blockchain networks.

Six-stage architecture:
    1. Load & clean transactions (+ optional per-address labels)
    2. Build NetworkX graph + per-node structural features
    3. Count per-node temporal motif fingerprints via --motif-backend
    4. Assemble feature matrix per --features {aggregates, motifs, full}
    5. Train/test (supervised) OR anomaly-detect + cluster (unsupervised)
    6. Visualize (unless --no-plot)

Examples:
    # Supervised, full feature set, δ sweep, HuggingFace GNN subset
    python main.py --mode supervised --dataset-type huggingface-gnn \
        --months 2025-06 2025-07 --features full --delta 3600 86400 --no-plot

    # Supervised on BLTE raw transactions
    python main.py --mode supervised --data data/raw/ethereum_dataset_labelled \
        --dataset-type blte

    # Unsupervised on one month of HF GNN data with DBSCAN
    python main.py --mode unsupervised --dataset-type huggingface-gnn \
        --months 2025-07 --features full --unsupervised-method dbscan

    # Quick synthetic-data sanity check (no dataset needed)
    python main.py --mode unsupervised
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from config import (
    DEFAULT_DELTA,
    DEFAULT_MOTIF_BACKEND,
    DEFAULT_N_ADDRESSES,
    DEFAULT_N_TRANSACTIONS,
    DEFAULT_SUPERVISED_MODEL,
    DEFAULT_UNSUPERVISED_METHOD,
)
from src.data_loading import (
    load_blte_dataset,
    load_ethereum_fraud_dataset,
    load_ethereum_fraud_gnn_dataset,
    load_ethereum_fraud_labels,
    load_transactions,
)
from src.graph_construction import build_transaction_graph, get_graph_stats, get_node_features
from src.synthetic_data import generate_synthetic_transactions


def main():
    args = parse_args()
    run_pipeline(args)


def run_pipeline(args) -> None:
    """Six-stage pipeline. `--mode graph` and `--mode motifs` short-circuit
    after stages 2 and 3 respectively; `supervised` / `unsupervised` run all stages.
    """
    txs, node_labels = _stage1_load_and_clean(args)
    G, node_features = _stage2_build_graph(
        txs,
        compute_node_features=(args.mode != "graph"),
    )

    # --- Early exit: --mode graph ---
    if args.mode == "graph":
        _save_graph(G, _graph_tag(args))
        if not args.no_plot:
            _plot_graph_only(G)
        return

    # δ-independent: compute once, reuse across the delta loop.
    aggregates = _aggregate_tx_features(txs)

    deltas = _deltas_for(args)
    last_ctx: dict | None = None
    all_results: list[dict] = []

    for delta in deltas:
        fingerprints = _stage3_count_motifs(txs, delta, args.motif_backend, node_labels)
        _summarize_fingerprint_sparsity(fingerprints)
        _save_fingerprints(fingerprints, _fingerprint_tag(args, delta))

        if args.mode == "motifs":
            last_ctx = {"G": G, "X": fingerprints, "fingerprints": fingerprints, "result": {}}
            continue

        X = _stage4_assemble_features(fingerprints, aggregates, node_features, args.features)

        tag = _build_result_tag(args, delta)
        result = _stage5_train_or_detect(X, node_labels, args, tag=tag)
        all_results.append({"delta": delta, **_metrics_only(result)})

        last_ctx = {
            "G": G,
            "X": X,
            "fingerprints": fingerprints,
            "result": result,
        }

    if args.mode == "supervised" and len([r for r in all_results if "f1" in r]) > 1:
        _print_sweep_summary([r for r in all_results if "f1" in r])

    if not args.no_plot and last_ctx is not None:
        _stage6_visualize(last_ctx, args)


# --- Stage 1: load + clean ---

def _stage1_load_and_clean(args) -> tuple[pd.DataFrame, pd.Series | None]:
    """Dispatch to the right loader and return cleaned (txs, node_labels)."""
    dataset_type = args.dataset_type
    node_labels: pd.Series | None = None

    if dataset_type == "realcats":
        if not args.data:
            raise ValueError(
                "--data <path-to-realcats_transactions> is required for Real-CATS. "
                "Run scripts/extract_realcats.py first to extract labeled-address "
                "JSONs from the Kaggle zip."
            )
        from src.data_loading import load_realcats_transactions
        txs, node_labels = load_realcats_transactions(args.data)

    if dataset_type == "blte":
        if not args.data:
            raise ValueError("--data <path-to-ethereum_dataset_labelled> is required for BLTE.")
        print(f"Loading BLTE dataset from {args.data}...")
        txs, address_features = load_blte_dataset(args.data)
        if not address_features.empty and "Scam" in address_features.columns:
            node_labels = address_features["Scam"].astype(int)

    elif dataset_type == "huggingface":
        # Default to 100k rows for the raw stream so we never accidentally pull the full 1.6B-row dataset.
        stream_rows = args.max_rows if args.max_rows else 100_000
        print(f"Loading HuggingFace dataset (streaming, max_rows={stream_rows})...")
        txs, _ = load_ethereum_fraud_dataset(max_rows=stream_rows)
        txs = txs.dropna(subset=["to_address"]).reset_index(drop=True)
        txs["from_address"] = txs["from_address"].str.lower()
        txs["to_address"] = txs["to_address"].str.lower()

        if args.mode == "supervised":
            print(f"Loading HuggingFace address labels (balanced={not args.full_labels})...")
            label_df = load_ethereum_fraud_labels(balanced=not args.full_labels)
            node_labels = (
                label_df[["address", "is_scam"]]
                .dropna(subset=["address"])
                .drop_duplicates("address")
                .set_index("address")["is_scam"]
                .astype(int)
            )

    elif dataset_type == "huggingface-gnn":
        print(f"Loading HuggingFace GNN edge subset (months={args.months or 'latest'})...")
        txs, node_labels = load_ethereum_fraud_gnn_dataset(
            months=args.months, use_full=args.use_full_edges,
        )

    elif args.data:
        print(f"Loading transactions from {args.data}...")
        txs = load_transactions(args.data, column_map=args.column_map)

    else:
        print("Generating synthetic transaction data...")
        txs = generate_synthetic_transactions(
            n_addresses=args.n_addresses,
            n_transactions=args.n_transactions,
        )

    # Shared cleaning
    txs = txs.dropna(subset=["to_address", "from_address"]).reset_index(drop=True)

    # Universal --max-rows cap: applies to every dataset type, not just the raw HF stream.
    # Downstream stages iterate per-row or per-node, so this is the cheapest place to cut.
    if args.max_rows and len(txs) > args.max_rows:
        txs = txs.head(args.max_rows).reset_index(drop=True)
        print(f"  Applied --max-rows={args.max_rows}: {len(txs)} transactions")
        if node_labels is not None:
            keep = set(txs["from_address"]) | set(txs["to_address"])
            node_labels = node_labels.loc[node_labels.index.intersection(keep)]

    if args.mode == "supervised" and node_labels is None:
        raise ValueError(
            f"Supervised mode requires per-address labels, but dataset_type "
            f"{dataset_type!r} did not produce any. Use --dataset-type "
            f"{{blte,huggingface,huggingface-gnn}} for supervised runs."
        )

    print(f"  Transactions: {len(txs)}")
    print(f"  Unique senders: {txs['from_address'].nunique()}")
    print(f"  Unique receivers: {txs['to_address'].nunique()}")
    if node_labels is not None:
        pos = int(node_labels.sum())
        neg = int((node_labels == 0).sum())
        print(f"  Labeled nodes: {len(node_labels)} ({pos} scam, {neg} non-scam)")
    print()

    return txs, node_labels


# --- Stage 2: build graph ---

def _stage2_build_graph(txs: pd.DataFrame, compute_node_features: bool = True) -> tuple:
    """Build NetworkX graph and (optionally) per-node structural features.

    `compute_node_features=False` skips the PageRank / clustering pass — useful
    for `--mode graph` where the feature matrix is never consumed.
    """
    print("Building transaction graph (NetworkX)...")
    G = build_transaction_graph(txs)
    _print_stats(get_graph_stats(G))
    node_features = get_node_features(G) if compute_node_features else None
    return G, node_features


# --- Stage 3: motif fingerprints ---

def _stage3_count_motifs(
    txs: pd.DataFrame,
    delta: int,
    backend: str,
    node_labels: pd.Series | None,
) -> pd.DataFrame:
    """Run the selected motif backend, optionally restricted to labeled nodes."""
    from src.motifs import get_backend
    count_motifs = get_backend(backend)
    node_filter = set(node_labels.index) if node_labels is not None else None

    print(f"\n=== δ = {delta}s ===")
    msg = f"Counting motif fingerprints via {backend} backend"
    if node_filter is not None:
        msg += f" (labeled nodes={len(node_filter)})"
    print(msg + "...")

    return count_motifs(txs, delta=delta, node_filter=node_filter)


# --- Stage 4: assemble features ---

def _stage4_assemble_features(
    fingerprints: pd.DataFrame,
    aggregates: pd.DataFrame,
    node_features: pd.DataFrame,
    feature_strategy: str,
) -> pd.DataFrame:
    """Select which features feed stage 5."""
    if feature_strategy == "aggregates":
        return aggregates
    if feature_strategy == "motifs":
        return fingerprints
    if feature_strategy == "full":
        return (
            fingerprints
            .join(aggregates, how="left")
            .join(node_features, how="left")
            .fillna(0)
        )
    raise ValueError(f"Unknown feature strategy: {feature_strategy!r}")


# --- Stage 5: train or detect ---

def _stage5_train_or_detect(
    X: pd.DataFrame,
    node_labels: pd.Series | None,
    args,
    *,
    tag: str,
) -> dict:
    if args.mode == "supervised":
        assert node_labels is not None, "stage 1 should have guaranteed labels"
        X_train, y = _prepare_xy(X, node_labels)
        if len(X_train) == 0:
            print("Error: No labeled nodes remain after cleanup — check that the "
                  "selected --months actually include labeled addresses.")
            return {}
        payload_extra = _build_payload_extra(args)
        metrics = _train_and_save(
            X_train, y,
            model_type=args.supervised_model,
            tag=tag,
            payload_extra=payload_extra,
            no_plot=args.no_plot,
        )
        return {"metrics": metrics}

    if args.mode == "unsupervised":
        return _run_unsupervised_detection(X, args, tag=tag)

    raise ValueError(f"Unknown mode: {args.mode!r}")


def _run_unsupervised_detection(X: pd.DataFrame, args, *, tag: str) -> dict:
    from src.detection import cluster_addresses, detect_anomalies, find_coordinated_clusters
    from src.evaluation import export_results
    from src.feature_engineering import normalize_features

    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.clip(lower=-1e30, upper=1e30)
    X = X[X.notna().all(axis=1)]

    if X.empty:
        print("Error: No valid rows remain after cleanup.")
        return {}

    features_scaled, _ = normalize_features(X)
    print(f"  Feature matrix: {features_scaled.shape[0]} addresses x {features_scaled.shape[1]} features")

    print(f"\nRunning anomaly detection ({args.unsupervised_method})...")
    anomaly_labels, _ = detect_anomalies(features_scaled, method=args.unsupervised_method)
    n_anomalies = int((anomaly_labels == -1).sum())
    print(f"  Anomalies detected: {n_anomalies}")

    print("Clustering addresses by feature similarity...")
    cluster_labels = cluster_addresses(features_scaled)
    n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
    print(f"  Clusters found: {n_clusters}")

    coordinated = find_coordinated_clusters(features_scaled, cluster_labels)
    if coordinated:
        print("\n  Coordinated clusters:")
        for c in coordinated[:5]:
            print(f"    Cluster {c['cluster_id']}: {c['size']} addresses, "
                  f"coordination_score={c['coordination_score']:.3f}")

    payload = {
        "mode": "unsupervised",
        "method": args.unsupervised_method,
        "features": args.features,
        "motif_backend": args.motif_backend,
        "n_anomalies": n_anomalies,
        "n_clusters": n_clusters,
        "coordinated_top": [
            {k: c[k] for k in ("cluster_id", "size", "coordination_score")}
            for c in coordinated[:10]
        ],
    }
    results_path = export_results(payload, filename=f"results_{tag}.json")
    print(f"\n  Results saved to {results_path}")

    return {
        "n_anomalies": n_anomalies,
        "n_clusters": n_clusters,
        "anomaly_labels": anomaly_labels,
        "cluster_labels": cluster_labels,
        "features_scaled": features_scaled,
    }


# --- Stage 6: visualize ---

def _stage6_visualize(ctx: dict, args) -> None:
    from src.temporal_motifs import compute_motif_distribution
    from src.visualization import (
        plot_cluster_tsne,
        plot_motif_distribution,
        plot_motif_fingerprint_heatmap,
        plot_transaction_graph,
    )

    G = ctx["G"]
    X = ctx["X"]
    fingerprints = ctx["fingerprints"]
    result = ctx["result"]

    highlight: list | None = None
    title = "Transaction Network"
    if args.mode == "unsupervised" and "anomaly_labels" in result:
        anomaly_labels = result["anomaly_labels"]
        highlight = [
            X.index[i] for i in range(len(anomaly_labels))
            if anomaly_labels[i] == -1
        ]
        title = "Transaction Network (anomalies in red)"
    elif args.mode == "motifs" and not fingerprints.empty:
        motif_mask = (fingerprints.values != 0).any(axis=1)
        highlight = [
            fingerprints.index[i] for i in range(len(motif_mask)) if motif_mask[i]
        ]
        title = "Transaction Network (motif participants in red)"

    plot_transaction_graph(G, highlight_nodes=highlight, title=title)

    if not fingerprints.empty:
        distribution = compute_motif_distribution(fingerprints.sum(axis=0).values)
        if distribution:
            plot_motif_distribution(distribution)
        plot_motif_fingerprint_heatmap(fingerprints)

    if args.mode == "unsupervised" and "features_scaled" in result:
        features_scaled = result["features_scaled"]
        cluster_labels = result["cluster_labels"]
        if len(features_scaled) > 5:
            plot_cluster_tsne(features_scaled, cluster_labels)
    # Supervised ROC/PR/confusion plots are emitted inside _train_and_save.


# --- Shared helpers (kept from previous revisions) ---

def _deltas_for(args) -> list[int]:
    """`motifs`/`full` sweep over args.delta; `aggregates` only runs once."""
    if args.features == "aggregates":
        return [args.delta[0] if args.delta else DEFAULT_DELTA]
    return args.delta if args.delta else [DEFAULT_DELTA]


def _build_result_tag(args, delta: int) -> str:
    dataset_part = (args.dataset_type or "synthetic").replace("-", "_")
    parts = [dataset_part, args.mode, args.features]
    if args.features in ("motifs", "full"):
        parts.append(args.motif_backend)
        parts.append(f"delta{delta}")
    return "_".join(parts)


def _graph_tag(args) -> str:
    dataset_part = (args.dataset_type or "synthetic").replace("-", "_")
    return f"{dataset_part}_graph"


def _fingerprint_tag(args, delta: int) -> str:
    dataset_part = (args.dataset_type or "synthetic").replace("-", "_")
    return f"{dataset_part}_fingerprints_{args.motif_backend}_delta{delta}"


def _save_graph(G, tag: str) -> None:
    """Persist the NetworkX graph to results/ as a pickle."""
    import pickle
    from config import RESULTS_DIR
    import os
    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = os.path.join(RESULTS_DIR, f"{tag}.pkl")
    with open(path, "wb") as f:
        pickle.dump(G, f)
    print(f"  Graph saved to {path}  ({G.number_of_nodes()} nodes, {G.number_of_edges()} edges)")


def _save_fingerprints(fp: pd.DataFrame, tag: str) -> None:
    """Persist a motif fingerprint DataFrame to results/ as Parquet."""
    from config import RESULTS_DIR
    import os
    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = os.path.join(RESULTS_DIR, f"{tag}.parquet")
    fp.to_parquet(path)
    print(f"  Fingerprints saved to {path}  ({fp.shape[0]} rows x {fp.shape[1]} cols)")


def _plot_graph_only(G) -> None:
    """Render just the transaction graph (no highlighted nodes)."""
    from src.visualization import plot_transaction_graph
    plot_transaction_graph(G, title="Transaction Network")


def _build_payload_extra(args) -> dict:
    payload = {
        "mode": args.mode,
        "dataset_type": args.dataset_type,
        "features": args.features,
    }
    if args.features in ("motifs", "full"):
        payload["motif_backend"] = args.motif_backend
    return payload


def _metrics_only(result: dict) -> dict:
    """Extract flat metric fields from the stage-5 result for sweep summaries."""
    if not result:
        return {}
    if "metrics" in result:
        return dict(result["metrics"])
    return {k: v for k, v in result.items() if isinstance(v, (int, float))}


def _summarize_fingerprint_sparsity(fp: pd.DataFrame) -> None:
    if fp.empty:
        print("  Fingerprint is empty.")
        return
    total_cells = fp.size
    nonzero_cells = int((fp.values != 0).sum())
    nonzero_rows = int((fp.values != 0).any(axis=1).sum())
    nonzero_cols = int((fp.values != 0).any(axis=0).sum())
    pct = 100 * nonzero_cells / total_cells if total_cells else 0.0
    print(f"  Fingerprint shape: {fp.shape}")
    print(f"  Populated cells: {nonzero_cells}/{total_cells} ({pct:.2f}%)")
    print(f"  Addresses with any motif: {nonzero_rows}/{fp.shape[0]}")
    print(f"  Motif types that ever fire: {nonzero_cols}/{fp.shape[1]}")


def _prepare_xy(
    features: pd.DataFrame, node_labels: pd.Series
) -> tuple[pd.DataFrame, pd.Series]:
    joined = features.join(node_labels.rename("is_scam"), how="inner")
    y = joined["is_scam"].astype(int)
    X = joined.drop(columns=["is_scam"])
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.clip(lower=-1e30, upper=1e30)
    mask = X.notna().all(axis=1)
    return X[mask], y[mask]


def _train_and_save(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    model_type: str,
    tag: str,
    payload_extra: dict,
    no_plot: bool,
) -> dict:
    from src.detection import predict_supervised, train_supervised_model
    from src.evaluation import (
        evaluate_classifier,
        export_results,
        plot_confusion_matrix,
        plot_roc_pr_curves,
    )

    print(f"  Labeled addresses: {len(X)} ({int(y.sum())} malicious, {int((y == 0).sum())} benign)")
    print(f"\nTraining {model_type} classifier...")
    model, X_test, y_test, cv_scores = train_supervised_model(X, y, model_type=model_type)
    print(f"  Cross-validation F1: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

    predictions, probabilities = predict_supervised(X_test, model)
    metrics = evaluate_classifier(y_test.values, predictions, probabilities)

    print("\n  Test set results:")
    for metric, value in metrics.items():
        print(f"    {metric}: {value:.4f}")

    payload = {"model": model_type, "metrics": metrics, **payload_extra}
    results_path = export_results(payload, filename=f"results_{tag}.json")
    print(f"\n  Results saved to {results_path}")

    if not no_plot:
        plot_roc_pr_curves(y_test.values, probabilities, title_prefix=f"{model_type} — ")
        plot_confusion_matrix(y_test.values, predictions)

    return metrics


def _aggregate_tx_features(txs: pd.DataFrame) -> pd.DataFrame:
    sender = txs.groupby("from_address").agg(
        tx_sent=("tx_hash", "count"),
        value_sent_sum=("value", "sum"),
        value_sent_mean=("value", "mean"),
        value_sent_std=("value", "std"),
        first_tx_sent=("timestamp", "min"),
        last_tx_sent=("timestamp", "max"),
    )
    receiver = txs.groupby("to_address").agg(
        tx_received=("tx_hash", "count"),
        value_received_sum=("value", "sum"),
        value_received_mean=("value", "mean"),
        value_received_std=("value", "std"),
        first_tx_received=("timestamp", "min"),
        last_tx_received=("timestamp", "max"),
    )
    features = sender.join(receiver, how="outer").fillna(0)
    features["active_span"] = (
        features[["last_tx_sent", "last_tx_received"]].max(axis=1)
        - features[["first_tx_sent", "first_tx_received"]].replace(0, np.nan).min(axis=1)
    ).fillna(0)
    features.index.name = "address"
    return features


def _print_sweep_summary(rows: list[dict]) -> None:
    headers = ["delta", "f1", "precision", "recall", "roc_auc", "pr_auc"]
    col_widths = [max(len(h), 10) for h in headers]
    print("\n=== δ sweep summary ===")
    print("  " + "  ".join(h.rjust(w) for h, w in zip(headers, col_widths)))
    for row in rows:
        values = [
            f"{row['delta']}",
            f"{row.get('f1', 0):.4f}",
            f"{row.get('precision', 0):.4f}",
            f"{row.get('recall', 0):.4f}",
            f"{row.get('roc_auc', 0):.4f}",
            f"{row.get('pr_auc', 0):.4f}",
        ]
        print("  " + "  ".join(v.rjust(w) for v, w in zip(values, col_widths)))


def _print_stats(stats: dict) -> None:
    print("  Graph statistics:")
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"    {key}: {value:.4f}")
        else:
            print(f"    {key}: {value}")
    print()


# --- CLI ---

def parse_args():
    parser = argparse.ArgumentParser(
        description="Detect coordinated malicious behaviour in blockchain transaction networks."
    )
    parser.add_argument("--mode", type=str, default="motifs",
                        choices=["graph", "motifs", "supervised", "unsupervised"],
                        help="Pipeline mode. 'graph' stops after stage 2 (save + plot the graph). "
                             "'motifs' stops after stage 3 (save + plot motif fingerprints). "
                             "'supervised' / 'unsupervised' run all six stages. Default: motifs.")
    parser.add_argument("--data", type=str, default=None,
                        help="Path to transaction dataset (CSV, JSON, Parquet) or labeled dataset dir")
    parser.add_argument("--dataset-type", type=str, default=None,
                        choices=["blte", "huggingface", "huggingface-gnn", "realcats"],
                        help="Dataset type. Required for supervised mode; omit for a raw file or "
                             "for synthetic data.")
    parser.add_argument("--max-rows", type=int, default=None,
                        help="Cap the transaction count for any dataset (applied at the end of stage 1). "
                             "For --dataset-type huggingface this also limits the stream; "
                             "falls back to 100000 there if unset.")
    parser.add_argument("--full-labels", action="store_true",
                        help="Use addr_labels_big.csv (~1M addresses) instead of balanced (~115k)")
    parser.add_argument("--months", type=str, nargs="+", default=None,
                        help="Months to load for huggingface-gnn (e.g. 2025-04 2025-05). "
                             "Default: latest month.")
    parser.add_argument("--use-full-edges", action="store_true",
                        help="Download the 21.5 GB full edge file for huggingface-gnn.")
    parser.add_argument("--features", type=str, default="motifs",
                        choices=["aggregates", "motifs", "full"],
                        help="Feature strategy: aggregates (tx stats only), motifs (fingerprint only), "
                             "or full (motifs + aggregates + graph features). Default: motifs.")
    parser.add_argument("--motif-backend", type=str, default=DEFAULT_MOTIF_BACKEND,
                        choices=["networkx", "snap", "raphtory"],
                        help="Motif counter backend (default: raphtory).")
    parser.add_argument("--column-map", type=str, default=None,
                        help="Column mapping preset (bigquery_ethereum, etherscan)")
    parser.add_argument("--n-addresses", type=int, default=DEFAULT_N_ADDRESSES)
    parser.add_argument("--n-transactions", type=int, default=DEFAULT_N_TRANSACTIONS)
    parser.add_argument("--delta", type=int, nargs="+", default=None,
                        help="Temporal window(s) in seconds (e.g. --delta 3600 86400)")
    parser.add_argument("--supervised-model", type=str, default=DEFAULT_SUPERVISED_MODEL,
                        choices=["random_forest", "xgboost"])
    parser.add_argument("--unsupervised-method", type=str, default=DEFAULT_UNSUPERVISED_METHOD,
                        choices=["isolation_forest", "dbscan"])
    parser.add_argument("--no-plot", action="store_true", help="Skip visualization")
    return parser.parse_args()


if __name__ == "__main__":
    main()
