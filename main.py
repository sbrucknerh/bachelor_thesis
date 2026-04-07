"""Main pipeline for coordinated malicious behaviour detection in blockchain networks.

Usage:
    python main.py                     # Run with synthetic data
    python main.py --data path/to.csv  # Run with a real dataset
"""

import argparse
import sys

from config import DEFAULT_DELTA, DEFAULT_MOTIF_THRESHOLD, DEFAULT_N_ADDRESSES, DEFAULT_N_TRANSACTIONS
from src.data_loading import load_transactions
from src.synthetic_data import generate_synthetic_transactions
from src.graph_construction import build_transaction_graph, get_graph_stats
from src.temporal_motifs import extract_temporal_motifs, compute_motif_distribution
from src.detection import detect_coordinated_behaviour
from src.visualization import plot_transaction_graph, plot_motif_distribution, plot_detection_results


def main():
    args = parse_args()

    # --- Step 1: Load or generate transaction data ---
    if args.data:
        print(f"Loading transactions from {args.data}...")
        df = load_transactions(args.data)
    else:
        print("Generating synthetic transaction data...")
        df = generate_synthetic_transactions(
            n_addresses=args.n_addresses,
            n_transactions=args.n_transactions,
        )

    print(f"  Transactions: {len(df)}")
    print(f"  Unique addresses: {df['from_address'].nunique() + df['to_address'].nunique()}")
    print()

    # --- Step 2: Build transaction graph ---
    print("Building transaction graph...")
    G = build_transaction_graph(df)
    stats = get_graph_stats(G)

    print("  Graph statistics:")
    for key, value in stats.items():
        print(f"    {key}: {value}")
    print()

    # --- Step 3: Extract temporal motifs ---
    print(f"Extracting temporal motifs (delta={args.delta}s)...")
    motifs = extract_temporal_motifs(G, delta=args.delta)
    distribution = compute_motif_distribution(motifs)

    print(f"  Total motifs found: {len(motifs)}")
    print("  Distribution:")
    for motif_type, count in sorted(distribution.items(), key=lambda x: -x[1]):
        print(f"    {motif_type}: {count}")
    print()

    # --- Step 4: Detect coordinated behaviour ---
    print(f"Detecting coordinated behaviour (threshold={args.threshold})...")
    flagged = detect_coordinated_behaviour(motifs, threshold=args.threshold)

    if flagged:
        print(f"  Flagged {len(flagged)} suspicious addresses:")
        for result in flagged[:5]:
            addr = result["address"][:16] + "..."
            print(f"    {addr}  risk_score={result['risk_score']:.1f}  motifs={result['motif_count']}")
    else:
        print("  No suspicious addresses detected.")
    print()

    # --- Step 5: Visualize results ---
    if not args.no_plot:
        highlight = [r["address"] for r in flagged] if flagged else None
        plot_transaction_graph(G, highlight_nodes=highlight)
        if distribution:
            plot_motif_distribution(distribution)
        if flagged:
            plot_detection_results(flagged)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Detect coordinated malicious behaviour in blockchain transaction networks."
    )
    parser.add_argument("--data", type=str, default=None, help="Path to transaction dataset (CSV or JSON)")
    parser.add_argument("--n-addresses", type=int, default=DEFAULT_N_ADDRESSES, help="Number of synthetic addresses")
    parser.add_argument("--n-transactions", type=int, default=DEFAULT_N_TRANSACTIONS, help="Number of synthetic transactions")
    parser.add_argument("--delta", type=int, default=DEFAULT_DELTA, help="Temporal motif time window in seconds")
    parser.add_argument("--threshold", type=int, default=DEFAULT_MOTIF_THRESHOLD, help="Detection threshold")
    parser.add_argument("--no-plot", action="store_true", help="Skip visualization")
    return parser.parse_args()


if __name__ == "__main__":
    main()
