"""Generate synthetic Ethereum-like transaction datasets for development and testing."""

from __future__ import annotations

import hashlib
import time

import numpy as np
import pandas as pd

from config import DEFAULT_N_ADDRESSES, DEFAULT_N_TRANSACTIONS


def _random_address(rng: np.random.Generator) -> str:
    """Generate a random Ethereum-like address."""
    raw = rng.bytes(20)
    return "0x" + raw.hex()


def generate_synthetic_transactions(
    n_addresses: int = DEFAULT_N_ADDRESSES,
    n_transactions: int = DEFAULT_N_TRANSACTIONS,
    seed: int = 42,
    include_malicious_patterns: bool = True,
) -> pd.DataFrame:
    """Generate a synthetic Ethereum transaction dataset.

    Args:
        n_addresses: Number of unique addresses to generate.
        n_transactions: Number of transactions to generate.
        seed: Random seed for reproducibility.
        include_malicious_patterns: If True, inject coordinated behaviour patterns.

    Returns:
        DataFrame with columns: from_address, to_address, value, timestamp, tx_hash.
    """
    rng = np.random.default_rng(seed)

    # Generate address pool
    addresses = [_random_address(rng) for _ in range(n_addresses)]

    # Generate random background transactions
    base_time = int(time.time()) - 86400 * 30  # 30 days ago

    from_addrs = [addresses[i] for i in rng.integers(0, n_addresses, size=n_transactions)]
    to_addrs = [addresses[i] for i in rng.integers(0, n_addresses, size=n_transactions)]
    values = rng.exponential(scale=1.0, size=n_transactions).round(6)
    timestamps = sorted(base_time + rng.integers(0, 86400 * 30, size=n_transactions))

    tx_hashes = [
        "0x" + hashlib.sha256(f"tx_{i}_{seed}".encode()).hexdigest()
        for i in range(n_transactions)
    ]

    df = pd.DataFrame({
        "from_address": from_addrs,
        "to_address": to_addrs,
        "value": values,
        "timestamp": timestamps,
        "tx_hash": tx_hashes,
    })

    # Remove self-loops
    df = df[df["from_address"] != df["to_address"]].reset_index(drop=True)

    if include_malicious_patterns and n_addresses >= 50:
        malicious_txs = _inject_malicious_patterns(addresses, base_time, rng)
        df = pd.concat([df, malicious_txs], ignore_index=True)
        df = df.sort_values("timestamp").reset_index(drop=True)

    return df


def _inject_malicious_patterns(
    addresses: list[str],
    base_time: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Inject known coordinated behaviour patterns into the dataset.

    Patterns injected:
    - Fan-out: One address sends to many in a short time window
    - Fan-in: Many addresses send to one in a short time window
    - Cycle: A -> B -> C -> A within a short time window
    """
    rows = []
    pattern_time = base_time + 86400 * 15  # Middle of the time range

    # Fan-out pattern: one source sends to 10 targets within 5 minutes
    source = addresses[0]
    targets = addresses[1:11]
    for i, target in enumerate(targets):
        rows.append({
            "from_address": source,
            "to_address": target,
            "value": round(rng.uniform(0.5, 1.5), 6),
            "timestamp": pattern_time + i * 30,  # 30 seconds apart
            "tx_hash": "0x" + hashlib.sha256(f"fanout_{i}".encode()).hexdigest(),
        })

    # Fan-in pattern: 10 sources send to one target within 5 minutes
    target = addresses[20]
    sources = addresses[21:31]
    for i, src in enumerate(sources):
        rows.append({
            "from_address": src,
            "to_address": target,
            "value": round(rng.uniform(0.5, 1.5), 6),
            "timestamp": pattern_time + 3600 + i * 30,
            "tx_hash": "0x" + hashlib.sha256(f"fanin_{i}".encode()).hexdigest(),
        })

    # Cycle pattern: A -> B -> C -> D -> A within 10 minutes
    cycle_addrs = addresses[40:44]
    for i in range(len(cycle_addrs)):
        src = cycle_addrs[i]
        dst = cycle_addrs[(i + 1) % len(cycle_addrs)]
        rows.append({
            "from_address": src,
            "to_address": dst,
            "value": round(rng.uniform(1.0, 2.0), 6),
            "timestamp": pattern_time + 7200 + i * 120,
            "tx_hash": "0x" + hashlib.sha256(f"cycle_{i}".encode()).hexdigest(),
        })

    return pd.DataFrame(rows)
