"""Load and validate Ethereum transaction datasets from CSV, JSON, or Parquet files."""

from __future__ import annotations

import os
import zipfile

import pandas as pd

from config import COLUMN_MAPS


REQUIRED_COLUMNS = ["from_address", "to_address", "value", "timestamp"]


def load_transactions(
    filepath: str,
    column_map: str | dict | None = None,
) -> pd.DataFrame:
    """Load transaction data from a CSV, JSON, or Parquet file.

    Args:
        filepath: Path to the transaction dataset file.
        column_map: Either a key from config.COLUMN_MAPS (e.g. "bigquery_ethereum",
            "etherscan") or a custom dict mapping standard names to source column names.
            If None, assumes columns already match the standard schema.

    Returns:
        DataFrame with columns: from_address, to_address, value, timestamp, tx_hash.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset not found: {filepath}")

    ext = os.path.splitext(filepath)[1].lower()

    if ext == ".csv":
        df = pd.read_csv(filepath)
    elif ext == ".json":
        df = pd.read_json(filepath)
    elif ext == ".parquet":
        df = pd.read_parquet(filepath)
    else:
        raise ValueError(f"Unsupported file format: {ext}. Use .csv, .json, or .parquet")

    if column_map is not None:
        df = _apply_column_map(df, column_map)

    df = _validate_and_normalize(df)
    return df



def load_ethereum_fraud_dataset(
    split: str = "train",
    max_rows: int | None = 100_000,
    streaming: bool = True,
) -> tuple[pd.DataFrame, pd.Series | None]:
    """Load the Ethereum Fraud Dataset by Activity from HuggingFace.

    Dataset: fesevu/ethereum_fraud_dataset_by_activity (~1.6B rows, 92GB Parquet).
    Requires the `datasets` package and HuggingFace authentication
    (run `huggingface-cli login` beforehand).

    The default config schema is: from_address, to_address, block_number, timestamp,
    value_wei, tx_fee_wei, nonce, input_data_size, contract_creation, tx_hash, day.
    Fraud labels (is_scam) are NOT included in the default config.

    Because the full dataset is 92GB, streaming mode is used by default with a
    row limit. Set `streaming=False` and `max_rows=None` only if you have
    enough memory and disk space for the full dataset.

    Args:
        split: Dataset split to load (default: "train").
        max_rows: Maximum rows to load. Set to None for the full dataset (caution:
            1.6B rows). Default: 100,000.
        streaming: If True, streams the dataset (no full download).

    Returns:
        Tuple of (transactions_df, labels) where:
        - transactions_df: DataFrame with standard columns (from_address, to_address,
          value, timestamp, tx_hash) plus block_number, nonce, input_data_size,
          contract_creation, tx_fee_wei.
        - labels: Binary Series (1 = scam) if `is_scam` is in the dataset, else None.
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError(
            "The `datasets` package is required to load HuggingFace datasets. "
            "Install it with: pip install datasets"
        )

    ds = load_dataset(
        "fesevu/ethereum_fraud_dataset_by_activity",
        split=split,
        streaming=streaming,
    )

    if streaming:
        if max_rows is None:
            raise ValueError("max_rows must be set when streaming=True.")
        ds = ds.take(max_rows)
        df = pd.DataFrame(list(ds))
    else:
        if max_rows is not None:
            ds = ds.select(range(min(max_rows, len(ds))))
        df = ds.to_pandas()

    # Rename source columns to standard names
    df = df.rename(columns={"value_wei": "value"})

    # Parse timestamp (string like "2019-01-13 00:08:24") to unix seconds
    if "timestamp" in df.columns and df["timestamp"].dtype != "int64":
        df["timestamp"] = (
            pd.to_datetime(df["timestamp"]).astype("datetime64[s]").astype("int64")
        )

    # Convert wei to ETH
    if "value" in df.columns:
        df["value"] = df["value"].astype(float) / 1e18

    labels = df["is_scam"].astype(int) if "is_scam" in df.columns else None
    return df, labels


def load_ethereum_fraud_labels(balanced: bool = True) -> pd.DataFrame:
    """Load per-address fraud labels from the HuggingFace dataset repo.

    The `fesevu/ethereum_fraud_dataset_by_activity` repo publishes scam labels
    in separate CSV files (not in the default parquet config). Labels come from
    Etherscan tags + curated scam lists.

    Args:
        balanced: If True, load addr_labels_balanced.csv (~115k balanced subset).
            If False, load addr_labels_big.csv (~1M+ raw addresses).

    Returns:
        DataFrame with columns:
        - address (str, lowercase 0x...)
        - is_scam (0/1)
        - description (e.g. "Verified", "Phishing")
        - activity_start_ts, activity_end_ts (timestamps)
        - is_contract (0/1)
    """
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        raise ImportError(
            "The `huggingface_hub` package is required. Install with: "
            "pip install huggingface_hub"
        )

    filename = "addr_labels_balanced.csv.zst" if balanced else "addr_labels_big.csv.zst"
    path = hf_hub_download(
        repo_id="fesevu/ethereum_fraud_dataset_by_activity",
        filename=filename,
        repo_type="dataset",
    )
    df = pd.read_csv(path, compression="zstd")
    df["address"] = df["address"].astype(str).str.lower()
    return df


def load_ethereum_fraud_gnn_dataset(
    months: list[str] | None = None,
    use_full: bool = False,
) -> tuple[pd.DataFrame, pd.Series]:
    """Load the pre-filtered GNN edge + label subset from the HF repo.

    This subset is curated around labeled addresses and is dramatically smaller
    than the raw 1.6B-transaction default config. Edges use hashed node IDs
    (`src_id`, `dst_id`); labels live in the same node_id space, so they join
    directly without needing the address mapping.

    The pipeline treats addresses as opaque string identifiers, so we stringify
    the node IDs and use them as `from_address` / `to_address`.

    Args:
        months: List of YYYY-MM strings (e.g. ["2023-04", "2023-05"]). If None,
            and use_full is False, loads the most recent available month.
        use_full: If True, download `gnn_dataset/edges_all/edges.parquet`
            (21.5 GB). Only use if months-based loading isn't enough.

    Returns:
        Tuple of (transactions_df, labels_series):
        - transactions_df: Standard pipeline schema with from_address, to_address,
          value (ETH), timestamp (unix seconds), tx_hash, block_number,
          contract_creation, tx_fee_wei.
        - labels_series: Binary is_scam label indexed by node_id string.
          Only labeled addresses appear.
    """
    try:
        from huggingface_hub import HfApi, hf_hub_download
    except ImportError:
        raise ImportError(
            "The `huggingface_hub` package is required. Install with: "
            "pip install huggingface_hub"
        )

    repo_id = "fesevu/ethereum_fraud_dataset_by_activity"

    # --- Load labels (small file, always needed) ---
    labels_path = hf_hub_download(
        repo_id=repo_id,
        filename="gnn_dataset/labels/targets_global.parquet",
        repo_type="dataset",
    )
    labels_df = pd.read_parquet(labels_path)
    labels_series = (
        labels_df.assign(node_id=labels_df["node_id"].astype(str))
        .set_index("node_id")["is_scam"]
        .astype(int)
    )

    # --- Load edges ---
    if use_full:
        print("  Downloading full edges (21.5 GB) ...")
        edge_paths = [
            hf_hub_download(
                repo_id=repo_id,
                filename="gnn_dataset/edges_all/edges.parquet",
                repo_type="dataset",
            )
        ]
    else:
        if months is None:
            # Discover available months and pick the latest
            api = HfApi()
            all_files = api.list_repo_files(repo_id, repo_type="dataset")
            month_files = sorted(
                f for f in all_files
                if f.startswith("gnn_dataset/edges_by_month/")
                and f.endswith("edges.parquet")
            )
            if not month_files:
                raise RuntimeError("No monthly edge files found in repo.")
            months = [month_files[-1].split("month=")[1].split("/")[0]]
            print(f"  No --months specified; using latest: {months}")

        edge_paths = [
            hf_hub_download(
                repo_id=repo_id,
                filename=f"gnn_dataset/edges_by_month/month={m}/edges.parquet",
                repo_type="dataset",
            )
            for m in months
        ]

    edges = pd.concat([pd.read_parquet(p) for p in edge_paths], ignore_index=True)

    # --- Normalize to standard pipeline schema ---
    txs = pd.DataFrame({
        "from_address": edges["src_id"].astype(str),
        "to_address": edges["dst_id"].astype(str),
        "timestamp": edges["ts"].astype("int64"),
        "value": edges["value_wei"].astype(str).astype(float) / 1e18,
        "tx_hash": edges["tx_hash"].astype(str),
        "block_number": edges["block_number"].astype("int64"),
        "contract_creation": edges["contract_creation"].astype(bool),
        "tx_fee_wei": edges["tx_fee_wei"].astype(str),
    })

    return txs, labels_series


def load_blte_dataset(
    dataset_dir: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load the BLTE (Benchmark Labeled Transactions of Ethereum) dataset.

    Contains raw labeled Ethereum transactions and pre-extracted address features.
    Source: https://github.com/AeSBL/Labeled-Transactions-based-Dataset-of-Ethereum-Network

    Args:
        dataset_dir: Path to the ethereum_dataset_labelled directory.

    Returns:
        Tuple of (transactions_df, address_features_df) where:
        - transactions_df: Raw transactions with standard columns (from_address,
          to_address, value, timestamp, tx_hash) plus scam labels (from_scam,
          to_scam, from_category, to_category).
        - address_features_df: Pre-extracted address-level features (24 columns)
          with a binary Scam label.
    """
    # Load raw transactions from Dataset.zip
    zip_path = os.path.join(dataset_dir, "Dataset.zip")
    if not os.path.exists(zip_path):
        raise FileNotFoundError(f"Dataset.zip not found in {dataset_dir}")

    with zipfile.ZipFile(zip_path, "r") as z:
        with z.open("Dataset/Dataset.csv") as f:
            txs = pd.read_csv(f)

    # Map to standard schema
    txs = txs.rename(columns={
        "hash": "tx_hash",
        "block_timestamp": "timestamp",
    })

    # Parse timestamp string to unix epoch (seconds)
    dt = pd.to_datetime(txs["timestamp"], format="mixed", utc=True)
    txs["timestamp"] = dt.values.astype("int64") // 10**6

    # Convert value from wei to ETH
    txs["value"] = pd.to_numeric(txs["value"], errors="coerce").fillna(0) / 1e18

    txs["from_address"] = txs["from_address"].astype(str)
    txs["to_address"] = txs["to_address"].astype(str)

    # Load pre-extracted address features
    features_path = os.path.join(dataset_dir, "Featuers_Extraction.csv")
    if os.path.exists(features_path):
        address_features = pd.read_csv(features_path)
        address_features = address_features.rename(columns={"Address": "address"})
        address_features = address_features.set_index("address")
    else:
        address_features = pd.DataFrame()

    return txs, address_features


def load_realcats_dataset(
    dataset_dir: str,
    blockchain: str = "ethereum",
) -> pd.DataFrame:
    """Load the Real-CATS (Cryptocurrency Addresses with Transaction profiles) dataset.

    Contains criminal and benign address features for Ethereum and Bitcoin.
    Source: https://github.com/RealCats-Lab/Real-CATS

    Args:
        dataset_dir: Path to the realcats_dataset directory.
        blockchain: "ethereum" (CE.tsv + BE.tsv, 52 features) or
                    "bitcoin" (CB.tsv + BB.tsv, 32 features).

    Returns:
        DataFrame with address features and a binary label column
        (1 = criminal, 0 = benign), indexed by address.
    """
    if blockchain == "ethereum":
        criminal_file = os.path.join(dataset_dir, "CE.tsv")
        benign_file = os.path.join(dataset_dir, "BE.tsv")
    elif blockchain == "bitcoin":
        criminal_file = os.path.join(dataset_dir, "CB.tsv")
        benign_file = os.path.join(dataset_dir, "BB.tsv")
    else:
        raise ValueError(f"Unknown blockchain: {blockchain}. Use 'ethereum' or 'bitcoin'.")

    criminal = pd.read_csv(criminal_file, sep="\t")
    benign = pd.read_csv(benign_file, sep="\t")

    # Normalize labels to binary
    criminal["label"] = 1
    benign["label"] = 0

    df = pd.concat([criminal, benign], ignore_index=True)
    df = df.set_index("address")

    return df


def load_realcats_transactions(
    data_dir: str,
    tx_types: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.Series]:
    """Load Real-CATS per-address transaction histories from the Kaggle release.

    The Kaggle dataset (https://www.kaggle.com/datasets/lvd312393/real-cats)
    provides a per-address transaction history under ETHdata/ETHdata/<address>/
    with five JSON files per address. This function defaults to
    normal_transactions.json (direct ETH transfers), which maps cleanly to the
    pipeline's standard schema.

    Use scripts/extract_realcats.py to selectively extract only the labeled
    addresses from the zip before calling this function.

    Labels are derived from the CE.tsv (criminal, label=1) and BE.tsv (benign,
    label=0) files in the realcats_dataset directory (GitHub release).

    Args:
        data_dir: Path containing ETHdata/ETHdata/ subdirectory (Kaggle) and
            CE.tsv / BE.tsv (GitHub release). Typically two separate dirs:
            pass the one containing ETHdata/ETHdata/.
        tx_types: Which transaction types to load. Options: "normal", "erc20",
            "erc721", "erc1155", "internal". Defaults to ["normal"].

    Returns:
        Tuple of (transactions_df, labels_series) where:
        - transactions_df: Standard pipeline schema (from_address, to_address,
          value, timestamp, tx_hash). Value is ETH for normal/internal (wei/1e18);
          raw token units for erc20/erc721/erc1155.
        - labels_series: Binary is_scam (1=criminal, 0=benign) indexed by
          address. Restricted to addresses present in both the transaction data
          and the CE.tsv / BE.tsv label files.
    """
    if tx_types is None:
        tx_types = ["normal"]

    # Actual filenames inside the zip / extracted dirs
    _FILENAME = {
        "normal":   "normal_transactions.json",
        "internal": "internal_transactions.json",
        "erc20":    "ERC_20_transactions.json",
        "erc721":   "ERC_721_transactions.json",
        "erc1155":  "ERC_1155_transactions.json",
    }
    for t in tx_types:
        if t not in _FILENAME:
            raise ValueError(f"Unknown tx_type {t!r}. Choose from: {list(_FILENAME)}")

    # Support both data/raw/realcats_transactions/ETHdata/ETHdata/ (Kaggle zip layout)
    # and a flat ETHdata/ layout.
    eth_data_dir = os.path.join(data_dir, "ETHdata", "ETHdata")
    if not os.path.isdir(eth_data_dir):
        eth_data_dir = os.path.join(data_dir, "ETHdata")
    if not os.path.isdir(eth_data_dir):
        raise FileNotFoundError(
            f"ETHdata/ directory not found under {data_dir}. "
            "Run scripts/extract_realcats.py first to extract labeled addresses "
            "from the Kaggle zip (https://www.kaggle.com/datasets/lvd312393/real-cats)."
        )

    # Labels live in the realcats_dataset dir (sibling of realcats_transactions)
    label_dir = data_dir
    _ce = os.path.join(label_dir, "CE.tsv")
    _be = os.path.join(label_dir, "BE.tsv")
    if not os.path.exists(_ce):
        # Try sibling directory convention
        label_dir = os.path.join(os.path.dirname(data_dir), "realcats_dataset")
        _ce = os.path.join(label_dir, "CE.tsv")
        _be = os.path.join(label_dir, "BE.tsv")
    if not os.path.exists(_ce):
        raise FileNotFoundError(
            f"CE.tsv not found in {data_dir} or its sibling realcats_dataset/. "
            "Ensure the GitHub Real-CATS release is present."
        )

    # Load labels from CE.tsv and BE.tsv
    criminal_labels = pd.read_csv(_ce, sep="\t", usecols=["address"])
    criminal_labels["is_scam"] = 1
    benign_labels = pd.read_csv(_be, sep="\t", usecols=["address"])
    benign_labels["is_scam"] = 0
    label_df = pd.concat([criminal_labels, benign_labels], ignore_index=True)
    labels_series = label_df.set_index("address")["is_scam"].astype(int)
    labeled_addresses = set(labels_series.index)

    address_dirs = [
        d for d in os.listdir(eth_data_dir)
        if os.path.isdir(os.path.join(eth_data_dir, d))
    ]
    print(f"Loading Real-CATS transactions for {len(address_dirs):,} addresses "
          f"(tx_types={tx_types})...")

    # value field name and whether to convert wei→ETH
    _VALUE_FIELD = {"normal": "value", "internal": "value",
                    "erc20": "value", "erc721": "tokenValue", "erc1155": "tokenValue"}
    _WEI_TYPES = {"normal", "internal"}

    import json
    all_rows = []
    for addr_name in address_dirs:
        addr_dir = os.path.join(eth_data_dir, addr_name)
        for tx_type in tx_types:
            json_path = os.path.join(addr_dir, _FILENAME[tx_type])
            if not os.path.exists(json_path):
                continue
            with open(json_path) as f:
                try:
                    data = json.load(f)
                except Exception:
                    continue
            records = data if isinstance(data, list) else data.get("root", [])
            if not records:
                continue
            for rec in records:
                from_addr = rec.get("from", "")
                to_addr = rec.get("to", "")
                if not from_addr or not to_addr:
                    continue
                # Skip failed transactions (normal/internal only)
                if rec.get("isError") == "1":
                    continue
                raw_value = rec.get(_VALUE_FIELD[tx_type], "0") or "0"
                try:
                    value_float = float(raw_value)
                    if tx_type in _WEI_TYPES:
                        value_float /= 1e18
                except (ValueError, TypeError):
                    value_float = 0.0
                try:
                    ts = int(rec.get("timeStamp", 0))
                except (ValueError, TypeError):
                    ts = 0
                all_rows.append({
                    "from_address": str(from_addr).lower(),
                    "to_address": str(to_addr).lower(),
                    "value": value_float,
                    "timestamp": ts,
                    "tx_hash": rec.get("hash", ""),
                    "tx_type": tx_type,
                })

    if not all_rows:
        raise RuntimeError(
            f"No transactions loaded from {eth_data_dir}. "
            "Check that the JSON files are present and non-empty."
        )

    txs = pd.DataFrame(all_rows)
    txs = txs[txs["timestamp"] > 0].reset_index(drop=True)

    # Restrict labels to addresses that actually appear in the transaction graph
    active = set(txs["from_address"]) | set(txs["to_address"])
    labels_series = labels_series.loc[labels_series.index.intersection(active & labeled_addresses)]

    print(f"  Loaded {len(txs):,} transactions, "
          f"{labels_series.sum():,} criminal / "
          f"{(labels_series == 0).sum():,} benign labeled addresses.")
    return txs, labels_series


def _apply_column_map(df: pd.DataFrame, column_map: str | dict) -> pd.DataFrame:
    """Rename columns from a source schema to the standard schema."""
    if isinstance(column_map, str):
        if column_map not in COLUMN_MAPS:
            raise ValueError(f"Unknown column map: {column_map}. Available: {list(COLUMN_MAPS.keys())}")
        mapping = COLUMN_MAPS[column_map]
    else:
        mapping = column_map

    # Invert: standard_name -> source_name becomes source_name -> standard_name
    rename_map = {source: standard for standard, source in mapping.items() if source in df.columns}
    df = df.rename(columns=rename_map)
    return df


def _validate_and_normalize(df: pd.DataFrame) -> pd.DataFrame:
    """Validate required columns exist and normalize types."""
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df["from_address"] = df["from_address"].astype(str)
    df["to_address"] = df["to_address"].astype(str)
    df["value"] = pd.to_numeric(df["value"], errors="coerce").fillna(0)
    df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce").astype(int)

    if "tx_hash" not in df.columns:
        df["tx_hash"] = [f"0x{i:064x}" for i in range(len(df))]

    return df
