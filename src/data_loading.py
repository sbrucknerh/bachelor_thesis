"""Load and validate Ethereum transaction datasets from CSV or JSON files."""

import os

import pandas as pd


REQUIRED_COLUMNS = ["from_address", "to_address", "value", "timestamp"]


def load_transactions(filepath: str) -> pd.DataFrame:
    """Load transaction data from a CSV or JSON file.

    Args:
        filepath: Path to the transaction dataset file.

    Returns:
        DataFrame with columns: from_address, to_address, value, timestamp,
        and optionally tx_hash.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If required columns are missing or file format is unsupported.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset not found: {filepath}")

    ext = os.path.splitext(filepath)[1].lower()

    if ext == ".csv":
        df = pd.read_csv(filepath)
    elif ext == ".json":
        df = pd.read_json(filepath)
    else:
        raise ValueError(f"Unsupported file format: {ext}. Use .csv or .json")

    df = _validate_and_normalize(df)
    return df


def _validate_and_normalize(df: pd.DataFrame) -> pd.DataFrame:
    """Validate required columns exist and normalize types."""
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Normalize types
    df["from_address"] = df["from_address"].astype(str)
    df["to_address"] = df["to_address"].astype(str)
    df["value"] = pd.to_numeric(df["value"], errors="coerce").fillna(0)
    df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce").astype(int)

    # Add tx_hash if missing
    if "tx_hash" not in df.columns:
        df["tx_hash"] = [f"0x{i:064x}" for i in range(len(df))]

    return df
