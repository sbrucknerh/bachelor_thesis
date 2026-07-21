"""Deterministic and sensitivity event orders for second-level timestamp ties."""

from __future__ import annotations

from contextlib import contextmanager
import hashlib
import os
from pathlib import Path
from typing import Iterator

import duckdb
import numpy as np
import pandas as pd

from src.experiments.resources import physical_memory_bytes


ORDER_VARIANTS = ("canonical", "reverse", "random_0", "random_1", "random_2", "random_3", "random_4")


def _stable_random_key(signature: str, seed: int) -> str:
    return hashlib.md5(
        f"{seed}:{signature}".encode("utf-8"), usedforsecurity=False
    ).hexdigest()


def duckdb_order_clause(variant: str) -> str:
    """Return the SQL ordering equivalent of :func:`order_events`."""
    if variant not in ORDER_VARIANTS:
        raise ValueError(f"Unknown ordering variant {variant!r}; choose from {ORDER_VARIANTS}")
    if variant == "canonical":
        tie_expression = "CAST(event_signature AS VARCHAR) ASC"
    elif variant == "reverse":
        tie_expression = "CAST(event_signature AS VARCHAR) DESC"
    else:
        seed = int(variant.split("_", maxsplit=1)[1]) + 42
        tie_expression = (
            f"md5('{seed}:' || CAST(event_signature AS VARCHAR)) ASC"
        )
    return ", ".join(
        (
            "CAST(timestamp AS BIGINT) ASC",
            "COALESCE(CAST(block_number AS BIGINT), -1) ASC",
            "(transaction_index IS NOT NULL) DESC",
            "COALESCE(CAST(transaction_index AS BIGINT), -1) ASC",
            tie_expression,
        )
    )


@contextmanager
def ordered_edge_relation(
    parquet_path: str | Path,
    variant: str,
    temporary_directory: str | Path,
) -> Iterator[duckdb.DuckDBPyRelation]:
    """Externally order a Parquet file and expose it as an Arrow stream."""
    temporary = Path(temporary_directory)
    temporary.mkdir(parents=True, exist_ok=True)
    connection = duckdb.connect()
    physical = physical_memory_bytes()
    memory_gib = 4 if physical is None else max(1, min(16, int(physical / 1024**3 * 0.15)))
    connection.execute("SET temp_directory = ?", [str(temporary)])
    connection.execute(f"SET memory_limit = '{memory_gib}GiB'")
    connection.execute(f"SET threads = {max(1, os.cpu_count() or 1)}")
    connection.execute("SET preserve_insertion_order = true")
    relation = connection.sql(
        f"""
        SELECT CAST(timestamp AS BIGINT) AS timestamp,
               CAST(from_address AS VARCHAR) AS from_address,
               CAST(to_address AS VARCHAR) AS to_address
        FROM read_parquet(?)
        ORDER BY {duckdb_order_clause(variant)}
        """,
        params=[str(Path(parquet_path))],
    )
    try:
        yield relation
    finally:
        connection.close()


def order_events(events: pd.DataFrame, variant: str = "canonical") -> pd.DataFrame:
    """Return events in a documented insertion order for Raphtory.

    Known transaction indices establish real within-block order. The final key is
    only used when the source does not expose a complete event order.
    """
    if variant not in ORDER_VARIANTS:
        raise ValueError(f"Unknown ordering variant {variant!r}; choose from {ORDER_VARIANTS}")

    ordered = events.copy()
    required = {"timestamp", "block_number", "transaction_index", "event_signature"}
    missing = required - set(ordered.columns)
    if missing:
        raise ValueError(f"Cannot order events; missing columns: {sorted(missing)}")

    ordered["_timestamp"] = pd.to_numeric(ordered["timestamp"], errors="raise").astype("int64")
    ordered["_block"] = pd.to_numeric(ordered["block_number"], errors="coerce").fillna(-1).astype("int64")
    tx_index = pd.to_numeric(ordered["transaction_index"], errors="coerce")
    ordered["_tx_known"] = tx_index.notna().astype("int8")
    ordered["_tx_index"] = tx_index.fillna(-1).astype("int64")
    signatures = ordered["event_signature"].astype(str)

    if variant == "canonical":
        ordered["_tie_key"] = signatures
        tie_ascending = True
    elif variant == "reverse":
        ordered["_tie_key"] = signatures
        tie_ascending = False
    else:
        seed = int(variant.split("_", maxsplit=1)[1]) + 42
        ordered["_tie_key"] = signatures.map(lambda value: _stable_random_key(value, seed))
        tie_ascending = True

    sort_columns = ["_timestamp", "_block", "_tx_known", "_tx_index", "_tie_key"]
    ascending = [True, True, False, True, tie_ascending]
    ordered = ordered.sort_values(sort_columns, ascending=ascending, kind="mergesort")
    return ordered.drop(columns=sort_columns[:-1] + ["_tie_key"])


def unresolved_tie_summary(events: pd.DataFrame) -> dict[str, int]:
    """Count events whose true order is unresolved by source fields."""
    tx_index = pd.to_numeric(events["transaction_index"], errors="coerce")
    unresolved = events.loc[tx_index.isna()].copy()
    if unresolved.empty:
        return {"unresolved_events": 0, "unresolved_groups": 0, "max_group_size": 0}
    sizes = unresolved.groupby(["timestamp", "block_number"], dropna=False).size()
    ambiguous = sizes[sizes > 1]
    return {
        "unresolved_events": int(ambiguous.sum()),
        "unresolved_groups": int(len(ambiguous)),
        "max_group_size": int(ambiguous.max()) if len(ambiguous) else 0,
    }
