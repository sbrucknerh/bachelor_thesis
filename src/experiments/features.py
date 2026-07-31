"""Leakage-resistant aggregate features and motif transformations."""

from __future__ import annotations

from collections.abc import Iterable
import os
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import pyarrow as pa

from src.experiments.resources import physical_memory_bytes


def _grouped_sets(frame: pd.DataFrame, key: str, value: str) -> dict[str, set[str]]:
    return {
        str(address): set(group[value].astype(str))
        for address, group in frame.groupby(key, sort=False)
    }


def aggregate_features(events: pd.DataFrame, targets: Iterable[str]) -> pd.DataFrame:
    """Compute aggregate/static features without absolute timestamp leakage."""
    target_index = pd.Index([str(target) for target in targets], name="address")
    target_set = set(target_index)
    working = events.copy()
    working["from_address"] = working["from_address"].astype(str)
    working["to_address"] = working["to_address"].astype(str)
    working["timestamp"] = pd.to_numeric(working["timestamp"], errors="raise").astype("int64")
    values = pd.to_numeric(working["value"], errors="coerce").fillna(0.0)
    if values.abs().max() > 1e12:
        values = values / 1e18
    working["_value"] = values.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    sent_events = working.loc[working["from_address"].isin(target_set)]
    received_events = working.loc[working["to_address"].isin(target_set)]
    sent = sent_events.groupby("from_address").agg(
        tx_sent=("tx_hash", "count"),
        value_sent_sum=("_value", "sum"),
        value_sent_mean=("_value", "mean"),
        value_sent_std=("_value", "std"),
        unique_out_degree=("to_address", "nunique"),
    )
    received = received_events.groupby("to_address").agg(
        tx_received=("tx_hash", "count"),
        value_received_sum=("_value", "sum"),
        value_received_mean=("_value", "mean"),
        value_received_std=("_value", "std"),
        unique_in_degree=("from_address", "nunique"),
    )
    features = sent.join(received, how="outer").reindex(target_index)

    incident_sender = sent_events[["from_address", "timestamp"]].rename(
        columns={"from_address": "address"}
    )
    incident_receiver = received_events.loc[
        received_events["from_address"] != received_events["to_address"],
        ["to_address", "timestamp"],
    ].rename(columns={"to_address": "address"})
    incident = pd.concat([incident_sender, incident_receiver], ignore_index=True)
    incident = incident.sort_values(["address", "timestamp"], kind="mergesort")
    incident["interarrival"] = incident.groupby("address")["timestamp"].diff()
    timing = incident.groupby("address").agg(
        active_span=("timestamp", lambda values_: int(values_.max() - values_.min())),
        interarrival_mean=("interarrival", "mean"),
        interarrival_median=("interarrival", "median"),
        interarrival_std=("interarrival", "std"),
    )
    features = features.join(timing, how="left")

    outgoing = _grouped_sets(sent_events, "from_address", "to_address")
    incoming = _grouped_sets(received_events, "to_address", "from_address")
    reciprocity = {}
    for address in target_index:
        out_neighbors = outgoing.get(address, set())
        in_neighbors = incoming.get(address, set())
        union = out_neighbors | in_neighbors
        reciprocity[address] = len(out_neighbors & in_neighbors) / len(union) if union else 0.0
    features["reciprocity"] = pd.Series(reciprocity)
    features["tx_total"] = features["tx_sent"].fillna(0) + features["tx_received"].fillna(0)
    features["value_total"] = (
        features["value_sent_sum"].fillna(0) + features["value_received_sum"].fillna(0)
    )
    return features.replace([np.inf, -np.inf], np.nan).fillna(0.0).astype("float64")


def aggregate_features_from_parquet(
    events_path: str | Path,
    targets: Iterable[str],
    temporary_directory: str | Path,
) -> pd.DataFrame:
    """Compute the baseline in DuckDB without materializing raw events in pandas."""
    target_values = [str(target) for target in targets]
    target_index = pd.Index(target_values, name="address")
    temporary = Path(temporary_directory)
    temporary.mkdir(parents=True, exist_ok=True)
    connection = duckdb.connect()
    physical = physical_memory_bytes()
    memory_gib = int(
        os.environ.get(
            "THESIS_DUCKDB_MEMORY_GIB",
            4 if physical is None else max(1, min(16, int(physical / 1024**3 * 0.15))),
        )
    )
    connection.execute("SET temp_directory = ?", [str(temporary)])
    connection.execute(f"SET memory_limit = '{memory_gib}GiB'")
    connection.execute(
        f"SET threads = {max(1, int(os.environ.get('THESIS_MODEL_THREADS', '1')))}"
    )
    connection.register("experiment_targets", pa.table({"address": target_values}))
    query = """
        WITH events AS (
            SELECT CAST(from_address AS VARCHAR) AS from_address,
                   CAST(to_address AS VARCHAR) AS to_address,
                   CAST(timestamp AS BIGINT) AS timestamp,
                   COALESCE(CAST(value AS DOUBLE), 0.0) AS value,
                   tx_hash
            FROM read_parquet(?)
        ),
        sent AS (
            SELECT e.from_address AS address,
                   COUNT(e.tx_hash) AS tx_sent,
                   SUM(e.value) AS value_sent_sum,
                   AVG(e.value) AS value_sent_mean,
                   STDDEV_SAMP(e.value) AS value_sent_std,
                   COUNT(DISTINCT e.to_address) AS unique_out_degree
            FROM events e
            INNER JOIN experiment_targets t ON e.from_address = t.address
            GROUP BY e.from_address
        ),
        received AS (
            SELECT e.to_address AS address,
                   COUNT(e.tx_hash) AS tx_received,
                   SUM(e.value) AS value_received_sum,
                   AVG(e.value) AS value_received_mean,
                   STDDEV_SAMP(e.value) AS value_received_std,
                   COUNT(DISTINCT e.from_address) AS unique_in_degree
            FROM events e
            INNER JOIN experiment_targets t ON e.to_address = t.address
            GROUP BY e.to_address
        ),
        incident AS (
            SELECT e.from_address AS address, e.timestamp
            FROM events e
            INNER JOIN experiment_targets t ON e.from_address = t.address
            UNION ALL
            SELECT e.to_address AS address, e.timestamp
            FROM events e
            INNER JOIN experiment_targets t ON e.to_address = t.address
            WHERE e.from_address <> e.to_address
        ),
        incident_gaps AS (
            SELECT address, timestamp,
                   timestamp - LAG(timestamp) OVER (
                       PARTITION BY address ORDER BY timestamp
                   ) AS interarrival
            FROM incident
        ),
        timing AS (
            SELECT address,
                   MAX(timestamp) - MIN(timestamp) AS active_span,
                   AVG(interarrival) AS interarrival_mean,
                   MEDIAN(interarrival) AS interarrival_median,
                   STDDEV_SAMP(interarrival) AS interarrival_std
            FROM incident_gaps
            GROUP BY address
        ),
        neighbour_rows AS (
            SELECT e.from_address AS address, e.to_address AS neighbour,
                   1 AS outgoing, 0 AS incoming
            FROM events e
            INNER JOIN experiment_targets t ON e.from_address = t.address
            UNION ALL
            SELECT e.to_address AS address, e.from_address AS neighbour,
                   0 AS outgoing, 1 AS incoming
            FROM events e
            INNER JOIN experiment_targets t ON e.to_address = t.address
        ),
        neighbours AS (
            SELECT address, neighbour, MAX(outgoing) AS outgoing, MAX(incoming) AS incoming
            FROM neighbour_rows
            GROUP BY address, neighbour
        ),
        reciprocal AS (
            SELECT address,
                   SUM(CASE WHEN outgoing = 1 AND incoming = 1 THEN 1 ELSE 0 END)
                   / CAST(COUNT(*) AS DOUBLE) AS reciprocity
            FROM neighbours
            GROUP BY address
        )
        SELECT t.address,
               COALESCE(s.tx_sent, 0) AS tx_sent,
               COALESCE(s.value_sent_sum, 0.0) AS value_sent_sum,
               COALESCE(s.value_sent_mean, 0.0) AS value_sent_mean,
               COALESCE(s.value_sent_std, 0.0) AS value_sent_std,
               COALESCE(s.unique_out_degree, 0) AS unique_out_degree,
               COALESCE(r.tx_received, 0) AS tx_received,
               COALESCE(r.value_received_sum, 0.0) AS value_received_sum,
               COALESCE(r.value_received_mean, 0.0) AS value_received_mean,
               COALESCE(r.value_received_std, 0.0) AS value_received_std,
               COALESCE(r.unique_in_degree, 0) AS unique_in_degree,
               COALESCE(tm.active_span, 0) AS active_span,
               COALESCE(tm.interarrival_mean, 0.0) AS interarrival_mean,
               COALESCE(tm.interarrival_median, 0.0) AS interarrival_median,
               COALESCE(tm.interarrival_std, 0.0) AS interarrival_std,
               COALESCE(rc.reciprocity, 0.0) AS reciprocity,
               COALESCE(s.tx_sent, 0) + COALESCE(r.tx_received, 0) AS tx_total,
               COALESCE(s.value_sent_sum, 0.0)
                   + COALESCE(r.value_received_sum, 0.0) AS value_total
        FROM experiment_targets t
        LEFT JOIN sent s USING (address)
        LEFT JOIN received r USING (address)
        LEFT JOIN timing tm USING (address)
        LEFT JOIN reciprocal rc USING (address)
        ORDER BY t.address
    """
    try:
        frame = connection.sql(query, params=[str(Path(events_path))]).df()
    finally:
        connection.close()
    frame = frame.set_index("address").reindex(target_index)
    return frame.replace([np.inf, -np.inf], np.nan).fillna(0.0).astype("float64")


def log_motif_counts(counts: pd.DataFrame) -> pd.DataFrame:
    if (counts < 0).any().any():
        raise ValueError("Motif counts cannot be negative")
    return np.log1p(counts.astype("float64"))


def hellinger_motif_composition(counts: pd.DataFrame) -> pd.DataFrame:
    """Hellinger-transform motif proportions independently within each layer."""
    if (counts < 0).any().any():
        raise ValueError("Motif counts cannot be negative")
    transformed: list[pd.DataFrame] = []
    layer_names = sorted({column.split("__", 1)[0] for column in counts.columns})
    if not layer_names:
        totals = counts.sum(axis=1).replace(0, np.nan)
        return np.sqrt(counts.div(totals, axis=0).fillna(0.0))
    for layer in layer_names:
        columns = [column for column in counts.columns if column.startswith(f"{layer}__")]
        layer_counts = counts.loc[:, columns].astype("float64")
        totals = layer_counts.sum(axis=1).replace(0, np.nan)
        transformed.append(np.sqrt(layer_counts.div(totals, axis=0).fillna(0.0)))
    return pd.concat(transformed, axis=1)


def motif_coverage(counts: pd.DataFrame, labels: pd.Series | None = None) -> pd.DataFrame:
    nonzero = counts.sum(axis=1) > 0
    if labels is None:
        return pd.DataFrame(
            {"n": [len(counts)], "nonzero": [int(nonzero.sum())], "coverage": [float(nonzero.mean())]}
        )
    frame = pd.DataFrame({"label": labels.reindex(counts.index), "nonzero": nonzero})
    summary = frame.groupby("label", dropna=False)["nonzero"].agg(["size", "sum", "mean"])
    return summary.rename(columns={"size": "n", "sum": "nonzero", "mean": "coverage"})
