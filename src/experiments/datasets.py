"""Streaming dataset preparation for the Results experiments."""

from __future__ import annotations

from collections import defaultdict
from contextlib import nullcontext
import math
import os
from pathlib import Path
import sys
from typing import Iterable
import zipfile

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from src.experiments.constants import (
    ACTIVITY_MATCHING_BINS,
    HF_COHORT_ELIGIBILITY,
    KNOWN_CAMPAIGN_PREFIXES,
    RANDOM_SEED,
)
from src.experiments.telemetry import RunTelemetry


HF_REPO_CACHE = "datasets--fesevu--ethereum_fraud_dataset_by_activity"
HF_DATASET_REVISION = "6a7a517fbdbe2bcc4ec65bee4c0ff15f1c0a3c70"
HF_PROJECT_SNAPSHOT = (
    Path(__file__).resolve().parents[2]
    / "data/raw/huggingface_ethereum_fraud"
)


def _is_hf_snapshot(candidate: Path) -> bool:
    return (
        (candidate / "gnn_dataset/edges_all/edges.parquet").exists()
        and (candidate / "gnn_dataset/labels/targets_global.parquet").exists()
        and (candidate / "addr_labels_balanced.csv.zst").exists()
    )


def discover_hf_snapshot(
    cache_root: str | Path | None = None,
    project_snapshot: str | Path | None = None,
) -> Path:
    """Return the pinned local Hugging Face snapshot containing full GNN data."""
    local = Path(project_snapshot) if project_snapshot else HF_PROJECT_SNAPSHOT
    if _is_hf_snapshot(local):
        return local
    root = Path(cache_root) if cache_root else Path.home() / ".cache" / "huggingface" / "hub"
    snapshots = root / HF_REPO_CACHE / "snapshots"
    pinned = snapshots / HF_DATASET_REVISION
    if _is_hf_snapshot(pinned):
        return pinned
    candidates = sorted(snapshots.glob("*")) if snapshots.exists() else []
    for candidate in reversed(candidates):
        if _is_hf_snapshot(candidate):
            return candidate
    raise FileNotFoundError(
        "Full Hugging Face GNN snapshot is missing. Run scripts/setup_results.sh "
        "or provide --hf-snapshot with edges_all, targets_global, and "
        "addr_labels_balanced."
    )


def campaign_group(description: object, address: str) -> str:
    """Map only audited event families to shared groups; everything else is singleton."""
    normalized = str(description or "").strip().lower()
    for family, prefix in KNOWN_CAMPAIGN_PREFIXES.items():
        if normalized.startswith(prefix):
            return f"campaign:{family}"
    return f"address:{address.lower()}"


def count_target_activity(
    edges_path: str | Path,
    target_ids: Iterable[int],
    batch_size: int = 262_144,
    telemetry: RunTelemetry | None = None,
    max_rows: int | None = None,
) -> pd.Series:
    """Count incident edges for target IDs without loading the edge table."""
    if max_rows is not None and max_rows <= 0:
        raise ValueError("max_rows must be positive")
    target_array = np.fromiter((int(value) for value in target_ids), dtype=np.uint64)
    value_set = pa.array(target_array, type=pa.uint64())
    counts: defaultdict[int, int] = defaultdict(int)
    parquet = pq.ParquetFile(edges_path)
    source_rows = int(parquet.metadata.num_rows)
    rows_to_scan = min(source_rows, max_rows) if max_rows is not None else source_rows
    total_batches = max(1, math.ceil(rows_to_scan / batch_size))
    rows_scanned = 0

    progress_context = (
        telemetry.task_progress("HF activity scan", total_batches, unit="batch")
        if telemetry
        else nullcontext(lambda _count=1: None)
    )
    with progress_context as advance:
        for batch_number, batch in enumerate(
            parquet.iter_batches(
                batch_size=batch_size,
                columns=["src_id", "dst_id"],
                use_threads=True,
            ),
            start=1,
        ):
            remaining = rows_to_scan - rows_scanned
            if remaining <= 0:
                break
            if len(batch) > remaining:
                batch = batch.slice(0, remaining)
            src = batch.column(0)
            dst = batch.column(1)
            src_mask = pc.is_in(src, value_set=value_set)
            dst_mask = pc.and_(
                pc.is_in(dst, value_set=value_set),
                pc.not_equal(src, dst),
            )
            for selected in (pc.filter(src, src_mask), pc.filter(dst, dst_mask)):
                values = selected.to_numpy(zero_copy_only=False)
                if not len(values):
                    continue
                unique, frequencies = np.unique(values, return_counts=True)
                for node_id, frequency in zip(unique, frequencies, strict=True):
                    counts[int(node_id)] += int(frequency)
            rows_scanned += len(batch)
            advance()
            if telemetry and (
                batch_number % 100 == 0 or batch_number == total_batches
            ):
                telemetry.progress(
                    "prepare.hf.activity_scan",
                    batches_complete=batch_number,
                    batches_total=total_batches,
                    rows_scanned=rows_scanned,
                    rows_total=rows_to_scan,
                    source_rows=source_rows,
                )

    result = pd.Series(counts, dtype="int64", name="activity")
    result.index.name = "node_id"
    return result


def load_hf_targets(snapshot: str | Path, activity: pd.Series) -> pd.DataFrame:
    """Join GNN node IDs to labels, descriptions, activity, and split groups."""
    snapshot_path = Path(snapshot)
    labels = pd.read_parquet(snapshot_path / "gnn_dataset/labels/targets_global.parquet")
    descriptions = pd.read_csv(
        snapshot_path / "addr_labels_balanced.csv.zst",
        compression="zstd",
        usecols=["address", "description"],
    )
    labels["ethereum_address"] = labels["address"].astype(str).str.lower()
    descriptions["ethereum_address"] = descriptions["address"].astype(str).str.lower()
    descriptions = descriptions.drop(columns="address").drop_duplicates("ethereum_address")
    labels = labels.merge(descriptions, on="ethereum_address", how="left", validate="one_to_one")
    labels["activity"] = labels["node_id"].map(activity).fillna(0).astype("int64")
    labels["address"] = labels["node_id"].astype(str)
    labels["label"] = labels["is_scam"].astype("int8")
    labels["is_contract"] = labels["is_contract"].astype("int8")
    labels["campaign_group"] = [
        campaign_group(description, address)
        for description, address in zip(
            labels["description"], labels["ethereum_address"], strict=True
        )
    ]

    eligible = labels["activity"] >= 3
    labels["activity_quintile"] = pd.Series(pd.NA, index=labels.index, dtype="Int64")
    for _, indices in labels.loc[eligible].groupby(["label", "is_contract"]).groups.items():
        ranked = labels.loc[indices, "activity"].rank(method="first")
        bins = pd.qcut(ranked, q=min(5, len(ranked)), labels=False, duplicates="drop")
        labels.loc[indices, "activity_quintile"] = bins.astype("int64")
    labels["stratum"] = (
        labels["label"].astype(str)
        + ":"
        + labels["is_contract"].astype(str)
        + ":"
        + labels["activity_quintile"].astype(str)
    )
    return labels


def select_dual_hf_cohorts(
    targets: pd.DataFrame,
    prevalence_size: int = 5_000,
    controlled_per_stratum: int = 250,
    seed: int = RANDOM_SEED,
    minimum_activity: int = HF_COHORT_ELIGIBILITY["minimum_incident_edges"],
    activity_upper_quantile: float = HF_COHORT_ELIGIBILITY["activity_upper_quantile"],
    excluded_node_ids: Iterable[int] = (),
    excluded_campaign_groups: Iterable[str] = (),
) -> tuple[pd.DataFrame, dict]:
    """Select cohorts after eligibility and optional prior-run exclusion rules."""
    if minimum_activity < 3:
        raise ValueError("Three-edge motifs require minimum_activity >= 3")
    if not 0 < activity_upper_quantile < 1:
        raise ValueError("activity_upper_quantile must be between zero and one")
    activity_eligible = targets.loc[targets["activity"] >= minimum_activity].copy()
    stratum_columns = ["label", "is_contract", "activity_quintile"]
    activity_eligible["activity_cutoff"] = activity_eligible.groupby(
        stratum_columns, dropna=False
    )["activity"].transform(lambda values: values.quantile(activity_upper_quantile))
    high_activity = activity_eligible["activity"] > activity_eligible["activity_cutoff"]
    eligible = activity_eligible.loc[~high_activity].drop(columns="activity_cutoff").copy()
    excluded_nodes = {int(value) for value in excluded_node_ids}
    excluded_groups = {str(value) for value in excluded_campaign_groups}
    node_exclusion = eligible["node_id"].isin(excluded_nodes)
    if excluded_groups and "campaign_group" not in eligible:
        raise ValueError("campaign_group is required for prior-run group exclusions")
    campaign_exclusion = (
        eligible["campaign_group"].astype(str).isin(excluded_groups)
        if "campaign_group" in eligible
        else pd.Series(False, index=eligible.index)
    )
    excluded = node_exclusion | campaign_exclusion
    available = eligible.loc[~excluded].copy()
    prevalence_n = min(prevalence_size, len(available))
    prevalence = available.sample(
        n=prevalence_n, random_state=seed, replace=False
    ).copy()
    prevalence["cohort"] = "prevalence"

    remaining = available.loc[
        ~available["node_id"].isin(prevalence["node_id"])
    ].copy()
    controlled_parts: list[pd.DataFrame] = []
    for _, group in remaining.groupby(["label", "is_contract", "activity_quintile"]):
        take = min(controlled_per_stratum, len(group))
        controlled_parts.append(group.sample(n=take, random_state=seed, replace=False))
    controlled = pd.concat(controlled_parts, ignore_index=True) if controlled_parts else remaining.head(0)
    controlled["cohort"] = "controlled"

    selected = pd.concat([prevalence, controlled], ignore_index=True)
    if selected["node_id"].duplicated().any():
        raise RuntimeError("Dual cohorts are not disjoint")
    selected = selected.sort_values(["cohort", "node_id"]).reset_index(drop=True)

    attrition = {
        "labelled_targets": int(len(targets)),
        "inactive_targets": int((targets["activity"] == 0).sum()),
        "insufficient_activity_targets": int(
            ((targets["activity"] > 0) & (targets["activity"] < minimum_activity)).sum()
        ),
        "minimum_incident_edges": int(minimum_activity),
        "activity_eligible_before_tail_exclusion": int(len(activity_eligible)),
        "high_activity_targets_excluded": int(high_activity.sum()),
        "activity_upper_quantile": float(activity_upper_quantile),
        "eligible_active_targets": int(len(eligible)),
        "prior_run_node_exclusions": int(node_exclusion.sum()),
        "prior_run_campaign_exclusions": int((campaign_exclusion & ~node_exclusion).sum()),
        "prior_run_exclusions_total": int(excluded.sum()),
        "available_after_prior_run_exclusions": int(len(available)),
        "prevalence_targets": int(len(prevalence)),
        "controlled_targets": int(len(controlled)),
        "high_activity_exclusions_by_stratum": {
            str(key): int(value)
            for key, value in activity_eligible.loc[high_activity]
            .groupby(stratum_columns, dropna=False)
            .size()
            .items()
        },
        "controlled_strata": {
            str(key): int(value)
            for key, value in controlled.groupby(
                ["label", "is_contract", "activity_quintile"], dropna=False
            ).size().items()
        },
    }
    return selected, attrition


def _hf_output_table(batch: pa.RecordBatch, mask: pa.Array) -> pa.Table:
    filtered = pa.Table.from_batches([batch]).filter(mask)
    src = filtered["src_id"]
    dst = filtered["dst_id"]
    hashes = filtered["tx_hash"]
    signatures = pc.binary_join_element_wise(
        pc.cast(hashes, pa.string()),
        pc.cast(src, pa.string()),
        pc.cast(dst, pa.string()),
        "|",
    )
    count = len(filtered)
    return pa.table(
        {
            "from_address": pc.cast(src, pa.string()),
            "to_address": pc.cast(dst, pa.string()),
            "timestamp": filtered["ts"],
            "value": pc.divide(
                pc.cast(filtered["value_wei"], pa.float64()), pa.scalar(1e18)
            ),
            "tx_hash": pc.cast(hashes, pa.string()),
            "block_number": filtered["block_number"],
            "transaction_index": pa.nulls(count, type=pa.int64()),
            "event_signature": signatures,
            "layer": pa.array(["normal"] * count, type=pa.string()),
        }
    )


def extract_target_edges(
    edges_path: str | Path,
    target_ids: Iterable[int],
    output_path: str | Path,
    batch_size: int = 262_144,
    telemetry: RunTelemetry | None = None,
    max_rows: int | None = None,
) -> dict[str, int]:
    """Write exactly the union of edges incident to selected targets."""
    if max_rows is not None and max_rows <= 0:
        raise ValueError("max_rows must be positive")
    output = Path(output_path)
    if output.exists():
        raise FileExistsError(output)
    target_array = np.fromiter((int(value) for value in target_ids), dtype=np.uint64)
    value_set = pa.array(target_array, type=pa.uint64())
    parquet = pq.ParquetFile(edges_path)
    columns = ["src_id", "dst_id", "ts", "value_wei", "block_number", "tx_hash"]
    writer: pq.ParquetWriter | None = None
    selected_rows = 0
    source_rows = int(parquet.metadata.num_rows)
    rows_to_scan = min(source_rows, max_rows) if max_rows is not None else source_rows
    total_batches = max(1, math.ceil(rows_to_scan / batch_size))
    rows_scanned = 0
    progress_context = (
        telemetry.task_progress("HF incident-edge extraction", total_batches, unit="batch")
        if telemetry
        else nullcontext(lambda _count=1: None)
    )
    try:
        with progress_context as advance:
            for batch_number, batch in enumerate(
                parquet.iter_batches(
                    batch_size=batch_size,
                    columns=columns,
                    use_threads=True,
                ),
                start=1,
            ):
                remaining = rows_to_scan - rows_scanned
                if remaining <= 0:
                    break
                if len(batch) > remaining:
                    batch = batch.slice(0, remaining)
                src = batch.column(0)
                dst = batch.column(1)
                mask = pc.or_(
                    pc.is_in(src, value_set=value_set),
                    pc.is_in(dst, value_set=value_set),
                )
                if pc.sum(pc.cast(mask, pa.int64())).as_py() != 0:
                    table = _hf_output_table(batch, mask)
                    if writer is None:
                        writer = pq.ParquetWriter(
                            output,
                            table.schema,
                            compression="zstd",
                        )
                    writer.write_table(table)
                    selected_rows += len(table)
                rows_scanned += len(batch)
                advance()
                if telemetry and (
                    batch_number % 100 == 0 or batch_number == total_batches
                ):
                    telemetry.progress(
                        "prepare.hf.edge_extraction",
                        batches_complete=batch_number,
                        batches_total=total_batches,
                        rows_scanned=rows_scanned,
                        rows_total=rows_to_scan,
                        source_rows=source_rows,
                        selected_rows=selected_rows,
                    )
    finally:
        if writer is not None:
            writer.close()
    if writer is None:
        raise RuntimeError("No edges were incident to the selected targets")
    audit = {"source_edges": source_rows, "selected_edges": selected_rows}
    if max_rows is not None:
        audit["source_edges_scanned"] = rows_scanned
    return audit


def extract_released_neighborhood_edges(
    edges_path: str | Path,
    incident_edges_path: str | Path,
    output_path: str | Path,
    batch_size: int = 262_144,
    telemetry: RunTelemetry | None = None,
    max_rows: int | None = None,
) -> dict[str, int | str | bool]:
    """Materialize the released two-hop closure needed for target triangles.

    For every selected target, both non-target vertices of any triangle must
    already occur in its incident-edge endpoint set. Retaining all released
    edges within the union of those endpoints therefore preserves every
    target-centred triangle present in the released graph. It cannot restore
    edges removed by the upstream dataset construction.
    """
    if max_rows is not None and max_rows <= 0:
        raise ValueError("max_rows must be positive")
    incident_path = Path(incident_edges_path)
    output = Path(output_path)
    if output.exists():
        raise FileExistsError(output)
    if not incident_path.is_file():
        raise FileNotFoundError(incident_path)

    incident_parquet = pq.ParquetFile(incident_path)
    parquet = pq.ParquetFile(edges_path)
    source_rows = int(parquet.metadata.num_rows)
    rows_to_scan = min(source_rows, max_rows) if max_rows is not None else source_rows
    incident_rows = int(incident_parquet.metadata.num_rows)

    import duckdb

    from src.experiments.resources import GIB, duckdb_memory_bytes, physical_memory_bytes

    scratch = output.parent / "duckdb_tmp" / "prepare_hf_neighborhood"
    scratch.mkdir(parents=True, exist_ok=True)
    connection = duckdb.connect()
    physical = physical_memory_bytes() or 64 * GIB
    memory_gib = max(1, duckdb_memory_bytes(physical) // GIB)
    threads = max(1, int(os.environ.get("THESIS_MODEL_THREADS", "1")))
    connection.execute("SET temp_directory = ?", [str(scratch)])
    connection.execute(f"SET memory_limit = '{memory_gib}GiB'")
    connection.execute(f"SET threads = {threads}")
    if sys.stderr.isatty() and os.environ.get("THESIS_DISABLE_TQDM", "0") != "1":
        connection.execute("PRAGMA enable_progress_bar")
        connection.execute("SET progress_bar_time = 2000")

    source_literal = "'" + str(Path(edges_path)).replace("'", "''") + "'"
    output_literal = "'" + str(output).replace("'", "''") + "'"
    source_relation = f"read_parquet({source_literal})"
    if max_rows is not None:
        source_relation = f"(SELECT * FROM {source_relation} LIMIT {rows_to_scan})"
    try:
        if telemetry:
            telemetry.progress(
                "prepare.hf.neighborhood_extraction",
                phase="build_endpoint_relation",
                incident_edges=incident_rows,
                threads=threads,
                duckdb_memory_gib=memory_gib,
            )
        connection.execute(
            """
            CREATE TEMP TABLE neighborhood_nodes AS
            SELECT CAST(from_address AS UBIGINT) AS node_id
            FROM read_parquet(?)
            UNION
            SELECT CAST(to_address AS UBIGINT) AS node_id
            FROM read_parquet(?)
            """,
            [str(incident_path), str(incident_path)],
        )
        endpoint_count = int(
            connection.execute(
                "SELECT COUNT(*) FROM neighborhood_nodes"
            ).fetchone()[0]
        )
        if endpoint_count <= 0:
            raise RuntimeError("The target-incident graph contains no endpoints")
        if telemetry:
            telemetry.progress(
                "prepare.hf.neighborhood_extraction",
                phase="scan_released_graph",
                rows_total=rows_to_scan,
                source_rows=source_rows,
                neighborhood_nodes=endpoint_count,
            )
        connection.execute(
            f"""
            COPY (
                SELECT CAST(e.src_id AS VARCHAR) AS from_address,
                       CAST(e.dst_id AS VARCHAR) AS to_address,
                       CAST(e.ts AS BIGINT) AS timestamp,
                       CAST(e.value_wei AS DOUBLE) / 1e18 AS value,
                       CAST(e.tx_hash AS VARCHAR) AS tx_hash,
                       CAST(e.block_number AS BIGINT) AS block_number,
                       CAST(NULL AS BIGINT) AS transaction_index,
                       CONCAT(
                           CAST(e.tx_hash AS VARCHAR), '|',
                           CAST(e.src_id AS VARCHAR), '|',
                           CAST(e.dst_id AS VARCHAR)
                       ) AS event_signature,
                       'normal' AS layer
                FROM {source_relation} e
                INNER JOIN neighborhood_nodes src_nodes
                    ON e.src_id = src_nodes.node_id
                INNER JOIN neighborhood_nodes dst_nodes
                    ON e.dst_id = dst_nodes.node_id
            ) TO {output_literal} (
                FORMAT PARQUET,
                COMPRESSION ZSTD,
                ROW_GROUP_SIZE {max(2_048, int(batch_size))}
            )
            """
        )
    finally:
        connection.close()
    if not output.is_file():
        raise RuntimeError("No released edges were retained for the target neighborhoods")

    selected_rows = int(pq.ParquetFile(output).metadata.num_rows)
    if selected_rows < incident_rows:
        raise RuntimeError(
            "Released-neighborhood extraction lost a target-incident edge"
        )
    audit: dict[str, int | str | bool] = {
        "source_edges": source_rows,
        "incident_edges": incident_rows,
        "neighborhood_nodes": endpoint_count,
        "selected_edges": selected_rows,
        "additional_neighborhood_edges": selected_rows - incident_rows,
        "scope": "released_graph_target_neighborhood_closure",
        "release_relative_completeness": True,
        "full_ethereum_completeness": False,
    }
    if max_rows is not None:
        audit["source_edges_scanned"] = rows_to_scan
    return audit


def load_blte_for_experiment(dataset_dir: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load BLTE events and endpoint-derived labels without precomputed feature leakage."""
    directory = Path(dataset_dir)
    with zipfile.ZipFile(directory / "Dataset.zip") as archive:
        with archive.open("Dataset/Dataset.csv") as handle:
            raw = pd.read_csv(handle)
    timestamp = (
        pd.to_datetime(raw["block_timestamp"], format="mixed", utc=True).astype("int64")
        // 10**9
    )
    events = pd.DataFrame(
        {
            "from_address": raw["from_address"].astype(str).str.lower(),
            "to_address": raw["to_address"].astype(str).str.lower(),
            "timestamp": timestamp.astype("int64"),
            "value": pd.to_numeric(raw["value"], errors="coerce").fillna(0) / 1e18,
            "tx_hash": raw["hash"].astype(str),
            "block_number": pd.to_numeric(raw["block_number"], errors="coerce").fillna(-1).astype("int64"),
            "transaction_index": pd.to_numeric(raw["transaction_index"], errors="coerce"),
            "event_signature": raw["hash"].astype(str),
            "layer": "normal",
        }
    )
    labels = pd.read_csv(
        directory / "Featuers_Extraction.csv", usecols=["Address", "Scam"]
    ).rename(columns={"Address": "address", "Scam": "label"})
    labels["address"] = labels["address"].astype(str).str.lower()
    active = set(events["from_address"]) | set(events["to_address"])
    labels = labels.loc[labels["address"].isin(active)].drop_duplicates("address")
    labels["campaign_group"] = "address:" + labels["address"]
    return events, labels.reset_index(drop=True)


def select_activity_matched_blte_targets(
    events: pd.DataFrame,
    targets: pd.DataFrame,
    *,
    seed: int = RANDOM_SEED,
    minimum_activity: int = 3,
) -> tuple[pd.DataFrame, dict]:
    """Balance BLTE classes within incident-event activity bins."""
    sent = events.groupby("from_address").size()
    received = events.loc[
        events["from_address"] != events["to_address"]
    ].groupby("to_address").size()
    activity = sent.add(received, fill_value=0).astype("int64")
    working = targets.copy()
    working["incident_events"] = (
        working["address"].map(activity).fillna(0).astype("int64")
    )
    source_counts = {
        str(key): int(value) for key, value in working["label"].value_counts().items()
    }
    working = working.loc[working["incident_events"] >= int(minimum_activity)].copy()
    boundaries = [*ACTIVITY_MATCHING_BINS, np.inf]
    bin_labels = [
        f"[{int(lower)},{'inf' if np.isinf(upper) else int(upper)})"
        for lower, upper in zip(boundaries[:-1], boundaries[1:], strict=True)
    ]
    working["activity_bin"] = pd.cut(
        working["incident_events"],
        bins=boundaries,
        labels=bin_labels,
        right=False,
        include_lowest=True,
    )
    rng = np.random.default_rng(int(seed))
    selected: list[pd.DataFrame] = []
    bin_audit: dict[str, dict[str, int]] = {}
    for activity_bin, scoped in working.groupby(
        "activity_bin", observed=True, sort=True
    ):
        negative = scoped.loc[scoped["label"].astype(int) == 0].sort_values(
            "address"
        )
        positive = scoped.loc[scoped["label"].astype(int) == 1].sort_values(
            "address"
        )
        matched = min(len(negative), len(positive))
        if matched:
            negative_indices = rng.choice(len(negative), matched, replace=False)
            positive_indices = rng.choice(len(positive), matched, replace=False)
            selected.extend(
                [
                    negative.iloc[np.sort(negative_indices)],
                    positive.iloc[np.sort(positive_indices)],
                ]
            )
        bin_audit[str(activity_bin)] = {
            "eligible_benign": int(len(negative)),
            "eligible_scam": int(len(positive)),
            "selected_per_class": int(matched),
        }
    if not selected:
        raise ValueError("No activity-matched BLTE cohort can be formed")
    result = pd.concat(selected, ignore_index=True).sort_values("address").reset_index(
        drop=True
    )
    result["label"] = result["label"].astype(int)
    result["activity_bin"] = result["activity_bin"].astype(str)
    counts = result["label"].value_counts()
    if len(counts) != 2 or counts.nunique() != 1:
        raise RuntimeError("BLTE activity matching did not produce balanced classes")
    return result, {
        "source_label_counts": source_counts,
        "minimum_incident_events": int(minimum_activity),
        "activity_used_as_model_feature": False,
        "bin_counts": bin_audit,
        "selected_label_counts": {
            str(key): int(value) for key, value in counts.items()
        },
    }
