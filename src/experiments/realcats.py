"""Auditable, layer-preserving conversion of Real-CATS ego histories."""

from __future__ import annotations

from collections import Counter, defaultdict
from contextlib import nullcontext
import hashlib
import json
from pathlib import Path
import shutil
import sqlite3
from typing import Any, Iterable
import zipfile

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from src.experiments.constants import (
    ACTIVITY_MATCHING_BINS,
    REALCATS_CATEGORY_ELIGIBILITY,
    REALCATS_LAYER_FILES,
)
from src.experiments.telemetry import RunTelemetry


_FILE_TO_LAYER = {filename: layer for layer, filename in REALCATS_LAYER_FILES.items()}


def ensure_free_space(path: str | Path, minimum_gib: int = 25) -> None:
    available = shutil.disk_usage(Path(path).resolve()).free
    minimum = minimum_gib * 1024**3
    if available < minimum:
        raise RuntimeError(
            f"Real-CATS conversion requires at least {minimum_gib} GiB free; "
            f"only {available / 1024**3:.1f} GiB is available"
        )


def load_criminal_labels(label_dir: str | Path) -> pd.DataFrame:
    labels = pd.read_csv(Path(label_dir) / "CE.tsv", sep="\t", usecols=["address", "label"])
    labels["address"] = labels["address"].astype(str).str.lower()
    labels["type_label"] = labels["label"].astype(str).str.strip()
    conflicts = labels.groupby("address")["type_label"].nunique()
    conflicts = conflicts[conflicts > 1]
    if not conflicts.empty:
        examples = sorted(conflicts.index.astype(str))[:10]
        raise ValueError(
            "CE.tsv contains addresses with conflicting criminal labels: "
            f"{examples}"
        )
    return labels.drop(columns="label").drop_duplicates(["address", "type_label"])


def load_binary_labels(label_dir: str | Path) -> pd.DataFrame:
    """Load Real-CATS criminal/benign labels and matching-only activity."""
    directory = Path(label_dir)
    frames: list[pd.DataFrame] = []
    for filename, label in (("CE.tsv", 1), ("BE.tsv", 0)):
        frame = pd.read_csv(
            directory / filename,
            sep="\t",
            usecols=["address", "transaction_number"],
        )
        frame["address"] = frame["address"].astype(str).str.lower()
        frame["source_activity"] = pd.to_numeric(
            frame.pop("transaction_number"), errors="raise"
        ).astype("int64")
        frame["label"] = int(label)
        if frame["address"].duplicated().any():
            raise ValueError(f"{filename} contains duplicate addresses")
        frames.append(frame)
    labels = pd.concat(frames, ignore_index=True)
    overlap = labels.groupby("address")["label"].nunique()
    conflicts = overlap[overlap > 1]
    if not conflicts.empty:
        raise ValueError(
            "Real-CATS CE.tsv and BE.tsv contain conflicting addresses: "
            f"{sorted(conflicts.index.astype(str))[:10]}"
        )
    return labels


def realcats_archive_addresses(zip_path: str | Path) -> set[str]:
    """Return addresses with at least one recognized history file in the archive."""
    observed: set[str] = set()
    with zipfile.ZipFile(zip_path) as archive:
        for name in archive.namelist():
            parts = name.split("/")
            if len(parts) == 4 and parts[3] in _FILE_TO_LAYER:
                observed.add(parts[2].lower())
    return observed


def _activity_bin(activity: pd.Series) -> pd.Series:
    boundaries = [*ACTIVITY_MATCHING_BINS, np.inf]
    labels = [
        f"[{int(lower)},{'inf' if np.isinf(upper) else int(upper)})"
        for lower, upper in zip(boundaries[:-1], boundaries[1:], strict=True)
    ]
    return pd.cut(
        activity,
        bins=boundaries,
        labels=labels,
        right=False,
        include_lowest=True,
    )


def select_activity_matched_binary_targets(
    labels: pd.DataFrame,
    archive_addresses: Iterable[str],
    excluded_addresses: Iterable[str],
    *,
    seed: int,
    minimum_activity: int = 3,
) -> tuple[pd.DataFrame, dict]:
    """Select an HF-disjoint 1:1 criminal/benign cohort within activity bins."""
    archive = {str(value).lower() for value in archive_addresses}
    excluded = {str(value).lower() for value in excluded_addresses}
    working = labels.copy()
    source_counts = {
        str(key): int(value) for key, value in working["label"].value_counts().items()
    }
    working = working.loc[working["source_activity"] >= int(minimum_activity)]
    after_activity = {
        str(key): int(value) for key, value in working["label"].value_counts().items()
    }
    working = working.loc[working["address"].isin(archive)]
    after_archive = {
        str(key): int(value) for key, value in working["label"].value_counts().items()
    }
    overlap_count = int(working["address"].isin(excluded).sum())
    working = working.loc[~working["address"].isin(excluded)].copy()
    working["activity_bin"] = _activity_bin(working["source_activity"])
    if working["activity_bin"].isna().any():
        raise RuntimeError("Real-CATS activity matching produced an unassigned bin")

    selected: list[pd.DataFrame] = []
    bin_audit: dict[str, dict[str, int]] = {}
    rng = np.random.default_rng(int(seed))
    for activity_bin, scoped in working.groupby(
        "activity_bin", observed=True, sort=True
    ):
        criminal = scoped.loc[scoped["label"] == 1].sort_values("address")
        benign = scoped.loc[scoped["label"] == 0].sort_values("address")
        matched = min(len(criminal), len(benign))
        if matched:
            criminal_indices = rng.choice(len(criminal), size=matched, replace=False)
            benign_indices = rng.choice(len(benign), size=matched, replace=False)
            selected.extend(
                [criminal.iloc[np.sort(criminal_indices)], benign.iloc[np.sort(benign_indices)]]
            )
        bin_audit[str(activity_bin)] = {
            "eligible_criminal": int(len(criminal)),
            "eligible_benign": int(len(benign)),
            "selected_per_class": int(matched),
        }
    if not selected:
        raise ValueError("No activity-matched Real-CATS binary cohort can be formed")
    result = pd.concat(selected, ignore_index=True).sort_values("address").reset_index(
        drop=True
    )
    if result["address"].duplicated().any():
        raise RuntimeError("Real-CATS binary matching selected duplicate addresses")
    if set(result["address"]) & excluded:
        raise RuntimeError("Real-CATS binary cohort overlaps the HF label universe")
    matched_counts = result["label"].value_counts()
    if len(matched_counts) != 2 or matched_counts.nunique() != 1:
        raise RuntimeError("Real-CATS binary cohort is not class balanced")
    result["activity_bin"] = result["activity_bin"].astype(str)
    result["campaign_group"] = "address:" + result["address"]
    audit = {
        "source_label_counts": source_counts,
        "after_minimum_source_activity": after_activity,
        "after_archive_presence": after_archive,
        "hf_label_universe_exclusions": overlap_count,
        "hf_label_universe_size": int(len(excluded)),
        "minimum_source_activity": int(minimum_activity),
        "activity_used_as_model_feature": False,
        "bin_counts": bin_audit,
        "selected_label_counts": {
            str(key): int(value) for key, value in matched_counts.items()
        },
    }
    return result, audit


def _clean_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _token_value(layer: str, record: dict) -> str:
    if layer == "erc20":
        return str(record.get("value", "0") or "0")
    return str(record.get("tokenValue", record.get("value", "0")) or "0")


def normalize_record(
    layer: str,
    record: dict,
    transaction_indices: dict[str, int] | None = None,
) -> dict | None:
    """Normalize one Etherscan record and construct a stable event signature."""
    if record.get("isError") == "1":
        return None
    from_address = str(record.get("from", "")).lower()
    to_address = str(record.get("to", "")).lower()
    timestamp = _clean_int(record.get("timeStamp"))
    if not from_address or not to_address or not timestamp or timestamp <= 0:
        return None

    tx_hash = str(record.get("hash", "")).lower()
    block_number = _clean_int(record.get("blockNumber")) or -1
    transaction_index = _clean_int(record.get("transactionIndex"))
    if transaction_index is None and transaction_indices:
        transaction_index = transaction_indices.get(tx_hash)

    if layer == "normal":
        identity = (layer, tx_hash)
        raw_value = str(record.get("value", "0") or "0")
        try:
            value = float(raw_value) / 1e18
        except (TypeError, ValueError, OverflowError):
            value = 0.0
    elif layer == "internal":
        trace_id = str(record.get("traceId", ""))
        identity = (
            layer,
            tx_hash,
            trace_id,
            from_address,
            to_address,
            str(record.get("value", "0") or "0"),
            str(record.get("type", "")),
        )
        try:
            value = float(record.get("value", 0) or 0) / 1e18
        except (TypeError, ValueError, OverflowError):
            value = 0.0
    else:
        token_value = _token_value(layer, record)
        identity = (
            layer,
            tx_hash,
            str(record.get("contractAddress", "")).lower(),
            from_address,
            to_address,
            str(record.get("tokenID", "")),
            token_value,
        )
        value = 0.0

    signature = "|".join(identity)
    return {
        "layer": layer,
        "from_address": from_address,
        "to_address": to_address,
        "timestamp": timestamp,
        "value": value,
        "tx_hash": tx_hash,
        "block_number": block_number,
        "transaction_index": transaction_index,
        "event_signature": signature,
    }


def _read_json(archive: zipfile.ZipFile, name: str) -> list[dict]:
    try:
        payload = json.loads(archive.read(name))
    except (json.JSONDecodeError, UnicodeDecodeError, OSError):
        return []
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict) and isinstance(payload.get("root"), list):
        return payload["root"]
    return []


def _create_event_index(path: Path, resume: bool = False) -> sqlite3.Connection:
    if path.exists() and not resume:
        raise FileExistsError(path)
    connection = sqlite3.connect(path)
    if resume and path.exists():
        try:
            check = connection.execute("PRAGMA quick_check(1)").fetchone()
        except sqlite3.DatabaseError as error:
            connection.close()
            raise RuntimeError(
                "The Real-CATS checkpoint is corrupt and cannot be resumed safely. "
                "Keep it for audit if needed, then prepare a new run ID."
            ) from error
        if not check or check[0] != "ok":
            connection.close()
            raise RuntimeError(
                "The Real-CATS checkpoint failed SQLite quick_check and cannot be "
                "resumed safely. Keep it for audit if needed, then prepare a new run ID."
            )
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA synchronous=FULL")
    connection.execute("PRAGMA temp_store=FILE")
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS events (
            event_id TEXT PRIMARY KEY,
            layer TEXT NOT NULL,
            from_address TEXT NOT NULL,
            to_address TEXT NOT NULL,
            timestamp INTEGER NOT NULL,
            value REAL NOT NULL,
            tx_hash TEXT NOT NULL,
            block_number INTEGER NOT NULL,
            transaction_index INTEGER,
            event_signature TEXT NOT NULL
        ) WITHOUT ROWID
        """
    )
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS processed_addresses (
            address TEXT PRIMARY KEY
        ) WITHOUT ROWID
        """
    )
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS source_audit (
            address TEXT NOT NULL,
            layer TEXT NOT NULL,
            source_records INTEGER NOT NULL,
            PRIMARY KEY (address, layer)
        ) WITHOUT ROWID
        """
    )
    connection.commit()
    return connection


def _insert_source_records(
    connection: sqlite3.Connection,
    layer: str,
    records: list[dict],
    transaction_indices: dict[str, int],
) -> tuple[int, int]:
    normalized = [
        item
        for record in records
        if (item := normalize_record(layer, record, transaction_indices)) is not None
    ]
    multiplicity = Counter(item["event_signature"] for item in normalized)
    representatives = {item["event_signature"]: item for item in normalized}
    before = connection.total_changes
    rows = []
    for signature, count in multiplicity.items():
        event = representatives[signature]
        for ordinal in range(count):
            event_id = hashlib.sha256(f"{signature}|{ordinal}".encode("utf-8")).hexdigest()
            rows.append(
                (
                    event_id,
                    event["layer"],
                    event["from_address"],
                    event["to_address"],
                    event["timestamp"],
                    event["value"],
                    event["tx_hash"],
                    event["block_number"],
                    event["transaction_index"],
                    event["event_signature"],
                )
            )
    connection.executemany(
        "INSERT OR IGNORE INTO events VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", rows
    )
    return len(normalized), connection.total_changes - before


def _export_layers(
    connection: sqlite3.Connection,
    output_dir: Path,
    telemetry: RunTelemetry | None = None,
) -> dict[str, Path]:
    output_paths: dict[str, Path] = {}
    columns = [
        "from_address",
        "to_address",
        "timestamp",
        "value",
        "tx_hash",
        "block_number",
        "transaction_index",
        "event_signature",
        "layer",
    ]
    schema = pa.schema(
        [
            ("from_address", pa.string()),
            ("to_address", pa.string()),
            ("timestamp", pa.int64()),
            ("value", pa.float64()),
            ("tx_hash", pa.string()),
            ("block_number", pa.int64()),
            ("transaction_index", pa.int64()),
            ("event_signature", pa.string()),
            ("layer", pa.string()),
        ]
    )
    for layer in REALCATS_LAYER_FILES:
        output = output_dir / f"{layer}.parquet"
        if output.exists():
            output_paths[layer] = output
            continue
        export_context = (
            telemetry.step(f"prepare.realcats.export.{layer}")
            if telemetry
            else nullcontext()
        )
        with export_context:
            temporary = output.with_suffix(".parquet.partial")
            temporary.unlink(missing_ok=True)
            cursor = connection.execute(
                """
                SELECT from_address, to_address, timestamp, value, tx_hash,
                       block_number, transaction_index, event_signature, layer
                FROM events WHERE layer = ?
                """,
                (layer,),
            )
            writer: pq.ParquetWriter | None = None
            exported_rows = 0
            while rows := cursor.fetchmany(100_000):
                frame = pd.DataFrame.from_records(rows, columns=columns)
                table = pa.Table.from_pandas(frame, preserve_index=False)
                if writer is None:
                    writer = pq.ParquetWriter(temporary, table.schema, compression="zstd")
                writer.write_table(table)
                exported_rows += len(table)
                if telemetry and exported_rows % 1_000_000 == 0:
                    telemetry.progress(
                        f"prepare.realcats.export.{layer}", exported_rows=exported_rows
                    )
            if writer is None:
                writer = pq.ParquetWriter(temporary, schema, compression="zstd")
            writer.close()
            temporary.replace(output)
            output_paths[layer] = output
    return output_paths


def convert_realcats_zip(
    zip_path: str | Path,
    label_dir: str | Path,
    output_dir: str | Path,
    minimum_free_gib: int = 25,
    keep_sqlite_index: bool = False,
    resume: bool = False,
    telemetry: RunTelemetry | None = None,
    selected_addresses: Iterable[str] | None = None,
    address_universe: Iterable[str] | None = None,
) -> dict:
    """Convert selected address histories into deduplicated layer Parquets."""
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=resume)
    ensure_free_space(destination.parent, minimum_free_gib)
    audit_path = destination / "conversion_audit.json"
    if resume and audit_path.exists():
        completed = json.loads(audit_path.read_text(encoding="utf-8"))
        if all(Path(path).exists() for path in completed.get("layer_paths", {}).values()):
            return completed
        raise RuntimeError("Real-CATS conversion audit exists but layer artifacts are missing")
    labels = load_criminal_labels(label_dir)
    all_labelled = set(labels["address"])
    validation_universe = (
        {str(address).lower() for address in address_universe}
        if address_universe is not None
        else all_labelled
    )
    requested_addresses = (
        {str(address).lower() for address in selected_addresses}
        if selected_addresses is not None
        else None
    )
    if requested_addresses is not None:
        unknown = requested_addresses - validation_universe
        if unknown:
            raise ValueError(
                "Selected Real-CATS addresses are absent from the locked label universe: "
                f"{sorted(unknown)[:10]}"
            )
    labelled = requested_addresses if requested_addresses is not None else all_labelled
    index_path = destination / "event_index.sqlite"
    connection = _create_event_index(index_path, resume=resume)
    audit = defaultdict(int)

    try:
        with zipfile.ZipFile(zip_path) as archive:
            by_address: defaultdict[str, dict[str, str]] = defaultdict(dict)
            for name in archive.namelist():
                parts = name.split("/")
                if len(parts) != 4 or parts[2].lower() not in labelled:
                    continue
                layer = _FILE_TO_LAYER.get(parts[3])
                if layer:
                    by_address[parts[2].lower()][layer] = name

            processed = {
                str(row[0])
                for row in connection.execute("SELECT address FROM processed_addresses")
            }
            audit["labelled_addresses"] = len(labelled)
            audit["release_labelled_addresses"] = len(all_labelled)
            audit["selection_universe_addresses"] = len(validation_universe)
            audit["address_selection_bounded"] = requested_addresses is not None
            audit["addresses_in_archive"] = len(by_address)
            audit["addresses_resumed"] = len(processed)
            remaining = len(by_address) - len(processed & set(by_address))
            completed_addresses = 0
            progress_context = (
                telemetry.task_progress(
                    "Real-CATS address conversion",
                    remaining,
                    unit="address",
                )
                if telemetry
                else nullcontext(lambda _count=1: None)
            )
            with progress_context as advance:
                for address, layer_files in by_address.items():
                    if address in processed:
                        continue
                    normal_records = (
                        _read_json(archive, layer_files["normal"])
                        if "normal" in layer_files
                        else []
                    )
                    transaction_indices = {
                        str(record.get("hash", "")).lower(): index
                        for record in normal_records
                        if (index := _clean_int(record.get("transactionIndex")))
                        is not None
                    }
                    for layer in REALCATS_LAYER_FILES:
                        name = layer_files.get(layer)
                        if not name:
                            continue
                        records = (
                            normal_records
                            if layer == "normal"
                            else _read_json(archive, name)
                        )
                        source_count, _ = _insert_source_records(
                            connection,
                            layer,
                            records,
                            transaction_indices,
                        )
                        connection.execute(
                            "INSERT OR REPLACE INTO source_audit VALUES (?, ?, ?)",
                            (address, layer, source_count),
                        )
                    connection.execute(
                        "INSERT OR IGNORE INTO processed_addresses VALUES (?)",
                        (address,),
                    )
                    connection.commit()
                    completed_addresses += 1
                    advance()
                    if telemetry and (
                        completed_addresses % 250 == 0
                        or completed_addresses == remaining
                    ):
                        telemetry.progress(
                            "prepare.realcats.address_conversion",
                            addresses_complete=completed_addresses,
                            addresses_total=remaining,
                            addresses_resumed=len(processed),
                        )

        connection.execute("CREATE INDEX IF NOT EXISTS events_layer ON events(layer)")
        connection.commit()
        layer_paths = _export_layers(connection, destination, telemetry=telemetry)
        for layer in REALCATS_LAYER_FILES:
            audit[f"{layer}_source_records"] = int(
                connection.execute(
                    "SELECT COALESCE(SUM(source_records), 0) FROM source_audit WHERE layer = ?",
                    (layer,),
                ).fetchone()[0]
            )
            audit[f"{layer}_unique_events"] = int(
                connection.execute(
                    "SELECT COUNT(*) FROM events WHERE layer = ?", (layer,)
                ).fetchone()[0]
            )
            audit[f"{layer}_exact_distinct_nodes"] = int(
                connection.execute(
                    """
                    SELECT COUNT(*) FROM (
                        SELECT from_address AS address FROM events WHERE layer = ?
                        UNION
                        SELECT to_address AS address FROM events WHERE layer = ?
                    )
                    """,
                    (layer, layer),
                ).fetchone()[0]
            )
    finally:
        try:
            connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        except sqlite3.DatabaseError:
            pass
        connection.close()

    audit["duplicate_copies_removed"] = sum(
        audit[f"{layer}_source_records"] - audit[f"{layer}_unique_events"]
        for layer in REALCATS_LAYER_FILES
    )
    audit["layer_paths"] = {layer: str(path) for layer, path in layer_paths.items()}
    completed = dict(audit)
    with audit_path.open("x", encoding="utf-8") as handle:
        json.dump(completed, handle, indent=2, sort_keys=True)
        handle.write("\n")
    if not keep_sqlite_index:
        index_path.unlink(missing_ok=True)
        Path(f"{index_path}-wal").unlink(missing_ok=True)
        Path(f"{index_path}-shm").unlink(missing_ok=True)
    return completed


def select_realcats_targets(
    labels: pd.DataFrame,
    layer_paths: dict[str, str | Path],
    minimum_class_size: int = REALCATS_CATEGORY_ELIGIBILITY[
        "minimum_active_addresses"
    ],
) -> tuple[pd.DataFrame, dict]:
    """Select active exact labels without merging or motif-coverage filtering."""
    labelled = set(labels["address"])
    active: set[str] = set()
    for path in layer_paths.values():
        parquet = pq.ParquetFile(path)
        for batch in parquet.iter_batches(columns=["from_address", "to_address"]):
            for column in (batch.column(0), batch.column(1)):
                values = pc.unique(column).to_pylist()
                for value in values:
                    normalized = str(value).lower()
                    if normalized in labelled:
                        active.add(normalized)

    selected = labels.loc[labels["address"].isin(active)].copy()
    excluded = set(REALCATS_CATEGORY_ELIGIBILITY["exclude_casefold_labels"])
    selected = selected.loc[
        ~selected["type_label"].str.casefold().isin(excluded)
    ]
    counts = selected["type_label"].value_counts()
    inferential = set(counts[counts >= minimum_class_size].index)
    selected["inference_eligible"] = selected["type_label"].isin(inferential)
    selected["campaign_group"] = "address:" + selected["address"]
    audit = {
        "criminal_labels": int(len(labels)),
        "active_criminal_labels": int(len(selected)),
        "exact_label_counts": {str(key): int(value) for key, value in counts.items()},
        "inferential_labels": sorted(inferential),
        "minimum_class_size": minimum_class_size,
    }
    return selected.reset_index(drop=True), audit


def count_target_usable_events(
    layer_paths: dict[str, str | Path], targets: Iterable[str]
) -> pd.Series:
    """Count deduplicated incident events across all converted layers."""
    target_index = pd.Index(
        sorted({str(address).lower() for address in targets}), name="address"
    )
    target_set = set(target_index)
    counts = Counter({address: 0 for address in target_index})
    for path in layer_paths.values():
        parquet = pq.ParquetFile(path)
        for batch in parquet.iter_batches(columns=["from_address", "to_address"]):
            senders = batch.column(0).to_pylist()
            receivers = batch.column(1).to_pylist()
            for sender, receiver in zip(senders, receivers, strict=True):
                sender_value = str(sender).lower()
                receiver_value = str(receiver).lower()
                if sender_value in target_set:
                    counts[sender_value] += 1
                if receiver_value in target_set and receiver_value != sender_value:
                    counts[receiver_value] += 1
    return pd.Series(counts, index=target_index, name="usable_events", dtype="int64")
