"""Read-only protocol-v6 resource estimates and execution admission checks."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
import math
import os
from pathlib import Path
import shutil

from src.experiments.constants import (
    CHECKPOINT_STAGING_MIN_MIB,
    COORDINATOR_ALLOWANCE_GIB,
    DISK_RESERVE_FRACTION,
    DISK_RESERVE_MIN_GIB,
    DISK_SORT_FACTOR,
    DUCKDB_MEMORY_FRACTION,
    DUCKDB_MEMORY_MAX_GIB,
    DUCKDB_MEMORY_MIN_GIB,
    GRAPH_COLUMN_EXPANSION_FACTOR,
    MOTIF_RESULT_BYTES_PER_NODE,
    RAM_ADMISSION_FRACTION,
    PROTOCOL_VERSION,
    REALCATS_RESEARCH_QUESTIONS,
    RESULT_STAGING_ALLOWANCE_GIB,
    SMOKE_DISK_RESERVE_GIB,
)


GIB = 1024**3
MIB = 1024**2


class ResourceLimitError(RuntimeError):
    """The locked experiment cannot be admitted on the current host."""


def physical_memory_bytes() -> int | None:
    try:
        return int(os.sysconf("SC_PAGE_SIZE")) * int(os.sysconf("SC_PHYS_PAGES"))
    except (AttributeError, OSError, ValueError):
        return None


def parquet_uncompressed_bytes(
    path: str | Path, columns: Iterable[str] | None = None
) -> int:
    import pyarrow.parquet as pq

    metadata = pq.ParquetFile(path).metadata
    selected = set(columns) if columns is not None else None
    total = 0
    found: set[str] = set()
    for row_group_index in range(metadata.num_row_groups):
        row_group = metadata.row_group(row_group_index)
        for column_index in range(row_group.num_columns):
            column = row_group.column(column_index)
            if selected is None or column.path_in_schema in selected:
                total += column.total_uncompressed_size
                found.add(column.path_in_schema)
    missing = set() if selected is None else selected - found
    if missing:
        raise ValueError(f"Parquet resource estimate missing columns: {sorted(missing)}")
    return int(total)


def duckdb_memory_bytes(admitted_memory: int) -> int:
    return min(
        DUCKDB_MEMORY_MAX_GIB * GIB,
        max(DUCKDB_MEMORY_MIN_GIB * GIB, int(admitted_memory * DUCKDB_MEMORY_FRACTION)),
    )


def python_model_allowance_bytes(targets: int, max_feature_columns: int) -> int:
    return max(2 * GIB, 8 * int(targets) * int(max_feature_columns) * 8)


def graph_resource_estimate(
    path: str | Path,
    *,
    node_count: int,
    targets: int,
    max_feature_columns: int,
    admitted_memory: int,
    columns: Iterable[str] = ("timestamp", "from_address", "to_address"),
) -> dict[str, int | float]:
    uncompressed = parquet_uncompressed_bytes(path, columns=columns)
    graph = int(uncompressed * GRAPH_COLUMN_EXPANSION_FACTOR)
    sorter = duckdb_memory_bytes(admitted_memory)
    motif = int(node_count) * MOTIF_RESULT_BYTES_PER_NODE
    python_model = python_model_allowance_bytes(targets, max_feature_columns)
    checkpoint = max(CHECKPOINT_STAGING_MIN_MIB * MIB, 2 * targets * 33 * 8)
    peak = graph + sorter + motif + python_model + checkpoint
    disk = int(uncompressed * DISK_SORT_FACTOR) + checkpoint + RESULT_STAGING_ALLOWANCE_GIB * GIB
    return {
        "parquet_uncompressed_bytes": uncompressed,
        "graph_bytes": graph,
        "duckdb_bytes": sorter,
        "motif_result_bytes": motif,
        "python_model_bytes": python_model,
        "checkpoint_staging_bytes": checkpoint,
        "estimated_peak_bytes": peak,
        "estimated_scratch_bytes": disk,
    }


def _dataset_reports(run_root: Path, protocol: dict, admitted_memory: int) -> list[dict]:
    reports: list[dict] = []
    resources = protocol.get("resource_estimates", {})
    selected = set(protocol.get("research_questions", []))
    if {"rq1", "rq2"} & selected:
        dataset = protocol["datasets"]["huggingface"]
        path = run_root / dataset["artifacts"]["events"]
        reports.append(
            {
                "dataset": "huggingface",
                "layer": None,
                "path": str(path),
                **graph_resource_estimate(
                    path,
                    node_count=int(resources["huggingface"]["node_count_admission"]),
                    targets=int(resources["huggingface"]["target_count"]),
                    max_feature_columns=int(resources["huggingface"]["max_feature_columns"]),
                    admitted_memory=admitted_memory,
                ),
            }
        )
    if selected & REALCATS_RESEARCH_QUESTIONS:
        dataset = protocol["datasets"]["realcats"]
        for layer, relative in sorted(dataset["artifacts"]["layers"].items()):
            path = run_root / relative
            layer_resource = resources["realcats"]["layers"][layer]
            reports.append(
                {
                    "dataset": "realcats",
                    "layer": layer,
                    "path": str(path),
                    **graph_resource_estimate(
                        path,
                        node_count=int(layer_resource["exact_distinct_nodes"]),
                        targets=int(resources["realcats"]["target_count"]),
                        max_feature_columns=int(resources["realcats"]["max_feature_columns"]),
                        admitted_memory=admitted_memory,
                    ),
                }
            )
    if "blte" in selected:
        dataset = protocol["datasets"]["blte"]
        path = run_root / dataset["artifacts"]["events"]
        metadata = resources["blte"]
        reports.append(
            {
                "dataset": "blte",
                "layer": None,
                "path": str(path),
                **graph_resource_estimate(
                    path,
                    node_count=int(metadata["exact_distinct_nodes"]),
                    targets=int(metadata["target_count"]),
                    max_feature_columns=int(metadata["max_feature_columns"]),
                    admitted_memory=admitted_memory,
                ),
            }
        )
    return reports


def estimate_run(run_root: str | Path) -> dict:
    """Return a resource report without creating or modifying any path."""
    from src.experiments.artifacts import read_locked_protocol

    root = Path(run_root)
    _, protocol = read_locked_protocol(root)
    version = int(protocol.get("protocol_version", 1))
    if version != PROTOCOL_VERSION:
        return {
            "run_root": str(root),
            "protocol_version": version,
            "admissible": False,
            "reason": "legacy protocols support read-only integrity inspection only",
        }
    physical = physical_memory_bytes()
    admitted = int(physical * RAM_ADMISSION_FRACTION) if physical else 64 * GIB
    reports = _dataset_reports(root, protocol, admitted)
    hf_peaks = [int(item["estimated_peak_bytes"]) for item in reports if item["dataset"] == "huggingface"]
    real_peaks = [int(item["estimated_peak_bytes"]) for item in reports if item["dataset"] == "realcats"]
    sequential_peak = max(hf_peaks + real_peaks + [0])
    scratch = max((int(item["estimated_scratch_bytes"]) for item in reports), default=0)
    usage = shutil.disk_usage(root)
    if not reports:
        reserve = 0
    elif protocol.get("execution_profile") == "smoke":
        reserve = SMOKE_DISK_RESERVE_GIB * GIB
    else:
        reserve = max(
            DISK_RESERVE_MIN_GIB * GIB,
            int(usage.total * DISK_RESERVE_FRACTION),
        )
    disk_required = scratch + reserve
    admissible = (physical is None or sequential_peak <= admitted) and usage.free >= disk_required
    return {
        "run_root": str(root),
        "protocol_version": version,
        "admissible": admissible,
        "physical_memory_bytes": physical,
        "ram_admission_bytes": admitted,
        "sequential_peak_bytes": sequential_peak,
        "scratch_bytes": scratch,
        "filesystem_device": int(root.stat().st_dev),
        "filesystem_total_bytes": usage.total,
        "filesystem_free_bytes": usage.free,
        "filesystem_reserve_bytes": reserve,
        "disk_required_bytes": disk_required,
        "components": reports,
    }


def preflight_runs(run_roots: Iterable[str | Path], concurrent: bool = False) -> dict:
    """Read-only admission for one run or a coordinated pair."""
    reports = [estimate_run(root) for root in run_roots]
    if any(
        int(item.get("protocol_version", 0)) != PROTOCOL_VERSION
        for item in reports
    ):
        return {"admissible": False, "concurrent": concurrent, "runs": reports}
    physical = physical_memory_bytes()
    admitted = int(physical * RAM_ADMISSION_FRACTION) if physical else math.inf
    peaks = [int(item["sequential_peak_bytes"]) for item in reports]
    combined_peak = (
        sum(peaks) + COORDINATOR_ALLOWANCE_GIB * GIB
        if concurrent
        else max(peaks, default=0)
    )
    by_device: defaultdict[int, dict[str, int]] = defaultdict(lambda: {"scratch": 0, "free": 0, "reserve": 0})
    for report in reports:
        device = int(report["filesystem_device"])
        by_device[device]["scratch"] += int(report["scratch_bytes"])
        by_device[device]["free"] = int(report["filesystem_free_bytes"])
        by_device[device]["reserve"] = max(
            by_device[device]["reserve"], int(report["filesystem_reserve_bytes"])
        )
    devices = {
        str(device): {
            **values,
            "required": values["scratch"] + values["reserve"],
            "admissible": values["free"] >= values["scratch"] + values["reserve"],
        }
        for device, values in by_device.items()
    }
    admissible = (
        all(item["admissible"] for item in reports)
        and combined_peak <= admitted
        and all(item["admissible"] for item in devices.values())
    )
    return {
        "admissible": admissible,
        "concurrent": concurrent,
        "physical_memory_bytes": physical,
        "ram_admission_bytes": None if math.isinf(admitted) else admitted,
        "combined_peak_bytes": combined_peak,
        "devices": devices,
        "runs": reports,
    }


def assert_preflight(report: dict) -> None:
    if not report.get("admissible"):
        raise ResourceLimitError(
            f"Protocol-v{PROTOCOL_VERSION} resource admission failed. Inspect the preflight report for "
            "RAM and per-filesystem disk requirements; no execution attempt was started."
        )


def require_in_memory_execution(
    path: str | Path,
    expansion_factor: float = GRAPH_COLUMN_EXPANSION_FACTOR,
    memory_fraction: float = RAM_ADMISSION_FRACTION,
    columns: Iterable[str] | None = None,
    external_sort_memory_fraction: float = DUCKDB_MEMORY_FRACTION,
) -> dict[str, float]:
    """Compatibility guard used by focused graph helpers and legacy tests."""
    uncompressed = parquet_uncompressed_bytes(path, columns=columns)
    physical = physical_memory_bytes()
    graph_estimate = int(uncompressed * expansion_factor)
    sorter_memory = duckdb_memory_bytes(physical or 64 * GIB)
    estimated_peak = graph_estimate + sorter_memory
    report = {
        "projected_parquet_uncompressed_gib": uncompressed / GIB,
        "graph_estimate_gib": graph_estimate / GIB,
        "external_sort_memory_gib": sorter_memory / GIB,
        "estimated_peak_gib": estimated_peak / GIB,
        "physical_memory_gib": physical / GIB if physical else float("nan"),
    }
    if physical is not None and estimated_peak > physical * memory_fraction:
        raise ResourceLimitError(
            f"Locked artifact {path} needs an estimated {report['estimated_peak_gib']:.1f} GiB; "
            f"this host has {report['physical_memory_gib']:.1f} GiB. The locked cohort "
            "was not trimmed or replaced."
        )
    return report
