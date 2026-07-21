"""Conservative resource checks that fail before partial experiment execution."""

from __future__ import annotations

import os
from pathlib import Path
from collections.abc import Iterable

import pyarrow.parquet as pq


class ResourceLimitError(RuntimeError):
    """The locked cohort cannot be evaluated on the current machine unchanged."""


def physical_memory_bytes() -> int | None:
    try:
        return int(os.sysconf("SC_PAGE_SIZE")) * int(os.sysconf("SC_PHYS_PAGES"))
    except (AttributeError, OSError, ValueError):
        return None


def parquet_uncompressed_bytes(
    path: str | Path, columns: Iterable[str] | None = None
) -> int:
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


def require_in_memory_execution(
    path: str | Path,
    expansion_factor: float = 3.0,
    memory_fraction: float = 0.8,
    columns: Iterable[str] | None = None,
    external_sort_memory_fraction: float = 0.15,
) -> dict[str, float]:
    """Reject a graph load that cannot fit with a conservative safety margin."""
    uncompressed = parquet_uncompressed_bytes(path, columns=columns)
    physical = physical_memory_bytes()
    graph_estimate = int(uncompressed * expansion_factor)
    sorter_memory = (
        min(16 * 1024**3, max(1 * 1024**3, int(physical * external_sort_memory_fraction)))
        if physical is not None
        else 0
    )
    estimated_peak = graph_estimate + sorter_memory
    report = {
        "projected_parquet_uncompressed_gib": uncompressed / 1024**3,
        "graph_estimate_gib": graph_estimate / 1024**3,
        "external_sort_memory_gib": sorter_memory / 1024**3,
        "estimated_peak_gib": estimated_peak / 1024**3,
        "physical_memory_gib": physical / 1024**3 if physical else float("nan"),
    }
    if physical is not None and estimated_peak > physical * memory_fraction:
        raise ResourceLimitError(
            f"Locked artifact {path} projects to "
            f"{report['projected_parquet_uncompressed_gib']:.1f} GiB for graph columns "
            f"and needs an estimated {report['estimated_peak_gib']:.1f} GiB "
            f"for the externally sorted Raphtory graph; this machine has "
            f"{report['physical_memory_gib']:.1f} GiB. "
            "The cohort was not trimmed or replaced. Execute this locked run on a "
            "larger-memory machine, or prepare a new explicitly resource-bounded protocol."
        )
    return report
