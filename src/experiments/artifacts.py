"""Immutable run directories, canonical JSON, hashes, and provenance helpers."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import sys
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from src.experiments.types import RunPaths


def json_ready(value: Any) -> Any:
    """Convert numpy, pathlib, and nested values into deterministic JSON values."""
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_ready(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    return value


def canonical_json_bytes(payload: dict) -> bytes:
    return json.dumps(
        json_ready(payload), sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def protocol_digest(payload: dict) -> str:
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def sha256_file(path: str | Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def create_run_paths(results_root: str | Path, run_id: str) -> RunPaths:
    if not run_id or run_id in {".", ".."} or "/" in run_id:
        raise ValueError("run_id must be a non-empty directory name")
    root = Path(results_root) / run_id
    root.mkdir(parents=True, exist_ok=False)
    paths = RunPaths.from_root(root)
    for path in (
        paths.data,
        paths.features,
        paths.predictions,
        paths.metrics,
        paths.figures,
        paths.tables,
    ):
        path.mkdir()
    return paths


def resume_run_paths(results_root: str | Path, run_id: str) -> RunPaths:
    """Open an incomplete preparation run without permitting result reuse."""
    root = Path(results_root) / run_id
    paths = RunPaths.from_root(root)
    if not root.is_dir():
        raise FileNotFoundError(root)
    if paths.protocol.exists() or paths.protocol_hash.exists():
        raise RuntimeError(f"Run {run_id!r} is already locked and cannot be resumed")
    for path in (
        paths.data,
        paths.features,
        paths.predictions,
        paths.metrics,
        paths.figures,
        paths.tables,
    ):
        if not path.is_dir():
            raise RuntimeError(f"Incomplete run is missing directory: {path}")
    if any(paths.features.iterdir()) or any(paths.predictions.iterdir()) or any(paths.metrics.iterdir()):
        raise RuntimeError("Cannot resume after feature, prediction, or metric generation")
    return paths


def write_json_exclusive(path: str | Path, payload: dict) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("x", encoding="utf-8") as handle:
        json.dump(json_ready(payload), handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def write_parquet_exclusive(path: str | Path, frame: pd.DataFrame) -> None:
    destination = Path(path)
    if destination.exists():
        raise FileExistsError(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(destination, index=False)


def write_csv_exclusive(path: str | Path, frame: pd.DataFrame) -> None:
    destination = Path(path)
    if destination.exists():
        raise FileExistsError(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(destination, index=False)


def write_locked_protocol(paths: RunPaths, protocol: dict) -> str:
    digest = protocol_digest(protocol)
    write_json_exclusive(paths.protocol, protocol)
    paths.protocol_hash.write_text(f"{digest}\n", encoding="ascii")
    return digest


def read_locked_protocol(run_root: str | Path) -> tuple[RunPaths, dict]:
    paths = RunPaths.from_root(Path(run_root))
    protocol = json.loads(paths.protocol.read_text(encoding="utf-8"))
    expected = paths.protocol_hash.read_text(encoding="ascii").strip()
    actual = protocol_digest(protocol)
    if expected != actual:
        raise RuntimeError(
            f"Protocol hash mismatch for {paths.root}: expected {expected}, got {actual}"
        )
    return paths, protocol


def dependency_versions(packages: Iterable[str]) -> dict[str, str]:
    versions: dict[str, str] = {}
    for package in packages:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "not-installed"
    return versions


def runtime_manifest() -> dict:
    repository_root = Path(__file__).resolve().parents[2]
    source_paths = sorted((repository_root / "src/experiments").glob("*.py"))
    runner = repository_root / "scripts/run_results.py"
    if runner.exists():
        source_paths.append(runner)
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "cpu_count": os.cpu_count(),
        "dependencies": dependency_versions(
            (
                "numpy",
                "pandas",
                "pyarrow",
                "duckdb",
                "scikit-learn",
                "xgboost",
                "raphtory",
                "matplotlib",
            )
        ),
        "source_sha256": {
            str(path.relative_to(repository_root)): sha256_file(path)
            for path in source_paths
        },
    }
