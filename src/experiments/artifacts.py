"""Immutable run directories, canonical JSON, hashes, and provenance helpers."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import secrets
import sys
from pathlib import Path
from typing import Any, Iterable

from src.experiments.types import RunPaths


def json_ready(value: Any) -> Any:
    """Convert numpy, pathlib, and nested values into deterministic JSON values."""
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_ready(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    # Avoid importing numerical libraries before execution thread limits are set.
    module = type(value).__module__
    if module == "numpy":
        if hasattr(value, "tolist"):
            return value.tolist()
        if hasattr(value, "item"):
            return value.item()
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


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_link_commit(temporary: Path, destination: Path) -> None:
    """Commit a fully fsynced temporary file without permitting replacement."""
    try:
        os.link(temporary, destination)
        _fsync_directory(destination.parent)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_write_bytes_exclusive(path: str | Path, payload: bytes) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.parent / f".{destination.name}.{secrets.token_hex(8)}.tmp"
    with temporary.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    _atomic_link_commit(temporary, destination)


def write_json_exclusive(path: str | Path, payload: dict) -> None:
    formatted = json.dumps(
        json_ready(payload), indent=2, sort_keys=True, allow_nan=False
    ).encode("utf-8") + b"\n"
    atomic_write_bytes_exclusive(path, formatted)


def write_parquet_exclusive(path: str | Path, frame: Any) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.parent / f".{destination.stem}.{secrets.token_hex(8)}{destination.suffix}"
    try:
        frame.to_parquet(temporary, index=False)
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        _atomic_link_commit(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def write_csv_exclusive(path: str | Path, frame: Any) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.parent / f".{destination.name}.{secrets.token_hex(8)}.tmp"
    try:
        with temporary.open("x", encoding="utf-8", newline="") as handle:
            frame.to_csv(handle, index=False)
            handle.flush()
            os.fsync(handle.fileno())
        _atomic_link_commit(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def write_locked_protocol(paths: RunPaths, protocol: dict) -> str:
    digest = protocol_digest(protocol)
    write_json_exclusive(paths.protocol, protocol)
    atomic_write_bytes_exclusive(paths.protocol_hash, f"{digest}\n".encode("ascii"))
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
                "scipy",
                "joblib",
                "threadpoolctl",
                "xgboost",
                "raphtory",
                "matplotlib",
                "zstandard",
                "datasets",
            )
        ),
        "source_sha256": {
            str(path.relative_to(repository_root)): sha256_file(path)
            for path in source_paths
        },
    }
