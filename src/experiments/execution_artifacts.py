"""Protocol-v5 checkpoint, stage, publication, and inventory primitives."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
from typing import Callable, Iterator

from src.experiments.artifacts import (
    atomic_write_bytes_exclusive,
    canonical_json_bytes,
    protocol_digest,
    sha256_file,
    write_json_exclusive,
    write_parquet_exclusive,
)
from src.experiments.constants import MOTIF_ROLE_NAMES
from src.experiments.types import RunPaths


PUBLIC_DIRECTORIES = ("features", "predictions", "metrics", "figures", "tables")
_STAGE_ID = re.compile(r"^[a-z0-9][a-z0-9._-]*$")


def _check_id(value: str, kind: str) -> str:
    if not _STAGE_ID.fullmatch(value):
        raise ValueError(f"Invalid {kind} identifier: {value!r}")
    return value


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _fsync_tree(root: Path) -> None:
    for path in sorted(root.rglob("*")):
        if path.is_file():
            with path.open("rb") as handle:
                os.fsync(handle.fileno())
    for path in sorted((item for item in root.rglob("*") if item.is_dir()), reverse=True):
        _fsync_directory(path)
    _fsync_directory(root)


def _payload_files(payload: Path) -> list[Path]:
    return sorted(path for path in payload.rglob("*") if path.is_file())


def _stage_manifest(stage_id: str, identity: dict, payload: Path) -> dict:
    return {
        "schema_version": 1,
        "stage_id": stage_id,
        "identity": identity,
        "payload": {
            str(path.relative_to(payload)): {
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
            }
            for path in _payload_files(payload)
        },
    }


def verify_stage(stage_dir: Path, expected_identity: dict | None = None) -> dict:
    manifest_path = stage_dir / "stage_manifest.json"
    if not manifest_path.is_file():
        raise RuntimeError(f"Committed stage has no manifest: {stage_dir}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if expected_identity is not None and manifest.get("identity") != expected_identity:
        raise RuntimeError(f"Committed stage identity mismatch: {stage_dir}")
    payload = stage_dir / "payload"
    recorded = manifest.get("payload", {})
    actual_names = {str(path.relative_to(payload)) for path in _payload_files(payload)}
    if actual_names != set(recorded):
        raise RuntimeError(f"Committed stage payload set changed: {stage_dir}")
    for relative, details in recorded.items():
        path = payload / relative
        if path.stat().st_size != int(details["size_bytes"]):
            raise RuntimeError(f"Committed stage size changed: {path}")
        if sha256_file(path) != details["sha256"]:
            raise RuntimeError(f"Committed stage hash changed: {path}")
    return manifest


def publish_stage(run_root: Path, stage_dir: Path, manifest: dict) -> None:
    """Create no-overwrite public hard links from a verified committed stage."""
    payload = stage_dir / "payload"
    for relative, details in sorted(manifest["payload"].items()):
        parts = Path(relative).parts
        if not parts or parts[0] not in PUBLIC_DIRECTORIES:
            raise RuntimeError(f"Stage payload is outside a public directory: {relative}")
        source = payload / relative
        destination = run_root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            if not destination.is_file() or sha256_file(destination) != details["sha256"]:
                raise RuntimeError(f"Existing public artifact conflicts with stage: {destination}")
            continue
        os.link(source, destination)
        _fsync_directory(destination.parent)


@dataclass(slots=True)
class StageWorkspace:
    stage_id: str
    directory: Path
    payload: Path
    paths: RunPaths
    reused: bool = False


class StageManager:
    """Commit immutable stage bundles and expose their payload through hard links."""

    def __init__(
        self,
        run_paths: RunPaths,
        attempt_id: str,
        base_identity: dict,
        telemetry=None,
    ) -> None:
        self.run_paths = run_paths
        self.attempt_id = attempt_id
        self.base_identity = dict(base_identity)
        self.telemetry = telemetry
        self.stages = run_paths.root / "stages"
        self.private = run_paths.root / "attempts" / attempt_id / "staging"
        self.stages.mkdir(exist_ok=True)
        self.private.mkdir(parents=True, exist_ok=True)

    def identity(self, stage_id: str, details: dict | None = None) -> dict:
        return {**self.base_identity, "stage_id": stage_id, "details": details or {}}

    @contextmanager
    def stage(
        self, stage_id: str, details: dict | None = None
    ) -> Iterator[StageWorkspace]:
        stage_id = _check_id(stage_id, "stage")
        identity = self.identity(stage_id, details)
        committed = self.stages / stage_id
        if committed.exists():
            manifest = verify_stage(committed, identity)
            publish_stage(self.run_paths.root, committed, manifest)
            if self.telemetry:
                self.telemetry.progress("execute.stage", stage_id=stage_id, reused=True)
            yield StageWorkspace(
                stage_id,
                committed,
                committed / "payload",
                self._paths(committed / "payload"),
                reused=True,
            )
            return

        staging = self.private / f"stage-{stage_id}"
        if staging.exists():
            raise RuntimeError(f"Uncommitted staging directory already exists: {staging}")
        payload = staging / "payload"
        for name in PUBLIC_DIRECTORIES:
            (payload / name).mkdir(parents=True, exist_ok=True)
        workspace = StageWorkspace(stage_id, staging, payload, self._paths(payload))
        try:
            yield workspace
            manifest = _stage_manifest(stage_id, identity, payload)
            write_json_exclusive(staging / "stage_manifest.json", manifest)
            _fsync_tree(staging)
            if committed.exists():
                raise FileExistsError(committed)
            os.rename(staging, committed)
            _fsync_directory(self.stages)
            publish_stage(self.run_paths.root, committed, manifest)
            if self.telemetry:
                self.telemetry.progress("execute.stage", stage_id=stage_id, reused=False)
        except BaseException:
            # Private staging is deliberately retained for audit and ignored by resume.
            raise

    def _paths(self, payload: Path) -> RunPaths:
        return RunPaths(
            root=self.run_paths.root,
            protocol=self.run_paths.protocol,
            protocol_hash=self.run_paths.protocol_hash,
            data=self.run_paths.data,
            features=payload / "features",
            predictions=payload / "predictions",
            metrics=payload / "metrics",
            figures=payload / "figures",
            tables=payload / "tables",
        )


def canonical_checkpoint_table(frame):
    """Return the fixed Arrow table used to hash motif-count checkpoint content."""
    import pyarrow as pa

    expected = list(MOTIF_ROLE_NAMES)
    work = frame.copy()
    if "address" not in work.columns:
        work = work.reset_index()
    if work["address"].isna().any():
        raise ValueError("Checkpoint addresses must be non-null")
    if list(work.columns) != ["address", *expected]:
        work = work.loc[:, ["address", *expected]]
    work["address"] = work["address"].astype(str)
    if work["address"].duplicated().any():
        raise ValueError("Checkpoint addresses must be non-null and unique")
    order = sorted(range(len(work)), key=lambda index: work.iloc[index]["address"].encode("utf-8"))
    work = work.iloc[order].reset_index(drop=True)
    arrays = [pa.array(work["address"], type=pa.string(), from_pandas=False)]
    for role in expected:
        values = work[role]
        if values.isna().any() or (values < 0).any():
            raise ValueError(f"Checkpoint role {role} must contain nonnegative integers")
        arrays.append(pa.array(values.astype("int64"), type=pa.int64(), from_pandas=False))
    return pa.Table.from_arrays(arrays, names=["address", *expected]).replace_schema_metadata(None)


def canonical_checkpoint_bytes(frame) -> bytes:
    import pyarrow as pa
    import pyarrow.ipc as ipc

    table = canonical_checkpoint_table(frame)
    sink = pa.BufferOutputStream()
    options = ipc.IpcWriteOptions(metadata_version=ipc.MetadataVersion.V5, use_legacy_format=False)
    with ipc.new_file(sink, table.schema, options=options) as writer:
        batches = table.combine_chunks().to_batches(max_chunksize=max(1, len(table)))
        if batches:
            writer.write_batch(batches[0])
        else:
            writer.write_batch(pa.RecordBatch.from_arrays(
                [pa.array([], type=field.type) for field in table.schema],
                schema=table.schema,
            ))
    return sink.getvalue().to_pybytes()


def canonical_checkpoint_hash(frame) -> str:
    return hashlib.sha256(canonical_checkpoint_bytes(frame)).hexdigest()


class CheckpointStore:
    def __init__(self, root: Path, base_identity: dict, attempt_id: str, telemetry=None) -> None:
        self.root = root / "checkpoints"
        self.private = root / "attempts" / attempt_id / "staging"
        self.base_identity = dict(base_identity)
        self.telemetry = telemetry
        self.root.mkdir(exist_ok=True)

    def get_or_compute(self, checkpoint_id: str, details: dict, compute: Callable[[], object]):
        import pandas as pd

        checkpoint_id = _check_id(checkpoint_id, "checkpoint")
        identity = {**self.base_identity, "checkpoint_id": checkpoint_id, "details": details}
        destination = self.root / checkpoint_id
        if destination.exists():
            frame = self._verify(destination, identity)
            if self.telemetry:
                self.telemetry.set_latest_checkpoint(checkpoint_id)
                self.telemetry.progress("execute.checkpoint", checkpoint_id=checkpoint_id, reused=True)
            return frame

        staging = self.private / f"checkpoint-{checkpoint_id}"
        if staging.exists():
            raise RuntimeError(f"Uncommitted checkpoint staging exists: {staging}")
        staging.mkdir(parents=True)
        frame = compute()
        canonical = canonical_checkpoint_hash(frame)
        write_parquet_exclusive(staging / "counts.parquet", canonical_checkpoint_table(frame).to_pandas())
        metadata = {
            "schema_version": 1,
            "identity": identity,
            "canonical_encoding": "arrow-ipc-v5-file-uncompressed-one-batch",
            "canonical_content_sha256": canonical,
            "row_count": int(len(frame)),
        }
        write_json_exclusive(staging / "metadata.json", metadata)
        _fsync_tree(staging)
        if destination.exists():
            raise FileExistsError(destination)
        os.rename(staging, destination)
        _fsync_directory(self.root)
        verified = self._verify(destination, identity)
        if self.telemetry:
            self.telemetry.set_latest_checkpoint(checkpoint_id)
            self.telemetry.progress("execute.checkpoint", checkpoint_id=checkpoint_id, reused=False)
        return verified

    @staticmethod
    def _verify(directory: Path, identity: dict):
        import pandas as pd

        data = directory / "counts.parquet"
        metadata_path = directory / "metadata.json"
        if not data.is_file() or not metadata_path.is_file():
            raise RuntimeError(f"Committed checkpoint is incomplete: {directory}")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("identity") != identity:
            raise RuntimeError(f"Checkpoint identity mismatch: {directory}")
        frame = pd.read_parquet(data)
        actual = canonical_checkpoint_hash(frame)
        if actual != metadata.get("canonical_content_sha256"):
            raise RuntimeError(f"Checkpoint content hash mismatch: {directory}")
        if len(frame) != int(metadata.get("row_count", -1)):
            raise RuntimeError(f"Checkpoint row count mismatch: {directory}")
        return frame.set_index("address")


def build_result_inventory(run_root: Path) -> dict:
    include: list[Path] = []
    definition = run_root / "execution_definition.json"
    if definition.is_file():
        include.append(definition)
    definition_sidecar = run_root / "execution_definition.sha256"
    if definition_sidecar.is_file():
        include.append(definition_sidecar)
    for checkpoint in sorted((run_root / "checkpoints").glob("*")):
        if checkpoint.is_dir():
            include.extend(path for path in checkpoint.rglob("*") if path.is_file())
    for stage in sorted((run_root / "stages").glob("*")):
        manifest = stage / "stage_manifest.json"
        if manifest.is_file():
            include.append(manifest)
    for directory in PUBLIC_DIRECTORIES:
        include.extend(path for path in (run_root / directory).rglob("*") if path.is_file())
    return {
        "schema_version": 1,
        "artifacts": {
            str(path.relative_to(run_root)): {
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
            }
            for path in sorted(set(include))
        },
    }


def write_inventory(run_root: Path) -> tuple[dict, str]:
    inventory = build_result_inventory(run_root)
    digest = protocol_digest(inventory)
    path = run_root / "result_artifacts.json"
    sidecar = run_root / "result_artifacts.sha256"
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing != inventory:
            raise RuntimeError("Existing result inventory conflicts with committed artifacts")
    else:
        write_json_exclusive(path, inventory)
    if sidecar.exists():
        if sidecar.read_text(encoding="ascii").strip() != digest:
            raise RuntimeError("Existing result inventory sidecar conflicts")
    else:
        atomic_write_bytes_exclusive(sidecar, f"{digest}\n".encode("ascii"))
    return inventory, digest


def verify_inventory(run_root: Path) -> dict:
    path = run_root / "result_artifacts.json"
    sidecar = run_root / "result_artifacts.sha256"
    inventory = json.loads(path.read_text(encoding="utf-8"))
    expected = sidecar.read_text(encoding="ascii").strip()
    if protocol_digest(inventory) != expected:
        raise RuntimeError("Result inventory sidecar hash mismatch")
    current = build_result_inventory(run_root)
    if current != inventory:
        raise RuntimeError("Result inventory does not match committed artifacts")
    return inventory
