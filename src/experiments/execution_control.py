"""Protocol-v6 execution ownership, state transitions, and immutable definitions."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import uuid

from src.experiments.artifacts import (
    atomic_write_bytes_exclusive,
    protocol_digest,
    write_json_exclusive,
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def available_threads() -> int:
    try:
        return max(1, len(os.sched_getaffinity(0)))
    except (AttributeError, OSError):
        pass
    return max(1, os.cpu_count() or 1)


def default_host_lock_path() -> Path:
    return Path(f"/tmp/bachelor-thesis-execution-{os.getuid()}.lock")


def configure_thread_environment(threads: int) -> None:
    """Set native pool limits before importing numerical libraries."""
    if threads <= 0:
        raise ValueError("threads must be positive")
    value = str(int(threads))
    for name in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "RAYON_NUM_THREADS",
        "ARROW_NUM_THREADS",
        "THESIS_MODEL_THREADS",
    ):
        os.environ[name] = value


class ExecutionLockError(RuntimeError):
    pass


class FileLock:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.handle = None

    def acquire(self) -> "FileLock":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = self.path.open("a+")
        try:
            fcntl.flock(self.handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            self.handle.close()
            self.handle = None
            raise ExecutionLockError(f"Execution lock is already held: {self.path}") from error
        return self

    @property
    def fd(self) -> int:
        if self.handle is None:
            raise RuntimeError("Lock is not acquired")
        return self.handle.fileno()

    def release(self) -> None:
        if self.handle is not None:
            fcntl.flock(self.handle.fileno(), fcntl.LOCK_UN)
            self.handle.close()
            self.handle = None

    def __enter__(self) -> "FileLock":
        return self.acquire()

    def __exit__(self, *_args) -> None:
        self.release()


def validate_inherited_lock(fd: int, path: Path) -> None:
    try:
        inherited = os.fstat(fd)
        expected = path.stat()
    except (OSError, FileNotFoundError) as error:
        raise ExecutionLockError("Coordinator host-lock descriptor is invalid") from error
    if (inherited.st_dev, inherited.st_ino) != (expected.st_dev, expected.st_ino):
        raise ExecutionLockError("Coordinator host-lock descriptor does not match host lock")


def execution_state(run_root: Path) -> str:
    definition = run_root / "execution_definition.json"
    manifest = run_root / "execution_manifest.json"
    attempts = run_root / "attempts"
    checkpoints = run_root / "checkpoints"
    stages = run_root / "stages"
    if manifest.exists() and (run_root / "execution_manifest.sha256").exists():
        return "completed"
    if not definition.exists():
        residue = any(
            path.exists() and (not path.is_dir() or any(path.iterdir()))
            for path in (attempts, checkpoints, stages)
        )
        residue = residue or (run_root / "execution_definition.sha256").exists()
        residue = residue or any(
            path.exists()
            for path in (
                run_root / "result_artifacts.json",
                run_root / "result_artifacts.sha256",
                run_root / "execution_manifest.sha256",
            )
        )
        residue = residue or any(
            directory.exists() and any(directory.rglob("*"))
            for directory in (
                run_root / "features",
                run_root / "predictions",
                run_root / "metrics",
                run_root / "figures",
                run_root / "tables",
            )
        )
        return "invalid_fresh" if residue else "fresh"
    if not attempts.exists() or not any(attempts.iterdir()):
        return "interrupted_initialization"
    return "interrupted"


def _definition_payload(
    protocol_sha256: str,
    threads: int,
    attempt_id: str,
    mode: str,
    coordinator_id: str | None,
    coordinator_definition_sha256: str | None,
) -> dict:
    try:
        physical_gib = int(os.sysconf("SC_PAGE_SIZE")) * int(os.sysconf("SC_PHYS_PAGES")) / 1024**3
    except (AttributeError, OSError, ValueError):
        physical_gib = 32
    duckdb_memory_gib = max(1, min(16, int(physical_gib * 0.15)))
    return {
        "schema_version": 1,
        "protocol_sha256": protocol_sha256,
        "initial_attempt_id": attempt_id,
        "thread_allocation": int(threads),
        "execution_mode": mode,
        "coordinator_id": coordinator_id,
        "coordinator_definition_sha256": coordinator_definition_sha256,
        "operational_model_parameters": {
            "random_forest_n_jobs": int(threads),
            "xgboost_n_jobs": int(threads),
            "blas_thread_limit": int(threads),
            "raphtory_threads": int(threads),
            "duckdb_threads": int(threads),
            "duckdb_memory_gib": duckdb_memory_gib,
            "arrow_threads": int(threads),
        },
    }


@dataclass(frozen=True, slots=True)
class ExecutionAttempt:
    attempt_id: str
    definition: dict
    definition_sha256: str
    directory: Path
    resume: bool


def start_attempt(
    run_root: Path,
    protocol_sha256: str,
    threads: int | None,
    resume: bool,
    mode: str = "direct",
    coordinator_id: str | None = None,
    coordinator_definition_sha256: str | None = None,
) -> ExecutionAttempt:
    state = execution_state(run_root)
    definition_path = run_root / "execution_definition.json"
    definition_hash_path = run_root / "execution_definition.sha256"
    requested_threads = available_threads() if threads is None else int(threads)
    if requested_threads <= 0 or requested_threads > available_threads():
        raise RuntimeError(
            f"Thread allocation {requested_threads} is invalid on a host with "
            f"{available_threads()} logical CPUs"
        )
    if state == "completed":
        raise RuntimeError("Completed runs reject execute and execute --resume")
    if resume:
        if state not in {"interrupted", "interrupted_initialization"}:
            raise RuntimeError(
                f"--resume requires an interrupted current-protocol execution, got {state}"
            )
        definition = json.loads(definition_path.read_text(encoding="utf-8"))
        locked_threads = int(definition["thread_allocation"])
        if locked_threads > available_threads():
            raise RuntimeError(
                f"Resume requires {locked_threads} locked threads but this host has "
                f"{available_threads()}"
            )
        if threads is not None and int(threads) != locked_threads:
            raise RuntimeError(
                f"Resume thread allocation must equal locked value {locked_threads}"
            )
        requested_threads = locked_threads
        expected_hash = definition_hash_path.read_text(encoding="ascii").strip()
        if protocol_digest(definition) != expected_hash:
            raise RuntimeError("Execution definition hash mismatch")
        if definition.get("protocol_sha256") != protocol_sha256:
            raise RuntimeError("Execution definition references a different protocol")
        definition_sha = expected_hash
    else:
        if state != "fresh":
            raise RuntimeError(f"Fresh execute requires pristine execution state, got {state}")
        initial_id = uuid.uuid4().hex
        definition = _definition_payload(
            protocol_sha256,
            requested_threads,
            initial_id,
            mode,
            coordinator_id,
            coordinator_definition_sha256,
        )
        definition_sha = protocol_digest(definition)
        write_json_exclusive(definition_path, definition)
        atomic_write_bytes_exclusive(
            definition_hash_path, f"{definition_sha}\n".encode("ascii")
        )

    attempt_id = (
        str(definition["initial_attempt_id"])
        if not resume
        else uuid.uuid4().hex
    )
    directory = run_root / "attempts" / attempt_id
    directory.mkdir(parents=True, exist_ok=False)
    (directory / "staging").mkdir()
    (directory / "scratch" / "duckdb").mkdir(parents=True)
    write_json_exclusive(
        directory / "started.json",
        {
            "attempt_id": attempt_id,
            "started_at_utc": utc_now(),
            "resume": bool(resume),
            "invocation_mode": mode,
            "execution_definition_sha256": definition_sha,
            "coordinator_id": coordinator_id,
            "coordinator_definition_sha256": coordinator_definition_sha256,
        },
    )
    return ExecutionAttempt(attempt_id, definition, definition_sha, directory, resume)


def finish_attempt(attempt: ExecutionAttempt, status: str, error: BaseException | None = None) -> None:
    if status not in {"completed", "failed"}:
        raise ValueError(status)
    payload = {
        "attempt_id": attempt.attempt_id,
        f"{status}_at_utc": utc_now(),
        "status": status,
    }
    if error is not None:
        payload.update(error_type=type(error).__name__, error=str(error))
    write_json_exclusive(attempt.directory / f"{status}.json", payload)
