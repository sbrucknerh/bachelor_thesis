"""Same-host coordinated HF and Real-CATS execution."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import uuid

from src.experiments.artifacts import (
    atomic_write_bytes_exclusive,
    protocol_digest,
    read_locked_protocol,
    write_json_exclusive,
)
from src.experiments.constants import PROTOCOL_VERSION, REALCATS_RESEARCH_QUESTIONS
from src.experiments.execution_control import (
    FileLock,
    available_threads,
    default_host_lock_path,
    execution_state,
    utc_now,
)


def _coordination_root(first: Path, second: Path) -> Path:
    common = Path(os.path.commonpath([first.resolve(), second.resolve()]))
    if common in {first.resolve(), second.resolve()}:
        common = common.parent
    return common / "coordination"


def _validate_scopes(hf_root: Path, realcats_root: Path) -> None:
    _, hf = read_locked_protocol(hf_root)
    _, realcats = read_locked_protocol(realcats_root)
    if (
        int(hf.get("protocol_version", 0)) != PROTOCOL_VERSION
        or int(realcats.get("protocol_version", 0)) != PROTOCOL_VERSION
    ):
        raise RuntimeError(
            f"Coordinated execution requires two protocol-v{PROTOCOL_VERSION} roots"
        )
    if set(hf.get("research_questions", [])) != {"rq1", "rq2"}:
        raise RuntimeError("The HF coordinated root must contain exactly rq1 and rq2")
    if set(realcats.get("research_questions", [])) != REALCATS_RESEARCH_QUESTIONS:
        raise RuntimeError(
            "The Real-CATS coordinated root must contain exactly rq3 and rq4"
        )


def _resume_threads(root: Path) -> int:
    definition = json.loads((root / "execution_definition.json").read_text(encoding="utf-8"))
    return int(definition["thread_allocation"])


def execute_parallel(
    hf_run_root: str | Path,
    realcats_run_root: str | Path,
    *,
    resume: bool = False,
    host_lock_path: str | Path | None = None,
    total_threads: int | None = None,
) -> dict:
    hf_root = Path(hf_run_root)
    realcats_root = Path(realcats_run_root)
    if hf_root.resolve() == realcats_root.resolve():
        raise RuntimeError("Coordinated run roots must be distinct")
    _validate_scopes(hf_root, realcats_root)
    expected_state = "interrupted" if resume else "fresh"
    states = (execution_state(hf_root), execution_state(realcats_root))
    valid_resume_states = {"interrupted", "interrupted_initialization"}
    if resume:
        if not all(state in valid_resume_states for state in states):
            raise RuntimeError(
                "Parallel resume requires both roots to be interrupted; resume a single "
                f"incomplete root directly when states are mixed (got {states})"
            )
    elif states != (expected_state, expected_state):
        raise RuntimeError(f"Fresh parallel execution requires two fresh roots, got {states}")

    available_cpus = available_threads()
    cpus = available_cpus if total_threads is None else int(total_threads)
    if cpus < 2:
        raise RuntimeError("Parallel execution requires at least two CPU threads")
    if cpus > available_cpus:
        raise RuntimeError(
            f"Requested {cpus} threads but this process can use only {available_cpus}"
        )
    if resume:
        hf_threads = _resume_threads(hf_root)
        realcats_threads = _resume_threads(realcats_root)
        locked_total = hf_threads + realcats_threads
        if total_threads is not None and cpus != locked_total:
            raise RuntimeError(
                f"Parallel resume requires the locked combined allocation of {locked_total} threads"
            )
        if locked_total > available_cpus:
            raise RuntimeError(
                "This host has fewer CPUs than the combined locked resume allocation"
            )
    else:
        hf_threads = (cpus + 1) // 2
        realcats_threads = cpus // 2

    from src.experiments.resources import assert_preflight, preflight_runs

    assert_preflight(preflight_runs([hf_root, realcats_root], concurrent=True))
    host_path = Path(host_lock_path) if host_lock_path else default_host_lock_path()
    with FileLock(host_path) as host_lock:
        # Avoid admitting a second coordinator after state changed during preflight.
        current_states = (execution_state(hf_root), execution_state(realcats_root))
        if current_states != states:
            raise RuntimeError(f"Execution states changed during admission: {current_states}")

        coordinator_id = uuid.uuid4().hex
        directory = _coordination_root(hf_root, realcats_root) / coordinator_id
        directory.mkdir(parents=True, exist_ok=False)
        definition = {
            "schema_version": 1,
            "coordinator_id": coordinator_id,
            "created_at_utc": utc_now(),
            "resume": bool(resume),
            "host_lock_path": str(host_path),
            "available_threads": available_cpus,
            "allocated_threads": cpus,
            "runs": {
                "huggingface": {"root": str(hf_root.resolve()), "threads": hf_threads},
                "realcats": {"root": str(realcats_root.resolve()), "threads": realcats_threads},
            },
        }
        definition_sha = protocol_digest(definition)
        write_json_exclusive(directory / "definition.json", definition)
        atomic_write_bytes_exclusive(
            directory / "definition.sha256", f"{definition_sha}\n".encode("ascii")
        )

        runner = Path(__file__).resolve().parents[2] / "scripts" / "run_results.py"
        os.set_inheritable(host_lock.fd, True)
        processes = {}
        for position, (name, root, threads) in enumerate(
            (
                ("huggingface", hf_root, hf_threads),
                ("realcats", realcats_root, realcats_threads),
            )
        ):
            command = [
                sys.executable,
                str(runner),
                "execute",
                str(root),
                "--threads",
                str(threads),
                "--host-lock",
                str(host_path),
                "--_coordinator-id",
                coordinator_id,
                "--_coordinator-definition-sha256",
                definition_sha,
                "--_host-lock-fd",
                str(host_lock.fd),
            ]
            if resume:
                command.append("--resume")
            environment = os.environ.copy()
            environment["THESIS_PROGRESS_POSITION"] = str(position)
            environment["THESIS_PROGRESS_LABEL"] = name
            processes[name] = subprocess.Popen(
                command,
                pass_fds=(host_lock.fd,),
                env=environment,
            )

        exit_codes = {name: process.wait() for name, process in processes.items()}
        outcome = {
            "schema_version": 1,
            "coordinator_id": coordinator_id,
            "definition_sha256": definition_sha,
            "completed_at_utc": utc_now(),
            "exit_codes": exit_codes,
            "status": "complete" if all(code == 0 for code in exit_codes.values()) else "failed",
        }
        write_json_exclusive(directory / "outcome.json", outcome)
        outcome_sha = protocol_digest(outcome)
        atomic_write_bytes_exclusive(
            directory / "outcome.sha256", f"{outcome_sha}\n".encode("ascii")
        )
    if outcome["status"] != "complete":
        raise RuntimeError(
            "One or more coordinated executions failed; the healthy worker was allowed "
            f"to finish. Exit codes: {exit_codes}"
        )
    return outcome
