"""One-command preparation, execution, and verification for thesis servers."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys

from src.experiments.constants import PROTOCOL_VERSION, REALCATS_RESEARCH_QUESTIONS


def _status(message: str) -> None:
    timestamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    print(f"[{timestamp}] THESIS   {message}", file=sys.stderr, flush=True)


def _default_run_id() -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return f"thesis_{stamp}_v{PROTOCOL_VERSION}"


def _validate_locked_root(
    root: Path,
    expected_rqs: set[str],
    replication_index: int,
    expected_hf_exclusions: tuple[Path, ...] = (),
) -> None:
    from src.experiments.artifacts import read_locked_protocol

    _, protocol = read_locked_protocol(root)
    if int(protocol.get("protocol_version", 0)) != PROTOCOL_VERSION:
        raise RuntimeError(
            f"{root} uses protocol v{protocol.get('protocol_version')}; "
            f"the current workflow requires v{PROTOCOL_VERSION}"
        )
    if set(protocol.get("research_questions", [])) != expected_rqs:
        raise RuntimeError(
            f"{root} has RQ scope {protocol.get('research_questions')}; "
            f"expected {sorted(expected_rqs)}"
        )
    if protocol.get("execution_profile") != "thesis":
        raise RuntimeError(f"{root} is not a thesis-profile run")
    if int(protocol.get("replication", {}).get("index", -1)) != int(
        replication_index
    ):
        raise RuntimeError(f"{root} has a different locked replication index")
    actual_exclusions = {
        Path(item["run_root"]).resolve()
        for item in protocol.get("replication", {}).get(
            "hf_exclusion_references", []
        )
    }
    if actual_exclusions != set(expected_hf_exclusions):
        raise RuntimeError(f"{root} has different locked HF exclusion roots")


def _prepare_roots(
    results_root: Path,
    base_run_id: str,
    hf_snapshot: Path,
    realcats_zip: Path,
    realcats_labels: Path,
    resume: bool,
    replication_index: int,
    exclude_hf_runs: tuple[Path, ...],
) -> tuple[Path, Path]:
    from src.experiments.prepare import prepare_experiment

    hf_id = f"{base_run_id}_hf"
    realcats_id = f"{base_run_id}_realcats"
    hf_root = _prepare_hf_root(
        results_root,
        hf_id,
        hf_snapshot,
        resume,
        replication_index,
        exclude_hf_runs,
    )
    realcats_root = results_root / realcats_id

    if (realcats_root / "protocol.json").is_file():
        if not resume:
            raise FileExistsError(
                f"{realcats_root} already exists; use --resume to continue this run"
            )
        _validate_locked_root(
            realcats_root,
            set(REALCATS_RESEARCH_QUESTIONS),
            replication_index,
        )
        _status(f"Real-CATS preparation already locked: {realcats_root}")
    elif realcats_root.exists():
        if not resume:
            raise FileExistsError(
                f"{realcats_root} is incomplete; rerun with --resume"
            )
        _status("Resuming Real-CATS RQ3/RQ4 preparation")
        prepare_experiment(
            results_root,
            realcats_id,
            sorted(REALCATS_RESEARCH_QUESTIONS),
            realcats_zip=realcats_zip,
            realcats_labels=realcats_labels,
            profile="thesis",
            resume=True,
            replication_index=replication_index,
        )
    else:
        _status("Preparing Real-CATS RQ3/RQ4 root")
        prepare_experiment(
            results_root,
            realcats_id,
            sorted(REALCATS_RESEARCH_QUESTIONS),
            realcats_zip=realcats_zip,
            realcats_labels=realcats_labels,
            profile="thesis",
            replication_index=replication_index,
        )

    return hf_root, realcats_root


def _prepare_hf_root(
    results_root: Path,
    run_id: str,
    hf_snapshot: Path,
    resume: bool,
    replication_index: int,
    exclude_hf_runs: tuple[Path, ...],
) -> Path:
    from src.experiments.prepare import prepare_experiment

    root = results_root / run_id
    if (root / "protocol.json").is_file():
        if not resume:
            raise FileExistsError(
                f"{root} already exists; use --resume to continue this run"
            )
        _validate_locked_root(
            root,
            {"rq1", "rq2"},
            replication_index,
            exclude_hf_runs,
        )
        _status(f"HF preparation already locked: {root}")
    elif root.exists():
        raise RuntimeError(
            "Hugging Face preparation was interrupted before protocol locking and "
            "is not resumable. Use a new --run-id; the full HF scan is intentionally "
            "never inferred from partial artifacts."
        )
    else:
        _status("Preparing Hugging Face RQ1/RQ2 root")
        prepare_experiment(
            results_root,
            run_id,
            ["rq1", "rq2"],
            hf_snapshot=hf_snapshot,
            profile="thesis",
            replication_index=replication_index,
            exclude_hf_runs=exclude_hf_runs,
        )
    return root


def run_hf_workflow(
    *,
    results_root: str | Path = "results/experiments",
    run_id: str | None = None,
    hf_snapshot: str | Path | None = None,
    threads: int | None = None,
    host_lock_path: str | Path | None = None,
    resume: bool = False,
    replication_index: int = 0,
    exclude_hf_runs: tuple[str | Path, ...] | list[str | Path] = (),
) -> dict:
    """Prepare, execute, and verify one citable RQ1/RQ2 run."""
    from src.experiments.datasets import discover_hf_snapshot
    from src.experiments.execution_control import available_threads
    from src.experiments.resources import assert_preflight, preflight_runs
    from src.experiments.verify import verify_run

    available = available_threads()
    allocated = available if threads is None else int(threads)
    if allocated <= 0 or allocated > available:
        raise ValueError(
            f"--threads must be between 1 and the available CPU count ({available})"
        )
    base_run_id = run_id or _default_run_id()
    if not base_run_id or base_run_id in {".", ".."} or "/" in base_run_id:
        raise ValueError("run_id must be a non-empty directory name")
    snapshot = Path(hf_snapshot) if hf_snapshot else discover_hf_snapshot()
    required = (
        snapshot / "gnn_dataset/edges_all/edges.parquet",
        snapshot / "gnn_dataset/labels/targets_global.parquet",
        snapshot / "addr_labels_balanced.csv.zst",
    )
    missing = [path for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Required Hugging Face inputs are missing: "
            + ", ".join(str(path) for path in missing)
        )
    exclusions = tuple(Path(path).resolve() for path in exclude_hf_runs)
    _status(
        f"Starting protocol-v{PROTOCOL_VERSION} HF run {base_run_id}; "
        f"threads={allocated}, replication={replication_index}"
    )
    _status(
        "Initial HF reference estimate: approximately 3-5 hours on the tested "
        "20-core server; the measured total ETA will replace this range"
    )
    root = _prepare_hf_root(
        Path(results_root),
        f"{base_run_id}_hf",
        snapshot,
        resume,
        replication_index,
        exclusions,
    )
    assert_preflight(preflight_runs([root], concurrent=False))
    _execute_one(root, allocated, host_lock_path)
    verification = verify_run(root, require_citable=True)
    _status("RQ1/RQ2 run completed and passed citable verification")
    return {
        "schema_version": 1,
        "protocol_version": PROTOCOL_VERSION,
        "run_id": base_run_id,
        "status": "complete",
        "thread_budget": int(verification.get("thread_allocation", allocated)),
        "replication_index": int(replication_index),
        "excluded_hf_runs": [str(path) for path in exclusions],
        "gpu_used": False,
        "run_root": str(root.resolve()),
        "verification": verification,
    }


def _execute_one(
    root: Path,
    threads: int,
    host_lock_path: str | Path | None,
) -> str:
    from src.experiments.execute import execute_experiment
    from src.experiments.execution_control import execution_state

    state = execution_state(root)
    if state == "completed":
        _status(f"Execution already complete: {root}")
        return state
    if state == "fresh":
        _status(f"Executing {root.name} with {threads} threads")
        execute_experiment(
            root,
            threads=threads,
            host_lock_path=host_lock_path,
        )
        return "completed"
    if state in {"interrupted", "interrupted_initialization"}:
        _status(f"Resuming {root.name} with its locked thread allocation")
        execute_experiment(
            root,
            resume=True,
            host_lock_path=host_lock_path,
        )
        return "completed"
    raise RuntimeError(f"Cannot execute {root}: state is {state}")


def run_thesis_workflow(
    *,
    results_root: str | Path = "results/experiments",
    run_id: str | None = None,
    hf_snapshot: str | Path | None = None,
    realcats_zip: str | Path = "data/raw/realcats_transactions/real-cats.zip",
    realcats_labels: str | Path = "data/raw/realcats_dataset",
    threads: int | None = None,
    execution_mode: str = "auto",
    host_lock_path: str | Path | None = None,
    resume: bool = False,
    replication_index: int = 0,
    exclude_hf_runs: tuple[str | Path, ...] | list[str | Path] = (),
) -> dict:
    """Run all primary thesis experiments and return verified run locations."""
    from src.experiments.artifacts import read_locked_protocol
    from src.experiments.datasets import discover_hf_snapshot
    from src.experiments.execution_control import available_threads, execution_state
    from src.experiments.resources import assert_preflight, preflight_runs
    from src.experiments.verify import verify_run

    if execution_mode not in {"auto", "parallel", "sequential"}:
        raise ValueError(f"Unknown execution mode: {execution_mode}")
    available = available_threads()
    allocated = available if threads is None else int(threads)
    if allocated <= 0 or allocated > available:
        raise ValueError(
            f"--threads must be between 1 and the available CPU count ({available})"
        )
    if execution_mode == "parallel" and allocated < 2:
        raise ValueError("Parallel execution requires at least two threads")

    base_run_id = run_id or _default_run_id()
    if not base_run_id or base_run_id in {".", ".."} or "/" in base_run_id:
        raise ValueError("run_id must be a non-empty directory name")
    result_path = Path(results_root)
    snapshot = Path(hf_snapshot) if hf_snapshot else discover_hf_snapshot()
    zip_path = Path(realcats_zip)
    label_path = Path(realcats_labels)
    exclusion_roots = tuple(Path(path).resolve() for path in exclude_hf_runs)
    missing = [
        path
        for path in (
            snapshot / "gnn_dataset/edges_all/edges.parquet",
            snapshot / "gnn_dataset/labels/targets_global.parquet",
            snapshot / "addr_labels_balanced.csv.zst",
            zip_path,
            label_path / "CE.tsv",
        )
        if not path.is_file()
    ]
    if missing:
        raise FileNotFoundError(
            "Required thesis inputs are missing: "
            + ", ".join(str(path) for path in missing)
        )

    _status(
        f"Starting protocol-v{PROTOCOL_VERSION} run {base_run_id}; "
        f"threads={allocated}, mode={execution_mode}, replication={replication_index}"
    )
    _status(
        "Initial reference estimate: approximately 5-7 hours on the tested 20-core server; "
        f"this run uses {allocated} threads and a measured total ETA will replace "
        "the range after preparation"
    )
    hf_root, realcats_root = _prepare_roots(
        result_path,
        base_run_id,
        snapshot,
        zip_path,
        label_path,
        resume,
        replication_index,
        exclusion_roots,
    )

    concurrent_report = preflight_runs(
        [hf_root, realcats_root],
        concurrent=True,
    )
    if execution_mode == "parallel":
        assert_preflight(concurrent_report)
        chosen_mode = "parallel"
    elif execution_mode == "sequential":
        assert_preflight(preflight_runs([hf_root, realcats_root], concurrent=False))
        chosen_mode = "sequential"
    elif concurrent_report["admissible"] and allocated >= 2:
        chosen_mode = "parallel"
    else:
        sequential_report = preflight_runs(
            [hf_root, realcats_root],
            concurrent=False,
        )
        assert_preflight(sequential_report)
        chosen_mode = "sequential"
        _status(
            "Concurrent RAM/disk admission failed; continuing sequentially with "
            "the full CPU allocation"
        )

    states = (execution_state(hf_root), execution_state(realcats_root))
    _status(
        f"Resource admission passed; execution={chosen_mode}, states={states}"
    )
    from src.experiments.progress import (
        build_execution_progress_plan,
        workflow_progress_display,
    )

    plans = {
        hf_root: build_execution_progress_plan(hf_root, read_locked_protocol(hf_root)[1]),
        realcats_root: build_execution_progress_plan(
            realcats_root, read_locked_protocol(realcats_root)[1]
        ),
    }
    progress_environment = {
        "THESIS_HIDE_ROOT_TOTAL": os.environ.get("THESIS_HIDE_ROOT_TOTAL"),
        "THESIS_PROGRESS_POSITION": os.environ.get("THESIS_PROGRESS_POSITION"),
    }
    os.environ["THESIS_HIDE_ROOT_TOTAL"] = "1"
    os.environ["THESIS_PROGRESS_POSITION"] = "1"
    try:
        with workflow_progress_display(plans):
            if chosen_mode == "parallel" and states in {
                ("fresh", "fresh"),
                ("interrupted", "interrupted"),
                ("interrupted_initialization", "interrupted_initialization"),
            }:
                from src.experiments.coordination import execute_parallel

                parallel_resume = states != ("fresh", "fresh")
                execute_parallel(
                    hf_root,
                    realcats_root,
                    resume=parallel_resume,
                    host_lock_path=host_lock_path,
                    total_threads=(
                        allocated
                        if not parallel_resume or threads is not None
                        else None
                    ),
                )
            else:
                _execute_one(hf_root, allocated, host_lock_path)
                _execute_one(realcats_root, allocated, host_lock_path)

            _status("Verifying both result roots as thesis-citable")
            hf_verification = verify_run(hf_root, require_citable=True)
            realcats_verification = verify_run(
                realcats_root, require_citable=True
            )
    finally:
        for name, value in progress_environment.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
    locked_thread_allocations = (
        int(hf_verification.get("thread_allocation", 0)),
        int(realcats_verification.get("thread_allocation", 0)),
    )
    actual_thread_budget = (
        sum(locked_thread_allocations)
        if chosen_mode == "parallel"
        else max(locked_thread_allocations)
    ) or allocated
    summary = {
        "schema_version": 1,
        "protocol_version": PROTOCOL_VERSION,
        "run_id": base_run_id,
        "status": "complete",
        "execution_mode": chosen_mode,
        "thread_budget": int(actual_thread_budget),
        "replication_index": int(replication_index),
        "excluded_hf_runs": [str(path) for path in exclusion_roots],
        "gpu_used": False,
        "run_roots": {
            "huggingface": str(hf_root.resolve()),
            "realcats": str(realcats_root.resolve()),
        },
        "verification": {
            "huggingface": hf_verification,
            "realcats": realcats_verification,
        },
    }
    _status(
        "All four research questions completed and passed citable verification"
    )
    return json.loads(json.dumps(summary))
