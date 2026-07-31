"""One-command preparation, execution, and verification for thesis servers."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import sys

from src.experiments.constants import PROTOCOL_VERSION, REALCATS_RESEARCH_QUESTIONS


def _status(message: str) -> None:
    timestamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    print(f"[{timestamp}] THESIS   {message}", file=sys.stderr, flush=True)


def _default_run_id() -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return f"thesis_{stamp}_v{PROTOCOL_VERSION}"


def _validate_locked_root(root: Path, expected_rqs: set[str]) -> None:
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


def _prepare_roots(
    results_root: Path,
    base_run_id: str,
    hf_snapshot: Path,
    realcats_zip: Path,
    realcats_labels: Path,
    resume: bool,
) -> tuple[Path, Path]:
    from src.experiments.prepare import prepare_experiment

    hf_id = f"{base_run_id}_hf"
    realcats_id = f"{base_run_id}_realcats"
    hf_root = results_root / hf_id
    realcats_root = results_root / realcats_id

    if (hf_root / "protocol.json").is_file():
        if not resume:
            raise FileExistsError(
                f"{hf_root} already exists; use --resume to continue this run"
            )
        _validate_locked_root(hf_root, {"rq1", "rq2"})
        _status(f"HF preparation already locked: {hf_root}")
    elif hf_root.exists():
        raise RuntimeError(
            "Hugging Face preparation was interrupted before protocol locking and "
            "is not resumable. Use a new --run-id; the full HF scan is intentionally "
            "never inferred from partial artifacts."
        )
    else:
        _status("Preparing Hugging Face RQ1/RQ2 root")
        prepare_experiment(
            results_root,
            hf_id,
            ["rq1", "rq2"],
            hf_snapshot=hf_snapshot,
            profile="thesis",
        )

    if (realcats_root / "protocol.json").is_file():
        if not resume:
            raise FileExistsError(
                f"{realcats_root} already exists; use --resume to continue this run"
            )
        _validate_locked_root(realcats_root, set(REALCATS_RESEARCH_QUESTIONS))
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
        )

    return hf_root, realcats_root


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
) -> dict:
    """Run all primary thesis experiments and return verified run locations."""
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
        f"threads={allocated}, mode={execution_mode}"
    )
    hf_root, realcats_root = _prepare_roots(
        result_path,
        base_run_id,
        snapshot,
        zip_path,
        label_path,
        resume,
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
            total_threads=allocated if not parallel_resume or threads is not None else None,
        )
    else:
        _execute_one(hf_root, allocated, host_lock_path)
        _execute_one(realcats_root, allocated, host_lock_path)

    _status("Verifying both result roots as thesis-citable")
    hf_verification = verify_run(hf_root, require_citable=True)
    realcats_verification = verify_run(realcats_root, require_citable=True)
    summary = {
        "schema_version": 1,
        "protocol_version": PROTOCOL_VERSION,
        "run_id": base_run_id,
        "status": "complete",
        "execution_mode": chosen_mode,
        "thread_budget": allocated,
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
