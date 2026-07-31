#!/usr/bin/env python3
"""Prepare, execute, resume, coordinate, and verify Results experiments."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

VALID_RQS = {"rq1", "rq2", "rq3", "rq4", "blte", "validation"}


def _research_questions(values: list[str]) -> list[str]:
    return ["rq1", "rq2", "rq3", "rq4"] if "all" in values else values


def _set_thread_environment(threads: int) -> None:
    value = str(max(1, int(threads)))
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare", help="Create data artifacts and lock protocol.json")
    prepare.add_argument("--run-id", required=True)
    prepare.add_argument("--results-root", default="results/experiments")
    prepare.add_argument("--profile", choices=["thesis", "development"], default="thesis")
    prepare.add_argument(
        "--rq", nargs="+", default=["all"], choices=["all", *sorted(VALID_RQS)]
    )
    prepare.add_argument("--hf-snapshot")
    prepare.add_argument("--realcats-zip", default="data/raw/realcats_transactions/real-cats.zip")
    prepare.add_argument("--realcats-labels", default="data/raw/realcats_dataset")
    prepare.add_argument("--blte-dir", default="data/raw/ethereum_dataset_labelled")
    prepare.add_argument("--permutations", type=int)
    prepare.add_argument("--bootstraps", type=int)
    prepare.add_argument(
        "--resume",
        action="store_true",
        help="Resume interrupted combined RQ3/RQ4 preparation before protocol locking",
    )

    execute = subparsers.add_parser("execute", help="Execute or resume one locked v5 run")
    execute.add_argument("run_root")
    execute.add_argument("--resume", action="store_true")
    execute.add_argument("--threads", type=int)
    execute.add_argument("--host-lock")
    execute.add_argument("--_coordinator-id", help=argparse.SUPPRESS)
    execute.add_argument("--_coordinator-definition-sha256", help=argparse.SUPPRESS)
    execute.add_argument("--_host-lock-fd", type=int, help=argparse.SUPPRESS)

    parallel = subparsers.add_parser(
        "execute-parallel", help="Coordinate one HF root and one Real-CATS root"
    )
    parallel.add_argument("hf_run_root")
    parallel.add_argument("realcats_run_root")
    parallel.add_argument("--resume", action="store_true")
    parallel.add_argument(
        "--threads",
        type=int,
        help="Total CPU-thread budget split between the two workers",
    )
    parallel.add_argument("--host-lock")

    thesis = subparsers.add_parser(
        "run-thesis",
        help="Prepare, execute, and verify all four thesis RQs in one command",
    )
    thesis.add_argument("--run-id")
    thesis.add_argument("--results-root", default="results/experiments")
    thesis.add_argument("--threads", type=int)
    thesis.add_argument(
        "--execution-mode",
        choices=["auto", "parallel", "sequential"],
        default="auto",
    )
    thesis.add_argument("--hf-snapshot")
    thesis.add_argument(
        "--realcats-zip",
        default="data/raw/realcats_transactions/real-cats.zip",
    )
    thesis.add_argument(
        "--realcats-labels",
        default="data/raw/realcats_dataset",
    )
    thesis.add_argument("--host-lock")
    thesis.add_argument("--resume", action="store_true")

    preflight = subparsers.add_parser("preflight", help="Read-only resource admission")
    preflight.add_argument("run_roots", nargs="+")

    verify = subparsers.add_parser("verify", help="Read-only integrity verification")
    verify.add_argument("run_root")
    verify.add_argument("--require-citable", action="store_true")

    smoke = subparsers.add_parser(
        "smoke", help="Run bounded, non-citable validation across every path"
    )
    smoke.add_argument("--run-id", required=True)
    smoke.add_argument("--results-root", default="results/smoke")
    smoke.add_argument("--hf-snapshot")
    smoke.add_argument("--realcats-zip", default="data/raw/realcats_transactions/real-cats.zip")
    smoke.add_argument("--realcats-labels", default="data/raw/realcats_dataset")
    smoke.add_argument("--blte-dir", default="data/raw/ethereum_dataset_labelled")
    smoke.add_argument("--sla-minutes", type=float, default=60.0)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "prepare":
        from src.experiments.prepare import prepare_experiment

        root = prepare_experiment(
            results_root=args.results_root,
            run_id=args.run_id,
            rqs=_research_questions(args.rq),
            hf_snapshot=args.hf_snapshot,
            realcats_zip=args.realcats_zip,
            realcats_labels=args.realcats_labels,
            blte_dir=args.blte_dir,
            permutations=args.permutations,
            bootstraps=args.bootstraps,
            profile=args.profile,
            resume=args.resume,
        )
        print(f"Prepared and locked: {root}")
        return

    if args.command == "preflight":
        from src.experiments.resources import preflight_runs

        report = preflight_runs(args.run_roots, concurrent=len(args.run_roots) > 1)
        print(json.dumps(report, indent=2, sort_keys=True))
        if not report["admissible"]:
            raise SystemExit(2)
        return

    if args.command == "verify":
        from src.experiments.verify import VerificationError, verify_run

        try:
            report = verify_run(args.run_root, require_citable=args.require_citable)
        except (VerificationError, OSError, RuntimeError, ValueError) as error:
            raise SystemExit(f"Verification failed: {error}") from error
        print(json.dumps(report, indent=2, sort_keys=True))
        return

    if args.command == "execute-parallel":
        from src.experiments.coordination import execute_parallel

        outcome = execute_parallel(
            args.hf_run_root,
            args.realcats_run_root,
            resume=args.resume,
            host_lock_path=args.host_lock,
            total_threads=args.threads,
        )
        print(json.dumps(outcome, indent=2, sort_keys=True))
        return

    if args.command == "run-thesis":
        requested_threads = args.threads or max(1, os.cpu_count() or 1)
        _set_thread_environment(requested_threads)
        from src.experiments.server_workflow import run_thesis_workflow

        outcome = run_thesis_workflow(
            results_root=args.results_root,
            run_id=args.run_id,
            hf_snapshot=args.hf_snapshot,
            realcats_zip=args.realcats_zip,
            realcats_labels=args.realcats_labels,
            threads=args.threads,
            execution_mode=args.execution_mode,
            host_lock_path=args.host_lock,
            resume=args.resume,
        )
        print(json.dumps(outcome, indent=2, sort_keys=True))
        return

    if args.command == "execute":
        requested_threads = args.threads or max(1, os.cpu_count() or 1)
        _set_thread_environment(requested_threads)
        from src.experiments.execute import execute_experiment
        from src.experiments.resources import ResourceLimitError

        try:
            manifest = execute_experiment(
                args.run_root,
                resume=args.resume,
                threads=args.threads,
                host_lock_path=args.host_lock,
                coordinator_id=args._coordinator_id,
                coordinator_definition_sha256=args._coordinator_definition_sha256,
                authorized_host_lock_fd=args._host_lock_fd,
            )
        except ResourceLimitError as error:
            raise SystemExit(f"Execution blocked before attempt creation: {error}") from error
        print(f"Completed: {args.run_root} ({manifest['status']}, citable={manifest['citable']})")
        return

    _set_thread_environment(max(1, os.cpu_count() or 1))
    from src.experiments.resources import ResourceLimitError
    from src.experiments.smoke import run_smoke_experiment

    if args.sla_minutes <= 0:
        raise SystemExit("--sla-minutes must be positive")
    try:
        root, _, summary = run_smoke_experiment(
            args.results_root,
            args.run_id,
            hf_snapshot=args.hf_snapshot,
            realcats_zip=args.realcats_zip,
            realcats_labels=args.realcats_labels,
            blte_dir=args.blte_dir,
            sla_seconds=args.sla_minutes * 60,
        )
    except ResourceLimitError as error:
        raise SystemExit(f"Smoke execution blocked by resource admission: {error}") from error
    print(
        f"Smoke completed: {root} "
        f"({summary['elapsed_seconds']:.1f}s, sla_met={summary['sla_met']})"
    )
    if not summary["sla_met"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
