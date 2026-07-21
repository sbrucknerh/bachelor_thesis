#!/usr/bin/env python3
"""Prepare and execute the locked Results-chapter experiments."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.experiments.prepare import VALID_RQS


def _research_questions(values: list[str]) -> list[str]:
    return sorted(VALID_RQS) if "all" in values else values


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare", help="Create data artifacts and lock protocol.json")
    prepare.add_argument("--run-id", required=True)
    prepare.add_argument("--results-root", default="results/experiments")
    prepare.add_argument(
        "--rq", nargs="+", default=["all"], choices=["all", *sorted(VALID_RQS)]
    )
    prepare.add_argument("--hf-snapshot")
    prepare.add_argument(
        "--realcats-zip", default="data/raw/realcats_transactions/real-cats.zip"
    )
    prepare.add_argument("--realcats-labels", default="data/raw/realcats_dataset")
    prepare.add_argument("--blte-dir", default="data/raw/ethereum_dataset_labelled")
    prepare.add_argument("--permutations", type=int, default=10_000)
    prepare.add_argument("--bootstraps", type=int, default=10_000)
    prepare.add_argument(
        "--resume",
        action="store_true",
        help="Resume an interrupted RQ3-only preparation before protocol locking",
    )

    execute = subparsers.add_parser("execute", help="Verify and execute a locked run")
    execute.add_argument("run_root")

    smoke = subparsers.add_parser(
        "smoke",
        help="Run bounded, non-citable validation across every experiment path",
    )
    smoke.add_argument("--run-id", required=True)
    smoke.add_argument("--results-root", default="results/smoke")
    smoke.add_argument("--hf-snapshot")
    smoke.add_argument(
        "--realcats-zip", default="data/raw/realcats_transactions/real-cats.zip"
    )
    smoke.add_argument("--realcats-labels", default="data/raw/realcats_dataset")
    smoke.add_argument("--blte-dir", default="data/raw/ethereum_dataset_labelled")
    smoke.add_argument(
        "--sla-minutes",
        type=float,
        default=60.0,
        help="Report a failed runtime SLA after completion; does not kill native work",
    )
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
            resume=args.resume,
        )
        print(f"Prepared and locked: {root}")
    elif args.command == "execute":
        from src.experiments.execute import execute_experiment
        from src.experiments.resources import ResourceLimitError

        try:
            manifest = execute_experiment(args.run_root)
        except ResourceLimitError as error:
            raise SystemExit(f"Execution blocked before metric generation: {error}") from error
        print(f"Completed: {args.run_root} ({manifest['status']})")
    else:
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
            raise SystemExit(
                f"Smoke execution blocked by the resource preflight: {error}"
            ) from error
        print(
            f"Smoke completed: {root} "
            f"({summary['elapsed_seconds']:.1f}s, sla_met={summary['sla_met']})"
        )
        if not summary["sla_met"]:
            raise SystemExit(2)


if __name__ == "__main__":
    main()
