"""Data-dependent preparation stage that locks cohorts and folds before evaluation."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable
from contextlib import nullcontext
import json

import pandas as pd
from sklearn.model_selection import StratifiedKFold

from src.experiments.artifacts import (
    create_run_paths,
    resume_run_paths,
    runtime_manifest,
    sha256_file,
    write_json_exclusive,
    write_locked_protocol,
    write_parquet_exclusive,
)
from src.experiments.constants import (
    CV_SEEDS,
    DEFAULT_RF_PARAMS,
    DEFAULT_XGB_PARAMS,
    DELTAS,
    EVENT_EXECUTION,
    HF_COHORT_ELIGIBILITY,
    MULTICLASS_LOGISTIC_PARAMS,
    N_BOOTSTRAPS,
    N_PERMUTATIONS,
    PROTOCOL_VERSION,
    RANDOM_SEED,
)
from src.experiments.datasets import (
    count_target_activity,
    discover_hf_snapshot,
    extract_target_edges,
    load_blte_for_experiment,
    load_hf_targets,
    select_dual_hf_cohorts,
)
from src.experiments.realcats import (
    convert_realcats_zip,
    load_criminal_labels,
    select_realcats_targets,
)
from src.experiments.splits import (
    grouped_outer_split,
    repeated_group_folds,
    stratified_outer_split,
)
from src.experiments.telemetry import RunTelemetry


VALID_RQS = {"rq1", "rq2", "rq3", "blte", "validation"}


def _relative(path: Path, root: Path) -> str:
    return str(path.relative_to(root))


def _prepare_hf(
    paths, snapshot: Path, telemetry: RunTelemetry | None = None
) -> dict:
    edge_source = snapshot / "gnn_dataset/edges_all/edges.parquet"
    label_source = snapshot / "gnn_dataset/labels/targets_global.parquet"
    labels = pd.read_parquet(label_source, columns=["node_id"])
    activity_context = (
        telemetry.step("prepare.hf.activity_scan", targets=len(labels))
        if telemetry
        else nullcontext()
    )
    with activity_context:
        activity = count_target_activity(
            edge_source, labels["node_id"], telemetry=telemetry
        )
    cohort_context = (
        telemetry.step("prepare.hf.cohort_and_splits")
        if telemetry
        else nullcontext()
    )
    with cohort_context:
        all_targets = load_hf_targets(snapshot, activity)
        selected, attrition = select_dual_hf_cohorts(all_targets)

        split_frames: list[pd.DataFrame] = []
        fold_frames: list[pd.DataFrame] = []
        for cohort, cohort_targets in selected.groupby("cohort", sort=True):
            cohort_targets = cohort_targets.copy()
            cohort_targets["split"] = grouped_outer_split(cohort_targets).values
            split_frames.append(cohort_targets)
            development = cohort_targets.loc[cohort_targets["split"] == "development"]
            folds = repeated_group_folds(development, seeds=CV_SEEDS)
            folds["cohort"] = cohort
            fold_frames.append(folds)
        selected = pd.concat(split_frames, ignore_index=True)
        folds = pd.concat(fold_frames, ignore_index=True)

    target_path = paths.data / "hf_targets.parquet"
    fold_path = paths.data / "hf_cv_folds.parquet"
    event_path = paths.data / "hf_incident_edges.parquet"
    write_parquet_exclusive(target_path, selected)
    write_parquet_exclusive(fold_path, folds)
    extraction_context = (
        telemetry.step("prepare.hf.edge_extraction", targets=len(selected))
        if telemetry
        else nullcontext()
    )
    with extraction_context:
        extraction = extract_target_edges(
            edge_source, selected["node_id"], event_path, telemetry=telemetry
        )
    hash_context = (
        telemetry.step("prepare.hf.hash_artifacts")
        if telemetry
        else nullcontext()
    )
    with hash_context:
        source_edge_hash = sha256_file(edge_source)
        source_label_hash = sha256_file(label_source)
        description_source = snapshot / "addr_labels_balanced.csv.zst"
        description_hash = sha256_file(description_source)
        target_hash = sha256_file(target_path)
        fold_hash = sha256_file(fold_path)
        event_hash = sha256_file(event_path)
    return {
        "snapshot_revision": snapshot.name,
        "sources": {
            "edges": {"path": str(edge_source), "sha256": source_edge_hash},
            "labels": {"path": str(label_source), "sha256": source_label_hash},
            "descriptions": {
                "path": str(description_source),
                "sha256": description_hash,
            },
        },
        "artifacts": {
            "targets": _relative(target_path, paths.root),
            "folds": _relative(fold_path, paths.root),
            "events": _relative(event_path, paths.root),
            "targets_sha256": target_hash,
            "folds_sha256": fold_hash,
            "events_sha256": event_hash,
        },
        "attrition": attrition,
        "edge_extraction": extraction,
    }


def _rq3_cv_assignments(targets: pd.DataFrame) -> pd.DataFrame:
    development = targets.loc[targets["split"] == "development"].reset_index(drop=True)
    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    records = []
    for fold, (_, validation) in enumerate(
        splitter.split(development, development["type_label"])
    ):
        records.extend(
            {"address": str(development.iloc[index]["address"]), "fold": fold}
            for index in validation
        )
    return pd.DataFrame.from_records(records)


def _write_or_verify_parquet(path: Path, frame: pd.DataFrame) -> None:
    if not path.exists():
        write_parquet_exclusive(path, frame)
        return
    existing = pd.read_parquet(path)
    pd.testing.assert_frame_equal(existing, frame.reset_index(drop=True), check_dtype=True)


def _prepare_realcats(
    paths,
    zip_path: Path,
    label_dir: Path,
    resume: bool = False,
    source_hashes: dict[str, str] | None = None,
    telemetry: RunTelemetry | None = None,
) -> dict:
    layer_dir = paths.data / "realcats_layers"
    conversion_context = (
        telemetry.step("prepare.realcats.conversion", resumed=resume)
        if telemetry
        else nullcontext()
    )
    with conversion_context:
        conversion = convert_realcats_zip(
            zip_path,
            label_dir,
            layer_dir,
            resume=resume,
            telemetry=telemetry,
        )
    layer_paths = {layer: Path(path) for layer, path in conversion["layer_paths"].items()}
    target_context = (
        telemetry.step("prepare.realcats.targets_and_splits")
        if telemetry
        else nullcontext()
    )
    with target_context:
        labels = load_criminal_labels(label_dir)
        targets, attrition = select_realcats_targets(labels, layer_paths)
        inferential = targets.loc[targets["inference_eligible"]].copy()
        inferential["split"] = stratified_outer_split(inferential).values
        targets = targets.merge(
            inferential[["address", "split"]],
            on="address",
            how="left",
            validate="one_to_one",
        )
        folds = _rq3_cv_assignments(inferential)

    target_path = paths.data / "realcats_targets.parquet"
    fold_path = paths.data / "realcats_cv_folds.parquet"
    _write_or_verify_parquet(target_path, targets)
    _write_or_verify_parquet(fold_path, folds)
    return {
        "sources": {
            "transactions": {
                "path": str(zip_path),
                "sha256": (source_hashes or {}).get("transactions")
                or sha256_file(zip_path),
            },
            "labels": {
                "path": str(label_dir / "CE.tsv"),
                "sha256": (source_hashes or {}).get("labels")
                or sha256_file(label_dir / "CE.tsv"),
            },
        },
        "artifacts": {
            "targets": _relative(target_path, paths.root),
            "folds": _relative(fold_path, paths.root),
            "layers": {
                layer: _relative(path, paths.root) for layer, path in layer_paths.items()
            },
            "targets_sha256": sha256_file(target_path),
            "folds_sha256": sha256_file(fold_path),
            "layers_sha256": {layer: sha256_file(path) for layer, path in layer_paths.items()},
        },
        "attrition": attrition,
        "conversion": conversion,
    }


def _prepare_blte(
    paths, dataset_dir: Path, telemetry: RunTelemetry | None = None
) -> dict:
    context = telemetry.step("prepare.blte") if telemetry else nullcontext()
    with context:
        events, targets = load_blte_for_experiment(dataset_dir)
    event_path = paths.data / "blte_events.parquet"
    target_path = paths.data / "blte_targets.parquet"
    write_parquet_exclusive(event_path, events)
    write_parquet_exclusive(target_path, targets)
    counts = targets["label"].value_counts()
    return {
        "eligible": bool(len(counts) == 2 and counts.min() >= 100),
        "active_label_counts": {str(key): int(value) for key, value in counts.items()},
        "sources": {
            "transactions": {
                "path": str(dataset_dir / "Dataset.zip"),
                "sha256": sha256_file(dataset_dir / "Dataset.zip"),
            },
            "labels": {
                "path": str(dataset_dir / "Featuers_Extraction.csv"),
                "sha256": sha256_file(dataset_dir / "Featuers_Extraction.csv"),
            },
        },
        "artifacts": {
            "events": _relative(event_path, paths.root),
            "targets": _relative(target_path, paths.root),
            "events_sha256": sha256_file(event_path),
            "targets_sha256": sha256_file(target_path),
        },
    }


def prepare_experiment(
    results_root: str | Path,
    run_id: str,
    rqs: Iterable[str],
    hf_snapshot: str | Path | None = None,
    realcats_zip: str | Path = "data/raw/realcats_transactions/real-cats.zip",
    realcats_labels: str | Path = "data/raw/realcats_dataset",
    blte_dir: str | Path = "data/raw/ethereum_dataset_labelled",
    permutations: int = N_PERMUTATIONS,
    bootstraps: int = N_BOOTSTRAPS,
    resume: bool = False,
) -> Path:
    selected_rqs = sorted(set(rqs))
    unknown = set(selected_rqs) - VALID_RQS
    if unknown:
        raise ValueError(f"Unknown research questions: {sorted(unknown)}")
    if resume and selected_rqs != ["rq3"]:
        raise ValueError("Checkpoint resume is supported only for an RQ3-only preparation")
    preparation_runtime = runtime_manifest()
    paths = (
        resume_run_paths(results_root, run_id)
        if resume
        else create_run_paths(results_root, run_id)
    )
    telemetry = RunTelemetry(paths.root)
    telemetry.emit(
        "prepare",
        "started",
        run_id=run_id,
        research_questions=selected_rqs,
        resumed=resume,
    )
    realcats_zip_path = Path(realcats_zip)
    realcats_label_dir = Path(realcats_labels)
    realcats_source_hashes = None
    if "rq3" in selected_rqs:
        with telemetry.step("prepare.realcats.hash_sources"):
            realcats_source_hashes = {
                "transactions": sha256_file(realcats_zip_path),
                "labels": sha256_file(realcats_label_dir / "CE.tsv"),
            }
    preparation_definition = {
        "protocol_version": PROTOCOL_VERSION,
        "run_id": run_id,
        "research_questions": selected_rqs,
        "permutations": int(permutations),
        "bootstraps": int(bootstraps),
        "inputs": {
            "hf_snapshot": str(Path(hf_snapshot).resolve()) if hf_snapshot else None,
            "realcats_zip": str(realcats_zip_path.resolve()),
            "realcats_labels": str(realcats_label_dir.resolve()),
            "blte_dir": str(Path(blte_dir).resolve()),
            "realcats_source_sha256": realcats_source_hashes,
        },
        "runtime": preparation_runtime,
    }
    preparation_state = paths.root / "preparation_state.json"
    if resume:
        if not preparation_state.exists():
            raise RuntimeError(
                "This incomplete run predates checkpoint locking and cannot be resumed "
                "safely; prepare a new run ID"
            )
        expected = json.loads(preparation_state.read_text(encoding="utf-8"))
        if expected != preparation_definition:
            raise RuntimeError(
                "Resume arguments, inputs, dependencies, or source files differ from "
                "the locked preparation state"
            )
    else:
        write_json_exclusive(preparation_state, preparation_definition)
    datasets: dict = {}
    if {"rq1", "rq2"} & set(selected_rqs):
        snapshot = Path(hf_snapshot) if hf_snapshot else discover_hf_snapshot()
        datasets["huggingface"] = _prepare_hf(paths, snapshot, telemetry=telemetry)
    if "rq3" in selected_rqs:
        datasets["realcats"] = _prepare_realcats(
            paths,
            realcats_zip_path,
            realcats_label_dir,
            resume=resume,
            source_hashes=realcats_source_hashes,
            telemetry=telemetry,
        )
    if "blte" in selected_rqs:
        datasets["blte"] = _prepare_blte(paths, Path(blte_dir), telemetry=telemetry)

    protocol = {
        "protocol_version": PROTOCOL_VERSION,
        "run_id": run_id,
        "research_questions": selected_rqs,
        "retrospective_classification_only": True,
        "primary_metric": "average_precision",
        "alpha": 0.05,
        "deltas": list(DELTAS),
        "random_seed": RANDOM_SEED,
        "cv_seeds": list(CV_SEEDS),
        "permutations": int(permutations),
        "bootstraps": int(bootstraps),
        "ordering_variants": [
            "canonical", "reverse", "random_0", "random_1", "random_2", "random_3", "random_4"
        ],
        "event_execution": EVENT_EXECUTION,
        "hf_cohort_eligibility": HF_COHORT_ELIGIBILITY,
        "model_parameters": {
            "random_forest": DEFAULT_RF_PARAMS,
            "xgboost": DEFAULT_XGB_PARAMS,
            "multiclass_logistic": MULTICLASS_LOGISTIC_PARAMS,
            "multiclass_standard_scaler": {"with_mean": True, "with_std": True},
        },
        "datasets": datasets,
        "runtime": preparation_runtime,
        "preparation_state_sha256": sha256_file(preparation_state),
    }
    protocol_context = telemetry.step("prepare.lock_protocol")
    with protocol_context:
        write_locked_protocol(paths, protocol)
    telemetry.emit("prepare", "completed", run_root=str(paths.root))
    return paths.root
