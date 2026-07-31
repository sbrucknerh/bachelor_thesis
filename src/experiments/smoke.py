"""Bounded, non-citable end-to-end validation of every experiment path."""

from __future__ import annotations

from pathlib import Path
from time import monotonic
from typing import Callable
import zipfile

import pandas as pd

from src.experiments.artifacts import (
    create_run_paths,
    protocol_digest,
    runtime_manifest,
    sha256_file,
    write_json_exclusive,
    write_locked_protocol,
    write_parquet_exclusive,
)
from src.experiments.constants import (
    DEFAULT_RF_PARAMS,
    DEFAULT_XGB_PARAMS,
    DELTA_SELECTION,
    EVENT_EXECUTION,
    HF_COHORT_ELIGIBILITY,
    MULTICLASS_LOGISTIC_PARAMS,
    PROTOCOL_VERSION,
    RANDOM_SEED,
    REALCATS_CATEGORY_ELIGIBILITY,
    REALCATS_LAYER_FILES,
    RESOURCE_CONSTANTS,
)
from src.experiments.datasets import (
    count_target_activity,
    discover_hf_snapshot,
    extract_target_edges,
    load_hf_targets,
    select_dual_hf_cohorts,
)
from src.experiments.execute import execute_experiment
from src.experiments.prepare import _prepare_blte, _rq3_cv_assignments
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


SMOKE_DELTAS = (300, 3_600, 604_800)
SMOKE_ORDER_VARIANTS = ("canonical", "reverse", "random_0")
SMOKE_HF_SOURCE_ROWS = 5_000_000
SMOKE_HF_PREVALENCE_SIZE = 200
SMOKE_HF_CONTROLLED_PER_STRATUM = 5
SMOKE_REALCATS_LABELS = (
    "Phishing Scam",
    "Impersonation Scam",
    "Hack Scam",
)
SMOKE_REALCATS_ADDRESSES_PER_LABEL = 20
SMOKE_REALCATS_MINIMUM_CLASS_SIZE = 10
SMOKE_REALCATS_MIN_ADDRESS_BYTES = 1_024
SMOKE_REALCATS_MAX_ADDRESS_BYTES = 2 * 1024 * 1024
SMOKE_RESAMPLES = 100


def _relative(path: Path, root: Path) -> str:
    return str(path.relative_to(root))


def _stat_identity(path: Path) -> dict[str, int | str]:
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "size_bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def _smoke_configuration(sla_seconds: float) -> dict:
    return {
        "sla_seconds": float(sla_seconds),
        "hf_source_rows": SMOKE_HF_SOURCE_ROWS,
        "hf_prevalence_size": SMOKE_HF_PREVALENCE_SIZE,
        "hf_controlled_per_stratum": SMOKE_HF_CONTROLLED_PER_STRATUM,
        "realcats_labels": list(SMOKE_REALCATS_LABELS),
        "realcats_addresses_per_label": SMOKE_REALCATS_ADDRESSES_PER_LABEL,
        "realcats_minimum_class_size": SMOKE_REALCATS_MINIMUM_CLASS_SIZE,
        "realcats_address_bytes": [
            SMOKE_REALCATS_MIN_ADDRESS_BYTES,
            SMOKE_REALCATS_MAX_ADDRESS_BYTES,
        ],
        "deltas": list(SMOKE_DELTAS),
        "ordering_variants": list(SMOKE_ORDER_VARIANTS),
        "cv_seeds": [RANDOM_SEED],
        "permutations": SMOKE_RESAMPLES,
        "bootstraps": SMOKE_RESAMPLES,
    }


def _prepare_smoke_hf(paths, snapshot: Path, telemetry: RunTelemetry) -> dict:
    edge_source = snapshot / "gnn_dataset/edges_all/edges.parquet"
    label_source = snapshot / "gnn_dataset/labels/targets_global.parquet"
    description_source = snapshot / "addr_labels_balanced.csv.zst"
    labels = pd.read_parquet(label_source, columns=["node_id"])
    with telemetry.step(
        "smoke.prepare.hf.activity_scan",
        rows_limit=SMOKE_HF_SOURCE_ROWS,
        targets=len(labels),
    ):
        activity = count_target_activity(
            edge_source,
            labels["node_id"],
            telemetry=telemetry,
            max_rows=SMOKE_HF_SOURCE_ROWS,
        )

    with telemetry.step("smoke.prepare.hf.cohorts_and_splits"):
        all_targets = load_hf_targets(snapshot, activity)
        selected, attrition = select_dual_hf_cohorts(
            all_targets,
            prevalence_size=SMOKE_HF_PREVALENCE_SIZE,
            controlled_per_stratum=SMOKE_HF_CONTROLLED_PER_STRATUM,
        )
        for cohort, cohort_targets in selected.groupby("cohort", sort=True):
            if cohort_targets["label"].nunique() != 2:
                raise RuntimeError(
                    f"HF smoke cohort {cohort!r} does not contain both labels"
                )
        selected["experiment_stratum"] = (
            selected["cohort"].astype(str) + ":" + selected["stratum"].astype(str)
        )
        selected["split"] = grouped_outer_split(
            selected, stratum_column="experiment_stratum"
        ).values
        development = selected.loc[selected["split"] == "development"]
        folds = repeated_group_folds(
            development,
            seeds=(RANDOM_SEED,),
            stratum_column="experiment_stratum",
        )
        folds = folds.merge(
            development[["address", "cohort"]],
            on="address",
            how="left",
            validate="many_to_one",
        )

    target_path = paths.data / "hf_targets.parquet"
    fold_path = paths.data / "hf_cv_folds.parquet"
    event_path = paths.data / "hf_incident_edges.parquet"
    write_parquet_exclusive(target_path, selected)
    write_parquet_exclusive(fold_path, folds)
    with telemetry.step(
        "smoke.prepare.hf.edge_extraction",
        rows_limit=SMOKE_HF_SOURCE_ROWS,
        targets=len(selected),
    ):
        extraction = extract_target_edges(
            edge_source,
            selected["node_id"],
            event_path,
            telemetry=telemetry,
            max_rows=SMOKE_HF_SOURCE_ROWS,
        )

    attrition["smoke_source_rows"] = SMOKE_HF_SOURCE_ROWS
    endpoints = pd.read_parquet(
        event_path, columns=["from_address", "to_address"]
    )
    approximate_nodes = int(
        pd.unique(
            pd.concat(
                [endpoints["from_address"], endpoints["to_address"]],
                ignore_index=True,
            )
        ).size
    )
    return {
        "snapshot_revision": snapshot.name,
        "smoke_bounded": True,
        "sources": {
            "edges": {
                **_stat_identity(edge_source),
                "verification": "bounded_prefix_and_prepared_artifact_sha256",
                "rows_considered": SMOKE_HF_SOURCE_ROWS,
            },
            "labels": {
                "path": str(label_source.resolve()),
                "sha256": sha256_file(label_source),
            },
            "descriptions": {
                "path": str(description_source.resolve()),
                "sha256": sha256_file(description_source),
            },
        },
        "artifacts": {
            "targets": _relative(target_path, paths.root),
            "folds": _relative(fold_path, paths.root),
            "events": _relative(event_path, paths.root),
            "targets_sha256": sha256_file(target_path),
            "folds_sha256": sha256_file(fold_path),
            "events_sha256": sha256_file(event_path),
        },
        "attrition": attrition,
        "edge_extraction": extraction,
        "resource_metadata": {
            "approximate_distinct_nodes": approximate_nodes,
            "target_count": int(len(selected)),
        },
    }


def select_realcats_smoke_addresses(
    zip_path: str | Path,
    label_dir: str | Path,
    labels_to_use: tuple[str, ...] = SMOKE_REALCATS_LABELS,
    addresses_per_label: int = SMOKE_REALCATS_ADDRESSES_PER_LABEL,
    minimum_bytes: int = SMOKE_REALCATS_MIN_ADDRESS_BYTES,
    maximum_bytes: int = SMOKE_REALCATS_MAX_ADDRESS_BYTES,
) -> tuple[pd.DataFrame, dict]:
    """Choose small deterministic real histories while covering every layer."""
    labels = load_criminal_labels(label_dir)
    labels = labels.loc[labels["type_label"].isin(labels_to_use)].copy()
    file_to_layer = {filename: layer for layer, filename in REALCATS_LAYER_FILES.items()}
    archive_entries: dict[str, dict[str, dict]] = {}
    with zipfile.ZipFile(zip_path) as archive:
        for info in archive.infolist():
            parts = info.filename.split("/")
            if len(parts) != 4:
                continue
            layer = file_to_layer.get(parts[3])
            if layer is None:
                continue
            address = parts[2].lower()
            archive_entries.setdefault(address, {})[layer] = {
                "name": info.filename,
                "file_size": int(info.file_size),
                "compress_size": int(info.compress_size),
                "crc": int(info.CRC),
            }

    selected_rows: list[dict] = []
    fingerprint_entries: list[dict] = []
    for label in labels_to_use:
        candidates: list[dict] = []
        for address in labels.loc[labels["type_label"] == label, "address"]:
            layers = archive_entries.get(address, {})
            total_bytes = sum(item["file_size"] for item in layers.values())
            if minimum_bytes <= total_bytes <= maximum_bytes:
                candidates.append(
                    {
                        "address": address,
                        "type_label": label,
                        "source_bytes": total_bytes,
                        "layers": layers,
                    }
                )
        candidates.sort(key=lambda item: (item["source_bytes"], item["address"]))
        if len(candidates) < addresses_per_label:
            raise RuntimeError(
                f"Real-CATS smoke selection found only {len(candidates)} bounded "
                f"addresses for {label!r}; requires {addresses_per_label}"
            )

        chosen: dict[str, dict] = {}
        for layer in REALCATS_LAYER_FILES:
            layer_candidates = [
                item
                for item in candidates
                if item["layers"].get(layer, {}).get("file_size", 0) > 2
            ]
            if not layer_candidates:
                raise RuntimeError(
                    f"Real-CATS smoke selection has no non-empty {layer} history "
                    f"for {label!r}"
                )
            item = layer_candidates[0]
            chosen[item["address"]] = item
        for item in candidates:
            if len(chosen) >= addresses_per_label:
                break
            chosen[item["address"]] = item
        if len(chosen) != addresses_per_label:
            raise RuntimeError(f"Incomplete Real-CATS smoke selection for {label!r}")

        for item in sorted(chosen.values(), key=lambda value: value["address"]):
            selected_rows.append(
                {
                    "address": item["address"],
                    "type_label": item["type_label"],
                    "source_bytes": item["source_bytes"],
                }
            )
            fingerprint_entries.extend(
                {
                    "address": item["address"],
                    "layer": layer,
                    **metadata,
                }
                for layer, metadata in sorted(item["layers"].items())
            )

    selected = pd.DataFrame.from_records(selected_rows).sort_values(
        ["type_label", "address"]
    ).reset_index(drop=True)
    counts = selected["type_label"].value_counts()
    if set(counts.index) != set(labels_to_use) or not (
        counts == addresses_per_label
    ).all():
        raise RuntimeError("Real-CATS smoke selection is not class balanced")
    audit = {
        "labels": list(labels_to_use),
        "addresses_per_label": addresses_per_label,
        "selected_addresses": len(selected),
        "minimum_address_bytes": minimum_bytes,
        "maximum_address_bytes": maximum_bytes,
        "selected_source_bytes": int(selected["source_bytes"].sum()),
        "selected_archive_entries_digest": protocol_digest(
            {"entries": fingerprint_entries}
        ),
    }
    return selected, audit


def _prepare_smoke_realcats(
    paths,
    zip_path: Path,
    label_dir: Path,
    telemetry: RunTelemetry,
) -> dict:
    with telemetry.step("smoke.prepare.realcats.select_addresses"):
        selection, selection_audit = select_realcats_smoke_addresses(
            zip_path, label_dir
        )
    selection_path = paths.data / "realcats_smoke_selection.parquet"
    write_parquet_exclusive(selection_path, selection)

    layer_dir = paths.data / "realcats_layers"
    with telemetry.step(
        "smoke.prepare.realcats.conversion", addresses=len(selection)
    ):
        conversion = convert_realcats_zip(
            zip_path,
            label_dir,
            layer_dir,
            minimum_free_gib=1,
            selected_addresses=selection["address"],
            telemetry=telemetry,
        )
    layer_paths = {
        layer: Path(path) for layer, path in conversion["layer_paths"].items()
    }
    missing_layers = set(REALCATS_LAYER_FILES) - set(layer_paths)
    if missing_layers:
        raise RuntimeError(
            f"Real-CATS smoke conversion produced no events for layers: "
            f"{sorted(missing_layers)}"
        )

    with telemetry.step("smoke.prepare.realcats.targets_and_splits"):
        labels = load_criminal_labels(label_dir)
        labels = labels.loc[labels["address"].isin(set(selection["address"]))]
        targets, attrition = select_realcats_targets(
            labels,
            layer_paths,
            minimum_class_size=SMOKE_REALCATS_MINIMUM_CLASS_SIZE,
        )
        inferential = targets.loc[targets["inference_eligible"]].copy()
        counts = inferential["type_label"].value_counts()
        if set(counts.index) != set(SMOKE_REALCATS_LABELS):
            raise RuntimeError(
                "Real-CATS smoke conversion did not retain every exact label: "
                f"{counts.to_dict()}"
            )
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
    write_parquet_exclusive(target_path, targets)
    write_parquet_exclusive(fold_path, folds)
    return {
        "smoke_bounded": True,
        "sources": {
            "transactions": {
                **_stat_identity(zip_path),
                "verification": "selected_zip_entries_digest",
                "selected_entries_sha256": selection_audit[
                    "selected_archive_entries_digest"
                ],
            },
            "labels": {
                "path": str((label_dir / "CE.tsv").resolve()),
                "sha256": sha256_file(label_dir / "CE.tsv"),
            },
        },
        "artifacts": {
            "selection": _relative(selection_path, paths.root),
            "targets": _relative(target_path, paths.root),
            "folds": _relative(fold_path, paths.root),
            "layers": {
                layer: _relative(path, paths.root)
                for layer, path in layer_paths.items()
            },
            "selection_sha256": sha256_file(selection_path),
            "targets_sha256": sha256_file(target_path),
            "folds_sha256": sha256_file(fold_path),
            "layers_sha256": {
                layer: sha256_file(path) for layer, path in layer_paths.items()
            },
        },
        "selection": selection_audit,
        "attrition": attrition,
        "conversion": conversion,
        "resource_metadata": {
            "target_count": int(len(inferential)),
            "exact_distinct_nodes_by_layer": {
                layer: int(conversion[f"{layer}_exact_distinct_nodes"])
                for layer in REALCATS_LAYER_FILES
            },
        },
    }


def prepare_smoke_experiment(
    results_root: str | Path,
    run_id: str,
    *,
    hf_snapshot: str | Path | None = None,
    realcats_zip: str | Path = "data/raw/realcats_transactions/real-cats.zip",
    realcats_labels: str | Path = "data/raw/realcats_dataset",
    blte_dir: str | Path = "data/raw/ethereum_dataset_labelled",
    sla_seconds: float = 3_600,
) -> Path:
    if sla_seconds <= 0:
        raise ValueError("sla_seconds must be positive")
    paths = create_run_paths(results_root, run_id)
    telemetry = RunTelemetry(paths.root)
    marker = paths.root / "SMOKE_ONLY.txt"
    with marker.open("x", encoding="ascii") as handle:
        handle.write(
            "NON-CITABLE INTERNAL PIPELINE VALIDATION\n"
            "Metrics in this directory are not thesis findings.\n"
        )
    snapshot = Path(hf_snapshot) if hf_snapshot else discover_hf_snapshot()
    realcats_zip_path = Path(realcats_zip)
    realcats_label_dir = Path(realcats_labels)
    blte_path = Path(blte_dir)
    configuration = _smoke_configuration(sla_seconds)
    runtime = runtime_manifest()
    preparation_state = {
        "protocol_version": PROTOCOL_VERSION,
        "execution_profile": "smoke",
        "citable": False,
        "citable_candidate": False,
        "run_id": run_id,
        "configuration": configuration,
        "inputs": {
            "hf_snapshot": str(snapshot.resolve()),
            "realcats_zip": str(realcats_zip_path.resolve()),
            "realcats_labels": str(realcats_label_dir.resolve()),
            "blte_dir": str(blte_path.resolve()),
        },
        "runtime": runtime,
    }
    write_json_exclusive(paths.root / "preparation_state.json", preparation_state)
    telemetry.emit(
        "smoke.prepare",
        "started",
        run_id=run_id,
        sla_seconds=float(sla_seconds),
    )
    datasets: dict = {}
    datasets["huggingface"] = _prepare_smoke_hf(paths, snapshot, telemetry)
    datasets["realcats"] = _prepare_smoke_realcats(
        paths, realcats_zip_path, realcats_label_dir, telemetry
    )
    datasets["blte"] = _prepare_blte(paths, blte_path, telemetry=telemetry)
    hf_resource = datasets["huggingface"].get(
        "resource_metadata", {"approximate_distinct_nodes": 0, "target_count": 0}
    )
    realcats_resource = datasets["realcats"].get(
        "resource_metadata",
        {
            "target_count": 0,
            "exact_distinct_nodes_by_layer": {
                layer: 0 for layer in REALCATS_LAYER_FILES
            },
        },
    )
    blte_resource = datasets["blte"].get(
        "resource_metadata", {"exact_distinct_nodes": 0, "target_count": 0}
    )
    protocol = {
        "protocol_version": PROTOCOL_VERSION,
        "execution_profile": "smoke",
        "citable": False,
        "citable_candidate": False,
        "run_id": run_id,
        "research_questions": [
            "blte",
            "rq1",
            "rq2",
            "rq3",
            "rq4",
            "validation",
        ],
        "retrospective_classification_only": True,
        "primary_metric": "average_precision",
        "delta_selection": DELTA_SELECTION,
        "realcats_category_eligibility": {
            **REALCATS_CATEGORY_ELIGIBILITY,
            "minimum_active_addresses": SMOKE_REALCATS_MINIMUM_CLASS_SIZE,
            "smoke_override": True,
        },
        "alpha": 0.05,
        "deltas": list(SMOKE_DELTAS),
        "random_seed": RANDOM_SEED,
        "cv_seeds": [RANDOM_SEED],
        "permutations": SMOKE_RESAMPLES,
        "bootstraps": SMOKE_RESAMPLES,
        "ordering_variants": list(SMOKE_ORDER_VARIANTS),
        "event_execution": EVENT_EXECUTION,
        "hf_cohort_eligibility": HF_COHORT_ELIGIBILITY,
        "smoke_configuration": configuration,
        "model_parameters": {
            "random_forest": DEFAULT_RF_PARAMS,
            "xgboost": DEFAULT_XGB_PARAMS,
            "multiclass_logistic": MULTICLASS_LOGISTIC_PARAMS,
            "multiclass_standard_scaler": {"with_mean": True, "with_std": True},
        },
        "datasets": datasets,
        "resource_estimates": {
            "huggingface": {
                "node_count_method": "exact_bounded_smoke_endpoints",
                "approximate_distinct_nodes": hf_resource["approximate_distinct_nodes"],
                "node_count_admission": hf_resource["approximate_distinct_nodes"],
                "target_count": hf_resource["target_count"],
                "max_feature_columns": 49,
            },
            "realcats": {
                "node_count_method": "exact_sqlite_union_distinct_per_layer",
                "target_count": realcats_resource["target_count"],
                "max_feature_columns": 160,
                "layers": {
                    layer: {"exact_distinct_nodes": count}
                    for layer, count in realcats_resource["exact_distinct_nodes_by_layer"].items()
                },
            },
            "blte": {
                "node_count_method": "exact_materialized_endpoint_union",
                "exact_distinct_nodes": blte_resource["exact_distinct_nodes"],
                "target_count": blte_resource["target_count"],
                "max_feature_columns": 49,
            },
        },
        "resource_constants": RESOURCE_CONSTANTS,
        "source_scope": {
            "huggingface_preparation": "bounded_source_prefix_smoke_only",
            "huggingface_motif_graph": "target_incident_subgraph",
            "realcats_layers": list(REALCATS_LAYER_FILES),
            "blte_scope": "smoke_only",
        },
        "runtime": runtime,
        "preparation_state_sha256": sha256_file(
            paths.root / "preparation_state.json"
        ),
    }
    write_locked_protocol(paths, protocol)
    telemetry.emit("smoke.prepare", "completed", run_root=str(paths.root))
    return paths.root


def _artifact_inventory(root: Path) -> list[dict[str, int | str]]:
    return [
        {
            "path": str(path.relative_to(root)),
            "size_bytes": int(path.stat().st_size),
        }
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.name != "smoke_summary.json"
    ]


def run_smoke_experiment(
    results_root: str | Path,
    run_id: str,
    *,
    hf_snapshot: str | Path | None = None,
    realcats_zip: str | Path = "data/raw/realcats_transactions/real-cats.zip",
    realcats_labels: str | Path = "data/raw/realcats_dataset",
    blte_dir: str | Path = "data/raw/ethereum_dataset_labelled",
    sla_seconds: float = 3_600,
    clock: Callable[[], float] = monotonic,
) -> tuple[Path, dict, dict]:
    """Prepare and execute a smoke run, reporting rather than killing overruns."""
    started = clock()
    expected_root = Path(results_root) / run_id
    try:
        root = prepare_smoke_experiment(
            results_root,
            run_id,
            hf_snapshot=hf_snapshot,
            realcats_zip=realcats_zip,
            realcats_labels=realcats_labels,
            blte_dir=blte_dir,
            sla_seconds=sla_seconds,
        )
        manifest = execute_experiment(root)
    except BaseException as error:
        elapsed = max(0.0, clock() - started)
        if expected_root.is_dir():
            summary_path = expected_root / "smoke_summary.json"
            if not summary_path.exists():
                write_json_exclusive(
                    summary_path,
                    {
                        "status": "failed",
                        "citable": False,
                        "elapsed_seconds": elapsed,
                        "sla_seconds": float(sla_seconds),
                        "sla_met": elapsed <= sla_seconds,
                        "error_type": type(error).__name__,
                        "error": str(error),
                        "artifacts": _artifact_inventory(expected_root),
                    },
                )
                RunTelemetry(expected_root).emit(
                    "smoke", "failed", elapsed_seconds=elapsed,
                    error_type=type(error).__name__, error=str(error)
                )
        raise

    elapsed = max(0.0, clock() - started)
    summary = {
        "status": "complete",
        "citable": False,
        "elapsed_seconds": elapsed,
        "sla_seconds": float(sla_seconds),
        "sla_met": elapsed <= sla_seconds,
        "execution_manifest": "execution_manifest.json",
        "artifacts": _artifact_inventory(root),
    }
    write_json_exclusive(root / "smoke_summary.json", summary)
    RunTelemetry(root).emit(
        "smoke",
        "completed",
        elapsed_seconds=elapsed,
        sla_seconds=float(sla_seconds),
        sla_met=summary["sla_met"],
    )
    return root, manifest, summary
