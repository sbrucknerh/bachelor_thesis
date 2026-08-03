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
    read_locked_protocol,
    resume_run_paths,
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
    DELTAS,
    EVENT_EXECUTION,
    HF_COHORT_ELIGIBILITY,
    HF_REPRESENTATION_ANALYSIS,
    MULTICLASS_LOGISTIC_PARAMS,
    N_BOOTSTRAPS,
    N_PERMUTATIONS,
    PROFILE_SETTINGS,
    PROTOCOL_VERSION,
    RANDOM_SEED,
    REALCATS_CATEGORY_ELIGIBILITY,
    REALCATS_LAYER_FILES,
    REALCATS_RESEARCH_QUESTIONS,
    RESOURCE_CONSTANTS,
    replication_seed_definition,
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


VALID_RQS = {
    "rq1",
    "rq2",
    *REALCATS_RESEARCH_QUESTIONS,
    "blte",
    "validation",
}


def _relative(path: Path, root: Path) -> str:
    return str(path.relative_to(root))


def _approx_distinct_endpoints(path: Path, temporary_directory: Path) -> int:
    import duckdb

    temporary_directory.mkdir(parents=True, exist_ok=True)
    connection = duckdb.connect()
    try:
        connection.execute("SET temp_directory = ?", [str(temporary_directory)])
        value = connection.execute(
            """
            SELECT approx_count_distinct(address) FROM (
                SELECT from_address AS address FROM read_parquet(?)
                UNION ALL
                SELECT to_address AS address FROM read_parquet(?)
            )
            """,
            [str(path), str(path)],
        ).fetchone()[0]
        return int(value or 0)
    finally:
        connection.close()


def _load_hf_exclusion_references(
    run_roots: Iterable[str | Path],
) -> tuple[set[int], set[str], list[dict]]:
    """Load immutable prior HF cohorts that a confirmation run must not reuse."""
    node_ids: set[int] = set()
    campaign_groups: set[str] = set()
    references: list[dict] = []
    observed_roots: set[Path] = set()
    for value in run_roots:
        root = Path(value).resolve()
        if root in observed_roots:
            raise ValueError(f"Duplicate --exclude-hf-run root: {root}")
        observed_roots.add(root)
        paths, protocol = read_locked_protocol(root)
        dataset = protocol.get("datasets", {}).get("huggingface")
        if dataset is None:
            raise ValueError(f"Prior run does not contain a Hugging Face dataset: {root}")
        artifacts = dataset.get("artifacts", {})
        relative = artifacts.get("targets")
        expected_hash = artifacts.get("targets_sha256")
        if not isinstance(relative, str) or not isinstance(expected_hash, str):
            raise ValueError(f"Prior HF run has no locked target artifact: {root}")
        target_path = paths.root / relative
        actual_hash = sha256_file(target_path)
        if actual_hash != expected_hash:
            raise RuntimeError(f"Prior HF target artifact hash mismatch: {target_path}")
        targets = pd.read_parquet(
            target_path, columns=["node_id", "campaign_group"]
        )
        if targets["node_id"].isna().any() or targets["campaign_group"].isna().any():
            raise ValueError(f"Prior HF targets contain null exclusion keys: {root}")
        if targets["node_id"].duplicated().any():
            raise ValueError(f"Prior HF targets contain duplicate node IDs: {root}")
        node_ids.update(int(value) for value in targets["node_id"])
        campaign_groups.update(str(value) for value in targets["campaign_group"])
        references.append(
            {
                "run_root": str(root),
                "run_id": str(protocol.get("run_id", root.name)),
                "protocol_version": int(protocol.get("protocol_version", 0)),
                "protocol_sha256": paths.protocol_hash.read_text(
                    encoding="ascii"
                ).strip(),
                "targets_sha256": actual_hash,
                "target_count": int(len(targets)),
                "source_edges_sha256": dataset.get("sources", {})
                .get("edges", {})
                .get("sha256"),
                "source_labels_sha256": dataset.get("sources", {})
                .get("labels", {})
                .get("sha256"),
                "source_descriptions_sha256": dataset.get("sources", {})
                .get("descriptions", {})
                .get("sha256"),
            }
        )
    return node_ids, campaign_groups, references


def _prepare_hf(
    paths,
    snapshot: Path,
    replication: dict,
    excluded_node_ids: set[int],
    excluded_campaign_groups: set[str],
    exclusion_references: list[dict],
    telemetry: RunTelemetry | None = None,
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
        selected, attrition = select_dual_hf_cohorts(
            all_targets,
            seed=int(replication["cohort_seed"]),
            excluded_node_ids=excluded_node_ids,
            excluded_campaign_groups=excluded_campaign_groups,
        )

        if set(selected["node_id"].astype(int)) & excluded_node_ids:
            raise RuntimeError("Confirmation cohort reused a prior-run HF node")
        if set(selected["campaign_group"].astype(str)) & excluded_campaign_groups:
            raise RuntimeError("Confirmation cohort reused a prior-run campaign group")

        selected["experiment_stratum"] = (
            selected["cohort"].astype(str) + ":" + selected["stratum"].astype(str)
        )
        selected["split"] = grouped_outer_split(
            selected,
            stratum_column="experiment_stratum",
            seed=int(replication["split_seed"]),
        ).values
        development = selected.loc[selected["split"] == "development"]
        folds = repeated_group_folds(
            development,
            seeds=replication["cv_seeds"],
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
    for reference in exclusion_references:
        if reference.get("source_edges_sha256") not in {None, source_edge_hash}:
            raise RuntimeError(
                "A prior HF exclusion run used a different edge source revision"
            )
        if reference.get("source_labels_sha256") not in {None, source_label_hash}:
            raise RuntimeError(
                "A prior HF exclusion run used a different label source revision"
            )
        if reference.get("source_descriptions_sha256") not in {
            None,
            description_hash,
        }:
            raise RuntimeError(
                "A prior HF exclusion run used a different description source revision"
            )
    approximate_nodes = _approx_distinct_endpoints(
        event_path, paths.data / "duckdb_tmp" / "prepare_hf_nodes"
    )
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
        "confirmation_exclusions": {
            "policy": "exclude_all_prior_target_node_ids_and_campaign_groups_before_sampling",
            "references": exclusion_references,
            "unique_prior_node_ids": int(len(excluded_node_ids)),
            "unique_prior_campaign_groups": int(len(excluded_campaign_groups)),
            "selected_node_overlap": 0,
            "selected_campaign_group_overlap": 0,
        },
        "edge_extraction": extraction,
        "resource_metadata": {
            "approximate_distinct_nodes": approximate_nodes,
            "target_count": int(len(selected)),
        },
    }


def _rq3_cv_assignments(targets: pd.DataFrame, seed: int) -> pd.DataFrame:
    development = targets.loc[targets["split"] == "development"].reset_index(drop=True)
    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=int(seed))
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
    split_seed: int = RANDOM_SEED,
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
        inferential["split"] = stratified_outer_split(
            inferential, seed=int(split_seed)
        ).values
        targets = targets.merge(
            inferential[["address", "split"]],
            on="address",
            how="left",
            validate="one_to_one",
        )
        folds = _rq3_cv_assignments(inferential, int(split_seed))

    raw_label_rows = int(
        len(pd.read_csv(label_dir / "CE.tsv", sep="\t", usecols=["address", "label"]))
    )

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
        "label_audit": {
            "source_rows": raw_label_rows,
            "unique_consistent_addresses": int(len(labels)),
            "identical_duplicate_rows_collapsed": raw_label_rows - int(len(labels)),
        },
        "resource_metadata": {
            "target_count": int(len(inferential)),
            "exact_distinct_nodes_by_layer": {
                layer: int(conversion[f"{layer}_exact_distinct_nodes"])
                for layer in REALCATS_LAYER_FILES
            },
        },
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
    distinct_nodes = int(
        pd.unique(
            pd.concat(
                [events["from_address"], events["to_address"]], ignore_index=True
            )
        ).size
    )
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
        "resource_metadata": {
            "exact_distinct_nodes": distinct_nodes,
            "target_count": int(len(targets)),
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
    permutations: int | None = None,
    bootstraps: int | None = None,
    profile: str = "thesis",
    resume: bool = False,
    replication_index: int = 0,
    exclude_hf_runs: Iterable[str | Path] = (),
) -> Path:
    selected_rqs = sorted(set(rqs))
    unknown = set(selected_rqs) - VALID_RQS
    if unknown:
        raise ValueError(f"Unknown research questions: {sorted(unknown)}")
    if profile not in PROFILE_SETTINGS:
        raise ValueError(f"Unknown preparation profile: {profile!r}")
    if "blte" in selected_rqs and selected_rqs != ["blte"]:
        raise ValueError("BLTE must be prepared as a separate optional secondary run")
    exclusion_roots = tuple(Path(value).resolve() for value in exclude_hf_runs)
    hf_selected = bool({"rq1", "rq2"} & set(selected_rqs))
    if exclusion_roots and not hf_selected:
        raise ValueError("--exclude-hf-run is valid only for an RQ1/RQ2 preparation")
    if hf_selected and int(replication_index) == 0 and exclusion_roots:
        raise ValueError("HF exclusion runs require --replication-index greater than zero")
    if hf_selected and int(replication_index) > 0 and not exclusion_roots:
        raise ValueError(
            "An HF confirmation replication must exclude at least one prior HF run"
        )
    realcats_selected = set(selected_rqs) & REALCATS_RESEARCH_QUESTIONS
    if realcats_selected and realcats_selected != REALCATS_RESEARCH_QUESTIONS:
        raise ValueError("RQ3 and RQ4 must be prepared together because they share one locked Real-CATS analysis")
    profile_settings = PROFILE_SETTINGS[profile]
    validation_only = selected_rqs == ["validation"]
    if profile == "thesis" and not validation_only and (
        permutations not in {None, N_PERMUTATIONS}
        or bootstraps not in {None, N_BOOTSTRAPS}
    ):
        raise ValueError(
            "The thesis profile locks 10,000 permutations and bootstraps; use "
            "--profile development for reduced resampling"
        )
    permutations = int(
        profile_settings["permutations"] if permutations is None else permutations
    )
    bootstraps = int(
        profile_settings["bootstraps"] if bootstraps is None else bootstraps
    )
    if permutations <= 0 or bootstraps <= 0:
        raise ValueError("permutations and bootstraps must be positive")
    if profile == "development" and (
        permutations > N_PERMUTATIONS or bootstraps > N_BOOTSTRAPS
    ):
        raise ValueError("The development profile may reduce, but not increase, resampling")
    if resume and set(selected_rqs) != REALCATS_RESEARCH_QUESTIONS:
        raise ValueError("Checkpoint resume is supported only for a combined RQ3/RQ4 preparation")
    replication = replication_seed_definition(replication_index)
    excluded_node_ids: set[int] = set()
    excluded_campaign_groups: set[str] = set()
    exclusion_references: list[dict] = []
    if exclusion_roots:
        (
            excluded_node_ids,
            excluded_campaign_groups,
            exclusion_references,
        ) = _load_hf_exclusion_references(exclusion_roots)
    replication["confirmation_design"] = (
        "disjoint_out_of_sample_confirmation"
        if exclusion_references
        else "initial_analysis"
    )
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
        profile=profile,
        resumed=resume,
    )
    realcats_zip_path = Path(realcats_zip)
    realcats_label_dir = Path(realcats_labels)
    realcats_source_hashes = None
    if REALCATS_RESEARCH_QUESTIONS <= set(selected_rqs):
        with telemetry.step("prepare.realcats.hash_sources"):
            realcats_source_hashes = {
                "transactions": sha256_file(realcats_zip_path),
                "labels": sha256_file(realcats_label_dir / "CE.tsv"),
            }
    preparation_definition = {
        "protocol_version": PROTOCOL_VERSION,
        "run_id": run_id,
        "research_questions": selected_rqs,
        "profile": profile,
        "permutations": int(permutations),
        "bootstraps": int(bootstraps),
        "delta_selection": DELTA_SELECTION,
        "replication": {
            **replication,
            "hf_exclusion_references": exclusion_references,
        },
        "inputs": {
            "hf_snapshot": str(Path(hf_snapshot).resolve()) if hf_snapshot else None,
            "realcats_zip": str(realcats_zip_path.resolve()),
            "realcats_labels": str(realcats_label_dir.resolve()),
            "blte_dir": str(Path(blte_dir).resolve()),
            "realcats_source_sha256": realcats_source_hashes,
            "exclude_hf_runs": [str(path) for path in exclusion_roots],
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
        datasets["huggingface"] = _prepare_hf(
            paths,
            snapshot,
            replication,
            excluded_node_ids,
            excluded_campaign_groups,
            exclusion_references,
            telemetry=telemetry,
        )
    if REALCATS_RESEARCH_QUESTIONS <= set(selected_rqs):
        datasets["realcats"] = _prepare_realcats(
            paths,
            realcats_zip_path,
            realcats_label_dir,
            resume=resume,
            source_hashes=realcats_source_hashes,
            split_seed=int(replication["split_seed"]),
            telemetry=telemetry,
        )
    if "blte" in selected_rqs:
        datasets["blte"] = _prepare_blte(paths, Path(blte_dir), telemetry=telemetry)

    resource_estimates = {}
    if "huggingface" in datasets:
        metadata = datasets["huggingface"]["resource_metadata"]
        resource_estimates["huggingface"] = {
            "node_count_method": "duckdb_approx_count_distinct_inflated_20_percent",
            "approximate_distinct_nodes": metadata["approximate_distinct_nodes"],
            "node_count_admission": int(metadata["approximate_distinct_nodes"] * 1.2 + 0.999999),
            "target_count": metadata["target_count"],
            "max_feature_columns": 50,
        }
    if "realcats" in datasets:
        metadata = datasets["realcats"]["resource_metadata"]
        resource_estimates["realcats"] = {
            "node_count_method": "exact_sqlite_union_distinct_per_layer",
            "target_count": metadata["target_count"],
            "max_feature_columns": 160,
            "layers": {
                layer: {"exact_distinct_nodes": value}
                for layer, value in metadata["exact_distinct_nodes_by_layer"].items()
            },
        }
    if "blte" in datasets:
        metadata = datasets["blte"]["resource_metadata"]
        resource_estimates["blte"] = {
            "node_count_method": "exact_materialized_endpoint_union",
            "exact_distinct_nodes": metadata["exact_distinct_nodes"],
            "target_count": metadata["target_count"],
            "max_feature_columns": 49,
        }

    protocol = {
        "protocol_version": PROTOCOL_VERSION,
        "run_id": run_id,
        "research_questions": selected_rqs,
        "execution_profile": profile,
        "citable_candidate": bool(profile_settings["citable_candidate"]),
        "retrospective_classification_only": True,
        "primary_metric": "average_precision",
        "delta_selection": DELTA_SELECTION,
        "hf_representation_analysis": HF_REPRESENTATION_ANALYSIS,
        "realcats_category_eligibility": REALCATS_CATEGORY_ELIGIBILITY,
        "alpha": 0.05,
        "deltas": list(DELTAS),
        "random_seed": RANDOM_SEED,
        "cv_seeds": list(replication["cv_seeds"]),
        "replication": {
            **replication,
            "hf_exclusion_references": exclusion_references,
        },
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
        "resource_estimates": resource_estimates,
        "resource_constants": RESOURCE_CONSTANTS,
        "source_scope": {
            "huggingface_preparation": "full_source_scan",
            "huggingface_motif_graph": "target_incident_subgraph",
            "realcats_layers": list(REALCATS_LAYER_FILES),
            "blte_scope": "separate_optional_secondary_rq1_run",
        },
        "runtime": preparation_runtime,
        "preparation_state_sha256": sha256_file(preparation_state),
    }
    protocol_context = telemetry.step("prepare.lock_protocol")
    with protocol_context:
        write_locked_protocol(paths, protocol)
    telemetry.emit("prepare", "completed", run_root=str(paths.root))
    return paths.root
