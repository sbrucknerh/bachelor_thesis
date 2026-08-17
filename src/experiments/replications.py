"""Read-only aggregation of independent, verified HF confirmation runs."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.experiments.artifacts import (
    read_locked_protocol,
    sha256_file,
    write_csv_exclusive,
    write_json_exclusive,
)
from src.experiments.constants import PROTOCOL_VERSION
from src.experiments.verify import verify_run


def _compatible_definition(protocol: dict) -> dict:
    return {
        "protocol_version": protocol.get("protocol_version"),
        "primary_metric": protocol.get("primary_metric"),
        "alpha": protocol.get("alpha"),
        "deltas": protocol.get("deltas"),
        "delta_selection": protocol.get("delta_selection"),
        "hf_representation_analysis": protocol.get("hf_representation_analysis"),
        "hf_cohort_eligibility": protocol.get("hf_cohort_eligibility"),
        "event_execution": protocol.get("event_execution"),
        "model_parameters": protocol.get("model_parameters"),
        "ordering_variants": protocol.get("ordering_variants"),
        "permutations": protocol.get("permutations"),
        "bootstraps": protocol.get("bootstraps"),
        "source_sha256": protocol.get("runtime", {}).get("source_sha256"),
    }


def _count_true(values: pd.Series) -> int:
    normalized = values.map(
        lambda value: value
        if isinstance(value, bool)
        else str(value).strip().casefold()
    )
    allowed = {True, False, "true", "false"}
    if not set(normalized) <= allowed:
        raise ValueError("Replication decision columns contain invalid booleans")
    return int(normalized.isin({True, "true"}).sum())


def _summarize_primary(frame: pd.DataFrame) -> pd.DataFrame:
    canonical = frame.loc[frame["ordering_variant"] == "canonical"]
    return (
        canonical.groupby("cohort", as_index=False)
        .agg(
            n_replications=("run_id", "nunique"),
            mean_delta_average_precision=("delta_average_precision", "mean"),
            std_delta_average_precision=("delta_average_precision", "std"),
            min_delta_average_precision=("delta_average_precision", "min"),
            max_delta_average_precision=("delta_average_precision", "max"),
            replications_positive=(
                "delta_average_precision",
                lambda values: int((values > 0).sum()),
            ),
            replications_rejecting_h0=(
                "reject_h0",
                _count_true,
            ),
        )
        .sort_values("cohort")
        .reset_index(drop=True)
    )


def _summarize_representations(frame: pd.DataFrame) -> pd.DataFrame:
    return (
        frame.groupby(["cohort", "comparison"], as_index=False)
        .agg(
            n_replications=("run_id", "nunique"),
            mean_delta_average_precision=("observed_delta_ap", "mean"),
            std_delta_average_precision=("observed_delta_ap", "std"),
            min_delta_average_precision=("observed_delta_ap", "min"),
            max_delta_average_precision=("observed_delta_ap", "max"),
            replications_positive=(
                "observed_delta_ap",
                lambda values: int((values > 0).sum()),
            ),
            replications_rejecting_h0_bh=(
                "reject_h0_bh",
                _count_true,
            ),
        )
        .sort_values(["cohort", "comparison"])
        .reset_index(drop=True)
    )


def aggregate_hf_replications(
    run_roots: list[str | Path] | tuple[str | Path, ...],
    output_directory: str | Path,
) -> Path:
    """Report all compatible disjoint HF reruns without selecting or pooling them."""
    roots = [Path(value).resolve() for value in run_roots]
    if len(roots) < 2:
        raise ValueError("At least two HF run roots are required for replication aggregation")
    if len(set(roots)) != len(roots):
        raise ValueError("HF replication roots must be unique")

    output = Path(output_directory)
    source_definition = None
    source_hashes = None
    observed_addresses: set[str] = set()
    observed_groups: set[str] = set()
    observed_indices: set[int] = set()
    source_records: list[dict] = []
    primary_frames: list[pd.DataFrame] = []
    representation_frames: list[pd.DataFrame] = []
    xgboost_frames: list[pd.DataFrame] = []
    delta_frames: list[pd.DataFrame] = []

    for root in roots:
        verification = verify_run(root, require_citable=True)
        paths, protocol = read_locked_protocol(root)
        if int(protocol.get("protocol_version", 0)) != PROTOCOL_VERSION:
            raise ValueError(f"Replication root is not protocol v{PROTOCOL_VERSION}: {root}")
        if not {"rq1", "rq2"} <= set(protocol.get("research_questions", [])):
            raise ValueError(f"Replication root does not contain RQ1/RQ2: {root}")
        definition = _compatible_definition(protocol)
        if source_definition is None:
            source_definition = definition
        elif definition != source_definition:
            raise ValueError("HF replication scientific definitions differ")

        dataset = protocol["datasets"]["huggingface"]
        hashes = {
            "edges": dataset["sources"]["edges"]["sha256"],
            "labels": dataset["sources"]["labels"]["sha256"],
            "descriptions": dataset["sources"]["descriptions"]["sha256"],
        }
        if source_hashes is None:
            source_hashes = hashes
        elif hashes != source_hashes:
            raise ValueError("HF replication source datasets differ")

        replication_index = int(protocol["replication"]["index"])
        if replication_index in observed_indices:
            raise ValueError("HF replication indices must be unique")
        observed_indices.add(replication_index)
        targets_path = root / dataset["artifacts"]["targets"]
        targets = pd.read_parquet(
            targets_path, columns=["address", "campaign_group"]
        )
        addresses = set(targets["address"].astype(str))
        groups = set(targets["campaign_group"].astype(str))
        if observed_addresses & addresses:
            raise ValueError("HF replication target addresses overlap")
        if observed_groups & groups:
            raise ValueError("HF replication campaign groups overlap")
        observed_addresses.update(addresses)
        observed_groups.update(groups)

        run_id = str(protocol.get("run_id", root.name))
        for relative, collection in (
            ("tables/rq1_hypothesis_tests.csv", primary_frames),
            ("tables/rq1_representation_tests.csv", representation_frames),
            ("tables/rq1_xgboost_robustness.csv", xgboost_frames),
            ("tables/rq2_delta_comparisons.csv", delta_frames),
        ):
            frame = pd.read_csv(root / relative)
            if "replication_index" not in frame or set(
                frame["replication_index"].astype(int)
            ) != {replication_index}:
                raise ValueError(
                    f"Replication index mismatch in {root / relative}"
                )
            frame.insert(0, "run_id", run_id)
            collection.append(frame)
        exclusion_references = protocol["replication"].get(
            "hf_exclusion_references", []
        )
        source_records.append(
            {
                "run_id": run_id,
                "run_root": str(root),
                "replication_index": replication_index,
                "protocol_sha256": paths.protocol_hash.read_text(
                    encoding="ascii"
                ).strip(),
                "targets_sha256": sha256_file(targets_path),
                "target_count": int(len(targets)),
                "confirmation_design": protocol["replication"].get(
                    "confirmation_design"
                ),
                "exclusion_references": exclusion_references,
                "verification": verification,
            }
        )
    ordered_sources = sorted(
        source_records, key=lambda item: int(item["replication_index"])
    )
    for position, current in enumerate(ordered_sources):
        required_prior_hashes = {
            item["targets_sha256"] for item in ordered_sources[:position]
        }
        referenced_hashes = {
            item.get("targets_sha256")
            for item in current["exclusion_references"]
        }
        if not required_prior_hashes <= referenced_hashes:
            raise ValueError(
                "HF replication lineage is incomplete: every later run must "
                "exclude every earlier supplied run"
            )
    primary = pd.concat(primary_frames, ignore_index=True)
    representations = pd.concat(representation_frames, ignore_index=True)
    xgboost = pd.concat(xgboost_frames, ignore_index=True)
    deltas = pd.concat(delta_frames, ignore_index=True)
    output.mkdir(parents=True, exist_ok=False)
    output_files: list[Path] = []

    def write_report(name: str, frame: pd.DataFrame) -> None:
        path = output / name
        write_csv_exclusive(path, frame)
        output_files.append(path)

    write_report("hf_replication_rq1_all_runs.csv", primary)
    write_report(
        "hf_replication_representation_all_runs.csv", representations
    )
    write_report("hf_replication_xgboost_all_runs.csv", xgboost)
    write_report("hf_replication_rq2_all_runs.csv", deltas)
    write_report(
        "hf_replication_rq1_summary.csv", _summarize_primary(primary)
    )
    write_report(
        "hf_replication_representation_summary.csv",
        _summarize_representations(representations),
    )
    selections = (
        primary.loc[primary["ordering_variant"] == "canonical", [
            "run_id",
            "replication_index",
            "cohort",
            "selected_delta",
            "one_se_delta",
        ]]
        .sort_values(["replication_index", "cohort"])
        .reset_index(drop=True)
    )
    write_report(
        "hf_replication_delta_selections.csv", selections
    )
    write_json_exclusive(
        output / "hf_replication_manifest.json",
        {
            "schema_version": 1,
            "protocol_version": PROTOCOL_VERSION,
            "status": "descriptive_replication_aggregation",
            "inferential_pooling": "none",
            "selection_policy": "all_supplied_verified_disjoint_runs_reported",
            "source_hashes": source_hashes,
            "scientific_definition": source_definition,
            "artifact_sha256": {
                path.name: sha256_file(path) for path in output_files
            },
            "address_overlap": 0,
            "campaign_group_overlap": 0,
            "lineage_checked": True,
            "sources": ordered_sources,
        },
    )
    return output
