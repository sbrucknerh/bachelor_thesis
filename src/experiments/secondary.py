"""BLTE appendix analysis and exact synthetic correctness validation."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.experiments.artifacts import (
    write_csv_exclusive,
    write_json_exclusive,
    write_parquet_exclusive,
)
from src.experiments.features import aggregate_features, log_motif_counts, motif_coverage
from src.experiments.modeling import (
    cv_delta_scores,
    fit_binary_scores,
    select_delta_maximum,
    select_delta_one_standard_error,
)
from src.experiments.motifs import RaphtoryExperimentGraph
from src.experiments.statistics import (
    binary_metrics,
    grouped_score_swap_test,
    stratified_group_bootstrap_delta,
)
from src.experiments.synthetic_fixtures import fixture_for_role
from src.experiments.telemetry import RunTelemetry


def execute_blte(paths, protocol: dict, telemetry: RunTelemetry | None = None) -> dict:
    dataset = protocol["datasets"]["blte"]
    if not dataset["eligible"]:
        result = {
            "status": "descriptive_only",
            "reason": "fewer than 100 active addresses in at least one class",
            "active_label_counts": dataset["active_label_counts"],
        }
        write_json_exclusive(paths.metrics / "blte_results.json", result)
        return result

    events = pd.read_parquet(paths.root / dataset["artifacts"]["events"])
    targets = pd.read_parquet(paths.root / dataset["artifacts"]["targets"])
    folds = pd.read_parquet(paths.root / dataset["artifacts"]["folds"])
    targets["address"] = targets["address"].astype(str)
    indexed = targets.set_index("address", drop=False)
    development = indexed.index[indexed["split"] == "development"]
    test = indexed.index[indexed["split"] == "test"]
    labels = indexed["label"].astype(int)
    baseline = aggregate_features(events, indexed.index)
    graph = RaphtoryExperimentGraph(events, "canonical", telemetry=telemetry)
    counts_by_delta = graph.sweep(protocol["deltas"], indexed.index)
    motif_features = {
        delta: log_motif_counts(counts) for delta, counts in counts_by_delta.items()
    }
    combined_features = {
        delta: baseline.join(features, how="left").fillna(0.0)
        for delta, features in motif_features.items()
    }
    cv_scores = cv_delta_scores(
        {delta: frame.loc[development] for delta, frame in combined_features.items()},
        labels.loc[development],
        folds,
    )
    selected_delta, selection = select_delta_maximum(cv_scores)
    one_se_delta, one_se_selection = select_delta_one_standard_error(cv_scores)
    selection = selection.merge(
        one_se_selection[["delta", "within_one_se", "selected_one_se"]],
        on="delta",
        how="left",
        validate="one_to_one",
    )
    coverage_records = []
    for delta, counts in counts_by_delta.items():
        frame = motif_coverage(counts, labels).reset_index()
        frame["delta"] = int(delta)
        coverage_records.append(frame)
    coverage = pd.concat(coverage_records, ignore_index=True)
    coverage_frame = coverage.loc[coverage["delta"] == selected_delta]
    minimum_coverage = 0.20
    if (coverage_frame["coverage"] < minimum_coverage).any():
        result = {
            "status": "descriptive_only",
            "reason": "at least one class has less than 20% nonzero motif coverage",
            "selected_delta": int(selected_delta),
            "one_se_delta": int(one_se_delta),
            "minimum_required_coverage": minimum_coverage,
            "coverage": coverage.to_dict(orient="records"),
        }
        write_csv_exclusive(paths.metrics / "blte_cv.csv", cv_scores)
        write_csv_exclusive(paths.metrics / "blte_delta_selection.csv", selection)
        write_csv_exclusive(paths.metrics / "blte_coverage.csv", coverage)
        write_json_exclusive(paths.metrics / "blte_results.json", result)
        return result
    motifs = motif_features[selected_delta]
    combined = combined_features[selected_delta]
    scores = {}
    for arm, features in (("baseline", baseline), ("motifs", motifs), ("combined", combined)):
        _, scores[arm] = fit_binary_scores(
            features.loc[development], labels.loc[development], features.loc[test]
        )
    result = {
        "status": "evaluated",
        "selected_delta": int(selected_delta),
        "one_se_delta": int(one_se_delta),
        "metrics": {arm: binary_metrics(labels.loc[test], values) for arm, values in scores.items()},
        "coverage": coverage_frame.to_dict(orient="records"),
        "randomization": grouped_score_swap_test(
            labels.loc[test],
            scores["baseline"],
            scores["combined"],
            indexed.loc[test, "campaign_group"],
            iterations=int(protocol["permutations"]),
            seed=int(protocol["random_seed"]),
        ),
        "bootstrap": stratified_group_bootstrap_delta(
            labels.loc[test],
            scores["baseline"],
            scores["combined"],
            indexed.loc[test, "campaign_group"],
            iterations=int(protocol["bootstraps"]),
            seed=int(protocol["random_seed"]),
        ),
    }
    predictions = pd.DataFrame(
        {
            "address": np.repeat(test.to_numpy(), 3),
            "label": np.repeat(labels.loc[test].to_numpy(), 3),
            "delta": int(selected_delta),
            "arm": np.tile(["baseline", "motifs", "combined"], len(test)),
            "score": np.column_stack([scores["baseline"], scores["motifs"], scores["combined"]]).reshape(-1),
        }
    )
    write_parquet_exclusive(paths.predictions / "blte.parquet", predictions)
    write_csv_exclusive(paths.metrics / "blte_cv.csv", cv_scores)
    write_csv_exclusive(paths.metrics / "blte_delta_selection.csv", selection)
    write_csv_exclusive(paths.metrics / "blte_coverage.csv", coverage)
    write_json_exclusive(paths.metrics / "blte_results.json", result)
    return result


def execute_synthetic_validation(
    paths, telemetry: RunTelemetry | None = None
) -> dict:
    role_checks = []
    for role_index in range(40):
        fixture = fixture_for_role(role_index)
        graph = RaphtoryExperimentGraph(fixture, telemetry=telemetry)
        counts = graph.count_all(2, ["A"])
        vector = counts.iloc[0].to_numpy()
        role_checks.append(
            {
                "role_index": role_index,
                "observed_index": int(np.flatnonzero(vector)[0]) if vector.sum() else None,
                "observed_count": int(vector.sum()),
                "passed": bool(vector.sum() == 1 and vector[role_index] == 1),
            }
        )

    boundary_fixture = fixture_for_role(0, start=100, step=5)
    boundary_graph = RaphtoryExperimentGraph(boundary_fixture, telemetry=telemetry)
    at_delta = int(boundary_graph.count(10, ["A"]).to_numpy().sum())
    below_delta = int(boundary_graph.count(9, ["A"]).to_numpy().sum())

    changed_roles = []
    for role_index in range(32):
        tied = fixture_for_role(role_index, start=100, step=0)
        tied["transaction_index"] = np.nan
        tied["block_number"] = 1
        canonical = RaphtoryExperimentGraph(
            tied, "canonical", telemetry=telemetry
        ).count(1, ["A"]).to_numpy()
        reverse = RaphtoryExperimentGraph(
            tied, "reverse", telemetry=telemetry
        ).count(1, ["A"]).to_numpy()
        if not np.array_equal(canonical, reverse):
            changed_roles.append(role_index)
    result = {
        "all_role_checks_passed": all(item["passed"] for item in role_checks),
        "role_checks": role_checks,
        "delta_boundary": {
            "span_equals_delta_count": at_delta,
            "span_exceeds_delta_count": below_delta,
            "passed": at_delta == 1 and below_delta == 0,
        },
        "tie_order_changes_fingerprint": bool(changed_roles),
        "tie_sensitive_roles": changed_roles,
    }
    result["passed"] = bool(
        result["all_role_checks_passed"]
        and result["delta_boundary"]["passed"]
        and result["tie_order_changes_fingerprint"]
    )
    write_json_exclusive(paths.metrics / "synthetic_validation.json", result)
    if not result["passed"]:
        raise RuntimeError("Synthetic motif validation failed")
    return result
