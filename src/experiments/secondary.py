"""BLTE appendix analysis and exact synthetic correctness validation."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.experiments.artifacts import write_json_exclusive, write_parquet_exclusive
from src.experiments.features import aggregate_features, log_motif_counts, motif_coverage
from src.experiments.modeling import fit_binary_scores
from src.experiments.motifs import RaphtoryExperimentGraph
from src.experiments.statistics import binary_metrics, grouped_score_swap_test
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
    targets["address"] = targets["address"].astype(str)
    indexed = targets.set_index("address", drop=False)
    development, test = train_test_split(
        indexed.index.to_numpy(),
        test_size=0.2,
        random_state=int(protocol["random_seed"]),
        stratify=indexed["label"],
    )
    development = pd.Index(development)
    test = pd.Index(test)
    labels = indexed["label"].astype(int)
    baseline = aggregate_features(events, indexed.index)
    graph = RaphtoryExperimentGraph(events, "canonical", telemetry=telemetry)
    counts = graph.count(604_800, indexed.index)
    coverage_frame = motif_coverage(counts, labels).reset_index()
    minimum_coverage = 0.20
    if (coverage_frame["coverage"] < minimum_coverage).any():
        result = {
            "status": "descriptive_only",
            "reason": "at least one class has less than 20% nonzero motif coverage",
            "delta": 604_800,
            "minimum_required_coverage": minimum_coverage,
            "coverage": coverage_frame.to_dict(orient="records"),
        }
        write_json_exclusive(paths.metrics / "blte_results.json", result)
        return result
    motifs = log_motif_counts(counts)
    combined = baseline.join(motifs, how="left").fillna(0.0)
    scores = {}
    for arm, features in (("baseline", baseline), ("motifs", motifs), ("combined", combined)):
        _, scores[arm] = fit_binary_scores(
            features.loc[development], labels.loc[development], features.loc[test]
        )
    result = {
        "status": "evaluated",
        "delta": 604_800,
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
    }
    predictions = pd.DataFrame(
        {
            "address": np.repeat(test.to_numpy(), 3),
            "label": np.repeat(labels.loc[test].to_numpy(), 3),
            "arm": np.tile(["baseline", "motifs", "combined"], len(test)),
            "score": np.column_stack([scores["baseline"], scores["motifs"], scores["combined"]]).reshape(-1),
        }
    )
    write_parquet_exclusive(paths.predictions / "blte.parquet", predictions)
    write_json_exclusive(paths.metrics / "blte_results.json", result)
    return result


def execute_synthetic_validation(
    paths, telemetry: RunTelemetry | None = None
) -> dict:
    role_checks = []
    for role_index in range(32):
        fixture = fixture_for_role(role_index)
        graph = RaphtoryExperimentGraph(fixture, telemetry=telemetry)
        counts = graph.count(2, ["A"])
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
