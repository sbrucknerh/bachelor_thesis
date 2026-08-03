from __future__ import annotations

import pandas as pd

from src.experiments.artifacts import create_run_paths
from src.experiments.constants import (
    HF_DIAGNOSTIC_ARM_NAMES,
    HF_DIAGNOSTIC_COMPARISONS,
    MOTIF_ROLE_NAMES,
)
from src.experiments.rq_binary import execute_hf
from src.experiments.rq_multiclass import execute_realcats
from src.experiments.secondary import execute_blte
from src.experiments.splits import grouped_outer_split, repeated_group_folds
from src.experiments.synthetic_fixtures import fixture_for_role


def _renamed_fixture(role: int, target: str, offset: int) -> pd.DataFrame:
    frame = fixture_for_role(role, start=100 + offset, step=1)
    mapping = {"A": target, "B": f"{target}-b", "C": f"{target}-c"}
    frame["from_address"] = frame["from_address"].map(mapping)
    frame["to_address"] = frame["to_address"].map(mapping)
    frame["tx_hash"] = target + "-" + frame["tx_hash"]
    frame["event_signature"] = frame["tx_hash"]
    frame["block_number"] += offset
    return frame


def test_rq1_rq2_orchestration_on_locked_fixture(tmp_path):
    paths = create_run_paths(tmp_path, "binary")
    target_rows = []
    event_frames = []
    for label in (0, 1):
        for index in range(20):
            address = f"target-{label}-{index}"
            target_rows.append(
                {
                    "address": address,
                    "node_id": label * 100 + index,
                    "label": label,
                    "is_contract": index % 2,
                    "activity_quintile": index % 5,
                    "activity": 3,
                    "stratum": str(label),
                    "campaign_group": f"address:{address}",
                    "cohort": "controlled",
                }
            )
            event_frames.append(_renamed_fixture(0 if label == 0 else 7, address, (label * 20 + index) * 10))
    targets = pd.DataFrame(target_rows)
    targets["split"] = grouped_outer_split(targets).values
    folds = repeated_group_folds(
        targets.loc[targets["split"] == "development"], seeds=(42,)
    )
    folds["cohort"] = "controlled"
    events = pd.concat(event_frames, ignore_index=True)
    targets.to_parquet(paths.data / "targets.parquet", index=False)
    folds.to_parquet(paths.data / "folds.parquet", index=False)
    events.to_parquet(paths.data / "events.parquet", index=False)
    protocol = {
        "alpha": 0.05,
        "deltas": [2, 10],
        "random_seed": 42,
        "permutations": 10,
        "bootstraps": 10,
        "ordering_variants": ["canonical"],
        "datasets": {
            "huggingface": {
                "artifacts": {
                    "targets": "data/targets.parquet",
                    "folds": "data/folds.parquet",
                    "events": "data/events.parquet",
                }
            }
        },
    }
    result = execute_hf(paths, protocol)
    assert "controlled" in result
    assert (paths.metrics / "hf_controlled_canonical_statistical_tests.json").exists()
    assert (paths.figures / "rq1_ablation_controlled.pdf").exists()
    metrics = pd.read_csv(paths.metrics / "hf_controlled_canonical_metrics.csv")
    selected_delta = int(
        pd.read_csv(paths.tables / "rq1_hypothesis_tests.csv").loc[0, "selected_delta"]
    )
    selected_arms = set(metrics.loc[metrics["delta"] == selected_delta, "arm"])
    assert {"baseline", "motifs", "combined", *HF_DIAGNOSTIC_ARM_NAMES} <= selected_arms
    representation = pd.read_csv(paths.tables / "rq1_representation_tests.csv")
    assert set(representation["comparison"]) == {
        comparison for comparison, _, _ in HF_DIAGNOSTIC_COMPARISONS
    }
    roles = pd.read_csv(paths.tables / "rq1_role_diagnostics.csv")
    assert set(roles["scope"]) == {"development", "test"}
    assert set(roles["role"]) == set(MOTIF_ROLE_NAMES)
    coverage = pd.read_csv(paths.tables / "rq2_motif_coverage.csv")
    assert set(coverage["scope"]) == {"all", "development", "test"}
    assert (paths.figures / "rq1_representation_ablation_controlled.pdf").exists()
    assert (paths.figures / "rq2_motif_coverage_controlled.pdf").exists()


def test_rq3_orchestration_uses_exact_labels_and_layer_features(tmp_path):
    paths = create_run_paths(tmp_path, "multiclass")
    target_rows = []
    event_frames = []
    exact_labels = ["Phishing Scam", "Hack Scam", "Rug Pull Scam"]
    for label_index, label in enumerate(exact_labels):
        for index in range(10):
            address = f"type-{label_index}-{index}"
            target_rows.append(
                {
                    "address": address,
                    "type_label": label,
                    "inference_eligible": True,
                    "split": "development" if index < 8 else "test",
                }
            )
            event_frames.append(_renamed_fixture(label_index, address, (label_index * 10 + index) * 10))
    targets = pd.DataFrame(target_rows)
    events = pd.concat(event_frames, ignore_index=True)
    fold_rows = []
    for label_index, _ in enumerate(exact_labels):
        for index in range(8):
            fold_rows.append(
                {"address": f"type-{label_index}-{index}", "fold": index % 5}
            )
    folds = pd.DataFrame(fold_rows)
    targets.to_parquet(paths.data / "targets.parquet", index=False)
    folds.to_parquet(paths.data / "folds.parquet", index=False)
    events.to_parquet(paths.data / "normal.parquet", index=False)
    protocol = {
        "alpha": 0.05,
        "deltas": [2, 10],
        "random_seed": 42,
        "permutations": 10,
        "bootstraps": 10,
        "ordering_variants": ["canonical"],
        "datasets": {
            "realcats": {
                "artifacts": {
                    "targets": "data/targets.parquet",
                    "folds": "data/folds.parquet",
                    "layers": {"normal": "data/normal.parquet"},
                }
            }
        },
    }
    result = execute_realcats(paths, protocol)
    assert result["ordering_robust_global_separability"] in {True, False}
    assert result["rq4_ordering_robust_category_specific_windows"] in {
        True,
        False,
    }
    output = pd.read_parquet(paths.predictions / "realcats_canonical.parquet")
    assert set(output["true_label"]) == set(exact_labels)
    assert (paths.figures / "rq3_label_delta_curves.pdf").exists()
    assert (paths.figures / "rq4_label_delta_curves.pdf").exists()
    assert (paths.tables / "rq4_global_results.csv").exists()
    assert (paths.tables / "rq4_category_results.csv").exists()


def test_blte_stops_before_models_when_motif_coverage_is_too_low(tmp_path):
    paths = create_run_paths(tmp_path, "blte")
    targets = []
    events = []
    for label in (0, 1):
        for index in range(100):
            address = f"target-{label}-{index}"
            targets.append(
                {
                    "address": address,
                    "label": label,
                    "campaign_group": f"address:{address}",
                }
            )
            events.append(
                {
                    "from_address": address,
                    "to_address": f"peer-{label}-{index}",
                    "timestamp": label * 1_000 + index,
                    "value": 1.0,
                    "tx_hash": f"tx-{label}-{index}",
                    "block_number": label * 1_000 + index,
                    "transaction_index": 0,
                    "event_signature": f"tx-{label}-{index}",
                    "layer": "normal",
                }
            )
    pd.DataFrame(events).to_parquet(paths.data / "events.parquet", index=False)
    pd.DataFrame(targets).to_parquet(paths.data / "targets.parquet", index=False)
    protocol = {
        "random_seed": 42,
        "permutations": 10,
        "datasets": {
            "blte": {
                "eligible": True,
                "artifacts": {
                    "events": "data/events.parquet",
                    "targets": "data/targets.parquet",
                },
            }
        },
    }

    result = execute_blte(paths, protocol)

    assert result["status"] == "descriptive_only"
    assert result["minimum_required_coverage"] == 0.20
    assert not (paths.predictions / "blte.parquet").exists()
