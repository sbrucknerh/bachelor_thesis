from __future__ import annotations

import pandas as pd
import pytest

from src.experiments.artifacts import (
    create_run_paths,
    read_locked_protocol,
    sha256_file,
    write_locked_protocol,
)
from src.experiments.constants import PROTOCOL_VERSION
from src.experiments.replications import aggregate_hf_replications


def _replication_root(
    tmp_path,
    run_id: str,
    index: int,
    address: str,
    group: str,
    prior_roots=(),
):
    paths = create_run_paths(tmp_path, run_id)
    targets = pd.DataFrame(
        {"address": [address], "campaign_group": [group], "cohort": ["controlled"]}
    )
    targets.to_parquet(paths.data / "hf_targets.parquet", index=False)
    exclusion_references = []
    for prior_root in prior_roots:
        prior_paths, prior_protocol = read_locked_protocol(prior_root)
        prior_dataset = prior_protocol["datasets"]["huggingface"]
        exclusion_references.append(
            {
                "targets_sha256": prior_dataset["artifacts"]["targets_sha256"],
                "protocol_sha256": prior_paths.protocol_hash.read_text(
                    encoding="ascii"
                ).strip(),
            }
        )
    protocol = {
        "protocol_version": PROTOCOL_VERSION,
        "run_id": run_id,
        "research_questions": ["rq1", "rq2"],
        "deltas": [300, 600],
        "delta_selection": {"primary": "fixture"},
        "hf_representation_analysis": {"fixture": True},
        "model_parameters": {"fixture": True},
        "ordering_variants": ["canonical"],
        "permutations": 10,
        "bootstraps": 10,
        "replication": {
            "index": index,
            "confirmation_design": (
                "disjoint_out_of_sample_confirmation"
                if exclusion_references
                else "initial_analysis"
            ),
            "hf_exclusion_references": exclusion_references,
        },
        "datasets": {
            "huggingface": {
                "sources": {
                    "edges": {"sha256": "e" * 64},
                    "labels": {"sha256": "l" * 64},
                    "descriptions": {"sha256": "d" * 64},
                },
                "artifacts": {
                    "targets": "data/hf_targets.parquet",
                    "targets_sha256": sha256_file(paths.data / "hf_targets.parquet"),
                },
            }
        },
    }
    write_locked_protocol(paths, protocol)
    pd.DataFrame(
        {
            "replication_index": [index],
            "cohort": ["controlled"],
            "ordering_variant": ["canonical"],
            "selected_delta": [600],
            "one_se_delta": [300],
            "delta_average_precision": [0.01 + index / 100],
            "ci_lower": [0.001],
            "ci_upper": [0.02],
            "p_value": [0.04],
            "reject_h0": [True],
        }
    ).to_csv(paths.tables / "rq1_hypothesis_tests.csv", index=False)
    pd.DataFrame(
        {
            "replication_index": [index],
            "cohort": ["controlled"],
            "ordering_variant": ["canonical"],
            "selected_delta": [600],
            "comparison": ["motifs_vs_volume"],
            "reference_arm": ["motif_volume"],
            "alternative_arm": ["motifs"],
            "observed_delta_ap": [0.1 + index / 100],
            "reject_h0_bh": [True],
        }
    ).to_csv(paths.tables / "rq1_representation_tests.csv", index=False)
    pd.DataFrame(
        {
            "replication_index": [index],
            "cohort": ["controlled"],
            "ordering_variant": ["canonical"],
            "selected_delta": [600],
            "observed_delta_ap": [0.02],
        }
    ).to_csv(paths.tables / "rq1_xgboost_robustness.csv", index=False)
    pd.DataFrame(
        {
            "replication_index": [index],
            "cohort": ["controlled"],
            "ordering_variant": ["canonical"],
            "selected_delta": [600],
            "one_se_delta": [300],
            "alternative_delta": [300],
            "observed_delta_ap": [0.01],
        }
    ).to_csv(paths.tables / "rq2_delta_comparisons.csv", index=False)
    return paths.root


def test_replication_aggregation_reports_every_disjoint_run(tmp_path, monkeypatch):
    first = _replication_root(tmp_path, "first", 0, "a", "group-a")
    second = _replication_root(
        tmp_path, "second", 1, "b", "group-b", prior_roots=[first]
    )
    monkeypatch.setattr(
        "src.experiments.replications.verify_run",
        lambda *_args, **_kwargs: {"integrity_ok": True, "citable": True},
    )

    output = aggregate_hf_replications(
        [first, second], tmp_path / "aggregation"
    )

    all_runs = pd.read_csv(output / "hf_replication_rq1_all_runs.csv")
    summary = pd.read_csv(output / "hf_replication_rq1_summary.csv")
    assert set(all_runs["run_id"]) == {"first", "second"}
    assert summary.loc[0, "n_replications"] == 2
    assert summary.loc[0, "replications_rejecting_h0"] == 2


def test_replication_aggregation_rejects_campaign_overlap(tmp_path, monkeypatch):
    first = _replication_root(tmp_path, "first", 0, "a", "shared")
    second = _replication_root(
        tmp_path, "second", 1, "b", "shared", prior_roots=[first]
    )
    monkeypatch.setattr(
        "src.experiments.replications.verify_run",
        lambda *_args, **_kwargs: {"integrity_ok": True, "citable": True},
    )
    output = tmp_path / "aggregation"

    with pytest.raises(ValueError, match="campaign groups overlap"):
        aggregate_hf_replications([first, second], output)
    assert not output.exists()


def test_replication_aggregation_rejects_missing_lineage(tmp_path, monkeypatch):
    first = _replication_root(tmp_path, "first", 0, "a", "group-a")
    second = _replication_root(tmp_path, "second", 1, "b", "group-b")
    monkeypatch.setattr(
        "src.experiments.replications.verify_run",
        lambda *_args, **_kwargs: {"integrity_ok": True, "citable": True},
    )

    with pytest.raises(ValueError, match="lineage is incomplete"):
        aggregate_hf_replications([first, second], tmp_path / "aggregation")
