from __future__ import annotations

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from src.experiments.constants import TRIANGLE_ROLE_NAMES
from src.experiments.datasets import (
    HF_DATASET_REVISION,
    campaign_group,
    count_target_activity,
    discover_hf_snapshot,
    extract_released_neighborhood_edges,
    extract_target_edges,
    select_activity_matched_blte_targets,
    select_dual_hf_cohorts,
)
from src.experiments.motifs import RaphtoryExperimentGraph
from src.experiments.realcats import select_activity_matched_binary_targets


def _write_hf_snapshot(root):
    for relative in (
        "gnn_dataset/edges_all/edges.parquet",
        "gnn_dataset/labels/targets_global.parquet",
        "addr_labels_balanced.csv.zst",
    ):
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()


def test_hf_discovery_prefers_project_data_then_pinned_cache(tmp_path):
    project_snapshot = tmp_path / "project"
    cache_root = tmp_path / "cache"
    pinned = (
        cache_root
        / "datasets--fesevu--ethereum_fraud_dataset_by_activity"
        / "snapshots"
        / HF_DATASET_REVISION
    )
    _write_hf_snapshot(pinned)

    assert (
        discover_hf_snapshot(cache_root, project_snapshot=project_snapshot)
        == pinned
    )

    _write_hf_snapshot(project_snapshot)
    assert (
        discover_hf_snapshot(cache_root, project_snapshot=project_snapshot)
        == project_snapshot
    )


def test_campaign_grouping_is_whitelisted():
    assert campaign_group("Multichain Hack Alert 12; take-action", "0x1") == "campaign:multichain_hack"
    assert campaign_group("Fake_Phishing3192", "0x2") == "address:0x2"
    assert campaign_group("Phishing", "0x3") == "address:0x3"


def test_activity_counts_self_loop_once_and_extracts_union(tmp_path):
    source = tmp_path / "edges.parquet"
    table = pa.table(
        {
            "src_id": pa.array([1, 1, 3], type=pa.uint64()),
            "dst_id": pa.array([1, 2, 1], type=pa.uint64()),
            "ts": [100, 101, 102],
            "value_wei": ["1", "2", "3"],
            "block_number": [1, 2, 3],
            "tx_hash": ["a", "b", "c"],
        }
    )
    pq.write_table(table, source)
    activity = count_target_activity(source, [1, 2])
    assert activity.loc[1] == 3
    assert activity.loc[2] == 1

    output = tmp_path / "selected.parquet"
    audit = extract_target_edges(source, [2], output)
    selected = pd.read_parquet(output)
    assert audit == {"source_edges": 3, "selected_edges": 1}
    assert selected.loc[0, "event_signature"] == "b|1|2"


def test_activity_and_edge_extraction_respect_exact_row_bound(tmp_path):
    source = tmp_path / "edges.parquet"
    pq.write_table(
        pa.table(
            {
                "src_id": pa.array([1, 1, 1, 1], type=pa.uint64()),
                "dst_id": pa.array([2, 2, 3, 4], type=pa.uint64()),
                "ts": [100, 101, 102, 103],
                "value_wei": ["1", "2", "3", "4"],
                "block_number": [1, 2, 3, 4],
                "tx_hash": ["a", "b", "c", "d"],
            }
        ),
        source,
        row_group_size=4,
    )

    activity = count_target_activity(source, [1, 4], batch_size=4, max_rows=3)
    assert activity.loc[1] == 3
    assert 4 not in activity.index

    output = tmp_path / "bounded.parquet"
    audit = extract_target_edges(
        source, [1], output, batch_size=4, max_rows=3
    )
    assert audit == {
        "source_edges": 4,
        "source_edges_scanned": 3,
        "selected_edges": 3,
    }
    assert list(pd.read_parquet(output)["tx_hash"]) == ["a", "b", "c"]


def test_released_neighborhood_extraction_restores_target_triangle_edges(tmp_path):
    source = tmp_path / "edges.parquet"
    pq.write_table(
        pa.table(
            {
                "src_id": pa.array([1, 2, 3, 8], type=pa.uint64()),
                "dst_id": pa.array([2, 3, 1, 9], type=pa.uint64()),
                "ts": [100, 101, 102, 103],
                "value_wei": ["1", "2", "3", "4"],
                "block_number": [1, 2, 3, 4],
                "tx_hash": ["target-out", "neighbor-edge", "target-in", "unrelated"],
            }
        ),
        source,
    )
    incident = tmp_path / "incident.parquet"
    extract_target_edges(source, [1], incident)
    neighborhood = tmp_path / "neighborhood.parquet"
    audit = extract_released_neighborhood_edges(source, incident, neighborhood)

    retained = pd.read_parquet(neighborhood)
    assert set(retained["tx_hash"]) == {"target-out", "neighbor-edge", "target-in"}
    assert audit == {
        "source_edges": 4,
        "incident_edges": 2,
        "neighborhood_nodes": 3,
        "selected_edges": 3,
        "additional_neighborhood_edges": 1,
        "scope": "released_graph_target_neighborhood_closure",
        "release_relative_completeness": True,
        "full_ethereum_completeness": False,
    }
    incident_counts = RaphtoryExperimentGraph(pd.read_parquet(incident)).count_all(
        10, ["1"]
    )
    neighborhood_counts = RaphtoryExperimentGraph(retained).count_all(10, ["1"])
    assert int(incident_counts.loc["1", TRIANGLE_ROLE_NAMES].sum()) == 0
    assert int(neighborhood_counts.loc["1", TRIANGLE_ROLE_NAMES].sum()) == 1


def test_dual_cohorts_are_disjoint_and_apply_locked_activity_eligibility():
    rows = []
    node = 0
    high_activity_node = None
    for label in (0, 1):
        for contract in (0, 1):
            for quintile in range(5):
                for offset in range(12):
                    node += 1
                    is_high_activity = (
                        label == 0 and contract == 0 and quintile == 4 and offset == 0
                    )
                    if is_high_activity:
                        high_activity_node = node
                    rows.append(
                        {
                            "node_id": node,
                            "address": str(node),
                            "label": label,
                            "is_contract": contract,
                            "activity_quintile": quintile,
                            "activity": 10**9 if is_high_activity else quintile + 1,
                            "campaign_group": f"address:{node}",
                        }
                    )
    rows.append(
        {"node_id": 9999, "address": "9999", "label": 0, "is_contract": 0,
         "activity_quintile": pd.NA, "activity": 0, "campaign_group": "address:9999"}
    )
    rows.append(
        {"node_id": 9998, "address": "9998", "label": 0, "is_contract": 0,
         "activity_quintile": pd.NA, "activity": 2, "campaign_group": "address:9998"}
    )
    targets = pd.DataFrame(rows)
    selected, audit = select_dual_hf_cohorts(
        targets, prevalence_size=20, controlled_per_stratum=2
    )
    prevalence = set(selected.loc[selected["cohort"] == "prevalence", "node_id"])
    controlled = set(selected.loc[selected["cohort"] == "controlled", "node_id"])
    assert prevalence.isdisjoint(controlled)
    assert 9999 not in set(selected["node_id"])
    assert 9998 not in set(selected["node_id"])
    assert high_activity_node not in set(selected["node_id"])
    assert audit["inactive_targets"] == 1
    assert audit["insufficient_activity_targets"] == 97
    assert audit["high_activity_targets_excluded"] == 1
    assert audit["minimum_incident_edges"] == 3

    eligible = targets.loc[
        (targets["activity"] >= 3) & (targets["node_id"] != high_activity_node)
    ]
    excluded_node = int(eligible.iloc[0]["node_id"])
    excluded_group = str(eligible.iloc[1]["campaign_group"])
    confirmation, confirmation_audit = select_dual_hf_cohorts(
        targets,
        prevalence_size=20,
        controlled_per_stratum=2,
        excluded_node_ids=[excluded_node],
        excluded_campaign_groups=[excluded_group],
    )
    assert excluded_node not in set(confirmation["node_id"])
    assert excluded_group not in set(confirmation["campaign_group"])
    assert confirmation_audit["prior_run_exclusions_total"] == 2


def test_mixed_blte_timestamp_formats_are_supported(tmp_path):
    import zipfile

    dataset = tmp_path / "blte"
    dataset.mkdir()
    transactions = pd.DataFrame(
        {
            "hash": ["a", "b"],
            "from_address": ["0xa", "0xb"],
            "to_address": ["0xb", "0xa"],
            "value": [1, 2],
            "block_timestamp": ["2017-10-16 05:26:53 UTC", "2017-10-16 08:31:44+00:00"],
            "block_number": [1, 2],
            "transaction_index": [0, 1],
        }
    )
    with zipfile.ZipFile(dataset / "Dataset.zip", "w") as archive:
        archive.writestr("Dataset/Dataset.csv", transactions.to_csv(index=False))
    pd.DataFrame({"Address": ["0xa", "0xb"], "Scam": [0, 1]}).to_csv(
        dataset / "Featuers_Extraction.csv", index=False
    )
    from src.experiments.datasets import load_blte_for_experiment

    events, labels = load_blte_for_experiment(dataset)
    assert len(events) == 2
    assert len(labels) == 2
    assert events["timestamp"].is_monotonic_increasing


def test_realcats_binary_matching_is_balanced_and_hf_disjoint():
    rows = []
    for label in (0, 1):
        for index, activity in enumerate((3, 4, 7, 8, 20, 21)):
            rows.append(
                {
                    "address": f"0x{label}{index}",
                    "source_activity": activity,
                    "label": label,
                }
            )
    labels = pd.DataFrame(rows)
    excluded = {"0x00"}
    selected, audit = select_activity_matched_binary_targets(
        labels,
        labels["address"],
        excluded,
        seed=42,
    )

    assert not (set(selected["address"]) & excluded)
    assert selected["label"].value_counts().nunique() == 1
    assert selected.groupby("activity_bin")["label"].value_counts().groupby(
        level=0
    ).nunique().eq(1).all()
    assert audit["hf_label_universe_exclusions"] == 1
    assert audit["activity_used_as_model_feature"] is False


def test_blte_activity_matching_uses_incident_events_only_for_selection():
    targets = []
    events = []
    for label in (0, 1):
        for index in range(4):
            address = f"target-{label}-{index}"
            targets.append(
                {
                    "address": address,
                    "label": label,
                    "campaign_group": f"address:{address}",
                }
            )
            for event_index in range(3 + index):
                events.append(
                    {
                        "from_address": address,
                        "to_address": f"peer-{label}-{index}-{event_index}",
                    }
                )
    selected, audit = select_activity_matched_blte_targets(
        pd.DataFrame(events), pd.DataFrame(targets), seed=42
    )

    assert selected["label"].value_counts().to_dict() == {0: 4, 1: 4}
    assert selected["incident_events"].min() >= 3
    assert audit["activity_used_as_model_feature"] is False
