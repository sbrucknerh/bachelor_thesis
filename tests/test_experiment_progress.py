from __future__ import annotations

import pyarrow as pa
import pyarrow.parquet as pq

from src.experiments.progress import (
    build_execution_progress_plan,
    completed_progress_items,
)


def _write_parquet(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.table(payload), path)


def test_progress_plan_uses_event_weights_and_infers_resume_state(tmp_path):
    event_path = tmp_path / "data/hf_incident_edges.parquet"
    target_path = tmp_path / "data/hf_targets.parquet"
    _write_parquet(
        event_path,
        {
            "timestamp": [1, 2, 3],
            "from_address": ["a", "b", "c"],
            "to_address": ["b", "c", "a"],
        },
    )
    _write_parquet(target_path, {"cohort": ["controlled", "prevalence"]})
    protocol = {
        "ordering_variants": ["canonical"],
        "deltas": [300, 1_800],
        "datasets": {
            "huggingface": {
                "artifacts": {
                    "events": "data/hf_incident_edges.parquet",
                    "targets": "data/hf_targets.parquet",
                }
            }
        },
    }

    plan = build_execution_progress_plan(tmp_path, protocol)
    graph_key = "step:motifs.graph_load.hf_incident_edges.canonical"
    checkpoints = (
        "checkpoint:hf-canonical-delta-300",
        "checkpoint:hf-canonical-delta-1800",
    )
    assert plan.description == "Hugging Face"
    assert plan.graph_requirements[graph_key] == checkpoints
    assert "stage:hf-eval-controlled-canonical" in plan.items
    assert "stage:hf-eval-prevalence-canonical" in plan.items
    assert plan.digest == build_execution_progress_plan(tmp_path, protocol).digest

    stage = tmp_path / "stages/hf-baseline"
    stage.mkdir(parents=True)
    (stage / "stage_manifest.json").write_text("{}")
    for key in checkpoints:
        checkpoint = tmp_path / "checkpoints" / key.removeprefix("checkpoint:")
        checkpoint.mkdir(parents=True)
        (checkpoint / "counts.parquet").touch()
        (checkpoint / "metadata.json").write_text("{}")

    completed = completed_progress_items(tmp_path, plan)
    assert "stage:hf-baseline" in completed
    assert set(checkpoints) <= completed
    assert graph_key in completed


def test_progress_plan_skips_graph_load_for_empty_realcats_layer(tmp_path):
    layer_path = tmp_path / "data/realcats_layers/erc1155.parquet"
    _write_parquet(
        layer_path,
        {
            "timestamp": pa.array([], type=pa.int64()),
            "from_address": pa.array([], type=pa.string()),
            "to_address": pa.array([], type=pa.string()),
        },
    )
    protocol = {
        "ordering_variants": ["canonical"],
        "deltas": [300],
        "datasets": {
            "realcats": {
                "artifacts": {
                    "layers": {"erc1155": "data/realcats_layers/erc1155.parquet"}
                }
            }
        },
    }

    plan = build_execution_progress_plan(tmp_path, protocol)
    assert "checkpoint:realcats-erc1155-canonical-delta-300" in plan.items
    assert "step:motifs.graph_load.erc1155.canonical" not in plan.items
    assert "unit:realcats-eval:canonical" in plan.items
