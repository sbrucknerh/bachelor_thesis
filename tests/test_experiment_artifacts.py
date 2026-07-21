from __future__ import annotations

import json

import pytest

from src.experiments.artifacts import (
    create_run_paths,
    read_locked_protocol,
    runtime_manifest,
    write_json_exclusive,
    write_locked_protocol,
)
from src.experiments.constants import (
    DEFAULT_RF_PARAMS,
    DEFAULT_XGB_PARAMS,
    EVENT_EXECUTION,
    HF_COHORT_ELIGIBILITY,
    MULTICLASS_LOGISTIC_PARAMS,
)
from src.experiments.execute import _verify_execution_definition
from src.experiments.resources import (
    ResourceLimitError,
    parquet_uncompressed_bytes,
    require_in_memory_execution,
)


def test_protocol_hash_rejects_tampering(tmp_path):
    paths = create_run_paths(tmp_path, "run")
    write_locked_protocol(paths, {"deltas": [300], "seed": 42})
    protocol = json.loads(paths.protocol.read_text())
    protocol["deltas"] = [600]
    paths.protocol.write_text(json.dumps(protocol))
    with pytest.raises(RuntimeError, match="hash mismatch"):
        read_locked_protocol(paths.root)


def test_artifacts_are_exclusive(tmp_path):
    destination = tmp_path / "result.json"
    write_json_exclusive(destination, {"value": 1})
    with pytest.raises(FileExistsError):
        write_json_exclusive(destination, {"value": 2})


def test_existing_run_directory_is_rejected(tmp_path):
    create_run_paths(tmp_path, "run")
    with pytest.raises(FileExistsError):
        create_run_paths(tmp_path, "run")


def test_resource_preflight_rejects_before_loading(monkeypatch, tmp_path):
    import pandas as pd

    artifact = tmp_path / "large.parquet"
    pd.DataFrame({"value": range(100)}).to_parquet(artifact, index=False)
    monkeypatch.setattr("src.experiments.resources.physical_memory_bytes", lambda: 1)
    with pytest.raises(ResourceLimitError, match="was not trimmed or replaced"):
        require_in_memory_execution(artifact)


def test_resource_estimate_can_project_only_graph_columns(tmp_path):
    import pandas as pd

    artifact = tmp_path / "events.parquet"
    pd.DataFrame(
        {
            "timestamp": range(1_000),
            "from_address": ["a"] * 1_000,
            "event_signature": [f"large-signature-{index}" for index in range(1_000)],
        }
    ).to_parquet(artifact, index=False)
    projected = parquet_uncompressed_bytes(
        artifact, columns=("timestamp", "from_address")
    )
    assert projected < parquet_uncompressed_bytes(artifact)


def test_protocol_v2_rejects_source_drift(monkeypatch):
    runtime = runtime_manifest()
    protocol = {
        "protocol_version": 2,
        "runtime": runtime,
        "model_parameters": {
            "random_forest": DEFAULT_RF_PARAMS,
            "xgboost": DEFAULT_XGB_PARAMS,
            "multiclass_logistic": MULTICLASS_LOGISTIC_PARAMS,
            "multiclass_standard_scaler": {"with_mean": True, "with_std": True},
        },
    }
    changed = {**runtime, "source_sha256": {"changed.py": "0" * 64}}
    monkeypatch.setattr("src.experiments.execute.runtime_manifest", lambda: changed)
    with pytest.raises(RuntimeError, match="source files changed"):
        _verify_execution_definition(protocol)


def test_protocol_v3_rejects_execution_backend_drift():
    protocol = {
        "protocol_version": 3,
        "runtime": runtime_manifest(),
        "model_parameters": {
            "random_forest": DEFAULT_RF_PARAMS,
            "xgboost": DEFAULT_XGB_PARAMS,
            "multiclass_logistic": MULTICLASS_LOGISTIC_PARAMS,
            "multiclass_standard_scaler": {"with_mean": True, "with_std": True},
        },
        "event_execution": {**EVENT_EXECUTION, "motif_roles_retained": 31},
        "hf_cohort_eligibility": HF_COHORT_ELIGIBILITY,
    }
    with pytest.raises(RuntimeError, match="execution definition changed"):
        _verify_execution_definition(protocol)
