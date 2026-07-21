"""Execute a prepared and hash-locked experiment without changing its protocol."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from src.experiments.artifacts import read_locked_protocol, runtime_manifest, sha256_file, write_json_exclusive
from src.experiments.constants import (
    DEFAULT_RF_PARAMS,
    DEFAULT_XGB_PARAMS,
    EVENT_EXECUTION,
    HF_COHORT_ELIGIBILITY,
    MULTICLASS_LOGISTIC_PARAMS,
)
from src.experiments.rq_binary import execute_hf
from src.experiments.rq_multiclass import execute_realcats
from src.experiments.secondary import execute_blte, execute_synthetic_validation
from src.experiments.telemetry import RunTelemetry


def _verify_prepared_artifacts(root: Path, dataset: dict) -> None:
    artifacts = dataset.get("artifacts", {})
    for name, expected in artifacts.items():
        if not name.endswith("_sha256"):
            continue
        artifact_name = name.removesuffix("_sha256")
        relative = artifacts.get(artifact_name)
        if isinstance(relative, str):
            actual = sha256_file(root / relative)
            if actual != expected:
                raise RuntimeError(
                    f"Prepared artifact changed: {relative}; expected {expected}, got {actual}"
                )
        elif isinstance(relative, dict) and isinstance(expected, dict):
            for key, relative_path in relative.items():
                actual = sha256_file(root / relative_path)
                if actual != expected[key]:
                    raise RuntimeError(f"Prepared artifact changed: {relative_path}")


def _verify_execution_definition(protocol: dict) -> None:
    """Reject source or model drift for protocol v2 and later."""
    if int(protocol.get("protocol_version", 1)) < 2:
        return
    current_runtime = runtime_manifest()
    expected_sources = protocol.get("runtime", {}).get("source_sha256")
    if not expected_sources or expected_sources != current_runtime["source_sha256"]:
        raise RuntimeError("Experiment source files changed after protocol preparation")
    expected_runtime = protocol.get("runtime", {})
    if expected_runtime.get("dependencies") != current_runtime["dependencies"]:
        raise RuntimeError("Dependency versions changed after protocol preparation")
    expected_python = str(expected_runtime.get("python", "")).split(maxsplit=1)[0]
    current_python = str(current_runtime["python"]).split(maxsplit=1)[0]
    if expected_python != current_python:
        raise RuntimeError("Python version changed after protocol preparation")
    current_models = {
        "random_forest": DEFAULT_RF_PARAMS,
        "xgboost": DEFAULT_XGB_PARAMS,
        "multiclass_logistic": MULTICLASS_LOGISTIC_PARAMS,
        "multiclass_standard_scaler": {"with_mean": True, "with_std": True},
    }
    if protocol.get("model_parameters") != current_models:
        raise RuntimeError("Model parameters changed after protocol preparation")
    if int(protocol.get("protocol_version", 1)) >= 3:
        if protocol.get("event_execution") != EVENT_EXECUTION:
            raise RuntimeError("Large-event execution definition changed after preparation")
        if protocol.get("hf_cohort_eligibility") != HF_COHORT_ELIGIBILITY:
            raise RuntimeError("Hugging Face cohort eligibility changed after preparation")


def execute_experiment(run_root: str | Path) -> dict:
    paths, protocol = read_locked_protocol(run_root)
    telemetry = RunTelemetry(paths.root)
    telemetry.emit("execute", "started", run_root=str(paths.root))
    with telemetry.step("execute.verify_protocol_and_artifacts"):
        _verify_execution_definition(protocol)
        for dataset in protocol["datasets"].values():
            _verify_prepared_artifacts(paths.root, dataset)

    results = {}
    selected = set(protocol["research_questions"])
    if {"rq1", "rq2"} & selected:
        with telemetry.step("execute.huggingface"):
            results["huggingface"] = execute_hf(paths, protocol, telemetry=telemetry)
    if "rq3" in selected:
        with telemetry.step("execute.realcats"):
            results["realcats"] = execute_realcats(paths, protocol, telemetry=telemetry)
    if "blte" in selected:
        with telemetry.step("execute.blte"):
            results["blte"] = execute_blte(paths, protocol, telemetry=telemetry)
    if "validation" in selected:
        with telemetry.step("execute.validation"):
            results["validation"] = execute_synthetic_validation(paths, telemetry=telemetry)

    manifest = {
        "status": "complete",
        "execution_profile": protocol.get("execution_profile", "thesis"),
        "citable": bool(protocol.get("citable", True)),
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol_sha256": paths.protocol_hash.read_text(encoding="ascii").strip(),
        "runtime": runtime_manifest(),
        "model_parameters": {
            "random_forest": DEFAULT_RF_PARAMS,
            "xgboost": DEFAULT_XGB_PARAMS,
            "multiclass_logistic": MULTICLASS_LOGISTIC_PARAMS,
            "multiclass_standard_scaler": {"with_mean": True, "with_std": True},
        },
        "results": results,
    }
    write_json_exclusive(paths.root / "execution_manifest.json", manifest)
    telemetry.emit("execute", "completed", manifest="execution_manifest.json")
    return manifest
