"""Resumable, integrity-chained protocol-v5 experiment execution."""

from __future__ import annotations

from contextlib import nullcontext
from datetime import datetime, timezone
import json
import os
from pathlib import Path

from src.experiments.artifacts import (
    atomic_write_bytes_exclusive,
    protocol_digest,
    read_locked_protocol,
    runtime_manifest,
    sha256_file,
    write_json_exclusive,
)
from src.experiments.constants import (
    DEFAULT_RF_PARAMS,
    DEFAULT_XGB_PARAMS,
    DELTA_SELECTION,
    DELTAS,
    EVENT_EXECUTION,
    HF_COHORT_ELIGIBILITY,
    MULTICLASS_LOGISTIC_PARAMS,
    N_BOOTSTRAPS,
    N_PERMUTATIONS,
    PROTOCOL_VERSION,
    REALCATS_CATEGORY_ELIGIBILITY,
    REALCATS_RESEARCH_QUESTIONS,
    RESOURCE_CONSTANTS,
)
from src.experiments.execution_control import (
    FileLock,
    configure_thread_environment,
    default_host_lock_path,
    finish_attempt,
    start_attempt,
    validate_inherited_lock,
)
from src.experiments.execution_artifacts import (
    CheckpointStore,
    StageManager,
    verify_inventory,
    write_inventory,
)


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
            if set(relative) != set(expected):
                raise RuntimeError(f"Prepared artifact hash keys changed for {artifact_name}")
            for key, relative_path in relative.items():
                if sha256_file(root / relative_path) != expected[key]:
                    raise RuntimeError(f"Prepared artifact changed: {relative_path}")


def _verify_locked_scientific_definition(protocol: dict) -> None:
    if int(protocol.get("protocol_version", 0)) != PROTOCOL_VERSION:
        raise RuntimeError(
            f"Only protocol v{PROTOCOL_VERSION} can execute or resume; legacy runs are read-only"
        )
    current_runtime = runtime_manifest()
    expected_runtime = protocol.get("runtime", {})
    if expected_runtime.get("source_sha256") != current_runtime["source_sha256"]:
        raise RuntimeError("Experiment source files changed after protocol preparation")
    if expected_runtime.get("dependencies") != current_runtime["dependencies"]:
        raise RuntimeError("Dependency versions changed after protocol preparation")
    expected_python = str(expected_runtime.get("python", "")).split(maxsplit=1)[0]
    current_python = str(current_runtime["python"]).split(maxsplit=1)[0]
    if expected_python != current_python:
        raise RuntimeError("Python version changed after protocol preparation")
    analytical_models = {
        "random_forest": DEFAULT_RF_PARAMS,
        "xgboost": DEFAULT_XGB_PARAMS,
        "multiclass_logistic": MULTICLASS_LOGISTIC_PARAMS,
        "multiclass_standard_scaler": {"with_mean": True, "with_std": True},
    }
    if protocol.get("model_parameters") != analytical_models:
        raise RuntimeError("Analytical model parameters changed after preparation")
    if protocol.get("event_execution") != EVENT_EXECUTION:
        raise RuntimeError("Large-event execution definition changed after preparation")
    if protocol.get("hf_cohort_eligibility") != HF_COHORT_ELIGIBILITY:
        raise RuntimeError("Hugging Face cohort eligibility changed after preparation")
    if protocol.get("delta_selection") != DELTA_SELECTION:
        raise RuntimeError("Temporal-window selection rules changed after preparation")
    expected_category_eligibility = REALCATS_CATEGORY_ELIGIBILITY
    if protocol.get("execution_profile") == "smoke":
        expected_category_eligibility = {
            **REALCATS_CATEGORY_ELIGIBILITY,
            "minimum_active_addresses": protocol["smoke_configuration"][
                "realcats_minimum_class_size"
            ],
            "smoke_override": True,
        }
    if (
        protocol.get("realcats_category_eligibility")
        != expected_category_eligibility
    ):
        raise RuntimeError("Real-CATS category eligibility changed after preparation")
    if protocol.get("resource_constants") != RESOURCE_CONSTANTS:
        raise RuntimeError(
            f"Protocol-v{PROTOCOL_VERSION} resource constants changed after preparation"
        )
    selected = set(protocol.get("research_questions", []))
    realcats_selected = selected & REALCATS_RESEARCH_QUESTIONS
    if realcats_selected and realcats_selected != REALCATS_RESEARCH_QUESTIONS:
        raise RuntimeError("RQ3 and RQ4 must share one locked Real-CATS execution")


def _verify_execution_definition(protocol: dict) -> None:
    """Compatibility verifier for read-only inspection of protocol v2/v3 tests."""
    version = int(protocol.get("protocol_version", 1))
    if version >= 4:
        _verify_locked_scientific_definition(protocol)
        return
    if version < 2:
        return
    current = runtime_manifest()
    expected = protocol.get("runtime", {})
    if expected.get("source_sha256") != current.get("source_sha256"):
        raise RuntimeError("Experiment source files changed after protocol preparation")
    if expected.get("dependencies") != current.get("dependencies"):
        raise RuntimeError("Dependency versions changed after protocol preparation")
    if version >= 3:
        if protocol.get("event_execution") != EVENT_EXECUTION:
            raise RuntimeError("Large-event execution definition changed after preparation")
        if protocol.get("hf_cohort_eligibility") != HF_COHORT_ELIGIBILITY:
            raise RuntimeError("Hugging Face cohort eligibility changed after preparation")


def _derive_citable(
    root: Path, protocol: dict, validation_result: dict, generated_audit: dict
) -> tuple[bool, list[str]]:
    reasons = []
    if protocol.get("execution_profile") != "thesis":
        reasons.append("execution profile is not thesis")
    if not protocol.get("citable_candidate", False):
        reasons.append("protocol is not a citable candidate")
    validation_only = set(protocol.get("research_questions", [])) == {"validation"}
    if not validation_only and int(protocol.get("permutations", 0)) != N_PERMUTATIONS:
        reasons.append("permutation count is not thesis-locked")
    if not validation_only and int(protocol.get("bootstraps", 0)) != N_BOOTSTRAPS:
        reasons.append("bootstrap count is not thesis-locked")
    if not validation_only and tuple(protocol.get("deltas", [])) != DELTAS:
        reasons.append("delta grid is not thesis-locked")
    if not validation_result.get("passed", False):
        reasons.append("synthetic validation did not pass")
    if generated_audit.get("files_checked", 0) <= 0:
        reasons.append("no generated result artifacts were schema-validated")
    selected = set(protocol.get("research_questions", []))
    orders = protocol.get("ordering_variants", [])
    deltas = protocol.get("deltas", [])
    if not validation_only and orders != [
        "canonical", "reverse", "random_0", "random_1", "random_2", "random_3", "random_4"
    ]:
        reasons.append("ordering-sensitivity variants are not thesis-locked")
    required_public: list[Path] = [root / "metrics" / "synthetic_validation.json"]
    expected_checkpoints = 0
    if {"rq1", "rq2"} & selected:
        expected_checkpoints += len(orders) * len(deltas)
        required_public.extend(
            [
                root / "features" / "hf_baseline.parquet",
                root / "tables" / "rq1_hypothesis_tests.csv",
                root / "tables" / "rq2_delta_comparisons.csv",
                root / "metrics" / "hf_ordering_sensitivity.json",
            ]
        )
    if selected & REALCATS_RESEARCH_QUESTIONS:
        layer_count = len(protocol["datasets"]["realcats"]["artifacts"]["layers"])
        expected_checkpoints += layer_count * len(orders) * len(deltas)
        required_public.extend(
            [
                root / "tables" / "rq3_global_results.csv",
                root / "tables" / "rq3_label_results.csv",
                root / "tables" / "rq4_global_results.csv",
                root / "tables" / "rq4_category_results.csv",
                root / "metrics" / "realcats_ordering_sensitivity.json",
                root / "metrics" / "realcats_rq4_ordering_sensitivity.json",
            ]
        )
    if "blte" in selected:
        required_public.append(root / "metrics" / "blte_results.json")
    actual_checkpoints = (
        sum(path.is_dir() for path in (root / "checkpoints").iterdir())
        if (root / "checkpoints").exists()
        else 0
    )
    if actual_checkpoints != expected_checkpoints:
        reasons.append(
            f"expected {expected_checkpoints} committed motif checkpoints, found {actual_checkpoints}"
        )
    missing = [str(path.relative_to(root)) for path in required_public if not path.is_file()]
    if missing:
        reasons.append(f"required result artifacts are missing: {missing}")
    return not reasons, reasons


def _write_execution_manifest(run_root: Path, payload: dict) -> dict:
    path = run_root / "execution_manifest.json"
    sidecar = run_root / "execution_manifest.sha256"
    if path.exists():
        manifest = json.loads(path.read_text(encoding="utf-8"))
        for key in ("protocol_sha256", "execution_definition_sha256", "result_artifacts_sha256"):
            if manifest.get(key) != payload.get(key):
                raise RuntimeError(f"Existing execution manifest conflicts on {key}")
    else:
        manifest = payload
        write_json_exclusive(path, manifest)
    digest = protocol_digest(manifest)
    if sidecar.exists():
        if sidecar.read_text(encoding="ascii").strip() != digest:
            raise RuntimeError("Execution manifest sidecar conflicts with manifest")
    else:
        atomic_write_bytes_exclusive(sidecar, f"{digest}\n".encode("ascii"))
    return manifest


def execute_experiment(
    run_root: str | Path,
    *,
    resume: bool = False,
    threads: int | None = None,
    host_lock_path: str | Path | None = None,
    coordinator_id: str | None = None,
    coordinator_definition_sha256: str | None = None,
    authorized_host_lock_fd: int | None = None,
) -> dict:
    root = Path(run_root)
    paths, protocol = read_locked_protocol(root)
    protocol_sha = paths.protocol_hash.read_text(encoding="ascii").strip()

    # These imports are intentionally delayed until the CLI has set native pool limits.
    from src.experiments.resources import assert_preflight, preflight_runs

    host_path = Path(host_lock_path) if host_lock_path else default_host_lock_path()
    host_context = nullcontext()
    if authorized_host_lock_fd is None:
        host_context = FileLock(host_path)
    else:
        validate_inherited_lock(authorized_host_lock_fd, host_path)

    with host_context:
        assert_preflight(preflight_runs([root]))
        with FileLock(root / ".execution.lock"):
            _verify_locked_scientific_definition(protocol)
            for dataset in protocol["datasets"].values():
                _verify_prepared_artifacts(root, dataset)

            attempt = start_attempt(
                root,
                protocol_sha,
                threads,
                resume,
                mode="coordinated" if coordinator_id else "direct",
                coordinator_id=coordinator_id,
                coordinator_definition_sha256=coordinator_definition_sha256,
            )
            locked_threads = int(attempt.definition["thread_allocation"])
            configure_thread_environment(locked_threads)
            os.environ["THESIS_DUCKDB_MEMORY_GIB"] = str(
                attempt.definition["operational_model_parameters"]["duckdb_memory_gib"]
            )

            from src.experiments.telemetry import RunTelemetry

            telemetry = RunTelemetry(root)
            telemetry.emit(
                "execute",
                "started",
                attempt_id=attempt.attempt_id,
                resumed=resume,
                threads=locked_threads,
            )
            base_identity = {
                "protocol_sha256": protocol_sha,
                "execution_definition_sha256": attempt.definition_sha256,
            }
            stages = StageManager(paths, attempt.attempt_id, base_identity, telemetry)
            checkpoints = CheckpointStore(
                root, base_identity, attempt.attempt_id, telemetry
            )
            results: dict = {}
            try:
                from threadpoolctl import threadpool_limits

                from src.experiments.secondary import (
                    execute_blte,
                    execute_synthetic_validation,
                )
                from src.experiments.validation import (
                    validate_generated_results,
                    validate_prepared_inputs,
                )

                with threadpool_limits(limits=locked_threads):
                    prepared_audit = validate_prepared_inputs(root, protocol)
                    with telemetry.step("execute.validation"):
                        with stages.stage("synthetic-validation") as stage:
                            if not stage.reused:
                                validation_result = execute_synthetic_validation(
                                    stage.paths, telemetry=telemetry
                                )
                            else:
                                validation_result = json.loads(
                                    (paths.metrics / "synthetic_validation.json").read_text(
                                        encoding="utf-8"
                                    )
                                )
                    if not validation_result.get("passed", False):
                        raise RuntimeError("Synthetic motif validation gate failed")
                    results["validation"] = validation_result

                    selected = set(protocol["research_questions"])
                    scratch = attempt.directory / "scratch" / "duckdb"
                    if {"rq1", "rq2"} & selected:
                        from src.experiments.rq_binary import execute_hf_v5

                        results["huggingface"] = execute_hf_v5(
                            paths,
                            protocol,
                            stages,
                            checkpoints,
                            scratch,
                            telemetry=telemetry,
                        )
                    if REALCATS_RESEARCH_QUESTIONS <= selected:
                        from src.experiments.rq_multiclass import execute_realcats_v5

                        results["realcats"] = execute_realcats_v5(
                            paths,
                            protocol,
                            stages,
                            checkpoints,
                            scratch,
                            telemetry=telemetry,
                        )
                    if "blte" in selected:
                        with stages.stage("blte-secondary-evaluation") as stage:
                            if not stage.reused:
                                results["blte"] = execute_blte(
                                    stage.paths, protocol, telemetry=telemetry
                                )
                            else:
                                results["blte"] = json.loads(
                                    (paths.metrics / "blte_results.json").read_text(
                                        encoding="utf-8"
                                    )
                                )

                    generated_audit = validate_generated_results(root, protocol)
                    with stages.stage("final-validation") as stage:
                        if not stage.reused:
                            write_json_exclusive(
                                stage.paths.metrics / "execution_validation.json",
                                {
                                    "prepared": prepared_audit,
                                    "generated": generated_audit,
                                    "passed": True,
                                },
                            )

                inventory, inventory_sha = write_inventory(root)
                verify_inventory(root)
                citable, citable_reasons = _derive_citable(
                    root, protocol, validation_result, generated_audit
                )
                manifest_payload = {
                    "schema_version": 1,
                    "status": "complete",
                    "execution_profile": protocol["execution_profile"],
                    "citable": citable,
                    "citable_reasons": citable_reasons,
                    "completed_at_utc": datetime.now(timezone.utc).isoformat(),
                    "protocol_sha256": protocol_sha,
                    "execution_definition_sha256": attempt.definition_sha256,
                    "result_artifacts_sha256": inventory_sha,
                    "runtime": runtime_manifest(),
                    "thread_allocation": locked_threads,
                    "operational_model_parameters": attempt.definition[
                        "operational_model_parameters"
                    ],
                    "analytical_model_parameters": protocol["model_parameters"],
                    "scientific_validation": {
                        "prepared": prepared_audit,
                        "generated": generated_audit,
                    },
                    "results": results,
                }
                manifest = _write_execution_manifest(root, manifest_payload)
                finish_attempt(attempt, "completed")
                telemetry.emit(
                    "execute", "completed", manifest="execution_manifest.json"
                )
                return manifest
            except BaseException as error:
                finish_attempt(attempt, "failed", error)
                telemetry.emit(
                    "execute",
                    "failed",
                    attempt_id=attempt.attempt_id,
                    error_type=type(error).__name__,
                    error=str(error),
                )
                raise
