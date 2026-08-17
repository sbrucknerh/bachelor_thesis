"""Read-only integrity and citability verification for experiment runs."""

from __future__ import annotations

import json
from pathlib import Path

from src.experiments.artifacts import protocol_digest, read_locked_protocol, sha256_file
from src.experiments.execution_artifacts import (
    PUBLIC_DIRECTORIES,
    canonical_checkpoint_hash,
    verify_inventory,
    verify_stage,
)
from src.experiments.constants import (
    DELTAS,
    N_BOOTSTRAPS,
    N_PERMUTATIONS,
    MIN_FULLY_VERIFIABLE_PROTOCOL_VERSION,
    PROTOCOL_VERSION,
    REALCATS_RESEARCH_QUESTIONS,
    SUPPORTED_FULL_PROTOCOL_VERSIONS,
    HF_CONFIRMATION_EXCLUSION_POLICY,
)


class VerificationError(RuntimeError):
    pass


def _derive_citable_read_only(root: Path, protocol: dict) -> tuple[bool, list[str]]:
    reasons = []
    selected = set(protocol.get("research_questions", []))
    validation_only = selected == {"validation"}
    if protocol.get("execution_profile") != "thesis":
        reasons.append("execution profile is not thesis")
    if not protocol.get("citable_candidate", False):
        reasons.append("protocol is not a citable candidate")
    if not validation_only and int(protocol.get("permutations", 0)) != N_PERMUTATIONS:
        reasons.append("permutation count is not thesis-locked")
    if not validation_only and int(protocol.get("bootstraps", 0)) != N_BOOTSTRAPS:
        reasons.append("bootstrap count is not thesis-locked")
    if not validation_only and tuple(protocol.get("deltas", [])) != DELTAS:
        reasons.append("delta grid is not thesis-locked")
    orders = protocol.get("ordering_variants", [])
    expected_orders = [
        "canonical", "reverse", "random_0", "random_1", "random_2", "random_3", "random_4"
    ]
    if not validation_only and orders != expected_orders:
        reasons.append("ordering-sensitivity variants are not thesis-locked")
    validation_path = root / "metrics" / "synthetic_validation.json"
    if not validation_path.is_file() or not json.loads(
        validation_path.read_text(encoding="utf-8")
    ).get("passed", False):
        reasons.append("synthetic validation gate did not pass")
    expected_checkpoints = 0
    if {"rq1", "rq2"} & selected:
        expected_checkpoints += len(orders) * len(protocol.get("deltas", []))
        version = int(protocol.get("protocol_version", 0))
        if version >= 7:
            triangle = protocol.get("hf_triangle_analysis", {})
            artifacts = protocol.get("datasets", {}).get("huggingface", {}).get(
                "artifacts", {}
            )
            if not triangle.get("enabled", False):
                reasons.append("HF triangle analysis is not enabled")
            if not artifacts.get("motif_events") or not artifacts.get(
                "motif_events_sha256"
            ):
                reasons.append("HF neighborhood graph is missing")
            if version == 7 and not protocol.get("replication", {}).get(
                "hf_exclusion_references", []
            ):
                reasons.append(
                    "protocol-v7 HF triangle extension has no prior-run exclusions"
                )
        if version >= 8:
            exclusions = protocol["datasets"]["huggingface"].get(
                "confirmation_exclusions", {}
            )
            if len(protocol.get("replication", {}).get(
                "hf_exclusion_references", []
            )) < int(HF_CONFIRMATION_EXCLUSION_POLICY["minimum_reference_runs"]):
                reasons.append("protocol-v8 HF exclusion run count is incomplete")
            if int(exclusions.get("unique_prior_node_ids", 0)) < int(
                HF_CONFIRMATION_EXCLUSION_POLICY["minimum_unique_prior_targets"]
            ):
                reasons.append(
                    "protocol-v8 HF prior-target exclusion coverage is incomplete"
                )
    if selected & REALCATS_RESEARCH_QUESTIONS:
        expected_checkpoints += (
            len(protocol["datasets"]["realcats"]["artifacts"]["layers"])
            * len(orders)
            * len(protocol.get("deltas", []))
        )
        if int(protocol.get("protocol_version", 0)) >= 8:
            expected_checkpoints += (
                len(
                    protocol["datasets"]["realcats"]["artifacts"].get(
                        "binary_layers", {}
                    )
                )
                * len(orders)
                * len(protocol.get("deltas", []))
            )
    actual_checkpoints = (
        sum(path.is_dir() for path in (root / "checkpoints").iterdir())
        if (root / "checkpoints").exists()
        else 0
    )
    if actual_checkpoints != expected_checkpoints:
        reasons.append("committed checkpoint cardinality is incomplete")
    return not reasons, reasons


def _verify_prepared(root: Path, protocol: dict) -> int:
    checked = 0
    for dataset in protocol.get("datasets", {}).values():
        artifacts = dataset.get("artifacts", {})
        for key, expected in artifacts.items():
            if not key.endswith("_sha256"):
                continue
            relative = artifacts.get(key.removesuffix("_sha256"))
            if isinstance(relative, str):
                if sha256_file(root / relative) != expected:
                    raise VerificationError(f"Prepared artifact hash mismatch: {relative}")
                checked += 1
            elif isinstance(relative, dict):
                if set(relative) != set(expected):
                    raise VerificationError(f"Prepared artifact keys mismatch: {key}")
                for name, path in relative.items():
                    if sha256_file(root / path) != expected[name]:
                        raise VerificationError(f"Prepared artifact hash mismatch: {path}")
                    checked += 1
    return checked


def _verify_definition(root: Path, protocol_sha: str) -> tuple[dict, str]:
    path = root / "execution_definition.json"
    sidecar = root / "execution_definition.sha256"
    definition = json.loads(path.read_text(encoding="utf-8"))
    digest = protocol_digest(definition)
    if sidecar.read_text(encoding="ascii").strip() != digest:
        raise VerificationError("Execution definition sidecar mismatch")
    if definition.get("protocol_sha256") != protocol_sha:
        raise VerificationError("Execution definition references another protocol")
    return definition, digest


def _verify_checkpoints(root: Path, protocol_sha: str, definition_sha: str) -> int:
    import pandas as pd

    checked = 0
    checkpoint_root = root / "checkpoints"
    if not checkpoint_root.exists():
        return checked
    for directory in sorted(checkpoint_root.iterdir()):
        if not directory.is_dir():
            raise VerificationError(f"Unexpected checkpoint entry: {directory}")
        metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
        identity = metadata.get("identity", {})
        if identity.get("protocol_sha256") != protocol_sha:
            raise VerificationError(f"Checkpoint protocol identity mismatch: {directory}")
        if identity.get("execution_definition_sha256") != definition_sha:
            raise VerificationError(f"Checkpoint execution identity mismatch: {directory}")
        if identity.get("checkpoint_id") != directory.name:
            raise VerificationError(f"Checkpoint ID mismatch: {directory}")
        frame = pd.read_parquet(directory / "counts.parquet")
        if canonical_checkpoint_hash(frame) != metadata.get("canonical_content_sha256"):
            raise VerificationError(f"Checkpoint canonical hash mismatch: {directory}")
        checked += 1
    return checked


def _verify_stages(root: Path, protocol_sha: str, definition_sha: str) -> int:
    checked = 0
    stage_root = root / "stages"
    if not stage_root.exists():
        return checked
    for directory in sorted(stage_root.iterdir()):
        manifest = verify_stage(directory)
        identity = manifest.get("identity", {})
        if identity.get("protocol_sha256") != protocol_sha:
            raise VerificationError(f"Stage protocol identity mismatch: {directory}")
        if identity.get("execution_definition_sha256") != definition_sha:
            raise VerificationError(f"Stage execution identity mismatch: {directory}")
        for relative in manifest["payload"]:
            source = directory / "payload" / relative
            public = root / relative
            if not public.is_file():
                raise VerificationError(f"Stage has no public artifact link: {public}")
            if source.stat().st_ino != public.stat().st_ino or source.stat().st_dev != public.stat().st_dev:
                raise VerificationError(f"Public artifact is not a stage hard link: {public}")
        checked += 1
    return checked


def verify_run(run_root: str | Path, require_citable: bool = False) -> dict:
    """Verify a run without creating locks, logs, attempts, or temporary paths."""
    root = Path(run_root)
    paths, protocol = read_locked_protocol(root)
    protocol_sha = paths.protocol_hash.read_text(encoding="ascii").strip()
    prepared = _verify_prepared(root, protocol)
    version = int(protocol.get("protocol_version", 1))
    if version < MIN_FULLY_VERIFIABLE_PROTOCOL_VERSION:
        report = {
            "integrity_ok": True,
            "protocol_version": version,
            "legacy_read_only": True,
            "completed": (root / "execution_manifest.json").exists(),
            "citable": False,
            "prepared_artifacts_checked": prepared,
        }
        if require_citable:
            raise VerificationError("Legacy protocol runs cannot be thesis-citable")
        return report
    if version not in SUPPORTED_FULL_PROTOCOL_VERSIONS:
        raise VerificationError(f"Unsupported protocol version: {version}")

    definition, definition_sha = _verify_definition(root, protocol_sha)
    checkpoints = _verify_checkpoints(root, protocol_sha, definition_sha)
    stages = _verify_stages(root, protocol_sha, definition_sha)
    verify_inventory(root)

    manifest_path = root / "execution_manifest.json"
    manifest_sidecar = root / "execution_manifest.sha256"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if protocol_digest(manifest) != manifest_sidecar.read_text(encoding="ascii").strip():
        raise VerificationError("Execution manifest sidecar mismatch")
    if manifest.get("protocol_sha256") != protocol_sha:
        raise VerificationError("Execution manifest protocol hash mismatch")
    if manifest.get("execution_definition_sha256") != definition_sha:
        raise VerificationError("Execution manifest definition hash mismatch")
    for key in ("source_sha256", "dependencies"):
        if manifest.get("runtime", {}).get(key) != protocol.get("runtime", {}).get(key):
            raise VerificationError(f"Execution runtime {key} differs from locked protocol")
    prepared_python = str(protocol.get("runtime", {}).get("python", "")).split(maxsplit=1)[0]
    executed_python = str(manifest.get("runtime", {}).get("python", "")).split(maxsplit=1)[0]
    if prepared_python != executed_python:
        raise VerificationError("Execution Python version differs from locked protocol")
    inventory_sha = (root / "result_artifacts.sha256").read_text(encoding="ascii").strip()
    if manifest.get("result_artifacts_sha256") != inventory_sha:
        raise VerificationError("Execution manifest inventory hash mismatch")
    from src.experiments.validation import (
        validate_generated_results,
        validate_prepared_inputs,
    )

    scientific = {
        "prepared": validate_prepared_inputs(root, protocol),
        "generated": validate_generated_results(root, protocol),
    }
    if manifest.get("scientific_validation") != scientific:
        raise VerificationError("Execution manifest scientific validation audit differs")
    citable, citable_reasons = _derive_citable_read_only(root, protocol)
    if bool(manifest.get("citable", False)) != citable:
        raise VerificationError("Execution manifest citability does not match derived state")
    if list(manifest.get("citable_reasons", [])) != citable_reasons:
        raise VerificationError("Execution manifest citability reasons do not match derived state")
    report = {
        "integrity_ok": True,
        "protocol_version": version,
        "legacy_read_only": version < PROTOCOL_VERSION,
        "completed": manifest.get("status") == "complete",
        "citable": citable,
        "prepared_artifacts_checked": prepared,
        "checkpoints_checked": checkpoints,
        "stages_checked": stages,
        "thread_allocation": definition["thread_allocation"],
    }
    if require_citable and not citable:
        reasons = "; ".join(citable_reasons)
        raise VerificationError(f"Run is integral but not thesis-citable: {reasons}")
    return report
