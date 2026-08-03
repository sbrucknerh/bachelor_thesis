from __future__ import annotations

import json
import os

import pandas as pd
import pytest

from src.experiments.artifacts import create_run_paths, write_json_exclusive
from src.experiments.constants import MOTIF_ROLE_NAMES
from src.experiments.execution_artifacts import (
    StageManager,
    canonical_checkpoint_hash,
    verify_stage,
)
from src.experiments.execution_control import execution_state, start_attempt
from src.experiments.prepare import prepare_experiment
from src.experiments.resources import preflight_runs
from src.experiments.validation import validate_prepared_inputs


def _counts(addresses=("0xb", "0xa")) -> pd.DataFrame:
    return pd.DataFrame(
        0,
        index=pd.Index(addresses, name="address"),
        columns=MOTIF_ROLE_NAMES,
        dtype="int64",
    )


def test_checkpoint_hash_has_canonical_address_order_and_types():
    first = _counts()
    second = first.iloc[::-1].copy()
    assert canonical_checkpoint_hash(first) == canonical_checkpoint_hash(second)
    first.iloc[0, 0] = -1
    with pytest.raises(ValueError, match="nonnegative"):
        canonical_checkpoint_hash(first)


def test_execution_definition_locks_resume_threads(tmp_path, monkeypatch):
    root = tmp_path / "run"
    root.mkdir()
    for name in ("features", "predictions", "metrics", "figures", "tables"):
        (root / name).mkdir()
    monkeypatch.setattr("src.experiments.execution_control.available_threads", lambda: 8)
    first = start_attempt(root, "a" * 64, 3, False)
    assert execution_state(root) == "interrupted"
    with pytest.raises(RuntimeError, match="locked value 3"):
        start_attempt(root, "a" * 64, 4, True)
    resumed = start_attempt(root, "a" * 64, None, True)
    assert resumed.definition["thread_allocation"] == 3
    assert resumed.attempt_id != first.attempt_id


def test_stage_publication_uses_verified_hard_links(tmp_path):
    paths = create_run_paths(tmp_path, "run")
    manager = StageManager(
        paths,
        "attempt",
        {"protocol_sha256": "p", "execution_definition_sha256": "e"},
    )
    with manager.stage("example") as stage:
        write_json_exclusive(stage.paths.metrics / "result.json", {"ok": True})
    committed = paths.root / "stages" / "example"
    verify_stage(committed)
    source = committed / "payload" / "metrics" / "result.json"
    public = paths.metrics / "result.json"
    assert (source.stat().st_dev, source.stat().st_ino) == (
        public.stat().st_dev,
        public.stat().st_ino,
    )


def test_preflight_command_is_read_only_for_validation_run(tmp_path):
    root = prepare_experiment(tmp_path, "validation", ["validation"])
    before = {
        str(path.relative_to(root)): (path.stat().st_mtime_ns, path.stat().st_size)
        for path in root.rglob("*")
        if path.is_file()
    }
    report = preflight_runs([root])
    after = {
        str(path.relative_to(root)): (path.stat().st_mtime_ns, path.stat().st_size)
        for path in root.rglob("*")
        if path.is_file()
    }
    assert report["admissible"] is True
    assert before == after
    assert not (root / ".execution.lock").exists()


def test_blte_must_be_a_separate_run(tmp_path):
    with pytest.raises(ValueError, match="separate optional secondary"):
        prepare_experiment(tmp_path, "mixed", ["rq1", "blte"])


def test_realcats_rq3_and_rq4_must_be_locked_together(tmp_path):
    with pytest.raises(ValueError, match="RQ3 and RQ4"):
        prepare_experiment(tmp_path, "rq3-only", ["rq3"])
    with pytest.raises(ValueError, match="RQ3 and RQ4"):
        prepare_experiment(tmp_path, "rq4-only", ["rq4"])


def test_hf_confirmation_requires_prior_run_exclusion(tmp_path):
    with pytest.raises(ValueError, match="must exclude at least one prior HF run"):
        prepare_experiment(
            tmp_path,
            "confirmation",
            ["rq1", "rq2"],
            replication_index=1,
        )


def test_hf_initial_analysis_rejects_exclusion_reference(tmp_path):
    with pytest.raises(ValueError, match="replication-index greater than zero"):
        prepare_experiment(
            tmp_path,
            "initial",
            ["rq1", "rq2"],
            exclude_hf_runs=[tmp_path / "prior"],
        )


def test_resume_reuses_committed_stage_after_finalization_failure(tmp_path, monkeypatch):
    import src.experiments.execute as execution
    from src.experiments.artifacts import sha256_file

    root = prepare_experiment(tmp_path, "resume", ["validation"])
    original = execution._write_execution_manifest
    calls = 0

    def interrupt_once(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise KeyboardInterrupt("finalization interrupted")
        return original(*args, **kwargs)

    monkeypatch.setattr(execution, "_write_execution_manifest", interrupt_once)
    with pytest.raises(KeyboardInterrupt, match="finalization interrupted"):
        execution.execute_experiment(root, threads=1)

    stage_file = root / "stages" / "synthetic-validation" / "payload" / "metrics" / "synthetic_validation.json"
    before = sha256_file(stage_file)
    manifest = execution.execute_experiment(root, resume=True, threads=1)
    after = sha256_file(stage_file)

    assert manifest["status"] == "complete"
    assert before == after
    assert len(list((root / "attempts").iterdir())) == 2


def _prepared_validation_fixture(tmp_path, targets, folds):
    root = tmp_path / "prepared"
    (root / "data").mkdir(parents=True)
    targets.to_parquet(root / "data" / "targets.parquet", index=False)
    folds.to_parquet(root / "data" / "folds.parquet", index=False)
    protocol = {
        "execution_profile": "thesis",
        "datasets": {
            "fixture": {
                "artifacts": {
                    "targets": "data/targets.parquet",
                    "folds": "data/folds.parquet",
                }
            }
        },
    }
    return root, protocol


def test_prepared_guard_rejects_cv_fold_missing_a_class(tmp_path):
    targets = pd.DataFrame(
        {
            "address": [f"a{index}" for index in range(10)],
            "label": [0, 1] * 5,
            "campaign_group": [f"g{index}" for index in range(10)],
            "split": ["development"] * 8 + ["test"] * 2,
        }
    )
    folds = pd.DataFrame(
        {
            "address": [f"a{index}" for index in range(8)],
            "fold": [0, 1] * 4,
        }
    )
    root, protocol = _prepared_validation_fixture(tmp_path, targets, folds)
    with pytest.raises(RuntimeError, match="missing required classes"):
        validate_prepared_inputs(root, protocol)


def test_prepared_guard_checks_campaign_isolation_without_repeat_column(tmp_path):
    targets = pd.DataFrame(
        {
            "address": [f"a{index}" for index in range(10)],
            "label": [0, 1] * 5,
            "campaign_group": ["shared", "shared", *[f"g{index}" for index in range(2, 10)]],
            "split": ["development"] * 8 + ["test"] * 2,
        }
    )
    folds = pd.DataFrame(
        {
            "address": [f"a{index}" for index in range(8)],
            "fold": [0, 1, 0, 1, 0, 1, 0, 1],
        }
    )
    root, protocol = _prepared_validation_fixture(tmp_path, targets, folds)
    with pytest.raises(RuntimeError, match="campaign groups straddle CV folds"):
        validate_prepared_inputs(root, protocol)
