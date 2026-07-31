from __future__ import annotations

import json

import pytest

from src.experiments.telemetry import RunTelemetry


def test_telemetry_records_completed_and_failed_steps(tmp_path):
    telemetry = RunTelemetry(tmp_path)
    with telemetry.step("successful", rows=3):
        pass
    with pytest.raises(ValueError, match="bad input"):
        with telemetry.step("failed"):
            raise ValueError("bad input")

    records = [
        json.loads(line)
        for line in telemetry.path.read_text(encoding="utf-8").splitlines()
    ]
    assert [record["status"] for record in records] == [
        "started",
        "completed",
        "started",
        "failed",
    ]
    assert records[1]["rows"] == 3
    assert records[1]["elapsed_seconds"] >= 0
    assert records[-1]["error_type"] == "ValueError"


def test_overall_progress_is_weighted_persisted_and_idempotent(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("THESIS_DISABLE_TQDM", "1")
    telemetry = RunTelemetry(tmp_path)
    telemetry.configure_overall(
        "test execution",
        {"first": 1.0, "second": 2.0},
        completed_items=set(),
        plan_sha256="a" * 64,
    )

    initial = json.loads(
        (tmp_path / "overall_progress.json").read_text(encoding="utf-8")
    )
    assert initial["percent_complete"] == 0.0
    assert initial["eta_seconds"] is None

    assert telemetry.mark_overall("first") is True
    assert telemetry.mark_overall("first") is False
    partial = telemetry.overall_snapshot()
    assert partial["completed_items"] == 1
    assert partial["completed_weight"] == 1.0
    assert partial["percent_complete"] == pytest.approx(100 / 3, abs=0.01)
    assert partial["eta_seconds"] is not None

    assert telemetry.mark_overall("second") is True
    telemetry.close_overall("completed")
    completed = json.loads(
        (tmp_path / "overall_progress.json").read_text(encoding="utf-8")
    )
    assert completed["status"] == "completed"
    assert completed["percent_complete"] == 100.0
    assert completed["eta_seconds"] == 0.0
    records = [
        json.loads(line)
        for line in telemetry.path.read_text(encoding="utf-8").splitlines()
    ]
    assert sum(row["stage"] == "execute.overall" for row in records) == 2
