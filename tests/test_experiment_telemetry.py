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
