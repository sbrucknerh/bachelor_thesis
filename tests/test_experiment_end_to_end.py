from __future__ import annotations

import json

from src.experiments.execute import execute_experiment
from src.experiments.prepare import prepare_experiment


def test_validation_prepare_execute_schema(tmp_path):
    run_root = prepare_experiment(
        tmp_path,
        "validation",
        ["validation"],
        permutations=10,
        bootstraps=10,
    )
    manifest = execute_experiment(run_root)
    assert manifest["status"] == "complete"
    assert manifest["execution_profile"] == "thesis"
    assert manifest["citable"] is True
    assert manifest["results"]["validation"]["passed"] is True
    assert (run_root / "preparation_state.json").exists()
    assert (run_root / "protocol.json").exists()
    assert (run_root / "protocol.sha256").exists()
    assert (run_root / "execution_manifest.json").exists()
    records = [
        json.loads(line)
        for line in (run_root / "run_log.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert any(
        row["stage"] == "execute.validation" and row["status"] == "completed"
        for row in records
    )
    assert all("timestamp_utc" in row for row in records)
