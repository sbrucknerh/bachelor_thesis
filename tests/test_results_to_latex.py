from __future__ import annotations

import json

import pytest

import scripts.results_to_latex as exporter


def _run(tmp_path, name, protocol, *, completed):
    root = tmp_path / name
    root.mkdir()
    (root / "protocol.json").write_text(json.dumps(protocol), encoding="utf-8")
    if completed:
        (root / "execution_manifest.json").write_text("{}\n", encoding="utf-8")
    return root


def test_exporter_uses_only_verified_completed_dataset_owner(tmp_path, monkeypatch):
    prepared = _run(
        tmp_path,
        "prepared",
        {
            "run_id": "prepared",
            "research_questions": ["rq1"],
            "datasets": {"huggingface": {"attrition": {"controlled_targets": 5_000}}},
        },
        completed=False,
    )
    smoke = _run(
        tmp_path,
        "smoke",
        {
            "run_id": "smoke",
            "execution_profile": "smoke",
            "research_questions": ["rq1"],
            "datasets": {"huggingface": {"attrition": {"controlled_targets": 100}}},
        },
        completed=True,
    )
    monkeypatch.setattr(
        exporter,
        "verify_run",
        lambda *_args, **_kwargs: {"completed": True, "citable": False},
    )

    runs = exporter.RunSet([prepared, smoke])

    assert runs.dataset("huggingface")["attrition"]["controlled_targets"] == 100
    assert runs.claimed == {"rq1"}
    assert runs.citable is False


def test_exporter_rejects_an_executed_run_that_fails_verification(tmp_path, monkeypatch):
    root = _run(
        tmp_path,
        "broken",
        {"run_id": "broken", "research_questions": ["rq1"], "datasets": {}},
        completed=True,
    )

    def fail(*_args, **_kwargs):
        raise RuntimeError("inventory mismatch")

    monkeypatch.setattr(exporter, "verify_run", fail)
    with pytest.raises(SystemExit, match="Run verification failed"):
        exporter.RunSet([root])


def test_exporter_does_not_fill_missing_citable_artifacts_from_smoke(
    tmp_path, monkeypatch
):
    thesis = _run(
        tmp_path,
        "thesis",
        {
            "run_id": "thesis",
            "execution_profile": "thesis",
            "research_questions": ["rq1"],
            "datasets": {"huggingface": {}},
        },
        completed=True,
    )
    smoke = _run(
        tmp_path,
        "smoke",
        {
            "run_id": "smoke",
            "execution_profile": "smoke",
            "research_questions": ["rq1"],
            "datasets": {"huggingface": {}},
        },
        completed=True,
    )
    for root in (thesis, smoke):
        for relative in exporter.REQUIRED["rq1"]:
            path = root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("placeholder\n", encoding="utf-8")
    (thesis / exporter.REQUIRED["rq1"][1]).unlink()

    monkeypatch.setattr(
        exporter,
        "verify_run",
        lambda root, **_kwargs: {
            "completed": True,
            "citable": root.name == "thesis",
        },
    )

    runs = exporter.RunSet([thesis, smoke])
    assert runs.owners["rq1"] == thesis
    with pytest.raises(SystemExit, match="thesis.*artifact"):
        runs.check_completeness()


def test_exporter_accepts_distinct_locked_replications(tmp_path, monkeypatch):
    first = _run(
        tmp_path,
        "first",
        {
            "run_id": "first",
            "execution_profile": "thesis",
            "research_questions": ["rq1", "rq2"],
            "replication": {"index": 1},
            "datasets": {"huggingface": {}},
        },
        completed=True,
    )
    second = _run(
        tmp_path,
        "second",
        {
            "run_id": "second",
            "execution_profile": "thesis",
            "research_questions": ["rq1", "rq2"],
            "replication": {"index": 2},
            "datasets": {"huggingface": {}},
        },
        completed=True,
    )
    monkeypatch.setattr(
        exporter,
        "verify_run",
        lambda *_args, **_kwargs: {"completed": True, "citable": True},
    )

    runs = exporter.RunSet([second, first])

    assert runs.sources_for("rq1") == [first, second]
    assert runs.sources_for("rq2") == [first, second]
    assert runs.owners["rq1"] == first


def test_exporter_rejects_equal_ranked_runs_without_replication_identity(
    tmp_path, monkeypatch
):
    first = _run(
        tmp_path,
        "first",
        {
            "run_id": "first",
            "execution_profile": "thesis",
            "research_questions": ["rq1"],
            "datasets": {"huggingface": {}},
        },
        completed=True,
    )
    second = _run(
        tmp_path,
        "second",
        {
            "run_id": "second",
            "execution_profile": "thesis",
            "research_questions": ["rq1"],
            "datasets": {"huggingface": {}},
        },
        completed=True,
    )
    monkeypatch.setattr(
        exporter,
        "verify_run",
        lambda *_args, **_kwargs: {"completed": True, "citable": True},
    )

    with pytest.raises(SystemExit, match="distinct locked replication indices"):
        exporter.RunSet([first, second])
