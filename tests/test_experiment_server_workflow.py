from __future__ import annotations

import json
from pathlib import Path

from src.experiments.server_workflow import run_thesis_workflow


def test_one_command_workflow_prepares_parallel_roots_and_verifies(
    tmp_path,
    monkeypatch,
):
    snapshot = tmp_path / "hf"
    (snapshot / "gnn_dataset/edges_all").mkdir(parents=True)
    (snapshot / "gnn_dataset/labels").mkdir(parents=True)
    (snapshot / "gnn_dataset/edges_all/edges.parquet").write_bytes(b"edges")
    (snapshot / "gnn_dataset/labels/targets_global.parquet").write_bytes(
        b"labels"
    )
    (snapshot / "addr_labels_balanced.csv.zst").write_bytes(b"descriptions")
    realcats_zip = tmp_path / "real-cats.zip"
    realcats_zip.write_bytes(b"zip")
    realcats_labels = tmp_path / "labels"
    realcats_labels.mkdir()
    (realcats_labels / "CE.tsv").write_text("address\tlabel\n")
    states: dict[str, str] = {}
    prepared_scopes: dict[str, set[str]] = {}

    def fake_prepare(results_root, run_id, rqs, **_kwargs):
        root = Path(results_root) / run_id
        root.mkdir(parents=True)
        (root / "protocol.json").write_text(
            json.dumps(
                {
                    "protocol_version": 5,
                    "research_questions": sorted(rqs),
                    "execution_profile": "thesis",
                }
            )
        )
        states[str(root)] = "fresh"
        prepared_scopes[run_id] = set(rqs)
        return root

    def fake_state(root):
        return states[str(Path(root))]

    def fake_parallel(hf_root, realcats_root, **kwargs):
        assert kwargs["total_threads"] == 20
        states[str(Path(hf_root))] = "completed"
        states[str(Path(realcats_root))] = "completed"
        return {"status": "complete"}

    monkeypatch.setattr(
        "src.experiments.prepare.prepare_experiment",
        fake_prepare,
    )
    monkeypatch.setattr(
        "src.experiments.execution_control.available_threads",
        lambda: 20,
    )
    monkeypatch.setattr(
        "src.experiments.execution_control.execution_state",
        fake_state,
    )
    monkeypatch.setattr(
        "src.experiments.resources.preflight_runs",
        lambda *_args, **_kwargs: {"admissible": True},
    )
    monkeypatch.setattr(
        "src.experiments.coordination.execute_parallel",
        fake_parallel,
    )
    monkeypatch.setattr(
        "src.experiments.verify.verify_run",
        lambda *_args, **_kwargs: {
            "integrity_ok": True,
            "completed": True,
            "citable": True,
        },
    )

    outcome = run_thesis_workflow(
        results_root=tmp_path / "results",
        run_id="final",
        hf_snapshot=snapshot,
        realcats_zip=realcats_zip,
        realcats_labels=realcats_labels,
        threads=20,
    )

    assert outcome["status"] == "complete"
    assert outcome["execution_mode"] == "parallel"
    assert prepared_scopes["final_hf"] == {"rq1", "rq2"}
    assert prepared_scopes["final_realcats"] == {"rq3", "rq4"}
