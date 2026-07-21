from __future__ import annotations

import json
from pathlib import Path
import zipfile

import pandas as pd
import pytest

import src.experiments.smoke as smoke
from scripts.run_results import build_parser
from src.experiments.artifacts import read_locked_protocol, write_json_exclusive
from src.experiments.realcats import convert_realcats_zip


def _record(tx_hash: str, source: str, destination: str) -> dict:
    return {
        "hash": tx_hash,
        "from": source,
        "to": destination,
        "timeStamp": "100",
        "blockNumber": "1",
        "transactionIndex": "0",
        "value": "1",
        "contractAddress": "0xcontract",
        "tokenID": "1",
        "tokenValue": "1",
    }


def _realcats_fixture(tmp_path: Path) -> tuple[Path, Path, tuple[str, ...]]:
    labels = ("Phishing Scam", "Impersonation Scam", "Hack Scam")
    rows = []
    archive_path = tmp_path / "realcats.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        for label_index, label in enumerate(labels):
            for address_index in range(2):
                address = f"0x{label_index}{address_index}"
                rows.append({"address": address, "label": label})
                prefix = f"ETHdata/ETHdata/{address}"
                for filename in smoke.REALCATS_LAYER_FILES.values():
                    archive.writestr(
                        f"{prefix}/{filename}",
                        json.dumps(
                            [_record(f"{address}-{filename}", address, "0xpeer")]
                        ),
                    )
    label_dir = tmp_path / "labels"
    label_dir.mkdir()
    pd.DataFrame(rows).to_csv(label_dir / "CE.tsv", sep="\t", index=False)
    return archive_path, label_dir, labels


def test_smoke_cli_has_isolated_defaults():
    args = build_parser().parse_args(["smoke", "--run-id", "check"])
    assert args.results_root == "results/smoke"
    assert args.sla_minutes == 60.0


def test_realcats_smoke_selection_is_balanced_deterministic_and_layered(tmp_path):
    archive, labels, exact_labels = _realcats_fixture(tmp_path)
    first, first_audit = smoke.select_realcats_smoke_addresses(
        archive,
        labels,
        labels_to_use=exact_labels,
        addresses_per_label=2,
        minimum_bytes=0,
        maximum_bytes=1_000_000,
    )
    second, second_audit = smoke.select_realcats_smoke_addresses(
        archive,
        labels,
        labels_to_use=exact_labels,
        addresses_per_label=2,
        minimum_bytes=0,
        maximum_bytes=1_000_000,
    )

    pd.testing.assert_frame_equal(first, second)
    assert first.groupby("type_label").size().eq(2).all()
    assert first_audit["selected_archive_entries_digest"] == second_audit[
        "selected_archive_entries_digest"
    ]


def test_realcats_conversion_can_be_bounded_to_selected_addresses(tmp_path):
    archive, labels, _ = _realcats_fixture(tmp_path)
    output = tmp_path / "converted"
    audit = convert_realcats_zip(
        archive,
        labels,
        output,
        minimum_free_gib=0,
        selected_addresses=["0x00"],
    )

    assert audit["address_selection_bounded"] is True
    assert audit["labelled_addresses"] == 1
    assert audit["release_labelled_addresses"] == 6
    assert set(pd.read_parquet(output / "normal.parquet")["from_address"]) == {
        "0x00"
    }


def test_smoke_protocol_is_non_citable_and_separate(monkeypatch, tmp_path):
    empty_dataset = {"artifacts": {}}
    monkeypatch.setattr(smoke, "_prepare_smoke_hf", lambda *args: empty_dataset)
    monkeypatch.setattr(
        smoke, "_prepare_smoke_realcats", lambda *args: empty_dataset
    )
    monkeypatch.setattr(smoke, "_prepare_blte", lambda *args, **kwargs: empty_dataset)

    root = smoke.prepare_smoke_experiment(
        tmp_path / "smoke",
        "profile",
        hf_snapshot=tmp_path / "hf",
        realcats_zip=tmp_path / "realcats.zip",
        realcats_labels=tmp_path / "labels",
        blte_dir=tmp_path / "blte",
    )
    _, protocol = read_locked_protocol(root)

    assert root.parent == tmp_path / "smoke"
    assert protocol["execution_profile"] == "smoke"
    assert protocol["citable"] is False
    assert protocol["deltas"] == list(smoke.SMOKE_DELTAS)
    assert protocol["ordering_variants"] == list(smoke.SMOKE_ORDER_VARIANTS)
    assert protocol["permutations"] == smoke.SMOKE_RESAMPLES
    assert (root / "SMOKE_ONLY.txt").exists()


def test_smoke_overrun_finishes_and_reports_failed_sla(monkeypatch, tmp_path):
    root = tmp_path / "smoke" / "overrun"

    def fake_prepare(results_root, run_id, **kwargs):
        root.mkdir(parents=True)
        return root

    def fake_execute(run_root):
        manifest = {"status": "complete", "citable": False}
        write_json_exclusive(Path(run_root) / "execution_manifest.json", manifest)
        return manifest

    times = iter([0.0, 61.0])
    monkeypatch.setattr(smoke, "prepare_smoke_experiment", fake_prepare)
    monkeypatch.setattr(smoke, "execute_experiment", fake_execute)

    _, _, summary = smoke.run_smoke_experiment(
        tmp_path / "smoke",
        "overrun",
        sla_seconds=60,
        clock=lambda: next(times),
    )

    assert summary["status"] == "complete"
    assert summary["sla_met"] is False
    persisted = json.loads((root / "smoke_summary.json").read_text())
    assert persisted["elapsed_seconds"] == 61.0


def test_smoke_failure_preserves_diagnostics(monkeypatch, tmp_path):
    root = tmp_path / "smoke" / "failed"

    def fake_prepare(results_root, run_id, **kwargs):
        root.mkdir(parents=True)
        raise ValueError("invalid smoke source")

    times = iter([0.0, 5.0])
    monkeypatch.setattr(smoke, "prepare_smoke_experiment", fake_prepare)

    with pytest.raises(ValueError, match="invalid smoke source"):
        smoke.run_smoke_experiment(
            tmp_path / "smoke",
            "failed",
            sla_seconds=60,
            clock=lambda: next(times),
        )

    summary = json.loads((root / "smoke_summary.json").read_text())
    assert summary["status"] == "failed"
    assert summary["error_type"] == "ValueError"
