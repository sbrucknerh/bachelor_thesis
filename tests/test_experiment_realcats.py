from __future__ import annotations

import json
import zipfile

import pandas as pd
import pytest

import src.experiments.realcats as realcats
from src.experiments.realcats import convert_realcats_zip, normalize_record


def _record(tx_hash: str, source: str, destination: str, **extra):
    return {
        "hash": tx_hash,
        "from": source,
        "to": destination,
        "timeStamp": "100",
        "blockNumber": "1",
        "transactionIndex": "0",
        "value": "1",
        **extra,
    }


def test_failed_and_incomplete_records_are_rejected():
    assert normalize_record("normal", _record("x", "a", "b", isError="1")) is None
    assert normalize_record("normal", _record("x", "", "b")) is None


def test_ego_copy_dedup_preserves_maximum_token_multiplicity(tmp_path):
    labels = tmp_path / "labels"
    labels.mkdir()
    pd.DataFrame(
        {"address": ["0xa", "0xb"], "label": ["Phishing Scam", "Phishing Scam"]}
    ).to_csv(labels / "CE.tsv", sep="\t", index=False)
    archive_path = tmp_path / "realcats.zip"
    normal = _record("0xtx", "0xa", "0xb")
    token = _record("0xtoken", "0xa", "0xb", contractAddress="0xc")
    with zipfile.ZipFile(archive_path, "w") as archive:
        for address, token_rows in (("0xa", [token, token]), ("0xb", [token])):
            prefix = f"ETHdata/ETHdata/{address}"
            archive.writestr(f"{prefix}/normal_transactions.json", json.dumps([normal]))
            archive.writestr(f"{prefix}/ERC_20_transactions.json", json.dumps(token_rows))

    output = tmp_path / "converted"
    audit = convert_realcats_zip(
        archive_path, labels, output, minimum_free_gib=0
    )
    normal_events = pd.read_parquet(output / "normal.parquet")
    token_events = pd.read_parquet(output / "erc20.parquet")
    assert len(normal_events) == 1
    assert len(token_events) == 2
    assert audit["duplicate_copies_removed"] == 2


def test_conversion_resumes_only_before_protocol_lock(monkeypatch, tmp_path):
    labels = tmp_path / "labels"
    labels.mkdir()
    pd.DataFrame(
        {"address": ["0xa", "0xb"], "label": ["Hack Scam", "Hack Scam"]}
    ).to_csv(labels / "CE.tsv", sep="\t", index=False)
    archive_path = tmp_path / "realcats.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        for address in ("0xa", "0xb"):
            prefix = f"ETHdata/ETHdata/{address}"
            archive.writestr(
                f"{prefix}/normal_transactions.json",
                json.dumps([_record(f"tx-{address}", address, "0xpeer")]),
            )

    original_read = realcats._read_json
    calls = 0

    def interrupt_second_address(archive, name):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise KeyboardInterrupt
        return original_read(archive, name)

    output = tmp_path / "converted"
    monkeypatch.setattr(realcats, "_read_json", interrupt_second_address)
    with pytest.raises(KeyboardInterrupt):
        convert_realcats_zip(archive_path, labels, output, minimum_free_gib=0)

    monkeypatch.setattr(realcats, "_read_json", original_read)
    audit = convert_realcats_zip(
        archive_path, labels, output, minimum_free_gib=0, resume=True
    )

    assert audit["addresses_resumed"] == 1
    assert audit["normal_source_records"] == 2
    assert audit["normal_unique_events"] == 2
    assert len(pd.read_parquet(output / "normal.parquet")) == 2
    assert not (output / "event_index.sqlite").exists()


def test_corrupt_checkpoint_is_rejected_without_recovery(tmp_path):
    labels = tmp_path / "labels"
    labels.mkdir()
    pd.DataFrame({"address": ["0xa"], "label": ["Hack Scam"]}).to_csv(
        labels / "CE.tsv", sep="\t", index=False
    )
    archive_path = tmp_path / "realcats.zip"
    with zipfile.ZipFile(archive_path, "w"):
        pass
    output = tmp_path / "converted"
    output.mkdir()
    (output / "event_index.sqlite").write_bytes(b"not a sqlite database")

    with pytest.raises(RuntimeError, match="cannot be resumed safely"):
        convert_realcats_zip(
            archive_path,
            labels,
            output,
            minimum_free_gib=0,
            resume=True,
        )
