from __future__ import annotations

import numpy as np
import pandas as pd

from src.experiments.features import (
    aggregate_features,
    aggregate_features_from_parquet,
    binary_motif_representations,
    hellinger_motif_composition,
)
from src.experiments.constants import INCOMING_STAR_ROLE_NAMES, MOTIF_ROLE_NAMES
from src.experiments.ordering import order_events
from src.experiments.synthetic_fixtures import fixture_for_role


def test_baseline_has_no_absolute_timestamps_and_keeps_inactive_target():
    events = fixture_for_role(0)
    features = aggregate_features(events, ["A", "B", "Z"])
    assert "Z" in features.index
    assert features.loc["Z"].sum() == 0
    assert not any("first" in column or "last" in column or "timestamp" in column for column in features)
    assert features.loc["A", "active_span"] == 2


def test_duckdb_baseline_matches_pandas_reference(tmp_path):
    events = pd.concat(
        [
            fixture_for_role(0),
            pd.DataFrame(
                {
                    "from_address": ["A", "C", "D"],
                    "to_address": ["A", "A", "A"],
                    "timestamp": [5, 8, 13],
                    "value": [2.5, np.nan, 4.0],
                    "tx_hash": ["self", "incoming", "incoming-2"],
                    "block_number": [1, 2, 3],
                    "transaction_index": [0, 0, 0],
                    "event_signature": ["self", "incoming", "incoming-2"],
                    "layer": ["normal", "normal", "normal"],
                }
            ),
        ],
        ignore_index=True,
    )
    path = tmp_path / "events.parquet"
    events.to_parquet(path, index=False)
    targets = ["A", "B", "Z"]
    expected = aggregate_features(events, targets)
    actual = aggregate_features_from_parquet(path, targets, tmp_path / "duckdb")
    pd.testing.assert_frame_equal(actual, expected, check_exact=False, rtol=1e-12, atol=1e-12)


def test_hellinger_is_layer_normalized_and_zero_safe():
    counts = pd.DataFrame(
        {"normal__a": [3, 0], "normal__b": [1, 0], "erc20__a": [0, 0]},
        index=["x", "y"],
    )
    transformed = hellinger_motif_composition(counts)
    assert np.isclose(transformed.loc["x", "normal__a"], np.sqrt(0.75))
    assert transformed.loc["y"].sum() == 0


def test_binary_motif_representations_separate_presence_volume_and_shape():
    counts = pd.DataFrame(
        0,
        index=["x", "y"],
        columns=MOTIF_ROLE_NAMES,
        dtype="int64",
    )
    counts.loc["x", "star_pre_iii"] = 3
    counts.loc["x", "star_post_iii"] = 1

    views = binary_motif_representations(counts, INCOMING_STAR_ROLE_NAMES)

    assert views["motif_presence"].loc["x", "motif_presence"] == 1
    assert views["motif_presence"].loc["y", "motif_presence"] == 0
    assert np.isclose(views["motif_volume"].loc["x", "motif_volume"], np.log1p(4))
    composition = views["motif_composition"]
    assert np.isclose(composition.loc["x", "star_pre_iii"], np.sqrt(0.75))
    assert np.isclose(composition.loc["x", "star_post_iii"], 0.5)
    assert composition.loc["y"].sum() == 0
    assert list(views["motif_incoming_stars"].columns) == list(
        INCOMING_STAR_ROLE_NAMES
    )


def test_reverse_changes_only_unresolved_same_block_order():
    events = fixture_for_role(25, start=100, step=0)
    events["block_number"] = 1
    events["transaction_index"] = np.nan
    canonical = order_events(events, "canonical")["event_signature"].tolist()
    reverse = order_events(events, "reverse")["event_signature"].tolist()
    assert reverse == canonical[::-1]

    events["transaction_index"] = [0, 1, 2]
    assert order_events(events, "canonical")["event_signature"].tolist() == order_events(
        events, "reverse"
    )["event_signature"].tolist()
