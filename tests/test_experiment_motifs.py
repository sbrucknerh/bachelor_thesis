from __future__ import annotations

import numpy as np
import pytest

from src.experiments.motifs import RaphtoryExperimentGraph
from src.experiments.synthetic_fixtures import fixture_for_role


def test_all_retained_raphtory_indices():
    for role_index in range(32):
        graph = RaphtoryExperimentGraph(fixture_for_role(role_index))
        counts = graph.count(2, ["A"])
        vector = counts.iloc[0].to_numpy()
        assert vector.sum() == 1
        assert vector[role_index] == 1


def test_delta_boundary_and_zero_row_retention():
    graph = RaphtoryExperimentGraph(fixture_for_role(0, start=100, step=5))
    assert graph.count(10, ["A"]).to_numpy().sum() == 1
    counts = graph.count(9, ["A", "Z"])
    assert counts.loc["A"].sum() == 0
    assert counts.loc["Z"].sum() == 0


@pytest.mark.parametrize(
    "variant",
    ["canonical", "reverse", "random_0", "random_1", "random_2", "random_3", "random_4"],
)
def test_duckdb_parquet_bulk_load_preserves_ordered_fingerprints(tmp_path, variant):
    events = fixture_for_role(25, start=100, step=0).sample(frac=1, random_state=7)
    events["transaction_index"] = np.nan
    events["block_number"] = 1
    path = tmp_path / "events.parquet"
    events.to_parquet(path, index=False)

    expected = RaphtoryExperimentGraph(events, variant).count(1, ["A"])
    observed = RaphtoryExperimentGraph.from_parquet(
        path, variant, tmp_path / "duckdb"
    ).count(1, ["A"])

    assert np.array_equal(expected.to_numpy(), observed.to_numpy())
