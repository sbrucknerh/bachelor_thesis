from __future__ import annotations

import numpy as np
import pandas as pd

from src.experiments.splits import grouped_outer_split
from src.experiments.statistics import (
    benjamini_hochberg,
    grouped_score_swap_test,
    stratified_group_bootstrap_delta,
)


def test_grouped_split_never_splits_campaign():
    rows = []
    for label in (0, 1):
        for index in range(50):
            rows.append(
                {
                    "address": f"{label}-{index}",
                    "stratum": str(label),
                    "campaign_group": f"group-{label}-{index // 2}",
                }
            )
    targets = pd.DataFrame(rows)
    targets["split"] = grouped_outer_split(targets)
    assert targets.groupby("campaign_group")["split"].nunique().max() == 1


def test_statistical_tests_are_seed_reproducible():
    y = np.array([0, 0, 0, 1, 1, 1])
    baseline = np.array([0.1, 0.4, 0.2, 0.6, 0.5, 0.7])
    combined = np.array([0.05, 0.2, 0.1, 0.8, 0.7, 0.9])
    groups = np.array(["a", "b", "c", "d", "e", "f"])
    first = grouped_score_swap_test(y, baseline, combined, groups, iterations=100, seed=7)
    second = grouped_score_swap_test(y, baseline, combined, groups, iterations=100, seed=7)
    assert first == second
    first_ci = stratified_group_bootstrap_delta(
        y, baseline, combined, groups, iterations=100, seed=7
    )
    second_ci = stratified_group_bootstrap_delta(
        y, baseline, combined, groups, iterations=100, seed=7
    )
    assert first_ci == second_ci


def test_benjamini_hochberg_is_monotone_in_rank():
    adjusted = benjamini_hochberg([0.01, 0.04, 0.03, 0.8])
    assert np.allclose(adjusted, [0.04, 0.05333333333333334, 0.05333333333333334, 0.8])
