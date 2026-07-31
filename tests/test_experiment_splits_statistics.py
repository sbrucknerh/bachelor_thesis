from __future__ import annotations

import numpy as np
import pandas as pd

from src.experiments.modeling import (
    select_delta_maximum,
    select_delta_one_standard_error,
)
from src.experiments.splits import grouped_outer_split, repeated_group_folds
from src.experiments.statistics import (
    benjamini_hochberg,
    grouped_score_swap_test,
    stratified_group_bootstrap_delta,
    stratified_paired_bootstrap_macro_ap_delta,
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


def test_campaign_assignments_are_consistent_across_cohorts_and_folds():
    rows = []
    for cohort in ("prevalence", "controlled"):
        for label in (0, 1):
            for index in range(40):
                rows.append(
                    {
                        "address": f"{cohort}-{label}-{index}",
                        "cohort": cohort,
                        "label": label,
                        "experiment_stratum": f"{cohort}:{label}",
                        "campaign_group": (
                            f"shared-{label}-{index // 2}"
                            if index < 10
                            else f"singleton-{cohort}-{label}-{index}"
                        ),
                    }
                )
    targets = pd.DataFrame(rows)
    targets["split"] = grouped_outer_split(
        targets, stratum_column="experiment_stratum"
    )
    assert targets.groupby("campaign_group")["split"].nunique().max() == 1

    development = targets.loc[targets["split"] == "development"]
    folds = repeated_group_folds(
        development, seeds=(42,), stratum_column="experiment_stratum"
    ).merge(development[["address", "campaign_group"]], on="address")
    assert folds.groupby(["campaign_group", "repeat"])["fold"].nunique().max() == 1


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


def test_rq2_uses_strict_cv_maximum_and_retains_one_se_as_secondary():
    scores = pd.DataFrame(
        {
            "delta": [300, 300, 3_600, 3_600],
            "average_precision": [0.79, 0.79, 0.70, 0.90],
        }
    )

    maximum, maximum_summary = select_delta_maximum(scores)
    one_se, one_se_summary = select_delta_one_standard_error(scores)

    assert maximum == 3_600
    assert one_se == 300
    assert maximum_summary.loc[
        maximum_summary["delta"] == 3_600, "selected_primary"
    ].item()
    assert one_se_summary.loc[
        one_se_summary["delta"] == 300, "selected_one_se"
    ].item()


def test_rq4_macro_bootstrap_is_seed_reproducible():
    labels = np.array(["a", "a", "a", "b", "b", "b", "c", "c", "c"])
    classes = ["a", "b", "c"]
    reference = np.full((9, 3), 1 / 3)
    category_specific = np.array(
        [
            [0.90, 0.05, 0.05],
            [0.85, 0.10, 0.05],
            [0.80, 0.10, 0.10],
            [0.05, 0.90, 0.05],
            [0.10, 0.85, 0.05],
            [0.10, 0.80, 0.10],
            [0.05, 0.05, 0.90],
            [0.05, 0.10, 0.85],
            [0.10, 0.10, 0.80],
        ]
    )

    first = stratified_paired_bootstrap_macro_ap_delta(
        labels,
        classes,
        reference,
        category_specific,
        iterations=100,
        seed=7,
    )
    second = stratified_paired_bootstrap_macro_ap_delta(
        labels,
        classes,
        reference,
        category_specific,
        iterations=100,
        seed=7,
    )

    assert first == second
    assert first["observed_delta_macro_ap"] > 0
