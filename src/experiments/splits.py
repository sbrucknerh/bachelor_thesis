"""Locked group-aware and class-stratified split construction."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, train_test_split


def _distribution_distance(full: pd.Series, test: pd.Series) -> float:
    full_distribution = full.value_counts(normalize=True)
    test_distribution = test.value_counts(normalize=True)
    categories = full_distribution.index.union(test_distribution.index)
    return float(
        (full_distribution.reindex(categories, fill_value=0)
         - test_distribution.reindex(categories, fill_value=0)).abs().sum()
    )


def grouped_outer_split(
    targets: pd.DataFrame,
    stratum_column: str = "stratum",
    group_column: str = "campaign_group",
    seed: int = 42,
) -> pd.Series:
    """Choose the most balanced 20% fold without consulting model outcomes."""
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    strata = targets[stratum_column].astype(str)
    groups = targets[group_column].astype(str)
    best: tuple[float, np.ndarray] | None = None
    for _, test_indices in splitter.split(targets, strata, groups):
        size_error = abs(len(test_indices) / len(targets) - 0.2)
        distribution_error = _distribution_distance(strata, strata.iloc[test_indices])
        objective = size_error + distribution_error
        if best is None or objective < best[0]:
            best = (objective, test_indices)
    if best is None:
        raise RuntimeError("No valid grouped outer split could be constructed")
    assignment = pd.Series("development", index=targets.index, name="split")
    assignment.iloc[best[1]] = "test"
    assert_group_isolation(targets.assign(split=assignment), group_column, "split")
    return assignment


def repeated_group_folds(
    development_targets: pd.DataFrame,
    seeds: Iterable[int] = (42, 43, 44),
    stratum_column: str = "stratum",
    group_column: str = "campaign_group",
) -> pd.DataFrame:
    """Return long-form validation fold assignments for repeated grouped CV."""
    records: list[dict] = []
    strata = development_targets[stratum_column].astype(str)
    groups = development_targets[group_column].astype(str)
    for repeat, seed in enumerate(seeds):
        splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=int(seed))
        for fold, (_, validation_indices) in enumerate(
            splitter.split(development_targets, strata, groups)
        ):
            for index in validation_indices:
                records.append(
                    {
                        "address": str(development_targets.iloc[index]["address"]),
                        "repeat": repeat,
                        "seed": int(seed),
                        "fold": fold,
                    }
                )
    result = pd.DataFrame.from_records(records)
    expected = len(development_targets) * len(tuple(seeds))
    if len(result) != expected:
        raise RuntimeError(f"Incomplete CV assignments: expected {expected}, got {len(result)}")
    return result


def stratified_outer_split(
    targets: pd.DataFrame,
    label_column: str = "type_label",
    seed: int = 42,
) -> pd.Series:
    indices = np.arange(len(targets))
    development, test = train_test_split(
        indices,
        test_size=0.2,
        random_state=seed,
        stratify=targets[label_column],
    )
    assignment = pd.Series(index=targets.index, dtype="object", name="split")
    assignment.iloc[development] = "development"
    assignment.iloc[test] = "test"
    return assignment


def assert_group_isolation(
    assignments: pd.DataFrame,
    group_column: str = "campaign_group",
    split_column: str = "split",
) -> None:
    crossing = assignments.groupby(group_column)[split_column].nunique()
    offenders = crossing[crossing > 1]
    if len(offenders):
        raise RuntimeError(f"Groups straddle splits: {offenders.index.tolist()[:10]}")


def assert_disjoint_addresses(*frames: pd.DataFrame) -> None:
    observed: set[str] = set()
    for frame in frames:
        addresses = set(frame["address"].astype(str))
        overlap = observed & addresses
        if overlap:
            raise RuntimeError(f"Address overlap between cohorts: {sorted(overlap)[:10]}")
        observed.update(addresses)
