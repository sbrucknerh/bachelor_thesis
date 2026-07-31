"""Prespecified metrics, randomization tests, bootstraps, and multiplicity control."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import label_binarize


def binary_metrics(y_true: Iterable[int], scores: Iterable[float], threshold: float = 0.5) -> dict:
    y = np.asarray(y_true, dtype=int)
    probability = np.asarray(scores, dtype=float)
    prediction = (probability >= threshold).astype(int)
    result = {
        "average_precision": float(average_precision_score(y, probability)),
        "f1": float(f1_score(y, prediction, zero_division=0)),
        "precision": float(precision_score(y, prediction, zero_division=0)),
        "recall": float(recall_score(y, prediction, zero_division=0)),
    }
    result["roc_auc"] = float(roc_auc_score(y, probability)) if len(np.unique(y)) == 2 else None
    return result


def multiclass_metrics(
    y_true: Iterable[str],
    probabilities: np.ndarray,
    classes: Iterable[str],
) -> dict:
    y = np.asarray(y_true)
    class_values = np.asarray(list(classes))
    encoded = label_binarize(y, classes=class_values)
    prediction = class_values[np.argmax(probabilities, axis=1)]
    per_label = {
        str(label): float(average_precision_score(encoded[:, index], probabilities[:, index]))
        for index, label in enumerate(class_values)
    }
    return {
        "macro_average_precision": float(np.mean(list(per_label.values()))),
        "macro_f1": float(f1_score(y, prediction, average="macro", zero_division=0)),
        "per_label_average_precision": per_label,
    }


def grouped_score_swap_test(
    y_true: Iterable[int],
    baseline_scores: Iterable[float],
    combined_scores: Iterable[float],
    groups: Iterable[str],
    iterations: int = 10_000,
    seed: int = 42,
) -> dict:
    """Paired randomization test that swaps complete campaign score vectors."""
    y = np.asarray(y_true, dtype=int)
    baseline = np.asarray(baseline_scores, dtype=float)
    combined = np.asarray(combined_scores, dtype=float)
    group_values = np.asarray(groups, dtype=str)
    unique_groups, group_index = np.unique(group_values, return_inverse=True)
    observed = average_precision_score(y, combined) - average_precision_score(y, baseline)
    rng = np.random.default_rng(seed)
    null = np.empty(iterations, dtype=float)
    for iteration in range(iterations):
        swap = rng.integers(0, 2, size=len(unique_groups), dtype=np.int8)[group_index].astype(bool)
        permuted_baseline = np.where(swap, combined, baseline)
        permuted_combined = np.where(swap, baseline, combined)
        null[iteration] = (
            average_precision_score(y, permuted_combined)
            - average_precision_score(y, permuted_baseline)
        )
    p_value = (1 + int(np.sum(null >= observed))) / (iterations + 1)
    return {
        "observed_delta_ap": float(observed),
        "p_value_one_sided": float(p_value),
        "iterations": iterations,
        "seed": seed,
    }


def stratified_group_bootstrap_delta(
    y_true: Iterable[int],
    baseline_scores: Iterable[float],
    combined_scores: Iterable[float],
    groups: Iterable[str],
    iterations: int = 10_000,
    seed: int = 42,
) -> dict:
    """Bootstrap whole groups independently within each binary class."""
    frame = pd.DataFrame(
        {
            "y": np.asarray(y_true, dtype=int),
            "baseline": np.asarray(baseline_scores, dtype=float),
            "combined": np.asarray(combined_scores, dtype=float),
            "group": np.asarray(groups, dtype=str),
        }
    )
    group_labels = frame.groupby("group")["y"].nunique()
    if (group_labels > 1).any():
        raise ValueError("Each bootstrap group must belong to one outcome class")
    groups_by_class = {
        label: frame.loc[frame["y"] == label, "group"].unique()
        for label in sorted(frame["y"].unique())
    }
    row_indices = {group: values.index.to_numpy() for group, values in frame.groupby("group")}
    rng = np.random.default_rng(seed)
    deltas = np.empty(iterations, dtype=float)
    for iteration in range(iterations):
        sampled_indices: list[int] = []
        for class_groups in groups_by_class.values():
            sampled = rng.choice(class_groups, size=len(class_groups), replace=True)
            for group in sampled:
                sampled_indices.extend(row_indices[group])
        sample = frame.iloc[sampled_indices]
        deltas[iteration] = (
            average_precision_score(sample["y"], sample["combined"])
            - average_precision_score(sample["y"], sample["baseline"])
        )
    lower, upper = np.quantile(deltas, [0.025, 0.975])
    p_value = (1 + int(np.sum(deltas <= 0))) / (iterations + 1)
    return {
        "observed_delta_ap": float(
            average_precision_score(frame["y"], frame["combined"])
            - average_precision_score(frame["y"], frame["baseline"])
        ),
        "ci_95": [float(lower), float(upper)],
        "p_value_one_sided": float(p_value),
        "iterations": iterations,
        "seed": seed,
    }


def stratified_group_bootstrap_ap(
    y_true: Iterable[int],
    scores: Iterable[float],
    groups: Iterable[str],
    iterations: int = 10_000,
    seed: int = 42,
) -> dict:
    frame = pd.DataFrame(
        {
            "y": np.asarray(y_true, dtype=int),
            "score": np.asarray(scores, dtype=float),
            "group": np.asarray(groups, dtype=str),
        }
    )
    if (frame.groupby("group")["y"].nunique() > 1).any():
        raise ValueError("Each bootstrap group must belong to one outcome class")
    groups_by_class = {
        label: frame.loc[frame["y"] == label, "group"].unique()
        for label in sorted(frame["y"].unique())
    }
    row_indices = {group: values.index.to_numpy() for group, values in frame.groupby("group")}
    rng = np.random.default_rng(seed)
    values = np.empty(iterations, dtype=float)
    for iteration in range(iterations):
        sampled_indices: list[int] = []
        for class_groups in groups_by_class.values():
            sampled = rng.choice(class_groups, size=len(class_groups), replace=True)
            for group in sampled:
                sampled_indices.extend(row_indices[group])
        sample = frame.iloc[sampled_indices]
        values[iteration] = average_precision_score(sample["y"], sample["score"])
    lower, upper = np.quantile(values, [0.025, 0.975])
    return {"ci_95": [float(lower), float(upper)], "iterations": iterations, "seed": seed}


def stratified_paired_bootstrap_delta(
    y_binary: Iterable[int],
    reference_scores: Iterable[float],
    alternative_scores: Iterable[float],
    iterations: int = 10_000,
    seed: int = 42,
) -> dict:
    y = np.asarray(y_binary, dtype=int)
    reference = np.asarray(reference_scores, dtype=float)
    alternative = np.asarray(alternative_scores, dtype=float)
    class_indices = {label: np.flatnonzero(y == label) for label in np.unique(y)}
    rng = np.random.default_rng(seed)
    deltas = np.empty(iterations, dtype=float)
    for iteration in range(iterations):
        sample = np.concatenate(
            [rng.choice(indices, size=len(indices), replace=True) for indices in class_indices.values()]
        )
        deltas[iteration] = (
            average_precision_score(y[sample], alternative[sample])
            - average_precision_score(y[sample], reference[sample])
        )
    lower, upper = np.quantile(deltas, [0.025, 0.975])
    p_value = (1 + int(np.sum(deltas <= 0))) / (iterations + 1)
    return {
        "observed_delta_ap": float(
            average_precision_score(y, alternative) - average_precision_score(y, reference)
        ),
        "ci_95": [float(lower), float(upper)],
        "p_value_one_sided": float(p_value),
        "iterations": iterations,
        "seed": seed,
    }


def stratified_paired_bootstrap_macro_ap_delta(
    y_true: Iterable[str],
    classes: Iterable[str],
    reference_scores: np.ndarray,
    category_specific_scores: np.ndarray,
    iterations: int = 10_000,
    seed: int = 42,
) -> dict:
    """Compare one global window with development-selected category windows."""
    y = np.asarray(y_true)
    class_values = np.asarray(list(classes))
    reference = np.asarray(reference_scores, dtype=float)
    category_specific = np.asarray(category_specific_scores, dtype=float)
    expected_shape = (len(y), len(class_values))
    if reference.shape != expected_shape or category_specific.shape != expected_shape:
        raise ValueError(
            "Multiclass score matrices must have shape "
            f"{expected_shape}, got {reference.shape} and {category_specific.shape}"
        )
    class_indices = {
        label: np.flatnonzero(y == label) for label in class_values
    }
    if any(len(indices) == 0 for indices in class_indices.values()):
        raise ValueError("Every class must occur in the bootstrap sample")

    def macro_ap(indices: np.ndarray) -> float:
        encoded = label_binarize(y[indices], classes=class_values)
        reference_aps = [
            average_precision_score(encoded[:, column], reference[indices, column])
            for column in range(len(class_values))
        ]
        category_aps = [
            average_precision_score(
                encoded[:, column], category_specific[indices, column]
            )
            for column in range(len(class_values))
        ]
        return float(np.mean(category_aps) - np.mean(reference_aps))

    full = np.arange(len(y))
    observed = macro_ap(full)
    rng = np.random.default_rng(seed)
    deltas = np.empty(iterations, dtype=float)
    for iteration in range(iterations):
        sample = np.concatenate(
            [
                rng.choice(indices, size=len(indices), replace=True)
                for indices in class_indices.values()
            ]
        )
        deltas[iteration] = macro_ap(sample)
    lower, upper = np.quantile(deltas, [0.025, 0.975])
    p_value = (1 + int(np.sum(deltas <= 0))) / (iterations + 1)
    return {
        "observed_delta_macro_ap": observed,
        "ci_95": [float(lower), float(upper)],
        "p_value_one_sided": float(p_value),
        "iterations": iterations,
        "seed": seed,
    }


def permute_multiclass_labels(
    y_true: Iterable[str],
    probabilities: np.ndarray,
    classes: Iterable[str],
    iterations: int = 10_000,
    seed: int = 42,
) -> dict:
    y = np.asarray(y_true)
    class_values = np.asarray(list(classes))
    observed = multiclass_metrics(y, probabilities, class_values)["macro_average_precision"]
    rng = np.random.default_rng(seed)
    null = np.empty(iterations, dtype=float)
    for iteration in range(iterations):
        permuted = rng.permutation(y)
        encoded = label_binarize(permuted, classes=class_values)
        aps = [
            average_precision_score(encoded[:, index], probabilities[:, index])
            for index in range(len(class_values))
        ]
        null[iteration] = np.mean(aps)
    p_value = (1 + int(np.sum(null >= observed))) / (iterations + 1)
    return {
        "observed_macro_ap": float(observed),
        "p_value_one_sided": float(p_value),
        "iterations": iterations,
        "seed": seed,
    }


def benjamini_hochberg(p_values: Iterable[float]) -> np.ndarray:
    values = np.asarray(list(p_values), dtype=float)
    if np.any((values < 0) | (values > 1)):
        raise ValueError("p-values must lie in [0, 1]")
    order = np.argsort(values)
    ranked = values[order] * len(values) / np.arange(1, len(values) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    adjusted = np.empty_like(ranked)
    adjusted[order] = np.clip(ranked, 0, 1)
    return adjusted
