"""Fixed model definitions and development-only temporal-window selection."""

from __future__ import annotations

from collections.abc import Iterable
import os

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.experiments.constants import (
    DEFAULT_RF_PARAMS,
    DEFAULT_XGB_PARAMS,
    DELTA_MEAN_TIE_TOLERANCE,
    MULTICLASS_LOGISTIC_PARAMS,
)


def make_binary_model(model_type: str, y_train: pd.Series | np.ndarray):
    threads = max(1, int(os.environ.get("THESIS_MODEL_THREADS", "1")))
    if model_type == "random_forest":
        return RandomForestClassifier(**DEFAULT_RF_PARAMS, n_jobs=threads)
    if model_type == "xgboost":
        from xgboost import XGBClassifier

        y = np.asarray(y_train, dtype=int)
        positive = int((y == 1).sum())
        negative = int((y == 0).sum())
        params = dict(DEFAULT_XGB_PARAMS)
        params["n_jobs"] = threads
        params["scale_pos_weight"] = negative / positive if positive else 1.0
        return XGBClassifier(**params)
    raise ValueError(f"Unknown binary model: {model_type}")


def make_multiclass_model() -> Pipeline:
    return Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "model",
                LogisticRegression(**MULTICLASS_LOGISTIC_PARAMS),
            ),
        ]
    )


def fit_binary_scores(
    train_features: pd.DataFrame,
    train_labels: pd.Series,
    test_features: pd.DataFrame,
    model_type: str = "random_forest",
) -> tuple[object, np.ndarray]:
    model = make_binary_model(model_type, train_labels)
    model.fit(train_features, train_labels)
    return model, model.predict_proba(test_features)[:, 1]


def cv_delta_scores(
    features_by_delta: dict[int, pd.DataFrame],
    labels: pd.Series,
    fold_assignments: pd.DataFrame,
    model_type: str = "random_forest",
) -> pd.DataFrame:
    """Evaluate each delta on identical persisted repeated-CV folds."""
    labels = labels.copy()
    labels.index = labels.index.astype(str)
    records: list[dict] = []
    for (repeat, fold), validation_rows in fold_assignments.groupby(["repeat", "fold"]):
        validation_addresses = set(validation_rows["address"].astype(str))
        validation = labels.index.isin(validation_addresses)
        training = ~validation
        for delta, features in features_by_delta.items():
            aligned = features.reindex(labels.index)
            _, scores = fit_binary_scores(
                aligned.loc[training], labels.loc[training], aligned.loc[validation], model_type
            )
            records.append(
                {
                    "repeat": int(repeat),
                    "fold": int(fold),
                    "delta": int(delta),
                    "average_precision": float(
                        average_precision_score(labels.loc[validation], scores)
                    ),
                }
            )
    return pd.DataFrame.from_records(records)


def _summarize_delta_scores(cv_scores: pd.DataFrame) -> pd.DataFrame:
    summary = cv_scores.groupby("delta")["average_precision"].agg(["mean", "std", "count"])
    summary["standard_error"] = summary["std"] / np.sqrt(summary["count"])
    return summary


def select_delta_maximum(
    cv_scores: pd.DataFrame,
    tie_tolerance: float = DELTA_MEAN_TIE_TOLERANCE,
) -> tuple[int, pd.DataFrame]:
    """Select the development-CV mean-AP maximum with a locked tie rule."""
    if tie_tolerance < 0:
        raise ValueError("tie_tolerance must be nonnegative")
    summary = _summarize_delta_scores(cv_scores)
    maximum = float(summary["mean"].max())
    tied = summary.index[
        np.isclose(summary["mean"], maximum, rtol=0.0, atol=tie_tolerance)
    ]
    selected = int(min(tied))
    summary["is_maximum"] = summary.index.isin(tied)
    summary["selected_primary"] = summary.index == selected
    return selected, summary.reset_index()


def select_delta_one_standard_error(cv_scores: pd.DataFrame) -> tuple[int, pd.DataFrame]:
    summary = _summarize_delta_scores(cv_scores)
    best_delta = int(summary["mean"].idxmax())
    cutoff = float(summary.loc[best_delta, "mean"] - summary.loc[best_delta, "standard_error"])
    qualifying = summary.index[summary["mean"] >= cutoff]
    selected = int(min(qualifying))
    summary["within_one_se"] = summary.index.isin(qualifying)
    summary["selected_one_se"] = summary.index == selected
    return selected, summary.reset_index()


def select_multiclass_deltas(
    features_by_delta: dict[int, pd.DataFrame],
    labels: pd.Series,
    folds: Iterable[tuple[np.ndarray, np.ndarray]],
) -> tuple[int, dict[str, int], pd.DataFrame]:
    """Select global and one-vs-rest deltas using development folds only."""
    classes = sorted(labels.unique())
    records: list[dict] = []
    for fold, (training, validation) in enumerate(folds):
        for delta, features in features_by_delta.items():
            aligned = features.reindex(labels.index)
            model = make_multiclass_model()
            model.fit(aligned.iloc[training], labels.iloc[training])
            probabilities = model.predict_proba(aligned.iloc[validation])
            model_classes = list(model.named_steps["model"].classes_)
            per_class = []
            for label in classes:
                column = model_classes.index(label)
                binary = (labels.iloc[validation] == label).astype(int)
                score = average_precision_score(binary, probabilities[:, column])
                per_class.append(score)
                records.append(
                    {"fold": fold, "delta": int(delta), "label": label, "average_precision": score}
                )
            records.append(
                {"fold": fold, "delta": int(delta), "label": "__macro__", "average_precision": np.mean(per_class)}
            )
    scores = pd.DataFrame.from_records(records)
    means = scores.groupby(["label", "delta"])["average_precision"].mean()
    global_delta = int(means.loc["__macro__"].idxmax())
    per_label = {label: int(means.loc[label].idxmax()) for label in classes}
    return global_delta, per_label, scores
