"""Publication-oriented figures generated only from persisted experiment metrics."""

from __future__ import annotations

import os
from pathlib import Path
import tempfile

_CACHE_ROOT = Path(tempfile.gettempdir()) / "bachelor_thesis_plot_cache"
_CACHE_ROOT.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_CACHE_ROOT / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(_CACHE_ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

from src.experiments.constants import PROTOCOL_VERSION


def _save_figure(fig, output: str | Path) -> None:
    metadata = {
        "Creator": f"bachelor-thesis protocol v{PROTOCOL_VERSION}",
        "Producer": "matplotlib",
        "CreationDate": None,
        "ModDate": None,
    }
    fig.savefig(output, dpi=200, bbox_inches="tight", metadata=metadata)


def plot_delta_performance(metrics: pd.DataFrame, output: str | Path, title: str) -> None:
    frame = metrics.sort_values("delta")
    fig, axis = plt.subplots(figsize=(7.2, 4.4))
    axis.plot(frame["delta"], frame["average_precision"], marker="o", label="Baseline + motifs")
    if "baseline_average_precision" in frame:
        axis.axhline(
            float(frame["baseline_average_precision"].iloc[0]),
            color="0.35",
            linestyle="--",
            label="Aggregate/static baseline",
        )
    if {"ci_lower", "ci_upper"}.issubset(frame.columns):
        axis.fill_between(frame["delta"], frame["ci_lower"], frame["ci_upper"], alpha=0.18)
    axis.set_xscale("log")
    axis.set_xlabel("Temporal window delta (seconds, log scale)")
    axis.set_ylabel("Average precision")
    axis.set_ylim(0, 1)
    axis.set_title(title)
    axis.legend(frameon=False)
    fig.tight_layout()
    _save_figure(fig, output)
    plt.close(fig)


def plot_ablation(metrics: dict[str, float], output: str | Path, title: str) -> None:
    labels = ["Baseline", "Motifs", "Combined"]
    values = [metrics["baseline"], metrics["motifs"], metrics["combined"]]
    fig, axis = plt.subplots(figsize=(6.2, 4.2))
    bars = axis.bar(labels, values, color=["#777777", "#2c7fb8", "#d95f0e"])
    axis.bar_label(bars, fmt="%.3f", padding=3)
    axis.set_ylabel("Average precision")
    axis.set_ylim(0, 1)
    axis.set_title(title)
    fig.tight_layout()
    _save_figure(fig, output)
    plt.close(fig)


def plot_representation_ablation(
    metrics: dict[str, float], output: str | Path, title: str
) -> None:
    arms = (
        "baseline",
        "baseline_presence",
        "baseline_volume",
        "baseline_composition_volume",
        "baseline_incoming_stars",
        "combined",
    )
    labels = (
        "Baseline",
        "+Presence",
        "+Volume",
        "+Composition + volume",
        "+Incoming",
        "+All roles",
    )
    values = [metrics[arm] for arm in arms]
    fig, axis = plt.subplots(figsize=(8.0, 4.5))
    bars = axis.bar(
        labels,
        values,
        color=["#666666", "#8c8c8c", "#4c78a8", "#59a14f", "#e3a329", "#d95f0e"],
    )
    axis.bar_label(bars, fmt="%.3f", padding=3, fontsize=8)
    axis.set_ylabel("Average precision")
    axis.set_ylim(0, 1)
    axis.set_title(title)
    axis.tick_params(axis="x", rotation=25)
    fig.tight_layout()
    _save_figure(fig, output)
    plt.close(fig)


def plot_motif_coverage(metrics: pd.DataFrame, output: str | Path, title: str) -> None:
    frame = metrics.copy()
    frame["label"] = frame["label"].astype(str)
    labels = {"all": "All addresses", "0": "Benign", "1": "Malicious"}
    fig, axis = plt.subplots(figsize=(7.2, 4.4))
    for label in ("all", "0", "1"):
        rows = frame.loc[frame["label"] == label].sort_values("delta")
        if len(rows):
            axis.plot(
                rows["delta"],
                rows["coverage"],
                marker="o",
                label=labels[label],
            )
    axis.set_xscale("log")
    axis.set_xlabel("Temporal window delta (seconds, log scale)")
    axis.set_ylabel("Nonzero fingerprint coverage")
    axis.set_ylim(0, 1)
    axis.set_title(title)
    axis.legend(frameon=False)
    fig.tight_layout()
    _save_figure(fig, output)
    plt.close(fig)


def plot_confusion(
    y_true: pd.Series | np.ndarray,
    y_predicted: pd.Series | np.ndarray,
    classes: list[str],
    output: str | Path,
) -> None:
    matrix = confusion_matrix(y_true, y_predicted, labels=classes)
    fig, axis = plt.subplots(figsize=(max(6, len(classes) * 0.85), max(5, len(classes) * 0.75)))
    image = axis.imshow(matrix, cmap="Blues")
    axis.set_xticks(range(len(classes)), classes, rotation=45, ha="right")
    axis.set_yticks(range(len(classes)), classes)
    axis.set_xlabel("Predicted label")
    axis.set_ylabel("True label")
    for row in range(len(classes)):
        for column in range(len(classes)):
            axis.text(column, row, str(matrix[row, column]), ha="center", va="center", fontsize=8)
    fig.colorbar(image, ax=axis)
    fig.tight_layout()
    _save_figure(fig, output)
    plt.close(fig)


def plot_fingerprint_heatmap(
    features: pd.DataFrame,
    labels: pd.Series,
    output: str | Path,
    max_features: int = 40,
) -> None:
    means = features.assign(_label=labels.reindex(features.index)).groupby("_label").mean()
    variation = means.var(axis=0).sort_values(ascending=False)
    selected = variation.head(max_features).index
    matrix = means.loc[:, selected]
    fig, axis = plt.subplots(figsize=(max(8, len(selected) * 0.22), max(4, len(means) * 0.55)))
    image = axis.imshow(matrix, aspect="auto", cmap="viridis")
    axis.set_yticks(range(len(matrix.index)), matrix.index)
    axis.set_xticks(range(len(selected)), selected, rotation=90, fontsize=6)
    axis.set_xlabel("Motif role")
    axis.set_ylabel("Exact Real-CATS label")
    fig.colorbar(image, ax=axis, label="Mean Hellinger component")
    fig.tight_layout()
    _save_figure(fig, output)
    plt.close(fig)


def plot_label_delta_curves(scores: pd.DataFrame, output: str | Path) -> None:
    means = scores.groupby(["label", "delta"])["average_precision"].mean().reset_index()
    means = means.loc[means["label"] != "__macro__"]
    fig, axis = plt.subplots(figsize=(7.5, 4.8))
    for label, rows in means.groupby("label"):
        axis.plot(rows["delta"], rows["average_precision"], marker="o", label=str(label))
    axis.set_xscale("log")
    axis.set_ylim(0, 1)
    axis.set_xlabel("Temporal window delta (seconds, log scale)")
    axis.set_ylabel("One-vs-rest average precision")
    axis.set_title("RQ4 category-specific temporal-window sensitivity")
    axis.legend(frameon=False, fontsize=7, ncol=2)
    fig.tight_layout()
    _save_figure(fig, output)
    plt.close(fig)
