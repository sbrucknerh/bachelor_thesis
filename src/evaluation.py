"""Evaluation metrics and reporting for detection models.

Provides standard ML evaluation (precision, recall, F1, ROC/AUC) and
comparison utilities for supervised and unsupervised approaches.
"""

from __future__ import annotations

import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    auc,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from config import FIGURE_SIZE, RESULTS_DIR


def evaluate_classifier(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray | None = None,
) -> dict:
    """Compute classification metrics.

    Args:
        y_true: Ground truth binary labels.
        y_pred: Predicted binary labels.
        y_prob: Predicted probabilities for the positive class (optional).

    Returns:
        Dictionary of metrics: accuracy, precision, recall, f1, and
        optionally roc_auc and pr_auc.
    """
    results = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }

    if y_prob is not None:
        try:
            results["roc_auc"] = roc_auc_score(y_true, y_prob)
        except ValueError:
            results["roc_auc"] = float("nan")

        precision_vals, recall_vals, _ = precision_recall_curve(y_true, y_prob)
        results["pr_auc"] = auc(recall_vals, precision_vals)

    return results


def plot_roc_pr_curves(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    title_prefix: str = "",
    save_path: str | None = None,
) -> None:
    """Plot ROC and Precision-Recall curves side by side.

    Args:
        y_true: Ground truth labels.
        y_prob: Predicted probabilities.
        title_prefix: Optional prefix for plot titles.
        save_path: If provided, save figure to this path instead of showing.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # ROC curve
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc_val = roc_auc_score(y_true, y_prob)
    ax1.plot(fpr, tpr, color="steelblue", lw=2, label=f"AUC = {roc_auc_val:.3f}")
    ax1.plot([0, 1], [0, 1], "k--", lw=1)
    ax1.set_xlabel("False Positive Rate")
    ax1.set_ylabel("True Positive Rate")
    ax1.set_title(f"{title_prefix}ROC Curve")
    ax1.legend(loc="lower right")

    # PR curve
    precision_vals, recall_vals, _ = precision_recall_curve(y_true, y_prob)
    pr_auc_val = auc(recall_vals, precision_vals)
    ax2.plot(recall_vals, precision_vals, color="crimson", lw=2, label=f"AUC = {pr_auc_val:.3f}")
    ax2.set_xlabel("Recall")
    ax2.set_ylabel("Precision")
    ax2.set_title(f"{title_prefix}Precision-Recall Curve")
    ax2.legend(loc="lower left")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: list[str] | None = None,
    save_path: str | None = None,
) -> None:
    """Plot a confusion matrix heatmap.

    Args:
        y_true: Ground truth labels.
        y_pred: Predicted labels.
        labels: Class label names (default: ["Benign", "Malicious"]).
        save_path: If provided, save figure to this path.
    """
    if labels is None:
        labels = ["Benign", "Malicious"]

    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))

    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")

    # Annotate cells
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black",
                    fontsize=14)

    plt.colorbar(im)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def compare_models(results_dict: dict[str, dict]) -> pd.DataFrame:
    """Create a comparison table of multiple model results.

    Args:
        results_dict: Dict mapping model names to their evaluation metric dicts.

    Returns:
        DataFrame with models as rows and metrics as columns.
    """
    df = pd.DataFrame(results_dict).T
    df.index.name = "model"
    return df.round(4)


def export_results(results: dict, filename: str = "results.json") -> str:
    """Save evaluation results to a JSON file.

    Args:
        results: Dictionary of results to save.
        filename: Output filename.

    Returns:
        Path to the saved file.
    """
    os.makedirs(RESULTS_DIR, exist_ok=True)
    filepath = os.path.join(RESULTS_DIR, filename)

    # Convert numpy types for JSON serialization
    serializable = _make_serializable(results)

    with open(filepath, "w") as f:
        json.dump(serializable, f, indent=2)

    return filepath


def _make_serializable(obj):
    """Recursively convert numpy types to Python types for JSON."""
    if isinstance(obj, dict):
        return {k: _make_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_make_serializable(v) for v in obj]
    elif isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj
