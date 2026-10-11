#!/usr/bin/env python3
"""Render readable thesis figures from the existing final experiment artifacts.

This changes presentation only. It does not train models or modify locked runs.
Run after results_to_latex.py, which copies the original experiment figures.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "thesis-feedback-mpl"))
os.environ.setdefault("XDG_CACHE_HOME", str(Path(tempfile.gettempdir()) / "thesis-feedback-cache"))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.experiments.features import hellinger_motif_composition
from src.experiments.statistics import stratified_group_bootstrap_ap

DEFAULT_RUNS = ROOT / "server-results/results_22_08_2026/extracted/results/experiments"
STYLE = {
    "font.family": "DejaVu Sans",
    "font.size": 12,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 11.5,
    "ytick.labelsize": 11.5,
    "legend.fontsize": 11.5,
    "pdf.fonttype": 42,
}
COLORS = ("#777777", "#2878a8", "#cc681d")
DELTA_TICKS = (300, 3600, 86400, 604800, 2592000)
DELTA_LABELS = ("5 min", "1 hour", "1 day", "1 week", "30 days")


def save(fig, output: Path) -> None:
    fig.savefig(
        output, bbox_inches="tight", pad_inches=0.07,
        metadata={"Creator": "Thesis presentation revision",
                  "CreationDate": None, "ModDate": None},
    )
    plt.close(fig)
    print(f"Rendered {output.name}", flush=True)


def window_axis(axis) -> None:
    axis.set_xscale("log")
    axis.set_xticks(DELTA_TICKS, DELTA_LABELS, rotation=40, ha="right")
    axis.minorticks_off()
    axis.set_ylim(0, 1)
    axis.set_yticks((0, .25, .5, .75, 1))
    axis.set_xlabel("Temporal window")
    axis.spines[["top", "right"]].set_visible(False)
    axis.grid(axis="y", alpha=.18)
    axis.set_axisbelow(True)


def binary_figures(hf: Path, out: Path, cache: Path, audit: dict) -> None:
    protocol = json.loads((hf / "protocol.json").read_text())
    n_bootstraps = int(protocol["bootstraps"])
    seed = int(protocol["random_seed"])
    all_coverage = pd.read_csv(hf / "tables/rq2_motif_coverage.csv")
    for cohort in ("controlled", "prevalence"):
        metrics = pd.read_csv(hf / f"metrics/hf_{cohort}_canonical_metrics.csv")
        summary = json.loads(
            (hf / f"metrics/hf_{cohort}_canonical_statistical_tests.json").read_text()
        )
        delta = int(summary["selected_delta"])
        selected = metrics.loc[metrics.delta == delta].set_index("arm")
        values = selected.loc[["baseline", "motifs", "combined"], "average_precision"]
        fig, axis = plt.subplots(figsize=(3.15, 2.9), layout="constrained")
        bars = axis.bar(["Baseline", "Roles\nonly", "Combined"], values, color=COLORS)
        axis.bar_label(bars, fmt="%.3f", padding=3, fontsize=12)
        axis.set_ylim(0, 1)
        axis.set_yticks((0, .25, .5, .75, 1))
        axis.set_ylabel("Average precision")
        axis.set_title(cohort.capitalize())
        axis.spines[["top", "right"]].set_visible(False)
        save(fig, out / f"rq1_ablation_{cohort}_rep3.pdf")

        predictions_file = hf / f"predictions/hf_{cohort}_canonical.parquet"
        cache_key = hashlib.sha256(
            predictions_file.read_bytes() + f"{n_bootstraps}:{seed}".encode()
        ).hexdigest()
        interval_file = cache / f"{cohort}-{cache_key}.json"
        if interval_file.exists():
            intervals = json.loads(interval_file.read_text())
        else:
            predictions = pd.read_parquet(predictions_file)
            intervals = {}
            for window in sorted(metrics.delta.unique()):
                scores = predictions.loc[
                    (predictions.delta == window) & (predictions.arm == "combined")
                ]
                print(f"Recovering original {cohort} interval at {window} s", flush=True)
                intervals[str(window)] = stratified_group_bootstrap_ap(
                    scores.label, scores.score, scores.campaign_group,
                    iterations=n_bootstraps, seed=seed,
                )["ci_95"]
            interval_file.write_text(json.dumps(intervals))
        curve = metrics.loc[metrics.arm == "combined"].sort_values("delta")
        bounds = np.array([intervals[str(window)] for window in curve.delta])
        baseline = float(values["baseline"])
        fig, axis = plt.subplots(figsize=(3.15, 3.25), layout="constrained")
        axis.plot(curve.delta, curve.average_precision, marker="o",
                  markersize=4, color=COLORS[1], label="Combined")
        axis.fill_between(curve.delta, bounds[:, 0], bounds[:, 1],
                          color=COLORS[1], alpha=.16)
        axis.axhline(baseline, color=COLORS[0], linestyle="--", label="Baseline")
        window_axis(axis)
        axis.set_ylabel("Average precision")
        axis.set_title(cohort.capitalize())
        axis.legend(frameon=False, loc="lower right")
        save(fig, out / f"rq2_delta_performance_{cohort}_rep3.pdf")

        coverage = all_coverage.loc[
            (all_coverage.cohort == cohort)
            & (all_coverage.variant == "canonical")
            & (all_coverage.scope == "all")
            & (all_coverage.representation == "all_40_roles")
        ].copy()
        fig, axis = plt.subplots(figsize=(3.15, 3.25), layout="constrained")
        for label, title, color, style, marker in (
            ("all", "All", "#333333", "-", "o"),
            ("0", "Benign", COLORS[1], "--", "s"),
            ("1", "Malicious", COLORS[2], ":", "^"),
        ):
            rows = coverage.loc[coverage.label.astype(str) == label].sort_values("delta")
            axis.plot(rows.delta, rows.coverage, label=title, color=color,
                      linestyle=style, marker=marker, markersize=4)
        window_axis(axis)
        axis.set_ylabel("Fingerprint coverage")
        axis.set_title(cohort.capitalize())
        axis.legend(frameon=False, loc="upper left")
        save(fig, out / f"rq2_motif_coverage_{cohort}_rep3.pdf")
        audit[cohort] = {
            "selected_delta": delta,
            "ablation": {arm: float(value) for arm, value in values.items()},
            "curve": curve[["delta", "average_precision"]].to_dict("records"),
            "intervals": intervals,
            "coverage": coverage[["delta", "label", "coverage"]].to_dict("records"),
        }


def category_name(label: str) -> str:
    return label.removesuffix(" Scam")


def role_label(column: str) -> str:
    layer, role = column.split("__", 1)
    prefixes = {"normal": "N", "internal": "I", "erc20": "E20",
                "erc721": "E721", "erc1155": "E1155"}
    role = role.replace("star_", "").replace("two_node_", "two:")
    return f"{prefixes[layer]} {role.replace('_', ':')}"


def category_figures(rc: Path, out: Path, audit: dict) -> None:
    summary = json.loads((rc / "metrics/realcats_canonical_results.json").read_text())
    delta = int(summary["global_delta"])
    targets = pd.read_parquet(rc / "data/realcats_targets.parquet").set_index("address")
    development = targets.index[targets.split == "development"]
    counts = pd.read_parquet(rc / "features/realcats_motifs_canonical.parquet")
    counts = counts.loc[counts.delta == delta].set_index("address").drop(columns=["delta", "variant"])
    composition = hellinger_motif_composition(counts).loc[development]
    means = composition.assign(_label=targets.type_label.reindex(development)).groupby("_label").mean()
    selected = means.var(axis=0).sort_values(ascending=False).head(40).index
    fig, axes = plt.subplots(2, 1, figsize=(6.15, 7.8), layout="constrained")
    maximum = float(means[selected].max().max())
    for panel, axis in enumerate(axes):
        columns = selected[panel * 20:(panel + 1) * 20]
        image = axis.imshow(means[columns], aspect="auto", cmap="viridis", vmin=0, vmax=maximum)
        axis.set_yticks(range(len(means)), [category_name(label) for label in means.index])
        axis.set_xticks(range(20), [role_label(column) for column in columns],
                       rotation=90, fontsize=11.5)
        axis.tick_params(length=0)
        axis.set_title(f"Components {panel * 20 + 1} to {(panel + 1) * 20}")
    fig.colorbar(image, ax=axes, orientation="horizontal", shrink=.68, aspect=35,
                 label="Mean Hellinger component", pad=.02)
    save(fig, out / "rq3_fingerprint_heatmap.pdf")

    predictions = pd.read_parquet(rc / "predictions/realcats_canonical.parquet")
    classes = sorted(predictions.score_label.unique())
    probabilities = predictions.pivot(index="address", columns="score_label", values="score")[classes]
    truth = predictions.drop_duplicates("address").set_index("address").true_label.reindex(probabilities.index)
    predicted = probabilities.idxmax(axis=1)
    matrix = confusion_matrix(truth, predicted, labels=classes)
    fig, axis = plt.subplots(figsize=(6.15, 6.15), layout="constrained")
    image = axis.imshow(matrix, cmap="Blues")
    axis.set_xticks(range(len(classes)), [category_name(label).replace(" ", "\n", 1)
                                         for label in classes], rotation=55, ha="right")
    axis.set_yticks(range(len(classes)), [category_name(label) for label in classes])
    axis.set_xlabel("Predicted category")
    axis.set_ylabel("Recorded category")
    for row, column in np.ndindex(matrix.shape):
        color = "white" if matrix[row, column] > matrix.max() * .55 else "#202020"
        axis.text(column, row, str(matrix[row, column]), color=color,
                  ha="center", va="center", fontsize=11.5)
    fig.colorbar(image, ax=axis, shrink=.72, label="Addresses")
    save(fig, out / "rq3_confusion_matrix.pdf")

    scores = pd.read_csv(rc / "metrics/realcats_canonical_cv.csv")
    scores = scores.loc[scores.label != "__macro__"]
    curves = scores.groupby(["label", "delta"]).average_precision.mean().reset_index()
    fig, axis = plt.subplots(figsize=(6.15, 5.7), layout="constrained")
    for index, (label, rows) in enumerate(curves.groupby("label")):
        axis.plot(rows.delta, rows.average_precision, label=category_name(label),
                  marker=("o", "s", "^")[index % 3], markersize=4,
                  linestyle=("-", "--", ":")[index // 3])
    window_axis(axis)
    axis.set_ylabel("One-vs-rest average precision")
    axis.legend(frameon=False, ncol=2, loc="upper center",
                bbox_to_anchor=(.5, -.30), columnspacing=1.0)
    save(fig, out / "rq4_label_delta_curves.pdf")
    audit["realcats"] = {
        "global_delta": delta,
        "selected_components": selected.tolist(),
        "category_means": means[selected].to_dict("index"),
        "classes": classes,
        "confusion_matrix": matrix.tolist(),
        "curves": curves.to_dict("records"),
    }


def diagram_canvas(height: float):
    fig = plt.figure(figsize=(6.1, height))
    axis = fig.add_axes((0, 0, 1, 1))
    axis.set(xlim=(0, 6.1), ylim=(0, height))
    axis.axis("off")
    return fig, axis


def box(axis, x, y, width, height, title, body, module=None, fill="#eeeeee"):
    axis.add_patch(FancyBboxPatch((x, y), width, height,
                   boxstyle="round,pad=0.03,rounding_size=0.05",
                   linewidth=.8, edgecolor="#888888", facecolor=fill))
    axis.text(x + width / 2, y + height - .14, title, ha="center", va="top",
              fontsize=12.5, weight="bold")
    axis.text(x + width / 2, y + height - .40, body, ha="center", va="top",
              fontsize=11.5, linespacing=1.3)
    if module:
        axis.text(x + width / 2, y + .10, module, ha="center", va="bottom",
                  fontsize=11, style="italic", color="#333333")


def arrow(axis, start, end):
    axis.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>",
                                  mutation_scale=12, color="#555555", linewidth=1,
                                  shrinkA=0, shrinkB=0))


def diagrams(out: Path) -> None:
    fig, axis = diagram_canvas(6.9)
    axis.text(3.05, 6.8, "Preparation", ha="center", va="top", fontsize=14, weight="bold")
    steps = [
        ("1. Load data", "Read transfers and labels.\nStream large source tables."),
        ("2. Prepare data", "Select targets and splits.\nStore the required transfers."),
        ("3. Build temporal graphs", "Keep direction, timestamps,\nand repeated transfers."),
        ("4. Count local roles", "Check validation templates.\nCount at nine time windows."),
        ("5. Build fingerprints", "Transform local role counts.\nAdd 17 baseline features."),
        ("6. Run detection", "Select windows on development.\nEvaluate held-out addresses."),
        ("7. Analyse results", "Compare features and windows.\nTest RQ1 to RQ4 across\nseven timestamp orderings."),
        ("8. Report results", "Write the tables and figures.\nRecord scores and uncertainty."),
    ]
    positions = []
    for index, (title, body) in enumerate(steps):
        row, column = divmod(index, 2)
        x = .08 + column * 3.12
        y = (5.30, 3.64, 2.02, .40)[row]
        box(axis, x, y, 2.82, 1.18, title, body)
        positions.append((x, y))
        if column:
            arrow(axis, (2.95, y + .59), (3.12, y + .59))
    axis.text(3.05, 5.10, "Execution", ha="center", va="top", fontsize=13.5, weight="bold")
    for index in (1, 3, 5):
        x, y = positions[index]
        next_x, next_y = positions[index + 1]
        # A connecting line runs through the gap between successive rows.
        middle_y = y - .12 if index == 1 else (y + next_y + 1.18) / 2
        axis.plot([x + 1.41, x + 1.41, next_x + 1.41],
                  [y - .04, middle_y, middle_y], color="#555555", linewidth=1)
        arrow(axis, (next_x + 1.41, middle_y), (next_x + 1.41, next_y + 1.22))
    save(fig, out / "05-method-steps.pdf")

    fig, axis = diagram_canvas(8.50)
    axis.text(3.05, 8.36, "Preparation", ha="center", va="top", fontsize=14, weight="bold")
    entries = [
        (7.03, "Load sources and select targets",
         "HuggingFace and Real-CATS, with optional BLTE.\nApply eligibility rules and select cohorts.",
         "datasets.py, realcats.py"),
        (5.79, "Fix the evaluation and graph scope",
         "Keep groups together in development and test.\nPrepare incident edges, closure edges, and layers.",
         "prepare.py, splits.py"),
        (3.52, "Verify and build temporal graphs",
         "Check the saved protocol, code, and data.\nApply seven orderings of equal-timestamp transfers.",
         "execute.py, ordering.py, motifs.py"),
        (2.32, "Count candidate subgraphs and build features",
         "Validate local roles, then count at nine windows.\nBuild fingerprints and the conventional baseline.",
         "motifs.py, features.py"),
        (1.12, "Fit models and compare predictions",
         "Select windows on development, then score test data.\nRun statistical tests and ordering checks.",
         "modeling.py, rq_binary.py, rq_multiclass.py, statistics.py"),
    ]
    for y, title, body, module in entries:
        box(axis, .10, y, 5.90, 1.08, title, body, module)
    arrow(axis, (3.05, 6.99), (3.05, 6.91))
    box(axis, .10, 5.05, 5.90, .58, "Fixed data and settings",
        "", fill="#e7edf3")
    axis.text(3.05, 5.16, "Protocol, cohorts, folds, transfers, and checksums",
              ha="center", fontsize=11.5)
    arrow(axis, (3.05, 5.75), (3.05, 5.67))
    axis.text(.15, 4.88, "Execution", ha="left", va="top", fontsize=14, weight="bold")
    arrow(axis, (3.05, 5.01), (3.05, 4.64))
    for y in (3.52, 2.32):
        arrow(axis, (3.05, y - .02), (3.05, y - .10))
    arrow(axis, (3.05, 1.08), (3.05, .98))
    box(axis, .10, .12, 5.90, .82, "Saved results",
        "Predictions, metrics, figures, tables, and verification records.",
        fill="#edf2e9")
    save(fig, out / "01-pipeline.pdf")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hf", type=Path, default=DEFAULT_RUNS / "thesis_final_v8_hf")
    parser.add_argument("--realcats", type=Path, default=DEFAULT_RUNS / "thesis_final_v8_realcats")
    parser.add_argument("--output", type=Path, default=ROOT / "writing/figures")
    parser.add_argument("--interval-cache", type=Path,
                        default=Path(tempfile.gettempdir()) / "thesis-feedback-intervals")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    args.interval_cache.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(STYLE)
    audit = {"sources": {"hf": str(args.hf.relative_to(ROOT)),
                         "realcats": str(args.realcats.relative_to(ROOT))}}
    diagrams(args.output)
    category_figures(args.realcats, args.output, audit)
    binary_figures(args.hf, args.output, args.interval_cache, audit)
    (args.output / "professor_feedback_sources.json").write_text(
        json.dumps(audit, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
