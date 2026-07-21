"""RQ3 exact-label and label-specific temporal-window evaluation."""

from __future__ import annotations

from pathlib import Path
from contextlib import nullcontext

import numpy as np
import pandas as pd

from src.experiments.artifacts import write_csv_exclusive, write_json_exclusive, write_parquet_exclusive
from src.experiments.features import (
    hellinger_motif_composition,
    log_motif_counts,
    motif_coverage,
)
from src.experiments.modeling import make_multiclass_model, select_multiclass_deltas
from src.experiments.motifs import count_layer_parquets
from src.experiments.statistics import (
    benjamini_hochberg,
    multiclass_metrics,
    permute_multiclass_labels,
    stratified_paired_bootstrap_delta,
)
from src.experiments.telemetry import RunTelemetry
from src.experiments.visualization import (
    plot_confusion,
    plot_fingerprint_heatmap,
    plot_label_delta_curves,
)


def _development_folds(
    development_addresses: pd.Index, assignments: pd.DataFrame
) -> list[tuple[np.ndarray, np.ndarray]]:
    positions = pd.Series(np.arange(len(development_addresses)), index=development_addresses)
    folds = []
    for _, rows in assignments.groupby("fold"):
        validation_addresses = rows["address"].astype(str)
        validation = positions.loc[validation_addresses].to_numpy()
        training = np.setdiff1d(np.arange(len(development_addresses)), validation)
        folds.append((training, validation))
    return folds


def _fit_multiclass(
    features: pd.DataFrame,
    labels: pd.Series,
    development: pd.Index,
    test: pd.Index,
) -> tuple[object, np.ndarray, list[str]]:
    model = make_multiclass_model()
    model.fit(features.loc[development], labels.loc[development])
    probabilities = model.predict_proba(features.loc[test])
    classes = [str(value) for value in model.named_steps["model"].classes_]
    return model, probabilities, classes


def execute_realcats(
    paths, protocol: dict, telemetry: RunTelemetry | None = None
) -> dict:
    dataset = protocol["datasets"]["realcats"]
    targets = pd.read_parquet(paths.root / dataset["artifacts"]["targets"])
    targets = targets.loc[targets["inference_eligible"]].copy()
    targets["address"] = targets["address"].astype(str)
    indexed = targets.set_index("address", drop=False)
    labels = indexed["type_label"].astype(str)
    development = indexed.index[indexed["split"] == "development"]
    test = indexed.index[indexed["split"] == "test"]
    assignments = pd.read_parquet(paths.root / dataset["artifacts"]["folds"])
    folds = _development_folds(development, assignments)
    layer_paths = {
        layer: str(paths.root / relative)
        for layer, relative in dataset["artifacts"]["layers"].items()
    }

    variant_summaries = []
    for variant in protocol["ordering_variants"]:
        if telemetry:
            telemetry.progress(
                "realcats.ordering_variants", variant=variant, status_detail="starting"
            )
        counts_by_delta = count_layer_parquets(
            layer_paths,
            indexed.index,
            protocol["deltas"],
            variant,
            temporary_directory=paths.data / "duckdb_tmp" / "realcats" / variant,
            telemetry=telemetry,
        )
        long_counts = pd.concat(
            [counts.reset_index().assign(delta=delta, variant=variant) for delta, counts in counts_by_delta.items()],
            ignore_index=True,
        )
        write_parquet_exclusive(paths.features / f"realcats_motifs_{variant}.parquet", long_counts)
        shape_features = {
            delta: hellinger_motif_composition(counts) for delta, counts in counts_by_delta.items()
        }
        development_features = {
            delta: features.loc[development] for delta, features in shape_features.items()
        }
        cv_context = (
            telemetry.step(
                f"realcats.cv.{variant}",
                deltas=len(protocol["deltas"]),
                labels=labels.nunique(),
            )
            if telemetry
            else nullcontext()
        )
        with cv_context:
            global_delta, per_label_delta, cv_scores = select_multiclass_deltas(
                development_features, labels.loc[development], folds
            )
        _, global_probabilities, classes = _fit_multiclass(
            shape_features[global_delta], labels, development, test
        )
        test_labels = labels.loc[test]
        metrics = multiclass_metrics(test_labels, global_probabilities, classes)
        permutation_context = (
            telemetry.step(
                f"realcats.label_permutation.{variant}",
                iterations=int(protocol["permutations"]),
            )
            if telemetry
            else nullcontext()
        )
        with permutation_context:
            permutation = permute_multiclass_labels(
                test_labels,
                global_probabilities,
                classes,
                iterations=int(protocol["permutations"]),
                seed=int(protocol["random_seed"]),
            )

        probabilities_by_delta = {global_delta: global_probabilities}
        for delta in sorted(set(per_label_delta.values()) - {global_delta}):
            _, probabilities, delta_classes = _fit_multiclass(
                shape_features[delta], labels, development, test
            )
            if delta_classes != classes:
                raise RuntimeError("Multiclass probability columns changed across deltas")
            probabilities_by_delta[delta] = probabilities

        comparisons = []
        comparison_context = (
            telemetry.step(
                f"realcats.label_delta_bootstrap.{variant}",
                labels=len(classes),
                bootstraps=int(protocol["bootstraps"]),
            )
            if telemetry
            else nullcontext()
        )
        with comparison_context:
            for column, label in enumerate(classes):
                selected_delta = per_label_delta[label]
                binary = (test_labels == label).astype(int).to_numpy()
                comparison = stratified_paired_bootstrap_delta(
                    binary,
                    global_probabilities[:, column],
                    probabilities_by_delta[selected_delta][:, column],
                    iterations=int(protocol["bootstraps"]),
                    seed=int(protocol["random_seed"]),
                )
                comparisons.append(
                    {
                        "label": label,
                        "global_delta": global_delta,
                        "label_delta": selected_delta,
                        **comparison,
                    }
                )
        adjusted = benjamini_hochberg(item["p_value_one_sided"] for item in comparisons)
        for comparison, adjusted_p in zip(comparisons, adjusted, strict=True):
            comparison["p_value_bh"] = float(adjusted_p)

        predictions = pd.DataFrame(
            {
                "address": np.repeat(test.to_numpy(), len(classes)),
                "true_label": np.repeat(test_labels.to_numpy(), len(classes)),
                "score_label": np.tile(classes, len(test)),
                "score": global_probabilities.reshape(-1),
                "variant": variant,
                "global_delta": global_delta,
            }
        )
        coverage = pd.concat(
            [
                motif_coverage(counts_by_delta[delta], labels)
                .reset_index()
                .assign(delta=delta, variant=variant)
                for delta in protocol["deltas"]
            ],
            ignore_index=True,
        )
        write_parquet_exclusive(paths.predictions / f"realcats_{variant}.parquet", predictions)
        write_csv_exclusive(paths.metrics / f"realcats_{variant}_cv.csv", cv_scores)
        write_csv_exclusive(paths.metrics / f"realcats_{variant}_coverage.csv", coverage)

        count_sensitivity = None
        if variant == "canonical":
            count_features = log_motif_counts(counts_by_delta[global_delta])
            _, count_probabilities, count_classes = _fit_multiclass(
                count_features, labels, development, test
            )
            count_sensitivity = multiclass_metrics(test_labels, count_probabilities, count_classes)
            prediction = np.asarray(classes)[np.argmax(global_probabilities, axis=1)]
            plot_confusion(
                test_labels,
                prediction,
                classes,
                paths.figures / "rq3_confusion_matrix.pdf",
            )
            plot_fingerprint_heatmap(
                shape_features[global_delta].loc[development],
                labels.loc[development],
                paths.figures / "rq3_fingerprint_heatmap.pdf",
            )
            plot_label_delta_curves(
                cv_scores,
                paths.figures / "rq3_label_delta_curves.pdf",
            )

        summary = {
            "variant": variant,
            "global_delta": global_delta,
            "per_label_delta": per_label_delta,
            "shape_metrics": metrics,
            "count_sensitivity_metrics": count_sensitivity,
            "global_label_permutation": permutation,
            "label_delta_comparisons": comparisons,
        }
        write_json_exclusive(paths.metrics / f"realcats_{variant}_results.json", summary)
        variant_summaries.append(summary)

    alpha = float(protocol["alpha"])
    decisions = [
        summary["global_label_permutation"]["p_value_one_sided"] < alpha
        for summary in variant_summaries
    ]
    sensitivity = {
        "ordering_robust_global_separability": len(set(decisions)) == 1 and all(decisions),
        "decisions": decisions,
        "global_deltas": [summary["global_delta"] for summary in variant_summaries],
        "per_label_deltas": [summary["per_label_delta"] for summary in variant_summaries],
    }
    global_rows = []
    label_rows = []
    for summary in variant_summaries:
        global_rows.append(
            {
                "ordering_variant": summary["variant"],
                "global_delta": summary["global_delta"],
                "macro_average_precision": summary["shape_metrics"]["macro_average_precision"],
                "macro_f1": summary["shape_metrics"]["macro_f1"],
                "permutation_p_value": summary["global_label_permutation"]["p_value_one_sided"],
                "reject_h0": summary["global_label_permutation"]["p_value_one_sided"] < alpha,
            }
        )
        for label, average_precision in summary["shape_metrics"][
            "per_label_average_precision"
        ].items():
            label_rows.append(
                {
                    "ordering_variant": summary["variant"],
                    "label": label,
                    "average_precision": average_precision,
                    "selected_delta": summary["per_label_delta"][label],
                }
            )
    write_csv_exclusive(paths.tables / "rq3_global_results.csv", pd.DataFrame(global_rows))
    write_csv_exclusive(paths.tables / "rq3_label_results.csv", pd.DataFrame(label_rows))
    write_json_exclusive(paths.metrics / "realcats_ordering_sensitivity.json", sensitivity)
    return sensitivity
