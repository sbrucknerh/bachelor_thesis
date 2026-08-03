"""RQ3 category separability and RQ4 category-specific window evaluation."""

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
from src.experiments.constants import MOTIF_ROLE_NAMES
from src.experiments.motifs import RaphtoryExperimentGraph, count_layer_parquets
from src.experiments.statistics import (
    benjamini_hochberg,
    multiclass_metrics,
    permute_multiclass_labels,
    stratified_paired_bootstrap_macro_ap_delta,
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


def _aggregate_realcats_results(
    summaries: list[dict], alpha: float
) -> tuple[dict, list[dict], list[dict], list[dict], list[dict]]:
    rq3_decisions = [
        summary["global_label_permutation"]["p_value_one_sided"] < alpha
        for summary in summaries
    ]
    rq4_decisions = [
        bool(summary["category_delta_macro_bootstrap"]["reject_h0"])
        for summary in summaries
    ]
    sensitivity = {
        "ordering_robust_global_separability": all(rq3_decisions),
        "decisions": rq3_decisions,
        "global_deltas": [summary["global_delta"] for summary in summaries],
        "per_label_deltas": [summary["per_label_delta"] for summary in summaries],
        "rq4_ordering_robust_category_specific_windows": all(rq4_decisions),
        "rq4_decisions": rq4_decisions,
    }
    rq3_global_rows: list[dict] = []
    rq3_label_rows: list[dict] = []
    rq4_global_rows: list[dict] = []
    rq4_category_rows: list[dict] = []
    for summary in summaries:
        rq3_global_rows.append(
            {
                "ordering_variant": summary["variant"],
                "global_delta": summary["global_delta"],
                "macro_average_precision": summary["shape_metrics"][
                    "macro_average_precision"
                ],
                "macro_f1": summary["shape_metrics"]["macro_f1"],
                "permutation_p_value": summary["global_label_permutation"][
                    "p_value_one_sided"
                ],
                "reject_h0": summary["global_label_permutation"][
                    "p_value_one_sided"
                ]
                < alpha,
            }
        )
        comparisons = {
            item["label"]: item for item in summary["label_delta_comparisons"]
        }
        for label, average_precision in summary["shape_metrics"][
            "per_label_average_precision"
        ].items():
            comparison = comparisons[label]
            rq3_label_rows.append(
                {
                    "ordering_variant": summary["variant"],
                    "label": label,
                    "average_precision": average_precision,
                    # Retained for compatibility; RQ4 owns the inference.
                    "selected_delta": summary["per_label_delta"][label],
                }
            )
            rq4_category_rows.append(
                {
                    "ordering_variant": summary["variant"],
                    "label": label,
                    "global_delta": comparison["global_delta"],
                    "category_delta": comparison["label_delta"],
                    "global_average_precision": average_precision,
                    "category_average_precision": (
                        average_precision + comparison["observed_delta_ap"]
                    ),
                    "observed_delta_ap": comparison["observed_delta_ap"],
                    "ci_lower": comparison["ci_95"][0],
                    "ci_upper": comparison["ci_95"][1],
                    "p_value": comparison["p_value_one_sided"],
                    "p_value_bh": comparison["p_value_bh"],
                    "reject_h0": comparison["reject_h0"],
                }
            )
        omnibus = summary["category_delta_macro_bootstrap"]
        rq4_global_rows.append(
            {
                "ordering_variant": summary["variant"],
                "global_delta": summary["global_delta"],
                "unique_category_deltas": len(
                    set(summary["per_label_delta"].values())
                ),
                "observed_delta_macro_ap": omnibus["observed_delta_macro_ap"],
                "ci_lower": omnibus["ci_95"][0],
                "ci_upper": omnibus["ci_95"][1],
                "p_value": omnibus["p_value_one_sided"],
                "reject_h0": omnibus["reject_h0"],
            }
        )
    return (
        sensitivity,
        rq3_global_rows,
        rq3_label_rows,
        rq4_global_rows,
        rq4_category_rows,
    )


def execute_realcats(
    paths,
    protocol: dict,
    telemetry: RunTelemetry | None = None,
    precomputed_counts_by_variant: dict[str, dict[int, pd.DataFrame]] | None = None,
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
        counts_by_delta = (
            precomputed_counts_by_variant[variant]
            if precomputed_counts_by_variant is not None
            else count_layer_parquets(
                layer_paths,
                indexed.index,
                protocol["deltas"],
                variant,
                temporary_directory=paths.data / "duckdb_tmp" / "realcats" / variant,
                telemetry=telemetry,
            )
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
            comparison["reject_h0"] = bool(
                comparison["label_delta"] != comparison["global_delta"]
                and comparison["observed_delta_ap"] > 0
                and adjusted_p < float(protocol["alpha"])
            )

        category_specific_scores = np.column_stack(
            [
                probabilities_by_delta[per_label_delta[label]][:, column]
                for column, label in enumerate(classes)
            ]
        )
        category_delta_macro = stratified_paired_bootstrap_macro_ap_delta(
            test_labels,
            classes,
            global_probabilities,
            category_specific_scores,
            iterations=int(protocol["bootstraps"]),
            seed=int(protocol["random_seed"]),
        )
        category_delta_macro["heterogeneous_selected_deltas"] = (
            len(set(per_label_delta.values())) > 1
        )
        category_delta_macro["reject_h0"] = bool(
            category_delta_macro["heterogeneous_selected_deltas"]
            and category_delta_macro["observed_delta_macro_ap"] > 0
            and category_delta_macro["p_value_one_sided"] < float(protocol["alpha"])
        )

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
            plot_label_delta_curves(
                cv_scores,
                paths.figures / "rq4_label_delta_curves.pdf",
            )

        summary = {
            "variant": variant,
            "global_delta": global_delta,
            "per_label_delta": per_label_delta,
            "shape_metrics": metrics,
            "count_sensitivity_metrics": count_sensitivity,
            "global_label_permutation": permutation,
            "label_delta_comparisons": comparisons,
            "category_delta_macro_bootstrap": category_delta_macro,
        }
        write_json_exclusive(paths.metrics / f"realcats_{variant}_results.json", summary)
        variant_summaries.append(summary)
        if telemetry:
            telemetry.mark_overall(f"unit:realcats-eval:{variant}")

    alpha = float(protocol["alpha"])
    (
        sensitivity,
        rq3_global_rows,
        rq3_label_rows,
        rq4_global_rows,
        rq4_category_rows,
    ) = _aggregate_realcats_results(variant_summaries, alpha)
    write_csv_exclusive(
        paths.tables / "rq3_global_results.csv", pd.DataFrame(rq3_global_rows)
    )
    write_csv_exclusive(
        paths.tables / "rq3_label_results.csv", pd.DataFrame(rq3_label_rows)
    )
    write_csv_exclusive(
        paths.tables / "rq4_global_results.csv", pd.DataFrame(rq4_global_rows)
    )
    write_csv_exclusive(
        paths.tables / "rq4_category_results.csv", pd.DataFrame(rq4_category_rows)
    )
    write_json_exclusive(paths.metrics / "realcats_ordering_sensitivity.json", sensitivity)
    write_json_exclusive(
        paths.metrics / "realcats_rq4_ordering_sensitivity.json",
        {
            "ordering_robust_category_specific_windows": sensitivity[
                "rq4_ordering_robust_category_specific_windows"
            ],
            "decisions": sensitivity["rq4_decisions"],
            "global_deltas": sensitivity["global_deltas"],
            "per_category_deltas": sensitivity["per_label_deltas"],
        },
    )
    return sensitivity


def execute_realcats_v6(
    paths,
    protocol: dict,
    stage_manager,
    checkpoints,
    scratch_directory: Path,
    telemetry: RunTelemetry | None = None,
) -> dict:
    """Run RQ3/RQ4 with shared layer/order/window motif checkpoints."""
    import json
    import pyarrow.parquet as pq

    dataset = protocol["datasets"]["realcats"]
    targets = pd.read_parquet(paths.root / dataset["artifacts"]["targets"])
    targets = targets.loc[targets["inference_eligible"]].copy()
    targets["address"] = targets["address"].astype(str)
    target_index = pd.Index(targets["address"], name="address")
    counts_by_variant: dict[str, dict[int, pd.DataFrame]] = {}

    for variant in protocol["ordering_variants"]:
        combined = {int(delta): pd.DataFrame(index=target_index) for delta in protocol["deltas"]}
        for layer, relative in sorted(dataset["artifacts"]["layers"].items()):
            layer_path = paths.root / relative
            row_count = int(pq.ParquetFile(layer_path).metadata.num_rows)
            graph = None
            checkpoint_ids = [
                f"realcats-{layer}-{variant}-delta-{int(delta)}"
                for delta in protocol["deltas"]
            ]
            if row_count and not all(
                (checkpoints.root / checkpoint_id).exists()
                for checkpoint_id in checkpoint_ids
            ):
                graph = RaphtoryExperimentGraph.from_parquet(
                    layer_path,
                    variant,
                    scratch_directory / "realcats" / variant / layer,
                    telemetry=telemetry,
                )
            progress_context = (
                telemetry.task_progress(
                    f"Real-CATS {layer} motif windows ({variant})",
                    len(protocol["deltas"]),
                    unit="window",
                )
                if telemetry
                else nullcontext(lambda _count=1: None)
            )
            with progress_context as advance:
                for delta in protocol["deltas"]:
                    checkpoint_id = (
                        f"realcats-{layer}-{variant}-delta-{int(delta)}"
                    )

                    def compute(delta=int(delta), graph=graph):
                        if graph is None:
                            return pd.DataFrame(
                                0,
                                index=target_index,
                                columns=MOTIF_ROLE_NAMES,
                                dtype="int64",
                            )
                        return graph.count(delta, target_index)

                    counts = checkpoints.get_or_compute(
                        checkpoint_id,
                        {
                            "dataset": "realcats",
                            "layer": layer,
                            "ordering_variant": variant,
                            "delta": int(delta),
                            "targets_sha256": dataset["artifacts"][
                                "targets_sha256"
                            ],
                            "events_sha256": dataset["artifacts"][
                                "layers_sha256"
                            ][layer],
                        },
                        compute,
                    )
                    combined[int(delta)] = combined[int(delta)].join(
                        counts.add_prefix(f"{layer}__"),
                        how="left",
                    )
                    advance()
            del graph
        counts_by_variant[variant] = {
            delta: frame.fillna(0).astype("int64") for delta, frame in combined.items()
        }

    stage_id = "realcats-ordering-evaluation"
    stage_result = None
    with stage_manager.stage(stage_id) as stage:
        if not stage.reused:
            stage_result = execute_realcats(
                stage.paths,
                protocol,
                telemetry=telemetry,
                precomputed_counts_by_variant=counts_by_variant,
            )
            # Cross-order products form the next integrity link and are committed
            # separately from per-ordering evaluation outputs.
            for path in (
                stage.paths.tables / "rq3_global_results.csv",
                stage.paths.tables / "rq3_label_results.csv",
                stage.paths.tables / "rq4_global_results.csv",
                stage.paths.tables / "rq4_category_results.csv",
                stage.paths.metrics / "realcats_ordering_sensitivity.json",
                stage.paths.metrics / "realcats_rq4_ordering_sensitivity.json",
            ):
                path.unlink()

    summaries = [
        json.loads(
            (paths.metrics / f"realcats_{variant}_results.json").read_text(encoding="utf-8")
        )
        for variant in protocol["ordering_variants"]
    ]
    alpha = float(protocol["alpha"])
    (
        sensitivity,
        rq3_global_rows,
        rq3_label_rows,
        rq4_global_rows,
        rq4_category_rows,
    ) = _aggregate_realcats_results(summaries, alpha)
    with stage_manager.stage("realcats-cross-order-aggregation") as stage:
        if not stage.reused:
            write_csv_exclusive(
                stage.paths.tables / "rq3_global_results.csv",
                pd.DataFrame(rq3_global_rows),
            )
            write_csv_exclusive(
                stage.paths.tables / "rq3_label_results.csv",
                pd.DataFrame(rq3_label_rows),
            )
            write_csv_exclusive(
                stage.paths.tables / "rq4_global_results.csv",
                pd.DataFrame(rq4_global_rows),
            )
            write_csv_exclusive(
                stage.paths.tables / "rq4_category_results.csv",
                pd.DataFrame(rq4_category_rows),
            )
            write_json_exclusive(
                stage.paths.metrics / "realcats_ordering_sensitivity.json", sensitivity
            )
            write_json_exclusive(
                stage.paths.metrics / "realcats_rq4_ordering_sensitivity.json",
                {
                    "ordering_robust_category_specific_windows": sensitivity[
                        "rq4_ordering_robust_category_specific_windows"
                    ],
                    "decisions": sensitivity["rq4_decisions"],
                    "global_deltas": sensitivity["global_deltas"],
                    "per_category_deltas": sensitivity["per_label_deltas"],
                },
            )
    return sensitivity
