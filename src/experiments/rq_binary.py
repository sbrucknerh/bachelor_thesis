"""RQ1/RQ2 execution on locked Hugging Face cohorts."""

from __future__ import annotations

from pathlib import Path
from contextlib import nullcontext

import pandas as pd

from src.experiments.artifacts import write_csv_exclusive, write_json_exclusive, write_parquet_exclusive
from src.experiments.features import (
    aggregate_features_from_parquet,
    log_motif_counts,
    motif_coverage,
)
from src.experiments.modeling import (
    cv_delta_scores,
    fit_binary_scores,
    select_delta_maximum,
    select_delta_one_standard_error,
)
from src.experiments.motifs import RaphtoryExperimentGraph
from src.experiments.resources import require_in_memory_execution
from src.experiments.statistics import (
    benjamini_hochberg,
    binary_metrics,
    grouped_score_swap_test,
    stratified_group_bootstrap_ap,
    stratified_group_bootstrap_delta,
)
from src.experiments.telemetry import RunTelemetry
from src.experiments.visualization import plot_ablation, plot_delta_performance


def _artifact_path(root: Path, relative: str) -> Path:
    return root / relative


def _save_indexed_parquet(path: Path, frame: pd.DataFrame) -> None:
    output = frame.copy()
    output.index.name = output.index.name or "address"
    write_parquet_exclusive(path, output.reset_index())


def _evaluate_cohort(
    paths,
    protocol: dict,
    targets: pd.DataFrame,
    folds: pd.DataFrame,
    baseline: pd.DataFrame,
    motif_counts: dict[int, pd.DataFrame],
    variant: str,
    telemetry: RunTelemetry | None = None,
) -> dict:
    cohort = str(targets["cohort"].iloc[0])
    indexed = targets.set_index(targets["address"].astype(str), drop=False)
    labels = indexed["label"].astype(int)
    development_addresses = indexed.index[indexed["split"] == "development"]
    test_addresses = indexed.index[indexed["split"] == "test"]
    development_labels = labels.loc[development_addresses]
    test_labels = labels.loc[test_addresses]
    development_folds = folds.loc[folds["cohort"] == cohort]

    motif_features = {delta: log_motif_counts(counts) for delta, counts in motif_counts.items()}
    combined_features = {
        delta: baseline.join(features, how="left").fillna(0.0)
        for delta, features in motif_features.items()
    }
    cv_context = (
        telemetry.step(f"hf.cv.{cohort}.{variant}", deltas=len(protocol["deltas"]))
        if telemetry
        else nullcontext()
    )
    with cv_context:
        cv_scores = cv_delta_scores(
            {
                delta: features.loc[development_addresses]
                for delta, features in combined_features.items()
            },
            development_labels,
            development_folds,
        )
    maximum_delta, selection = select_delta_maximum(cv_scores)
    one_se_delta, one_se_selection = select_delta_one_standard_error(cv_scores)
    selection = selection.merge(
        one_se_selection[
            ["delta", "within_one_se", "selected_one_se"]
        ],
        on="delta",
        how="left",
        validate="one_to_one",
    )
    selected_delta = maximum_delta

    baseline_model_context = (
        telemetry.step(f"hf.model_baseline.{cohort}.{variant}")
        if telemetry
        else nullcontext()
    )
    with baseline_model_context:
        _, baseline_scores = fit_binary_scores(
            baseline.loc[development_addresses],
            development_labels,
            baseline.loc[test_addresses],
        )
    prediction_records: list[dict] = []
    metric_records: list[dict] = []
    scores_by_delta: dict[int, dict[str, object]] = {}
    for delta in protocol["deltas"]:
        model_context = (
            telemetry.step(f"hf.models.{cohort}.{variant}.delta_{delta}")
            if telemetry
            else nullcontext()
        )
        with model_context:
            _, motif_scores = fit_binary_scores(
                motif_features[delta].loc[development_addresses],
                development_labels,
                motif_features[delta].loc[test_addresses],
            )
            _, combined_scores = fit_binary_scores(
                combined_features[delta].loc[development_addresses],
                development_labels,
                combined_features[delta].loc[test_addresses],
            )
        scores_by_delta[delta] = {"motifs": motif_scores, "combined": combined_scores}
        for arm, scores in (
            ("baseline", baseline_scores),
            ("motifs", motif_scores),
            ("combined", combined_scores),
        ):
            metrics = binary_metrics(test_labels, scores)
            metric_records.append(
                {"cohort": cohort, "variant": variant, "delta": delta, "arm": arm, **metrics}
            )
            prediction_records.extend(
                {
                    "address": address,
                    "label": int(label),
                    "campaign_group": str(indexed.loc[address, "campaign_group"]),
                    "cohort": cohort,
                    "variant": variant,
                    "delta": delta,
                    "arm": arm,
                    "score": float(score),
                }
                for address, label, score in zip(
                    test_addresses, test_labels, scores, strict=True
                )
            )

    groups = indexed.loc[test_addresses, "campaign_group"].astype(str)
    selected_scores = scores_by_delta[selected_delta]["combined"]
    statistics_context = (
        telemetry.step(
            f"hf.rq1_statistics.{cohort}.{variant}",
            permutations=int(protocol["permutations"]),
            bootstraps=int(protocol["bootstraps"]),
        )
        if telemetry
        else nullcontext()
    )
    with statistics_context:
        randomization = grouped_score_swap_test(
            test_labels,
            baseline_scores,
            selected_scores,
            groups,
            iterations=int(protocol["permutations"]),
            seed=int(protocol["random_seed"]),
        )
        confidence_interval = stratified_group_bootstrap_delta(
            test_labels,
            baseline_scores,
            selected_scores,
            groups,
            iterations=int(protocol["bootstraps"]),
            seed=int(protocol["random_seed"]),
        )

    comparisons = []
    if variant == "canonical":
        xgboost_context = (
            telemetry.step(f"hf.model_xgboost.{cohort}.{variant}")
            if telemetry
            else nullcontext()
        )
        with xgboost_context:
            _, xgboost_scores = fit_binary_scores(
                combined_features[selected_delta].loc[development_addresses],
                development_labels,
                combined_features[selected_delta].loc[test_addresses],
                model_type="xgboost",
            )
        xgboost_metrics = binary_metrics(test_labels, xgboost_scores)
        metric_records.append(
            {
                "cohort": cohort,
                "variant": variant,
                "delta": selected_delta,
                "arm": "combined_xgboost",
                **xgboost_metrics,
            }
        )
        prediction_records.extend(
            {
                "address": address,
                "label": int(label),
                "campaign_group": str(indexed.loc[address, "campaign_group"]),
                "cohort": cohort,
                "variant": variant,
                "delta": selected_delta,
                "arm": "combined_xgboost",
                "score": float(score),
            }
            for address, label, score in zip(
                test_addresses, test_labels, xgboost_scores, strict=True
            )
        )
    rq2_context = (
        telemetry.step(
            f"hf.rq2_statistics.{cohort}.{variant}",
            comparisons=len(protocol["deltas"]) - 1,
            bootstraps=int(protocol["bootstraps"]),
        )
        if telemetry
        else nullcontext()
    )
    with rq2_context:
        for delta in protocol["deltas"]:
            if delta == selected_delta:
                continue
            comparison = stratified_group_bootstrap_delta(
                test_labels,
                scores_by_delta[delta]["combined"],
                selected_scores,
                groups,
                iterations=int(protocol["bootstraps"]),
                seed=int(protocol["random_seed"]),
            )
            comparisons.append({"alternative_delta": delta, **comparison})
    if comparisons:
        adjusted = benjamini_hochberg(
            comparison["p_value_one_sided"] for comparison in comparisons
        )
        for comparison, adjusted_p in zip(comparisons, adjusted, strict=True):
            comparison["p_value_bh"] = float(adjusted_p)

    predictions = pd.DataFrame.from_records(prediction_records)
    metrics = pd.DataFrame.from_records(metric_records)
    coverage = pd.concat(
        [
            motif_coverage(motif_counts[delta].loc[indexed.index], labels)
            .reset_index()
            .assign(delta=delta, cohort=cohort, variant=variant)
            for delta in protocol["deltas"]
        ],
        ignore_index=True,
    )
    stem = f"hf_{cohort}_{variant}"
    write_parquet_exclusive(paths.predictions / f"{stem}.parquet", predictions)
    write_csv_exclusive(paths.metrics / f"{stem}_metrics.csv", metrics)
    write_csv_exclusive(paths.metrics / f"{stem}_cv.csv", cv_scores)
    write_csv_exclusive(paths.metrics / f"{stem}_delta_selection.csv", selection)
    write_csv_exclusive(paths.metrics / f"{stem}_coverage.csv", coverage)
    statistical = {
        "cohort": cohort,
        "variant": variant,
        "selected_delta": selected_delta,
        "max_mean_delta": maximum_delta,
        "one_se_delta": one_se_delta,
        "selection_objective": "development_cv_mean_combined_average_precision",
        "rq1_randomization": randomization,
        "rq1_group_bootstrap": confidence_interval,
        "rq2_selected_vs_alternatives": comparisons,
    }
    write_json_exclusive(paths.metrics / f"{stem}_statistical_tests.json", statistical)

    if variant == "canonical":
        selected_metrics = metrics.loc[metrics["delta"] == selected_delta].set_index("arm")
        plot_ablation(
            {
                arm: float(selected_metrics.loc[arm, "average_precision"])
                for arm in ("baseline", "motifs", "combined")
            },
            paths.figures / f"rq1_ablation_{cohort}.pdf",
            f"RQ1 ablation ({cohort} cohort)",
        )
        delta_metrics = metrics.loc[metrics["arm"] == "combined", ["delta", "average_precision"]]
        delta_metrics["baseline_average_precision"] = float(
            selected_metrics.loc["baseline", "average_precision"]
        )
        ci_rows = []
        for delta in protocol["deltas"]:
            ci = stratified_group_bootstrap_ap(
                test_labels,
                scores_by_delta[delta]["combined"],
                groups,
                iterations=int(protocol["bootstraps"]),
                seed=int(protocol["random_seed"]),
            )
            ci_rows.append({"delta": delta, "ci_lower": ci["ci_95"][0], "ci_upper": ci["ci_95"][1]})
        delta_metrics = delta_metrics.merge(pd.DataFrame(ci_rows), on="delta", validate="one_to_one")
        plot_delta_performance(
            delta_metrics,
            paths.figures / f"rq2_delta_performance_{cohort}.pdf",
            f"RQ2 temporal-window sensitivity ({cohort} cohort)",
        )

    return statistical


def execute_hf(
    paths, protocol: dict, telemetry: RunTelemetry | None = None
) -> dict:
    dataset = protocol["datasets"]["huggingface"]
    targets = pd.read_parquet(_artifact_path(paths.root, dataset["artifacts"]["targets"]))
    folds = pd.read_parquet(_artifact_path(paths.root, dataset["artifacts"]["folds"]))
    event_path = _artifact_path(paths.root, dataset["artifacts"]["events"])
    preflight_context = (
        telemetry.step("hf.resource_preflight", source=str(event_path))
        if telemetry
        else nullcontext()
    )
    with preflight_context:
        report = require_in_memory_execution(
            event_path,
            expansion_factor=4.0,
            columns=("timestamp", "from_address", "to_address"),
        )
        if telemetry:
            telemetry.progress("hf.resource_preflight", **report)
    target_addresses = targets["address"].astype(str)
    baseline_context = (
        telemetry.step("hf.baseline.duckdb", targets=len(target_addresses))
        if telemetry
        else nullcontext()
    )
    with baseline_context:
        baseline = aggregate_features_from_parquet(
            event_path,
            target_addresses,
            paths.data / "duckdb_tmp" / "hf_baseline",
        )
    _save_indexed_parquet(paths.features / "hf_baseline.parquet", baseline)

    variant_results: list[dict] = []
    for variant in protocol["ordering_variants"]:
        if telemetry:
            telemetry.progress("hf.ordering_variants", variant=variant, status_detail="starting")
        graph = RaphtoryExperimentGraph.from_parquet(
            event_path,
            variant,
            paths.data / "duckdb_tmp" / "hf" / variant,
            telemetry=telemetry,
        )
        motif_counts = graph.sweep(protocol["deltas"], target_addresses)
        del graph
        long_counts = pd.concat(
            [counts.reset_index().assign(delta=delta, variant=variant) for delta, counts in motif_counts.items()],
            ignore_index=True,
        )
        write_parquet_exclusive(paths.features / f"hf_motifs_{variant}.parquet", long_counts)
        for _, cohort_targets in targets.groupby("cohort", sort=True):
            cohort = str(cohort_targets["cohort"].iloc[0])
            evaluation_context = (
                telemetry.step(f"hf.evaluate.{cohort}.{variant}")
                if telemetry
                else nullcontext()
            )
            with evaluation_context:
                variant_results.append(
                    _evaluate_cohort(
                        paths,
                        protocol,
                        cohort_targets,
                        folds,
                        baseline,
                        motif_counts,
                        variant,
                        telemetry=telemetry,
                    )
                )

    alpha = float(protocol["alpha"])
    sensitivity = {}
    for cohort in sorted(targets["cohort"].unique()):
        results = [result for result in variant_results if result["cohort"] == cohort]
        decisions = [
            result["rq1_randomization"]["p_value_one_sided"] < alpha
            for result in results
        ]
        directions = [result["rq1_randomization"]["observed_delta_ap"] > 0 for result in results]
        sensitivity[cohort] = {
            "ordering_robust": len(set(decisions)) == 1 and all(directions),
            "rq1_rejected_in_all_variants": all(decisions),
            "decisions": decisions,
            "effect_directions_positive": directions,
            "selected_deltas": [result["selected_delta"] for result in results],
            "one_se_deltas": [result["one_se_delta"] for result in results],
            "rq2_unique_optimum_in_all_variants": all(
                result["rq2_selected_vs_alternatives"]
                and all(
                    comparison["observed_delta_ap"] > 0
                    and comparison["p_value_bh"] < alpha
                    for comparison in result["rq2_selected_vs_alternatives"]
                )
                for result in results
            ),
        }
    sensitivity["general_signal"] = all(
        details["ordering_robust"] and details["rq1_rejected_in_all_variants"]
        for key, details in sensitivity.items()
        if key != "general_signal"
    )
    hypothesis_rows = []
    delta_rows = []
    for result in variant_results:
        hypothesis_rows.append(
            {
                "cohort": result["cohort"],
                "ordering_variant": result["variant"],
                "selected_delta": result["selected_delta"],
                "one_se_delta": result["one_se_delta"],
                "delta_average_precision": result["rq1_randomization"]["observed_delta_ap"],
                "ci_lower": result["rq1_group_bootstrap"]["ci_95"][0],
                "ci_upper": result["rq1_group_bootstrap"]["ci_95"][1],
                "p_value": result["rq1_randomization"]["p_value_one_sided"],
                "reject_h0": result["rq1_randomization"]["p_value_one_sided"] < alpha,
            }
        )
        for comparison in result["rq2_selected_vs_alternatives"]:
            delta_rows.append(
                {
                    "cohort": result["cohort"],
                    "ordering_variant": result["variant"],
                    "selected_delta": result["selected_delta"],
                    "one_se_delta": result["one_se_delta"],
                    **comparison,
                }
            )
    write_csv_exclusive(paths.tables / "rq1_hypothesis_tests.csv", pd.DataFrame(hypothesis_rows))
    write_csv_exclusive(paths.tables / "rq2_delta_comparisons.csv", pd.DataFrame(delta_rows))
    write_json_exclusive(paths.metrics / "hf_ordering_sensitivity.json", sensitivity)
    return sensitivity


def execute_hf_v5(
    paths,
    protocol: dict,
    stage_manager,
    checkpoints,
    scratch_directory: Path,
    telemetry: RunTelemetry | None = None,
) -> dict:
    """Checkpointed v5 execution preserving the established public filenames."""
    import json

    dataset = protocol["datasets"]["huggingface"]
    targets = pd.read_parquet(
        _artifact_path(paths.root, dataset["artifacts"]["targets"])
    )
    folds = pd.read_parquet(
        _artifact_path(paths.root, dataset["artifacts"]["folds"])
    )
    event_path = _artifact_path(paths.root, dataset["artifacts"]["events"])
    target_addresses = targets["address"].astype(str)

    with stage_manager.stage("hf-baseline", {"dataset": "huggingface"}) as stage:
        if not stage.reused:
            baseline = aggregate_features_from_parquet(
                event_path,
                target_addresses,
                scratch_directory / "hf-baseline",
            )
            _save_indexed_parquet(stage.paths.features / "hf_baseline.parquet", baseline)
    baseline_frame = pd.read_parquet(paths.features / "hf_baseline.parquet")
    baseline_frame["address"] = baseline_frame["address"].astype(str)
    baseline = baseline_frame.set_index("address")

    variant_results: list[dict] = []
    for variant in protocol["ordering_variants"]:
        if telemetry:
            telemetry.progress("hf.ordering_variants", variant=variant, status_detail="starting")
        checkpoint_ids = [
            f"hf-{variant}-delta-{int(delta)}" for delta in protocol["deltas"]
        ]
        graph = None
        if not all((checkpoints.root / checkpoint_id).exists() for checkpoint_id in checkpoint_ids):
            graph = RaphtoryExperimentGraph.from_parquet(
                event_path,
                variant,
                scratch_directory / "hf" / variant,
                telemetry=telemetry,
            )
        motif_counts = {}
        progress_context = (
            telemetry.task_progress(
                f"HF motif windows ({variant})",
                len(protocol["deltas"]),
                unit="window",
            )
            if telemetry
            else nullcontext(lambda _count=1: None)
        )
        with progress_context as advance:
            for delta in protocol["deltas"]:
                checkpoint_id = f"hf-{variant}-delta-{int(delta)}"
                motif_counts[int(delta)] = checkpoints.get_or_compute(
                    checkpoint_id,
                    {
                        "dataset": "huggingface",
                        "ordering_variant": variant,
                        "delta": int(delta),
                        "targets_sha256": dataset["artifacts"]["targets_sha256"],
                        "events_sha256": dataset["artifacts"]["events_sha256"],
                    },
                    lambda delta=int(delta), graph=graph: graph.count(
                        delta,
                        target_addresses,
                    ),
                )
                advance()
        del graph

        with stage_manager.stage(
            f"hf-motifs-{variant}", {"ordering_variant": variant}
        ) as stage:
            if not stage.reused:
                long_counts = pd.concat(
                    [
                        counts.reset_index().assign(delta=delta, variant=variant)
                        for delta, counts in motif_counts.items()
                    ],
                    ignore_index=True,
                )
                write_parquet_exclusive(
                    stage.paths.features / f"hf_motifs_{variant}.parquet", long_counts
                )

        for _, cohort_targets in targets.groupby("cohort", sort=True):
            cohort = str(cohort_targets["cohort"].iloc[0])
            stage_id = f"hf-eval-{cohort}-{variant}"
            with stage_manager.stage(
                stage_id, {"cohort": cohort, "ordering_variant": variant}
            ) as stage:
                if not stage.reused:
                    result = _evaluate_cohort(
                        stage.paths,
                        protocol,
                        cohort_targets,
                        folds,
                        baseline,
                        motif_counts,
                        variant,
                        telemetry=telemetry,
                    )
                else:
                    result = None
            if result is None:
                result = json.loads(
                    (paths.metrics / f"hf_{cohort}_{variant}_statistical_tests.json").read_text(
                        encoding="utf-8"
                    )
                )
            variant_results.append(result)

    alpha = float(protocol["alpha"])
    sensitivity = {}
    for cohort in sorted(targets["cohort"].unique()):
        results = [result for result in variant_results if result["cohort"] == cohort]
        decisions = [
            result["rq1_randomization"]["p_value_one_sided"] < alpha
            for result in results
        ]
        directions = [
            result["rq1_randomization"]["observed_delta_ap"] > 0 for result in results
        ]
        sensitivity[cohort] = {
            "ordering_robust": len(set(decisions)) == 1 and all(directions),
            "rq1_rejected_in_all_variants": all(decisions),
            "decisions": decisions,
            "effect_directions_positive": directions,
            "selected_deltas": [result["selected_delta"] for result in results],
            "one_se_deltas": [result["one_se_delta"] for result in results],
            "rq2_unique_optimum_in_all_variants": all(
                result["rq2_selected_vs_alternatives"]
                and all(
                    comparison["observed_delta_ap"] > 0
                    and comparison["p_value_bh"] < alpha
                    for comparison in result["rq2_selected_vs_alternatives"]
                )
                for result in results
            ),
        }
    sensitivity["general_signal"] = all(
        details["ordering_robust"] and details["rq1_rejected_in_all_variants"]
        for key, details in sensitivity.items()
        if key != "general_signal"
    )
    hypothesis_rows = []
    delta_rows = []
    for result in variant_results:
        hypothesis_rows.append(
            {
                "cohort": result["cohort"],
                "ordering_variant": result["variant"],
                "selected_delta": result["selected_delta"],
                "one_se_delta": result["one_se_delta"],
                "delta_average_precision": result["rq1_randomization"]["observed_delta_ap"],
                "ci_lower": result["rq1_group_bootstrap"]["ci_95"][0],
                "ci_upper": result["rq1_group_bootstrap"]["ci_95"][1],
                "p_value": result["rq1_randomization"]["p_value_one_sided"],
                "reject_h0": result["rq1_randomization"]["p_value_one_sided"] < alpha,
            }
        )
        for comparison in result["rq2_selected_vs_alternatives"]:
            delta_rows.append(
                {
                    "cohort": result["cohort"],
                    "ordering_variant": result["variant"],
                    "selected_delta": result["selected_delta"],
                    "one_se_delta": result["one_se_delta"],
                    **comparison,
                }
            )
    with stage_manager.stage("hf-cross-order-aggregation") as stage:
        if not stage.reused:
            write_csv_exclusive(
                stage.paths.tables / "rq1_hypothesis_tests.csv",
                pd.DataFrame(hypothesis_rows),
            )
            write_csv_exclusive(
                stage.paths.tables / "rq2_delta_comparisons.csv",
                pd.DataFrame(delta_rows),
            )
            write_json_exclusive(
                stage.paths.metrics / "hf_ordering_sensitivity.json", sensitivity
            )
    return sensitivity
