"""RQ1/RQ2 execution on locked Hugging Face cohorts."""

from __future__ import annotations

from pathlib import Path
from contextlib import nullcontext

import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from src.experiments.artifacts import write_csv_exclusive, write_json_exclusive, write_parquet_exclusive
from src.experiments.constants import (
    HF_REPRESENTATION_ANALYSIS,
    HF_TRIANGLE_WINDOW_ANALYSIS,
    MOTIF_ROLE_NAMES,
    TRIANGLE_ROLE_NAMES,
)
from src.experiments.features import (
    aggregate_features_from_parquet,
    binary_motif_representations,
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
from src.experiments.visualization import (
    plot_ablation,
    plot_delta_performance,
    plot_motif_coverage,
    plot_representation_ablation,
)


def _artifact_path(root: Path, relative: str) -> Path:
    return root / relative


def _save_indexed_parquet(path: Path, frame: pd.DataFrame) -> None:
    output = frame.copy()
    output.index.name = output.index.name or "address"
    write_parquet_exclusive(path, output.reset_index())


def _joined_baseline(baseline: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
    return baseline.join(features, how="left").fillna(0.0)


def _diagnostic_feature_arms(
    baseline: pd.DataFrame,
    counts: pd.DataFrame,
    incoming_star_roles: list[str],
) -> dict[str, pd.DataFrame]:
    representations = binary_motif_representations(counts, incoming_star_roles)
    arms = {
        **representations,
        "baseline": baseline,
        "baseline_presence": _joined_baseline(
            baseline, representations["motif_presence"]
        ),
        "baseline_volume": _joined_baseline(
            baseline, representations["motif_volume"]
        ),
        "baseline_composition": _joined_baseline(
            baseline, representations["motif_composition"]
        ),
        "baseline_composition_volume": _joined_baseline(
            baseline, representations["motif_composition_volume"]
        ),
        "baseline_incoming_stars": _joined_baseline(
            baseline, representations["motif_incoming_stars"]
        ),
        "combined": _joined_baseline(baseline, representations["motifs"]),
    }
    if set(TRIANGLE_ROLE_NAMES) <= set(counts.columns):
        without_triangles = log_motif_counts(counts.loc[:, MOTIF_ROLE_NAMES])
        triangles = log_motif_counts(counts.loc[:, TRIANGLE_ROLE_NAMES])
        arms.update(
            {
                "triangles_only": triangles,
                "baseline_triangles": _joined_baseline(baseline, triangles),
                "motifs_without_triangles": without_triangles,
                "combined_without_triangles": _joined_baseline(
                    baseline, without_triangles
                ),
            }
        )
    return arms


def _representation_comparison_specs(definition: dict) -> tuple[tuple[str, str, str], ...]:
    return tuple(
        (
            str(item["comparison"]),
            str(item["reference_arm"]),
            str(item["alternative_arm"]),
        )
        for item in definition["formal_comparisons"]
    )


def _append_binary_output(
    *,
    metric_records: list[dict],
    prediction_records: list[dict],
    indexed: pd.DataFrame,
    test_addresses: pd.Index,
    test_labels: pd.Series,
    cohort: str,
    variant: str,
    delta: int,
    arm: str,
    scores,
    replication_index: int,
) -> None:
    metric_records.append(
        {
            "replication_index": int(replication_index),
            "cohort": cohort,
            "variant": variant,
            "delta": int(delta),
            "arm": arm,
            **binary_metrics(test_labels, scores),
        }
    )
    prediction_records.extend(
        {
            "address": address,
            "label": int(label),
            "campaign_group": str(indexed.loc[address, "campaign_group"]),
            "replication_index": int(replication_index),
            "cohort": cohort,
            "variant": variant,
            "delta": int(delta),
            "arm": arm,
            "score": float(score),
        }
        for address, label, score in zip(
            test_addresses, test_labels, scores, strict=True
        )
    )


def _role_diagnostics(
    counts: pd.DataFrame,
    indexed: pd.DataFrame,
    delta: int,
    replication_index: int,
) -> pd.DataFrame:
    records: list[dict] = []
    for scope in ("development", "test"):
        addresses = indexed.index[indexed["split"] == scope]
        labels = indexed.loc[addresses, "label"].astype(int)
        scoped = counts.loc[addresses]
        for role in scoped.columns:
            values = scoped[role].astype("float64")
            nonzero = values.gt(0)
            record = {
                "replication_index": int(replication_index),
                "scope": scope,
                "delta": int(delta),
                "role": str(role),
                "n": int(len(values)),
                "average_precision_direct_count": float(
                    average_precision_score(labels, values)
                ),
                "roc_auc_direct_count": float(roc_auc_score(labels, values)),
                "nonzero_coverage": float(nonzero.mean()),
            }
            for label in (0, 1):
                selected = labels.eq(label)
                record[f"mean_count_label_{label}"] = float(values.loc[selected].mean())
                record[f"nonzero_coverage_label_{label}"] = float(
                    nonzero.loc[selected].mean()
                )
            records.append(record)
    return pd.DataFrame.from_records(records)


def _coverage_summary(
    counts_by_delta: dict[int, pd.DataFrame],
    indexed: pd.DataFrame,
    cohort: str,
    variant: str,
    replication_index: int,
) -> pd.DataFrame:
    records: list[pd.DataFrame] = []
    for delta, counts in counts_by_delta.items():
        representations = {"all_40_roles": counts}
        if set(TRIANGLE_ROLE_NAMES) <= set(counts.columns):
            representations.update(
                {
                    "non_triangle_32_roles": counts.loc[:, MOTIF_ROLE_NAMES],
                    "triangles_only": counts.loc[:, TRIANGLE_ROLE_NAMES],
                }
            )
        for representation, representation_counts in representations.items():
            for scope in ("all", "development", "test"):
                addresses = (
                    indexed.index
                    if scope == "all"
                    else indexed.index[indexed["split"] == scope]
                )
                scoped_counts = representation_counts.loc[addresses]
                labels = indexed.loc[addresses, "label"].astype(int)
                by_label = motif_coverage(scoped_counts, labels).reset_index()
                by_label["label"] = by_label["label"].astype(str)
                overall = motif_coverage(scoped_counts).assign(label="all")
                frame = pd.concat([overall, by_label], ignore_index=True)
                frame["zero"] = frame["n"] - frame["nonzero"]
                frame["zero_fraction"] = 1.0 - frame["coverage"]
                frame = frame.assign(
                    replication_index=int(replication_index),
                    delta=int(delta),
                    cohort=cohort,
                    variant=variant,
                    scope=scope,
                    representation=representation,
                )
                records.append(frame)
    return pd.concat(records, ignore_index=True)


def _delta_selection_tables(
    cv_scores: pd.DataFrame,
) -> tuple[int, int, pd.DataFrame]:
    maximum_delta, selection = select_delta_maximum(cv_scores)
    one_se_delta, one_se_selection = select_delta_one_standard_error(cv_scores)
    selection = selection.merge(
        one_se_selection[["delta", "within_one_se", "selected_one_se"]],
        on="delta",
        how="left",
        validate="one_to_one",
    )
    return maximum_delta, one_se_delta, selection


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
    replication_index = int(protocol.get("replication", {}).get("index", 0))
    representation_definition = protocol.get(
        "hf_representation_analysis", HF_REPRESENTATION_ANALYSIS
    )
    incoming_star_roles = list(representation_definition["incoming_star_roles"])
    diagnostic_arm_names = tuple(representation_definition["diagnostic_arms"])
    comparison_specs = _representation_comparison_specs(representation_definition)

    motif_features = {
        delta: log_motif_counts(counts) for delta, counts in motif_counts.items()
    }
    non_triangle_features = {
        delta: log_motif_counts(counts.loc[:, MOTIF_ROLE_NAMES])
        for delta, counts in motif_counts.items()
    }
    triangle_features = {
        delta: log_motif_counts(counts.loc[:, TRIANGLE_ROLE_NAMES])
        for delta, counts in motif_counts.items()
    }
    combined_features = {
        delta: _joined_baseline(baseline, features)
        for delta, features in motif_features.items()
    }
    combined_non_triangle_features = {
        delta: _joined_baseline(baseline, features)
        for delta, features in non_triangle_features.items()
    }
    baseline_triangle_features = {
        delta: _joined_baseline(baseline, features)
        for delta, features in triangle_features.items()
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
        cv_scores_32 = cv_delta_scores(
            {
                delta: features.loc[development_addresses]
                for delta, features in combined_non_triangle_features.items()
            },
            development_labels,
            development_folds,
        )
        cv_scores_triangles = cv_delta_scores(
            {
                delta: features.loc[development_addresses]
                for delta, features in triangle_features.items()
            },
            development_labels,
            development_folds,
        )
    maximum_delta, one_se_delta, selection = _delta_selection_tables(cv_scores)
    maximum_delta_32, one_se_delta_32, selection_32 = _delta_selection_tables(
        cv_scores_32
    )
    maximum_delta_triangles, one_se_delta_triangles, selection_triangles = (
        _delta_selection_tables(cv_scores_triangles)
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
            _, non_triangle_scores = fit_binary_scores(
                non_triangle_features[delta].loc[development_addresses],
                development_labels,
                non_triangle_features[delta].loc[test_addresses],
            )
            _, combined_non_triangle_scores = fit_binary_scores(
                combined_non_triangle_features[delta].loc[development_addresses],
                development_labels,
                combined_non_triangle_features[delta].loc[test_addresses],
            )
            _, triangle_scores = fit_binary_scores(
                triangle_features[delta].loc[development_addresses],
                development_labels,
                triangle_features[delta].loc[test_addresses],
            )
            _, baseline_triangle_scores = fit_binary_scores(
                baseline_triangle_features[delta].loc[development_addresses],
                development_labels,
                baseline_triangle_features[delta].loc[test_addresses],
            )
        scores_by_delta[delta] = {
            "motifs": motif_scores,
            "combined": combined_scores,
            "motifs_without_triangles": non_triangle_scores,
            "combined_without_triangles": combined_non_triangle_scores,
            "triangles_only": triangle_scores,
            "baseline_triangles": baseline_triangle_scores,
        }
        for arm, scores in (
            ("baseline", baseline_scores),
            ("motifs", motif_scores),
            ("combined", combined_scores),
            ("motifs_without_triangles", non_triangle_scores),
            ("combined_without_triangles", combined_non_triangle_scores),
            ("triangles_only", triangle_scores),
            ("baseline_triangles", baseline_triangle_scores),
        ):
            _append_binary_output(
                metric_records=metric_records,
                prediction_records=prediction_records,
                indexed=indexed,
                test_addresses=test_addresses,
                test_labels=test_labels,
                cohort=cohort,
                variant=variant,
                delta=int(delta),
                arm=arm,
                scores=scores,
                replication_index=replication_index,
            )

    diagnostic_features = _diagnostic_feature_arms(
        baseline,
        motif_counts[selected_delta],
        incoming_star_roles,
    )
    diagnostic_scores = {
        "baseline": baseline_scores,
        **scores_by_delta[selected_delta],
    }
    diagnostic_context = (
        telemetry.step(
            f"hf.representation_models.{cohort}.{variant}",
            arms=len(diagnostic_arm_names),
            delta=int(selected_delta),
        )
        if telemetry
        else nullcontext()
    )
    with diagnostic_context:
        for arm in diagnostic_arm_names:
            if arm in diagnostic_scores:
                continue
            features = diagnostic_features[arm]
            _, scores = fit_binary_scores(
                features.loc[development_addresses],
                development_labels,
                features.loc[test_addresses],
            )
            diagnostic_scores[arm] = scores
            _append_binary_output(
                metric_records=metric_records,
                prediction_records=prediction_records,
                indexed=indexed,
                test_addresses=test_addresses,
                test_labels=test_labels,
                cohort=cohort,
                variant=variant,
                delta=int(selected_delta),
                arm=arm,
                scores=scores,
                replication_index=replication_index,
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

    triangle_window_reference = scores_by_delta[maximum_delta_32][
        "combined_without_triangles"
    ]
    triangle_window_alternative = scores_by_delta[maximum_delta]["combined"]
    triangle_window_comparison = {
        "definition": protocol.get(
            "hf_triangle_window_analysis", HF_TRIANGLE_WINDOW_ANALYSIS
        ),
        "all_40_selected_delta": int(maximum_delta),
        "all_40_one_se_delta": int(one_se_delta),
        "non_triangle_32_selected_delta": int(maximum_delta_32),
        "non_triangle_32_one_se_delta": int(one_se_delta_32),
        "triangles_only_selected_delta": int(maximum_delta_triangles),
        "triangles_only_one_se_delta": int(one_se_delta_triangles),
        "all_40_average_precision": float(
            average_precision_score(test_labels, triangle_window_alternative)
        ),
        "non_triangle_32_average_precision": float(
            average_precision_score(test_labels, triangle_window_reference)
        ),
        "triangles_only_average_precision": float(
            average_precision_score(
                test_labels,
                scores_by_delta[maximum_delta_triangles]["triangles_only"],
            )
        ),
    }
    triangle_window_comparison["observed_delta_ap"] = float(
        triangle_window_comparison["all_40_average_precision"]
        - triangle_window_comparison["non_triangle_32_average_precision"]
    )
    if variant == "canonical":
        triangle_window_permutation = grouped_score_swap_test(
            test_labels,
            triangle_window_reference,
            triangle_window_alternative,
            groups,
            iterations=int(protocol["permutations"]),
            seed=int(protocol["random_seed"]),
        )
        triangle_window_bootstrap = stratified_group_bootstrap_delta(
            test_labels,
            triangle_window_reference,
            triangle_window_alternative,
            groups,
            iterations=int(protocol["bootstraps"]),
            seed=int(protocol["random_seed"]),
        )
        triangle_window_comparison.update(
            {
                "permutation": triangle_window_permutation,
                "bootstrap": triangle_window_bootstrap,
            }
        )

    representation_effects = {
        comparison: float(
            average_precision_score(test_labels, diagnostic_scores[alternative])
            - average_precision_score(test_labels, diagnostic_scores[reference])
        )
        for comparison, reference, alternative in comparison_specs
    }
    representation_comparisons: list[dict] = []
    if variant == representation_definition["formal_ordering_variant"]:
        representation_context = (
            telemetry.step(
                f"hf.representation_statistics.{cohort}.{variant}",
                comparisons=len(comparison_specs),
                permutations=int(protocol["permutations"]),
                bootstraps=int(protocol["bootstraps"]),
            )
            if telemetry
            else nullcontext()
        )
        with representation_context:
            for comparison, reference, alternative in comparison_specs:
                permutation = grouped_score_swap_test(
                    test_labels,
                    diagnostic_scores[reference],
                    diagnostic_scores[alternative],
                    groups,
                    iterations=int(protocol["permutations"]),
                    seed=int(protocol["random_seed"]),
                )
                bootstrap = stratified_group_bootstrap_delta(
                    test_labels,
                    diagnostic_scores[reference],
                    diagnostic_scores[alternative],
                    groups,
                    iterations=int(protocol["bootstraps"]),
                    seed=int(protocol["random_seed"]),
                )
                representation_comparisons.append(
                    {
                        "comparison": comparison,
                        "reference_arm": reference,
                        "alternative_arm": alternative,
                        "reference_average_precision": float(
                            average_precision_score(
                                test_labels, diagnostic_scores[reference]
                            )
                        ),
                        "alternative_average_precision": float(
                            average_precision_score(
                                test_labels, diagnostic_scores[alternative]
                            )
                        ),
                        **permutation,
                        "ci_95": bootstrap["ci_95"],
                        "bootstrap_p_value_one_sided": bootstrap[
                            "p_value_one_sided"
                        ],
                    }
                )
        adjusted = benjamini_hochberg(
            item["p_value_one_sided"] for item in representation_comparisons
        )
        for item, adjusted_p in zip(
            representation_comparisons, adjusted, strict=True
        ):
            item["p_value_bh"] = float(adjusted_p)
            item["reject_h0_bh"] = bool(
                adjusted_p < float(protocol["alpha"])
            )

    comparisons = []
    xgboost_robustness = None
    if variant == "canonical":
        xgboost_context = (
            telemetry.step(f"hf.model_xgboost.{cohort}.{variant}")
            if telemetry
            else nullcontext()
        )
        xgboost_scores: dict[str, object] = {}
        with xgboost_context:
            for arm, features in (
                ("baseline_xgboost", diagnostic_features["baseline"]),
                ("motifs_xgboost", diagnostic_features["motifs"]),
                ("combined_xgboost", diagnostic_features["combined"]),
            ):
                _, scores = fit_binary_scores(
                    features.loc[development_addresses],
                    development_labels,
                    features.loc[test_addresses],
                    model_type="xgboost",
                )
                xgboost_scores[arm] = scores
                _append_binary_output(
                    metric_records=metric_records,
                    prediction_records=prediction_records,
                    indexed=indexed,
                    test_addresses=test_addresses,
                    test_labels=test_labels,
                    cohort=cohort,
                    variant=variant,
                    delta=int(selected_delta),
                    arm=arm,
                    scores=scores,
                    replication_index=replication_index,
                )
        xgb_permutation = grouped_score_swap_test(
            test_labels,
            xgboost_scores["baseline_xgboost"],
            xgboost_scores["combined_xgboost"],
            groups,
            iterations=int(protocol["permutations"]),
            seed=int(protocol["random_seed"]),
        )
        xgb_bootstrap = stratified_group_bootstrap_delta(
            test_labels,
            xgboost_scores["baseline_xgboost"],
            xgboost_scores["combined_xgboost"],
            groups,
            iterations=int(protocol["bootstraps"]),
            seed=int(protocol["random_seed"]),
        )
        xgboost_robustness = {
            "reference_arm": "baseline_xgboost",
            "alternative_arm": "combined_xgboost",
            **xgb_permutation,
            "ci_95": xgb_bootstrap["ci_95"],
            "bootstrap_p_value_one_sided": xgb_bootstrap[
                "p_value_one_sided"
            ],
        }
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
    coverage = _coverage_summary(
        motif_counts,
        indexed,
        cohort,
        variant,
        replication_index,
    )
    stem = f"hf_{cohort}_{variant}"
    write_parquet_exclusive(paths.predictions / f"{stem}.parquet", predictions)
    write_csv_exclusive(paths.metrics / f"{stem}_metrics.csv", metrics)
    write_csv_exclusive(paths.metrics / f"{stem}_cv.csv", cv_scores)
    write_csv_exclusive(paths.metrics / f"{stem}_cv_32.csv", cv_scores_32)
    write_csv_exclusive(
        paths.metrics / f"{stem}_cv_triangles.csv", cv_scores_triangles
    )
    write_csv_exclusive(paths.metrics / f"{stem}_delta_selection.csv", selection)
    write_csv_exclusive(
        paths.metrics / f"{stem}_delta_selection_32.csv", selection_32
    )
    write_csv_exclusive(
        paths.metrics / f"{stem}_delta_selection_triangles.csv",
        selection_triangles,
    )
    write_csv_exclusive(paths.metrics / f"{stem}_coverage.csv", coverage)
    if variant == representation_definition["formal_ordering_variant"]:
        role_diagnostics = _role_diagnostics(
            motif_counts[selected_delta],
            indexed,
            selected_delta,
            replication_index,
        )
        role_diagnostics.insert(1, "cohort", cohort)
        role_diagnostics.insert(2, "variant", variant)
        write_csv_exclusive(
            paths.metrics / f"{stem}_role_diagnostics.csv", role_diagnostics
        )
    statistical = {
        "replication_index": replication_index,
        "cohort": cohort,
        "variant": variant,
        "selected_delta": selected_delta,
        "max_mean_delta": maximum_delta,
        "one_se_delta": one_se_delta,
        "selection_objective": "development_cv_mean_combined_average_precision",
        "rq1_randomization": randomization,
        "rq1_group_bootstrap": confidence_interval,
        "representation_analysis": {
            "definition": representation_definition,
            "selected_delta": selected_delta,
            "effects": representation_effects,
            "formal_comparisons": representation_comparisons,
        },
        "triangle_window_analysis": triangle_window_comparison,
        "xgboost_robustness": xgboost_robustness,
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
        plot_representation_ablation(
            {
                arm: float(selected_metrics.loc[arm, "average_precision"])
                for arm in (
                    "baseline",
                    "baseline_volume",
                    "baseline_incoming_stars",
                    "baseline_triangles",
                    "combined_without_triangles",
                    "combined",
                )
                if arm in selected_metrics.index
            },
            paths.figures / f"rq1_representation_ablation_{cohort}.pdf",
            f"RQ1 representation diagnostics ({cohort} cohort)",
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
        plot_motif_coverage(
            coverage.loc[
                (coverage["scope"] == "all")
                & (coverage["representation"] == "all_40_roles")
            ],
            paths.figures / f"rq2_motif_coverage_{cohort}.pdf",
            f"RQ2 motif coverage ({cohort} cohort)",
        )

    return statistical


def _aggregate_hf_results(
    output_paths,
    source_paths,
    protocol: dict,
    targets: pd.DataFrame,
    variant_results: list[dict],
) -> dict:
    alpha = float(protocol["alpha"])
    replication_index = int(protocol.get("replication", {}).get("index", 0))
    sensitivity: dict = {}
    representation_sensitivity: dict = {
        "replication_index": replication_index,
        "formal_ordering_variant": protocol.get(
            "hf_representation_analysis", HF_REPRESENTATION_ANALYSIS
        )["formal_ordering_variant"],
        "cohorts": {},
    }
    for cohort in sorted(targets["cohort"].unique()):
        results = [result for result in variant_results if result["cohort"] == cohort]
        decisions = [
            result["rq1_randomization"]["p_value_one_sided"] < alpha
            for result in results
        ]
        directions = [
            result["rq1_randomization"]["observed_delta_ap"] > 0
            for result in results
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
        effects = {
            comparison: [
                float(result["representation_analysis"]["effects"][comparison])
                for result in results
            ]
            for comparison, _, _ in _representation_comparison_specs(
                protocol.get("hf_representation_analysis", HF_REPRESENTATION_ANALYSIS)
            )
        }
        formal = next(
            (
                result["representation_analysis"]["formal_comparisons"]
                for result in results
                if result["representation_analysis"]["formal_comparisons"]
            ),
            [],
        )
        representation_sensitivity["cohorts"][cohort] = {
            "selected_deltas": [int(result["selected_delta"]) for result in results],
            "effects_by_comparison": effects,
            "positive_in_all_orderings": {
                comparison: all(value > 0 for value in values)
                for comparison, values in effects.items()
            },
            "formal_bh_decisions": {
                item["comparison"]: bool(item["reject_h0_bh"])
                for item in formal
            },
        }
    sensitivity["general_signal"] = all(
        details["ordering_robust"] and details["rq1_rejected_in_all_variants"]
        for key, details in sensitivity.items()
        if key != "general_signal"
    )

    hypothesis_rows = []
    delta_rows = []
    representation_rows = []
    xgboost_rows = []
    triangle_window_rows = []
    representation_delta_rows = []
    for result in variant_results:
        hypothesis_rows.append(
            {
                "replication_index": replication_index,
                "cohort": result["cohort"],
                "ordering_variant": result["variant"],
                "selected_delta": result["selected_delta"],
                "one_se_delta": result["one_se_delta"],
                "delta_average_precision": result["rq1_randomization"][
                    "observed_delta_ap"
                ],
                "ci_lower": result["rq1_group_bootstrap"]["ci_95"][0],
                "ci_upper": result["rq1_group_bootstrap"]["ci_95"][1],
                "p_value": result["rq1_randomization"]["p_value_one_sided"],
                "reject_h0": result["rq1_randomization"]["p_value_one_sided"]
                < alpha,
            }
        )
        for comparison in result["rq2_selected_vs_alternatives"]:
            delta_rows.append(
                {
                    "replication_index": replication_index,
                    "cohort": result["cohort"],
                    "ordering_variant": result["variant"],
                    "selected_delta": result["selected_delta"],
                    "one_se_delta": result["one_se_delta"],
                    **comparison,
                }
            )
        for comparison in result["representation_analysis"]["formal_comparisons"]:
            representation_rows.append(
                {
                    "replication_index": replication_index,
                    "cohort": result["cohort"],
                    "ordering_variant": result["variant"],
                    "selected_delta": result["selected_delta"],
                    "comparison": comparison["comparison"],
                    "reference_arm": comparison["reference_arm"],
                    "alternative_arm": comparison["alternative_arm"],
                    "reference_average_precision": comparison[
                        "reference_average_precision"
                    ],
                    "alternative_average_precision": comparison[
                        "alternative_average_precision"
                    ],
                    "observed_delta_ap": comparison["observed_delta_ap"],
                    "ci_lower": comparison["ci_95"][0],
                    "ci_upper": comparison["ci_95"][1],
                    "p_value": comparison["p_value_one_sided"],
                    "p_value_bh": comparison["p_value_bh"],
                    "bootstrap_p_value": comparison[
                        "bootstrap_p_value_one_sided"
                    ],
                    "reject_h0_bh": comparison["reject_h0_bh"],
                }
            )
        xgboost = result.get("xgboost_robustness")
        if xgboost:
            xgboost_rows.append(
                {
                    "replication_index": replication_index,
                    "cohort": result["cohort"],
                    "ordering_variant": result["variant"],
                    "selected_delta": result["selected_delta"],
                    "reference_arm": xgboost["reference_arm"],
                    "alternative_arm": xgboost["alternative_arm"],
                    "observed_delta_ap": xgboost["observed_delta_ap"],
                    "ci_lower": xgboost["ci_95"][0],
                    "ci_upper": xgboost["ci_95"][1],
                    "p_value": xgboost["p_value_one_sided"],
                    "bootstrap_p_value": xgboost[
                        "bootstrap_p_value_one_sided"
                    ],
                    "reject_h0": xgboost["p_value_one_sided"] < alpha,
                }
            )
        triangle_window = result["triangle_window_analysis"]
        triangle_window_row = {
            "replication_index": replication_index,
            "cohort": result["cohort"],
            "ordering_variant": result["variant"],
            "all_40_selected_delta": triangle_window["all_40_selected_delta"],
            "non_triangle_32_selected_delta": triangle_window[
                "non_triangle_32_selected_delta"
            ],
            "triangles_only_selected_delta": triangle_window[
                "triangles_only_selected_delta"
            ],
            "all_40_average_precision": triangle_window[
                "all_40_average_precision"
            ],
            "non_triangle_32_average_precision": triangle_window[
                "non_triangle_32_average_precision"
            ],
            "triangles_only_average_precision": triangle_window[
                "triangles_only_average_precision"
            ],
            "observed_delta_ap": triangle_window["observed_delta_ap"],
        }
        if "permutation" in triangle_window:
            triangle_window_row.update(
                {
                    "p_value": triangle_window["permutation"][
                        "p_value_one_sided"
                    ],
                    "ci_lower": triangle_window["bootstrap"]["ci_95"][0],
                    "ci_upper": triangle_window["bootstrap"]["ci_95"][1],
                }
            )
            triangle_window_rows.append(triangle_window_row)
        for representation, selected_key, one_se_key, ap_key in (
            (
                "all_40_roles",
                "all_40_selected_delta",
                "all_40_one_se_delta",
                "all_40_average_precision",
            ),
            (
                "non_triangle_32_roles",
                "non_triangle_32_selected_delta",
                "non_triangle_32_one_se_delta",
                "non_triangle_32_average_precision",
            ),
            (
                "triangles_only",
                "triangles_only_selected_delta",
                "triangles_only_one_se_delta",
                "triangles_only_average_precision",
            ),
        ):
            representation_delta_rows.append(
                {
                    "replication_index": replication_index,
                    "cohort": result["cohort"],
                    "ordering_variant": result["variant"],
                    "representation": representation,
                    "selected_delta": triangle_window[selected_key],
                    "one_se_delta": triangle_window[one_se_key],
                    "test_average_precision": triangle_window[ap_key],
                }
            )

    role_frames = [
        pd.read_csv(
            source_paths.metrics
            / f"hf_{cohort}_{representation_sensitivity['formal_ordering_variant']}_role_diagnostics.csv"
        )
        for cohort in sorted(targets["cohort"].unique())
    ]
    coverage_frames = [
        pd.read_csv(
            source_paths.metrics
            / f"hf_{cohort}_{representation_sensitivity['formal_ordering_variant']}_coverage.csv"
        )
        for cohort in sorted(targets["cohort"].unique())
    ]
    write_csv_exclusive(
        output_paths.tables / "rq1_hypothesis_tests.csv",
        pd.DataFrame(hypothesis_rows),
    )
    representation_frame = pd.DataFrame(representation_rows)
    write_csv_exclusive(
        output_paths.tables / "rq1_representation_tests.csv",
        representation_frame,
    )
    triangle_comparisons = {
        "baseline_triangles_vs_baseline",
        "combined_vs_combined_without_triangles",
    }
    write_csv_exclusive(
        output_paths.tables / "rq1_triangle_ablation.csv",
        representation_frame.loc[
            representation_frame["comparison"].isin(triangle_comparisons)
        ].reset_index(drop=True),
    )
    write_csv_exclusive(
        output_paths.tables / "rq1_triangle_window_analysis.csv",
        pd.DataFrame(triangle_window_rows),
    )
    write_csv_exclusive(
        output_paths.tables / "rq1_role_diagnostics.csv",
        pd.concat(role_frames, ignore_index=True),
    )
    write_csv_exclusive(
        output_paths.tables / "rq1_xgboost_robustness.csv",
        pd.DataFrame(xgboost_rows),
    )
    write_csv_exclusive(
        output_paths.tables / "rq2_delta_comparisons.csv",
        pd.DataFrame(delta_rows),
    )
    write_csv_exclusive(
        output_paths.tables / "rq2_motif_coverage.csv",
        pd.concat(coverage_frames, ignore_index=True),
    )
    write_csv_exclusive(
        output_paths.tables / "rq2_representation_delta_selection.csv",
        pd.DataFrame(representation_delta_rows),
    )
    write_json_exclusive(
        output_paths.metrics / "hf_ordering_sensitivity.json", sensitivity
    )
    write_json_exclusive(
        output_paths.metrics / "hf_representation_sensitivity.json",
        representation_sensitivity,
    )
    return sensitivity


def execute_hf(
    paths, protocol: dict, telemetry: RunTelemetry | None = None
) -> dict:
    dataset = protocol["datasets"]["huggingface"]
    targets = pd.read_parquet(_artifact_path(paths.root, dataset["artifacts"]["targets"]))
    folds = pd.read_parquet(_artifact_path(paths.root, dataset["artifacts"]["folds"]))
    event_path = _artifact_path(paths.root, dataset["artifacts"]["events"])
    motif_event_path = _artifact_path(
        paths.root,
        dataset["artifacts"].get("motif_events", dataset["artifacts"]["events"]),
    )
    preflight_context = (
        telemetry.step("hf.resource_preflight", source=str(motif_event_path))
        if telemetry
        else nullcontext()
    )
    with preflight_context:
        report = require_in_memory_execution(
            motif_event_path,
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
            motif_event_path,
            variant,
            paths.data / "duckdb_tmp" / "hf" / variant,
            telemetry=telemetry,
        )
        motif_counts = graph.sweep_all(protocol["deltas"], target_addresses)
        del graph
        long_counts = pd.concat(
            [
                counts.reset_index().assign(delta=delta, variant=variant)
                for delta, counts in motif_counts.items()
            ],
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

    return _aggregate_hf_results(paths, paths, protocol, targets, variant_results)


def execute_hf_v8(
    paths,
    protocol: dict,
    stage_manager,
    checkpoints,
    scratch_directory: Path,
    telemetry: RunTelemetry | None = None,
) -> dict:
    """Checkpointed v8 HF execution preserving established public filenames."""
    import json

    dataset = protocol["datasets"]["huggingface"]
    targets = pd.read_parquet(
        _artifact_path(paths.root, dataset["artifacts"]["targets"])
    )
    folds = pd.read_parquet(
        _artifact_path(paths.root, dataset["artifacts"]["folds"])
    )
    event_path = _artifact_path(paths.root, dataset["artifacts"]["events"])
    motif_event_path = _artifact_path(
        paths.root,
        dataset["artifacts"].get("motif_events", dataset["artifacts"]["events"]),
    )
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
                motif_event_path,
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
                        "motif_events_sha256": dataset["artifacts"].get(
                            "motif_events_sha256",
                            dataset["artifacts"]["events_sha256"],
                        ),
                        "motif_roles_retained": 40,
                    },
                    lambda delta=int(delta), graph=graph: graph.count_all(
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

    with stage_manager.stage("hf-cross-order-aggregation") as stage:
        if not stage.reused:
            sensitivity = _aggregate_hf_results(
                stage.paths,
                paths,
                protocol,
                targets,
                variant_results,
            )
        else:
            sensitivity = json.loads(
                (paths.metrics / "hf_ordering_sensitivity.json").read_text(
                    encoding="utf-8"
                )
            )
    return sensitivity
