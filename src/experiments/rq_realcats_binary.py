"""HF-disjoint activity-matched Real-CATS binary replication for RQ1/RQ2."""

from __future__ import annotations

from contextlib import nullcontext
import json
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
from sklearn.metrics import average_precision_score

from src.experiments.artifacts import (
    write_csv_exclusive,
    write_json_exclusive,
    write_parquet_exclusive,
)
from src.experiments.constants import MOTIF_ROLE_NAMES
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
from src.experiments.statistics import (
    benjamini_hochberg,
    binary_metrics,
    grouped_score_swap_test,
    stratified_group_bootstrap_delta,
)
from src.experiments.telemetry import RunTelemetry


def _save_indexed(path: Path, frame: pd.DataFrame) -> None:
    output = frame.copy()
    output.index.name = "address"
    write_parquet_exclusive(path, output.reset_index())


def _build_layered_baseline(
    layer_paths: dict[str, Path],
    addresses: pd.Index,
    scratch: Path,
) -> pd.DataFrame:
    frames = []
    for layer, path in sorted(layer_paths.items()):
        layer_features = aggregate_features_from_parquet(
            path,
            addresses,
            scratch / layer,
        )
        frames.append(layer_features.add_prefix(f"{layer}__"))
    return pd.concat(frames, axis=1).reindex(addresses).fillna(0.0)


def _coverage_rows(
    counts_by_delta: dict[int, pd.DataFrame],
    targets: pd.DataFrame,
    variant: str,
) -> pd.DataFrame:
    indexed = targets.set_index("address")
    records: list[pd.DataFrame] = []
    for delta, counts in counts_by_delta.items():
        for scope in ("all", "development", "test"):
            addresses = (
                indexed.index
                if scope == "all"
                else indexed.index[indexed["split"] == scope]
            )
            scoped = counts.loc[addresses]
            labels = indexed.loc[addresses, "label"].astype(int)
            labelled = motif_coverage(scoped, labels).reset_index()
            labelled["label"] = labelled["label"].astype(str)
            overall = motif_coverage(scoped).assign(label="all")
            frame = pd.concat([overall, labelled], ignore_index=True)
            frame["zero"] = frame["n"] - frame["nonzero"]
            frame["zero_fraction"] = 1.0 - frame["coverage"]
            frame = frame.assign(
                ordering_variant=variant,
                delta=int(delta),
                scope=scope,
            )
            records.append(frame)
    return pd.concat(records, ignore_index=True)


def _evaluate_variant(
    output_paths,
    protocol: dict,
    targets: pd.DataFrame,
    folds: pd.DataFrame,
    baseline: pd.DataFrame,
    counts_by_delta: dict[int, pd.DataFrame],
    variant: str,
    telemetry: RunTelemetry | None,
) -> dict:
    indexed = targets.set_index("address")
    labels = indexed["label"].astype(int)
    development = indexed.index[indexed["split"] == "development"]
    test = indexed.index[indexed["split"] == "test"]
    motif_features = {
        delta: log_motif_counts(counts) for delta, counts in counts_by_delta.items()
    }
    combined_features = {
        delta: baseline.join(features, how="left").fillna(0.0)
        for delta, features in motif_features.items()
    }
    cv_context = (
        telemetry.step(f"realcats_binary.cv.{variant}")
        if telemetry
        else nullcontext()
    )
    with cv_context:
        cv_scores = cv_delta_scores(
            {
                delta: frame.loc[development]
                for delta, frame in combined_features.items()
            },
            labels.loc[development],
            folds,
        )
    selected_delta, selection = select_delta_maximum(cv_scores)
    one_se_delta, one_se = select_delta_one_standard_error(cv_scores)
    selection = selection.merge(
        one_se[["delta", "within_one_se", "selected_one_se"]],
        on="delta",
        how="left",
        validate="one_to_one",
    )

    _, baseline_scores = fit_binary_scores(
        baseline.loc[development], labels.loc[development], baseline.loc[test]
    )
    scores_by_delta: dict[int, dict[str, object]] = {}
    metrics: list[dict] = []
    predictions: list[dict] = []
    for delta in protocol["deltas"]:
        _, motif_scores = fit_binary_scores(
            motif_features[delta].loc[development],
            labels.loc[development],
            motif_features[delta].loc[test],
        )
        _, combined_scores = fit_binary_scores(
            combined_features[delta].loc[development],
            labels.loc[development],
            combined_features[delta].loc[test],
        )
        scores_by_delta[int(delta)] = {
            "baseline": baseline_scores,
            "motifs": motif_scores,
            "combined": combined_scores,
        }
        for arm, scores in scores_by_delta[int(delta)].items():
            metrics.append(
                {
                    "ordering_variant": variant,
                    "delta": int(delta),
                    "arm": arm,
                    **binary_metrics(labels.loc[test], scores),
                }
            )
            predictions.extend(
                {
                    "address": str(address),
                    "label": int(label),
                    "campaign_group": str(
                        indexed.loc[address, "campaign_group"]
                    ),
                    "variant": variant,
                    "delta": int(delta),
                    "arm": arm,
                    "score": float(score),
                }
                for address, label, score in zip(
                    test,
                    labels.loc[test],
                    scores,
                    strict=True,
                )
            )

    groups = indexed.loc[test, "campaign_group"].astype(str)
    selected_scores = scores_by_delta[selected_delta]["combined"]
    randomization = grouped_score_swap_test(
        labels.loc[test],
        baseline_scores,
        selected_scores,
        groups,
        iterations=int(protocol["permutations"]),
        seed=int(protocol["random_seed"]),
    )
    bootstrap = stratified_group_bootstrap_delta(
        labels.loc[test],
        baseline_scores,
        selected_scores,
        groups,
        iterations=int(protocol["bootstraps"]),
        seed=int(protocol["random_seed"]),
    )
    alternatives = []
    for delta in protocol["deltas"]:
        if int(delta) == selected_delta:
            continue
        comparison = stratified_group_bootstrap_delta(
            labels.loc[test],
            scores_by_delta[int(delta)]["combined"],
            selected_scores,
            groups,
            iterations=int(protocol["bootstraps"]),
            seed=int(protocol["random_seed"]),
        )
        alternatives.append({"alternative_delta": int(delta), **comparison})
    adjusted = benjamini_hochberg(
        item["p_value_one_sided"] for item in alternatives
    )
    for item, adjusted_p in zip(alternatives, adjusted, strict=True):
        item["p_value_bh"] = float(adjusted_p)

    coverage = _coverage_rows(counts_by_delta, targets, variant)
    write_parquet_exclusive(
        output_paths.predictions / f"realcats_binary_{variant}.parquet",
        pd.DataFrame.from_records(predictions),
    )
    write_csv_exclusive(
        output_paths.metrics / f"realcats_binary_{variant}_metrics.csv",
        pd.DataFrame.from_records(metrics),
    )
    write_csv_exclusive(
        output_paths.metrics / f"realcats_binary_{variant}_cv.csv", cv_scores
    )
    write_csv_exclusive(
        output_paths.metrics / f"realcats_binary_{variant}_delta_selection.csv",
        selection,
    )
    write_csv_exclusive(
        output_paths.metrics / f"realcats_binary_{variant}_coverage.csv", coverage
    )
    result = {
        "ordering_variant": variant,
        "selected_delta": int(selected_delta),
        "one_se_delta": int(one_se_delta),
        "baseline_average_precision": float(
            average_precision_score(labels.loc[test], baseline_scores)
        ),
        "combined_average_precision": float(
            average_precision_score(labels.loc[test], selected_scores)
        ),
        "rq1_randomization": randomization,
        "rq1_group_bootstrap": bootstrap,
        "rq2_selected_vs_alternatives": alternatives,
    }
    write_json_exclusive(
        output_paths.metrics / f"realcats_binary_{variant}_results.json", result
    )
    return result


def execute_realcats_binary(
    paths,
    protocol: dict,
    stage_manager,
    checkpoints,
    scratch_directory: Path,
    telemetry: RunTelemetry | None = None,
) -> dict:
    """Execute the locked external binary replication and publish its tables."""
    dataset = protocol["datasets"]["realcats"]
    artifacts = dataset["artifacts"]
    targets = pd.read_parquet(paths.root / artifacts["binary_targets"])
    targets["address"] = targets["address"].astype(str)
    folds = pd.read_parquet(paths.root / artifacts["binary_folds"])
    target_index = pd.Index(targets["address"], name="address")
    layer_paths = {
        layer: paths.root / relative
        for layer, relative in artifacts["binary_layers"].items()
    }

    with stage_manager.stage("realcats-binary-baseline") as stage:
        if not stage.reused:
            baseline = _build_layered_baseline(
                layer_paths,
                target_index,
                scratch_directory / "realcats-binary-baseline",
            )
            _save_indexed(
                stage.paths.features / "realcats_binary_baseline.parquet", baseline
            )
    baseline_frame = pd.read_parquet(
        paths.features / "realcats_binary_baseline.parquet"
    )
    baseline_frame["address"] = baseline_frame["address"].astype(str)
    baseline = baseline_frame.set_index("address").reindex(target_index)

    results: list[dict] = []
    for variant in protocol["ordering_variants"]:
        combined = {
            int(delta): pd.DataFrame(index=target_index)
            for delta in protocol["deltas"]
        }
        for layer, layer_path in sorted(layer_paths.items()):
            row_count = int(pq.ParquetFile(layer_path).metadata.num_rows)
            checkpoint_ids = [
                f"realcats-binary-{layer}-{variant}-delta-{int(delta)}"
                for delta in protocol["deltas"]
            ]
            graph = None
            if row_count and not all(
                (checkpoints.root / checkpoint_id).exists()
                for checkpoint_id in checkpoint_ids
            ):
                graph = RaphtoryExperimentGraph.from_parquet(
                    layer_path,
                    variant,
                    scratch_directory / "realcats-binary" / variant / layer,
                    telemetry=telemetry,
                    source_label=f"realcats_binary_{layer}",
                )
            for delta in protocol["deltas"]:
                checkpoint_id = (
                    f"realcats-binary-{layer}-{variant}-delta-{int(delta)}"
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
                        "dataset": "realcats_binary",
                        "layer": layer,
                        "ordering_variant": variant,
                        "delta": int(delta),
                        "targets_sha256": artifacts["binary_targets_sha256"],
                        "events_sha256": artifacts["binary_layers_sha256"][layer],
                        "motif_roles_retained": 32,
                    },
                    compute,
                )
                combined[int(delta)] = combined[int(delta)].join(
                    counts.add_prefix(f"{layer}__"), how="left"
                )
            del graph
        counts_by_delta = {
            delta: frame.fillna(0).astype("int64")
            for delta, frame in combined.items()
        }
        stage_id = f"realcats-binary-eval-{variant}"
        with stage_manager.stage(stage_id, {"ordering_variant": variant}) as stage:
            if not stage.reused:
                result = _evaluate_variant(
                    stage.paths,
                    protocol,
                    targets,
                    folds,
                    baseline,
                    counts_by_delta,
                    variant,
                    telemetry,
                )
            else:
                result = json.loads(
                    (
                        paths.metrics / f"realcats_binary_{variant}_results.json"
                    ).read_text(encoding="utf-8")
                )
        results.append(result)

    with stage_manager.stage("realcats-binary-aggregation") as stage:
        if not stage.reused:
            rq1_rows = []
            rq2_rows = []
            for result in results:
                rq1_rows.append(
                    {
                        "ordering_variant": result["ordering_variant"],
                        "selected_delta": result["selected_delta"],
                        "one_se_delta": result["one_se_delta"],
                        "baseline_average_precision": result[
                            "baseline_average_precision"
                        ],
                        "combined_average_precision": result[
                            "combined_average_precision"
                        ],
                        "observed_delta_ap": result["rq1_randomization"][
                            "observed_delta_ap"
                        ],
                        "ci_lower": result["rq1_group_bootstrap"]["ci_95"][0],
                        "ci_upper": result["rq1_group_bootstrap"]["ci_95"][1],
                        "p_value": result["rq1_randomization"][
                            "p_value_one_sided"
                        ],
                    }
                )
                rq2_rows.extend(
                    {
                        "ordering_variant": result["ordering_variant"],
                        "selected_delta": result["selected_delta"],
                        "one_se_delta": result["one_se_delta"],
                        **comparison,
                    }
                    for comparison in result["rq2_selected_vs_alternatives"]
                )
            write_csv_exclusive(
                stage.paths.tables / "realcats_binary_rq1_results.csv",
                pd.DataFrame(rq1_rows),
            )
            write_csv_exclusive(
                stage.paths.tables / "realcats_binary_rq2_results.csv",
                pd.DataFrame(rq2_rows),
            )
            summary = {
                "status": "evaluated",
                "classification": "secondary_external_replication",
                "ordering_variants": [item["ordering_variant"] for item in results],
                "selected_deltas": [item["selected_delta"] for item in results],
                "positive_effect_all_orderings": all(
                    item["rq1_randomization"]["observed_delta_ap"] > 0
                    for item in results
                ),
                "reject_h0_all_orderings": all(
                    item["rq1_randomization"]["p_value_one_sided"]
                    < float(protocol["alpha"])
                    for item in results
                ),
            }
            write_json_exclusive(
                stage.paths.metrics / "realcats_binary_ordering_sensitivity.json",
                summary,
            )
        else:
            summary = json.loads(
                (
                    paths.metrics / "realcats_binary_ordering_sensitivity.json"
                ).read_text(encoding="utf-8")
            )
    return summary
