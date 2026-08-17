"""Scientific and schema guards for prepared and generated experiment artifacts."""

from __future__ import annotations

import json
from pathlib import Path


def _finite_numeric(frame, context: str) -> None:
    import numpy as np

    numeric = frame.select_dtypes(include=["number"])
    if numeric.size and not np.isfinite(numeric.to_numpy(dtype=float)).all():
        raise RuntimeError(f"{context} contains non-finite numerical values")


def validate_prepared_inputs(run_root: Path, protocol: dict) -> dict:
    import pandas as pd

    from src.experiments.constants import (
        HF_CONFIRMATION_EXCLUSION_POLICY,
        HF_RELEASE_IDENTITY,
    )

    audit: dict = {}
    enforce_class_coverage = protocol.get("execution_profile") != "smoke"
    for name, dataset in protocol.get("datasets", {}).items():
        artifacts = dataset.get("artifacts", {})
        target_relative = artifacts.get("targets")
        if not target_relative:
            continue
        targets = pd.read_parquet(run_root / target_relative)
        address_column = "address" if "address" in targets else "node_id"
        if targets[address_column].isna().any():
            raise RuntimeError(f"{name} targets contain null addresses")
        addresses = targets[address_column].astype(str)
        if addresses.duplicated().any():
            raise RuntimeError(f"{name} targets are not unique and non-null")
        label_column = next(
            (
                column
                for column in ("label", "type_label", "is_scam")
                if column in targets
            ),
            None,
        )
        if "split" in targets:
            assigned = targets["split"].dropna()
            if not set(assigned.unique()) <= {"development", "test"}:
                raise RuntimeError(f"{name} contains invalid split labels")
            if enforce_class_coverage and label_column is not None and not assigned.empty:
                expected_classes = set(targets.loc[targets["split"].notna(), label_column])
                for split_name in ("development", "test"):
                    observed = set(
                        targets.loc[targets["split"] == split_name, label_column]
                    )
                    if observed != expected_classes:
                        raise RuntimeError(
                            f"{name} {split_name} split is missing required classes"
                        )
            if "campaign_group" in targets:
                groups = targets.loc[targets["split"].notna()].groupby("campaign_group")["split"].nunique()
                if (groups > 1).any():
                    raise RuntimeError(f"{name} campaign groups straddle development/test")
        fold_relative = artifacts.get("folds")
        if fold_relative:
            folds = pd.read_parquet(run_root / fold_relative)
            fold_keys = [column for column in ("address", "repeat") if column in folds]
            if folds.assign(address=folds["address"].astype(str)).duplicated(fold_keys).any():
                raise RuntimeError(f"{name} fold assignments contain duplicates")
            if not set(folds["address"].astype(str)) <= set(addresses):
                raise RuntimeError(f"{name} folds contain non-target addresses")
            target_metadata = [address_column]
            target_metadata.extend(
                column
                for column in ("campaign_group", label_column)
                if column is not None and column not in target_metadata
            )
            fold_audit = folds.assign(address=folds["address"].astype(str)).merge(
                targets[target_metadata].assign(
                    **{address_column: targets[address_column].astype(str)}
                ).rename(columns={address_column: "address"}),
                on="address",
                how="left",
                validate="many_to_one",
            )
            if "repeat" not in fold_audit:
                fold_audit["repeat"] = 0
            if "campaign_group" in targets:
                crossing = fold_audit.groupby(
                    ["campaign_group", "repeat"], dropna=False
                )["fold"].nunique()
                if (crossing > 1).any():
                    raise RuntimeError(f"{name} campaign groups straddle CV folds")
            if enforce_class_coverage and label_column is not None:
                scopes = [column for column in ("cohort", "repeat") if column in fold_audit]
                for scope_values, scoped in fold_audit.groupby(scopes, dropna=False):
                    expected_classes = set(scoped[label_column])
                    for fold_number, held_out in scoped.groupby("fold", dropna=False):
                        if set(held_out[label_column]) != expected_classes:
                            raise RuntimeError(
                                f"{name} CV fold {fold_number} in scope {scope_values} "
                                "is missing required classes"
                            )
        audit[name] = {
            "targets": int(len(targets)),
            "label_column": label_column,
        }
        if name == "huggingface" and artifacts.get("motif_events"):
            extraction = dataset.get("edge_extraction", {}).get(
                "released_neighborhood", {}
            )
            if extraction.get("scope") != "released_graph_target_neighborhood_closure":
                raise RuntimeError("HF triangle graph has an invalid extraction scope")
            if extraction.get("full_ethereum_completeness") is not False:
                raise RuntimeError(
                    "HF triangle graph must not claim full-Ethereum completeness"
                )
            incident_edges = int(extraction.get("incident_edges", -1))
            selected_edges = int(extraction.get("selected_edges", -1))
            if incident_edges <= 0 or selected_edges < incident_edges:
                raise RuntimeError("HF triangle graph lost target-incident edges")
            audit[name]["motif_graph_edges"] = selected_edges
            audit[name]["triangle_scope"] = extraction["scope"]
            if int(protocol.get("protocol_version", 0)) >= 8 and protocol.get(
                "execution_profile"
            ) != "smoke":
                release = dataset.get("release_identity", {})
                if not release.get("complete_pinned_release_match", False):
                    raise RuntimeError("HF input is not the complete pinned release")
                if release.get("observed_sha256") != HF_RELEASE_IDENTITY["sha256"]:
                    raise RuntimeError("HF release hashes differ from the locked identity")
                exclusions = dataset.get("confirmation_exclusions", {})
                if int(exclusions.get("unique_prior_node_ids", 0)) < int(
                    HF_CONFIRMATION_EXCLUSION_POLICY[
                        "minimum_unique_prior_targets"
                    ]
                ):
                    raise RuntimeError("HF prior-target exclusion lineage is incomplete")
        if name == "realcats" and int(protocol.get("protocol_version", 0)) >= 8:
            binary_targets = pd.read_parquet(run_root / artifacts["binary_targets"])
            binary_folds = pd.read_parquet(run_root / artifacts["binary_folds"])
            hf_exclusions = pd.read_parquet(
                run_root / artifacts["binary_hf_exclusions"]
            )
            binary_targets["address"] = binary_targets["address"].astype(str)
            if binary_targets["address"].duplicated().any():
                raise RuntimeError("Real-CATS binary targets are not unique")
            if set(binary_targets["label"].astype(int)) != {0, 1}:
                raise RuntimeError("Real-CATS binary targets do not contain both classes")
            counts = binary_targets["label"].astype(int).value_counts()
            if counts.nunique() != 1:
                raise RuntimeError("Real-CATS binary targets are not activity matched 1:1")
            if binary_targets["usable_events"].lt(3).any():
                raise RuntimeError("Real-CATS binary target has fewer than three events")
            overlap = set(binary_targets["address"]) & set(
                hf_exclusions["address"].astype(str)
            )
            if overlap:
                raise RuntimeError("Real-CATS binary targets overlap HF labels")
            if set(binary_targets["split"]) != {"development", "test"}:
                raise RuntimeError("Real-CATS binary split is incomplete")
            for split, scoped in binary_targets.groupby("split"):
                if set(scoped["label"].astype(int)) != {0, 1}:
                    raise RuntimeError(
                        f"Real-CATS binary {split} split is missing a class"
                    )
            if not set(binary_folds["address"].astype(str)) <= set(
                binary_targets.loc[
                    binary_targets["split"] == "development", "address"
                ]
            ):
                raise RuntimeError("Real-CATS binary folds contain non-development targets")
            audit[name]["binary_targets"] = int(len(binary_targets))
            audit[name]["binary_hf_overlap"] = 0
    return audit


def validate_generated_results(run_root: Path, protocol: dict) -> dict:
    import numpy as np
    import pandas as pd

    from src.experiments.constants import (
        ALL_MOTIF_ROLE_NAMES,
        MOTIF_ROLE_NAMES,
    )

    audit: dict = {"files_checked": 0}
    for path in sorted((run_root / "features").glob("*.parquet")):
        frame = pd.read_parquet(path)
        _finite_numeric(frame, str(path))
        motif_columns = [
            column
            for column in frame
            if "star_" in column
            or "two_node_" in column
            or "triangle_" in column
        ]
        if motif_columns and (frame[motif_columns] < 0).any().any():
            raise RuntimeError(f"{path} contains negative motif counts")
        if "address" in frame and "delta" not in frame and frame["address"].duplicated().any():
            raise RuntimeError(f"{path} must contain one row per target address")
        if {"address", "delta"} <= set(frame) and frame.duplicated(["address", "delta"]).any():
            raise RuntimeError(f"{path} must contain one row per address and delta")
        audit["files_checked"] += 1

    for path in sorted((run_root / "predictions").glob("*.parquet")):
        frame = pd.read_parquet(path)
        _finite_numeric(frame, str(path))
        score = pd.to_numeric(frame["score"], errors="raise")
        if ((score < 0) | (score > 1)).any():
            raise RuntimeError(f"{path} contains probabilities outside [0, 1]")
        if "score_label" in frame:
            keys = [column for column in ("address", "variant", "global_delta") if column in frame]
            sums = frame.groupby(keys, dropna=False)["score"].sum()
            if not np.allclose(sums.to_numpy(), 1.0, atol=1e-8, rtol=1e-8):
                raise RuntimeError(f"{path} multiclass probability rows do not sum to one")
            if frame.duplicated([*keys, "score_label"]).any():
                raise RuntimeError(f"{path} contains duplicate class probability rows")
        else:
            keys = [column for column in ("address", "cohort", "variant", "delta", "arm") if column in frame]
            if frame.duplicated(keys).any():
                raise RuntimeError(f"{path} contains duplicate binary prediction rows")
        audit["files_checked"] += 1

    for path in sorted((run_root / "metrics").glob("*.csv")):
        frame = pd.read_csv(path)
        _finite_numeric(frame, str(path))
        audit["files_checked"] += 1
    for path in sorted((run_root / "tables").glob("*.csv")):
        frame = pd.read_csv(path)
        _finite_numeric(frame, str(path))
        audit["files_checked"] += 1
    for path in sorted((run_root / "metrics").glob("*.json")):
        import json

        if path.name == "execution_validation.json":
            continue
        json.loads(path.read_text(encoding="utf-8"))
        audit["files_checked"] += 1

    selected = set(protocol.get("research_questions", []))
    deltas = set(int(delta) for delta in protocol.get("deltas", []))
    if "rq1" in selected:
        representation = pd.read_csv(
            run_root / "tables" / "rq1_representation_tests.csv"
        )
        required_representation = {
            "cohort",
            "ordering_variant",
            "selected_delta",
            "comparison",
            "reference_arm",
            "alternative_arm",
            "observed_delta_ap",
            "ci_lower",
            "ci_upper",
            "p_value_bh",
            "reject_h0_bh",
        }
        if not required_representation <= set(representation):
            raise RuntimeError("RQ1 representation table is missing required columns")
        representation_definition = protocol["hf_representation_analysis"]
        expected_comparisons = {
            str(item["comparison"])
            for item in representation_definition["formal_comparisons"]
        }
        expected_cohorts = set(
            pd.read_parquet(
                run_root
                / protocol["datasets"]["huggingface"]["artifacts"]["targets"],
                columns=["cohort"],
            )["cohort"].astype(str)
        )
        if set(representation["cohort"].astype(str)) != expected_cohorts:
            raise RuntimeError("RQ1 representation table is missing a locked cohort")
        for cohort, rows in representation.groupby("cohort"):
            if set(rows["comparison"].astype(str)) != expected_comparisons:
                raise RuntimeError(
                    f"RQ1 representation comparisons are incomplete for {cohort}"
                )
            if set(rows["ordering_variant"].astype(str)) != {"canonical"}:
                raise RuntimeError("RQ1 formal representation tests must be canonical only")

        if int(protocol.get("protocol_version", 0)) >= 7:
            triangle_ablation = pd.read_csv(
                run_root / "tables" / "rq1_triangle_ablation.csv"
            )
            expected_triangle_comparisons = {
                "baseline_triangles_vs_baseline",
                "combined_vs_combined_without_triangles",
            }
            if set(triangle_ablation["comparison"].astype(str)) != (
                expected_triangle_comparisons
            ):
                raise RuntimeError("RQ1 triangle ablation comparisons are incomplete")
            if set(triangle_ablation["cohort"].astype(str)) != expected_cohorts:
                raise RuntimeError("RQ1 triangle ablation is missing a locked cohort")
        if int(protocol.get("protocol_version", 0)) >= 8:
            triangle_windows = pd.read_csv(
                run_root / "tables" / "rq1_triangle_window_analysis.csv"
            )
            required_triangle_windows = {
                "cohort",
                "ordering_variant",
                "all_40_selected_delta",
                "non_triangle_32_selected_delta",
                "triangles_only_selected_delta",
                "observed_delta_ap",
            }
            if not required_triangle_windows <= set(triangle_windows):
                raise RuntimeError("RQ1 triangle-window table is incomplete")
            if set(triangle_windows["cohort"].astype(str)) != expected_cohorts:
                raise RuntimeError("RQ1 triangle-window table is missing a cohort")
            if set(triangle_windows["ordering_variant"].astype(str)) != {
                "canonical"
            }:
                raise RuntimeError("RQ1 triangle-window inference must be canonical only")

        roles = pd.read_csv(run_root / "tables" / "rq1_role_diagnostics.csv")
        role_required = {
            "cohort",
            "variant",
            "scope",
            "delta",
            "role",
            "average_precision_direct_count",
            "roc_auc_direct_count",
            "nonzero_coverage",
        }
        if not role_required <= set(roles):
            raise RuntimeError("RQ1 role diagnostics are missing required columns")
        expected_roles = (
            ALL_MOTIF_ROLE_NAMES
            if int(
                protocol.get("event_execution", {}).get(
                    "hf_motif_roles_retained",
                    protocol.get("event_execution", {}).get(
                        "motif_roles_retained", len(MOTIF_ROLE_NAMES)
                    ),
                )
            )
            == len(ALL_MOTIF_ROLE_NAMES)
            else MOTIF_ROLE_NAMES
        )
        for (cohort, scope), rows in roles.groupby(["cohort", "scope"]):
            if set(rows["role"].astype(str)) != set(expected_roles):
                raise RuntimeError(
                    f"RQ1 role diagnostics are incomplete for {cohort}/{scope}"
                )
        if set(roles["scope"].astype(str)) != {"development", "test"}:
            raise RuntimeError("RQ1 role diagnostics must separate development and test")

        xgboost = pd.read_csv(
            run_root / "tables" / "rq1_xgboost_robustness.csv"
        )
        if set(xgboost["cohort"].astype(str)) != expected_cohorts or set(
            xgboost["ordering_variant"].astype(str)
        ) != {"canonical"}:
            raise RuntimeError("RQ1 XGBoost robustness rows are incomplete")

        expected_selected_arms = {
            "baseline",
            "motifs",
            "combined",
            *representation_definition["diagnostic_arms"],
        }
        expected_xgboost_arms = {
            "baseline_xgboost",
            "motifs_xgboost",
            "combined_xgboost",
        }
        for cohort in expected_cohorts:
            for variant in protocol["ordering_variants"]:
                stem = f"hf_{cohort}_{variant}"
                metrics = pd.read_csv(run_root / "metrics" / f"{stem}_metrics.csv")
                statistical = json.loads(
                    (run_root / "metrics" / f"{stem}_statistical_tests.json").read_text(
                        encoding="utf-8"
                    )
                )
                selected_delta = int(statistical["selected_delta"])
                at_selected = set(
                    metrics.loc[metrics["delta"].astype(int) == selected_delta, "arm"].astype(str)
                )
                required_arms = set(expected_selected_arms)
                if variant == "canonical":
                    required_arms.update(expected_xgboost_arms)
                if not required_arms <= at_selected:
                    missing = sorted(required_arms - at_selected)
                    raise RuntimeError(
                        f"{stem} is missing selected-window arms: {missing}"
                    )
                primary = metrics.loc[
                    metrics["arm"].isin(["baseline", "motifs", "combined"])
                ]
                for arm, rows in primary.groupby("arm"):
                    if set(rows["delta"].astype(int)) != deltas:
                        raise RuntimeError(
                            f"{stem} primary arm {arm} is missing a locked delta"
                        )
                if int(protocol.get("protocol_version", 0)) >= 8:
                    for arm in (
                        "motifs_without_triangles",
                        "combined_without_triangles",
                        "triangles_only",
                        "baseline_triangles",
                    ):
                        rows = metrics.loc[metrics["arm"] == arm]
                        if set(rows["delta"].astype(int)) != deltas:
                            raise RuntimeError(
                                f"{stem} v8 arm {arm} is missing a locked delta"
                            )
    if "rq2" in selected:
        rq2 = pd.read_csv(run_root / "tables" / "rq2_delta_comparisons.csv")
        required = {
            "selected_delta",
            "one_se_delta",
            "alternative_delta",
            "observed_delta_ap",
            "p_value_bh",
        }
        if not required <= set(rq2):
            raise RuntimeError("RQ2 comparison table is missing locked protocol columns")
        observed_deltas = set(
            pd.concat(
                [
                    rq2["selected_delta"],
                    rq2["one_se_delta"],
                    rq2["alternative_delta"],
                ]
            ).astype(int)
        )
        if not observed_deltas <= deltas:
            raise RuntimeError("RQ2 comparison table contains an unlocked delta")
        coverage = pd.read_csv(run_root / "tables" / "rq2_motif_coverage.csv")
        coverage_required = {
            "cohort",
            "variant",
            "scope",
            "label",
            "delta",
            "n",
            "nonzero",
            "zero",
            "coverage",
            "zero_fraction",
        }
        if not coverage_required <= set(coverage):
            raise RuntimeError("RQ2 coverage table is missing required columns")
        if set(coverage["delta"].astype(int)) != deltas:
            raise RuntimeError("RQ2 coverage table is missing a locked delta")
        if set(coverage["scope"].astype(str)) != {"all", "development", "test"}:
            raise RuntimeError("RQ2 coverage table is missing a required population scope")
        if int(protocol.get("protocol_version", 0)) >= 8:
            if "representation" not in coverage or set(
                coverage["representation"].astype(str)
            ) != {"all_40_roles", "non_triangle_32_roles", "triangles_only"}:
                raise RuntimeError("RQ2 coverage table is missing a v8 representation")
            representation_deltas = pd.read_csv(
                run_root / "tables" / "rq2_representation_delta_selection.csv"
            )
            if set(representation_deltas["representation"].astype(str)) != {
                "all_40_roles",
                "non_triangle_32_roles",
                "triangles_only",
            }:
                raise RuntimeError("RQ2 representation window selections are incomplete")
    if {"rq3", "rq4"} <= selected:
        rq4_global = pd.read_csv(
            run_root / "tables" / "rq4_global_results.csv"
        )
        rq4_categories = pd.read_csv(
            run_root / "tables" / "rq4_category_results.csv"
        )
        global_required = {
            "ordering_variant",
            "global_delta",
            "observed_delta_macro_ap",
            "ci_lower",
            "ci_upper",
            "p_value",
            "reject_h0",
        }
        category_required = {
            "ordering_variant",
            "label",
            "global_delta",
            "category_delta",
            "observed_delta_ap",
            "p_value_bh",
            "reject_h0",
        }
        if not global_required <= set(rq4_global):
            raise RuntimeError("RQ4 global table is missing required columns")
        if not category_required <= set(rq4_categories):
            raise RuntimeError("RQ4 category table is missing required columns")
        result_deltas = set(
            pd.concat(
                [
                    rq4_global["global_delta"],
                    rq4_categories["global_delta"],
                    rq4_categories["category_delta"],
                ]
            ).astype(int)
        )
        if not result_deltas <= deltas:
            raise RuntimeError("RQ4 tables contain an unlocked delta")
        if int(protocol.get("protocol_version", 0)) >= 8:
            binary_rq1 = pd.read_csv(
                run_root / "tables" / "realcats_binary_rq1_results.csv"
            )
            binary_rq2 = pd.read_csv(
                run_root / "tables" / "realcats_binary_rq2_results.csv"
            )
            expected_variants = set(protocol["ordering_variants"])
            if set(binary_rq1["ordering_variant"].astype(str)) != expected_variants:
                raise RuntimeError("Real-CATS binary RQ1 variants are incomplete")
            if set(binary_rq2["ordering_variant"].astype(str)) != expected_variants:
                raise RuntimeError("Real-CATS binary RQ2 variants are incomplete")
            if not set(binary_rq1["selected_delta"].astype(int)) <= deltas:
                raise RuntimeError("Real-CATS binary RQ1 selected an unlocked delta")
            for variant in expected_variants:
                metrics = pd.read_csv(
                    run_root
                    / "metrics"
                    / f"realcats_binary_{variant}_metrics.csv"
                )
                for arm in ("baseline", "motifs", "combined"):
                    rows = metrics.loc[metrics["arm"] == arm]
                    if set(rows["delta"].astype(int)) != deltas:
                        raise RuntimeError(
                            f"Real-CATS binary {variant}/{arm} misses a delta"
                        )
    return audit
