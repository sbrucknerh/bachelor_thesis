"""Scientific and schema guards for prepared and generated experiment artifacts."""

from __future__ import annotations

from pathlib import Path


def _finite_numeric(frame, context: str) -> None:
    import numpy as np

    numeric = frame.select_dtypes(include=["number"])
    if numeric.size and not np.isfinite(numeric.to_numpy(dtype=float)).all():
        raise RuntimeError(f"{context} contains non-finite numerical values")


def validate_prepared_inputs(run_root: Path, protocol: dict) -> dict:
    import pandas as pd

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
    return audit


def validate_generated_results(run_root: Path, protocol: dict) -> dict:
    import numpy as np
    import pandas as pd

    audit: dict = {"files_checked": 0}
    for path in sorted((run_root / "features").glob("*.parquet")):
        frame = pd.read_parquet(path)
        _finite_numeric(frame, str(path))
        motif_columns = [column for column in frame if "star_" in column or "two_node_" in column]
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
            raise RuntimeError("RQ2 comparison table is missing protocol-v5 columns")
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
    return audit
