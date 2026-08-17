"""Locked experiment constants and the verified Raphtory index catalogue."""

from __future__ import annotations


PROTOCOL_VERSION = 8
MIN_FULLY_VERIFIABLE_PROTOCOL_VERSION = 6
SUPPORTED_FULL_PROTOCOL_VERSIONS = (6, 7, PROTOCOL_VERSION)
RANDOM_SEED = 42
CV_SEEDS = (42, 43, 44)
DELTAS = (
    300,
    1_800,
    3_600,
    21_600,
    86_400,
    259_200,
    604_800,
    1_209_600,
    2_592_000,
)
N_PERMUTATIONS = 10_000
N_BOOTSTRAPS = 10_000
HF_RESEARCH_QUESTIONS = frozenset({"rq1", "rq2"})
REALCATS_RESEARCH_QUESTIONS = frozenset({"rq3", "rq4"})
DELTA_MEAN_TIE_TOLERANCE = 1e-12
DELTA_SELECTION = {
    "rq2_primary": "development_cv_mean_combined_average_precision_argmax",
    "rq2_tie_break": "smallest_delta_within_absolute_tolerance",
    "rq2_tie_tolerance": DELTA_MEAN_TIE_TOLERANCE,
    "rq2_secondary": "shortest_delta_within_one_standard_error",
    "rq4_global": "development_cv_macro_average_precision_argmax",
    "rq4_per_category": "development_cv_one_vs_rest_average_precision_argmax",
    "rq4_test": "category_specific_vs_global_macro_average_precision_paired_bootstrap",
}
REALCATS_CATEGORY_ELIGIBILITY = {
    "label_source": "CE.tsv exact label",
    "minimum_active_addresses": 100,
    "exclude_casefold_labels": ["other"],
    "merge_labels": False,
    "coordination_ground_truth_available": False,
}

STAR_CLASSES = ("pre", "mid", "post")
DIRECTION_CODES = ("iii", "iio", "ioi", "ioo", "oii", "oio", "ooi", "ooo")
STAR_ROLE_NAMES = tuple(
    f"star_{star_class}_{directions}"
    for star_class in STAR_CLASSES
    for directions in DIRECTION_CODES
)
TWO_NODE_ROLE_NAMES = tuple(f"two_node_{directions}" for directions in DIRECTION_CODES)
TRIANGLE_ROLE_NAMES = tuple(f"triangle_{index + 1}" for index in range(8))
ALL_MOTIF_ROLE_NAMES = STAR_ROLE_NAMES + TWO_NODE_ROLE_NAMES + TRIANGLE_ROLE_NAMES
MOTIF_ROLE_NAMES = STAR_ROLE_NAMES + TWO_NODE_ROLE_NAMES
INCOMING_STAR_ROLE_NAMES = tuple(
    f"star_{star_class}_iii" for star_class in STAR_CLASSES
)

# Raphtory emits triangle roles in this verified order. Keeping the established
# column names preserves old artifacts; these definitions make each index
# interpretable without renaming protocol-v7 results.
TRIANGLE_ROLE_DEFINITIONS = {
    "triangle_1": ["i_to_j", "k_to_j", "i_to_k"],
    "triangle_2": ["i_to_j", "k_to_j", "k_to_i"],
    "triangle_3": ["i_to_j", "j_to_k", "i_to_k"],
    "triangle_4": ["i_to_j", "j_to_k", "k_to_i"],
    "triangle_5": ["i_to_j", "k_to_i", "j_to_k"],
    "triangle_6": ["i_to_j", "k_to_i", "k_to_j"],
    "triangle_7": ["i_to_j", "i_to_k", "j_to_k"],
    "triangle_8": ["i_to_j", "i_to_k", "k_to_j"],
}

HF_DIAGNOSTIC_ARM_NAMES = (
    "motif_presence",
    "motif_volume",
    "motif_composition",
    "motif_composition_volume",
    "motif_incoming_stars",
    "baseline_presence",
    "baseline_volume",
    "baseline_composition",
    "baseline_composition_volume",
    "baseline_incoming_stars",
    "triangles_only",
    "baseline_triangles",
    "motifs_without_triangles",
    "combined_without_triangles",
)

HF_DIAGNOSTIC_COMPARISONS = (
    ("motifs_vs_volume", "motif_volume", "motifs"),
    (
        "composition_volume_vs_volume",
        "motif_volume",
        "motif_composition_volume",
    ),
    (
        "motifs_vs_composition_volume",
        "motif_composition_volume",
        "motifs",
    ),
    ("baseline_volume_vs_baseline", "baseline", "baseline_volume"),
    (
        "baseline_composition_volume_vs_baseline_volume",
        "baseline_volume",
        "baseline_composition_volume",
    ),
    (
        "combined_vs_baseline_composition_volume",
        "baseline_composition_volume",
        "combined",
    ),
    ("baseline_incoming_vs_baseline", "baseline", "baseline_incoming_stars"),
    ("combined_vs_baseline_incoming", "baseline_incoming_stars", "combined"),
    ("baseline_triangles_vs_baseline", "baseline", "baseline_triangles"),
    (
        "combined_vs_combined_without_triangles",
        "combined_without_triangles",
        "combined",
    ),
)

HF_REPRESENTATION_ANALYSIS = {
    "status": "prespecified_protocol_v8_confirmation_extension",
    "evaluation_delta": "rq2_primary_development_cv_selected_delta",
    "formal_ordering_variant": "canonical",
    "multiple_testing": "benjamini_hochberg_across_ten_representation_comparisons_per_cohort",
    "feature_definitions": {
        "motif_presence": "indicator(sum(raw_40_role_counts)>0)",
        "motif_volume": "log1p(sum(raw_40_role_counts))",
        "motif_composition": "sqrt(raw_role_count/sum(raw_40_role_counts)); zero vector retained",
        "motif_composition_volume": "motif_composition joined with motif_volume",
        "motifs": "log1p(raw_40_role_counts)",
        "motif_incoming_stars": (
            "log1p(raw counts for the three prespecified all-incoming star roles)"
        ),
        "triangles_only": "log1p(raw counts for the eight triangle roles)",
        "baseline_triangles": "locked aggregate/static baseline joined with triangles_only",
        "motifs_without_triangles": "log1p(raw 24 star and 8 two-node role counts)",
        "combined_without_triangles": (
            "locked aggregate/static baseline joined with motifs_without_triangles"
        ),
        "baseline_*": "locked aggregate/static baseline joined with the named motif representation",
        "combined": "locked aggregate/static baseline joined with motifs",
    },
    "incoming_star_roles": list(INCOMING_STAR_ROLE_NAMES),
    "diagnostic_arms": list(HF_DIAGNOSTIC_ARM_NAMES),
    "formal_comparisons": [
        {
            "comparison": comparison,
            "reference_arm": reference,
            "alternative_arm": alternative,
            "alternative": "alternative_arm_average_precision_greater_than_reference_arm",
        }
        for comparison, reference, alternative in HF_DIAGNOSTIC_COMPARISONS
    ],
    "xgboost_robustness_arms": [
        "baseline_xgboost",
        "motifs_xgboost",
        "combined_xgboost",
    ],
}

HF_TRIANGLE_ANALYSIS = {
    "enabled": True,
    "scope": "released_graph_target_neighborhood_closure",
    "extraction": (
        "retain every released edge whose endpoints both occur in the union of "
        "endpoints of selected-target incident edges"
    ),
    "release_relative_completeness": True,
    "full_ethereum_completeness": False,
    "upstream_release_filters_apply": True,
    "upstream_release_constraints": [
        "normal_transactions_only",
        "labelled_seed_activity_windows",
        "value_wei_raw_at_least_1e14",
        "released_neighbor_and_pair_caps",
    ],
    "roles_retained": 40,
    "triangle_roles_retained": 8,
    "primary_features": "all_40_roles",
    "incremental_test": "combined_40_roles_vs_combined_32_roles",
    "role_definitions": TRIANGLE_ROLE_DEFINITIONS,
}

HF_TRIANGLE_WINDOW_ANALYSIS = {
    "enabled": True,
    "fold_contract": "same_addresses_labels_splits_and_repeated_cv_folds",
    "representations": {
        "all_40_roles": "baseline joined with log1p(all 40 role counts)",
        "non_triangle_32_roles": (
            "baseline joined with log1p(24 star and 8 two-node role counts)"
        ),
        "triangles_only": "log1p(the eight triangle role counts)",
    },
    "selection": "independent development_cv_mean_average_precision_argmax",
    "test_comparison": (
        "all_40_roles_at_its_selected_delta_vs_non_triangle_32_roles_at_its_"
        "selected_delta"
    ),
    "coverage": "all_40_roles_non_triangle_32_roles_and_triangles_only_by_delta",
    "classification": "secondary_prespecified_protocol_v8_analysis",
}

HF_RELEASE_IDENTITY = {
    "repository": "fesevu/ethereum_fraud_dataset_by_activity",
    "revision": "6a7a517fbdbe2bcc4ec65bee4c0ff15f1c0a3c70",
    "edges_rows": 380_566_661,
    "targets_rows": 114_739,
    "sha256": {
        "edges": "bbe483a60b6587982e6d6c5013ca6e5c4bd786d54653911308d3c5bc597c99a6",
        "labels": "6a50328d1d7e76da7d3dccc41ed3193863f7c4785f17bf65f05a46901bf7cc9b",
        "descriptions": "52941005bd44d2597037e2e1c932e4d25fcf9c855980f442896cb497d1827b39",
    },
    "unique_graph_representation": "gnn_dataset/edges_all/edges.parquet",
    "alternate_repository_copies": [
        "raw string-address GNN parquet",
        "weekly partitions",
        "monthly partitions",
    ],
    "alternate_copies_add_unique_released_edges": False,
    "upstream_sampling_cannot_be_reversed_from_release": True,
}

HF_CONFIRMATION_EXCLUSION_POLICY = {
    "minimum_reference_runs": 3,
    "minimum_unique_prior_targets": 27_600,
    "minimum_replication_index": 3,
    "scope": "thesis_profile_protocol_v8_hf",
}

ACTIVITY_MATCHING_BINS = (3, 5, 10, 25, 100)

REALCATS_BINARY_REPLICATION = {
    "enabled": True,
    "purpose": "secondary_external_rq1_rq2_replication",
    "labels": "CE.tsv criminal versus BE.tsv benign",
    "hf_overlap_policy": "exclude_every_address_in_pinned_hf_targets_global",
    "minimum_usable_events": 3,
    "activity_source": "transaction_number_used_for_matching_only_not_as_feature",
    "activity_bins_left_closed": list(ACTIVITY_MATCHING_BINS) + ["infinity"],
    "matching": "one_to_one_within_activity_bin_without_replacement",
    "layers": ["normal", "internal", "erc20", "erc721", "erc1155"],
    "motif_roles_per_layer": 32,
    "ordering_variants": "same_as_primary_protocol",
    "campaign_grouping": "singleton_addresses_dataset_has_no_campaign_identifier",
}

REPLICATION_SEED_STRIDE = 10_000
MAX_REPLICATION_INDEX = 100_000


def replication_seed_definition(index: int) -> dict:
    """Return deterministic sampling/split seeds without changing model semantics."""
    value = int(index)
    if value < 0 or value > MAX_REPLICATION_INDEX:
        raise ValueError(
            f"replication_index must be between 0 and {MAX_REPLICATION_INDEX}"
        )
    offset = value * REPLICATION_SEED_STRIDE
    return {
        "index": value,
        "cohort_seed": RANDOM_SEED + offset,
        "split_seed": RANDOM_SEED + offset,
        "cv_seeds": [seed + offset for seed in CV_SEEDS],
        "model_and_resampling_seed": RANDOM_SEED,
        "seed_stride": REPLICATION_SEED_STRIDE,
    }

EVENT_EXECUTION = {
    "external_sort": "duckdb",
    "graph_loader": "raphtory_load_edges_arrow_batches",
    "large_event_pandas_materialization": False,
    "raphtory_roles_computed": 40,
    "hf_motif_roles_retained": 40,
    "hf_triangle_roles_retained": 8,
    "realcats_motif_roles_retained_per_layer": 32,
    "realcats_triangle_roles_retained_per_layer": 0,
}

HF_COHORT_ELIGIBILITY = {
    "minimum_incident_edges": 3,
    "activity_upper_quantile": 0.995,
    "strata": ["label", "is_contract", "activity_quintile"],
}

REALCATS_LAYER_FILES = {
    "normal": "normal_transactions.json",
    "internal": "internal_transactions.json",
    "erc20": "ERC_20_transactions.json",
    "erc721": "ERC_721_transactions.json",
    "erc1155": "ERC_1155_transactions.json",
}

KNOWN_CAMPAIGN_PREFIXES = {
    "multichain_hack": "multichain hack alert",
    "upbit_hack": "upbit hacker",
    "sorbet_finance_hack": "sorbet finance hack alert",
}

DEFAULT_RF_PARAMS = {
    "n_estimators": 100,
    "max_depth": 10,
    "class_weight": "balanced",
    "random_state": RANDOM_SEED,
}

DEFAULT_XGB_PARAMS = {
    "n_estimators": 100,
    "max_depth": 6,
    "learning_rate": 0.1,
    "random_state": RANDOM_SEED,
    "eval_metric": "logloss",
}

MULTICLASS_LOGISTIC_PARAMS = {
    "class_weight": "balanced",
    "max_iter": 2_000,
    "random_state": RANDOM_SEED,
}

# Protocol-v8 resource-accounting constants. These are conservative admission
# assumptions, not operating-system-enforced limits for Raphtory.
GRAPH_COLUMN_EXPANSION_FACTOR = 4.0
MOTIF_RESULT_BYTES_PER_NODE = 512
DUCKDB_MEMORY_FRACTION = 0.15
DUCKDB_MEMORY_MIN_GIB = 1
DUCKDB_MEMORY_MAX_GIB = 16
RAM_ADMISSION_FRACTION = 0.85
DISK_SORT_FACTOR = 2.5
DISK_RESERVE_FRACTION = 0.20
DISK_RESERVE_MIN_GIB = 20
SMOKE_DISK_RESERVE_GIB = 2
RESULT_STAGING_ALLOWANCE_GIB = 2
CHECKPOINT_STAGING_MIN_MIB = 256
COORDINATOR_ALLOWANCE_GIB = 1

RESOURCE_CONSTANTS = {
    "graph_column_expansion_factor": GRAPH_COLUMN_EXPANSION_FACTOR,
    "motif_result_bytes_per_node": MOTIF_RESULT_BYTES_PER_NODE,
    "duckdb_memory_fraction": DUCKDB_MEMORY_FRACTION,
    "duckdb_memory_min_gib": DUCKDB_MEMORY_MIN_GIB,
    "duckdb_memory_max_gib": DUCKDB_MEMORY_MAX_GIB,
    "ram_admission_fraction": RAM_ADMISSION_FRACTION,
    "disk_sort_factor": DISK_SORT_FACTOR,
    "disk_reserve_fraction": DISK_RESERVE_FRACTION,
    "disk_reserve_min_gib": DISK_RESERVE_MIN_GIB,
    "smoke_disk_reserve_gib": SMOKE_DISK_RESERVE_GIB,
    "result_staging_allowance_gib": RESULT_STAGING_ALLOWANCE_GIB,
    "checkpoint_staging_min_mib": CHECKPOINT_STAGING_MIN_MIB,
    "coordinator_allowance_gib": COORDINATOR_ALLOWANCE_GIB,
}

PROFILE_SETTINGS = {
    "thesis": {
        "permutations": N_PERMUTATIONS,
        "bootstraps": N_BOOTSTRAPS,
        "citable_candidate": True,
    },
    "development": {
        "permutations": 100,
        "bootstraps": 100,
        "citable_candidate": False,
    },
}
