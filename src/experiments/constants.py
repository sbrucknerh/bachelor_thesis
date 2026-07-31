"""Locked experiment constants and the verified Raphtory index catalogue."""

from __future__ import annotations


PROTOCOL_VERSION = 5
RANDOM_SEED = 42
CV_SEEDS = (42, 43, 44)
DELTAS = (300, 1_800, 3_600, 21_600, 86_400, 259_200, 604_800)
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

EVENT_EXECUTION = {
    "external_sort": "duckdb",
    "graph_loader": "raphtory_load_edges_arrow_batches",
    "large_event_pandas_materialization": False,
    "raphtory_roles_computed": 40,
    "motif_roles_retained": 32,
    "triangle_roles_retained": 0,
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

# Protocol-v5 resource-accounting constants. These are conservative admission
# assumptions, not operating-system-enforced limits for Raphtory.
GRAPH_COLUMN_EXPANSION_FACTOR = 4.0
HF_NODE_COUNT_INFLATION = 1.20
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
    "hf_node_count_inflation": HF_NODE_COUNT_INFLATION,
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
