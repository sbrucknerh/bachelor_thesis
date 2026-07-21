"""Locked experiment constants and the verified Raphtory index catalogue."""

from __future__ import annotations


PROTOCOL_VERSION = 3
RANDOM_SEED = 42
CV_SEEDS = (42, 43, 44)
DELTAS = (300, 1_800, 3_600, 21_600, 86_400, 259_200, 604_800)
N_PERMUTATIONS = 10_000
N_BOOTSTRAPS = 10_000

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
    "n_jobs": -1,
    "random_state": RANDOM_SEED,
}

DEFAULT_XGB_PARAMS = {
    "n_estimators": 100,
    "max_depth": 6,
    "learning_rate": 0.1,
    "n_jobs": -1,
    "random_state": RANDOM_SEED,
    "eval_metric": "logloss",
}

MULTICLASS_LOGISTIC_PARAMS = {
    "class_weight": "balanced",
    "max_iter": 2_000,
    "random_state": RANDOM_SEED,
}
