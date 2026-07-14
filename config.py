"""Central configuration for the thesis project."""

import os

# Project root directory
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# Data paths
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")

# Synthetic data defaults
DEFAULT_N_ADDRESSES = 100
DEFAULT_N_TRANSACTIONS = 500

# Temporal motif parameters
DEFAULT_DELTA = 3600  # Time window in seconds (1 hour)
DEFAULT_DELTAS = [3600, 86400, 604800]  # 1 hour, 1 day, 1 week

# Number of motif types in Paranjape et al. (2017) catalogue
NUM_MOTIF_TYPES = 36

# Detection thresholds (fallback for non-ML baseline)
DEFAULT_MOTIF_THRESHOLD = 5

# ML model defaults
DEFAULT_SUPERVISED_MODEL = "random_forest"  # "random_forest" or "xgboost"
DEFAULT_UNSUPERVISED_METHOD = "isolation_forest"  # "isolation_forest" or "dbscan"
DEFAULT_MOTIF_BACKEND = "raphtory"  # "networkx", "snap", or "raphtory"
RANDOM_STATE = 42
TEST_SIZE = 0.2

# Column mappings for different data sources
COLUMN_MAPS = {
    "default": {
        "from_address": "from_address",
        "to_address": "to_address",
        "value": "value",
        "timestamp": "timestamp",
        "tx_hash": "tx_hash",
    },
    "bigquery_ethereum": {
        "from_address": "from_address",
        "to_address": "to_address",
        "value": "value",
        "timestamp": "block_timestamp",
        "tx_hash": "hash",
    },
    "etherscan": {
        "from_address": "from",
        "to_address": "to",
        "value": "value",
        "timestamp": "timeStamp",
        "tx_hash": "hash",
    },
}

# Visualization
FIGURE_SIZE = (12, 8)
NODE_SIZE = 50
EDGE_ALPHA = 0.3
