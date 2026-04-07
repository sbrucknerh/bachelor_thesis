"""Central configuration for the thesis project."""

import os

# Project root directory
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# Data paths
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")

# Synthetic data defaults
DEFAULT_N_ADDRESSES = 100
DEFAULT_N_TRANSACTIONS = 500

# Temporal motif parameters
DEFAULT_DELTA = 3600  # Time window in seconds (1 hour)

# Detection thresholds
DEFAULT_MOTIF_THRESHOLD = 5  # Minimum motif count to flag as suspicious

# Visualization
FIGURE_SIZE = (12, 8)
NODE_SIZE = 50
EDGE_ALPHA = 0.3
