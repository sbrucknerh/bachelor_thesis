#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_DIR="${VENV_DIR:-$PROJECT_ROOT/venv}"
CHECK_ONLY=false
RUN_SMOKE=false

USER_ID="$(id -u)"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-${TMPDIR:-/tmp}/bachelor-thesis-pip-$USER_ID}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-${TMPDIR:-/tmp}/bachelor-thesis-matplotlib-$USER_ID}"

HF_REPO="fesevu/ethereum_fraud_dataset_by_activity"
HF_REVISION="6a7a517fbdbe2bcc4ec65bee4c0ff15f1c0a3c70"
HF_LOCAL_DIR="$PROJECT_ROOT/data/raw/huggingface_ethereum_fraud"
HF_CACHE_ROOT="${HF_HUB_CACHE:-${HF_HOME:-$HOME/.cache/huggingface}/hub}"
HF_CACHED_DIR="$HF_CACHE_ROOT/datasets--fesevu--ethereum_fraud_dataset_by_activity/snapshots/$HF_REVISION"

REALCATS_LABEL_URL="https://github.com/sjdseu/Real-CATS.git"
REALCATS_LABEL_REVISION="99623a1e588500d84379d58f17ec38e2d1816ed3"
REALCATS_LABEL_DIR="$PROJECT_ROOT/data/raw/realcats_dataset"
REALCATS_ZIP="$PROJECT_ROOT/data/raw/realcats_transactions/real-cats.zip"

BLTE_URL="https://github.com/salam-ammari/Labeled-Transactions-based-Dataset-of-Ethereum-Network.git"
BLTE_REVISION="497df9faf15c914becb3fcd121618e7a20da9c24"
BLTE_DIR="$PROJECT_ROOT/data/raw/ethereum_dataset_labelled"

usage() {
    cat <<'EOF'
Usage: ./scripts/setup_results.sh [--check-only] [--smoke]

Creates the thesis virtual environment, downloads the required datasets, and
validates the installation. The script is idempotent and reuses valid files.

Options:
  --check-only  Validate an existing installation without changing anything.
  --smoke       Run the bounded end-to-end smoke test after setup.
  -h, --help    Show this help.

Environment:
  PYTHON_BIN    Python executable used to create the venv (default: python3).
  VENV_DIR      Virtual-environment path (default: <project>/venv).

Kaggle authentication is required only when real-cats.zip is not already
present. Set KAGGLE_API_TOKEN, use ~/.kaggle/access_token, or run
`venv/bin/kaggle auth login` before rerunning this script.
EOF
}

log() {
    printf '[setup] %s\n' "$*"
}

fail() {
    printf '[setup] ERROR: %s\n' "$*" >&2
    exit 1
}

for argument in "$@"; do
    case "$argument" in
        --check-only) CHECK_ONLY=true ;;
        --smoke) RUN_SMOKE=true ;;
        -h|--help) usage; exit 0 ;;
        *) fail "Unknown option: $argument" ;;
    esac
done

if $CHECK_ONLY && $RUN_SMOKE; then
    fail "--check-only and --smoke cannot be used together"
fi

if [[ "$VENV_DIR" != /* ]]; then
    VENV_DIR="$PROJECT_ROOT/$VENV_DIR"
fi
VENV_PYTHON="$VENV_DIR/bin/python"
VENV_PIP="$VENV_DIR/bin/pip"
HF_CLI="$VENV_DIR/bin/hf"
KAGGLE_CLI="$VENV_DIR/bin/kaggle"

check_python_version() {
    "$1" -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 11) else 1)' \
        || fail "Python 3.11 or newer is required: $1"
}

hf_snapshot_valid() {
    local root="$1"
    [[ -s "$root/gnn_dataset/edges_all/edges.parquet" ]] \
        && [[ -s "$root/gnn_dataset/labels/targets_global.parquet" ]] \
        && [[ -s "$root/addr_labels_balanced.csv.zst" ]]
}

clone_pinned_dataset() {
    local name="$1"
    local url="$2"
    local revision="$3"
    local destination="$4"
    local required_file="$5"

    if [[ -s "$destination/$required_file" ]]; then
        log "$name already present"
        return
    fi
    [[ ! -e "$destination" ]] || fail \
        "$destination exists but $required_file is missing; move it aside and rerun"
    log "Downloading $name"
    git clone --quiet "$url" "$destination" \
        || fail "Could not clone $name from $url"
    git -C "$destination" checkout --quiet --detach "$revision" \
        || fail "Could not check out pinned $name revision $revision"
    [[ -s "$destination/$required_file" ]] \
        || fail "$name download did not contain $required_file"
}

if $CHECK_ONLY; then
    [[ -x "$VENV_PYTHON" ]] \
        || fail "Virtual environment not found at $VENV_DIR; run setup without --check-only"
else
    command -v git >/dev/null 2>&1 \
        || fail "git is required to download the dataset metadata"
    command -v "$PYTHON_BIN" >/dev/null 2>&1 \
        || fail "Python executable not found: $PYTHON_BIN"
    check_python_version "$PYTHON_BIN"
    if [[ ! -x "$VENV_PYTHON" ]]; then
        log "Creating virtual environment at $VENV_DIR"
        "$PYTHON_BIN" -m venv "$VENV_DIR"
    else
        log "Reusing virtual environment at $VENV_DIR"
    fi
fi

check_python_version "$VENV_PYTHON"

if ! $CHECK_ONLY; then
    log "Installing pinned thesis dependencies"
    PIP_DISABLE_PIP_VERSION_CHECK=1 "$VENV_PIP" install --quiet \
        -r requirements-results.txt
    log "Installing the Kaggle download client"
    PIP_DISABLE_PIP_VERSION_CHECK=1 "$VENV_PIP" install --quiet kaggle==2.2.4
fi

"$VENV_PYTHON" -m pip check >/dev/null \
    || fail "The virtual environment contains incompatible packages"
"$VENV_PYTHON" -c \
    'import datasets, duckdb, matplotlib, numpy, pandas, pyarrow, raphtory, scipy, sklearn, tqdm, xgboost, zstandard' \
    || fail "One or more thesis dependencies cannot be imported"

if ! $CHECK_ONLY; then
    mkdir -p "$PROJECT_ROOT/data/raw/realcats_transactions"
fi

if ! $CHECK_ONLY && [[ ! -s "$REALCATS_ZIP" ]]; then
    KAGGLE_CONFIG_PATH="${KAGGLE_CONFIG_DIR:-$HOME/.kaggle}"
    if [[ -z "${KAGGLE_API_TOKEN:-}" ]] \
        && [[ ! -s "$KAGGLE_CONFIG_PATH/access_token" ]] \
        && [[ ! -s "$KAGGLE_CONFIG_PATH/kaggle.json" ]] \
        && { [[ -z "${KAGGLE_USERNAME:-}" ]] || [[ -z "${KAGGLE_KEY:-}" ]]; }; then
        cat >&2 <<EOF
[setup] Kaggle authentication is required before dataset downloads.
Create a token at https://www.kaggle.com/settings/api, then use either:
  export KAGGLE_API_TOKEN="PASTE_TOKEN_HERE"
  $KAGGLE_CLI auth login
Rerun ./scripts/setup_results.sh after authentication.
EOF
        exit 2
    fi
fi

if hf_snapshot_valid "$HF_LOCAL_DIR"; then
    HF_SNAPSHOT="$HF_LOCAL_DIR"
    log "Hugging Face data already present in data/raw"
elif hf_snapshot_valid "$HF_CACHED_DIR"; then
    HF_SNAPSHOT="$HF_CACHED_DIR"
    log "Reusing pinned Hugging Face cache snapshot"
elif $CHECK_ONLY; then
    fail "Pinned Hugging Face snapshot is missing"
else
    log "Downloading the three required Hugging Face files (about 21.6 GB)"
    "$HF_CLI" download "$HF_REPO" \
        gnn_dataset/edges_all/edges.parquet \
        gnn_dataset/labels/targets_global.parquet \
        addr_labels_balanced.csv.zst \
        --repo-type dataset \
        --revision "$HF_REVISION" \
        --local-dir "$HF_LOCAL_DIR" \
        || fail "Hugging Face download failed; check the network and rerun"
    hf_snapshot_valid "$HF_LOCAL_DIR" \
        || fail "Hugging Face download completed without all required files"
    HF_SNAPSHOT="$HF_LOCAL_DIR"
fi

if $CHECK_ONLY; then
    [[ -s "$REALCATS_LABEL_DIR/CE.tsv" ]] || fail "Real-CATS CE.tsv is missing"
    [[ -s "$REALCATS_LABEL_DIR/BE.tsv" ]] || fail "Real-CATS BE.tsv is missing"
else
    clone_pinned_dataset \
        "Real-CATS labels" \
        "$REALCATS_LABEL_URL" \
        "$REALCATS_LABEL_REVISION" \
        "$REALCATS_LABEL_DIR" \
        "CE.tsv"
    [[ -s "$REALCATS_LABEL_DIR/BE.tsv" ]] \
        || fail "Real-CATS label checkout did not contain BE.tsv"
fi

if [[ ! -s "$REALCATS_ZIP" ]]; then
    $CHECK_ONLY && fail "Real-CATS transaction archive is missing: $REALCATS_ZIP"
    log "Downloading Real-CATS transactions from Kaggle (authentication required)"
    if ! "$KAGGLE_CLI" datasets download lvd312393/real-cats \
        --file real-cats.zip \
        --path "$(dirname "$REALCATS_ZIP")"; then
        cat >&2 <<EOF
[setup] Kaggle download failed. Authenticate with one of:
  export KAGGLE_API_TOKEN="PASTE_TOKEN_HERE"
  $KAGGLE_CLI auth login
Then rerun ./scripts/setup_results.sh; completed downloads will be reused.
EOF
        exit 2
    fi
else
    log "Real-CATS transaction archive already present"
fi

if $CHECK_ONLY; then
    [[ -s "$BLTE_DIR/Dataset.zip" ]] || fail "BLTE Dataset.zip is missing"
    [[ -s "$BLTE_DIR/Featuers_Extraction.csv" ]] \
        || fail "BLTE Featuers_Extraction.csv is missing"
else
    clone_pinned_dataset \
        "BLTE" \
        "$BLTE_URL" \
        "$BLTE_REVISION" \
        "$BLTE_DIR" \
        "Dataset.zip"
fi

log "Validating dataset schemas and archives"
"$VENV_PYTHON" - \
    "$HF_SNAPSHOT" \
    "$REALCATS_LABEL_DIR/CE.tsv" \
    "$REALCATS_LABEL_DIR/BE.tsv" \
    "$REALCATS_ZIP" \
    "$BLTE_DIR/Dataset.zip" \
    "$BLTE_DIR/Featuers_Extraction.csv" <<'PY'
from pathlib import Path
import sys
import zipfile

import pandas as pd
import pyarrow.parquet as pq

hf, ce_path, be_path, realcats_zip, blte_zip, blte_labels = map(Path, sys.argv[1:])
edges = pq.ParquetFile(hf / "gnn_dataset/edges_all/edges.parquet")
targets = pq.ParquetFile(hf / "gnn_dataset/labels/targets_global.parquet")
if not {"src_id", "dst_id", "ts"}.issubset(edges.schema.names):
    raise SystemExit("Hugging Face edge schema is incompatible")
if not {"node_id", "is_scam", "is_contract", "address"}.issubset(targets.schema.names):
    raise SystemExit("Hugging Face target schema is incompatible")
if edges.metadata.num_rows != 380_566_661 or targets.metadata.num_rows != 114_739:
    raise SystemExit("Hugging Face files do not match the pinned release row counts")
descriptions = pd.read_csv(hf / "addr_labels_balanced.csv.zst", nrows=1)
if not {"address", "description"}.issubset(descriptions.columns):
    raise SystemExit("Hugging Face description schema is incompatible")

ce_columns = set(ce_path.open(encoding="utf-8").readline().rstrip().split("\t"))
if not {"address", "label", "transaction_number"}.issubset(ce_columns):
    raise SystemExit("Real-CATS CE.tsv schema is incompatible")
be_columns = set(be_path.open(encoding="utf-8").readline().rstrip().split("\t"))
if not {"address", "label", "transaction_number"}.issubset(be_columns):
    raise SystemExit("Real-CATS BE.tsv schema is incompatible")
with zipfile.ZipFile(realcats_zip) as archive:
    if not any(item.filename.endswith("/normal_transactions.json") for item in archive.infolist()):
        raise SystemExit("Real-CATS ZIP does not contain address transaction histories")
with zipfile.ZipFile(blte_zip) as archive:
    if "Dataset/Dataset.csv" not in archive.namelist():
        raise SystemExit("BLTE ZIP does not contain Dataset/Dataset.csv")
blte_columns = set(pd.read_csv(blte_labels, nrows=1).columns)
if not {"Address", "Scam"}.issubset(blte_columns):
    raise SystemExit("BLTE label schema is incompatible")

print(f"Hugging Face edges: {edges.metadata.num_rows:,}")
print(f"Hugging Face targets: {targets.metadata.num_rows:,}")
print(f"Hugging Face snapshot: {hf}")
PY

log "Setup validation passed"

if $RUN_SMOKE; then
    SMOKE_ID="setup_smoke_$(date -u +%Y%m%d_%H%M%S)"
    log "Running smoke test: $SMOKE_ID"
    "$VENV_PYTHON" scripts/run_results.py smoke \
        --run-id "$SMOKE_ID" \
        --hf-snapshot "$HF_SNAPSHOT" \
        --sla-minutes 60
fi

if $CHECK_ONLY; then
    COMPLETION_MESSAGE="Validation complete."
else
    COMPLETION_MESSAGE="Setup complete."
fi

cat <<EOF

$COMPLETION_MESSAGE

Run the smoke test before server execution:
  "$VENV_PYTHON" scripts/run_results.py smoke --run-id setup_check --sla-minutes 60

The citable v8 command requires the three earlier HF exclusion roots. See the
step-by-step "Full thesis run" command in README.md.

Run setup validation again with:
  ./scripts/setup_results.sh --check-only
EOF
