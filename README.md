# Coordinated Malicious Behaviour Detection with Temporal Motifs

This repository contains the experiment pipeline for a bachelor thesis on
detecting malicious Ethereum addresses with temporal motifs.

- `scripts/run_results.py` runs the reproducible thesis experiments.
- `main.py` is an exploratory pipeline. Its outputs are not thesis evidence.

## Requirements

- Python 3.11 or newer
- Git and an internet connection for the initial setup
- A Kaggle account and API token for downloading Real-CATS
- A local filesystem with enough free space for prepared data and checkpoints

The thesis environment is pinned in `requirements-results.txt`. Raphtory motif
counting is CPU-based; the pipeline does not require a GPU.

## Setup

Clone the project first:

```bash
git clone <repository-url>
cd bachelor_thesis
```

### Option A: Installation Script (recommended)

Create a token on the
[Kaggle API settings page](https://www.kaggle.com/settings/api), then run:

```bash
export KAGGLE_API_TOKEN="PASTE_TOKEN_HERE"
./scripts/setup_results.sh
```

The script is safe to rerun. It:

- creates `venv/` and installs the pinned dependencies;
- downloads the three required Hugging Face files, approximately 21.6 GB;
- downloads the Real-CATS labels and transaction archive;
- downloads BLTE for the smoke test;
- validates the dependency, Parquet, ZIP, and label schemas.

Use `--smoke` to run the bounded end-to-end test immediately after setup, or
`--check-only` to validate an existing installation without downloading data:

```bash
./scripts/setup_results.sh --smoke
./scripts/setup_results.sh --check-only
```

### Option B: Manual Setup

1. Create the environment:

```bash
python3 -m venv venv
venv/bin/pip install -r requirements-results.txt
venv/bin/pip install kaggle==2.2.4
```

2. Download the three Hugging Face files:

```bash
venv/bin/hf download fesevu/ethereum_fraud_dataset_by_activity \
  gnn_dataset/edges_all/edges.parquet \
  gnn_dataset/labels/targets_global.parquet \
  addr_labels_balanced.csv.zst \
  --repo-type dataset \
  --revision 6a7a517fbdbe2bcc4ec65bee4c0ff15f1c0a3c70 \
  --local-dir data/raw/huggingface_ethereum_fraud
```

3. Download Real-CATS:

```bash
git clone https://github.com/sjdseu/Real-CATS.git data/raw/realcats_dataset
git -C data/raw/realcats_dataset checkout --detach \
  99623a1e588500d84379d58f17ec38e2d1816ed3

mkdir -p data/raw/realcats_transactions
export KAGGLE_API_TOKEN="PASTE_TOKEN_HERE"
venv/bin/kaggle datasets download lvd312393/real-cats \
  --file real-cats.zip \
  --path data/raw/realcats_transactions
```

4. Download BLTE and validate the installation:

```bash
git clone \
  https://github.com/salam-ammari/Labeled-Transactions-based-Dataset-of-Ethereum-Network.git \
  data/raw/ethereum_dataset_labelled
git -C data/raw/ethereum_dataset_labelled checkout --detach \
  497df9faf15c914becb3fcd121618e7a20da9c24

./scripts/setup_results.sh --check-only
```

Both setup methods use these sources:

- [Hugging Face Ethereum Fraud Dataset](https://huggingface.co/datasets/fesevu/ethereum_fraud_dataset_by_activity)
- [Real-CATS labels](https://github.com/sjdseu/Real-CATS)
- [Real-CATS transactions](https://www.kaggle.com/datasets/lvd312393/real-cats)
- [BLTE](https://github.com/salam-ammari/Labeled-Transactions-based-Dataset-of-Ethereum-Network)

During experiments, Hugging Face data is streamed in bounded batches from the
local Parquet file rather than loaded completely into RAM. It must therefore be
downloaded once by either setup method.

## Usage

### Step 1: Run the smoke test

Always run this before starting the multi-day experiment:

```bash
venv/bin/python scripts/run_results.py smoke \
  --run-id smoke_$(date +%Y%m%d_%H%M%S) \
  --sla-minutes 60
```

The smoke test exercises preparation, motif counting, model fitting,
statistical tests, figures, tables, and integrity verification. It normally
finishes well within one hour; the last local protocol-v5 run took 76.4
seconds.

Smoke results are stored under `results/smoke/<run-id>`. They are deliberately
marked `citable: false` because they use bounded data and reduced resampling.

### Step 2: Start the complete thesis run

After the smoke test passes, the complete workflow is one command:

```bash
venv/bin/python scripts/run_results.py run-thesis \
  --run-id thesis_final_20260731 \
  --threads 20
```

The command automatically:

1. checks the input datasets;
2. prepares and locks the Hugging Face and Real-CATS experiments;
3. checks available memory and storage;
4. runs both experiments in parallel when the machine can admit them;
5. falls back to sequential execution when necessary;
6. creates resumable checkpoints, metrics, tables, and figures;
7. verifies both result directories as thesis-citable.

On the provided 20-core, 128 GB RAM, 4 TB server, parallel execution normally
assigns 10 CPU threads to each dataset. The GPU is not used. The current full
runtime estimate is approximately 3-10 days, but the measured ETA shown during
execution is more useful than this planning estimate.

`--run-id` is optional. If omitted, the runner creates a timestamped ID. An
explicit ID is recommended because it is needed for simple recovery after an
interruption.

Run the command inside `tmux` or another persistent server session. The
terminal displays `tqdm` progress bars with elapsed time and ETA. Detailed,
fsynced events and five-minute heartbeats are written to `run_log.jsonl`.

### Step 3: Monitor the run

The command creates two result directories:

```text
results/experiments/thesis_final_20260731_hf/
results/experiments/thesis_final_20260731_realcats/
```

Monitor either worker with:

```bash
tail -f results/experiments/thesis_final_20260731_hf/run_log.jsonl
tail -f results/experiments/thesis_final_20260731_realcats/run_log.jsonl
```

### Step 4: Resume after an interruption

Use the same run ID and thread budget:

```bash
venv/bin/python scripts/run_results.py run-thesis \
  --run-id thesis_final_20260731 \
  --threads 20 \
  --resume
```

Committed checkpoints are validated and reused. Existing results are never
silently overwritten.

### Step 5: Verify the finished results

`run-thesis` performs this verification automatically. It can also be repeated
manually:

```bash
venv/bin/python scripts/run_results.py verify \
  results/experiments/thesis_final_20260731_hf \
  --require-citable

venv/bin/python scripts/run_results.py verify \
  results/experiments/thesis_final_20260731_realcats \
  --require-citable
```

Both commands must exit successfully before their results are used in the
thesis.

### Step 6: Generate the LaTeX results

```bash
venv/bin/python scripts/results_to_latex.py writing/generated \
  results/experiments/thesis_final_20260731_hf \
  results/experiments/thesis_final_20260731_realcats
```

This creates the RQ1-RQ4 table fragments and macros in `writing/generated/`.
Figures remain in each verified run's `figures/` directory.

## Manual Execution

The one-command workflow above is recommended. For debugging, the same work can
be controlled stage by stage:

```bash
# Prepare the two locked experiments
venv/bin/python scripts/run_results.py prepare \
  --run-id hf_rq12_v5 --profile thesis --rq rq1 rq2

venv/bin/python scripts/run_results.py prepare \
  --run-id realcats_rq34_v5 --profile thesis --rq rq3 rq4

# Check combined resource admission without changing either run
venv/bin/python scripts/run_results.py preflight \
  results/experiments/hf_rq12_v5 \
  results/experiments/realcats_rq34_v5

# Execute both roots with a combined 20-thread budget
venv/bin/python scripts/run_results.py execute-parallel \
  results/experiments/hf_rq12_v5 \
  results/experiments/realcats_rq34_v5 \
  --threads 20
```

Use `execute --resume` for one interrupted root or `execute-parallel --resume`
when both coordinated workers were interrupted. Run
`scripts/run_results.py <command> --help` for less common path and lock options.

## Experiment Summary

| Question | Dataset | Analysis |
|---|---|---|
| RQ1 | Hugging Face | Baseline versus baseline plus temporal motifs on identical grouped splits |
| RQ2 | Hugging Face | Development-CV delta sweep; primary choice maximizes average precision |
| RQ3 | Real-CATS | Multiclass category distinguishability with a held-out label-permutation test |
| RQ4 | Real-CATS | Global versus category-specific delta selection with paired bootstrap inference |

RQ2 selects the delta with the highest mean development-CV average precision;
exact ties choose the smaller delta. The one-standard-error choice is reported
only as a secondary parsimony result.

Real-CATS categories use exact, non-`Other` labels with at least 100 active
addresses. Labels are not merged after inspecting results. Because Real-CATS
does not provide a separate coordination label, RQ3 and RQ4 concern temporal
fingerprints associated with its malicious-behaviour categories, not a direct
supervised test of coordination itself.

The Hugging Face source is scanned from local Parquet in bounded batches, while
motif counting uses the selected target-incident subgraph. Real-CATS retains
normal, internal, ERC-20, ERC-721, and ERC-1155 transfers as separate layers.

BLTE is optional secondary validation and is not required for the primary run
to be citable. Synthetic data is used only for motif-catalogue and boundary
validation.

Result roots contain `features/`, `predictions/`, `metrics/`, `tables/`, and
`figures/`, plus hash-verified checkpoints and manifests. Protocol v1-v4 runs
cannot be accepted as protocol-v5 thesis evidence.

Run `venv/bin/pytest -q` for development checks. Use
`venv/bin/python scripts/run_results.py --help` for command details. `main.py`
remains available for quick exploration, but its outputs are not citable.
