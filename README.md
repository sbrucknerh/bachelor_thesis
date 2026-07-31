# Coordinated Malicious Behaviour Detection with Temporal Motifs

This repository contains the experiment pipeline for a bachelor thesis on
detecting malicious Ethereum addresses with temporal motifs.

- `scripts/run_results.py` runs the reproducible thesis experiments.
- `main.py` is an exploratory pipeline. Its outputs are not thesis evidence.

## Requirements

- Python 3.11 or newer
- The Hugging Face and Real-CATS datasets stored locally
- A local filesystem with enough free space for prepared data and checkpoints

The thesis environment is pinned in `requirements-results.txt`. Raphtory motif
counting is CPU-based; the pipeline does not require a GPU.

## Setup

### 1. Install the environment

```bash
git clone <repository-url>
cd bachelor_thesis

python3 -m venv venv
venv/bin/pip install -r requirements-results.txt
```

### 2. Place the datasets

#### Hugging Face (RQ1/RQ2)

Source: [Ethereum Fraud Dataset by Activity](https://huggingface.co/datasets/fesevu/ethereum_fraud_dataset_by_activity)

Download only the three files used by the pipeline into the standard Hugging
Face cache. This is approximately 21.6 GB, not the complete dataset repository:

```bash
venv/bin/hf download fesevu/ethereum_fraud_dataset_by_activity \
  gnn_dataset/edges_all/edges.parquet \
  gnn_dataset/labels/targets_global.parquet \
  addr_labels_balanced.csv.zst \
  --repo-type dataset
```

The downloaded snapshot must contain:

```text
gnn_dataset/edges_all/edges.parquet
gnn_dataset/labels/targets_global.parquet
addr_labels_balanced.csv.zst
```

The runner automatically searches the local Hugging Face cache. "Streaming"
in this project means that PyArrow reads this local Parquet file in bounded
batches instead of loading it fully into RAM. It is not remote network
streaming, so the files must be downloaded once before the run. When the
snapshot is elsewhere, pass `--hf-snapshot /path/to/snapshot` to the run
command.

#### Real-CATS (RQ3/RQ4)

- Labels: [official Real-CATS GitHub repository](https://github.com/sjdseu/Real-CATS)
- Transactions: [official Real-CATS Kaggle dataset](https://www.kaggle.com/datasets/lvd312393/real-cats)

Clone the labels repository into the expected location:

```bash
mkdir -p data/raw/realcats_transactions
git clone --depth 1 https://github.com/sjdseu/Real-CATS.git \
  data/raw/realcats_dataset
```

Download the transaction archive from Kaggle and place it without extracting
it at:

```text
data/raw/realcats_transactions/real-cats.zip
```

The transaction ZIP is read directly. Do not process it with
`scripts/extract_realcats.py` for the thesis experiments.

#### BLTE (smoke test only)

Source: [authors' labeled Ethereum transaction repository](https://github.com/salam-ammari/Labeled-Transactions-based-Dataset-of-Ethereum-Network)

```bash
git clone --depth 1 \
  https://github.com/salam-ammari/Labeled-Transactions-based-Dataset-of-Ethereum-Network.git \
  data/raw/ethereum_dataset_labelled
```

The smoke test reads these two files from that clone:

```text
data/raw/ethereum_dataset_labelled/Dataset.zip
data/raw/ethereum_dataset_labelled/Featuers_Extraction.csv
```

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
