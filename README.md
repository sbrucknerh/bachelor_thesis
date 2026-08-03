# Coordinated Malicious Behaviour Detection with Temporal Motifs

This repository contains the experiment pipeline for a bachelor thesis on
detecting malicious Ethereum addresses with temporal motifs.

- `scripts/run_results.py` runs the reproducible thesis experiments.
- `main.py` is an exploratory pipeline. Its outputs are not thesis evidence.

## Requirements

- Python 3.11 or newer
- Git and an internet connection for the initial setup
- A Kaggle account and API token for the Real-CATS transaction archive
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

The Hugging Face dataset is public and requires no Hugging Face token.
Real-CATS transactions are hosted on Kaggle, so create a Kaggle token on the
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

### 1. Run the smoke test

Run this after setup and before using server time:

```bash
venv/bin/python scripts/run_results.py smoke \
  --run-id smoke_$(date +%Y%m%d_%H%M%S) \
  --sla-minutes 60
```

It checks every execution path but uses bounded data and reduced resampling, so
`results/smoke/<run-id>` is always marked `citable: false`.

### 2. Run all four RQs

For this thesis, the new representation analysis was designed after inspecting
the v5 HF run. The first v6 run must therefore use a disjoint cohort:

```bash
venv/bin/python scripts/run_results.py run-thesis \
  --run-id thesis_confirmation_1 \
  --threads 20 \
  --replication-index 1 \
  --exclude-hf-run results/experiments/thesis_final_20260731_hf
```

Copy the prior HF result root to the server first. Only `protocol.json`,
`protocol.sha256`, and `data/hf_targets.parquet` are needed, with that directory
layout preserved. For a genuinely new study with no inspected pilot cohort, use
`--run-id thesis_v6_initial --threads 20` and omit the last two options.

The command prepares and locks HF and Real-CATS, performs resource admission,
runs them in parallel when possible, checkpoints each window, creates all
artifacts, and requires citable verification before returning success.

On the provided 20-core, 128 GB RAM, 4 TB server, parallel execution normally
assigns 10 CPU threads to each dataset. The GPU is not used. The previous
seven-window run took about four hours; protocol v6 adds two windows and
secondary diagnostics, so plan for approximately 5-7 hours. The live ETA is
the better estimate once motif checkpoints start completing.

Run inside `tmux`. The terminal and `overall_progress.json` show total and
current-step ETAs; `run_log.jsonl` contains fsynced events and five-minute
heartbeats. If `--run-id` is omitted, a timestamped ID is generated.

### 3. Run the independent HF confirmation

Use all 20 cores for the second, HF-only confirmation. Exclude every earlier HF
cohort and increment the replication index:

```bash
venv/bin/python scripts/run_results.py run-hf \
  --run-id thesis_confirmation_2 \
  --threads 20 \
  --replication-index 2 \
  --exclude-hf-run results/experiments/thesis_final_20260731_hf \
  --exclude-hf-run results/experiments/thesis_confirmation_1_hf
```

The runner rejects a confirmation index without an exclusion reference. Never
select the best rerun. Report both v6 runs with:

```bash
venv/bin/python scripts/run_results.py aggregate-hf-replications \
  results/replications/hf_confirmation \
  results/experiments/thesis_confirmation_1_hf \
  results/experiments/thesis_confirmation_2_hf
```

The aggregator rejects overlap or incompatible definitions, reports every
supplied run, and does not pool p-values.

### 4. Monitor or resume

```bash
tail -f results/experiments/thesis_confirmation_1_hf/run_log.jsonl
```

Resume the complete run with exactly the same replication and exclusion options:

```bash
venv/bin/python scripts/run_results.py run-thesis \
  --run-id thesis_confirmation_1 \
  --threads 20 \
  --replication-index 1 \
  --exclude-hf-run results/experiments/thesis_final_20260731_hf \
  --resume
```

For `run-hf`, use the same pattern with that command and its original options.
Committed checkpoints are hash-checked and reused; outputs are never replaced.

### 5. Verify and export

The workflows verify automatically. Repeat the checks manually before writing:

```bash
venv/bin/python scripts/run_results.py verify \
  results/experiments/thesis_confirmation_1_hf \
  --require-citable

venv/bin/python scripts/run_results.py verify \
  results/experiments/thesis_confirmation_1_realcats \
  --require-citable
```

```bash
venv/bin/python scripts/results_to_latex.py writing/generated \
  results/experiments/thesis_confirmation_1_hf \
  results/experiments/thesis_confirmation_1_realcats
```

Both verifications must succeed. The exporter creates RQ1-RQ4 fragments in
`writing/generated/`; figures remain in each run's `figures/` directory.

## Manual Execution

The one-command workflow above is recommended. For debugging, the same work can
be controlled stage by stage:

```bash
# Prepare the two locked experiments
venv/bin/python scripts/run_results.py prepare \
  --run-id hf_confirmation_1 --profile thesis --rq rq1 rq2 \
  --replication-index 1 \
  --exclude-hf-run results/experiments/thesis_final_20260731_hf

venv/bin/python scripts/run_results.py prepare \
  --run-id realcats_confirmation_1 --profile thesis --rq rq3 rq4 \
  --replication-index 1

# Check combined resource admission without changing either run
venv/bin/python scripts/run_results.py preflight \
  results/experiments/hf_confirmation_1 \
  results/experiments/realcats_confirmation_1

# Execute both roots with a combined 20-thread budget
venv/bin/python scripts/run_results.py execute-parallel \
  results/experiments/hf_confirmation_1 \
  results/experiments/realcats_confirmation_1 \
  --threads 20
```

Use `execute --resume` for one interrupted root or `execute-parallel --resume`
when both coordinated workers were interrupted. Run
`scripts/run_results.py <command> --help` for less common path and lock options.

## Experiment Summary

| Question | Dataset | Analysis |
|---|---|---|
| RQ1 | Hugging Face | Baseline versus baseline plus motifs; prespecified volume, composition, presence, and incoming-role diagnostics |
| RQ2 | Hugging Face | Development-CV sweep from 5 minutes to 30 days; primary choice maximizes average precision |
| RQ3 | Real-CATS | Multiclass category distinguishability with a held-out label-permutation test |
| RQ4 | Real-CATS | Global versus category-specific delta selection with paired bootstrap inference |

RQ2 selects from 5 minutes, 30 minutes, 1 hour, 6 hours, 1 day, 3 days,
1 week, 2 weeks, and 30 days. It uses the highest mean development-CV AP;
exact ties choose the smaller delta. The one-standard-error choice is reported
only as a secondary parsimony result.

The RQ1 representation tests all use that same development-selected window and
the same held-out addresses. Canonical-order p-values are corrected across the
eight locked secondary contrasts. The three `star_*_iii` roles are specified
before execution; role-level direct-count scores are descriptive, not fitted
model performance.

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
`figures/`, plus hash-verified checkpoints and manifests. Protocol v1-v5 runs
are legacy read-only artifacts and cannot replace a verified protocol-v6 run.

Run `venv/bin/pytest -q` for development checks. Use
`venv/bin/python scripts/run_results.py --help` for command details. `main.py`
remains available for quick exploration, but its outputs are not citable.
