# Coordinated Malicious Behaviour Detection with Temporal Motifs

This repository contains the reproducible experiment pipeline for a bachelor
thesis on malicious Ethereum address detection. Use `scripts/run_results.py` for
thesis evidence. `main.py` is exploratory and its outputs are not citable.

## Requirements

- Python 3.11 or newer
- Git and enough local storage for the datasets and result checkpoints
- A Kaggle account only for downloading the Real-CATS transaction archive

Dependencies are pinned in `requirements-results.txt`. Motif counting uses CPU
threads; the GPU is not used.

## Automatic Setup

Create a Kaggle token at [Kaggle settings](https://www.kaggle.com/settings/api),
then run:

```bash
export KAGGLE_API_TOKEN="PASTE_TOKEN_HERE"
./scripts/setup_results.sh
```

The idempotent script creates `venv/`, installs dependencies, downloads the
pinned datasets, and validates their schemas. Useful variants are:

```bash
./scripts/setup_results.sh --check-only
./scripts/setup_results.sh --smoke
```

## Manual Setup

1. Create the environment:

```bash
python3 -m venv venv
venv/bin/pip install -r requirements-results.txt
venv/bin/pip install kaggle==2.2.4
```

2. Download the three pinned Hugging Face files:

```bash
venv/bin/hf download fesevu/ethereum_fraud_dataset_by_activity \
  gnn_dataset/edges_all/edges.parquet \
  gnn_dataset/labels/targets_global.parquet \
  addr_labels_balanced.csv.zst \
  --repo-type dataset \
  --revision 6a7a517fbdbe2bcc4ec65bee4c0ff15f1c0a3c70 \
  --local-dir data/raw/huggingface_ethereum_fraud
```

3. Download Real-CATS labels and transactions:

```bash
git clone https://github.com/sjdseu/Real-CATS.git data/raw/realcats_dataset
git -C data/raw/realcats_dataset checkout --detach \
  99623a1e588500d84379d58f17ec38e2d1816ed3

mkdir -p data/raw/realcats_transactions
export KAGGLE_API_TOKEN="PASTE_TOKEN_HERE"
venv/bin/kaggle datasets download lvd312393/real-cats \
  --file real-cats.zip --path data/raw/realcats_transactions
```

4. Download BLTE and validate everything:

```bash
git clone \
  https://github.com/salam-ammari/Labeled-Transactions-based-Dataset-of-Ethereum-Network.git \
  data/raw/ethereum_dataset_labelled
git -C data/raw/ethereum_dataset_labelled checkout --detach \
  497df9faf15c914becb3fcd121618e7a20da9c24
./scripts/setup_results.sh --check-only
```

Sources: [Hugging Face](https://huggingface.co/datasets/fesevu/ethereum_fraud_dataset_by_activity),
[Real-CATS labels](https://github.com/sjdseu/Real-CATS),
[Real-CATS transactions](https://www.kaggle.com/datasets/lvd312393/real-cats), and
[BLTE](https://github.com/salam-ammari/Labeled-Transactions-based-Dataset-of-Ethereum-Network).

## Run Procedure

### 1. Smoke test

Run the non-citable, bounded test before using server time:

```bash
venv/bin/python scripts/run_results.py smoke \
  --run-id smoke_$(date +%Y%m%d_%H%M%S) --sla-minutes 60
```

### 2. Full thesis run

Protocol v8 is a final post-hoc confirmation. Copy the three earlier HF result
roots to the server, preserving `protocol.json`, `protocol.sha256`, and
`data/hf_targets.parquet`. Then all four RQs and the external Real-CATS binary
replication run with one command:

```bash
venv/bin/python scripts/run_results.py run-thesis \
  --run-id thesis_final_v8 \
  --threads 20 \
  --replication-index 3 \
  --exclude-hf-run results/experiments/thesis_final_20260731_hf \
  --exclude-hf-run results/experiments/thesis_confirmation_1_hf \
  --exclude-hf-run results/experiments/thesis_confirmation_2_hf
```

The runner rejects incomplete exclusion lineage, source-file mismatches,
overlapping cohorts, insufficient resources, and non-citable output. It chooses
parallel or sequential HF/Real-CATS execution from a read-only resource check.
On the supplied 20-core, 128 GB, 4 TB server, allow roughly 12-30 hours for v8;
the terminal replaces this initial range with a measured total ETA.

For an HF-only v8 confirmation, replace `run-thesis` with `run-hf` and keep the
same replication and exclusion options.

### 3. Monitor or resume

Run the job inside `tmux`. Monitor the root-level total ETA and fsynced log:

```bash
tail -f results/experiments/thesis_final_v8_hf/run_log.jsonl
```

Rerun the same command with `--resume` after interruption. Committed checkpoints
are hash-verified and reused; public artifacts are never overwritten.

### 4. Verify and export

The one-command workflow verifies automatically. Manual checks are:

```bash
venv/bin/python scripts/run_results.py verify \
  results/experiments/thesis_final_v8_hf --require-citable
venv/bin/python scripts/run_results.py verify \
  results/experiments/thesis_final_v8_realcats --require-citable

venv/bin/python scripts/results_to_latex.py writing/generated \
  results/experiments/thesis_final_v8_hf \
  results/experiments/thesis_final_v8_realcats
```

Do not cite a run unless both verification commands succeed.

## Manual Execution

For debugging, prepare, inspect, and execute each locked root separately:

```bash
venv/bin/python scripts/run_results.py prepare \
  --run-id hf_v8 --profile thesis --rq rq1 rq2 \
  --replication-index 3 \
  --exclude-hf-run results/experiments/thesis_final_20260731_hf \
  --exclude-hf-run results/experiments/thesis_confirmation_1_hf \
  --exclude-hf-run results/experiments/thesis_confirmation_2_hf

venv/bin/python scripts/run_results.py prepare \
  --run-id realcats_v8 --profile thesis --rq rq3 rq4 \
  --replication-index 3

venv/bin/python scripts/run_results.py preflight \
  results/experiments/hf_v8 results/experiments/realcats_v8

venv/bin/python scripts/run_results.py execute-parallel \
  results/experiments/hf_v8 results/experiments/realcats_v8 --threads 20
```

BLTE remains an optional, separate activity-matched replication:

```bash
venv/bin/python scripts/run_results.py prepare \
  --run-id blte_v8 --profile thesis --rq blte
venv/bin/python scripts/run_results.py execute \
  results/experiments/blte_v8 --threads 20
```

## Experiment Scope

| Question | Primary data | Analysis |
|---|---|---|
| RQ1 | Hugging Face | Baseline versus motifs; 32-role, triangle-only, and 40-role comparisons on identical folds |
| RQ2 | Hugging Face | Development-CV window sweep from 5 minutes to 30 days for each representation |
| RQ3 | Real-CATS | Exact-category multiclass fingerprint distinguishability |
| RQ4 | Real-CATS | Global versus category-specific temporal windows |

Real-CATS also supplies an HF-disjoint secondary RQ1/RQ2 check. It compares
`CE.tsv` with `BE.tsv`, requires at least three usable events, and matches
criminal and benign addresses one-to-one within fixed activity bins. Source
activity is used only for cohort matching and never as a model feature. All five
transaction layers are retained separately. BLTE applies the same activity-bin
matching principle.

The pinned Hugging Face `edges_all` file contains 380,566,661 rows in the
canonical released graph table. The repository's raw-address, weekly, and
monthly Parquets are alternate representations or copies and do not add
relationships beyond that canonical release, so they
are not downloaded. Preparation scans the full local file in batches; it does
not load it all into pandas. The release itself contains upstream activity,
value, neighbor, and pair caps. Reversing those caps would require a new chain
query, which this project intentionally does not perform. No BigQuery account or
paid data service is used.

HF triangle counts are complete only within the released target-neighborhood
closure, not within the full Ethereum chain. Real-CATS RQ3/RQ4 uses the 32 star
and two-node roles because its ego histories cannot guarantee neighbor-to-neighbor
triangle completeness.

Result roots contain immutable `features/`, `predictions/`, `metrics/`,
`tables/`, `figures/`, checkpoints, and integrity manifests. Protocol v6 and v7
runs remain read-only verifiable; new execution uses protocol v8.

Run `venv/bin/pytest -q` for development checks and
`venv/bin/python scripts/run_results.py --help` for all CLI options.
