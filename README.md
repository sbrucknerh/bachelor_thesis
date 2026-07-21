# Coordinated Malicious Behaviour Detection in Blockchain Networks Using Temporal Motifs

Bachelor thesis project implementing a pipeline for detecting coordinated malicious behaviour in Ethereum transaction networks through temporal motif analysis.

The system models blockchain transactions as a temporal directed multigraph, extracts structural and temporal motif features per address, and applies supervised and unsupervised ML methods. `main.py` is the exploratory workflow. The locked, leakage-controlled thesis experiments are run through `scripts/run_results.py` and `src/experiments/`.

## Project Structure

```
bachelor_thesis/
├── main.py                        Entry point — orchestrates the pipeline
├── config.py                      Central configuration and parameters
├── requirements.txt               Python dependencies
├── src/
│   ├── data_loading.py            Load transactions from CSV / JSON / Parquet / HuggingFace
│   ├── synthetic_data.py          Generate synthetic datasets with injected patterns
│   ├── graph_construction.py      Build NetworkX graph, extract per-node structural features
│   ├── temporal_motifs.py         Thin re-export layer for motif utilities
│   ├── experiments/               Locked cohorts, features, inference, and artifacts
│   │   └── smoke.py               Bounded non-citable all-path validation profile
│   ├── motifs/                    Three interchangeable motif-counter backends
│   │   ├── base.py                Shared contract (see module docstring)
│   │   ├── networkx_backend.py    Pure-Python 5-type counter
│   │   ├── snap_backend.py        SNAP.py 36-type counter (placeholder, broken)
│   │   └── raphtory_backend.py    Raphtory 40-type counter (default)
│   ├── feature_engineering.py     Combine motif fingerprints with graph/temporal features
│   ├── detection.py               Supervised (RF, XGBoost) and unsupervised (IF, DBSCAN)
│   ├── evaluation.py              Metrics, ROC/PR curves, model comparison
│   └── visualization.py           Network plots, heatmaps, t-SNE, sensitivity charts
├── data/raw/                      Raw datasets (gitignored)
├── models/                        Saved trained models (gitignored)
├── results/                       Evaluation outputs and figures
├── notebooks/                     Jupyter notebooks for exploration (empty placeholder)
├── scripts/run_results.py         Prepare/execute entry point for thesis Results runs
└── tests/                         Unit and miniature end-to-end experiment tests
```

## Setup

### Prerequisites

- Python 3.11 or newer
- Enough local disk for prepared Parquet artifacts, DuckDB external-sort spill files,
  and temporary Real-CATS deduplication
- A higher-memory machine for a locked run when the execution preflight rejects the local machine

### Full-run server requirements

The Results pipeline is CPU and memory-bandwidth bound. A GPU is not used by
Raphtory, DuckDB, Random Forest, XGBoost, or the statistical procedures in this
configuration. CUDA is therefore unnecessary. Use a local NVMe SSD; a hard
disk or network-mounted working directory makes the external sorts and
Real-CATS SQLite conversion substantially slower.

| Resource | Practical minimum | Recommended full-run server | Notes |
|---|---:|---:|---|
| CPU | 16 modern physical cores | 24-32 physical cores | Raphtory uses all available cores inside one motif call. More cores reduce counting time, but the runner does not execute order variants or datasets concurrently. |
| RAM | 32 GB if the locked preflight accepts it | 64 GB; 128 GB for additional margin | A 16 GB machine is insufficient for the existing large HF artifact. Do not treat swap as usable graph memory. The actual requirement is run-dependent and the preflight is authoritative. |
| Free local SSD for RQ1/RQ2 | 100 GB | 150 GB | Includes the approximately 21 GB HF cache, prepared incident edges, DuckDB spill space, fingerprints, and outputs. |
| Free local SSD for RQ3 | 120 GB | 200 GB | The Real-CATS ZIP is approximately 12 GB and lists about 72 GB of uncompressed per-address JSON in total; RQ3 reads the criminal-address subset. Peak usage includes SQLite, layer Parquets, and sort spill. |
| Free local SSD for all retained runs | 250 GB | 350 GB or more on a 500 GB volume | Assumes both raw sources and both locked run directories remain available. Keep at least 20% of the filesystem free for temporary work. |
| GPU | Not required | Not required | The locked estimators use CPU implementations. |
| Network | Required only for initial downloads | Not required during prepare/execute | HuggingFace "streaming" means bounded reads from the locally cached Parquet, not repeated remote access. |

Cloud vendors usually advertise SMT threads as vCPUs rather than physical
cores. In that terminology, target approximately 32-64 dedicated vCPUs and
avoid heavily oversubscribed burst instances.

The 32 GB RAM figure is a floor, not a guarantee. As a concrete sizing anchor,
the existing local HF preparation contains about 140 million incident edges;
the current conservative preflight projects approximately 14.1 GiB peak and
rejects a 16 GB machine because execution would exceed the 80% memory safety
limit. A newly prepared protocol may select a different edge volume. Always run
the preflight on the intended server, and use its decision instead of the table
when they differ.

Recommended server profile for the complete thesis evaluation:

```text
Linux x86-64 recommended; otherwise verify all pinned wheels first
24-32 physical CPU cores
64-128 GB RAM
500 GB local NVMe SSD
Python and dependency versions exactly matching protocol.json
No GPU required
```

Run RQ1/RQ2 and RQ3 sequentially. Starting both executions at once competes for
RAM, SSD bandwidth, and all CPU cores, and makes both slower.

### Approximate full-run time

The following ranges are capacity-planning estimates, not guaranteed runtimes.
They assume that the datasets are already downloaded, local NVMe storage, no
other heavy workload, and a 24-32-core server with 64-128 GB RAM. The first
complete protocol-v3 run should be used to replace these estimates with the
measured durations from `run_log.jsonl`.

| Stage | Approximate time | Main cost |
|---|---:|---|
| Synthetic validation | Under 5 minutes | Small exact Raphtory fixtures and artifact checks |
| Hybrid internal smoke run | Target under 1 hour; 67.9 seconds observed locally | Bounded real HF and Real-CATS data, full BLTE, all RQ orchestration, and synthetic validation |
| HuggingFace RQ1/RQ2 preparation | 10-45 minutes | Three full local reads of the approximately 21 GB edge Parquet: activity, extraction, and SHA-256 |
| Real-CATS RQ3 preparation | 3-12 hours | ZIP JSON parsing, durable SQLite deduplication, checkpoint commits, and five Parquet exports |
| HuggingFace RQ1/RQ2 execution | 2-7 days | Seven graph builds and 49 full Raphtory motif calls: seven order variants times seven delta values |
| Real-CATS RQ3 execution | 2-7 days | Five layers processed for seven order variants and seven delta values, followed by exact-label inference |
| Both primary executions, sequentially | About 4-14 days | Depends mainly on prepared event counts, graph density, CPU memory bandwidth, and motif-window density |

On an 8-core laptop, preparation remains feasible when storage is sufficient,
but execution can take roughly one to three weeks and may fail the RAM
preflight. Renting a 24-32-core, 64-128 GB RAM server is the more predictable
option. Increasing GPU capacity will not shorten the current pipeline.

The model fits are not the main bottleneck. The dominant stage is Raphtory's
`local_temporal_three_node_motifs`: it computes all 40 local roles for every
graph node at every delta, even though the pipeline retains only 32 roles.
Graph reuse removes repeated construction within a single order variant, but
the seven sensitivity orders still require seven separately ordered graph
loads. Equal-timestamp sensitivity and the 10,000-iteration inference settings
are part of the locked scientific protocol and should not be removed merely to
meet a preferred deadline.

### Installation

```bash
# Clone the repository
git clone <repo-url>
cd bachelor_thesis

# Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies (includes Raphtory for the default motif backend)
pip install -r requirements.txt
```

### HuggingFace datasets

Some dataset types (`huggingface`, `huggingface-gnn`) require authenticated access:

```bash
huggingface-cli login
```

### Real-CATS dataset

Download `real-cats.zip` from Kaggle (`lvd312393/real-cats`) and place the files as follows:

```bash
data/raw/realcats_transactions/real-cats.zip
data/raw/realcats_dataset/CE.tsv
```

The Results runner reads all five transaction layers directly from the ZIP and globally deduplicates overlapping address histories. Do not run `scripts/extract_realcats.py` for RQ3; that script extracts only normal transactions for the exploratory `main.py` workflow.

All five layers are retained intentionally. Normal and internal transfers describe native-ETH and contract-mediated flows, while ERC-20, ERC-721, and ERC-1155 transfers are needed for type-specific behavior such as liquidity exits and NFT airdrops. Motifs are counted separately per layer, so unlike token standards are not mixed into one event sequence.

For exploratory use only:

```bash
python scripts/extract_realcats.py
python main.py --mode supervised --dataset-type realcats \
  --data data/raw/realcats_transactions --features full --no-plot
```

### Optional: SNAP.py backend

The `snap` motif backend (Paranjape et al.'s original C++ implementation via SNAP.py) is kept as a placeholder but is currently non-functional — the installed `snap-stanford` 6.0 does not expose the API this project was written against. SNAP.py also has no PyPI wheels for Python ≥ 3.10, so using it at all would require a separate Python 3.9 venv.

Use `--motif-backend raphtory` (default) or `--motif-backend networkx` instead. The SNAP code path raises a clear `RuntimeError` if invoked.

## Usage

### One-hour internal smoke test

Run this before committing several days to the complete thesis protocol. The
smoke command uses the real local source formats but applies deterministic
bounds, then passes the prepared artifacts through the normal execution path:

```bash
python scripts/run_results.py smoke \
  --run-id smoke_$(date +%Y%m%d_%H%M%S) \
  --sla-minutes 60
```

The default output root is `results/smoke/<run-id>`, never
`results/experiments`. A smoke run includes:

- the first 5,000,000 HuggingFace edges, 200 prevalence targets, and up to five
  controlled targets per label × contract × activity-quintile stratum;
- 20 bounded archive-backed addresses from each of `Phishing Scam`,
  `Impersonation Scam`, and `Hack Scam`, with all five Real-CATS layers;
- the complete small BLTE release and all deterministic 32-role validation
  fixtures;
- three delta values (5 minutes, 1 hour, and 1 week), canonical/reverse/one
  seeded timestamp order, five-fold CV, and 100 permutation/bootstrap draws;
- baseline and motif features, Random Forest, XGBoost, multinomial logistic
  regression, statistical-test code, figures, tables, predictions, protocol
  verification, and the final manifest.

On the current 8-core M1 Pro with 16 GB RAM, the verified hybrid run completed
in 67.9 seconds, created 90 files, and occupied 63 MB. Treat that as a local
reference rather than a runtime guarantee; source revisions and machine load
can change it.

Inspect the final status with:

```bash
jq '{status, elapsed_seconds, sla_seconds, sla_met, citable}' \
  results/smoke/<run-id>/smoke_summary.json
jq '{status, execution_profile, citable, results: (.results | keys)}' \
  results/smoke/<run-id>/execution_manifest.json
```

Every smoke directory contains `SMOKE_ONLY.txt`, and both `protocol.json` and
`execution_manifest.json` set `execution_profile` to `smoke` and `citable` to
`false`. Smoke metrics are implementation diagnostics, not thesis findings.
They do not validate full-dataset memory consumption, full-run runtime, final
cohort composition, seven-order robustness, or 10,000-iteration inference.

The 60-minute limit is an SLA rather than a force-kill. If the bounded run
exceeds it, the current native operation is allowed to finish so DuckDB,
SQLite, Raphtory, and Parquet state is not interrupted. The command writes
`sla_met: false`, preserves all diagnostics, and exits with status 2 after
completion. Pipeline failures exit nonzero and write their exception details to
`smoke_summary.json`. Smoke runs are immutable; use a new run ID instead of
resuming or overwriting one.

### Thesis Results experiments

The thesis workflow is separate from `main.py`. It has two required stages:

1. `prepare` reads raw data, locks cohorts and folds, and creates auditable data artifacts.
2. `execute` verifies the locked protocol and artifacts before computing any feature or metric.

The protocol and prepared inputs are immutable after `protocol.json` is written. Execution may create previously absent result artifacts and append `run_log.jsonl`, but it never overwrites an existing output. Never edit a protocol, fold file, prepared Parquet file, or result in place; use a new run ID.

#### 1. Verify the installation

Run the exact synthetic motif fixtures before a full dataset preparation:

```bash
python scripts/run_results.py prepare \
  --run-id validation_20260715 \
  --rq validation
python scripts/run_results.py execute \
  results/experiments/validation_20260715
```

The validation must report `"passed": true` in
`results/experiments/validation_20260715/execution_manifest.json`.

#### 2. Prepare RQ1 and RQ2

The HuggingFace preparation makes two analytical passes over the complete 380,566,661-edge Parquet: one for target activity and one for incident-edge extraction. A separate bytewise pass records the source SHA-256 digest. It does not load a monthly head sample. Before cohort sampling, targets must have at least three incident events and addresses above the 99.5th activity percentile within each label × contract-type × activity-quintile stratum are excluded. These rules are fixed before modeling, independent of observed performance, and locked in `protocol.json`; exact attrition is recorded by stratum. The prevalence cohort therefore represents the eligible non-extreme population, not every labelled address.

```bash
python scripts/run_results.py prepare \
  --run-id hf_rq12_20260715_v3 \
  --rq rq1 rq2
```

Preparation is complete only when both files exist:

```bash
test -f results/experiments/hf_rq12_20260715_v3/protocol.json
test -f results/experiments/hf_rq12_20260715_v3/protocol.sha256
```

#### 3. Prepare RQ3

Run RQ3 separately because checkpoint resume is intentionally limited to an RQ3-only preparation:

```bash
python scripts/run_results.py prepare \
  --run-id realcats_rq3_20260715_v3 \
  --rq rq3
```

Real-CATS preparation can run for hours. Before `protocol.json` is created, it writes `preparation_state.json`, a WAL-backed SQLite event index, per-address checkpoints, and source audit counters. If the command is interrupted, rerun the same command with `--resume` and identical arguments:

```bash
python scripts/run_results.py prepare \
  --run-id realcats_rq3_20260715_v3 \
  --rq rq3 \
  --resume
```

Resume is rejected when the run is already locked, source checksums changed, code or dependency versions changed, command arguments differ, or SQLite integrity validation fails. Do not repair or reuse a rejected index for thesis results. Preserve it under another name and start a new run ID:

```bash
mv results/experiments/realcats_rq3_20260715_v3 \
   results/experiments/realcats_rq3_20260715_v3_abandoned
```

Runs created before protocol version 3 use the superseded in-memory execution path and must not be cited or resumed. This includes the local incomplete `primary_realcats_v1`.

#### 4. Execute a locked run

Execution is allowed only after preparation has written and hashed the protocol:

```bash
python scripts/run_results.py execute \
  results/experiments/hf_rq12_20260715_v3

python scripts/run_results.py execute \
  results/experiments/realcats_rq3_20260715_v3
```

The runner checks protocol, prepared-artifact, source-code, model-parameter, Python, dependency, and execution-backend definitions before loading experiment data. Large event Parquets are externally sorted by DuckDB and streamed to Raphtory as Arrow record batches; pandas is used only for address-level feature matrices, predictions, metrics, and model input. The preflight therefore estimates the sorted Raphtory graph footprint rather than a pandas copy of the raw event table. If it rejects the machine, no metrics are written and the cohort is not trimmed or replaced. Copy the entire locked run directory and exact repository revision to a machine with more memory, recreate the recorded dependency versions, and execute there.

This removes the largest avoidable RAM copy but does not remove the principal compute cost. Raphtory 0.17.0 exposes one local three-node motif call that always computes all 40 roles for every graph node. The runner retains the scientifically usable 24 star and 8 two-node roles and discards the 8 triangle roles after counting; it cannot ask Raphtory to skip those triangle calculations. Each of the seven temporal windows and seven equal-timestamp order variants is still evaluated as locked in the protocol.

Do not lower `--permutations` or `--bootstraps` from their default 10,000 for thesis runs. Smaller values are appropriate only for tests.

#### 5. Inspect artifacts

Each run uses the following layout:

```text
results/experiments/<run-id>/
├── preparation_state.json   Resume definition before protocol locking
├── protocol.json            Locked experimental definition and provenance
├── protocol.sha256          Protocol integrity digest
├── run_log.jsonl            Append-only step status, progress, and elapsed time
├── data/                    Cohorts, folds, and prepared event Parquets
├── features/                Baselines and per-window motif fingerprints
├── predictions/             Held-out address-level scores
├── metrics/                 CV scores, tests, confidence intervals, coverage
├── figures/                 PDF chapter figures
├── tables/                  Long-form CSV chapter tables
└── execution_manifest.json  Completion status and runtime provenance
```

Useful checks while working manually:

```bash
du -sh results/experiments/<run-id>
tail -f results/experiments/<run-id>/run_log.jsonl
jq '.protocol_version, .research_questions, .datasets' \
  results/experiments/<run-id>/protocol.json
jq '.status, .results' \
  results/experiments/<run-id>/execution_manifest.json
```

Every long operation writes a `started`, periodic `progress`, and `completed` or `failed` JSON record. The same concise records are printed to stderr while the command runs. To inspect the most recent completed timings manually:

```bash
jq -r 'select(.status == "completed") | [.stage, .elapsed_seconds] | @tsv' \
  results/experiments/<run-id>/run_log.jsonl | tail -n 30
```

During HF preparation, progress is emitted every 100 Parquet batches for the activity scan and incident-edge extraction. Real-CATS conversion reports every 250 newly checkpointed addresses and times each layer export. Execution times the resource preflight, aggregate query, external sort and graph load, every motif window, cross-validation, model evaluation, and statistical-test block. A `failed` record contains the exception type and message; rerun with `--resume` only for an incomplete RQ3 preparation, following the restrictions above.

Use `main.py` only for exploration. Its `results/results_*.json` files are not thesis findings.

### Exploratory pipeline

Every invocation of `main.py` walks through the same six-stage pipeline, stopping early when `--mode graph` or `--mode motifs` is chosen:

```
1. Load + clean transactions (+ per-address labels, if any)
2. Build NetworkX graph → per-node structural features                  ← `--mode graph` stops here
3. Count per-node temporal motif fingerprints (selected backend)        ← `--mode motifs` stops here
4. Assemble feature matrix per --features {aggregates, motifs, full}
5. Train + evaluate  (supervised)  OR  detect + cluster  (unsupervised)
6. Visualize (skipped with --no-plot)
```

### Quick start

```bash
# Build and inspect the graph only — no motifs, no ML
python main.py --mode graph

# Compute motif fingerprints and plot them — no ML
python main.py --mode motifs

# Full unsupervised pipeline on synthetic data
python main.py --mode unsupervised

# Supervised on the HuggingFace GNN subset (requires huggingface-cli login)
python main.py --mode supervised --dataset-type huggingface-gnn \
  --months 2025-06 2025-07 --features full --no-plot
```

### CLI Options

| Flag | Description | Default |
|---|---|---|
| `--mode` | `graph`, `motifs`, `supervised`, or `unsupervised` | `motifs` |
| `--data` | Path to raw transaction file or a labeled dataset directory | None (synthetic) |
| `--dataset-type` | Labeled dataset: `blte`, `huggingface`, `huggingface-gnn`, or `realcats` | None |
| `--features` | Which features feed stage 5: `aggregates`, `motifs`, `full` | `motifs` |
| `--motif-backend` | Backend for stage 3: `raphtory`, `networkx`, `snap` | `raphtory` |
| `--delta` | Temporal window(s) in seconds; multiple values trigger a δ sweep | `3600` |
| `--supervised-model` | Classifier: `random_forest` or `xgboost` | `random_forest` |
| `--unsupervised-method` | Anomaly method: `isolation_forest` or `dbscan` | `isolation_forest` |
| `--months` | Months to load for `huggingface-gnn` (e.g. `2025-06 2025-07`) | latest |
| `--max-rows` | Cap the transaction count for any dataset (applied at the end of stage 1); the raw `huggingface` stream falls back to 100k if unset | None |
| `--full-labels` | Use `addr_labels_big.csv.zst` (~1M) instead of balanced (~115k) | False |
| `--use-full-edges` | Download the 21.5 GB full-edge file for `huggingface-gnn` | False |
| `--column-map` | Column preset for `--data` files: `bigquery_ethereum`, `etherscan` | None |
| `--n-addresses` | Number of addresses for synthetic data | 100 |
| `--n-transactions` | Number of transactions for synthetic data | 500 |
| `--no-plot` | Skip all visualizations | False |

### Feature strategies

| `--features` | What feeds stage 5 | When to pick it |
|---|---|---|
| `aggregates` | Per-address tx counts + value stats (13 columns) | Fast baseline; no motif signal needed. |
| `motifs` *(default)* | Motif fingerprint only (5 or 40 columns, backend-dependent) | Thesis ablation: "motifs alone". |
| `full` | `motifs` ⋈ `aggregates` ⋈ graph features (degree, clustering, PageRank, net_flow) | Best F1 in practice. |

### Motif backends

| Backend | Install | Dimensions | Algorithm | Status |
|---|---|---|---|---|
| `raphtory` *(default)* | `pip install raphtory==0.17.0` | 40 local roles (24 star + 8 two-node + 8 triangle) | Paranjape et al. 2017, Rust-parallel | Working |
| `networkx` | none (stdlib only) | 5 collapsed types (fan_out, fan_in, cycle, chain, reciprocal) | Pure-Python sliding window | Working |
| `snap` | `pip install snap-stanford` (Python 3.9 only) | 36 (Paranjape three-node) | SNAP.py C++ | Broken, API mismatch |

### Motif types reference

#### NetworkX backend (5 simplified types)

Each labeled column in the fingerprint DataFrame.

| Column | Pattern (within δ) | Shape | Typical signal |
|---|---|---|---|
| `motif_fan_out` | `A→B, A→C, A→D` | One sender, many targets | Airdrops, mass distribution, wash-trade sources |
| `motif_fan_in` | `B→A, C→A, D→A` | Many senders, one target | Fund consolidation, money muling |
| `motif_cycle` | `A→B→…→A` | Closed directed loop | Wash trading, circular transfers |
| `motif_chain` | `A→B→C` | Sequential forwarding | Layering, multi-hop routing |
| `motif_reciprocal` | `A→B and B→A` | Bidirectional two-node exchange | OTC trades, normal dApp interactions |

#### Raphtory backend (40 local roles — Paranjape et al. 2017)

Raphtory 0.17.0 emits 40 local role counts: 24 centred three-node star roles (indices 0–23), 8 two-node roles (24–31), and 8 triangle roles (32–39). Star counts are attributed to the centre, two-node counts to both participants, and triangle counts to all three participants. The Results runner names these roles explicitly and retains only indices 0–31 because an incident-edge cohort does not observe neighbour-to-neighbour edges reliably. This is a feature-selection rule, not a counter optimization: the Raphtory API computes all 40 roles internally.

| Class | Structural pattern (example) | How it's counted |
|---|---|---|
| Three-node star roles (0–23) | `B→A, C→A, C→A` or mixed directions | Centre only |
| Two-node roles (24–31) | Three events between `A` and `B` | Both nodes |
| Three-node triangle roles (32–39) | `A→B, B→C, C→A` | All three nodes |

The exploratory pipeline labels columns `motif_0` … `motif_39`; the Results runner uses stable role names from `src/experiments/constants.py`. For the catalogue, see Paranjape 2017 Figure 2 — [paper PDF](https://cs.stanford.edu/people/jure/pubs/motifs-wsdm17.pdf).

#### SNAP backend (36 types — Paranjape global)

Same six structural classes × six temporal orderings as Raphtory, minus the 4 extra local-perspective indices. Currently non-functional in this repo (kept as a placeholder for future thesis comparison).

### δ sweep

Passing multiple `--delta` values runs stage 3 once per δ, trains a classifier per δ, and prints a comparison table at the end. Each δ writes its own results file:

```bash
python main.py --mode supervised --dataset-type huggingface-gnn \
  --months 2025-01 2025-02 2025-03 2025-04 2025-05 2025-06 2025-07 2025-08 \
  --features full --delta 3600 86400 604800 --no-plot
```

`--features aggregates` ignores `--delta` entirely (aggregates are δ-independent) and runs only once.

Before each training run, a sparsity diagnostic reports whether motifs actually fire at the chosen δ. If most cells are zero, widen δ until the matrix fills in:

```
=== δ = 3600s ===
Counting motif fingerprints via raphtory backend (labeled nodes=114739)...
  Fingerprint shape: (2477, 40)
  Populated cells: 312/99080 (0.32%)
  Addresses with any motif: 187/2477
  Motif types that ever fire: 8/40
```

After the sweep finishes:

```
=== δ sweep summary ===
       delta          f1   precision      recall     roc_auc      pr_auc
        3600      0.2847      0.7321      0.1767      0.5728      0.6355
       86400      0.6412      0.5880      0.7048      0.7312      0.7108
      604800      0.7430      0.6904      0.8053      0.8105      0.7922
```

### Modes in detail

**`graph`** — stops after stage 2
- Loads data, builds the NetworkX graph, computes graph statistics.
- Saves the pickled graph to `results/<dataset>_graph.pkl` for later reuse (`pickle.load(open(path, "rb"))`).
- If `--no-plot` is absent, renders the transaction network.
- Skips PageRank / clustering coefficient computation (those are only needed by later stages).
- Useful for: inspecting graph structure, exporting to external graph tools, or caching a heavy graph build before iterating on later stages.

For fast iteration on huge months, combine with `--max-rows` to truncate the tx frame before stage 2:

```bash
# ~1 second on warm cache — only builds a 1000-edge graph
python main.py --mode graph --dataset-type huggingface-gnn --months 2025-07 --max-rows 1000
```

**`motifs`** — stops after stage 3 (default)
- Runs stages 1–3. No classifier, no anomaly detector.
- Saves the fingerprint DataFrame to `results/<dataset>_fingerprints_<backend>_delta<N>.parquet` (one file per δ if multiple are passed).
- If `--no-plot`: exits silently after saving. Otherwise plots the transaction network with motif participants highlighted, the motif distribution bar chart, and the fingerprint heatmap. See [Motif types reference](#motif-types-reference) to decode the columns.
- Useful for: exploring what motifs fire on a dataset before committing to a full training run.

**`supervised`**
- Stage 1 must produce per-address labels (`blte`, `huggingface`, `huggingface-gnn`, `realcats`; raw `--data` files are rejected unless you supply labels another way).
- Stage 5 runs an 80/20 stratified split, 5-fold CV, fits RF or XGBoost with automatic minority-class re-weighting (`class_weight='balanced'` / `scale_pos_weight`).
- Writes `results/results_<tag>.json` with accuracy, precision, recall, F1, ROC-AUC, PR-AUC.
- Plots ROC/PR curves + confusion matrix (unless `--no-plot`).

**`unsupervised`**
- Labels are ignored (works on any data source, including synthetic).
- Stage 5 runs Isolation Forest or DBSCAN for anomalies, plus DBSCAN clustering to surface coordinated groups.
- Writes `results/results_<tag>.json` with anomaly count, cluster count, and the top-10 coordinated clusters.
- Plots the transaction network with anomalies highlighted, motif distribution, fingerprint heatmap, and a t-SNE projection of clusters.

### Artifact filename schema

Every artifact lives under `results/`:

| Mode | File | Format |
|---|---|---|
| `graph` | `<dataset>_graph.pkl` | pickled NetworkX graph |
| `motifs` | `<dataset>_fingerprints_<backend>_delta<N>.parquet` | per-δ motif fingerprint matrix |
| `supervised` / `unsupervised` | `results_<tag>.json` where tag = `{dataset}_{mode}_{features}[_backend_deltaN]` | metrics dict |

Examples:
- `results/synthetic_graph.pkl`
- `results/huggingface_gnn_fingerprints_raphtory_delta3600.parquet`
- `results/results_synthetic_unsupervised_motifs_raphtory_delta3600.json`
- `results/results_huggingface_gnn_supervised_full_raphtory_delta86400.json`

## Exploratory Configuration

The exploratory defaults are centralized in `config.py`. The Results runner does not import these values; it locks its own protocol and verified 32-role feature set.

| Parameter | Default | Description |
|---|---|---|
| `DEFAULT_DELTA` | 3600 | Temporal motif window (seconds) |
| `DEFAULT_DELTAS` | [3600, 86400, 604800] | Multi-delta sweep values |
| `NUM_MOTIF_TYPES` | 36 | Paranjape catalogue size |
| `DEFAULT_MOTIF_THRESHOLD` | 5 | Baseline detection threshold |
| `DEFAULT_SUPERVISED_MODEL` | random_forest | Supervised classifier |
| `DEFAULT_UNSUPERVISED_METHOD` | isolation_forest | Anomaly detector |
| `DEFAULT_MOTIF_BACKEND` | raphtory | Motif counter for `--features motifs` |
| `RANDOM_STATE` | 42 | Reproducibility seed |
| `TEST_SIZE` | 0.2 | Train/test split ratio |

Column mappings for BigQuery and Etherscan data formats are also defined in `config.py` under `COLUMN_MAPS`.

## Data Sources

The pipeline accepts transaction data from multiple sources:

- **Local files** — CSV, JSON, or Parquet with columns: `from_address`, `to_address`, `value`, `timestamp`
- **Google BigQuery** — Export from `bigquery-public-data.crypto_ethereum.transactions` and load as Parquet
- **Etherscan API** — Use `--column-map etherscan` to auto-map column names
- **BLTE** — Benchmark Labeled Transactions of Ethereum Network (~71k raw transactions with scam labels + 73k pre-extracted address features). Loaded via `load_blte_dataset("data/raw/ethereum_dataset_labelled")`
- **Real-CATS** — 28,591 labeled Ethereum addresses (12,572 criminal / 16,019 benign) with 10.5 M raw normal ETH transactions spanning 2015–2024 (Kaggle release). Wired into the pipeline as `--dataset-type realcats --data data/raw/realcats_transactions`. Requires prior extraction — see [Real-CATS dataset](#real-cats-dataset) setup below. The GitHub release also ships pre-extracted address features (52 columns) accessible via `load_realcats_dataset("data/raw/realcats_dataset")`.
- **Ethereum Fraud Dataset** — `fesevu/ethereum_fraud_dataset_by_activity` from HuggingFace. Three access modes:
  - **`huggingface-gnn` (recommended)** — pre-filtered graph subset at `gnn_dataset/`, sharded by month. Use `--months 2023-04 2023-05` to pick ranges. Edges use hashed node IDs and labels live in the same space, so supervision works out of the box.
  - **`huggingface`** — raw 1.6B-row parquet stream with separate address label file. Use `--max-rows N` to cap the download; expect low overlap between random samples and labeled addresses.
  - **`--full-labels`** — use the 1M-address `addr_labels_big.csv.zst` instead of the 115k balanced file.
  - Requires `huggingface-cli login`.
- **Synthetic** — Auto-generated when no `--data` is provided (includes injected fan-out, fan-in, and cycle patterns)
