# Coordinated Malicious Behaviour Detection in Blockchain Networks Using Temporal Motifs

Bachelor thesis project implementing a pipeline for detecting coordinated malicious behaviour in Ethereum transaction networks through temporal motif analysis.

The system models blockchain transactions as a temporal directed multigraph, extracts structural and temporal motif features per address, and applies supervised and unsupervised ML methods. `main.py` is the exploratory workflow. The locked, leakage-controlled thesis experiments are run through `scripts/run_results.py` and `src/experiments/`.

## Project Structure

```
bachelor_thesis/
├── main.py                        Entry point — orchestrates the pipeline
├── config.py                      Central configuration and parameters
├── requirements.txt               Python dependencies
├── requirements-results.txt       Exact protocol-v5 numerical environment
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
├── scripts/run_results.py         Prepare/execute/resume/verify Results entry point
├── scripts/results_to_latex.py    Verified-run tables/macros for the thesis
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
| CPU | 16 modern physical cores | 20-32 physical cores | A direct run receives the selected thread budget. The coordinated launcher splits logical CPUs between HF and Real-CATS, with an odd extra thread assigned to HF. |
| RAM | 32 GB for an admitted sequential run | 64 GB sequential; 128 GB for coordinated parallel execution | Parallel admission sums the HF peak and the largest sequential Real-CATS layer peak. Do not treat swap as usable graph memory. The read-only preflight is authoritative. |
| Free local SSD for RQ1/RQ2 | 100 GB | 150 GB | Includes the approximately 21 GB HF cache, prepared incident edges, DuckDB spill space, fingerprints, and outputs. |
| Free local SSD for RQ3/RQ4 | 120 GB | 200 GB | The Real-CATS ZIP is approximately 12 GB and lists about 72 GB of uncompressed per-address JSON in total; RQ3/RQ4 share the criminal-address subset and motif checkpoints. Peak usage includes SQLite, layer Parquets, and sort spill. |
| Free local SSD for all retained runs | 250 GB | 350 GB or more on a 500 GB volume | Assumes both raw sources and both locked run directories remain available. Keep at least 20% of the filesystem free for temporary work. |
| GPU | Not required | Not required | The locked estimators use CPU implementations. |
| Network | Required only for initial downloads | Not required during prepare/execute | HuggingFace "streaming" means bounded reads from the locally cached Parquet, not repeated remote access. |

Cloud vendors usually advertise SMT threads as vCPUs rather than physical
cores. Avoid heavily oversubscribed burst instances.

The supervisor-provided server is suitable for the full run:

| Provided hardware | Pipeline use |
|---|---|
| 20 CPU cores | Used fully. The one-command launcher normally runs HF and Real-CATS concurrently with 10 Raphtory threads each. |
| 48 GPU cores, approximately RTX 5070 class | Not used. Raphtory motif counting is CPU-only, and enabling GPU execution for one small XGBoost robustness fit would not materially reduce total runtime. |
| 128 GB shared system RAM/VRAM | Suitable for coordinated execution when preflight passes. Because memory is shared, do not run a separate GPU workload during the thesis run. |
| 4 TB storage | More than sufficient. Keep at least 20% free because the locked admission rule reserves 20% per filesystem for staging and external sorts. |

The 32 GB RAM figure is a floor, not a guarantee. As a concrete sizing anchor,
the existing local HF preparation contains about 140 million incident edges;
the current conservative preflight projects approximately 14.1 GiB peak and
rejects a 16 GB machine because execution would exceed the 85% memory safety
limit. A newly prepared protocol may select a different edge volume. Always run
the preflight on the intended server, and use its decision instead of the table
when they differ.

Recommended server profile for the complete thesis evaluation:

```text
Linux x86-64 recommended; otherwise verify all pinned wheels first
20-32 physical CPU cores
64-128 GB RAM
500 GB local NVMe SSD
Python and dependency versions exactly matching protocol.json
No GPU required
```

RQ1/RQ2 and RQ3/RQ4 may run sequentially or through the coordinated parallel
launcher. Do not start two ordinary `execute` commands manually: each assumes
exclusive host ownership. `execute-parallel` allocates disjoint thread budgets,
checks their combined RAM and per-device disk demand, and owns one host lock
while both workers run.

### Approximate full-run time

The following ranges are capacity-planning estimates, not guaranteed runtimes.
They assume that the datasets are already downloaded, local NVMe storage, no
other heavy workload, and a 20-32-core server with 64-128 GB RAM. The first
complete protocol-v5 run should be used to replace these estimates with the
measured durations from `run_log.jsonl`.

| Stage | Approximate time | Main cost |
|---|---:|---|
| Synthetic validation | Under 5 minutes | Small exact Raphtory fixtures and artifact checks |
| Hybrid internal smoke run | Target under 1 hour; 76.4 seconds observed locally | Bounded real HF and Real-CATS data, full BLTE, all RQ orchestration, and synthetic validation |
| HuggingFace RQ1/RQ2 preparation | 10-45 minutes | Three full local reads of the approximately 21 GB edge Parquet: activity, extraction, and SHA-256 |
| Real-CATS RQ3/RQ4 preparation | 3-12 hours | ZIP JSON parsing, durable SQLite deduplication, checkpoint commits, and five Parquet exports |
| HuggingFace RQ1/RQ2 execution | 2-7 days | Seven graph builds and 49 full Raphtory motif calls: seven order variants times seven delta values |
| Real-CATS RQ3/RQ4 execution | 2-7 days | Five layers processed for seven order variants and seven delta values, followed by shared exact-label inference |
| Both primary executions, sequentially | About 4-14 days | Conservative option when the server cannot admit both graphs together |
| Both primary executions, coordinated in parallel on 20 CPU cores | About 3-10 days | Requires combined admission; storage bandwidth contention and per-delta variation can reduce the ideal speed-up |

On an 8-core laptop, preparation remains feasible when storage is sufficient,
but execution can take roughly one to three weeks and may fail the RAM
preflight. A 20-core, 128 GB RAM server is a substantially more predictable
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

# Install the pinned thesis-results environment
pip install -r requirements-results.txt
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

The Results runner reads all five transaction layers directly from the ZIP and globally deduplicates overlapping address histories. Do not run `scripts/extract_realcats.py` for RQ3/RQ4; that script extracts only normal transactions for the exploratory `main.py` workflow.

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

### Professor/server: complete thesis run with one command

After installing `requirements-results.txt` and placing the local Hugging Face
snapshot and Real-CATS files as described above, the complete citable workflow
is:

```bash
venv/bin/python scripts/run_results.py run-thesis \
  --run-id thesis_final_20260731 \
  --threads 20
```

`--run-id` is optional for this command; if omitted, a UTC timestamped protocol-v5
ID is generated and printed. Supplying it explicitly makes restart and handover
easier. The command performs all of the following without manual intervention:

1. validates that both source datasets exist locally;
2. prepares and locks separate `thesis_final_20260731_hf` and
   `thesis_final_20260731_realcats` roots;
3. runs read-only combined RAM and disk admission;
4. uses coordinated parallel execution when admitted, splitting 20 threads as
   10 for HF and 10 for Real-CATS;
5. falls back to sequential 20-thread execution if only sequential memory
   admission succeeds;
6. executes synthetic validation, RQ1/RQ2, and the shared RQ3/RQ4 analysis;
7. verifies both roots with the thesis-citable requirement before returning
   success.

The terminal displays `tqdm` progress bars with elapsed time, throughput, and
rolling ETA for HF scan batches, Real-CATS address conversion, and every
seven-window motif sweep. ETA becomes informative only after at least one
comparable unit has completed; different deltas and transaction layers can have
substantially different costs. `run_log.jsonl` remains the authoritative,
fsynced audit log and records five-minute heartbeats during a single long native
Raphtory call.

For an SSH session, run the command inside `tmux` or another persistent job
session. After an interruption, use the same ID and hardware allocation:

```bash
venv/bin/python scripts/run_results.py run-thesis \
  --run-id thesis_final_20260731 \
  --threads 20 \
  --resume
```

Committed checkpoints are verified and reused. The command resumes the
incomplete root directly if its peer already completed. It never overwrites an
existing result or silently replaces an invalid checkpoint.

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
- the complete small BLTE release and deterministic fixtures for all 40
  Raphtory catalogue positions (32 retained roles and 8 excluded triangles);
- three delta values (5 minutes, 1 hour, and 1 week), canonical/reverse/one
  seeded timestamp order, five-fold CV, and 100 permutation/bootstrap draws;
- baseline and motif features, Random Forest, XGBoost, multinomial logistic
  regression, statistical-test code, figures, tables, predictions, protocol
  verification, and the final manifest.

On the current 8-core M1 Pro with 16 GB RAM, the final protocol-v5 verification
run completed in 76.4 seconds, created 508 filesystem entries including stage
payloads and their public hard links, and occupied 65 MB. Treat that as a local
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
Smoke admission keeps a fixed 2 GiB filesystem reserve; thesis and development
profiles retain the full-run reserve of 20% of the filesystem or 20 GiB,
whichever is larger.

The 60-minute limit is an SLA rather than a force-kill. If the bounded run
exceeds it, the current native operation is allowed to finish so DuckDB,
SQLite, Raphtory, and Parquet state is not interrupted. The command writes
`sla_met: false`, preserves all diagnostics, and exits with status 2 after
completion. Pipeline failures exit nonzero and write their exception details to
`smoke_summary.json`. Smoke runs are immutable; use a new run ID instead of
resuming or overwriting one.

### Thesis Results experiments

The evidence-producing workflow is separate from `main.py` and uses protocol
version 5. Preparation locks the scientific definition and prepared data.
Execution creates an immutable operational definition, commits resumable motif
checkpoints and result stages, and finishes with a verifiable integrity chain.
`verify` is read-only and is the final authority for integrity and citability.

Protocol v5 reflects the four current research questions. RQ2 reports the δ
with the highest mean combined-model average precision on the persisted
development CV folds; exact numerical ties choose the smaller δ. The shorter
window within one standard error is retained and reported only as a secondary
parsimony analysis. RQ3 tests global multiclass category separability above
chance. RQ4 selects global and category-specific windows on development data,
then compares category-specific scores against the global-window scores on the
untouched test set using an omnibus paired bootstrap plus Benjamini-Hochberg
adjusted per-category comparisons.

A run ID is only the directory name below the selected results root. It has no
statistical meaning. Use a unique, descriptive value such as
`hf_rq12_20260721_v5`; omitting `--run-id` from `prepare` or `smoke` is an error,
and reusing an existing ID is rejected.

#### 1. Install and run the bounded smoke profile

Use the exact numerical environment for thesis work:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements-results.txt
python scripts/run_results.py smoke \
  --run-id smoke_$(date +%Y%m%d_%H%M%S) \
  --sla-minutes 60
python scripts/run_results.py verify results/smoke/<run-id>
```

Normal verification returns zero when schemas and integrity are valid even
though smoke is deliberately non-citable. This command must fail:

```bash
python scripts/run_results.py verify \
  results/smoke/<run-id> --require-citable
```

#### 2. Prepare the primary runs

Prepare HF and Real-CATS as separate roots so they can run sequentially or under
the coordinated launcher. The `thesis` profile is the default and fixes all
10,000 resamples. `development` may only reduce permutation and bootstrap
counts and can never become citable.

```bash
python scripts/run_results.py prepare \
  --run-id hf_rq12_20260721_v5 \
  --profile thesis \
  --rq rq1 rq2

python scripts/run_results.py prepare \
  --run-id realcats_rq34_20260721_v5 \
  --profile thesis \
  --rq rq3 rq4
```

HF preparation scans the complete 380,566,661-edge local source for activity
and incident-edge extraction. Only the target-incident subgraph is used for
motif counting. “Streaming” means bounded local Parquet batches, not remote
network access. Downloading the dataset first is therefore required and avoids
network instability, but it does not remove the full local scans. Split and CV
assignments are group-aware across both HF cohorts, so a known campaign cannot
land on different sides through separate cohort assignment.

Real-CATS preparation exports all five transaction layers. Empty layers still
receive schema-correct Parquets. Identical duplicate labels are audited and
collapsed; conflicting labels abort preparation. An interrupted RQ3/RQ4 preparation
before `protocol.json` exists may be resumed with the identical command plus
`--resume`. This preparation resume is separate from execution resume.

Behaviour categories are exact non-`Other` CE.tsv labels with at least 100 active
addresses; labels are not merged after inspecting results. Real-CATS does not
provide a separate ground-truth coordination flag. The pipeline therefore
tests whether dataset-defined malicious behaviour categories exhibit
coordination-related temporal fingerprints; it must not be described as a
directly supervised coordination classifier.

#### 3. Run read-only admission

Preflight creates no lock, telemetry, attempt, or temporary file:

```bash
python scripts/run_results.py preflight \
  results/experiments/hf_rq12_20260721_v5

python scripts/run_results.py preflight \
  results/experiments/hf_rq12_20260721_v5 \
  results/experiments/realcats_rq34_20260721_v5
```

One root checks sequential admission. Multiple roots check concurrent RAM and
sum scratch/staging demand by filesystem device, applying the disk reserve once
per device. The report is a conservative recorded budget, not an OS-enforced
Raphtory memory cap. A rejection never trims the cohort.

#### 4. Execute directly or in parallel

For sequential execution, optionally set a fixed thread count:

```bash
python scripts/run_results.py execute \
  results/experiments/hf_rq12_20260721_v5 --threads 20
python scripts/run_results.py execute \
  results/experiments/realcats_rq34_20260721_v5 --threads 20
```

For coordinated same-host execution:

```bash
python scripts/run_results.py execute-parallel \
  results/experiments/hf_rq12_20260721_v5 \
  results/experiments/realcats_rq34_20260721_v5 \
  --threads 20
```

Fresh parallel execution splits available logical CPUs, assigning an odd extra
thread to HF. The coordinator holds the per-user host lock and passes proof of
that lock to both workers. Each worker has a separate run lock and attempt
scratch directory. `--threads 20` fixes the combined budget before the 10/10
split. Raphtory receives each worker's concrete thread count. If one fails, the
other is allowed to finish.

Every expensive motif call is committed separately: HF uses one checkpoint per
ordering and delta; Real-CATS uses one per layer, ordering, and delta. After an
interruption, resume a direct run with:

```bash
python scripts/run_results.py execute \
  results/experiments/hf_rq12_20260721_v5 --resume
```

The locked thread count is reused. Supplying `--threads` on resume is optional;
if supplied, it must match exactly. Parallel resume is allowed only when both
roots are interrupted and the host can provide their combined locked threads:

```bash
python scripts/run_results.py execute-parallel \
  results/experiments/hf_rq12_20260721_v5 \
  results/experiments/realcats_rq34_20260721_v5 \
  --threads 20 \
  --resume
```

Mixed fresh/interrupted/completed pairs are rejected. Resume the one incomplete
root directly when its peer is complete. A committed artifact that fails its
hash or schema check is never replaced; start a new run rather than repairing it.

BLTE is optional and separate from the primary roots and their citability:

```bash
python scripts/run_results.py prepare \
  --run-id blte_rq1_20260721_v5 --profile thesis --rq blte
python scripts/run_results.py execute \
  results/experiments/blte_rq1_20260721_v5
```

#### 5. Verify and inspect

Final thesis acceptance requires:

```bash
python scripts/run_results.py verify \
  results/experiments/hf_rq12_20260721_v5 --require-citable
python scripts/run_results.py verify \
  results/experiments/realcats_rq34_20260721_v5 --require-citable
```

Protocol v1-v4 runs support limited read-only inspection only. They cannot be
executed, resumed, or accepted as citable.

#### 6. Generate the thesis tables

After both primary roots pass `--require-citable`, generate the LaTeX inputs
from those exact directories. Add the optional BLTE root only after it has also
completed and verified:

```bash
python scripts/results_to_latex.py writing/generated \
  results/experiments/hf_rq12_20260721_v5 \
  results/experiments/realcats_rq34_20260721_v5 \
  results/experiments/blte_rq1_20260721_v5
```

The exporter performs read-only verification of every completed input. It
never takes metrics from an unexecuted preparation and never combines cohort
metadata with results owned by another run. Non-citable or incomplete inputs
produce a visible preliminary banner; they are suitable only for checking the
chapter layout. Upload `writing/generated/*.tex` and the referenced PDFs from
the verified runs' `figures/` directories to Overleaf.

```text
results/experiments/<run-id>/
├── preparation_state.json
├── protocol.json / protocol.sha256
├── execution_definition.json / execution_definition.sha256
├── attempts/<attempt-id>/
│   ├── started.json and completed.json or failed.json
│   ├── scratch/duckdb/
│   └── staging/
├── checkpoints/<checkpoint-id>/
│   ├── counts.parquet
│   └── metadata.json
├── stages/<stage-id>/
│   ├── payload/
│   └── stage_manifest.json
├── features/ predictions/ metrics/ figures/ tables/
├── result_artifacts.json / result_artifacts.sha256
├── execution_manifest.json / execution_manifest.sha256
└── run_log.jsonl
```

`stages/` is the immutable source of truth. Public result directories contain
no-overwrite hard links with the same filenames used by earlier pipeline
versions. The integrity order is protocol/prepared data, execution definition,
checkpoints and stages, result inventory, then execution manifest. Inventory
and manifest do not hash themselves.

Operational monitoring:

```bash
tail -f results/experiments/<run-id>/run_log.jsonl
jq -r 'select(.status == "completed") | [.stage, .elapsed_seconds] | @tsv' \
  results/experiments/<run-id>/run_log.jsonl | tail -n 30
du -sh results/experiments/<run-id>
```

Long native calls emit a fsynced heartbeat every five minutes with elapsed
time, thread count, free disk, and latest checkpoint. A heartbeat proves
liveness, not percentage complete. Interactive runs also show `tqdm` elapsed
time and ETA for bounded scan, conversion, and motif-window units. Use
`main.py` only for exploration; its `results/results_*.json` files are not
thesis findings.

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
