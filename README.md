# Coordinated Malicious Behaviour Detection in Blockchain Networks Using Temporal Motifs

Bachelor thesis project implementing a pipeline for detecting coordinated malicious behaviour in Ethereum transaction networks through temporal motif analysis.

The system models blockchain transactions as a temporal directed multigraph, extracts structural and temporal motif features per address, and applies both supervised and unsupervised ML methods to flag suspicious coordinated activity such as wash trading, layering, and airdrop farming.

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
└── tests/                         Unit tests (placeholder — none yet)
```

## Setup

### Prerequisites

- Python 3.10+ (Raphtory, the default motif backend, requires 3.10 or newer)

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

Download `real-cats.zip` from Kaggle (`lvd312393/real-cats`) into `data/raw/realcats_transactions/`, then run:

```bash
python scripts/extract_realcats.py
```

This selectively extracts `normal_transactions.json` for all 28,591 labeled addresses (already-extracted files are skipped; ~2,100 zero-transaction addresses absent from the archive are silently skipped). Once complete:

```bash
python main.py --mode supervised --dataset-type realcats \
  --data data/raw/realcats_transactions --features full --no-plot
```

### Optional: SNAP.py backend

The `snap` motif backend (Paranjape et al.'s original C++ implementation via SNAP.py) is kept as a placeholder but is currently non-functional — the installed `snap-stanford` 6.0 does not expose the API this project was written against. SNAP.py also has no PyPI wheels for Python ≥ 3.10, so using it at all would require a separate Python 3.9 venv.

Use `--motif-backend raphtory` (default) or `--motif-backend networkx` instead. The SNAP code path raises a clear `RuntimeError` if invoked.

## Usage

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
| `raphtory` *(default)* | `pip install raphtory` | 40 (36 three-node + 4 two-node) | Paranjape et al. 2017, Rust-parallel | Working |
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

#### Raphtory backend (40 types — Paranjape et al. 2017)

The 40-dim vector from `local_temporal_three_node_motifs` groups into **six structural classes** (Paranjape, Benson & Leskovec, WSDM 2017). Each class has 6 temporal orderings (which edge came first, second, third), so 6 × 6 = **36 "global" motifs**. Four extra indices appear in the **local per-node** catalogue because two nodes in a same-direction two-node triple see it from different perspectives — hence 36 + 4 = **40**.

| Class | Structural pattern (example) | How it's counted |
|---|---|---|
| Two-node, same direction | `A→B, A→B, A→B` | Both nodes record it |
| Two-node, mixed directions | `A→B, B→A, A→B` | Both nodes record it |
| Three-node star — all incoming | `B→A, C→A, C→A` (center receives) | Centre only |
| Three-node star — all outgoing | `A→B, A→C, A→C` (center sends) | Centre only |
| Three-node star — mixed | `A→B, C→A, A→C` | Centre only |
| Three-node triangle | `A→B, B→C, C→A` (closed cycle) | All three nodes |

Fingerprint columns are labelled `motif_0` … `motif_39` in the order Raphtory emits them. For the exact per-index visual catalogue, see Paranjape 2017 Figure 2 — [paper PDF](https://cs.stanford.edu/people/jure/pubs/motifs-wsdm17.pdf).

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

## Configuration

All defaults are centralized in `config.py`:

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
