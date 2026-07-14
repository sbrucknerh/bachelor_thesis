# Architecture

This document describes the runtime architecture of the temporal-motif coordinated-behaviour
detection pipeline. Diagrams are written in [Mermaid](https://mermaid.js.org/) and render
directly on GitHub.

For *what* each flag/mode does, see [`README.md`](README.md). This document covers *how the
code is wired together*.

Rendered copies of every diagram (SVG / PNG / PDF) live in [`diagrams/`](diagrams/).

---

## 0. System overview (all-in-one)

A single figure folding together the data sources, the six-stage spine, the stage-3 backend
strategy, the stage-5 supervised/unsupervised branch, and the `results/` outputs. The sections
below break each part out in more detail.

```mermaid
flowchart TD
    subgraph SRC["Data sources — stage 1 selects one"]
        direction LR
        SYN["Synthetic<br/><i>synthetic_data.py</i>"]
        FILE["Local file<br/>CSV / JSON / Parquet"]
        BLTE["BLTE<br/>labeled"]
        HF["HuggingFace<br/>raw + GNN"]
    end

    CFG["config.py<br/>constants · COLUMN_MAPS"]

    SRC --> S1["<b>Stage 1</b> · Load &amp; clean + labels + tx aggregates<br/><i>data_loading.py</i>"]
    S1 --> S2["<b>Stage 2</b> · Build MultiDiGraph + node features<br/><i>graph_construction.py</i>"]
    S2 --> S3

    subgraph S3G["Stage 3 · Temporal motif fingerprints (per node, per δ)"]
        direction LR
        S3["get_backend(--motif-backend)<br/><i>motifs/__init__.py</i>"]
        S3 --> NX["networkx<br/>5 types"]
        S3 --> RAPH["raphtory<br/>40 types · default"]
        S3 --> SNAP["snap<br/>broken"]
    end

    S3G --> S4["<b>Stage 4</b> · Assemble matrix · aggregates / motifs / full<br/><i>_stage4_assemble_features</i>"]
    S4 --> MODE{"--mode"}

    MODE -->|supervised| SUP["<b>Stage 5a</b> · train RandomForest / XGBoost<br/>→ evaluate_classifier<br/><i>detection.py · evaluation.py</i>"]
    MODE -->|unsupervised| UNS["<b>Stage 5b</b> · IsolationForest / DBSCAN<br/>+ cluster + coordinated groups<br/><i>detection.py · feature_engineering.py</i>"]

    SUP --> S6["<b>Stage 6</b> · Visualize (unless --no-plot)<br/><i>visualization.py</i>"]
    UNS --> S6

    S2 -.->|"--mode graph"| OG[("results/*_graph.pkl")]
    S3G -.->|"--mode motifs"| OFP[("results/*_fingerprints_*.parquet")]
    SUP --> OJ[("results/results_*.json")]
    UNS --> OJ
    S6 --> OFIG[["matplotlib figures"]]

    CFG -.-> S1
    CFG -.-> SUP
    CFG -.-> UNS
    CFG -.-> S6

    classDef out fill:#e2f0d9,stroke:#548235;
    classDef cfg fill:#f2dede,stroke:#a94442;
    class OG,OFP,OJ,OFIG out;
    class CFG cfg;
```

---

## 1. Pipeline data flow

`main.py:run_pipeline()` walks a fixed six-stage pipeline. `--mode graph` and `--mode motifs`
exit early; `supervised`/`unsupervised` run all six stages. With multiple `--delta` values,
stages 3–5 repeat once per δ (the δ sweep).

```mermaid
flowchart TD
    subgraph SRC["Data sources (stage 1 picks one)"]
        direction LR
        SYN["Synthetic generator<br/><i>synthetic_data.py</i>"]
        FILE["Local file<br/>CSV / JSON / Parquet"]
        BLTE["BLTE<br/>labeled dataset"]
        HF["HuggingFace<br/>raw stream + GNN subset"]
    end

    SRC --> S1

    S1["<b>Stage 1</b> · Load &amp; clean transactions<br/>+ per-address labels &amp; tx aggregates<br/><i>data_loading.py · _aggregate_tx_features</i>"]
    S2["<b>Stage 2</b> · Build MultiDiGraph<br/>+ per-node structural features<br/><i>graph_construction.py</i>"]
    S3["<b>Stage 3</b> · Temporal motif fingerprints<br/>per node, per δ<br/><i>motifs/ via get_backend()</i>"]
    S4["<b>Stage 4</b> · Assemble feature matrix<br/>aggregates · motifs · full<br/><i>_stage4_assemble_features</i>"]
    S5["<b>Stage 5</b> · Train / detect<br/><i>detection.py + evaluation.py</i>"]
    S6["<b>Stage 6</b> · Visualize<br/><i>visualization.py</i>"]

    S1 --> S2 --> S3 --> S4 --> S5 --> S6

    S2 -.->|"--mode graph stops here"| G[("results/&lt;ds&gt;_graph.pkl")]
    S3 -.->|"--mode motifs stops here"| FP[("results/&lt;ds&gt;_fingerprints_&lt;backend&gt;_delta&lt;N&gt;.parquet")]
    S5 --> J[("results/results_&lt;tag&gt;.json")]
    S6 --> FIG[["matplotlib figures"]]

    classDef exit fill:#fff3cd,stroke:#d39e00;
    classDef out fill:#e2f0d9,stroke:#548235;
    class G,FP,J,FIG out;
```

---

## 2. Module dependency graph

`main.py` imports the data/graph/synthetic modules at the top level; everything else is
imported lazily inside the stage that needs it (so a `--mode graph` run never imports sklearn
or raphtory). All leaf modules read their constants from `config.py`.

```mermaid
flowchart LR
    MAIN["<b>main.py</b><br/>orchestrator + CLI"]
    CFG["config.py<br/>constants · COLUMN_MAPS"]

    subgraph CORE["src/ core modules"]
        DL["data_loading"]
        SD["synthetic_data"]
        GC["graph_construction"]
        FE["feature_engineering"]
        DET["detection"]
        EV["evaluation"]
        VIZ["visualization"]
        TM["temporal_motifs<br/><i>re-export shim</i>"]
    end

    subgraph MOT["src/motifs/ — backend strategy"]
        DISP["__init__<br/>get_backend(name)"]
        BASE["base.py<br/><i>contract docstring</i>"]
        NX["networkx_backend<br/>5 collapsed types"]
        RAPH["raphtory_backend<br/>40 types (default)"]
        SNAP["snap_backend<br/>broken placeholder"]
    end

    MAIN --> DL
    MAIN --> SD
    MAIN --> GC
    MAIN -.->|"stage 3"| DISP
    MAIN -.->|"stage 5 unsup."| FE
    MAIN -.->|"stage 5"| DET
    MAIN -.->|"stage 5/6"| EV
    MAIN -.->|"stage 6"| VIZ
    MAIN -.->|"stage 6"| TM

    DISP --> NX
    DISP --> RAPH
    DISP --> SNAP
    TM --> NX
    NX --> GC
    NX -.->|implements| BASE
    RAPH -.->|implements| BASE
    SNAP -.->|implements| BASE

    DL --> CFG
    SD --> CFG
    DET --> CFG
    EV --> CFG
    VIZ --> CFG

    classDef cfg fill:#f2dede,stroke:#a94442;
    class CFG cfg;
```

**Note:** `feature_engineering.build_feature_matrix()` exists but is **not** on the main path —
stage 4 inlines the join itself. Only `feature_engineering.normalize_features()` is called
(by the unsupervised branch of stage 5).

---

## 3. Motif backend contract (strategy pattern)

The three backends are interchangeable behind a single function contract, selected at runtime
by `--motif-backend`. `get_backend(name)` returns the matching `count_per_node_motifs`.

```mermaid
flowchart TD
    CALL["_stage3_count_motifs(txs, δ, backend, labels)"] --> GB["get_backend(backend)"]
    GB --> CONTRACT["count_per_node_motifs(df, delta, node_filter)<br/>→ DataFrame indexed by address"]

    CONTRACT --> NXr["networkx_backend<br/>pure-Python sliding window<br/>cols: motif_fan_out … motif_reciprocal (5)"]
    CONTRACT --> RAPHr["raphtory_backend<br/>Rust local_temporal_three_node_motifs<br/>cols: motif_0 … motif_39 (40)"]
    CONTRACT --> SNAPr["snap_backend<br/>raises RuntimeError<br/>(would be motif_0 … motif_35)"]
```

---

## 4. Stage 5 — supervised vs. unsupervised branch

```mermaid
flowchart TD
    X["Feature matrix X<br/>(rows = addresses)"] --> MODE{"--mode"}

    MODE -->|supervised| SUP["train_supervised_model<br/>RandomForest / XGBoost<br/>80/20 stratified split · 5-fold CV<br/>minority-class re-weighting"]
    SUP --> EVAL["evaluate_classifier<br/>acc · precision · recall · F1<br/>ROC-AUC · PR-AUC"]

    MODE -->|unsupervised| UNS["normalize_features (StandardScaler)<br/>↓<br/>detect_anomalies (IsolationForest / DBSCAN)<br/>cluster_addresses (DBSCAN)<br/>find_coordinated_clusters"]

    EVAL --> JSON[("results_&lt;tag&gt;.json")]
    UNS --> JSON
```

---

## Legend

| Notation | Meaning |
|---|---|
| solid arrow | top-level import / direct call |
| dashed arrow | lazy import (inside a stage) or "implements contract" |
| cylinder `( )` | file artifact written to `results/` |
| `<i>italic</i>` | the module/function that implements the step |

## External libraries by module

| Module | Key external deps |
|---|---|
| `graph_construction`, `visualization` | networkx |
| `data_loading` | pandas, `datasets` / `huggingface_hub`, zstandard |
| `motifs/raphtory_backend` | raphtory (Rust) |
| `detection`, `feature_engineering` | scikit-learn |
| `evaluation`, `visualization` | matplotlib, scikit-learn (`metrics`, `manifold`) |
