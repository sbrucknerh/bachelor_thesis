# Protocol v4 thesis changes for Overleaf

This file lists the edits already applied to `writing/thesis.tex`. Use the
section names, not local line numbers, when transferring them to Overleaf because
line numbers will move as the thesis changes.

## Scope deliberately left unchanged

- The three research questions in the Introduction.
- The exact-label RQ3 definition and minimum class size.
- Cohort construction, metrics, window selection, statistical tests, and
  ordering-robustness decisions.
- No thesis-profile numerical result or hypothesis decision was invented.

## Results chapter integration

The Results skeleton now uses `\input{generated/...}` files produced by
`scripts/results_to_latex.py`. The exporter verifies each completed source run,
selects data and metrics from the same artifact owner, and marks any legacy,
development, smoke, or incomplete input as preliminary and non-citable. Copy
the generated `.tex` files and the referenced figure PDFs to Overleaf only after
the two primary roots pass `verify --require-citable`.

## Method / System overview

In `System overview`, the preparation/execution description now states that a
locked execution is checkpointed. The motif-counting step now begins with the
synthetic correctness gate and commits each expensive completed count before the
next one.

## Method / Data acquisition

The BLTE suitability paragraph now states that BLTE is a separate optional run
and does not control the citability of the primary HF and Real-CATS runs.

## Method / Detection models

The Random Forest table entry changed from “all CPU cores” to “allocated CPU
threads”. Model semantics remain unchanged; only operational parallelism is
separated from the analytical parameter definition.

## Implementation details / Language and libraries

The text now explains that:

- execution is CPU-only;
- each run receives a recorded thread allocation;
- native thread limits are set before numerical libraries are imported;
- coordinated HF and Real-CATS workers cannot each claim every core.

The existing library versions remain the versions installed in the verified
environment. `requirements-results.txt` now pins them for server installation.

## Implementation details / Project structure

The layout listing adds:

- `requirements-results.txt`;
- `execution_control.py`;
- `execution_artifacts.py`;
- `coordination.py`;
- `verify.py`;
- `validation.py`.

The entry-point paragraph also lists `preflight`, `verify`, `execute --resume`,
and `execute-parallel`.

## Implementation details / Engineering challenges

This subsection was expanded into three subsubsections.

### Locating the bottleneck

- HF is scanned from a local Parquet in bounded batches.
- Local download removes network cost but not full local scans.
- Raphtory motif counting, not pandas or model fitting, is the main cost.
- One graph is reused across all deltas for an ordering; Real-CATS does this one
  layer at a time.

### Alternatives considered

- Selective motif computation was rejected because Raphtory 0.17.0 exposes one
  40-role operation. The pipeline retains 32 roles after verifying all 40.
- Spark was rejected because Raphtory still requires a single-process graph.
  PyArrow and DuckDB already handle the large event operations; pandas holds
  only address-level objects.
- Real-CATS layers were not reduced because layer-specific behavior is part of
  RQ3. The five layers are processed sequentially to bound peak memory.

### Long-run execution

- Preflight records RAM and per-device disk budgets without changing the run.
- HF checkpoints are ordering-by-delta.
- Real-CATS checkpoints are layer-by-ordering-by-delta.
- Invalid committed data stop the run and are never overwritten.
- A same-host coordinator can run HF and RQ3 concurrently with disjoint threads;
  one failed worker does not terminate the healthy worker.

## Implementation details / Reproducibility

The constant listing changes `PROTOCOL_VERSION = 3` to
`PROTOCOL_VERSION = 4`.

New text describes:

- thesis and development preparation profiles;
- immutable `execution_definition.json`;
- direct/coordinated mode and locked operational thread parameters;
- exact reproducibility for motif counts, addresses, splits, deltas, and
  decisions;
- tolerance-based comparison for independently recomputed floating results;
- byte-preserving reuse of existing stages/checkpoints;
- normalized PDF metadata;
- exact synthetic fixtures for all 40 returned catalogue positions, including
  the eight triangle roles excluded from experiment features;
- target, split, fold, finite-value, probability, and cardinality guards;
- rejection of conflicting Real-CATS duplicate labels;
- mandatory schema-correct Parquets for all five layers;
- canonical Arrow IPC hashing for checkpoint content.

## Implementation details / Result artifacts

This subsection was replaced because the former statement that a locked
execution could never resume was no longer true. The replacement explains:

- fsynced no-overwrite file publication;
- immutable stage bundles and public hard links;
- separate preparation and execution resume rules;
- attempt IDs and attempt-scoped DuckDB scratch;
- the acyclic protocol-to-manifest integrity chain;
- ordinary verification versus `--require-citable`;
- five-minute fsynced heartbeat records.

The artifact listing now includes execution definitions, attempts, checkpoints,
stages, result inventory sidecars, and execution-manifest sidecars.

## Author checks before submission

1. Run both primary roots with `--profile thesis`.
2. Require successful `verify --require-citable` for each primary root.
3. Insert only tables and figures from those verified roots into Results.
4. Replace runtime estimates in the README with observed stage timings if the
   server run differs materially.
5. Keep smoke and development outputs out of numerical Results claims.
