"""Weighted, resumable progress plans for long-running experiment execution."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import sys
import threading
from typing import Iterator, Mapping


_ROWS_PER_WORK_UNIT = 1_000_000
_MINOR_WORK_UNIT = 0.05
_EVALUATION_EVENT_PASS_FRACTION = 0.25


def _event_work(rows: int) -> float:
    return max(_MINOR_WORK_UNIT, int(rows) / _ROWS_PER_WORK_UNIT)


@dataclass(frozen=True)
class ExecutionProgressPlan:
    """Named weighted work items and their resume relationships."""

    description: str
    items: Mapping[str, float]
    graph_requirements: Mapping[str, tuple[str, ...]]

    @property
    def total_weight(self) -> float:
        return float(sum(self.items.values()))

    @property
    def digest(self) -> str:
        payload = {
            "description": self.description,
            "items": dict(sorted(self.items.items())),
            "graph_requirements": {
                key: list(value)
                for key, value in sorted(self.graph_requirements.items())
            },
        }
        encoded = json.dumps(
            payload, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()


def build_execution_progress_plan(
    run_root: str | Path, protocol: dict
) -> ExecutionProgressPlan:
    """Build an operational ETA plan from locked event counts.

    One work unit represents approximately one pass over one million events.
    Model evaluation and small publication stages receive conservative fractional
    weights. The plan affects telemetry only, never experimental computation.
    """
    import pyarrow.parquet as pq

    root = Path(run_root)
    variants = [str(value) for value in protocol.get("ordering_variants", [])]
    deltas = [int(value) for value in protocol.get("deltas", [])]
    datasets = protocol.get("datasets", {})
    items: dict[str, float] = {
        "stage:synthetic-validation": _MINOR_WORK_UNIT,
        "stage:final-validation": _MINOR_WORK_UNIT,
        "finalize:inventory": _MINOR_WORK_UNIT,
        "finalize:manifest": _MINOR_WORK_UNIT,
    }
    graph_requirements: dict[str, tuple[str, ...]] = {}
    labels: list[str] = []

    if "huggingface" in datasets:
        labels.append("Hugging Face")
        dataset = datasets["huggingface"]
        artifacts = dataset["artifacts"]
        event_path = root / artifacts["events"]
        rows = int(pq.ParquetFile(event_path).metadata.num_rows)
        event_weight = _event_work(rows)
        items["stage:hf-baseline"] = event_weight

        target_table = pq.read_table(
            root / artifacts["targets"], columns=["cohort"]
        )
        cohorts = sorted(
            {
                str(value.as_py())
                for value in target_table.column("cohort")
                if value.as_py() is not None
            }
        )
        for variant in variants:
            checkpoint_keys = tuple(
                f"checkpoint:hf-{variant}-delta-{delta}" for delta in deltas
            )
            graph_key = f"step:motifs.graph_load.{event_path.stem}.{variant}"
            items[graph_key] = event_weight
            graph_requirements[graph_key] = checkpoint_keys
            for key in checkpoint_keys:
                items[key] = event_weight
            items[f"stage:hf-motifs-{variant}"] = _MINOR_WORK_UNIT
            for cohort in cohorts:
                items[f"stage:hf-eval-{cohort}-{variant}"] = max(
                    _MINOR_WORK_UNIT,
                    event_weight * _EVALUATION_EVENT_PASS_FRACTION,
                )
        items["stage:hf-cross-order-aggregation"] = _MINOR_WORK_UNIT

    if "realcats" in datasets:
        labels.append("Real-CATS")
        dataset = datasets["realcats"]
        layers: dict[str, tuple[Path, int, float]] = {}
        for layer, relative in sorted(dataset["artifacts"]["layers"].items()):
            path = root / relative
            rows = int(pq.ParquetFile(path).metadata.num_rows)
            layers[str(layer)] = (path, rows, _event_work(rows))
        total_layer_weight = sum(weight for _, _, weight in layers.values())
        for variant in variants:
            for layer, (path, rows, event_weight) in layers.items():
                checkpoint_keys = tuple(
                    f"checkpoint:realcats-{layer}-{variant}-delta-{delta}"
                    for delta in deltas
                )
                if rows:
                    graph_key = f"step:motifs.graph_load.{path.stem}.{variant}"
                    items[graph_key] = event_weight
                    graph_requirements[graph_key] = checkpoint_keys
                for key in checkpoint_keys:
                    items[key] = event_weight
            items[f"unit:realcats-eval:{variant}"] = max(
                _MINOR_WORK_UNIT,
                total_layer_weight * _EVALUATION_EVENT_PASS_FRACTION,
            )
        items["stage:realcats-ordering-evaluation"] = _MINOR_WORK_UNIT
        items["stage:realcats-cross-order-aggregation"] = _MINOR_WORK_UNIT

    if "blte" in datasets:
        labels.append("BLTE")
        items["stage:blte-secondary-evaluation"] = 1.0

    description = "+".join(labels) if labels else "Validation"
    return ExecutionProgressPlan(description, items, graph_requirements)


def completed_progress_items(
    run_root: str | Path, plan: ExecutionProgressPlan
) -> set[str]:
    """Infer valid completed work from immutable committed artifacts."""
    root = Path(run_root)
    if (root / "execution_manifest.json").is_file():
        return set(plan.items)

    completed: set[str] = set()
    stage_root = root / "stages"
    checkpoint_root = root / "checkpoints"
    for key in plan.items:
        if key.startswith("stage:"):
            stage_id = key.removeprefix("stage:")
            if (stage_root / stage_id / "stage_manifest.json").is_file():
                completed.add(key)
        elif key.startswith("checkpoint:"):
            checkpoint_id = key.removeprefix("checkpoint:")
            directory = checkpoint_root / checkpoint_id
            if (directory / "counts.parquet").is_file() and (
                directory / "metadata.json"
            ).is_file():
                completed.add(key)

    for graph_key, requirements in plan.graph_requirements.items():
        if requirements and all(key in completed for key in requirements):
            completed.add(graph_key)
    if "stage:realcats-ordering-evaluation" in completed:
        completed.update(
            key for key in plan.items if key.startswith("unit:realcats-eval:")
        )
    if (root / "result_artifacts.json").is_file():
        completed.add("finalize:inventory")
    return completed & set(plan.items)


def _read_root_progress(root: Path, plan: ExecutionProgressPlan) -> float | None:
    path = root / "overall_progress.json"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None
    if payload.get("plan_sha256") != plan.digest:
        return None
    value = float(payload.get("completed_weight", 0.0))
    return min(plan.total_weight, max(0.0, value))


@contextmanager
def workflow_progress_display(
    plans: Mapping[Path, ExecutionProgressPlan],
    *,
    description: str = "Thesis total execution",
    poll_seconds: float = 2.0,
) -> Iterator[None]:
    """Display one aggregate ETA while one or more roots execute."""
    from tqdm import tqdm

    if poll_seconds <= 0:
        raise ValueError("poll_seconds must be positive")
    initial_by_root = {
        Path(root): sum(
            plan.items[key]
            for key in completed_progress_items(root, plan)
        )
        for root, plan in plans.items()
    }
    total = sum(plan.total_weight for plan in plans.values())
    initial = sum(initial_by_root.values())
    disabled = os.environ.get("THESIS_DISABLE_TQDM", "0") == "1"
    bar = tqdm(
        total=total,
        initial=initial,
        desc=description,
        unit="work",
        position=0,
        dynamic_ncols=sys.stderr.isatty(),
        mininterval=2.0,
        leave=True,
        disable=disabled,
        file=sys.stderr,
        bar_format="{desc}: {percentage:3.0f}%|{bar}| {elapsed}<{remaining}",
    )
    stopped = threading.Event()
    lock = threading.RLock()
    observed = dict(initial_by_root)

    def refresh() -> None:
        target = 0.0
        for root, plan in plans.items():
            current = _read_root_progress(root, plan)
            if current is not None:
                observed[root] = max(observed[root], current)
            target += observed[root]
        target = min(total, target)
        with lock:
            if target > bar.n:
                bar.update(target - bar.n)
            bar.refresh()

    def poll() -> None:
        while not stopped.wait(poll_seconds):
            refresh()

    thread = threading.Thread(
        target=poll,
        name="thesis-total-progress",
        daemon=True,
    )
    thread.start()
    failed = False
    try:
        yield
    except BaseException:
        failed = True
        raise
    finally:
        stopped.set()
        thread.join(timeout=max(1.0, poll_seconds * 2))
        refresh()
        if not failed and bar.n < total:
            bar.update(max(0.0, total - bar.n))
        bar.close()
