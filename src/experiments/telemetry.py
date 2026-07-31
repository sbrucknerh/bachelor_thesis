"""Console and JSONL telemetry for long-running experiment stages."""

from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import sys
import threading
from time import monotonic
from typing import Any, Iterator

from src.experiments.artifacts import json_ready


class RunTelemetry:
    """Append-only operational log; it is not an experimental result artifact."""

    def __init__(self, run_root: str | Path) -> None:
        self.root = Path(run_root)
        self.path = self.root / "run_log.jsonl"
        self._write_lock = threading.RLock()
        self._latest_checkpoint: str | None = None
        self._active_bars: list[Any] = []

    def set_latest_checkpoint(self, checkpoint_id: str) -> None:
        with self._write_lock:
            self._latest_checkpoint = checkpoint_id

    def emit(
        self,
        stage: str,
        status: str,
        *,
        elapsed_seconds: float | None = None,
        **details: Any,
    ) -> None:
        record = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "stage": stage,
            "status": status,
            **details,
        }
        if elapsed_seconds is not None:
            record["elapsed_seconds"] = round(float(elapsed_seconds), 6)
        ready = json_ready(record)
        with self._write_lock:
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(ready, sort_keys=True, allow_nan=False) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
        elapsed = (
            f" elapsed={elapsed_seconds:.1f}s" if elapsed_seconds is not None else ""
        )
        detail_text = " ".join(
            f"{key}={value}" for key, value in details.items() if value is not None
        )
        suffix = f" {detail_text}" if detail_text else ""
        message = (
            f"[{record['timestamp_utc']}] {status.upper():9s} "
            f"{stage}{elapsed}{suffix}"
        )
        with self._write_lock:
            if self._active_bars:
                from tqdm import tqdm

                tqdm.write(message, file=sys.stderr)
            else:
                print(message, file=sys.stderr, flush=True)

    @contextmanager
    def step(self, stage: str, **details: Any) -> Iterator[None]:
        started = monotonic()
        self.emit(stage, "started", **details)
        stopped = threading.Event()

        def heartbeat() -> None:
            while not stopped.wait(300.0):
                with self._write_lock:
                    latest = self._latest_checkpoint
                self.emit(
                    stage,
                    "heartbeat",
                    elapsed_seconds=monotonic() - started,
                    process_threads=threading.active_count(),
                    free_disk_bytes=shutil.disk_usage(self.root).free,
                    latest_checkpoint=latest,
                )

        thread = threading.Thread(
            target=heartbeat,
            name=f"telemetry-{stage[:32]}",
            daemon=True,
        )
        thread.start()
        try:
            yield
        except BaseException as error:
            stopped.set()
            thread.join(timeout=1.0)
            self.emit(
                stage,
                "failed",
                elapsed_seconds=monotonic() - started,
                error_type=type(error).__name__,
                error=str(error),
                **details,
            )
            raise
        stopped.set()
        thread.join(timeout=1.0)
        self.emit(stage, "completed", elapsed_seconds=monotonic() - started, **details)

    def progress(self, stage: str, **details: Any) -> None:
        self.emit(stage, "progress", **details)

    @contextmanager
    def task_progress(
        self,
        description: str,
        total: int,
        *,
        unit: str = "task",
        initial: int = 0,
    ) -> Iterator[Any]:
        """Display a durable task counter with tqdm's elapsed time and ETA."""
        from tqdm import tqdm

        if total < 0 or not 0 <= initial <= total:
            raise ValueError("Invalid progress total or initial value")
        worker = os.environ.get("THESIS_PROGRESS_LABEL")
        label = f"{worker}: {description}" if worker else description
        position = int(os.environ.get("THESIS_PROGRESS_POSITION", "0"))
        disabled = os.environ.get("THESIS_DISABLE_TQDM", "0") == "1"
        bar = tqdm(
            total=total,
            initial=initial,
            desc=label,
            unit=unit,
            position=position,
            dynamic_ncols=sys.stderr.isatty(),
            mininterval=5.0,
            leave=True,
            disable=disabled,
            file=sys.stderr,
        )
        with self._write_lock:
            self._active_bars.append(bar)
        try:
            yield bar.update
        finally:
            with self._write_lock:
                if bar in self._active_bars:
                    self._active_bars.remove(bar)
            bar.close()
