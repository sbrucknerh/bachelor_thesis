"""Console and JSONL telemetry for long-running experiment stages."""

from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
import json
import math
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
        self._overall_items: dict[str, float] = {}
        self._overall_completed: set[str] = set()
        self._overall_session_weight = 0.0
        self._overall_started: float | None = None
        self._overall_description: str | None = None
        self._overall_plan_sha256: str | None = None
        self._overall_bar: Any | None = None

    def configure_overall(
        self,
        description: str,
        work_items: dict[str, float],
        *,
        completed_items: set[str] | None = None,
        plan_sha256: str,
    ) -> None:
        """Start a persistent weighted total bar for one execution root."""
        from tqdm import tqdm

        if self._overall_started is not None:
            raise RuntimeError("Overall progress is already configured")
        if not work_items:
            raise ValueError("Overall progress requires at least one work item")
        normalized = {str(key): float(value) for key, value in work_items.items()}
        if any(not math.isfinite(value) or value <= 0 for value in normalized.values()):
            raise ValueError("Overall progress weights must be finite and positive")
        completed = set(completed_items or ()) & set(normalized)
        initial = sum(normalized[key] for key in completed)
        worker = os.environ.get("THESIS_PROGRESS_LABEL")
        label = f"{worker}: {description}" if worker else description
        hidden = os.environ.get("THESIS_HIDE_ROOT_TOTAL", "0") == "1"
        disabled = os.environ.get("THESIS_DISABLE_TQDM", "0") == "1"
        position = int(os.environ.get("THESIS_PROGRESS_POSITION", "0"))

        with self._write_lock:
            self._overall_items = normalized
            self._overall_completed = completed
            self._overall_started = monotonic()
            self._overall_description = description
            self._overall_plan_sha256 = plan_sha256
            if not hidden:
                self._overall_bar = tqdm(
                    total=sum(normalized.values()),
                    initial=initial,
                    desc=f"{label} total",
                    unit="work",
                    position=position,
                    dynamic_ncols=sys.stderr.isatty(),
                    mininterval=2.0,
                    leave=True,
                    disable=disabled,
                    file=sys.stderr,
                    bar_format=(
                        "{desc}: {percentage:3.0f}%|{bar}| "
                        "{elapsed}<{remaining}"
                    ),
                )
                self._active_bars.append(self._overall_bar)
            self._write_overall_state_locked(status="running")

    def _overall_payload_locked(
        self, *, status: str, current_item: str | None = None
    ) -> dict[str, Any]:
        if self._overall_started is None:
            return {}
        now = monotonic()
        elapsed = max(0.0, now - self._overall_started)
        total = float(sum(self._overall_items.values()))
        completed = float(
            sum(self._overall_items[key] for key in self._overall_completed)
        )
        remaining = max(0.0, total - completed)
        rate = (
            self._overall_session_weight / elapsed
            if elapsed > 0 and self._overall_session_weight > 0
            else None
        )
        eta = 0.0 if remaining == 0 else (remaining / rate if rate else None)
        completion = (
            datetime.now(timezone.utc) + timedelta(seconds=eta)
            if eta is not None
            else None
        )
        return {
            "schema_version": 1,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "status": status,
            "description": self._overall_description,
            "plan_sha256": self._overall_plan_sha256,
            "completed_items": len(self._overall_completed),
            "total_items": len(self._overall_items),
            "completed_weight": round(completed, 6),
            "total_weight": round(total, 6),
            "percent_complete": round(100.0 * completed / total, 4),
            "elapsed_seconds": round(elapsed, 6),
            "estimated_rate_work_per_second": (
                round(rate, 9) if rate is not None else None
            ),
            "eta_seconds": round(eta, 3) if eta is not None else None,
            "estimated_completion_utc": (
                completion.isoformat() if completion is not None else None
            ),
            "current_item": current_item,
        }

    def _write_overall_state_locked(
        self, *, status: str, current_item: str | None = None
    ) -> dict[str, Any]:
        payload = self._overall_payload_locked(
            status=status, current_item=current_item
        )
        if not payload:
            return payload
        path = self.root / "overall_progress.json"
        temporary = self.root / f".overall_progress.{os.getpid()}.tmp"
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(json_ready(payload), handle, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory_fd = os.open(self.root, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        return payload

    def overall_snapshot(self, *, persist: bool = False) -> dict[str, Any]:
        with self._write_lock:
            if persist:
                return self._write_overall_state_locked(status="running")
            return self._overall_payload_locked(status="running")

    def mark_overall(self, item: str) -> bool:
        """Mark one named item complete and refresh the total ETA."""
        with self._write_lock:
            if item not in self._overall_items or item in self._overall_completed:
                return False
            weight = self._overall_items[item]
            self._overall_completed.add(item)
            self._overall_session_weight += weight
            if self._overall_bar is not None:
                completed_weight = sum(
                    self._overall_items[key] for key in self._overall_completed
                )
                target = min(float(self._overall_bar.total), completed_weight)
                if target > self._overall_bar.n:
                    self._overall_bar.update(target - self._overall_bar.n)
            payload = self._write_overall_state_locked(
                status="running", current_item=item
            )
        self.emit(
            "execute.overall",
            "progress",
            completed_items=payload["completed_items"],
            total_items=payload["total_items"],
            percent_complete=payload["percent_complete"],
            eta_seconds=payload["eta_seconds"],
            estimated_completion_utc=payload["estimated_completion_utc"],
            completed_item=item,
        )
        return True

    def close_overall(self, status: str) -> None:
        with self._write_lock:
            if self._overall_started is None:
                return
            self._write_overall_state_locked(status=status)
            if self._overall_bar is not None:
                if self._overall_bar in self._active_bars:
                    self._active_bars.remove(self._overall_bar)
                self._overall_bar.close()
                self._overall_bar = None

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
    def step(
        self,
        stage: str,
        *,
        overall_key: str | None = None,
        **details: Any,
    ) -> Iterator[None]:
        started = monotonic()
        self.emit(stage, "started", **details)
        stopped = threading.Event()

        def heartbeat() -> None:
            while not stopped.wait(300.0):
                with self._write_lock:
                    latest = self._latest_checkpoint
                overall = self.overall_snapshot(persist=True)
                self.emit(
                    stage,
                    "heartbeat",
                    elapsed_seconds=monotonic() - started,
                    process_threads=threading.active_count(),
                    free_disk_bytes=shutil.disk_usage(self.root).free,
                    latest_checkpoint=latest,
                    overall_percent_complete=overall.get("percent_complete"),
                    overall_eta_seconds=overall.get("eta_seconds"),
                    overall_estimated_completion_utc=overall.get(
                        "estimated_completion_utc"
                    ),
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
        if overall_key is not None:
            self.mark_overall(overall_key)

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
        if self._overall_bar is not None:
            position += 1
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
