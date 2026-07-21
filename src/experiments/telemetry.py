"""Console and JSONL telemetry for long-running experiment stages."""

from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from time import monotonic
from typing import Any, Iterator

from src.experiments.artifacts import json_ready


class RunTelemetry:
    """Append-only operational log; it is not an experimental result artifact."""

    def __init__(self, run_root: str | Path) -> None:
        self.path = Path(run_root) / "run_log.jsonl"

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
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(ready, sort_keys=True, allow_nan=False) + "\n")
        elapsed = (
            f" elapsed={elapsed_seconds:.1f}s" if elapsed_seconds is not None else ""
        )
        detail_text = " ".join(
            f"{key}={value}" for key, value in details.items() if value is not None
        )
        suffix = f" {detail_text}" if detail_text else ""
        print(
            f"[{record['timestamp_utc']}] {status.upper():9s} {stage}{elapsed}{suffix}",
            file=sys.stderr,
            flush=True,
        )

    @contextmanager
    def step(self, stage: str, **details: Any) -> Iterator[None]:
        started = monotonic()
        self.emit(stage, "started", **details)
        try:
            yield
        except BaseException as error:
            self.emit(
                stage,
                "failed",
                elapsed_seconds=monotonic() - started,
                error_type=type(error).__name__,
                error=str(error),
                **details,
            )
            raise
        self.emit(stage, "completed", elapsed_seconds=monotonic() - started, **details)

    def progress(self, stage: str, **details: Any) -> None:
        self.emit(stage, "progress", **details)
