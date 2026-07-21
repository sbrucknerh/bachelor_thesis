"""Data contracts shared by preparation, feature extraction, and evaluation."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd


EVENT_COLUMNS = {
    "from_address",
    "to_address",
    "timestamp",
    "value",
    "tx_hash",
    "block_number",
    "transaction_index",
    "event_signature",
    "layer",
}


@dataclass(slots=True)
class ExperimentDataset:
    """In-memory experiment dataset with explicit labels and provenance."""

    name: str
    events: pd.DataFrame
    targets: pd.DataFrame
    provenance: dict = field(default_factory=dict)

    def validate(self) -> None:
        missing_events = sorted(EVENT_COLUMNS - set(self.events.columns))
        if missing_events:
            raise ValueError(f"{self.name}: missing event columns: {missing_events}")
        if self.events["from_address"].isna().any() or self.events["to_address"].isna().any():
            raise ValueError(f"{self.name}: null event endpoints are not allowed")
        if (pd.to_numeric(self.events["timestamp"], errors="coerce") <= 0).any():
            raise ValueError(f"{self.name}: timestamps must be positive integer seconds")
        if "address" not in self.targets.columns:
            raise ValueError(f"{self.name}: targets require an address column")
        if self.targets["address"].duplicated().any():
            raise ValueError(f"{self.name}: duplicate target addresses")

    @property
    def target_ids(self) -> set[str]:
        return set(self.targets["address"].astype(str))


@dataclass(frozen=True, slots=True)
class RunPaths:
    """Filesystem locations for one immutable experiment run."""

    root: Path
    protocol: Path
    protocol_hash: Path
    data: Path
    features: Path
    predictions: Path
    metrics: Path
    figures: Path
    tables: Path

    @classmethod
    def from_root(cls, root: Path) -> "RunPaths":
        return cls(
            root=root,
            protocol=root / "protocol.json",
            protocol_hash=root / "protocol.sha256",
            data=root / "data",
            features=root / "features",
            predictions=root / "predictions",
            metrics=root / "metrics",
            figures=root / "figures",
            tables=root / "tables",
        )
