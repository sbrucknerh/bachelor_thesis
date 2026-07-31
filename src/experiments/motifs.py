"""Raphtory graph reuse and verified per-address motif fingerprints."""

from __future__ import annotations

from collections.abc import Iterable
from contextlib import nullcontext
import os
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

from src.experiments.constants import ALL_MOTIF_ROLE_NAMES, MOTIF_ROLE_NAMES
from src.experiments.ordering import order_events, ordered_edge_relation
from src.experiments.resources import require_in_memory_execution
from src.experiments.telemetry import RunTelemetry


class RaphtoryExperimentGraph:
    """One ordered Raphtory graph reused for all temporal windows."""

    def __init__(
        self,
        events: pd.DataFrame,
        ordering_variant: str = "canonical",
        telemetry: RunTelemetry | None = None,
    ) -> None:
        try:
            from raphtory import Graph
        except ImportError as error:
            raise ImportError("raphtory is required for motif experiments") from error

        self.source_label = "dataframe"
        stage = f"motifs.graph_load.{self.source_label}.{ordering_variant}"
        context = (
            telemetry.step(
                stage,
                overall_key=f"step:{stage}",
                events=len(events),
                source="dataframe",
            )
            if telemetry
            else nullcontext()
        )
        with context:
            ordered = order_events(events, ordering_variant)
            edge_columns = ordered.loc[:, ["timestamp", "from_address", "to_address"]]
            self.graph = Graph()
            self.graph.load_edges(
                edge_columns,
                time="timestamp",
                src="from_address",
                dst="to_address",
            )
        self.ordering_variant = ordering_variant
        self.event_count = len(ordered)
        self.telemetry = telemetry

    @classmethod
    def from_parquet(
        cls,
        events_path: str | Path,
        ordering_variant: str,
        temporary_directory: str | Path,
        telemetry: RunTelemetry | None = None,
    ) -> "RaphtoryExperimentGraph":
        try:
            from raphtory import Graph
        except ImportError as error:
            raise ImportError("raphtory is required for motif experiments") from error

        path = Path(events_path)
        event_count = int(pq.ParquetFile(path).metadata.num_rows)
        source_label = path.stem
        stage = f"motifs.graph_load.{source_label}.{ordering_variant}"
        context = (
            telemetry.step(
                stage,
                overall_key=f"step:{stage}",
                events=event_count,
                source=str(path),
            )
            if telemetry
            else nullcontext()
        )
        instance = cls.__new__(cls)
        with context:
            with ordered_edge_relation(path, ordering_variant, temporary_directory) as relation:
                reader = relation.to_arrow_reader(batch_size=1_000_000)
                instance.graph = Graph()
                instance.graph.load_edges(
                    reader,
                    time="timestamp",
                    src="from_address",
                    dst="to_address",
                )
        instance.ordering_variant = ordering_variant
        instance.event_count = event_count
        instance.telemetry = telemetry
        instance.source_label = source_label
        return instance

    def _count(
        self, delta: int, targets: Iterable[str], *, retain_triangles: bool
    ) -> pd.DataFrame:
        from raphtory.algorithms import local_temporal_three_node_motifs

        target_index = pd.Index([str(target) for target in targets], name="address")
        threads = max(1, int(os.environ.get("THESIS_MODEL_THREADS", "1")))
        stage = (
            f"motifs.count.{self.source_label}.{self.ordering_variant}.delta_{int(delta)}"
        )
        context = (
            self.telemetry.step(
                stage,
                events=self.event_count,
                targets=len(target_index),
                threads=threads,
            )
            if self.telemetry
            else nullcontext()
        )
        with context:
            result = local_temporal_three_node_motifs(
                self.graph,
                int(delta),
                threads=threads,
            )
            records: list[dict] = []
            target_set = set(target_index)
            for node, payload in result.items():
                address = str(node.name if hasattr(node, "name") else node)
                if address not in target_set:
                    continue
                counts = payload["motif_counter"] if isinstance(payload, dict) else payload
                if len(counts) != len(ALL_MOTIF_ROLE_NAMES):
                    raise RuntimeError(
                        f"Raphtory returned {len(counts)} roles; expected "
                        f"{len(ALL_MOTIF_ROLE_NAMES)}"
                    )
                records.append(
                    {"address": address, **dict(zip(ALL_MOTIF_ROLE_NAMES, counts, strict=True))}
                )

        if records:
            frame = pd.DataFrame.from_records(records).set_index("address")
        else:
            frame = pd.DataFrame(index=target_index, columns=ALL_MOTIF_ROLE_NAMES).fillna(0)
        frame = frame.reindex(target_index, fill_value=0)
        columns = ALL_MOTIF_ROLE_NAMES if retain_triangles else MOTIF_ROLE_NAMES
        return frame.loc[:, columns].astype("int64")

    def count(self, delta: int, targets: Iterable[str]) -> pd.DataFrame:
        """Return the 32 roles valid on a target-incident experiment graph."""
        return self._count(delta, targets, retain_triangles=False)

    def count_all(self, delta: int, targets: Iterable[str]) -> pd.DataFrame:
        """Return all 40 roles for exact full-neighbourhood validation fixtures."""
        return self._count(delta, targets, retain_triangles=True)

    def sweep(self, deltas: Iterable[int], targets: Iterable[str]) -> dict[int, pd.DataFrame]:
        target_values = tuple(str(target) for target in targets)
        return {int(delta): self.count(int(delta), target_values) for delta in deltas}


def count_layered_motifs(
    events: pd.DataFrame,
    targets: Iterable[str],
    deltas: Iterable[int],
    ordering_variant: str = "canonical",
) -> dict[int, pd.DataFrame]:
    """Count each event layer separately and concatenate layer-prefixed roles."""
    target_values = tuple(str(target) for target in targets)
    combined = {
        int(delta): pd.DataFrame(index=pd.Index(target_values, name="address"))
        for delta in deltas
    }
    for layer, layer_events in events.groupby("layer", sort=True):
        graph = RaphtoryExperimentGraph(layer_events, ordering_variant)
        layer_counts = graph.sweep(deltas, target_values)
        for delta, counts in layer_counts.items():
            prefixed = counts.add_prefix(f"{layer}__")
            combined[delta] = combined[delta].join(prefixed, how="left")
    return {delta: frame.fillna(0).astype("int64") for delta, frame in combined.items()}


def count_layer_parquets(
    layer_paths: dict[str, str],
    targets: Iterable[str],
    deltas: Iterable[int],
    ordering_variant: str = "canonical",
    temporary_directory: str | Path | None = None,
    telemetry: RunTelemetry | None = None,
) -> dict[int, pd.DataFrame]:
    """Layer-preserving counting without materializing all Real-CATS layers together."""
    target_values = tuple(str(target) for target in targets)
    combined = {
        int(delta): pd.DataFrame(index=pd.Index(target_values, name="address"))
        for delta in deltas
    }
    for layer, path in sorted(layer_paths.items()):
        report = require_in_memory_execution(
            path,
            expansion_factor=4.0,
            columns=("timestamp", "from_address", "to_address"),
        )
        if telemetry:
            telemetry.progress(f"realcats.resource_preflight.{layer}", **report)
        graph = RaphtoryExperimentGraph.from_parquet(
            path,
            ordering_variant,
            temporary_directory or Path(path).parent / "duckdb_tmp",
            telemetry=telemetry,
        )
        for delta, counts in graph.sweep(deltas, target_values).items():
            combined[delta] = combined[delta].join(counts.add_prefix(f"{layer}__"), how="left")
        del graph
    return {delta: frame.fillna(0).astype("int64") for delta, frame in combined.items()}
