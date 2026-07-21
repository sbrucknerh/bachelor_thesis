"""Exact temporal-motif fixtures with no random background edges."""

from __future__ import annotations

import pandas as pd

from src.experiments.constants import DIRECTION_CODES


def _edge_for_direction(direction: str, center: str, leaf: str) -> tuple[str, str]:
    return (leaf, center) if direction == "i" else (center, leaf)


def star_fixture(role_index: int, start: int = 100, step: int = 1) -> pd.DataFrame:
    """Create exactly one star role in Raphtory indices 0..23 at node A."""
    if not 0 <= role_index < 24:
        raise ValueError("star role_index must be in 0..23")
    star_class, direction_index = divmod(role_index, 8)
    leaves = (("B", "B", "C"), ("B", "C", "B"), ("B", "C", "C"))[star_class]
    directions = DIRECTION_CODES[direction_index]
    edges = [
        _edge_for_direction(direction, "A", leaf)
        for direction, leaf in zip(directions, leaves, strict=True)
    ]
    return _events(edges, start, step, f"star_{role_index}")


def two_node_fixture(role_index: int, start: int = 100, step: int = 1) -> pd.DataFrame:
    """Create exactly one two-node role in Raphtory indices 24..31 at node A."""
    if not 24 <= role_index < 32:
        raise ValueError("two-node role_index must be in 24..31")
    directions = DIRECTION_CODES[role_index - 24]
    edges = [_edge_for_direction(direction, "A", "B") for direction in directions]
    return _events(edges, start, step, f"two_{role_index}")


def fixture_for_role(role_index: int, start: int = 100, step: int = 1) -> pd.DataFrame:
    if role_index < 24:
        return star_fixture(role_index, start, step)
    return two_node_fixture(role_index, start, step)


def _events(
    edges: list[tuple[str, str]], start: int, step: int, prefix: str
) -> pd.DataFrame:
    rows = []
    for index, (source, destination) in enumerate(edges):
        signature = f"{prefix}_{index}"
        rows.append(
            {
                "from_address": source,
                "to_address": destination,
                "timestamp": start + index * step,
                "value": 1.0,
                "tx_hash": signature,
                "block_number": index + 1,
                "transaction_index": index,
                "event_signature": signature,
                "layer": "normal",
            }
        )
    return pd.DataFrame.from_records(rows)
