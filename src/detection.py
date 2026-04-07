"""Coordinated malicious behaviour detection based on temporal motif analysis.

This module identifies suspicious address clusters by analyzing the frequency
and patterns of temporal motifs associated with each address.
"""

from collections import defaultdict

from config import DEFAULT_MOTIF_THRESHOLD


def detect_coordinated_behaviour(
    motifs: list[dict],
    threshold: int = DEFAULT_MOTIF_THRESHOLD,
) -> list[dict]:
    """Detect addresses involved in coordinated behaviour based on motif participation.

    Flags addresses that participate in an unusually high number of temporal motifs
    of suspicious types (fan_out, fan_in, cycle).

    Args:
        motifs: List of motif dictionaries from temporal_motifs.extract_temporal_motifs.
        threshold: Minimum number of suspicious motif participations to flag an address.

    Returns:
        List of detection results, each containing:
        - address: the flagged address
        - motif_count: total motif participations
        - suspicious_types: dict of suspicious motif type counts
        - risk_score: simple risk metric based on motif participation
    """
    suspicious_types = {"fan_out", "fan_in", "cycle"}

    # Count motif participation per address
    address_motifs = defaultdict(lambda: defaultdict(int))

    for motif in motifs:
        if motif["motif_type"] in suspicious_types:
            for node in motif["nodes"]:
                address_motifs[node][motif["motif_type"]] += 1

    # Flag addresses exceeding the threshold
    flagged = []
    for address, type_counts in address_motifs.items():
        total = sum(type_counts.values())
        if total >= threshold:
            flagged.append({
                "address": address,
                "motif_count": total,
                "suspicious_types": dict(type_counts),
                "risk_score": _compute_risk_score(type_counts),
            })

    # Sort by risk score descending
    flagged.sort(key=lambda x: x["risk_score"], reverse=True)
    return flagged


def _compute_risk_score(type_counts: dict[str, int]) -> float:
    """Compute a simple risk score from motif type counts.

    Weights:
    - fan_out: 1.0 (common in wash trading, airdrop farming)
    - fan_in: 1.0 (common in fund consolidation)
    - cycle: 1.5 (strong indicator of wash trading)
    """
    weights = {"fan_out": 1.0, "fan_in": 1.0, "cycle": 1.5}
    return sum(count * weights.get(mtype, 0.5) for mtype, count in type_counts.items())
