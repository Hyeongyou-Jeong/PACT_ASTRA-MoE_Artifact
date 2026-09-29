"""GPU routed-token coverage (and secondary activation coverage)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

from ther.evaluation.types import EventKey, ReplayEventRecord

@dataclass(frozen=True)
class ResidencyCoverage:
    activation_coverage: float
    gpu_routed_token_coverage: float
    ssd_routed_tokens: int


def _coverage(
    records: Sequence[ReplayEventRecord],
    schedule: Mapping[EventKey, frozenset[int]],
) -> ResidencyCoverage:
    activation_total = 0
    activation_hit = 0
    routed_total = 0
    routed_hit = 0
    for record in records:
        residents = schedule[record.key]
        activation_total += len(record.transition.activated)
        activation_hit += len(record.transition.activated & residents)
        for expert, count in record.routed_token_counts:
            routed_total += count
            if expert in residents:
                routed_hit += count
    return ResidencyCoverage(
        activation_coverage=activation_hit / activation_total
        if activation_total
        else 0.0,
        gpu_routed_token_coverage=routed_hit / routed_total if routed_total else 0.0,
        ssd_routed_tokens=routed_total - routed_hit,
    )

__all__ = ["ResidencyCoverage", "_coverage"]
