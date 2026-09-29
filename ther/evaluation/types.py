"""Event keys and per-step replay records used by policy evaluation."""
from __future__ import annotations

from dataclasses import dataclass

@dataclass(frozen=True)
class PriorityChange:
    expert_id: int
    before: float
    after: float
    reason: str


@dataclass(frozen=True)
class CacheTransition:
    layer: int
    activated: frozenset[int]
    persistent_before: frozenset[int]
    persistent_after: frozenset[int]
    hpc_hits: frozenset[int]
    mpc_hits: frozenset[int]
    lpc_hits: frozenset[int]
    misses: frozenset[int]
    promotions: frozenset[int]
    evictions: frozenset[int]
    locally_hot_candidates: frozenset[int]
    priority_changes: tuple[PriorityChange, ...]
    lpc_insert_count: int
    lpc_clear_count: int
    useful_prefetches: frozenset[int]
    wasted_prefetches: frozenset[int]

@dataclass(frozen=True, order=True)
class EventKey:
    replay_batch_id: int
    generation_iteration_id: int
    moe_layer_index: int


@dataclass(frozen=True)
class ReplayEventRecord:
    key: EventKey
    transformer_layer_id: int
    request_ids: tuple[str, ...]
    routed_token_counts: tuple[tuple[int, int], ...]
    transition: CacheTransition
    prefetched_experts: frozenset[int] = frozenset()



__all__ = [
    "EventKey",
    "ReplayEventRecord",
    "CacheTransition",
    "PriorityChange",
]
