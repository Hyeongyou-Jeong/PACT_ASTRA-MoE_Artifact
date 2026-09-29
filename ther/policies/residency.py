"""Expert-residency schedules for GPU routed-token coverage.

- Static: fixed expert ids {0..B-1}
- LRU: binary recency within each F-step interval; refresh every F
- LFU: binary activation counts within each F-step interval; refresh every F
- Cumulative Weighted-LFU (CSV key WeightedLFU): full-history routed-token
  counts; refresh every F (no finite W)
- THER: routed-token counts over a finite window W; refresh every F
- Oracle: next-F-step routed-token counts (offline upper bound)
"""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Sequence

from ther.evaluation.types import EventKey, ReplayEventRecord
from ther.policies.window import LayerIntervalRoutingTable


def build_static_schedule(
    records: Sequence[ReplayEventRecord],
    *,
    gpu_budget: int,
) -> dict[EventKey, frozenset[int]]:
    residents = frozenset(range(max(0, int(gpu_budget))))
    return {record.key: residents for record in records}


def build_lru_schedule(
    records: Sequence[ReplayEventRecord],
    *,
    gpu_budget: int,
    swap_frequency: int,
) -> tuple[dict[EventKey, frozenset[int]], frozenset[EventKey]]:
    if swap_frequency <= 0:
        raise ValueError("swap_frequency must be positive")
    last_used: dict[int, dict[int, int]] = defaultdict(dict)
    clock: Counter[int] = Counter()
    step_count: Counter[int] = Counter()
    residents: dict[int, frozenset[int]] = defaultdict(frozenset)
    schedule: dict[EventKey, frozenset[int]] = {}
    ready: set[EventKey] = set()
    for record in records:
        layer = record.key.moe_layer_index
        clock[layer] += 1
        for expert, count in record.routed_token_counts:
            if count > 0:
                last_used[layer][int(expert)] = clock[layer]
        step_count[layer] += 1
        if step_count[layer] % swap_frequency == 0:
            ranked = sorted(
                last_used[layer].items(),
                key=lambda kv: (kv[1], -kv[0]),
                reverse=True,
            )
            residents[layer] = frozenset(eid for eid, _ in ranked[:gpu_budget])
            last_used[layer].clear()
            clock[layer] = 0
        if step_count[layer] >= swap_frequency:
            ready.add(record.key)
        schedule[record.key] = residents[layer]
    return schedule, frozenset(ready)


def build_lfu_schedule(
    records: Sequence[ReplayEventRecord],
    *,
    gpu_budget: int,
    swap_frequency: int,
) -> tuple[dict[EventKey, frozenset[int]], frozenset[EventKey]]:
    """Step-count LFU (expert active in a step increments by 1), refresh every F."""
    if swap_frequency <= 0:
        raise ValueError("swap_frequency must be positive")
    access: dict[int, Counter[int]] = defaultdict(Counter)
    last_used: dict[int, dict[int, int]] = defaultdict(dict)
    clock: Counter[int] = Counter()
    step_count: Counter[int] = Counter()
    residents: dict[int, frozenset[int]] = defaultdict(frozenset)
    schedule: dict[EventKey, frozenset[int]] = {}
    ready: set[EventKey] = set()
    for record in records:
        layer = record.key.moe_layer_index
        clock[layer] += 1
        for expert, count in record.routed_token_counts:
            if count > 0:
                eid = int(expert)
                access[layer][eid] += 1
                last_used[layer][eid] = clock[layer]
        step_count[layer] += 1
        if step_count[layer] % swap_frequency == 0:
            ranked = sorted(
                access[layer].items(),
                key=lambda kv: (kv[1], last_used[layer].get(kv[0], -1), -kv[0]),
                reverse=True,
            )
            residents[layer] = frozenset(eid for eid, _ in ranked[:gpu_budget])
            access[layer].clear()
            last_used[layer].clear()
            clock[layer] = 0
        if step_count[layer] >= swap_frequency:
            ready.add(record.key)
        schedule[record.key] = residents[layer]
    return schedule, frozenset(ready)


def build_weighted_lfu_schedule(
    records: Sequence[ReplayEventRecord],
    *,
    gpu_budget: int,
    swap_frequency: int,
) -> tuple[dict[EventKey, frozenset[int]], frozenset[EventKey]]:
    """Cumulative routed-token LFU; no finite W (CSV key: WeightedLFU)."""
    if swap_frequency <= 0:
        raise ValueError("swap_frequency must be positive")
    scores: dict[int, Counter[int]] = defaultdict(Counter)
    step_count: Counter[int] = Counter()
    residents: dict[int, frozenset[int]] = defaultdict(frozenset)
    schedule: dict[EventKey, frozenset[int]] = {}
    ready: set[EventKey] = set()
    for record in records:
        layer = record.key.moe_layer_index
        scores[layer].update(Counter(dict(record.routed_token_counts)))
        step_count[layer] += 1
        if step_count[layer] % swap_frequency == 0:
            ranked = sorted(
                ((eid, cnt) for eid, cnt in scores[layer].items() if cnt > 0),
                key=lambda item: (-item[1], item[0]),
            )
            residents[layer] = frozenset(eid for eid, _ in ranked[:gpu_budget])
        if step_count[layer] >= swap_frequency:
            ready.add(record.key)
        schedule[record.key] = residents[layer]
    return schedule, frozenset(ready)


def build_oracle_schedule(
    records: Sequence[ReplayEventRecord],
    *,
    gpu_budget: int,
    swap_frequency: int,
) -> tuple[dict[EventKey, frozenset[int]], frozenset[EventKey]]:
    """Offline upper bound: at each refresh, use the next F steps' token counts."""
    if swap_frequency <= 0:
        raise ValueError("swap_frequency must be positive")
    by_layer: dict[int, list[ReplayEventRecord]] = defaultdict(list)
    for record in records:
        by_layer[record.key.moe_layer_index].append(record)

    schedule: dict[EventKey, frozenset[int]] = {}
    ready: set[EventKey] = set()
    for layer, layer_records in by_layer.items():
        residents: frozenset[int] = frozenset()
        for idx, record in enumerate(layer_records):
            step = idx + 1
            if step % swap_frequency == 0:
                future = Counter()
                for ahead in layer_records[idx : idx + swap_frequency]:
                    future.update(Counter(dict(ahead.routed_token_counts)))
                ranked = sorted(
                    ((eid, cnt) for eid, cnt in future.items() if cnt > 0),
                    key=lambda item: (-item[1], item[0]),
                )
                residents = frozenset(eid for eid, _ in ranked[:gpu_budget])
            if step >= swap_frequency:
                ready.add(record.key)
            schedule[record.key] = residents
    return schedule, frozenset(ready)



def build_ther_schedule(
    records: Sequence[ReplayEventRecord],
    *,
    gpu_budget: int,
    window_size: int,
    swap_frequency: int,
) -> tuple[dict[EventKey, frozenset[int]], frozenset[EventKey]]:
    """Refresh every F steps using routed-token counts from the previous W steps."""

    if window_size <= 0 or swap_frequency <= 0:
        raise ValueError("THER window_size and swap_frequency must be positive")
    table = LayerIntervalRoutingTable(window_size)
    step_count: Counter[int] = Counter()
    residents: dict[int, frozenset[int]] = defaultdict(frozenset)
    schedule: dict[EventKey, frozenset[int]] = {}
    ready: set[EventKey] = set()
    for record in records:
        layer = record.key.moe_layer_index
        counter = Counter(dict(record.routed_token_counts))
        table.observe_step(layer, counter)
        step_count[layer] += 1
        if step_count[layer] % swap_frequency == 0:
            residents[layer] = frozenset(
                table.current_ranked_residents(layer, gpu_budget)
            )
        if step_count[layer] >= swap_frequency:
            ready.add(record.key)
        schedule[record.key] = residents[layer]
    return schedule, frozenset(ready)


def build_schedule(
    name: str,
    records: Sequence[ReplayEventRecord],
    *,
    gpu_budget: int,
    window_size: int,
    swap_frequency: int,
) -> tuple[dict[EventKey, frozenset[int]], frozenset[EventKey]]:
    key = name.lower().replace("-", "").replace("_", "")
    if key == "static":
        sched = build_static_schedule(records, gpu_budget=gpu_budget)
        return sched, frozenset(sched)
    if key == "lru":
        return build_lru_schedule(
            records, gpu_budget=gpu_budget, swap_frequency=swap_frequency
        )
    if key == "lfu":
        return build_lfu_schedule(
            records, gpu_budget=gpu_budget, swap_frequency=swap_frequency
        )
    if key == "weightedlfu":
        return build_weighted_lfu_schedule(
            records, gpu_budget=gpu_budget, swap_frequency=swap_frequency
        )
    if key == "ther":
        return build_ther_schedule(
            records,
            gpu_budget=gpu_budget,
            window_size=window_size,
            swap_frequency=swap_frequency,
        )
    if key == "oracle":
        return build_oracle_schedule(
            records, gpu_budget=gpu_budget, swap_frequency=swap_frequency
        )
    raise ValueError(f"unknown policy: {name}")


__all__ = [
    "build_static_schedule",
    "build_lru_schedule",
    "build_lfu_schedule",
    "build_weighted_lfu_schedule",
    "build_ther_schedule",
    "build_oracle_schedule",
    "build_schedule",
    "LayerIntervalRoutingTable",
]
