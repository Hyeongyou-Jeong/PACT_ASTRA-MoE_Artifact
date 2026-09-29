#!/usr/bin/env python3
"""Minimal synthetic checks for residency policy semantics."""

from __future__ import annotations

from ther.scripts.path_guard import apply as _apply_path_guard
_apply_path_guard()

from collections import Counter

from ther.evaluation.types import CacheTransition, EventKey, ReplayEventRecord
from ther.policies.residency import (
    build_lfu_schedule,
    build_lru_schedule,
    build_oracle_schedule,
    build_static_schedule,
    build_ther_schedule,
    build_weighted_lfu_schedule,
)


def _rec(batch: int, step: int, layer: int, counts: dict[int, int]) -> ReplayEventRecord:
    activated = frozenset(e for e, c in counts.items() if c > 0)
    empty = frozenset()
    return ReplayEventRecord(
        key=EventKey(batch, step, layer),
        transformer_layer_id=layer,
        request_ids=("r0",),
        routed_token_counts=tuple(sorted(counts.items())),
        transition=CacheTransition(
            layer=layer,
            activated=activated,
            persistent_before=empty,
            persistent_after=empty,
            hpc_hits=empty,
            mpc_hits=empty,
            lpc_hits=empty,
            misses=activated,
            promotions=empty,
            evictions=empty,
            locally_hot_candidates=empty,
            priority_changes=(),
            lpc_insert_count=0,
            lpc_clear_count=0,
            useful_prefetches=empty,
            wasted_prefetches=empty,
        ),
    )


def main() -> None:
    # One layer, B=2, F=2, W=2
    # steps: e0 heavy, e1 heavy, e2 heavy, e0 heavy
    records = [
        _rec(0, 0, 0, {0: 10, 1: 1}),
        _rec(0, 1, 0, {0: 1, 1: 10}),
        _rec(0, 2, 0, {2: 20, 1: 1}),
        _rec(0, 3, 0, {0: 20, 2: 1}),
    ]
    B, F, W = 2, 2, 2

    static = build_static_schedule(records, gpu_budget=B)
    assert static[records[0].key] == frozenset({0, 1})

    lru, lru_ready = build_lru_schedule(records, gpu_budget=B, swap_frequency=F)
    # After steps 0,1 refresh: most recent are e1 then e0
    assert records[1].key in lru_ready
    assert lru[records[1].key] == frozenset({0, 1})

    lfu, _ = build_lfu_schedule(records, gpu_budget=B, swap_frequency=F)
    # After first 2 steps both e0 and e1 active once each -> {0,1}
    assert lfu[records[1].key] == frozenset({0, 1})

    wlfu, _ = build_weighted_lfu_schedule(records, gpu_budget=B, swap_frequency=F)
    # Tokens after 2 steps: e0=11, e1=11 -> {0,1}
    assert wlfu[records[1].key] == frozenset({0, 1})
    # After 4 steps cumulative: e0=31, e1=12, e2=21 -> top2 {0,2}
    assert wlfu[records[3].key] == frozenset({0, 2})

    ther, ther_ready = build_ther_schedule(
        records, gpu_budget=B, window_size=W, swap_frequency=F
    )
    assert records[1].key in ther_ready
    # Window of last 2 at step2 refresh (indices 2,3 not yet): after step1 window={0,1}
    assert ther[records[1].key] == frozenset({0, 1})
    # After step3 refresh, window is steps 2-3: e2=21, e0=20, e1=1 -> {2,0}
    assert ther[records[3].key] == frozenset({0, 2})

    oracle, _ = build_oracle_schedule(records, gpu_budget=B, swap_frequency=F)
    # At refresh on step1 (idx1), future is steps1-2? Our oracle uses idx:idx+F
    # at step%F==0 when step=2 (idx1): future records[1:3] = step1+step2 counts
    # e0=1, e1=11, e2=20 -> {2,1}
    assert oracle[records[1].key] == frozenset({1, 2})

    print("synthetic policy tests OK")


if __name__ == "__main__":
    main()
