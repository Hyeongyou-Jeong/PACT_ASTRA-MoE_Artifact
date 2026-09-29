#!/usr/bin/env python3
"""Optional sanity: W-matched token-weighted Top-B == THER.

Not a competitive baseline. Does not alter primary six-policy results.
"""

from __future__ import annotations

from ther.scripts.path_guard import apply as _apply_path_guard

_apply_path_guard()

import argparse
import csv
from collections import Counter, defaultdict, deque
from pathlib import Path

from ther.evaluation.types import CacheTransition, EventKey, ReplayEventRecord
from ther.evaluation.trace_source import DiffMoETraceSource
from ther.policies.residency import build_ther_schedule


def _empty(layer: int, activated: frozenset[int]) -> CacheTransition:
    e: frozenset[int] = frozenset()
    return CacheTransition(
        layer=layer,
        activated=activated,
        persistent_before=e,
        persistent_after=e,
        hpc_hits=e,
        mpc_hits=e,
        lpc_hits=e,
        misses=activated,
        promotions=e,
        evictions=e,
        locally_hot_candidates=e,
        priority_changes=(),
        lpc_insert_count=0,
        lpc_clear_count=0,
        useful_prefetches=e,
        wasted_prefetches=e,
    )


def load_records(trace_dir: Path, logical_batch_size: int = 16):
    records: list[ReplayEventRecord] = []
    with DiffMoETraceSource(trace_dir) as source:
        for event in source.iter_layer_events(
            splits=("profile/train", "validation", "test"),
            batch_size=logical_batch_size,
            batch_mode="logical_manifest",
        ):
            records.append(
                ReplayEventRecord(
                    key=EventKey(
                        event.replay_batch_id,
                        event.generation_iteration_id,
                        event.moe_layer_index,
                    ),
                    transformer_layer_id=event.transformer_layer_id,
                    request_ids=tuple(s.request_id for s in event.samples),
                    routed_token_counts=tuple(
                        sorted(event.routed_token_counts.items())
                    ),
                    transition=_empty(
                        event.moe_layer_index, event.activated_experts
                    ),
                )
            )
    return records


def matched_weighted_lfu_schedule(
    records: list[ReplayEventRecord],
    *,
    gpu_budget: int,
    window_size: int,
    swap_frequency: int,
) -> dict[EventKey, frozenset[int]]:
    """Independent sliding-W routed-token Top-B (same rules as THER)."""
    hist: dict[int, deque[Counter[int]]] = defaultdict(deque)
    scores: dict[int, Counter[int]] = defaultdict(Counter)
    step_count: Counter[int] = Counter()
    residents: dict[int, frozenset[int]] = defaultdict(frozenset)
    schedule: dict[EventKey, frozenset[int]] = {}
    for record in records:
        layer = record.key.moe_layer_index
        step_c = Counter(
            {int(e): int(n) for e, n in record.routed_token_counts if n}
        )
        h = hist[layer]
        s = scores[layer]
        if len(h) == window_size:
            old = h.popleft()
            for e, n in old.items():
                s[e] -= n
                if s[e] <= 0:
                    del s[e]
        h.append(step_c)
        for e, n in step_c.items():
            s[e] += n
        step_count[layer] += 1
        if step_count[layer] % swap_frequency == 0:
            ranked = sorted(
                ((e, n) for e, n in s.items() if n > 0),
                key=lambda item: (-item[1], item[0]),
            )
            residents[layer] = frozenset(e for e, _ in ranked[:gpu_budget])
        schedule[record.key] = residents[layer]
    return schedule


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--trace-root",
        type=Path,
        default=Path("ther/traces/data"),
    )
    p.add_argument(
        "--output",
        type=Path,
        default=Path("ther/results/supplemental/matched_weighted_lfu_check.csv"),
    )
    p.add_argument("--gpu-budget", type=int, default=12)
    p.add_argument("--window-size", type=int, default=8)
    p.add_argument("--swap-frequency", type=int, default=4)
    args = p.parse_args()

    workloads = sorted(
        d.name
        for d in args.trace_root.iterdir()
        if d.is_dir() and (d / "raw_trace.h5").is_file()
    )
    rows = []
    for wl in workloads:
        records = load_records(args.trace_root / wl)
        ther, _ = build_ther_schedule(
            records,
            gpu_budget=args.gpu_budget,
            window_size=args.window_size,
            swap_frequency=args.swap_frequency,
        )
        matched = matched_weighted_lfu_schedule(
            records,
            gpu_budget=args.gpu_budget,
            window_size=args.window_size,
            swap_frequency=args.swap_frequency,
        )
        n_diff = sum(1 for k in ther if ther[k] != matched[k])
        rows.append(
            {
                "workload": wl,
                "n_events": len(ther),
                "n_resident_set_diffs": n_diff,
                "max_abs_difference": 0 if n_diff == 0 else 1,
                "identical_to_ther": n_diff == 0,
            }
        )
        print(f"[matched-wlf] {wl}: diffs={n_diff}/{len(ther)}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "workload",
                "n_events",
                "n_resident_set_diffs",
                "max_abs_difference",
                "identical_to_ther",
            ],
        )
        w.writeheader()
        w.writerows(rows)
    print(f"[matched-wlf] wrote {args.output}")
    if any(not r["identical_to_ther"] for r in rows):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
