#!/usr/bin/env python3
"""Evaluate residency policies on a collected raw_trace directory."""

from __future__ import annotations

from ther.scripts.path_guard import apply as _apply_path_guard
_apply_path_guard()

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Sequence

from ther.evaluation.coverage import _coverage
from ther.evaluation.trace_source import DiffMoETraceSource
from ther.evaluation.types import CacheTransition, EventKey, ReplayEventRecord
from ther.policies.residency import build_schedule


POLICIES = ("Static", "LRU", "LFU", "WeightedLFU", "THER", "Oracle")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--trace-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--logical-batch-size", type=int, default=16)
    p.add_argument("--gpu-budget", type=int, default=12)
    p.add_argument("--window-size", type=int, default=8)
    p.add_argument("--swap-frequency", type=int, default=4)
    p.add_argument("--workload-name", default="")
    return p.parse_args()


def _empty_transition(layer: int, activated: frozenset[int]) -> CacheTransition:
    empty: frozenset[int] = frozenset()
    return CacheTransition(
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
    )


def load_records(
    trace_dir: Path, *, logical_batch_size: int
) -> tuple[list[ReplayEventRecord], int, int]:
    records: list[ReplayEventRecord] = []
    with DiffMoETraceSource(trace_dir) as source:
        num_experts = source.num_experts
        num_layers = source.num_layers
        for event in source.iter_layer_events(
            splits=("profile/train", "validation", "test"),
            batch_size=logical_batch_size,
            batch_mode="logical_manifest",
        ):
            records.append(
                ReplayEventRecord(
                    key=EventKey(
                        replay_batch_id=event.replay_batch_id,
                        generation_iteration_id=event.generation_iteration_id,
                        moe_layer_index=event.moe_layer_index,
                    ),
                    transformer_layer_id=event.transformer_layer_id,
                    request_ids=tuple(s.request_id for s in event.samples),
                    routed_token_counts=tuple(
                        sorted(event.routed_token_counts.items())
                    ),
                    transition=_empty_transition(
                        event.moe_layer_index, event.activated_experts
                    ),
                )
            )
    return records, num_experts, num_layers


def per_decode_coverage(
    records: Sequence[ReplayEventRecord],
    schedule: dict[EventKey, frozenset[int]],
    compared_keys: set[EventKey],
) -> list[dict]:
    """Mean routed-token coverage across MoE layers for each decode step."""
    by_step: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for record in records:
        if record.key not in compared_keys:
            continue
        residents = schedule[record.key]
        hit = 0
        total = 0
        for expert, count in record.routed_token_counts:
            total += count
            if expert in residents:
                hit += count
        by_step[int(record.key.generation_iteration_id)].append((hit, total))
    rows = []
    for step in sorted(by_step):
        hit = sum(h for h, _ in by_step[step])
        total = sum(t for _, t in by_step[step])
        rows.append(
            {
                "decode_step": step,
                "gpu_routed_token_coverage": (hit / total) if total else 0.0,
                "routed_token_hit": hit,
                "routed_token_total": total,
            }
        )
    return rows


def main() -> None:
    args = parse_args()
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    workload = args.workload_name or args.trace_dir.name

    records, num_experts, num_layers = load_records(
        args.trace_dir, logical_batch_size=args.logical_batch_size
    )
    if not records:
        raise SystemExit(f"no records loaded from {args.trace_dir}")

    # Warmup filter: intersection of ready sets for refresh-based policies.
    ready_sets = []
    schedules = {}
    for name in POLICIES:
        sched, ready = build_schedule(
            name,
            records,
            gpu_budget=args.gpu_budget,
            window_size=args.window_size,
            swap_frequency=args.swap_frequency,
        )
        schedules[name] = sched
        if name != "Static":
            ready_sets.append(ready)
    compared_keys = set.intersection(*(set(r) for r in ready_sets))
    compared = [r for r in records if r.key in compared_keys]
    if not compared:
        raise SystemExit("no events remain after warmup filtering")

    agg_rows = []
    for name in POLICIES:
        cov = _coverage(compared, {r.key: schedules[name][r.key] for r in compared})
        agg_rows.append(
            {
                "workload": workload,
                "policy": name,
                "gpu_routed_token_coverage": cov.gpu_routed_token_coverage,
                "activation_coverage": cov.activation_coverage,
                "ssd_routed_tokens": cov.ssd_routed_tokens,
                "n_events_compared": len(compared),
                "n_events_total": len(records),
                "gpu_budget": args.gpu_budget,
                "window_size": args.window_size,
                "swap_frequency": args.swap_frequency,
                "num_experts": num_experts,
                "num_layers": num_layers,
            }
        )
        step_rows = per_decode_coverage(records, schedules[name], compared_keys)
        step_path = out / f"per_step_{name}.csv"
        with step_path.open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(
                f,
                fieldnames=[
                    "decode_step",
                    "gpu_routed_token_coverage",
                    "routed_token_hit",
                    "routed_token_total",
                ],
            )
            w.writeheader()
            w.writerows(step_rows)

    agg_path = out / "aggregate_coverage.csv"
    with agg_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(agg_rows[0].keys()))
        w.writeheader()
        w.writerows(agg_rows)

    # Combined temporal CSV for plotting
    temporal_path = out / "temporal_coverage.csv"
    with temporal_path.open("w", encoding="utf-8", newline="") as f:
        fields = ["decode_step", "policy", "gpu_routed_token_coverage"]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for name in POLICIES:
            for row in per_decode_coverage(records, schedules[name], compared_keys):
                w.writerow(
                    {
                        "decode_step": row["decode_step"],
                        "policy": name,
                        "gpu_routed_token_coverage": row["gpu_routed_token_coverage"],
                    }
                )

    meta = {
        "workload": workload,
        "trace_dir": str(args.trace_dir),
        "n_records": len(records),
        "n_compared": len(compared),
        "policies": list(POLICIES),
        "gpu_budget": args.gpu_budget,
        "window_size": args.window_size,
        "swap_frequency": args.swap_frequency,
        "exclude_warmup_until_first_refresh": True,
        "metric": "gpu_routed_token_coverage",
        "model": "Qwen3-30B-A3B",
    }
    (out / "eval_meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(json.dumps(agg_rows, indent=2))
    print(f"[eval] wrote {agg_path}")


if __name__ == "__main__":
    main()
