from __future__ import annotations

from collections import Counter
from typing import Any

import numpy as np

from .event import BatchLayerEvent, PerRequestExperts


def synthetic_four_request_expectation() -> dict[str, Any]:
    """Canonical 4-request composition used in unit tests / docs."""

    per_request = [
        PerRequestExperts("R0", (1, 2)),
        PerRequestExperts("R1", (1, 3)),
        PerRequestExperts("R2", (1, 3)),
        PerRequestExperts("R3", (1, 4)),
    ]
    event = BatchLayerEvent.from_per_request(
        logical_batch_id=0,
        decode_iteration=0,
        layer_id=0,
        per_request=per_request,
    )
    return {
        "activated_experts": set(event.activated_experts),
        "batch_activation": {
            expert: 1 for expert in sorted(event.activated_experts)
        },
        "routed_tokens": dict(event.routed_tokens_per_expert),
        "diff_priority_delta": {
            expert: 1 for expert in sorted(event.activated_experts)
        },
        "ther_load": dict(event.routed_tokens_per_expert),
        "total_route_assignments": event.total_route_assignments,
        "event": event,
    }


def validate_synthetic_composition() -> dict[str, Any]:
    got = synthetic_four_request_expectation()
    expected_activated = {1, 2, 3, 4}
    expected_routed = {1: 4, 2: 1, 3: 2, 4: 1}
    ok = (
        got["activated_experts"] == expected_activated
        and got["routed_tokens"] == expected_routed
        and got["batch_activation"] == {1: 1, 2: 1, 3: 1, 4: 1}
        and got["diff_priority_delta"] == {1: 1, 2: 1, 3: 1, 4: 1}
        and got["ther_load"] == expected_routed
        and got["total_route_assignments"] == 8
    )
    return {
        "ok": ok,
        "expected_activated_experts": sorted(expected_activated),
        "expected_routed_tokens": expected_routed,
        "got_activated_experts": sorted(got["activated_experts"]),
        "got_routed_tokens": got["routed_tokens"],
    }


def diagnose_logical_batch(
    events: list[BatchLayerEvent],
    *,
    max_events: int = 5,
    expected_batch_size: int = 16,
    expected_top_k: int = 8,
) -> dict[str, Any]:
    if not events:
        return {"ok": False, "error": "no events"}
    sample = events[:max_events]
    rows = []
    ok = True
    expected_routes = expected_batch_size * expected_top_k
    for event in sample:
        row = {
            "logical_batch_id": event.logical_batch_id,
            "decode_iteration": event.decode_iteration,
            "layer_id": event.layer_id,
            "logical_batch_size": event.logical_batch_size,
            "total_route_assignments": event.total_route_assignments,
            "unique_activated_experts": len(event.activated_experts),
            "max_routed_tokens_per_expert": max(
                event.routed_tokens_per_expert.values(), default=0
            ),
            "mean_routed_tokens_per_activated_expert": (
                (
                    sum(event.routed_tokens_per_expert.values())
                    / max(1, len(event.activated_experts))
                )
                if event.activated_experts
                else 0.0
            ),
        }
        if event.logical_batch_size != expected_batch_size:
            ok = False
            row["error"] = "batch_size_mismatch"
        if event.total_route_assignments != expected_routes:
            ok = False
            row["error"] = "route_assignment_mismatch"
        if len(event.activated_experts) > expected_routes:
            ok = False
            row["error"] = "activated_exceeds_routes"
        rows.append(row)
    return {
        "ok": ok,
        "expected_logical_batch_size": expected_batch_size,
        "expected_route_assignments": expected_routes,
        "sampled_events": rows,
    }


def frequency_vs_load_from_events(
    events: list[BatchLayerEvent],
    *,
    window: int,
) -> dict[str, Any]:
    """Windowed activation frequency vs routed-token load (Spearman-ready)."""

    if window <= 0:
        raise ValueError("window must be positive")
    # Group by (batch, layer) then slide over decode iterations.
    by_key: dict[tuple[int, int], list[BatchLayerEvent]] = {}
    for event in events:
        by_key.setdefault((event.logical_batch_id, event.layer_id), []).append(event)
    for key in by_key:
        by_key[key].sort(key=lambda event: event.decode_iteration)

    from ther.evaluation.stats import spearman_rank_correlation

    spearman_values: list[float] = []
    tokens_per_activation: list[float] = []
    mismatch_examples: list[dict[str, Any]] = []
    for (batch_id, layer_id), series in by_key.items():
        for start in range(0, max(0, len(series) - window + 1)):
            window_events = series[start : start + window]
            activation = Counter()
            routed = Counter()
            for event in window_events:
                activation.update(event.activated_experts)
                routed.update(event.routed_tokens_per_expert)
            experts = sorted(set(activation) | set(routed))
            if len(experts) < 2:
                continue
            act_vec = np.asarray(
                [float(activation.get(expert, 0)) for expert in experts],
                dtype=np.float64,
            )
            route_vec = np.asarray(
                [float(routed.get(expert, 0)) for expert in experts],
                dtype=np.float64,
            )
            corr = float(spearman_rank_correlation(act_vec, route_vec))
            if corr == corr:  # not NaN
                spearman_values.append(corr)
            for expert in experts:
                a = activation.get(expert, 0)
                r = routed.get(expert, 0)
                if a > 0:
                    tokens_per_activation.append(r / a)
            # Rank mismatch: high activation rank, low load rank (or reverse).
            act_rank = {
                expert: rank
                for rank, expert in enumerate(
                    sorted(experts, key=lambda e: (-activation.get(e, 0), e))
                )
            }
            load_rank = {
                expert: rank
                for rank, expert in enumerate(
                    sorted(experts, key=lambda e: (-routed.get(e, 0), e))
                )
            }
            for expert in experts:
                delta = abs(act_rank[expert] - load_rank[expert])
                if delta >= max(3, len(experts) // 4) and len(mismatch_examples) < 20:
                    mismatch_examples.append(
                        {
                            "logical_batch_id": batch_id,
                            "layer_id": layer_id,
                            "window_start": start,
                            "expert": expert,
                            "activation_count": activation.get(expert, 0),
                            "routed_tokens": routed.get(expert, 0),
                            "activation_rank": act_rank[expert],
                            "load_rank": load_rank[expert],
                        }
                    )

    def _summary(values: list[float]) -> dict[str, float]:
        if not values:
            return {
                "mean": 0.0,
                "median": 0.0,
                "p10": 0.0,
                "p90": 0.0,
                "min": 0.0,
                "max": 0.0,
                "count": 0,
            }
        import numpy as np

        arr = np.asarray(values, dtype=np.float64)
        return {
            "mean": float(arr.mean()),
            "median": float(np.median(arr)),
            "p10": float(np.percentile(arr, 10)),
            "p90": float(np.percentile(arr, 90)),
            "min": float(arr.min()),
            "max": float(arr.max()),
            "count": int(arr.size),
        }

    return {
        "window": window,
        "spearman": _summary(spearman_values),
        "tokens_per_activation": _summary(tokens_per_activation),
        "mismatch_examples": mismatch_examples[:10],
    }
