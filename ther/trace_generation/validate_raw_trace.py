from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from ther.evaluation.raw_trace.loader import RawTraceLoader
from ther.evaluation.raw_trace.schema import Stage

try:
    from ther.trace_generation.convert_to_astra_trace import convert_raw_to_astra  # optional; not shipped
except ModuleNotFoundError:
    convert_raw_to_astra = None  # type: ignore


def _jsonl_tree(root: Path) -> dict[str, list[dict[str, Any]]]:
    return {
        str(path.relative_to(root)): [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        for path in sorted(root.rglob("layer_*.jsonl"))
    }


def validate_trace(
    trace: str | Path,
    *,
    legacy_dir: str | Path | None = None,
    probability_atol: float = 3e-3,
    weight_atol: float = 5e-3,
    chunk_tokens: int = 256,
) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    checks: dict[str, Any] = {}
    with RawTraceLoader(trace) as loader:
        handle = loader._handle
        n = loader.num_tokens
        layers = len(loader.moe_layer_ids)
        experts = loader.num_experts
        top_k = loader.top_k
        full = loader.stores_full_router
        finemoe = loader.stores_finemoe_tensors
        checks["trace_level"] = loader.trace_level
        expected_shapes = {
            "routing/selected_expert_ids": (n, layers, top_k),
            "routing/selected_expert_weights": (n, layers, top_k),
        }
        if finemoe:
            expected_shapes.update(
                {
                    "embedding": (n, int(handle.attrs["hidden_size"])),
                    "routing/router_probabilities": (n, layers, experts),
                }
            )
        if full:
            expected_shapes["routing/router_logits"] = (n, layers, experts)
        if loader.trace_level == "routing_only":
            for path in (
                "embedding",
                "routing/router_logits",
                "routing/router_probabilities",
            ):
                if path in handle:
                    errors.append(
                        f"routing_only trace unexpectedly contains dataset {path}"
                    )
        elif loader.trace_level == "fine_only":
            if "routing/router_logits" in handle:
                errors.append(
                    "fine_only trace unexpectedly contains routing/router_logits"
                )
        for path, shape in expected_shapes.items():
            if path not in handle or handle[path].shape != shape:
                actual = handle[path].shape if path in handle else None
                errors.append(f"{path}: shape {actual} != {shape}")
        checks["shape"] = not any("shape" in item for item in errors)

        max_probability_sum_error = 0.0
        topk_mismatches = 0
        weight_mismatches = 0
        nonfinite_values = 0
        assignment_count = 0
        norm_topk = bool(
            loader.metadata.get("router_semantics", {}).get(
                "top_k_weights_normalized", handle.attrs.get("norm_topk_prob", False)
            )
        )
        for start in range(0, n, max(1, chunk_tokens)):
            stop = min(n, start + max(1, chunk_tokens))
            selected_ids = handle["routing/selected_expert_ids"][start:stop].astype(
                np.int64
            )
            selected_weights = handle["routing/selected_expert_weights"][
                start:stop
            ].astype(np.float32)
            nonfinite_values += int(
                np.size(selected_weights)
                - np.count_nonzero(np.isfinite(selected_weights))
            )
            if np.any((selected_ids < 0) | (selected_ids >= experts)):
                errors.append(f"selected expert ID out of range in rows {start}:{stop}")
            assignment_count += int(selected_ids.size)
            if not finemoe:
                continue
            probabilities = handle["routing/router_probabilities"][start:stop].astype(
                np.float32
            )
            nonfinite_values += int(
                np.size(probabilities)
                - np.count_nonzero(np.isfinite(probabilities))
            )
            sums = probabilities.sum(axis=-1)
            max_probability_sum_error = max(
                max_probability_sum_error, float(np.max(np.abs(sums - 1.0)))
            )
            selected_probability = np.take_along_axis(
                probabilities, selected_ids, axis=-1
            )
            expected_weights = selected_probability
            if norm_topk:
                denominator = expected_weights.sum(axis=-1, keepdims=True)
                expected_weights = expected_weights / np.maximum(denominator, 1e-12)
            weight_mismatches += int(
                np.count_nonzero(
                    np.max(np.abs(expected_weights - selected_weights), axis=-1)
                    > weight_atol
                )
            )
            if full:
                logits = handle["routing/router_logits"][start:stop].astype(np.float32)
                nonfinite_values += int(
                    np.size(logits) - np.count_nonzero(np.isfinite(logits))
                )
                selected_logits = np.take_along_axis(logits, selected_ids, axis=-1)
                threshold = np.min(selected_logits, axis=-1)
                mask = np.ones(logits.shape, dtype=bool)
                np.put_along_axis(mask, selected_ids, False, axis=-1)
                largest_unselected = np.max(
                    np.where(mask, logits, -np.inf), axis=-1
                )
                # Float16 logits can tie at the top-k boundary. Such ties are valid
                # because the original float32 decision is preserved in selected IDs.
                topk_mismatches += int(
                    np.count_nonzero(largest_unselected > threshold + 2e-3)
                )

        if finemoe and max_probability_sum_error > probability_atol:
            errors.append(
                f"probability sum max error {max_probability_sum_error:.6g} "
                f"> {probability_atol}"
            )
        if full and topk_mismatches:
            errors.append(f"{topk_mismatches} token/layer top-k mismatches")
        if finemoe and weight_mismatches:
            errors.append(f"{weight_mismatches} token/layer weight mismatches")
        if nonfinite_values:
            errors.append(f"{nonfinite_values} non-finite routing values")
        expected_assignments = n * layers * top_k
        if assignment_count != expected_assignments:
            errors.append(
                f"route assignments {assignment_count} != {expected_assignments}"
            )
        checks.update(
            {
                "probability_sum": (
                    True
                    if not finemoe
                    else max_probability_sum_error <= probability_atol
                ),
                "top_k_consistency": True if not full else topk_mismatches == 0,
                "selected_weight_consistency": (
                    True if not finemoe else weight_mismatches == 0
                ),
                "routed_token_consistency": assignment_count == expected_assignments,
                "max_probability_sum_error": max_probability_sum_error,
                "route_assignment_count": assignment_count,
                "full_router_checks_skipped": not full,
                "finemoe_checks_skipped": not finemoe,
            }
        )

        known_requests = set(loader.requests)
        traced_requests = set(loader._request_ids.tolist())
        if known_requests:
            unknown = traced_requests - known_requests
            missing = known_requests - traced_requests
            if unknown:
                errors.append(f"trace contains unknown request IDs: {sorted(unknown)}")
            if missing:
                errors.append(f"request manifest IDs missing from trace: {sorted(missing)}")
        else:
            warnings.append("requests.json absent; request/output mapping not validated")

        iteration_errors = 0
        position_errors = 0
        stage_errors = 0
        for request_id, request in loader.requests.items():
            indices = loader.token_indices(request_id)
            positions = handle["index/token_position"][indices].astype(int)
            expected_last = int(request["input_length"]) + int(
                request["output_length"]
            ) - 2
            expected_positions = np.arange(max(0, expected_last + 1))
            if not np.array_equal(np.unique(positions), expected_positions):
                position_errors += 1
            generation = handle["index/generation_iteration_id"][indices].astype(int)
            expected_generation = np.arange(int(request["output_length"]))
            actual_generation = np.sort(generation[generation >= 0])
            if not np.array_equal(actual_generation, expected_generation):
                iteration_errors += 1
            prompt_length = int(request["input_length"])
            stages = handle["index/stage"][indices].astype(int)
            expected_stages = np.where(
                positions < prompt_length, Stage.PREFILL, Stage.DECODE
            )
            if not np.array_equal(stages, expected_stages):
                stage_errors += 1
        if position_errors:
            errors.append(f"{position_errors} requests have incomplete token positions")
        if iteration_errors:
            errors.append(
                f"{iteration_errors} requests have non-sequential generation iterations"
            )
        if stage_errors:
            errors.append(f"{stage_errors} requests have incorrect prefill/decode stage")
        checks["request_mapping"] = not position_errors and not (
            traced_requests - known_requests if known_requests else False
        )
        checks["iteration_consistency"] = iteration_errors == 0
        checks["stage_consistency"] = stage_errors == 0
        layer_paths = [
            "routing/selected_expert_ids",
            "routing/selected_expert_weights",
        ]
        if full:
            layer_paths = [
                "routing/router_logits",
                "routing/router_probabilities",
                *layer_paths,
            ]
        checks["layer_completeness"] = all(
            handle[path].shape[1] == layers for path in layer_paths
        )

    if legacy_dir is not None:
        if convert_raw_to_astra is None:
            raise ModuleNotFoundError(
                "legacy_dir validation requires convert_to_astra_trace, "
                "which is outside the public THER artifact scope"
            )
        legacy_root = Path(legacy_dir)
        with tempfile.TemporaryDirectory(prefix="astramoe-legacy-") as temporary:
            converted = Path(temporary)
            convert_raw_to_astra(trace, converted)
            if list(legacy_root.glob("layer_*.jsonl")):
                converted_datasets = [
                    path for path in converted.iterdir() if path.is_dir()
                ]
                actual_root = (
                    converted_datasets[0]
                    if len(converted_datasets) == 1
                    else converted
                )
            else:
                actual_root = converted
            actual = _jsonl_tree(actual_root)
            expected = _jsonl_tree(legacy_root)
            checks["legacy_astra_compatibility"] = actual == expected
            if actual != expected:
                actual_keys = set(actual)
                expected_keys = set(expected)
                errors.append(
                    "legacy ASTRA mismatch: "
                    f"missing_files={sorted(expected_keys - actual_keys)}, "
                    f"extra_files={sorted(actual_keys - expected_keys)}"
                )
    else:
        checks["legacy_astra_compatibility"] = None

    return {
        "valid": not errors,
        "errors": errors,
        "warnings": warnings,
        "checks": checks,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate a Qwen3 raw routing trace.")
    parser.add_argument("trace")
    parser.add_argument("--legacy-dir")
    parser.add_argument("--report")
    parser.add_argument("--probability-atol", type=float, default=3e-3)
    parser.add_argument("--weight-atol", type=float, default=5e-3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = validate_trace(
        args.trace,
        legacy_dir=args.legacy_dir,
        probability_atol=args.probability_atol,
        weight_atol=args.weight_atol,
    )
    rendered = json.dumps(report, indent=2)
    print(rendered)
    if args.report:
        Path(args.report).write_text(rendered + "\n", encoding="utf-8")
    if not report["valid"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
