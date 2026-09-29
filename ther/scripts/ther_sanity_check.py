#!/usr/bin/env python3
"""Lightweight THER-facing sanity checks on a collected raw_trace directory."""

from __future__ import annotations

from ther.scripts.path_guard import apply as _apply_path_guard
_apply_path_guard()

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from ther.evaluation.raw_trace.loader import RawTraceLoader
from ther.evaluation.raw_trace.schema import Stage
from ther.trace_generation.validate_raw_trace import validate_trace


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("trace_dir", type=Path)
    p.add_argument("--expect-requests", type=int, default=16)
    p.add_argument("--expect-layers", type=int, default=48)
    p.add_argument("--expect-experts", type=int, default=128)
    p.add_argument("--expect-topk", type=int, default=8)
    p.add_argument("--min-decode-steps", type=int, default=1)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    trace_dir = args.trace_dir.resolve()
    errors: list[str] = []

    base = validate_trace(trace_dir)
    if base.get("errors"):
        errors.extend(base["errors"])

    with RawTraceLoader(trace_dir) as loader:
        h = loader._handle
        n_tok = loader.num_tokens
        layers = len(loader.moe_layer_ids)
        experts = loader.num_experts
        top_k = loader.top_k
        req_ids = sorted(set(loader._request_ids.tolist()))
        stages = h["index/stage"][:].astype(int)
        gen_steps = h["index/generation_iteration_id"][:].astype(int)
        decode_mask = stages == int(Stage.DECODE)
        decode_steps = sorted(
            {int(s) for s, m in zip(gen_steps.tolist(), decode_mask.tolist()) if m}
        )

        if layers != args.expect_layers:
            errors.append(f"moe layers={layers} expected {args.expect_layers}")
        if experts != args.expect_experts:
            errors.append(f"experts={experts} expected {args.expect_experts}")
        if top_k != args.expect_topk:
            errors.append(f"top_k={top_k} expected {args.expect_topk}")
        if len(req_ids) != args.expect_requests:
            errors.append(
                f"request_ids={len(req_ids)} expected {args.expect_requests}"
            )
        if len(decode_steps) < args.min_decode_steps:
            errors.append(f"decode_steps={len(decode_steps)} too few")
        if decode_steps:
            expected = set(range(min(decode_steps), max(decode_steps) + 1))
            missing = sorted(expected - set(decode_steps))
            continuity_ok = len(missing) == 0
        else:
            continuity_ok = False
            missing = []

        ids = np.asarray(h["routing/selected_expert_ids"][:])
        if ids.size and (int(ids.min()) < 0 or int(ids.max()) >= experts):
            errors.append(
                f"expert id out of range [{ids.min()}, {ids.max()}] vs {experts}"
            )
        if ids.ndim >= 3 and ids.shape[-1] != top_k:
            errors.append(f"selected_expert last dim {ids.shape[-1]} != top_k {top_k}")
        if ids.shape[0] != n_tok:
            errors.append(f"token rows {ids.shape[0]} != num_tokens {n_tok}")
        if ids.shape[1] != layers:
            errors.append(f"layer dim {ids.shape[1]} != moe layers {layers}")

        report = {
            "trace_dir": str(trace_dir),
            "ok": not errors,
            "errors": errors,
            "warnings": list(base.get("warnings") or []),
            "num_tokens": int(n_tok),
            "num_requests": len(req_ids),
            "request_ids": req_ids,
            "num_moe_layers": layers,
            "num_experts": experts,
            "top_k": top_k,
            "decode_step_min": min(decode_steps) if decode_steps else None,
            "decode_step_max": max(decode_steps) if decode_steps else None,
            "decode_step_count": len(decode_steps),
            "decode_continuity_ok": continuity_ok,
            "decode_missing_steps": missing[:20],
            "trace_level": loader.trace_level,
            "base_validate_errors": base.get("errors"),
            "base_checks": base.get("checks"),
        }

    out = trace_dir / "ther_sanity.json"
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if errors:
        print(f"[ther_sanity] FAIL ({len(errors)} errors)", file=sys.stderr)
        return 1
    print("[ther_sanity] OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
