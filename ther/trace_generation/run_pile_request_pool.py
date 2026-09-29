from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

from vllm import LLM, SamplingParams
from vllm.inputs import TokensPrompt

from ther.trace_generation.logical_batch.composer import LogicalBatchComposer
from ther.trace_generation.logical_batch.validation import diagnose_logical_batch
from ther.trace_generation.pile_workload.builder import (
    WorkloadConfig,
    build_workload_manifest,
    estimate_workload_cost,
    load_workload_manifest,
)
from ther.evaluation.raw_trace.run_metadata import (
    add_raw_trace_arguments,
    build_run_metadata,
    enable_vllm_raw_trace,
    write_json_atomic,
)

FIXED_LENGTH_SEMANTICS = (
    "EOS/stop generation is suppressed until the configured max_tokens "
    "generated tokens. This is a controlled fixed-length decode workload and "
    "does not represent the natural output-length distribution."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Collect physical BS=1 Qwen3 request traces for offline logical "
            "batch composition (logical BS configurable, default 16)."
        )
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--manifest", default="")
    parser.add_argument("--num-batches", type=int, default=1)
    parser.add_argument(
        "--logical-batch-size",
        type=int,
        default=16,
        help="Offline simulation batch size (not physical vLLM batch size)",
    )
    parser.add_argument("--input-length", type=int, default=4096)
    parser.add_argument("--output-length", type=int, default=4096)
    parser.add_argument("--sources-per-batch", type=int, default=8)
    parser.add_argument("--requests-per-source", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--crop-seed", type=int, default=42)
    parser.add_argument("--max-scan-per-source", type=int, default=50_000)
    parser.add_argument("--model-name", default="Qwen/Qwen3-30B-A3B")
    parser.add_argument("--cache-dir", default="")
    parser.add_argument("--hf-token", default=os.getenv("HF_TOKEN", ""))
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--tensor-parallel-size", type=int, default=2)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.95)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--estimate-only", action="store_true")
    parser.add_argument("--build-manifest-only", action="store_true")
    parser.add_argument(
        "--logical-batch-ids",
        default="",
        help="Comma-separated logical batch IDs to collect; empty = all",
    )
    parser.add_argument(
        "--compose-after",
        action="store_true",
        help="After tracing, build logical_batch_manifest.json and diagnostics",
    )
    add_raw_trace_arguments(parser)
    parser.set_defaults(trace_level="routing_only")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    physical_batch_size = 1
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.logical_batch_size != args.sources_per_batch * args.requests_per_source:
        raise ValueError(
            "logical_batch_size must equal sources_per_batch * requests_per_source"
        )
    config = WorkloadConfig(
        model_name=args.model_name,
        batch_size=args.logical_batch_size,
        input_length=args.input_length,
        output_length=args.output_length,
        num_batches=args.num_batches,
        sources_per_batch=args.sources_per_batch,
        requests_per_source=args.requests_per_source,
        seed=args.seed,
        crop_seed=args.crop_seed,
        split_seed=args.split_seed,
        profile_train_ratio=args.profile_train_ratio,
        validation_ratio=args.validation_ratio,
        max_scan_per_source=args.max_scan_per_source,
        tokenizer_cache_dir=args.cache_dir,
        hf_token=args.hf_token,
        selection_rationale=(
            "Fixed offline logical composition (default 8 sources × 2 requests). "
            "Domain mix is a workload construction choice, not an experimental "
            "independent variable. Traces are collected at physical BS=1."
        ),
    )
    cost = estimate_workload_cost(config)
    cost["physical_batch_size"] = physical_batch_size
    cost["logical_batch_size"] = args.logical_batch_size
    print(json.dumps({"cost_estimate": cost}, indent=2))
    if args.estimate_only:
        return

    manifest_path = (
        Path(args.manifest)
        if args.manifest
        else output_dir / "workload_manifest.json"
    )
    if args.manifest:
        manifest = load_workload_manifest(manifest_path)
    else:
        print("[workload] building request pool / logical-batch membership...")
        manifest = build_workload_manifest(config)
        manifest.save(manifest_path)
        print(f"[workload] wrote {manifest_path}")
    if args.build_manifest_only:
        return

    batch_ids = (
        [int(value) for value in args.logical_batch_ids.split(",") if value.strip()]
        if args.logical_batch_ids.strip()
        else list(range(config.num_batches))
    )
    selected_requests = [
        request
        for request in manifest.requests
        if request.logical_batch_id in set(batch_ids)
    ]
    selected_requests.sort(
        key=lambda row: (row.logical_batch_id, row.batch_position, row.request_id)
    )
    write_json_atomic(
        output_dir / "workload_manifest.selected.json",
        {
            **manifest.to_json(),
            "selected_logical_batch_ids": batch_ids,
            "physical_batch_size": physical_batch_size,
            "logical_batch_size": args.logical_batch_size,
            "requests": [request.to_json() for request in selected_requests],
        },
    )

    args.batch_size = args.logical_batch_size
    args.seed = config.seed
    if not str(getattr(args, "raw_trace_dir", "") or "").strip():
        args.raw_trace_dir = str(output_dir)
    raw_trace_dir = enable_vllm_raw_trace(args)
    assert raw_trace_dir is not None

    started = time.perf_counter()
    llm = LLM(
        model=args.model_name,
        tensor_parallel_size=args.tensor_parallel_size,
        dtype=args.dtype,
        enforce_eager=True,
        trust_remote_code=True,
        enable_return_routed_experts=True,
        download_dir=args.cache_dir or None,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        max_num_seqs=physical_batch_size,
        disable_custom_all_reduce=True,
        enable_prefix_caching=False,
    )
    mc = llm.llm_engine.vllm_config.model_config.hf_text_config
    generation_config = {
        "temperature": 0.0,
        "min_tokens": config.output_length,
        "max_tokens": config.output_length,
        "ignore_eos": False,
        "semantics": FIXED_LENGTH_SEMANTICS,
    }
    write_json_atomic(
        raw_trace_dir / "metadata.json",
        {
            **build_run_metadata(
                args=args,
                hf_config=mc,
                dataset_name="pile_request_pool_bs1",
                request_ids=[request.request_id for request in selected_requests],
                generation_config=generation_config,
            ),
            "workload_manifest": str(manifest_path),
            "selected_logical_batch_ids": batch_ids,
            "trace_level": args.trace_level,
            "prefix_caching_enabled": False,
            "physical_batch_size": physical_batch_size,
            "logical_batch_size": args.logical_batch_size,
            "trace_collection": {
                "physical_batch_size": physical_batch_size,
                "note": (
                    "Each request is inferred independently with physical BS=1. "
                    "Do not interpret this as a physical BS=16 measurement."
                ),
            },
            "simulation": {
                "logical_batch_size": args.logical_batch_size,
                "note": (
                    "We collect per-request routing traces from Qwen3 and compose "
                    "them into batched workloads offline for trace-driven simulation."
                ),
            },
            "workload": {
                "input_tokens": config.input_length,
                "output_tokens": config.output_length,
            },
        },
    )
    sampling = SamplingParams(
        temperature=0.0,
        min_tokens=config.output_length,
        max_tokens=config.output_length,
        ignore_eos=False,
    )
    request_manifest: list[dict[str, Any]] = []
    per_request_latency: list[dict[str, Any]] = []
    for request in selected_requests:
        prompt = TokensPrompt(prompt_token_ids=list(request.input_token_ids))
        req_started = time.perf_counter()
        outputs = llm.generate([prompt], sampling)
        latency = time.perf_counter() - req_started
        if len(outputs) != 1:
            raise RuntimeError(
                f"{request.request_id}: expected 1 output, got {len(outputs)}"
            )
        req_out = outputs[0]
        prompt_token_ids = [int(value) for value in req_out.prompt_token_ids]
        output_token_ids = [int(value) for value in req_out.outputs[0].token_ids]
        if len(prompt_token_ids) != config.input_length:
            raise RuntimeError(
                f"{request.request_id}: prompt length "
                f"{len(prompt_token_ids)} != {config.input_length}"
            )
        if len(output_token_ids) != config.output_length:
            raise RuntimeError(
                f"{request.request_id}: output length "
                f"{len(output_token_ids)} != {config.output_length}"
            )
        request_manifest.append(
            {
                "request_id": str(req_out.request_id),
                "logical_request_id": request.request_id,
                "prompt_index": (
                    request.logical_batch_id * config.batch_size
                    + request.batch_position
                    + 1
                ),
                "dataset": "pile_request_pool_bs1",
                "split": request.split,
                "logical_batch_id": request.logical_batch_id,
                "batch_position": request.batch_position,
                "source_key": request.source_key,
                "pile_set_name": request.pile_set_name,
                "category": request.category,
                "document_id": request.document_id,
                "document_content_sha256": request.document_content_sha256,
                "crop_start_token": request.crop_start_token,
                "source_token_count": request.source_token_count,
                "input_length": len(prompt_token_ids),
                "output_length": len(output_token_ids),
                "target_output_tokens": config.output_length,
                "physical_batch_size": physical_batch_size,
                "prompt_token_ids": prompt_token_ids,
                "output_token_ids": output_token_ids,
            }
        )
        per_request_latency.append(
            {
                "logical_request_id": request.request_id,
                "vllm_request_id": str(req_out.request_id),
                "latency_seconds": latency,
                "generated_tokens": config.output_length,
            }
        )
        write_json_atomic(raw_trace_dir / "requests.json", request_manifest)
        print(
            f"[inference][bs1] {request.request_id} "
            f"latency_s={latency:.2f} vllm_id={req_out.request_id}"
        )

    # Flush writer attrs before reading storage stats.
    try:
        del llm
    except Exception:
        pass
    total_latency = time.perf_counter() - started
    h5_path = raw_trace_dir / "raw_trace.h5"
    write_seconds = 0.0
    tokens_written = 0
    trace_level = args.trace_level
    for _ in range(60):
        try:
            import h5py

            with h5py.File(h5_path, "r") as handle:
                write_seconds = float(handle.attrs.get("write_seconds", 0.0))
                tokens_written = int(handle.attrs.get("tokens_written", 0))
                trace_level = str(handle.attrs.get("trace_level", args.trace_level))
            if tokens_written > 0 and write_seconds > 0:
                break
        except Exception:
            pass
        time.sleep(0.5)
    trace_bytes = h5_path.stat().st_size if h5_path.exists() else 0
    generated_tokens = sum(row["output_length"] for row in request_manifest)
    report = {
        "output_dir": str(output_dir),
        "trace_level": trace_level,
        "physical_batch_size": physical_batch_size,
        "logical_batch_size": args.logical_batch_size,
        "selected_logical_batch_ids": batch_ids,
        "requests": len(request_manifest),
        "generated_tokens": generated_tokens,
        "total_wall_seconds": total_latency,
        "per_request_latency": per_request_latency,
        "trace_bytes": trace_bytes,
        "bytes_per_request": trace_bytes / max(1, len(request_manifest)),
        "bytes_per_generated_token": trace_bytes / max(1, generated_tokens),
        "write_seconds": write_seconds,
        "write_bandwidth_bytes_per_s": (
            trace_bytes / write_seconds if write_seconds > 0 else None
        ),
        "tokens_written": tokens_written,
        "generation_configuration": generation_config,
        "prefix_caching_enabled": False,
        "trace_collection": {"physical_batch_size": physical_batch_size},
        "simulation": {"logical_batch_size": args.logical_batch_size},
    }
    write_json_atomic(output_dir / "run_report.json", report)

    if args.compose_after:
        with LogicalBatchComposer(
            output_dir, logical_batch_size=args.logical_batch_size
        ) as composer:
            logical_manifest = composer.build_manifest(
                seed=config.seed, physical_batch_size=physical_batch_size
            )
            logical_manifest.save(output_dir / "logical_batch_manifest.json")
            events = list(composer.iter_batch_layer_events())
            diag = diagnose_logical_batch(
                events,
                max_events=5,
                expected_batch_size=args.logical_batch_size,
                expected_top_k=8,
            )
            write_json_atomic(output_dir / "logical_batch_diagnostics.json", diag)
            report["logical_batch_diagnostics"] = diag
            write_json_atomic(output_dir / "run_report.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
