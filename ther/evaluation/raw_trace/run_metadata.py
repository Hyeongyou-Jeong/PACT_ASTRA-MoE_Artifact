from __future__ import annotations

import hashlib
import json
import os
import platform
import sys
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Iterable

from .schema import FORMAT_VERSION


def add_raw_trace_arguments(parser: Any) -> None:
    parser.add_argument(
        "--raw-trace-dir",
        type=str,
        default="",
        help="Enable the repository-local vLLM plugin and write raw_trace.h5 here.",
    )
    parser.add_argument(
        "--trace-logits-dtype",
        choices=["float16", "float32"],
        default="float16",
    )
    parser.add_argument(
        "--trace-probabilities-dtype",
        choices=["float16", "float32"],
        default="float16",
    )
    parser.add_argument(
        "--trace-embeddings-dtype",
        choices=["float16", "float32"],
        default="float16",
    )
    parser.add_argument(
        "--trace-weights-dtype",
        choices=["float16", "float32"],
        default="float32",
    )
    parser.add_argument("--trace-queue-size", type=int, default=2)
    parser.add_argument("--trace-flush-every", type=int, default=1)
    parser.add_argument(
        "--trace-compression", choices=["lzf", "gzip", "none"], default="lzf"
    )
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--profile-train-ratio", type=float, default=0.7)
    parser.add_argument("--validation-ratio", type=float, default=0.1)
    parser.add_argument(
        "--trace-level",
        choices=["full", "fine_only", "routing_only"],
        default="full",
        help=(
            "full: embeddings+logits+probabilities+selected; "
            "fine_only: embeddings+probabilities+selected (FineMoE min); "
            "routing_only: selected experts/weights for Diff-MoE/THER"
        ),
    )


def deterministic_split(
    request_id: str,
    *,
    seed: int,
    train_ratio: float,
    validation_ratio: float,
) -> str:
    if train_ratio < 0 or validation_ratio < 0:
        raise ValueError("split ratios must be non-negative")
    if train_ratio + validation_ratio > 1:
        raise ValueError("train_ratio + validation_ratio must be <= 1")
    digest = hashlib.sha256(f"{seed}:{request_id}".encode()).digest()
    value = int.from_bytes(digest[:8], "big") / float(1 << 64)
    if value < train_ratio:
        return "profile/train"
    if value < train_ratio + validation_ratio:
        return "validation"
    return "test"


def enable_vllm_raw_trace(args: Any) -> Path | None:
    raw_dir = str(getattr(args, "raw_trace_dir", "") or "").strip()
    if not raw_dir:
        return None
    path = Path(raw_dir).resolve()
    path.mkdir(parents=True, exist_ok=True)
    os.environ["ASTRAMOE_RAW_TRACE_DIR"] = str(path)
    os.environ["ASTRAMOE_TRACE_LOGITS_DTYPE"] = args.trace_logits_dtype
    os.environ["ASTRAMOE_TRACE_PROBABILITIES_DTYPE"] = (
        args.trace_probabilities_dtype
    )
    os.environ["ASTRAMOE_TRACE_EMBEDDINGS_DTYPE"] = args.trace_embeddings_dtype
    os.environ["ASTRAMOE_TRACE_WEIGHTS_DTYPE"] = args.trace_weights_dtype
    os.environ["ASTRAMOE_TRACE_QUEUE_SIZE"] = str(args.trace_queue_size)
    os.environ["ASTRAMOE_TRACE_FLUSH_EVERY"] = str(args.trace_flush_every)
    os.environ["ASTRAMOE_TRACE_COMPRESSION"] = args.trace_compression
    os.environ["ASTRAMOE_TRACE_LEVEL"] = str(
        getattr(args, "trace_level", "full") or "full"
    )
    # This makes worker-side IDs identical to RequestOutput.request_id so the
    # trace and request completion manifest can be joined without guesswork.
    os.environ["VLLM_DISABLE_REQUEST_ID_RANDOMIZATION"] = "1"
    return path


def build_run_metadata(
    *,
    args: Any,
    hf_config: Any,
    dataset_name: str,
    request_ids: Iterable[str],
    generation_config: dict[str, Any],
) -> dict[str, Any]:
    model_name = str(args.model_name)
    revision, checkpoint = resolve_hf_revision(model_name, str(args.cache_dir))
    source_root = Path(__file__).resolve().parents[2]
    source_paths = [
        source_root / "ther/evaluation/raw_trace/schema.py",
        source_root / "ther/evaluation/raw_trace/writer.py",
        source_root / "ther/evaluation/raw_trace/vllm_plugin.py",
        source_root / "ther/trace_generation/qwen_integrateddataset.py",
        source_root / "ther/trace_generation/qwen_seperateddataset.py",
    ]
    moe_layers = [
        layer_id
        for layer_id in range(int(hf_config.num_hidden_layers))
        if layer_id not in set(getattr(hf_config, "mlp_only_layers", []))
        and (layer_id + 1) % int(hf_config.decoder_sparse_step) == 0
    ]
    return {
        "trace_format": "astramoe-qwen3-raw",
        "trace_format_version": FORMAT_VERSION,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "model_name": model_name,
        "huggingface_revision": revision,
        "checkpoint_path": checkpoint,
        "architecture": list(getattr(hf_config, "architectures", [])),
        "number_of_transformer_layers": int(hf_config.num_hidden_layers),
        "number_of_moe_layers": len(moe_layers),
        "transformer_moe_layer_ids": moe_layers,
        "number_of_experts_per_layer": int(hf_config.num_experts),
        "top_k": int(hf_config.num_experts_per_tok),
        "hidden_dimension": int(hf_config.hidden_size),
        "model_dtype": str(args.dtype),
        "trace_level": str(getattr(args, "trace_level", "full") or "full"),
        "trace_dtypes": {
            "router_logits": (
                None
                if str(getattr(args, "trace_level", "full"))
                in ("routing_only", "fine_only")
                else args.trace_logits_dtype
            ),
            "router_probabilities": (
                None
                if str(getattr(args, "trace_level", "full")) == "routing_only"
                else args.trace_probabilities_dtype
            ),
            "selected_expert_weights": args.trace_weights_dtype,
            "semantic_embedding": (
                None
                if str(getattr(args, "trace_level", "full")) == "routing_only"
                else args.trace_embeddings_dtype
            ),
            "selected_expert_ids": (
                "uint8" if int(hf_config.num_experts) <= 256 else "uint16"
            ),
        },
        "batch_size": int(args.batch_size),
        "tensor_parallel_size": int(args.tensor_parallel_size),
        "dataset_name": dataset_name,
        "request_ids": [str(value) for value in request_ids],
        "generation_configuration": generation_config,
        "random_seed": int(args.seed),
        "split": {
            "method": "sha256(seed:request_id) normalized to [0,1)",
            "seed": int(args.split_seed),
            "profile_train_ratio": float(args.profile_train_ratio),
            "validation_ratio": float(args.validation_ratio),
            "test_ratio": float(
                1.0 - args.profile_train_ratio - args.validation_ratio
            ),
        },
        "router_semantics": {
            "router_logits": "raw bias-free gate linear output",
            "router_probabilities": (
                "float32 softmax over all experts immediately before top-k; "
                "stored using configured trace dtype"
            ),
            "selected_expert_ids": (
                "actual logical top-k expert IDs selected by vLLM, before EPLB mapping"
            ),
            "selected_expert_weights": (
                "actual vLLM top-k weights used for expert output aggregation"
            ),
            "top_k_weights_normalized": bool(hf_config.norm_topk_prob),
        },
        "iteration_semantics": {
            "physical_stage": "prefill when token_position < prompt_length, else decode",
            "iteration_id": "-1 for prefill; token_position-prompt_length for decode",
            "generation_iteration_id": (
                "-1 before final prompt token; final prompt token is generation "
                "iteration 0; subsequent decode input tokens are 1,2,..."
            ),
        },
        "prefill_aggregation": None,
        "runtime": {
            "python": sys.version,
            "platform": platform.platform(),
            "vllm": package_version("vllm"),
            "transformers": package_version("transformers"),
            "torch": package_version("torch"),
            "numpy": package_version("numpy"),
            "h5py": package_version("h5py"),
        },
        "source_sha256": {
            str(path.relative_to(source_root)): sha256_file(path)
            for path in source_paths
            if path.exists()
        },
    }


def write_json_atomic(path: str | Path, value: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    temporary.replace(target)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def package_version(name: str) -> str:
    try:
        return version(name)
    except PackageNotFoundError:
        return "not-installed"


def resolve_hf_revision(model_name: str, cache_dir: str) -> tuple[str | None, str]:
    path = Path(model_name)
    if path.exists():
        return None, str(path.resolve())
    hub = Path(cache_dir) / f"models--{model_name.replace('/', '--')}"
    ref = hub / "refs/main"
    if ref.exists():
        revision = ref.read_text(encoding="utf-8").strip()
        snapshot = hub / "snapshots" / revision
        return revision or None, str(snapshot if snapshot.exists() else hub)
    return None, model_name
