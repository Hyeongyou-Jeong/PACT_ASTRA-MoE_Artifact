"""vLLM MoE router hook for astramoe routing_only traces.

Monkey-patches Qwen3 MoE router execution to record selected expert ids/weights.
This is integration code around vLLM APIs, not a vendored copy of vLLM itself.
"""

from __future__ import annotations

import atexit
import functools
import logging
import os
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .schema import (
    TRACE_LEVEL_FINE_ONLY,
    TRACE_LEVEL_FULL,
    TRACE_LEVEL_ROUTING_ONLY,
    TRACE_LEVELS,
    Stage,
)
from .writer import AsyncHDF5TraceWriter, TraceChunk

logger = logging.getLogger(__name__)
_PATCHED = False


@dataclass
class _ForwardIndex:
    request_ids: list[str]
    batch_id: np.ndarray
    batch_position: np.ndarray
    stage: np.ndarray
    iteration_id: np.ndarray
    generation_iteration_id: np.ndarray
    token_position: np.ndarray
    token_id: np.ndarray
    forward_id: np.ndarray


class _VLLMRawTraceCollector:
    def __init__(self) -> None:
        self.trace_dir = os.getenv("ASTRAMOE_RAW_TRACE_DIR", "").strip()
        self.enabled = bool(self.trace_dir)
        self._lock = threading.Lock()
        self._configured = False
        self._writer: AsyncHDF5TraceWriter | None = None
        self._router_layers: dict[int, int] = {}
        self._layer_ids: list[int] = []
        self._forward_counter = 0
        self._current: dict[str, Any] | None = None
        self._is_writer_rank = False
        level = os.getenv("ASTRAMOE_TRACE_LEVEL", TRACE_LEVEL_FULL).strip()
        self.trace_level = level if level in TRACE_LEVELS else TRACE_LEVEL_FULL

    def configure(self, runner: Any, modules: list[tuple[Any, int]]) -> None:
        if not self.enabled or self._configured:
            return
        try:
            from vllm.distributed import get_tensor_model_parallel_rank

            tp_rank = int(get_tensor_model_parallel_rank())
        except Exception:
            tp_rank = int(getattr(runner.parallel_config, "tensor_parallel_rank", 0))
        dp_rank = int(getattr(runner.parallel_config, "data_parallel_rank", 0))
        self._is_writer_rank = tp_rank == 0
        if not self._is_writer_rank:
            self._configured = True
            return

        from vllm.model_executor.layers.fused_moe.router.fused_topk_router import (
            FusedTopKRouter,
        )

        ordered = sorted(modules, key=lambda pair: int(pair[1]))
        if not ordered:
            raise RuntimeError("raw tracing requested but no FusedMoE routers were found")
        for router, layer_id in ordered:
            if not isinstance(router, FusedTopKRouter):
                raise RuntimeError(
                    f"unsupported router {type(router).__name__}; raw trace currently "
                    "requires vLLM FusedTopKRouter"
                )
            if router.scoring_func != "softmax":
                raise RuntimeError(
                    f"unsupported routing score {router.scoring_func!r}; expected softmax"
                )
            self._router_layers[id(router)] = int(layer_id)
        self._layer_ids = [int(layer_id) for _, layer_id in ordered]

        config = runner.vllm_config.model_config.hf_text_config
        if getattr(config, "model_type", None) != "qwen3_moe":
            raise RuntimeError("raw trace plugin currently supports model_type=qwen3_moe")
        if bool(getattr(runner.parallel_config, "enable_eplb", False)):
            logger.warning(
                "EPLB is enabled: selected_expert_ids contain logical IDs captured "
                "before logical-to-physical mapping"
            )
        output = Path(self.trace_dir)
        output.mkdir(parents=True, exist_ok=True)
        path = output / ("raw_trace.h5" if dp_rank == 0 else f"raw_trace.dp{dp_rank}.h5")
        self._writer = AsyncHDF5TraceWriter(
            path,
            num_moe_layers=len(self._layer_ids),
            num_experts=int(config.num_experts),
            top_k=int(config.num_experts_per_tok),
            hidden_size=int(config.hidden_size),
            transformer_layer_ids=self._layer_ids,
            logits_dtype=os.getenv("ASTRAMOE_TRACE_LOGITS_DTYPE", "float16"),
            probabilities_dtype=os.getenv(
                "ASTRAMOE_TRACE_PROBABILITIES_DTYPE", "float16"
            ),
            embeddings_dtype=os.getenv("ASTRAMOE_TRACE_EMBEDDINGS_DTYPE", "float16"),
            weights_dtype=os.getenv("ASTRAMOE_TRACE_WEIGHTS_DTYPE", "float32"),
            queue_size=int(os.getenv("ASTRAMOE_TRACE_QUEUE_SIZE", "2")),
            flush_every=int(os.getenv("ASTRAMOE_TRACE_FLUSH_EVERY", "1")),
            compression=os.getenv("ASTRAMOE_TRACE_COMPRESSION", "lzf"),
            trace_level=self.trace_level,
            metadata={
                "model_type": str(config.model_type),
                "model_dtype": str(runner.model_config.dtype),
                "vllm_version": _package_version("vllm"),
                "transformers_version": _package_version("transformers"),
                "tensor_parallel_size": int(runner.parallel_config.tensor_parallel_size),
                "data_parallel_rank": dp_rank,
                "eplb_enabled": bool(
                    getattr(runner.parallel_config, "enable_eplb", False)
                ),
                "router_scoring_func": "softmax",
                "norm_topk_prob": bool(config.norm_topk_prob),
                "trace_level": self.trace_level,
            },
        )
        self._configured = True
        logger.info(
            "Qwen3 raw trace writer initialized at %s (trace_level=%s)",
            path,
            self.trace_level,
        )

    def begin_forward(
        self,
        runner: Any,
        scheduler_output: Any,
        num_scheduled_tokens: np.ndarray,
    ) -> None:
        if not self.enabled or not self._is_writer_rank:
            return
        if self._writer is None:
            raise RuntimeError("raw trace collector was not configured")
        if self._current is not None:
            raise RuntimeError("raw trace forward context already active")

        total = int(scheduler_output.total_num_scheduled_tokens)
        num_reqs = int(runner.input_batch.num_reqs)
        req_indices = np.repeat(np.arange(num_reqs, dtype=np.int32), num_scheduled_tokens)
        positions = runner.positions.np[:total].astype(np.int32, copy=True)
        token_ids = runner.input_ids.cpu[:total].numpy().astype(np.int32, copy=True)
        request_ids_by_position = [str(value) for value in runner.input_batch.req_ids]
        request_ids = [request_ids_by_position[int(idx)] for idx in req_indices]
        prompt_lengths_by_req = runner.input_batch.num_prompt_tokens[:num_reqs].astype(
            np.int32, copy=True
        )
        prompt_lengths = prompt_lengths_by_req[req_indices]
        stage = np.where(positions < prompt_lengths, Stage.PREFILL, Stage.DECODE).astype(
            np.uint8
        )
        iteration = np.where(
            stage == Stage.DECODE, positions - prompt_lengths, -1
        ).astype(np.int32)
        generation_iteration = np.where(
            positions >= prompt_lengths - 1, positions - prompt_lengths + 1, -1
        ).astype(np.int32)
        batch_id = np.full(total, self._forward_counter, dtype=np.uint64)
        forward_id = batch_id.copy()
        index = _ForwardIndex(
            request_ids=request_ids,
            batch_id=batch_id,
            batch_position=req_indices.astype(np.uint32),
            stage=stage,
            iteration_id=iteration,
            generation_iteration_id=generation_iteration,
            token_position=positions,
            token_id=token_ids,
            forward_id=forward_id,
        )
        import torch

        pin = bool(torch.cuda.is_available())
        config = runner.vllm_config.model_config.hf_text_config
        n_layers = len(self._layer_ids)
        weight_dtype = _torch_dtype(
            os.getenv("ASTRAMOE_TRACE_WEIGHTS_DTYPE", "float32")
        )
        current: dict[str, Any] = {
            "index": index,
            "token_count": total,
            "selected_expert_ids": torch.empty(
                (total, n_layers, int(config.num_experts_per_tok)),
                dtype=torch.int32,
                device="cpu",
                pin_memory=pin,
            ),
            "selected_expert_weights": torch.empty(
                (total, n_layers, int(config.num_experts_per_tok)),
                dtype=weight_dtype,
                device="cpu",
                pin_memory=pin,
            ),
            "captured_layers": set(),
            "embedding_captured": self.trace_level
            not in (TRACE_LEVEL_FULL, TRACE_LEVEL_FINE_ONLY),
        }
        if self.trace_level in (TRACE_LEVEL_FULL, TRACE_LEVEL_FINE_ONLY):
            logits_dtype = _torch_dtype(
                os.getenv("ASTRAMOE_TRACE_LOGITS_DTYPE", "float16")
            )
            probs_dtype = _torch_dtype(
                os.getenv("ASTRAMOE_TRACE_PROBABILITIES_DTYPE", "float16")
            )
            emb_dtype = _torch_dtype(
                os.getenv("ASTRAMOE_TRACE_EMBEDDINGS_DTYPE", "float16")
            )
            current["embeddings"] = torch.empty(
                (total, int(config.hidden_size)),
                dtype=emb_dtype,
                device="cpu",
                pin_memory=pin,
            )
            current["router_probabilities"] = torch.empty(
                (total, n_layers, int(config.num_experts)),
                dtype=probs_dtype,
                device="cpu",
                pin_memory=pin,
            )
            if self.trace_level == TRACE_LEVEL_FULL:
                current["router_logits"] = torch.empty(
                    (total, n_layers, int(config.num_experts)),
                    dtype=logits_dtype,
                    device="cpu",
                    pin_memory=pin,
                )
            current["embedding_captured"] = False
        self._current = current
        self._forward_counter += 1

    def capture_embedding(self, embeddings: Any) -> None:
        current = self._current
        if current is None or not self._is_writer_rank:
            return
        if self.trace_level not in (TRACE_LEVEL_FULL, TRACE_LEVEL_FINE_ONLY):
            return
        n = int(current["token_count"])
        if embeddings.shape[0] < n:
            raise RuntimeError(
                f"embedding rows {embeddings.shape[0]} < scheduled tokens {n}"
            )
        current["embeddings"].copy_(embeddings[:n], non_blocking=True)
        current["embedding_captured"] = True

    def capture_router(
        self, router: Any, router_logits: Any, topk_weights: Any, topk_ids: Any
    ) -> None:
        current = self._current
        if current is None or not self._is_writer_rank:
            return
        transformer_layer_id = self._router_layers.get(id(router))
        if transformer_layer_id is None:
            raise RuntimeError("encountered an unregistered MoE router")
        moe_layer = self._layer_ids.index(transformer_layer_id)
        n = int(current["token_count"])
        if router_logits.shape[0] < n:
            raise RuntimeError(
                f"router rows {router_logits.shape[0]} < scheduled tokens {n}"
            )
        import torch

        if self.trace_level in (TRACE_LEVEL_FULL, TRACE_LEVEL_FINE_ONLY):
            full_probabilities = torch.softmax(router_logits[:n].float(), dim=-1)
            current["router_probabilities"][:, moe_layer].copy_(
                full_probabilities, non_blocking=True
            )
            if self.trace_level == TRACE_LEVEL_FULL:
                current["router_logits"][:, moe_layer].copy_(
                    router_logits[:n], non_blocking=True
                )
        current["selected_expert_ids"][:, moe_layer].copy_(
            topk_ids[:n].to(torch.int32), non_blocking=True
        )
        current["selected_expert_weights"][:, moe_layer].copy_(
            topk_weights[:n], non_blocking=True
        )
        current["captured_layers"].add(transformer_layer_id)

    def finish_forward(self) -> None:
        current = self._current
        if current is None or not self._is_writer_rank:
            return
        self._current = None
        missing = set(self._layer_ids) - set(current["captured_layers"])
        if missing:
            raise RuntimeError(f"raw trace missing MoE layers: {sorted(missing)}")
        if (
            self.trace_level in (TRACE_LEVEL_FULL, TRACE_LEVEL_FINE_ONLY)
            and not current["embedding_captured"]
        ):
            raise RuntimeError("raw trace did not capture model embedding output")
        import torch

        event = torch.cuda.Event()
        event.record(torch.cuda.current_stream())
        index: _ForwardIndex = current["index"]
        assert self._writer is not None
        self._writer.submit(
            TraceChunk(
                request_ids=index.request_ids,
                batch_id=index.batch_id,
                batch_position=index.batch_position,
                stage=index.stage,
                iteration_id=index.iteration_id,
                generation_iteration_id=index.generation_iteration_id,
                token_position=index.token_position,
                token_id=index.token_id,
                forward_id=index.forward_id,
                embeddings=current.get("embeddings"),
                router_logits=current.get("router_logits"),
                router_probabilities=current.get("router_probabilities"),
                selected_expert_ids=current["selected_expert_ids"],
                selected_expert_weights=current["selected_expert_weights"],
                ready_event=event,
                trace_level=self.trace_level,
            )
        )

    def close(self) -> None:
        with self._lock:
            if self._writer is not None:
                self._writer.close()
                self._writer = None


_COLLECTOR = _VLLMRawTraceCollector()
atexit.register(_COLLECTOR.close)


def _package_version(name: str) -> str:
    from importlib.metadata import version

    try:
        return version(name)
    except Exception:
        return "unknown"


def _torch_dtype(name: str) -> Any:
    import torch

    mapping = {
        "float16": torch.float16,
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
    }
    try:
        return mapping[name]
    except KeyError as exc:
        raise ValueError(f"unsupported trace dtype: {name}") from exc


def register() -> None:
    """vLLM general-plugin entry point."""

    global _PATCHED
    if _PATCHED:
        return
    _PATCHED = True
    if not _COLLECTOR.enabled:
        return

    from vllm.model_executor.layers.fused_moe.router.base_router import BaseRouter
    from vllm.model_executor.models.qwen3_moe import Qwen3MoeModel
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner
    from vllm.v1.worker.gpu_worker import Worker

    original_bind = GPUModelRunner._bind_routed_experts_capturer
    original_prepare = GPUModelRunner._prepare_inputs
    original_execute = GPUModelRunner.execute_model
    original_shutdown = Worker.shutdown
    original_select = BaseRouter.select_experts
    original_embed = Qwen3MoeModel.embed_input_ids

    @functools.wraps(original_bind)
    def bind(self: Any, capturer: Any) -> None:
        original_bind(self, capturer)
        from vllm.model_executor.layers.fused_moe.layer import FusedMoE
        from vllm.model_executor.layers.fused_moe.router.base_router import BaseRouter

        modules: list[tuple[Any, int]] = []
        for module in self.compilation_config.static_forward_context.values():
            if isinstance(module, FusedMoE) and isinstance(module.router, BaseRouter):
                modules.append((module.router, int(module.layer_id)))
        _COLLECTOR.configure(self, modules)

    @functools.wraps(original_prepare)
    def prepare(self: Any, scheduler_output: Any, num_scheduled_tokens: Any) -> Any:
        result = original_prepare(self, scheduler_output, num_scheduled_tokens)
        _COLLECTOR.begin_forward(self, scheduler_output, num_scheduled_tokens)
        return result

    @functools.wraps(original_execute)
    def execute(self: Any, *args: Any, **kwargs: Any) -> Any:
        try:
            return original_execute(self, *args, **kwargs)
        finally:
            _COLLECTOR.finish_forward()

    @functools.wraps(original_select)
    def select(self: Any, hidden_states: Any, router_logits: Any) -> Any:
        topk_weights, topk_ids = original_select(self, hidden_states, router_logits)
        _COLLECTOR.capture_router(self, router_logits, topk_weights, topk_ids)
        return topk_weights, topk_ids

    @functools.wraps(original_embed)
    def embed(self: Any, input_ids: Any) -> Any:
        embeddings = original_embed(self, input_ids)
        _COLLECTOR.capture_embedding(embeddings)
        return embeddings

    @functools.wraps(original_shutdown)
    def shutdown(self: Any) -> None:
        _COLLECTOR.close()
        original_shutdown(self)

    GPUModelRunner._bind_routed_experts_capturer = bind
    GPUModelRunner._prepare_inputs = prepare
    GPUModelRunner.execute_model = execute
    BaseRouter.select_experts = select
    Qwen3MoeModel.embed_input_ids = embed
    Worker.shutdown = shutdown
    logger.info("Enabled ASTRA-MoE Qwen3 raw trace plugin")
