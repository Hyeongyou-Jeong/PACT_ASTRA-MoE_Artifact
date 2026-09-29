from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .schema import (
    FORMAT_VERSION,
    TRACE_LEVEL_FINE_ONLY,
    TRACE_LEVEL_FULL,
    TRACE_LEVEL_ROUTING_ONLY,
    TRACE_LEVELS,
    expert_id_dtype,
)


def _numpy(value: Any) -> np.ndarray:
    if isinstance(value, np.ndarray):
        return value
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu") and getattr(value, "device", None) is not None:
        value = value.cpu()
    if hasattr(value, "numpy"):
        return value.numpy()
    return np.asarray(value)


@dataclass
class TraceChunk:
    request_ids: list[str]
    batch_id: np.ndarray
    batch_position: np.ndarray
    stage: np.ndarray
    iteration_id: np.ndarray
    generation_iteration_id: np.ndarray
    token_position: np.ndarray
    token_id: np.ndarray
    forward_id: np.ndarray
    embeddings: Any | None
    router_logits: Any | None
    router_probabilities: Any | None
    selected_expert_ids: Any
    selected_expert_weights: Any
    ready_event: Any | None = None
    trace_level: str = TRACE_LEVEL_FULL

    @property
    def token_count(self) -> int:
        return len(self.request_ids)

    def wait_and_numpy(self) -> dict[str, np.ndarray]:
        if self.ready_event is not None:
            self.ready_event.synchronize()
        arrays = {
            "batch_id": _numpy(self.batch_id),
            "batch_position": _numpy(self.batch_position),
            "stage": _numpy(self.stage),
            "iteration_id": _numpy(self.iteration_id),
            "generation_iteration_id": _numpy(self.generation_iteration_id),
            "token_position": _numpy(self.token_position),
            "token_id": _numpy(self.token_id),
            "forward_id": _numpy(self.forward_id),
            "selected_expert_ids": _numpy(self.selected_expert_ids),
            "selected_expert_weights": _numpy(self.selected_expert_weights),
        }
        if self.trace_level == TRACE_LEVEL_FULL:
            arrays["embeddings"] = _numpy(self.embeddings)
            arrays["router_logits"] = _numpy(self.router_logits)
            arrays["router_probabilities"] = _numpy(self.router_probabilities)
        elif self.trace_level == TRACE_LEVEL_FINE_ONLY:
            arrays["embeddings"] = _numpy(self.embeddings)
            arrays["router_probabilities"] = _numpy(self.router_probabilities)
        self._validate(arrays)
        return arrays

    def _validate(self, arrays: Mapping[str, np.ndarray]) -> None:
        n = self.token_count
        if n <= 0:
            raise ValueError("trace chunk must contain at least one token")
        if self.trace_level not in TRACE_LEVELS:
            raise ValueError(f"unsupported trace_level={self.trace_level!r}")
        for name, array in arrays.items():
            if array.shape[0] != n:
                raise ValueError(f"{name} first dimension {array.shape[0]} != {n}")
        ids = arrays["selected_expert_ids"]
        weights = arrays["selected_expert_weights"]
        if ids.ndim != 3 or weights.shape != ids.shape:
            raise ValueError(
                f"selected arrays must be [tokens,layers,top_k], got "
                f"{ids.shape} and {weights.shape}"
            )
        if self.trace_level == TRACE_LEVEL_ROUTING_ONLY:
            return
        probs = arrays["router_probabilities"]
        if probs.ndim != 3:
            raise ValueError(
                f"router_probabilities must be [tokens,layers,experts], got "
                f"{probs.shape}"
            )
        if ids.shape[:2] != probs.shape[:2]:
            raise ValueError("routing layer/token dimensions do not match")
        if arrays["embeddings"].ndim != 2:
            raise ValueError("embeddings must be [tokens,hidden_size]")
        if self.trace_level == TRACE_LEVEL_FULL:
            logits = arrays["router_logits"]
            if logits.shape != probs.shape:
                raise ValueError(
                    f"router logits/probabilities shape mismatch: "
                    f"{logits.shape} vs {probs.shape}"
                )


class AsyncHDF5TraceWriter:
    """Single-writer, bounded-queue HDF5 appender.

    GPU callbacks enqueue pinned CPU tensors plus a CUDA event. Only this
    thread touches h5py; waiting for the copy event therefore does not block
    the model thread unless the bounded queue fills.
    """

    def __init__(
        self,
        path: str | Path,
        *,
        num_moe_layers: int,
        num_experts: int,
        top_k: int,
        hidden_size: int,
        transformer_layer_ids: list[int],
        logits_dtype: str = "float16",
        probabilities_dtype: str = "float16",
        embeddings_dtype: str = "float16",
        weights_dtype: str = "float32",
        queue_size: int = 2,
        flush_every: int = 1,
        compression: str | None = "lzf",
        metadata: Mapping[str, Any] | None = None,
        trace_level: str = TRACE_LEVEL_FULL,
    ) -> None:
        if len(transformer_layer_ids) != num_moe_layers:
            raise ValueError("transformer_layer_ids length must equal num_moe_layers")
        if trace_level not in TRACE_LEVELS:
            raise ValueError(
                f"trace_level must be one of {TRACE_LEVELS}, got {trace_level!r}"
            )
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.trace_level = str(trace_level)
        self.num_moe_layers = int(num_moe_layers)
        self.num_experts = int(num_experts)
        self.top_k = int(top_k)
        self.hidden_size = int(hidden_size)
        self.transformer_layer_ids = np.asarray(transformer_layer_ids, dtype=np.int32)
        self.logits_dtype = np.dtype(logits_dtype)
        self.probabilities_dtype = np.dtype(probabilities_dtype)
        self.embeddings_dtype = np.dtype(embeddings_dtype)
        self.weights_dtype = np.dtype(weights_dtype)
        self.compression = None if compression in (None, "none") else compression
        self.flush_every = max(1, int(flush_every))
        self.metadata = dict(metadata or {})
        self._queue: queue.Queue[TraceChunk | None] = queue.Queue(
            maxsize=max(1, int(queue_size))
        )
        self._error: BaseException | None = None
        self._closed = False
        self._stats_lock = threading.Lock()
        self._tokens_written = 0
        self._chunks_written = 0
        self._write_seconds = 0.0
        self._thread = threading.Thread(
            target=self._run, name="raw-trace-hdf5-writer", daemon=True
        )
        self._thread.start()

    def submit(self, chunk: TraceChunk) -> None:
        if self._closed:
            raise RuntimeError("trace writer is closed")
        if chunk.trace_level != self.trace_level:
            raise ValueError(
                f"chunk trace_level={chunk.trace_level!r} != writer "
                f"{self.trace_level!r}"
            )
        self._raise_if_failed()
        self._queue.put(chunk)
        self._raise_if_failed()

    def close(self) -> None:
        if self._closed:
            self._raise_if_failed()
            return
        self._closed = True
        self._queue.put(None)
        self._thread.join()
        self._raise_if_failed()

    @property
    def stats(self) -> dict[str, float | int | str]:
        with self._stats_lock:
            size = self.path.stat().st_size if self.path.exists() else 0
            return {
                "tokens_written": self._tokens_written,
                "chunks_written": self._chunks_written,
                "write_seconds": self._write_seconds,
                "trace_bytes": size,
                "write_bandwidth_bytes_per_s": (
                    size / self._write_seconds if self._write_seconds > 0 else 0.0
                ),
                "trace_level": self.trace_level,
            }

    def _raise_if_failed(self) -> None:
        if self._error is not None:
            raise RuntimeError("raw trace writer failed") from self._error

    def _run(self) -> None:
        try:
            import h5py

            with h5py.File(self.path, "w", libver="latest") as handle:
                self._initialize_file(handle, h5py)
                while True:
                    chunk = self._queue.get()
                    if chunk is None:
                        break
                    started = time.perf_counter()
                    arrays = chunk.wait_and_numpy()
                    self._append(handle, chunk.request_ids, arrays)
                    elapsed = time.perf_counter() - started
                    with self._stats_lock:
                        self._tokens_written += chunk.token_count
                        self._chunks_written += 1
                        self._write_seconds += elapsed
                        chunks = self._chunks_written
                    if chunks % self.flush_every == 0:
                        stats = self.stats
                        for key, value in stats.items():
                            handle.attrs[key] = value
                        handle.flush()
                stats = self.stats
                for key, value in stats.items():
                    handle.attrs[key] = value
                handle.flush()
        except BaseException as exc:
            self._error = exc

    def _initialize_file(self, handle: Any, h5py: Any) -> None:
        handle.attrs["trace_format"] = "astramoe-qwen3-raw"
        handle.attrs["trace_format_version"] = FORMAT_VERSION
        handle.attrs["trace_level"] = self.trace_level
        handle.attrs["num_moe_layers"] = self.num_moe_layers
        handle.attrs["num_experts"] = self.num_experts
        handle.attrs["top_k"] = self.top_k
        handle.attrs["hidden_size"] = self.hidden_size
        handle.attrs["router_logits_dtype"] = self.logits_dtype.name
        handle.attrs["router_probabilities_dtype"] = self.probabilities_dtype.name
        handle.attrs["selected_expert_weights_dtype"] = self.weights_dtype.name
        handle.attrs["embedding_dtype"] = self.embeddings_dtype.name
        handle.attrs["router_score_semantics"] = (
            "full float32 softmax(raw_router_logits) before top-k selection"
            if self.trace_level in (TRACE_LEVEL_FULL, TRACE_LEVEL_FINE_ONLY)
            else "not stored in routing_only mode"
        )
        handle.attrs["selected_weight_semantics"] = (
            "actual vLLM top-k weights used by expert output aggregation"
        )
        if self.trace_level == TRACE_LEVEL_FINE_ONLY:
            handle.attrs["fine_only_note"] = (
                "stores embeddings + full router_probabilities + selected; "
                "omits router_logits"
            )
        for key, value in self.metadata.items():
            if value is None or isinstance(value, (dict, list, tuple)):
                continue
            handle.attrs[str(key)] = value

        index = handle.create_group("index")
        routing = handle.create_group("routing")
        model = handle.create_group("model")
        str_dtype = h5py.string_dtype(encoding="utf-8")
        index.create_dataset("request_id", shape=(0,), maxshape=(None,), dtype=str_dtype)
        for name, dtype in (
            ("batch_id", "uint64"),
            ("batch_position", "uint32"),
            ("stage", "uint8"),
            ("iteration_id", "int32"),
            ("generation_iteration_id", "int32"),
            ("token_position", "int32"),
            ("token_id", "int32"),
            ("forward_id", "uint64"),
        ):
            index.create_dataset(name, shape=(0,), maxshape=(None,), dtype=dtype)
        model.create_dataset(
            "transformer_layer_ids", data=self.transformer_layer_ids, dtype="int32"
        )
        model.create_dataset(
            "moe_layer_ids",
            data=np.arange(self.num_moe_layers, dtype=np.int32),
            dtype="int32",
        )
        if self.trace_level in (TRACE_LEVEL_FULL, TRACE_LEVEL_FINE_ONLY):
            self._create_tensor_dataset(
                handle,
                "embedding",
                (self.hidden_size,),
                self.embeddings_dtype,
            )
            self._create_tensor_dataset(
                routing,
                "router_probabilities",
                (self.num_moe_layers, self.num_experts),
                self.probabilities_dtype,
            )
        if self.trace_level == TRACE_LEVEL_FULL:
            self._create_tensor_dataset(
                routing,
                "router_logits",
                (self.num_moe_layers, self.num_experts),
                self.logits_dtype,
            )
        self._create_tensor_dataset(
            routing,
            "selected_expert_ids",
            (self.num_moe_layers, self.top_k),
            expert_id_dtype(self.num_experts),
        )
        self._create_tensor_dataset(
            routing,
            "selected_expert_weights",
            (self.num_moe_layers, self.top_k),
            self.weights_dtype,
        )

    def _create_tensor_dataset(
        self, parent: Any, name: str, tail_shape: tuple[int, ...], dtype: Any
    ) -> None:
        chunk_tokens = max(1, min(256, 1_048_576 // max(1, int(np.prod(tail_shape)))))
        parent.create_dataset(
            name,
            shape=(0, *tail_shape),
            maxshape=(None, *tail_shape),
            chunks=(chunk_tokens, *tail_shape),
            dtype=dtype,
            compression=self.compression,
        )

    def _append(
        self, handle: Any, request_ids: list[str], arrays: Mapping[str, np.ndarray]
    ) -> None:
        start = int(handle["index/request_id"].shape[0])
        end = start + len(request_ids)
        index = handle["index"]
        for name in (
            "request_id",
            "batch_id",
            "batch_position",
            "stage",
            "iteration_id",
            "generation_iteration_id",
            "token_position",
            "token_id",
            "forward_id",
        ):
            dataset = index[name]
            dataset.resize((end,))
            dataset[start:end] = request_ids if name == "request_id" else arrays[name]
        paths: list[tuple[str, str, Any]] = [
            (
                "routing/selected_expert_ids",
                "selected_expert_ids",
                np.dtype(expert_id_dtype(self.num_experts)),
            ),
            (
                "routing/selected_expert_weights",
                "selected_expert_weights",
                self.weights_dtype,
            ),
        ]
        if self.trace_level == TRACE_LEVEL_FULL:
            paths = [
                ("embedding", "embeddings", self.embeddings_dtype),
                ("routing/router_logits", "router_logits", self.logits_dtype),
                (
                    "routing/router_probabilities",
                    "router_probabilities",
                    self.probabilities_dtype,
                ),
                *paths,
            ]
        elif self.trace_level == TRACE_LEVEL_FINE_ONLY:
            paths = [
                ("embedding", "embeddings", self.embeddings_dtype),
                (
                    "routing/router_probabilities",
                    "router_probabilities",
                    self.probabilities_dtype,
                ),
                *paths,
            ]
        for path, key, dtype in paths:
            dataset = handle[path]
            dataset.resize((end, *dataset.shape[1:]))
            dataset[start:end] = arrays[key].astype(dtype, copy=False)

    def __enter__(self) -> "AsyncHDF5TraceWriter":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()
