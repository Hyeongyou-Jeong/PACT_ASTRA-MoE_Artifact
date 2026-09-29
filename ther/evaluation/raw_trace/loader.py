from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterator

import numpy as np

from .schema import (
    FORMAT_VERSION,
    TRACE_LEVEL_FINE_ONLY,
    TRACE_LEVEL_FULL,
    TRACE_LEVEL_ROUTING_ONLY,
    Stage,
)


class AmbiguousTraceQuery(ValueError):
    pass


class RawTraceLoader:
    def __init__(self, trace: str | Path) -> None:
        import h5py

        path = Path(trace)
        self.run_dir = path if path.is_dir() else path.parent
        self.path = path / "raw_trace.h5" if path.is_dir() else path
        self._handle = h5py.File(self.path, "r")
        version = str(self._handle.attrs.get("trace_format_version", ""))
        if version != FORMAT_VERSION:
            raise ValueError(
                f"unsupported trace version {version!r}; expected {FORMAT_VERSION!r}"
            )
        metadata_path = self.run_dir / "metadata.json"
        self.metadata: dict[str, Any] = (
            json.loads(metadata_path.read_text(encoding="utf-8"))
            if metadata_path.exists()
            else {}
        )
        request_path = self.run_dir / "requests.json"
        request_rows = (
            json.loads(request_path.read_text(encoding="utf-8"))
            if request_path.exists()
            else []
        )
        self.requests = {str(row["request_id"]): row for row in request_rows}
        self.transformer_layer_ids = self._handle[
            "model/transformer_layer_ids"
        ][:].astype(int)
        self.moe_layer_ids = self._handle["model/moe_layer_ids"][:].astype(int)
        self._transformer_to_moe = {
            int(layer): i for i, layer in enumerate(self.transformer_layer_ids)
        }
        raw_ids = self._handle["index/request_id"][:]
        self._request_ids = np.asarray(
            [value.decode() if isinstance(value, bytes) else str(value) for value in raw_ids]
        )
        self.trace_level = str(
            self._handle.attrs.get(
                "trace_level",
                self.metadata.get("trace_level", TRACE_LEVEL_FULL),
            )
        )

    def close(self) -> None:
        self._handle.close()

    def __enter__(self) -> "RawTraceLoader":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()

    @property
    def num_tokens(self) -> int:
        return int(self._request_ids.shape[0])

    @property
    def num_experts(self) -> int:
        return int(self._handle.attrs["num_experts"])

    @property
    def top_k(self) -> int:
        return int(self._handle.attrs["top_k"])

    @property
    def stores_full_router(self) -> bool:
        return self.trace_level == TRACE_LEVEL_FULL

    @property
    def stores_finemoe_tensors(self) -> bool:
        return self.trace_level in (TRACE_LEVEL_FULL, TRACE_LEVEL_FINE_ONLY)

    def _require_full(self, feature: str) -> None:
        if not self.stores_full_router:
            raise ValueError(
                f"{feature} requires trace_level='full'; this trace is "
                f"{self.trace_level!r}"
            )

    def _require_finemoe(self, feature: str) -> None:
        if not self.stores_finemoe_tensors:
            raise ValueError(
                f"{feature} requires trace_level in {{'full','fine_only'}}; "
                f"this trace is {self.trace_level!r}"
            )

    def token_indices(
        self,
        request_id: str,
        *,
        stage: str | Stage | None = None,
        iteration_id: int | None = None,
        generation_iteration_id: int | None = None,
        token_position: int | None = None,
    ) -> np.ndarray:
        mask = self._request_ids == str(request_id)
        if stage is not None:
            stage_value = (
                Stage[stage.upper()].value if isinstance(stage, str) else int(stage)
            )
            mask &= self._handle["index/stage"][:] == stage_value
        if iteration_id is not None:
            mask &= self._handle["index/iteration_id"][:] == int(iteration_id)
        if generation_iteration_id is not None:
            mask &= (
                self._handle["index/generation_iteration_id"][:]
                == int(generation_iteration_id)
            )
        if token_position is not None:
            mask &= self._handle["index/token_position"][:] == int(token_position)
        return np.flatnonzero(mask)

    def _moe_layer_index(self, layer_id: int, *, original_layer: bool) -> int:
        if original_layer:
            try:
                return self._transformer_to_moe[int(layer_id)]
            except KeyError as exc:
                raise KeyError(f"transformer layer {layer_id} is not an MoE layer") from exc
        idx = int(layer_id)
        if not 0 <= idx < len(self.moe_layer_ids):
            raise IndexError(f"MoE layer index out of range: {layer_id}")
        return idx

    def get_router_distribution(
        self,
        request_id: str,
        iteration_id: int,
        layer_id: int,
        *,
        stage: str = "decode",
        token_position: int | None = None,
        original_layer: bool = True,
    ) -> np.ndarray:
        """Return one full-expert distribution.

        Decode queries use generation iteration semantics. Generation iteration
        zero is the final prompt token evaluated by the prefill forward. A
        prefill query with multiple matching tokens is deliberately rejected;
        callers must select a token or explicitly aggregate token-level values.
        """

        self._require_full("get_router_distribution")
        moe_layer = self._moe_layer_index(layer_id, original_layer=original_layer)
        if stage.lower() == "decode":
            indices = self.token_indices(
                request_id,
                generation_iteration_id=iteration_id,
                token_position=token_position,
            )
        elif stage.lower() == "prefill":
            indices = self.token_indices(
                request_id,
                stage=Stage.PREFILL,
                iteration_id=iteration_id,
                token_position=token_position,
            )
        else:
            raise ValueError("stage must be 'prefill' or 'decode'")
        if len(indices) == 0:
            raise KeyError(
                f"no router distribution for request={request_id!r}, "
                f"iteration={iteration_id}, layer={layer_id}"
            )
        if len(indices) != 1:
            raise AmbiguousTraceQuery(
                f"query matched {len(indices)} tokens; specify token_position or "
                "use get_prefill_router_distributions() and aggregate explicitly"
            )
        return self._handle["routing/router_probabilities"][
            int(indices[0]), moe_layer
        ].astype(np.float32)

    def get_prefill_router_distributions(
        self, request_id: str, layer_id: int, *, original_layer: bool = True
    ) -> tuple[np.ndarray, np.ndarray]:
        self._require_full("get_prefill_router_distributions")
        moe_layer = self._moe_layer_index(layer_id, original_layer=original_layer)
        indices = self.token_indices(request_id, stage=Stage.PREFILL)
        positions = self._handle["index/token_position"][indices].astype(np.int32)
        order = np.argsort(positions, kind="stable")
        return (
            positions[order],
            self._handle["routing/router_probabilities"][
                indices[order], moe_layer
            ].astype(np.float32),
        )

    def get_embeddings(
        self, request_id: str, *, stage: str | Stage | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        self._require_full("get_embeddings")
        indices = self.token_indices(request_id, stage=stage)
        positions = self._handle["index/token_position"][indices].astype(np.int32)
        order = np.argsort(positions, kind="stable")
        return positions[order], self._handle["embedding"][indices[order]].astype(
            np.float32
        )

    def selected_experts(
        self,
        request_id: str,
        generation_iteration_id: int,
        layer_id: int,
        *,
        original_layer: bool = True,
    ) -> tuple[np.ndarray, np.ndarray]:
        moe_layer = self._moe_layer_index(layer_id, original_layer=original_layer)
        indices = self.token_indices(
            request_id, generation_iteration_id=generation_iteration_id
        )
        if len(indices) != 1:
            raise AmbiguousTraceQuery(f"selected-expert query matched {len(indices)} rows")
        row = int(indices[0])
        return (
            self._handle["routing/selected_expert_ids"][row, moe_layer].astype(
                np.int64
            ),
            self._handle["routing/selected_expert_weights"][row, moe_layer].astype(
                np.float32
            ),
        )

    def routed_token_counts(
        self,
        *,
        layer_id: int,
        generation_iteration_id: int,
        request_ids: list[str] | None = None,
        original_layer: bool = True,
    ) -> np.ndarray:
        """Return exact token assignments per expert for a replay step."""

        moe_layer = self._moe_layer_index(layer_id, original_layer=original_layer)
        mask = (
            self._handle["index/generation_iteration_id"][:]
            == int(generation_iteration_id)
        )
        if request_ids is not None:
            mask &= np.isin(self._request_ids, np.asarray(request_ids, dtype=str))
        ids = self._handle["routing/selected_expert_ids"][
            np.flatnonzero(mask), moe_layer
        ]
        return np.bincount(ids.reshape(-1).astype(np.int64), minlength=self.num_experts)

    def activation_event_counts(
        self,
        *,
        layer_id: int,
        generation_iteration_id: int,
        request_ids: list[str] | None = None,
        original_layer: bool = True,
    ) -> np.ndarray:
        """Count expert activation once per request/iteration, independent of load."""

        moe_layer = self._moe_layer_index(layer_id, original_layer=original_layer)
        mask = (
            self._handle["index/generation_iteration_id"][:]
            == int(generation_iteration_id)
        )
        if request_ids is not None:
            mask &= np.isin(self._request_ids, np.asarray(request_ids, dtype=str))
        counts: Counter[int] = Counter()
        indices = np.flatnonzero(mask)
        for request_id in np.unique(self._request_ids[indices]):
            request_indices = indices[self._request_ids[indices] == request_id]
            ids = self._handle["routing/selected_expert_ids"][
                request_indices, moe_layer
            ]
            counts.update(set(int(value) for value in ids.reshape(-1)))
        result = np.zeros(self.num_experts, dtype=np.uint64)
        for expert_id, count in counts.items():
            result[expert_id] = count
        return result

    def iter_request_token_rows(self, request_id: str) -> Iterator[dict[str, Any]]:
        indices = self.token_indices(request_id)
        positions = self._handle["index/token_position"][indices]
        for idx in indices[np.argsort(positions, kind="stable")]:
            yield {
                "row_index": int(idx),
                "request_id": str(request_id),
                "batch_id": int(self._handle["index/batch_id"][idx]),
                "batch_position": int(self._handle["index/batch_position"][idx]),
                "stage": Stage(int(self._handle["index/stage"][idx])).name.lower(),
                "iteration_id": int(self._handle["index/iteration_id"][idx]),
                "generation_iteration_id": int(
                    self._handle["index/generation_iteration_id"][idx]
                ),
                "token_position": int(self._handle["index/token_position"][idx]),
                "token_id": int(self._handle["index/token_id"][idx]),
            }
