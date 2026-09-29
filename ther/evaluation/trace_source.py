"""Iterate MoE layer events from a routing_only HDF5 trace."""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Sequence

import numpy as np

from ther.evaluation.raw_trace.loader import RawTraceLoader


@dataclass(frozen=True)
class SampleActivation:
    request_id: str
    generation_iteration_id: int
    expert_ids: tuple[int, ...]
    row_index: int


@dataclass(frozen=True)
class LayerEvent:
    replay_batch_id: int
    source_forward_ids: tuple[int, ...]
    generation_iteration_id: int
    moe_layer_index: int
    transformer_layer_id: int
    samples: tuple[SampleActivation, ...]
    activated_experts: frozenset[int]
    routed_token_counts: Counter[int]
    active_token_count: int

    @property
    def total_routes(self) -> int:
        return sum(self.routed_token_counts.values())


@dataclass(frozen=True)
class RequestIterationSequence:
    request_id: str
    generation_iteration_id: int
    activations_by_layer: tuple[tuple[int, ...], ...]


class DiffMoETraceSource:
    """Read-only, causal replay view over a raw trace.

    Policy iteration exposes only the current LayerEvent. `samples` preserves
    request identity and inter-layer sequences, while `activated_experts`
    applies the Qwen3 binary-per-iteration adaptation. Routed counts remain
    separate and are never used by the cache priority update.
    """

    def __init__(self, trace: str | Path) -> None:
        self.loader = RawTraceLoader(trace)
        self._handle = self.loader._handle
        if not self.loader.requests:
            raise ValueError("Diff-MoE replay requires requests.json")
        self.request_rows = sorted(
            self.loader.requests.values(),
            key=lambda row: int(row.get("prompt_index", 0)),
        )
        self.request_split = {
            str(row["request_id"]): str(row.get("split", ""))
            for row in self.request_rows
        }
        self._generation = self._handle["index/generation_iteration_id"][:].astype(
            np.int32
        )
        self._forward_ids = self._handle["index/forward_id"][:].astype(np.uint64)
        self._selected = self._handle["routing/selected_expert_ids"]
        self._request_generation_row: dict[tuple[str, int], int] = {}
        for row_index, request_id in enumerate(self.loader._request_ids):
            generation = int(self._generation[row_index])
            if generation < 0:
                continue
            key = (str(request_id), generation)
            if key in self._request_generation_row:
                raise ValueError(
                    f"duplicate raw row for request={key[0]!r}, iteration={key[1]}"
                )
            self._request_generation_row[key] = row_index

    def close(self) -> None:
        self.loader.close()

    def __enter__(self) -> "DiffMoETraceSource":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    @property
    def num_experts(self) -> int:
        return self.loader.num_experts

    @property
    def num_layers(self) -> int:
        return len(self.loader.moe_layer_ids)

    @property
    def transformer_layer_ids(self) -> tuple[int, ...]:
        return tuple(int(value) for value in self.loader.transformer_layer_ids)

    def request_ids_for_splits(self, splits: Iterable[str]) -> list[str]:
        allowed = set(splits)
        return [
            str(row["request_id"])
            for row in self.request_rows
            if self.request_split[str(row["request_id"])] in allowed
        ]

    def _row_for(self, request_id: str, generation_iteration_id: int) -> int:
        key = (request_id, generation_iteration_id)
        if key not in self._request_generation_row:
            raise ValueError(
                f"missing raw row for request={request_id!r}, "
                f"iteration={generation_iteration_id}"
            )
        return self._request_generation_row[key]

    def iter_request_sequences(
        self, *, splits: Sequence[str]
    ) -> Iterator[RequestIterationSequence]:
        """Request → generation iteration → all MoE layers."""

        allowed = set(splits)
        for request in self.request_rows:
            request_id = str(request["request_id"])
            if self.request_split[request_id] not in allowed:
                continue
            for iteration in range(int(request["output_length"])):
                row = self._row_for(request_id, iteration)
                selected = self._selected[row]
                yield RequestIterationSequence(
                    request_id=request_id,
                    generation_iteration_id=iteration,
                    activations_by_layer=tuple(
                        tuple(int(value) for value in selected[layer])
                        for layer in range(self.num_layers)
                    ),
                )

    def iter_layer_events(
        self,
        *,
        splits: Sequence[str],
        batch_size: int | None = None,
        batch_mode: str = "manifest",
    ) -> Iterator[LayerEvent]:
        if batch_mode == "physical_forward":
            yield from self._iter_physical_events(splits=splits)
            return
        if batch_mode == "logical_manifest":
            yield from self._iter_logical_manifest_events(
                splits=splits, batch_size=batch_size
            )
            return
        if batch_mode != "manifest":
            raise ValueError(
                "batch_mode must be 'manifest', 'logical_manifest', or "
                "'physical_forward'"
            )
        size = int(
            batch_size
            or self.loader.metadata.get("logical_batch_size")
            or self.loader.metadata.get("batch_size")
            or len(self.request_rows)
        )
        if size <= 0:
            raise ValueError("batch_size must be positive")
        allowed = set(splits)
        requests = [
            row
            for row in self.request_rows
            if self.request_split[str(row["request_id"])] in allowed
        ]
        for batch_start in range(0, len(requests), size):
            batch = requests[batch_start : batch_start + size]
            replay_batch_id = batch_start // size
            max_iterations = max(
                (int(row["output_length"]) for row in batch), default=0
            )
            for iteration in range(max_iterations):
                rows: list[tuple[str, int]] = []
                for request in batch:
                    if iteration >= int(request["output_length"]):
                        continue
                    request_id = str(request["request_id"])
                    rows.append((request_id, self._row_for(request_id, iteration)))
                if rows:
                    yield from self._events_from_rows(
                        replay_batch_id=replay_batch_id,
                        generation_iteration_id=iteration,
                        rows=rows,
                    )

    def _iter_logical_manifest_events(
        self,
        *,
        splits: Sequence[str],
        batch_size: int | None = None,
    ) -> Iterator[LayerEvent]:
        """Group by requests.json logical_batch_id (offline composition).

        Physical HDF5 batch_id / forward_id are ignored. Each logical decode
        step t aligns request generation_iteration_id == t.
        """

        allowed = set(splits)
        expected = int(
            batch_size
            or self.loader.metadata.get("logical_batch_size")
            or self.loader.metadata.get("batch_size")
            or 0
        )
        grouped: dict[int, list[dict]] = defaultdict(list)
        for row in self.request_rows:
            request_id = str(row["request_id"])
            if self.request_split[request_id] not in allowed:
                continue
            if "logical_batch_id" not in row:
                raise ValueError(
                    "logical_manifest mode requires logical_batch_id in requests.json"
                )
            grouped[int(row["logical_batch_id"])].append(row)
        for replay_batch_id in sorted(grouped):
            batch = sorted(
                grouped[replay_batch_id],
                key=lambda row: (
                    int(row.get("batch_position", row.get("prompt_index", 0))),
                    str(row["request_id"]),
                ),
            )
            if expected and len(batch) != expected:
                raise ValueError(
                    f"logical_batch_id={replay_batch_id} size {len(batch)} "
                    f"!= expected {expected}"
                )
            splits_in_batch = {str(row.get("split", "")) for row in batch}
            if len(splits_in_batch) != 1:
                raise ValueError(
                    f"logical_batch_id={replay_batch_id} mixes splits: "
                    f"{sorted(splits_in_batch)}"
                )
            lengths = [int(row["output_length"]) for row in batch]
            if len(set(lengths)) != 1:
                raise ValueError(
                    f"logical_batch_id={replay_batch_id} uneven lengths: {lengths}"
                )
            for iteration in range(lengths[0]):
                rows = [
                    (str(row["request_id"]), self._row_for(str(row["request_id"]), iteration))
                    for row in batch
                ]
                yield from self._events_from_rows(
                    replay_batch_id=replay_batch_id,
                    generation_iteration_id=iteration,
                    rows=rows,
                )

    def _iter_physical_events(
        self, *, splits: Sequence[str]
    ) -> Iterator[LayerEvent]:
        allowed_ids = set(self.request_ids_for_splits(splits))
        grouped: dict[tuple[int, int], list[tuple[str, int]]] = defaultdict(list)
        for row_index, request_id in enumerate(self.loader._request_ids):
            iteration = int(self._generation[row_index])
            if iteration < 0 or request_id not in allowed_ids:
                continue
            key = (int(self._forward_ids[row_index]), iteration)
            grouped[key].append((str(request_id), row_index))
        for replay_batch_id, ((forward_id, iteration), rows) in enumerate(
            sorted(grouped.items())
        ):
            yield from self._events_from_rows(
                replay_batch_id=replay_batch_id,
                generation_iteration_id=iteration,
                rows=rows,
                source_forward_ids=(forward_id,),
            )

    def _events_from_rows(
        self,
        *,
        replay_batch_id: int,
        generation_iteration_id: int,
        rows: list[tuple[str, int]],
        source_forward_ids: tuple[int, ...] | None = None,
    ) -> Iterator[LayerEvent]:
        if source_forward_ids is None:
            source_forward_ids = tuple(
                sorted({int(self._forward_ids[row]) for _, row in rows})
            )
        for moe_layer, transformer_layer in enumerate(self.transformer_layer_ids):
            samples: list[SampleActivation] = []
            routed_counts: Counter[int] = Counter()
            for request_id, row in rows:
                expert_ids = tuple(
                    int(value) for value in self._selected[row, moe_layer]
                )
                samples.append(
                    SampleActivation(
                        request_id=request_id,
                        generation_iteration_id=generation_iteration_id,
                        expert_ids=expert_ids,
                        row_index=row,
                    )
                )
                routed_counts.update(expert_ids)
            yield LayerEvent(
                replay_batch_id=replay_batch_id,
                source_forward_ids=source_forward_ids,
                generation_iteration_id=generation_iteration_id,
                moe_layer_index=moe_layer,
                transformer_layer_id=transformer_layer,
                samples=tuple(samples),
                activated_experts=frozenset(routed_counts),
                routed_token_counts=routed_counts,
                active_token_count=len(samples),
            )
