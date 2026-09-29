from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Iterator, Sequence

from ther.evaluation.trace_source import LayerEvent, SampleActivation
from ther.evaluation.raw_trace.loader import RawTraceLoader

from .event import BatchLayerEvent, PerRequestExperts
from .manifest import LogicalBatchManifest, LogicalBatchSpec


def _logical_request_id(row: dict) -> str:
    return str(row.get("logical_request_id") or row.get("request_id"))


def _trace_request_id(row: dict) -> str:
    """HDF5 / Diff-MoE index key (vLLM runtime id)."""

    return str(row["request_id"])


class LogicalBatchComposer:
    """Compose physical BS=1 request rows into logical BatchLayerEvents.

    Decode alignment: logical step t == each request's generation_iteration_id t.
    """

    def __init__(
        self,
        trace: str | Path,
        *,
        logical_batch_size: int = 16,
        top_k: int | None = None,
    ) -> None:
        self.loader = RawTraceLoader(trace)
        if not self.loader.requests:
            raise ValueError("LogicalBatchComposer requires requests.json")
        self.logical_batch_size = int(logical_batch_size)
        if self.logical_batch_size <= 0:
            raise ValueError("logical_batch_size must be positive")
        self.top_k = int(top_k or self.loader.top_k)
        self._generation = self.loader._handle["index/generation_iteration_id"][:]
        self._selected = self.loader._handle["routing/selected_expert_ids"]
        self._forward = self.loader._handle["index/forward_id"][:]
        self._row_by_key: dict[tuple[str, int], int] = {}
        for row_index, request_id in enumerate(self.loader._request_ids):
            generation = int(self._generation[row_index])
            if generation < 0:
                continue
            key = (str(request_id), generation)
            if key in self._row_by_key:
                raise ValueError(
                    f"duplicate row for request={key[0]!r} iteration={key[1]}"
                )
            self._row_by_key[key] = row_index
        self._rows_by_logical = self._index_requests()

    def close(self) -> None:
        self.loader.close()

    def __enter__(self) -> "LogicalBatchComposer":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def _index_requests(self) -> dict[int, list[dict]]:
        grouped: dict[int, list[dict]] = defaultdict(list)
        for row in self.loader.requests.values():
            if "logical_batch_id" not in row:
                continue
            grouped[int(row["logical_batch_id"])].append(row)
        for batch_id, rows in grouped.items():
            rows.sort(
                key=lambda row: (
                    int(row.get("batch_position", row.get("prompt_index", 0))),
                    _logical_request_id(row),
                )
            )
            if len(rows) != self.logical_batch_size:
                raise ValueError(
                    f"logical_batch_id={batch_id} has {len(rows)} requests; "
                    f"expected {self.logical_batch_size}"
                )
            splits = {str(row.get("split", "")) for row in rows}
            if len(splits) != 1:
                raise ValueError(
                    f"logical_batch_id={batch_id} mixes splits: {sorted(splits)}"
                )
        return dict(grouped)

    def build_manifest(
        self,
        *,
        seed: int = 42,
        physical_batch_size: int = 1,
    ) -> LogicalBatchManifest:
        batches: list[LogicalBatchSpec] = []
        for batch_id in sorted(self._rows_by_logical):
            rows = self._rows_by_logical[batch_id]
            request_ids = tuple(_logical_request_id(row) for row in rows)
            source_by_request = {
                _logical_request_id(row): str(
                    row.get("source_key") or row.get("pile_set_name") or ""
                )
                for row in rows
            }
            batches.append(
                LogicalBatchSpec(
                    logical_batch_id=batch_id,
                    request_ids=request_ids,
                    source_by_request=source_by_request,
                    split=str(rows[0].get("split", "")),
                )
            )
        return LogicalBatchManifest(
            logical_batch_size=self.logical_batch_size,
            physical_batch_size=physical_batch_size,
            seed=seed,
            composition="mixed_sources_fixed",
            batches=batches,
            notes={
                "trace_collection": {"physical_batch_size": physical_batch_size},
                "simulation": {"logical_batch_size": self.logical_batch_size},
            },
        )

    def iter_batch_layer_events(
        self,
        *,
        logical_batch_ids: Sequence[int] | None = None,
    ) -> Iterator[BatchLayerEvent]:
        batch_ids = (
            list(logical_batch_ids)
            if logical_batch_ids is not None
            else sorted(self._rows_by_logical)
        )
        for batch_id in batch_ids:
            rows = self._rows_by_logical[batch_id]
            output_lengths = [int(row["output_length"]) for row in rows]
            if len(set(output_lengths)) != 1:
                raise ValueError(
                    f"logical_batch_id={batch_id} has uneven output lengths: "
                    f"{output_lengths}"
                )
            n_steps = output_lengths[0]
            for iteration in range(n_steps):
                for layer_id in range(len(self.loader.moe_layer_ids)):
                    per_request: list[PerRequestExperts] = []
                    for row in rows:
                        trace_id = _trace_request_id(row)
                        key = (trace_id, iteration)
                        if key not in self._row_by_key:
                            raise ValueError(
                                f"missing decode step: request={trace_id!r} "
                                f"iteration={iteration}"
                            )
                        hdf_row = self._row_by_key[key]
                        expert_ids = tuple(
                            int(value)
                            for value in self._selected[hdf_row, layer_id]
                        )
                        if len(expert_ids) != self.top_k:
                            raise ValueError(
                                f"expected top_k={self.top_k}, got {len(expert_ids)}"
                            )
                        per_request.append(
                            PerRequestExperts(
                                request_id=_logical_request_id(row),
                                selected_expert_ids=expert_ids,
                            )
                        )
                    yield BatchLayerEvent.from_per_request(
                        logical_batch_id=batch_id,
                        decode_iteration=iteration,
                        layer_id=layer_id,
                        per_request=per_request,
                    )

    def iter_layer_events(
        self,
        *,
        logical_batch_ids: Sequence[int] | None = None,
    ) -> Iterator[LayerEvent]:
        """Emit Diff-MoE LayerEvent objects without changing Diff-MoE algorithms."""

        transformer_ids = tuple(int(v) for v in self.loader.transformer_layer_ids)
        batch_ids = (
            list(logical_batch_ids)
            if logical_batch_ids is not None
            else sorted(self._rows_by_logical)
        )
        for batch_id in batch_ids:
            rows = self._rows_by_logical[batch_id]
            n_steps = int(rows[0]["output_length"])
            for iteration in range(n_steps):
                keyed_rows: list[tuple[str, int]] = []
                forward_ids: list[int] = []
                for row in rows:
                    trace_id = _trace_request_id(row)
                    hdf_row = self._row_by_key[(trace_id, iteration)]
                    keyed_rows.append((trace_id, hdf_row))
                    forward_ids.append(int(self._forward[hdf_row]))
                source_forward_ids = tuple(sorted(set(forward_ids)))
                for moe_layer, transformer_layer in enumerate(transformer_ids):
                    samples: list[SampleActivation] = []
                    from collections import Counter

                    routed: Counter[int] = Counter()
                    for request_id, hdf_row in keyed_rows:
                        expert_ids = tuple(
                            int(value)
                            for value in self._selected[hdf_row, moe_layer]
                        )
                        samples.append(
                            SampleActivation(
                                request_id=request_id,
                                generation_iteration_id=iteration,
                                expert_ids=expert_ids,
                                row_index=hdf_row,
                            )
                        )
                        routed.update(expert_ids)
                    yield LayerEvent(
                        replay_batch_id=batch_id,
                        source_forward_ids=source_forward_ids,
                        generation_iteration_id=iteration,
                        moe_layer_index=moe_layer,
                        transformer_layer_id=transformer_layer,
                        samples=tuple(samples),
                        activated_experts=frozenset(routed),
                        routed_token_counts=routed,
                        active_token_count=len(samples),
                    )


def compose_batches(
    request_traces: str | Path,
    *,
    logical_batch_size: int = 16,
    seed: int = 42,
) -> tuple[LogicalBatchManifest, list[BatchLayerEvent]]:
    with LogicalBatchComposer(
        request_traces, logical_batch_size=logical_batch_size
    ) as composer:
        manifest = composer.build_manifest(seed=seed, physical_batch_size=1)
        events = list(composer.iter_batch_layer_events())
    return manifest, events
