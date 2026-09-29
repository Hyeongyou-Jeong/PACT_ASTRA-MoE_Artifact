from __future__ import annotations

from collections import Counter
from dataclasses import dataclass


@dataclass(frozen=True)
class PerRequestExperts:
    request_id: str
    selected_expert_ids: tuple[int, ...]


@dataclass(frozen=True)
class BatchLayerEvent:
    """One logical batch × decode step × MoE layer.

    Diff-MoE consumes activated_experts (binary union).
    THER / ASTRA consume routed_tokens_per_expert (multiplicity).
    """

    logical_batch_id: int
    decode_iteration: int
    layer_id: int
    request_ids: tuple[str, ...]
    per_request_selected_experts: tuple[PerRequestExperts, ...]
    activated_experts: frozenset[int]
    routed_tokens_per_expert: Counter[int]
    total_route_assignments: int

    @property
    def logical_batch_size(self) -> int:
        return len(self.request_ids)

    @classmethod
    def from_per_request(
        cls,
        *,
        logical_batch_id: int,
        decode_iteration: int,
        layer_id: int,
        per_request: list[PerRequestExperts],
    ) -> "BatchLayerEvent":
        routed: Counter[int] = Counter()
        for sample in per_request:
            routed.update(sample.selected_expert_ids)
        return cls(
            logical_batch_id=logical_batch_id,
            decode_iteration=decode_iteration,
            layer_id=layer_id,
            request_ids=tuple(sample.request_id for sample in per_request),
            per_request_selected_experts=tuple(per_request),
            activated_experts=frozenset(routed),
            routed_tokens_per_expert=routed,
            total_route_assignments=sum(routed.values()),
        )
