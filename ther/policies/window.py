"""Per-layer sliding window of routed-token counts used by THER.

Holds the last W per-step count vectors and maintains running totals for
Top-B residency selection.
"""
from __future__ import annotations

from collections import Counter, defaultdict, deque
from typing import Deque

class LayerIntervalRoutingTable:
    """Per-layer sliding-window routed-token table for THER."""

    def __init__(self, window_size: int) -> None:
        self.window_size = max(1, int(window_size))
        self._step_history: dict[int, Deque[Counter[int]]] = defaultdict(deque)
        self._scores: dict[int, Counter[int]] = defaultdict(Counter)
        self._step_count: dict[int, int] = defaultdict(int)

    def reset(self) -> None:
        self._step_history.clear()
        self._scores.clear()
        self._step_count.clear()

    def reset_layer(self, layer_idx: int) -> None:
        self._step_history.pop(layer_idx, None)
        self._scores.pop(layer_idx, None)
        self._step_count[layer_idx] = 0

    def observe_step(self, layer_idx: int, routed_tokens: Counter[int]) -> None:
        step_counter = Counter({eid: int(c) for eid, c in routed_tokens.items()})
        scores = self._scores[layer_idx]
        history = self._step_history[layer_idx]
        if len(history) == self.window_size:
            expired = history.popleft()
            for expert_id, cnt in expired.items():
                scores[expert_id] -= cnt
                if scores[expert_id] <= 0:
                    del scores[expert_id]
        history.append(step_counter)
        for expert_id, cnt in step_counter.items():
            scores[expert_id] += cnt
        self._step_count[layer_idx] += 1

    def current_step_count(self, layer_idx: int) -> int:
        return int(self._step_count.get(layer_idx, 0))

    def current_scores(self, layer_idx: int) -> Counter[int]:
        return Counter(self._scores.get(layer_idx, Counter()))

    def current_ranked_residents(self, layer_idx: int, capacity: int) -> tuple[int, ...]:
        if capacity <= 0:
            return ()
        scores = self._scores.get(layer_idx)
        if not scores:
            return ()
        ranked = sorted(
            ((eid, cnt) for eid, cnt in scores.items() if cnt > 0),
            key=lambda item: (-item[1], item[0]),
        )
        return tuple(eid for eid, _ in ranked[:capacity])




__all__ = ["LayerIntervalRoutingTable"]
