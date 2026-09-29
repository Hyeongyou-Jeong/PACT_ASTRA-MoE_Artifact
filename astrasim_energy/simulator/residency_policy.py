from __future__ import annotations

from collections import Counter, defaultdict, deque
from dataclasses import dataclass, field

try:
    from .config import ModelConfig
except ImportError:  # Allows direct imports when running simulator/main.py.
    from config import ModelConfig


@dataclass(frozen=True)
class ResidencyUpdate:
    decode_step: int
    layer: int
    loaded_experts: frozenset[int]
    evicted_experts: frozenset[int]


@dataclass
class ResidencyState:
    residents_by_layer: dict[int, set[int]]
    updates: list[ResidencyUpdate] = field(default_factory=list)


class ResidencyPolicy:
    def residents_for_layer(self, layer: int) -> set[int]:
        raise NotImplementedError

    def observe_step(self, decode_step: int, counts_by_layer: dict[int, dict[int, int]]) -> None:
        raise NotImplementedError


class StaticResidencyPolicy(ResidencyPolicy):
    def __init__(self, model: ModelConfig, gpu_expert_budget: int) -> None:
        budget = max(0, min(gpu_expert_budget, model.num_experts))
        residents = set(range(budget))
        self._residents_by_layer = {
            layer: set(residents)
            for layer in range(model.num_layers)
        }

    def residents_for_layer(self, layer: int) -> set[int]:
        return self._residents_by_layer.get(layer, set())

    def observe_step(self, decode_step: int, counts_by_layer: dict[int, dict[int, int]]) -> None:
        return None


class THERResidencyPolicy(ResidencyPolicy):
    """Sliding-window Temporal Hot Expert Residency policy."""

    def __init__(self, model: ModelConfig, gpu_expert_budget: int, window_size: int) -> None:
        self.model = model
        self.gpu_expert_budget = max(0, min(gpu_expert_budget, model.num_experts))
        self.window_size = max(1, window_size)
        initial = set(range(self.gpu_expert_budget))
        self.residents_by_layer: dict[int, set[int]] = {
            layer: set(initial)
            for layer in range(model.num_layers)
        }
        self.history_by_layer: dict[int, deque[Counter[int]]] = defaultdict(deque)
        self.scores_by_layer: dict[int, Counter[int]] = defaultdict(Counter)
        self.updates: list[ResidencyUpdate] = []

    def residents_for_layer(self, layer: int) -> set[int]:
        return self.residents_by_layer.get(layer, set())

    def observe_step(self, decode_step: int, counts_by_layer: dict[int, dict[int, int]]) -> None:
        for layer in range(self.model.num_layers):
            counts = Counter(counts_by_layer.get(layer, {}))
            history = self.history_by_layer[layer]
            scores = self.scores_by_layer[layer]

            history.append(counts)
            scores.update(counts)
            while len(history) > self.window_size:
                old_counts = history.popleft()
                for expert_id, count in old_counts.items():
                    scores[expert_id] -= count
                    if scores[expert_id] <= 0:
                        del scores[expert_id]

            if self.gpu_expert_budget == 0:
                new_residents: set[int] = set()
            elif (decode_step + 1) % self.window_size == 0:
                ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
                selected = {expert_id for expert_id, _ in ranked[: self.gpu_expert_budget]}
                if len(selected) < self.gpu_expert_budget:
                    for expert_id in range(self.model.num_experts):
                        selected.add(expert_id)
                        if len(selected) >= self.gpu_expert_budget:
                            break
                new_residents = selected
            else:
                continue

            old_residents = self.residents_by_layer[layer]
            loaded = frozenset(new_residents - old_residents)
            evicted = frozenset(old_residents - new_residents)
            if loaded or evicted:
                self.updates.append(
                    ResidencyUpdate(
                        decode_step=decode_step,
                        layer=layer,
                        loaded_experts=loaded,
                        evicted_experts=evicted,
                    )
                )
            self.residents_by_layer[layer] = new_residents


def build_residency_policy(
    name: str,
    model: ModelConfig,
    gpu_expert_budget: int,
    ther_window_size: int = 16,
) -> ResidencyPolicy:
    normalized = name.lower()
    if normalized in {"static", "all"}:
        budget = model.num_experts if normalized == "all" else gpu_expert_budget
        return StaticResidencyPolicy(model, budget)
    if normalized == "ther":
        return THERResidencyPolicy(model, gpu_expert_budget, ther_window_size)
    raise ValueError(f"Unsupported residency policy: {name}")
