from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator

try:
    from .config import ModelConfig
except ImportError:  # Allows direct imports when running simulator/main.py.
    from config import ModelConfig


CountsByLayer = dict[int, dict[int, int]]


@dataclass(frozen=True)
class TraceStep:
    batch_index: int
    decode_step: int
    token_count: int
    counts_by_layer: CountsByLayer


@dataclass(frozen=True)
class TraceWorkload:
    steps: list[TraceStep]
    total_generated_tokens: int
    batch_size: int
    source: str

    @property
    def num_steps(self) -> int:
        return len(self.steps)


@dataclass(frozen=True)
class RoutedRow:
    sample_index: int
    decode_step: int
    layer: int
    routed_experts: tuple[int, ...]


def resolve_log_files(path: Path, pattern: str = "*.jsonl") -> list[Path]:
    if path.is_file():
        return [path]
    if not path.exists():
        raise FileNotFoundError(f"routed expert log path not found: {path}")
    files = sorted(path.glob(pattern))
    if files:
        return files
    nested_files = sorted(p for child in path.iterdir() if child.is_dir() for p in child.glob(pattern))
    if nested_files:
        return nested_files
    raise FileNotFoundError(f"No JSONL routed expert logs found under: {path}")


def iter_jsonl_rows(files: Iterable[Path]) -> Iterator[dict[str, Any]]:
    for file_path in files:
        with file_path.open("r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                raw = line.strip()
                if not raw:
                    continue
                try:
                    row = json.loads(raw)
                except json.JSONDecodeError:
                    print(f"[WARN] JSON parse failed: {file_path}:{line_no}")
                    continue
                if isinstance(row, dict):
                    yield row


def _sample_index(row: dict[str, Any]) -> int | None:
    if isinstance(row.get("sample_id"), int):
        return int(row["sample_id"])
    if isinstance(row.get("prompt_index"), int):
        return int(row["prompt_index"]) - 1
    return None


def normalize_routed_row(row: dict[str, Any], model: ModelConfig) -> RoutedRow | None:
    sample_index = _sample_index(row)
    decode_step = row.get("decode_step")
    layer = row.get("layer")
    routed = row.get("routed_experts", row.get("expert_indices"))
    if (
        sample_index is None
        or not isinstance(decode_step, int)
        or not isinstance(layer, int)
        or not isinstance(routed, list)
    ):
        return None

    experts = tuple(
        int(expert_id)
        for expert_id in routed
        if isinstance(expert_id, int) and 0 <= expert_id < model.num_experts
    )
    if not experts:
        return None
    if not 0 <= layer < model.num_layers:
        return None
    return RoutedRow(sample_index, decode_step, layer, experts)


def load_routed_expert_log(
    log_path: str | Path,
    model: ModelConfig,
    batch_size: int,
    max_samples: int | None = None,
    max_decode_steps: int | None = None,
    pattern: str = "*.jsonl",
) -> TraceWorkload:
    files = resolve_log_files(Path(log_path), pattern=pattern)
    counts: dict[tuple[int, int], dict[int, Counter[int]]] = defaultdict(lambda: defaultdict(Counter))
    tokens_by_step: dict[tuple[int, int], set[tuple[int, int]]] = defaultdict(set)

    for raw_row in iter_jsonl_rows(files):
        row = normalize_routed_row(raw_row, model)
        if row is None:
            continue
        if max_samples is not None and row.sample_index >= max_samples:
            continue
        if max_decode_steps is not None and row.decode_step >= max_decode_steps:
            continue

        batch_index = row.sample_index // batch_size
        step_key = (batch_index, row.decode_step)
        counts_by_layer = counts[step_key]
        for expert_id in row.routed_experts:
            counts_by_layer[row.layer][expert_id] += 1
        tokens_by_step[step_key].add((row.sample_index, row.decode_step))

    steps = [
        TraceStep(
            batch_index=batch_index,
            decode_step=decode_step,
            token_count=len(tokens_by_step[(batch_index, decode_step)]),
            counts_by_layer={
                layer: dict(layer_counts)
                for layer, layer_counts in sorted(counts_by_layer.items())
            },
        )
        for (batch_index, decode_step), counts_by_layer in sorted(counts.items())
    ]
    total_generated_tokens = sum(step.token_count for step in steps)
    return TraceWorkload(
        steps=steps,
        total_generated_tokens=total_generated_tokens,
        batch_size=batch_size,
        source="log",
    )


def generate_synthetic_workload(
    model: ModelConfig,
    batch_size: int,
    sequence_length: int,
    total_samples: int,
) -> TraceWorkload:
    steps: list[TraceStep] = []
    hot_experts = max(1, min(8, model.num_experts))

    for batch_start in range(0, total_samples, batch_size):
        batch_index = batch_start // batch_size
        token_count = min(batch_size, total_samples - batch_start)
        for decode_step in range(sequence_length):
            counts_by_layer: CountsByLayer = {}
            for layer in range(model.num_layers):
                layer_counts: Counter[int] = Counter()
                for local_token in range(token_count):
                    sample_index = batch_start + local_token
                    for route_idx in range(model.top_k):
                        if route_idx < max(1, model.top_k - 2):
                            expert_id = (decode_step + layer + sample_index + route_idx) % hot_experts
                        else:
                            expert_id = (
                                decode_step * 17
                                + layer * 31
                                + sample_index * 13
                                + route_idx
                            ) % model.num_experts
                        layer_counts[expert_id] += 1
                counts_by_layer[layer] = dict(layer_counts)
            steps.append(
                TraceStep(
                    batch_index=batch_index,
                    decode_step=decode_step,
                    token_count=token_count,
                    counts_by_layer=counts_by_layer,
                )
            )

    return TraceWorkload(
        steps=steps,
        total_generated_tokens=total_samples * sequence_length,
        batch_size=batch_size,
        source="synthetic",
    )


def load_or_generate_workload(
    log_path: str | Path | None,
    model: ModelConfig,
    batch_size: int,
    synthetic_sequence_length: int,
    synthetic_total_samples: int,
    max_samples: int | None = None,
    max_decode_steps: int | None = None,
    pattern: str = "*.jsonl",
) -> TraceWorkload:
    if log_path is None:
        return generate_synthetic_workload(
            model=model,
            batch_size=batch_size,
            sequence_length=synthetic_sequence_length,
            total_samples=synthetic_total_samples,
        )

    path = Path(log_path)
    if not path.exists():
        print(f"[WARN] Routed expert log path does not exist; using synthetic workload: {path}")
        return generate_synthetic_workload(
            model=model,
            batch_size=batch_size,
            sequence_length=synthetic_sequence_length,
            total_samples=synthetic_total_samples,
        )

    return load_routed_expert_log(
        log_path=path,
        model=model,
        batch_size=batch_size,
        max_samples=max_samples,
        max_decode_steps=max_decode_steps,
        pattern=pattern,
    )
