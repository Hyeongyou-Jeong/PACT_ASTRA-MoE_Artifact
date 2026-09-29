from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from ther.evaluation.raw_trace.run_metadata import write_json_atomic


@dataclass(frozen=True)
class LogicalBatchSpec:
    logical_batch_id: int
    request_ids: tuple[str, ...]
    source_by_request: dict[str, str]
    split: str

    def to_json(self) -> dict[str, Any]:
        return {
            "logical_batch_id": self.logical_batch_id,
            "request_ids": list(self.request_ids),
            "source_by_request": dict(self.source_by_request),
            "split": self.split,
            "logical_batch_size": len(self.request_ids),
        }


@dataclass
class LogicalBatchManifest:
    logical_batch_size: int
    physical_batch_size: int = 1
    seed: int = 42
    composition: str = "mixed_sources_fixed"
    batches: list[LogicalBatchSpec] = field(default_factory=list)
    notes: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        return {
            "logical_batch_size": self.logical_batch_size,
            "physical_batch_size": self.physical_batch_size,
            "seed": self.seed,
            "composition": self.composition,
            "notes": {
                **self.notes,
                "semantics": (
                    "Logical batches are offline compositions of independently "
                    "collected physical BS=1 Qwen3 routing traces. They are not "
                    "physical multi-request vLLM batches."
                ),
            },
            "batches": [batch.to_json() for batch in self.batches],
        }

    def save(self, path: str | Path) -> None:
        write_json_atomic(path, self.to_json())

    @classmethod
    def load(cls, path: str | Path) -> "LogicalBatchManifest":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        batches = [
            LogicalBatchSpec(
                logical_batch_id=int(row["logical_batch_id"]),
                request_ids=tuple(str(value) for value in row["request_ids"]),
                source_by_request={
                    str(key): str(value)
                    for key, value in row.get("source_by_request", {}).items()
                },
                split=str(row.get("split", "")),
            )
            for row in payload["batches"]
        ]
        return cls(
            logical_batch_size=int(payload["logical_batch_size"]),
            physical_batch_size=int(payload.get("physical_batch_size", 1)),
            seed=int(payload.get("seed", 42)),
            composition=str(payload.get("composition", "mixed_sources_fixed")),
            batches=batches,
            notes=dict(payload.get("notes", {})),
        )
