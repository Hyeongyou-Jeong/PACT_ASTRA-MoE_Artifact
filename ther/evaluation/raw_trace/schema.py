from __future__ import annotations

from enum import IntEnum

FORMAT_VERSION = "1.0.0"

# full: embeddings + full router logits/probabilities + selected.
# fine_only: FineMoE minimum — embeddings + full probabilities + selected
#   (no router logits).
# routing_only: Diff-MoE/THER identity + selected experts/weights only.
TRACE_LEVEL_FULL = "full"
TRACE_LEVEL_FINE_ONLY = "fine_only"
TRACE_LEVEL_ROUTING_ONLY = "routing_only"
TRACE_LEVELS = (TRACE_LEVEL_FULL, TRACE_LEVEL_FINE_ONLY, TRACE_LEVEL_ROUTING_ONLY)

FINE_ONLY_DATASETS = (
    "router_probabilities",
    "selected_expert_ids",
    "selected_expert_weights",
)


class Stage(IntEnum):
    """Physical model-forward stage."""

    PREFILL = 0
    DECODE = 1


TOKEN_DATASETS = {
    "request_id": "utf8",
    "batch_id": "uint64",
    "batch_position": "uint32",
    "stage": "uint8",
    "iteration_id": "int32",
    "generation_iteration_id": "int32",
    "token_position": "int32",
    "token_id": "int32",
    "forward_id": "uint64",
}

ROUTING_DATASETS = (
    "router_logits",
    "router_probabilities",
    "selected_expert_ids",
    "selected_expert_weights",
)

ROUTING_ONLY_DATASETS = (
    "selected_expert_ids",
    "selected_expert_weights",
)


def expert_id_dtype(num_experts: int) -> str:
    if num_experts <= 0:
        raise ValueError("num_experts must be positive")
    if num_experts <= 256:
        return "uint8"
    if num_experts <= 65_536:
        return "uint16"
    return "uint32"
