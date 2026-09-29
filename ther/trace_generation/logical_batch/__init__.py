"""Offline logical-batch composition over physical BS=1 request traces.

Raw traces remain one real Qwen request (physical_batch_size=1).
LogicalBatchComposer builds BS=N BatchLayerEvent sequences for Diff-MoE / THER /
ASTRA without claiming physical multi-request inference.
"""

from .composer import LogicalBatchComposer, compose_batches
from .event import BatchLayerEvent, PerRequestExperts
from .manifest import LogicalBatchManifest, LogicalBatchSpec
from .validation import (
    diagnose_logical_batch,
    synthetic_four_request_expectation,
    validate_synthetic_composition,
)

__all__ = [
    "BatchLayerEvent",
    "LogicalBatchComposer",
    "LogicalBatchManifest",
    "LogicalBatchSpec",
    "PerRequestExperts",
    "compose_batches",
    "diagnose_logical_batch",
    "synthetic_four_request_expectation",
    "validate_synthetic_composition",
]
