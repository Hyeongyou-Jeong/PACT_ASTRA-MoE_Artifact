# Qwen3 routing independence for offline logical batching
#
# Configuration used by ASTRA-MoE Qwen3 raw-trace collection:
# - Router: FusedTopKRouter, softmax scoring, token-choice Top-K
# - No expert-choice routing
# - No expert capacity limit / token dropping in the capture path
# - No batch-level load-balancing reroute of expert IDs
# - EPLB disabled on tracing runners; if enabled, plugin captures logical IDs
#   before remapping
#
# Therefore, for a fixed token hidden state, selected experts are independent of
# which other requests share the same physical vLLM batch.
#
# Offline composition of physical BS=1 request traces into logical BS=16 events
# is valid for Diff-MoE / THER / ASTRA *routing-policy* simulation under this
# configuration.
#
# Caveats (documented, not paper claims of physical BS=16 measurement):
# - Attention / matmul numerics and decode-token divergence can still make a
#   physical BS=16 run disagree with 16× BS=1 runs on later decode steps.
# - Optional cross-check CLI measures agreement; bitwise identity is not required.
# - Results must be described as: "per-request routing traces composed offline
#   into batched workloads for trace-driven simulation."
