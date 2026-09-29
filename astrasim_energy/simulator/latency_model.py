from __future__ import annotations

from typing import Any


def _transfer_latency(num_bytes: float, bandwidth_Bps: float) -> float:
    if num_bytes <= 0:
        return 0.0
    if bandwidth_Bps <= 0:
        return float("inf")
    return num_bytes / bandwidth_Bps


def pcie_latency(num_bytes: float, pcie_bw_Bps: float) -> float:
    return _transfer_latency(num_bytes, pcie_bw_Bps)


def flash_latency(num_bytes: float, flash_bw_Bps: float) -> float:
    return _transfer_latency(num_bytes, flash_bw_Bps)


def epu_latency(num_flops: float, epu_flops_per_s: float) -> float:
    if num_flops <= 0:
        return 0.0
    if epu_flops_per_s <= 0:
        return float("inf")
    return num_flops / epu_flops_per_s


def gpu_attention_latency(batch_size: int, sequence_length: int, hardware_config: Any) -> float:
    # Lightweight proxy until an attention-kernel model is connected.
    hidden_size = getattr(hardware_config, "attention_hidden_size", 4096)
    flops = batch_size * max(1, sequence_length) * hidden_size * hidden_size
    gpu_flops_per_s = hardware_config.gpu.flops_per_s
    if gpu_flops_per_s <= 0:
        return float("inf")
    return flops / gpu_flops_per_s


def gpu_moe_latency(routed_token_count: int, model_config: Any, hardware_config: Any) -> float:
    if routed_token_count <= 0:
        return 0.0
    flops = routed_token_count * model_config.expert_flops_per_token
    gpu_flops_per_s = hardware_config.gpu.flops_per_s
    if gpu_flops_per_s <= 0:
        return float("inf")
    return flops / gpu_flops_per_s


def ssd_core_latency(t_flash: float, t_epu: float, t_controller: float) -> float:
    return max(t_flash, t_epu) + max(0.0, t_controller)


def total_latency_overlap(t_gpu: float, t_ssd: float, t_pcie: float) -> float:
    return max(t_gpu, t_ssd + t_pcie)


def total_latency(t_gpu: float, t_ssd: float, t_pcie: float) -> float:
    return total_latency_overlap(t_gpu, t_ssd, t_pcie)
