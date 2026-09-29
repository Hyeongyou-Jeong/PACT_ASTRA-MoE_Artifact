from __future__ import annotations

try:
    from .config import HardwareConfig, ModelConfig, SystemConfig
except ImportError:  # Allows direct imports when running simulator/main.py.
    from config import HardwareConfig, ModelConfig, SystemConfig


def kv_cache_bytes_per_token(model: ModelConfig) -> float:
    kv_hidden_size = model.kv_cache_hidden_size or model.hidden_size
    kv_layers = model.kv_cache_num_layers or model.num_layers
    return kv_layers * kv_hidden_size * model.bytes_per_param * 2


def kv_cache_bytes_per_sequence(model: ModelConfig, sequence_length: int) -> float:
    return max(1, sequence_length) * kv_cache_bytes_per_token(model)


def resident_expert_count_per_layer(system: SystemConfig, model: ModelConfig) -> int:
    name = system.name.lower()
    if name == "vllm":
        return model.num_experts
    if name == "deepspeed_nvme":
        return 0
    if name == "astra_moe":
        return max(0, min(system.gpu_expert_budget, model.num_experts))
    return max(0, min(system.gpu_expert_budget, model.num_experts))


def resident_expert_bytes(system: SystemConfig, model: ModelConfig) -> float:
    resident_layers = model.resident_expert_num_layers or model.num_layers
    return resident_expert_count_per_layer(system, model) * model.expert_size_bytes * resident_layers


def auto_batch_size(
    system: SystemConfig,
    model: ModelConfig,
    hardware: HardwareConfig,
    sequence_length: int,
) -> int:
    gpu_capacity_bytes = max(1, system.num_gpus) * hardware.gpu.memory_capacity_bytes
    available_kv_bytes = max(0.0, gpu_capacity_bytes - resident_expert_bytes(system, model))
    per_sequence_bytes = kv_cache_bytes_per_sequence(model, sequence_length)
    batch_size = max(1, int(available_kv_bytes // per_sequence_bytes))

    # Keep the existing ASTRA simulator convention: non-vLLM systems use even
    # batches because sub-batches are commonly defined as batch_size / 2.
    if system.name.lower() != "vllm":
        batch_size = max(2, (batch_size // 2) * 2)
    return batch_size


def resolve_batch_size(
    system: SystemConfig,
    model: ModelConfig,
    hardware: HardwareConfig,
    sequence_length: int,
) -> int:
    if isinstance(system.batch_size, int):
        return max(1, system.batch_size)
    if str(system.batch_size).lower() == "auto":
        return auto_batch_size(system, model, hardware, sequence_length)
    raise ValueError(f"Unsupported batch_size value for {system.name}: {system.batch_size}")


def resolve_system_batch_sizes(
    systems: list[SystemConfig],
    model: ModelConfig,
    hardware: HardwareConfig,
    sequence_length: int,
) -> list[SystemConfig]:
    return [
        system.with_batch_size(resolve_batch_size(system, model, hardware, sequence_length))
        for system in systems
    ]
