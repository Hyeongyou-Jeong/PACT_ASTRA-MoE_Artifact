from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class GPUConfig:
    name: str
    num_devices: int
    memory_capacity_bytes: int
    flops_per_s: float
    communication_bandwidth_Bps: float
    communication_energy_pj_per_bit: float
    static_power_w: float
    active_power_w: float
    idle_power_w: float


@dataclass(frozen=True)
class PCIeConfig:
    bandwidth_Bps: float
    energy_pj_per_bit: float


@dataclass(frozen=True)
class EPUConfig:
    flops_per_s: float
    energy_pj_per_flop: float


@dataclass(frozen=True)
class FlashConfig:
    bandwidth_Bps: float
    active_power_w: float


@dataclass(frozen=True)
class MemoryEnergyConfig:
    energy_pj_per_bit: float


@dataclass(frozen=True)
class ControllerConfig:
    active_power_w: float
    latency_s: float


@dataclass(frozen=True)
class HardwareConfig:
    gpu: GPUConfig
    pcie: PCIeConfig
    epu: EPUConfig
    flash: FlashConfig
    dram: MemoryEnergyConfig
    spm: MemoryEnergyConfig
    controller: ControllerConfig


@dataclass(frozen=True)
class ModelConfig:
    name: str
    num_layers: int
    num_experts: int
    top_k: int
    hidden_size: int
    ffn_dim: int
    bytes_per_param: int
    expert_size_bytes: int
    parameter_size_bytes: int
    sequence_length: int
    dense_flops_per_token: float
    expert_flops_per_token: float
    activation_bytes_per_token: int
    kv_cache_hidden_size: int | None = None
    kv_cache_num_layers: int | None = None
    resident_expert_num_layers: int | None = None

    @property
    def layer_num(self) -> int:
        return self.num_layers

    @property
    def expert_num(self) -> int:
        return self.num_experts

    @property
    def precision_bytes(self) -> int:
        return self.bytes_per_param

    @property
    def expert_intermediate_size(self) -> int:
        return self.ffn_dim


@dataclass(frozen=True)
class SystemConfig:
    name: str
    batch_size: int | str
    gpu_expert_budget: int
    num_gpus: int
    residency_policy: str
    ther_window_size: int = 16

    def with_batch_size(self, batch_size: int) -> "SystemConfig":
        return SystemConfig(
            name=self.name,
            batch_size=batch_size,
            gpu_expert_budget=self.gpu_expert_budget,
            num_gpus=self.num_gpus,
            residency_policy=self.residency_policy,
            ther_window_size=self.ther_window_size,
        )


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"YAML root must be a mapping: {path}")
    return data


def load_hardware_config(path: str | Path) -> HardwareConfig:
    data = load_yaml(path)
    return HardwareConfig(
        gpu=GPUConfig(**data["gpu"]),
        pcie=PCIeConfig(**data["pcie"]),
        epu=EPUConfig(**data["epu"]),
        flash=FlashConfig(**data["flash"]),
        dram=MemoryEnergyConfig(**data["dram"]),
        spm=MemoryEnergyConfig(**data["spm"]),
        controller=ControllerConfig(**data["controller"]),
    )


def load_model_config(path: str | Path) -> ModelConfig:
    data = load_yaml(path)
    return ModelConfig(**data)


def load_system_configs(path: str | Path) -> list[SystemConfig]:
    data = load_yaml(path)
    systems = data.get("systems")
    if not isinstance(systems, list):
        raise ValueError(f"systems.yaml must contain a 'systems' list: {path}")
    return [SystemConfig(**item) for item in systems]
