from __future__ import annotations

from dataclasses import dataclass, field

try:
    from .config import HardwareConfig, ModelConfig, SystemConfig
    from .energy_model import (
        controller_energy,
        dram_energy,
        epu_energy,
        flash_internal_read_energy,
        flash_interface_energy,
        gpu_comm_energy,
        gpu_compute_energy,
        gpu_dram_energy,
        gpu_l1_energy,
        gpu_l2_energy,
        gpu_shared_energy,
        pcie_energy,
        spm_energy,
    )
    from .latency_model import (
        epu_latency,
        flash_latency,
        pcie_latency,
        ssd_core_latency,
        total_latency_overlap,
    )
    from .residency_policy import build_residency_policy
    from .trace_loader import TraceStep, TraceWorkload
except ImportError:  # Allows direct imports when running simulator/main.py.
    from config import HardwareConfig, ModelConfig, SystemConfig
    from energy_model import (
        controller_energy,
        dram_energy,
        epu_energy,
        flash_internal_read_energy,
        flash_interface_energy,
        gpu_comm_energy,
        gpu_compute_energy,
        gpu_dram_energy,
        gpu_l1_energy,
        gpu_l2_energy,
        gpu_shared_energy,
        pcie_energy,
        spm_energy,
    )
    from latency_model import (
        epu_latency,
        flash_latency,
        pcie_latency,
        ssd_core_latency,
        total_latency_overlap,
    )
    from residency_policy import build_residency_policy
    from trace_loader import TraceStep, TraceWorkload


@dataclass
class ComponentBreakdown:
    latency_s: float = 0.0
    energy_j: float = 0.0


@dataclass
class SimulationResult:
    system: str
    batch_size: int
    total_generated_tokens: int
    num_steps: int
    components: dict[str, ComponentBreakdown] = field(default_factory=dict)
    total_latency_s: float = 0.0
    total_energy_j: float = 0.0
    gpu_expert_count: int = 0
    ssd_expert_count: int = 0

    def add(self, component: str, latency_s: float, energy_j: float) -> None:
        if component not in self.components:
            self.components[component] = ComponentBreakdown()
        self.components[component].latency_s += max(0.0, latency_s)
        self.components[component].energy_j += max(0.0, energy_j)

    def add_total_latency(self, latency_s: float) -> None:
        self.total_latency_s += max(0.0, latency_s)

    def add_expert_counts(self, gpu_experts: int, ssd_experts: int) -> None:
        self.gpu_expert_count += max(0, gpu_experts)
        self.ssd_expert_count += max(0, ssd_experts)

    def finalize(self) -> None:
        self.total_energy_j = sum(component.energy_j for component in self.components.values())
        self.components["total"] = ComponentBreakdown(
            latency_s=self.total_latency_s,
            energy_j=self.total_energy_j,
        )

    @property
    def energy_per_token_j(self) -> float:
        if self.total_generated_tokens <= 0:
            return 0.0
        return self.total_energy_j / self.total_generated_tokens

    @property
    def throughput_tokens_per_s(self) -> float:
        if self.total_latency_s <= 0:
            return 0.0
        return self.total_generated_tokens / self.total_latency_s


def _gpu_latency(num_flops: float, hardware: HardwareConfig, num_gpus: int) -> float:
    effective_flops = hardware.gpu.flops_per_s * max(1, num_gpus)
    if num_flops <= 0:
        return 0.0
    if effective_flops <= 0:
        return float("inf")
    return num_flops / effective_flops


def _gpu_comm_latency(num_bytes: float, hardware: HardwareConfig) -> float:
    if num_bytes <= 0:
        return 0.0
    if hardware.gpu.communication_bandwidth_Bps <= 0:
        return float("inf")
    return num_bytes / hardware.gpu.communication_bandwidth_Bps


def _step_routed_token_count(step: TraceStep) -> int:
    return sum(sum(layer_counts.values()) for layer_counts in step.counts_by_layer.values())


def _attention_flops(step: TraceStep, model: ModelConfig) -> float:
    return step.token_count * model.num_layers * model.dense_flops_per_token


def _moe_flops(routed_token_count: int, model: ModelConfig) -> float:
    return routed_token_count * model.expert_flops_per_token


def _attention_hbm_bytes(step: TraceStep, model: ModelConfig) -> float:
    return step.token_count * model.num_layers * model.hidden_size * model.bytes_per_param * 4


def _gpu_moe_hbm_bytes(active_expert_count: int, model: ModelConfig) -> float:
    return active_expert_count * model.expert_size_bytes


def _active_experts_by_layer(step: TraceStep) -> int:
    return sum(
        sum(1 for count in layer_counts.values() if count > 0)
        for layer_counts in step.counts_by_layer.values()
    )


def _gpu_experts_for_all_resident(step: TraceStep) -> int:
    return _active_experts_by_layer(step)


def _expert_owner_gpu(expert_id: int, model: ModelConfig, num_gpus: int) -> int:
    devices = max(1, num_gpus)
    experts_per_gpu = max(1, (model.num_experts + devices - 1) // devices)
    return min(devices - 1, expert_id // experts_per_gpu)


def _moe_imbalance_stall_latency(step: TraceStep, model: ModelConfig, hardware: HardwareConfig, num_gpus: int) -> float:
    """Estimate vLLM expert-parallel stall from routed-token imbalance."""
    devices = max(1, num_gpus)
    if devices == 1:
        return 0.0

    stall_s = 0.0
    for layer_counts in step.counts_by_layer.values():
        tokens_by_gpu = [0 for _ in range(devices)]
        for expert_id, count in layer_counts.items():
            tokens_by_gpu[_expert_owner_gpu(expert_id, model, devices)] += count
        times = [
            _gpu_latency(tokens * model.expert_flops_per_token, hardware, 1)
            for tokens in tokens_by_gpu
        ]
        avg_time = sum(times) / devices
        stall_s += max(0.0, max(times) - avg_time)
    return stall_s


def _empty_result(system: str, cfg: SystemConfig, workload: TraceWorkload) -> SimulationResult:
    return SimulationResult(
        system=system,
        batch_size=cfg.batch_size,
        total_generated_tokens=workload.total_generated_tokens,
        num_steps=workload.num_steps,
    )


def simulate_vllm_gpu_only(
    hardware: HardwareConfig,
    model: ModelConfig,
    system: SystemConfig,
    workload: TraceWorkload,
) -> SimulationResult:
    """vLLM/GPU-only expert parallelism: all experts are GPU-resident."""
    result = _empty_result("vllm", system, workload)
    num_gpus = max(1, system.num_gpus)

    for step in workload.steps:
        routed_tokens = _step_routed_token_count(step)
        active_gpu_experts = _gpu_experts_for_all_resident(step)
        attention_flops = _attention_flops(step, model)
        moe_flops = _moe_flops(routed_tokens, model)
        gpu_flops = attention_flops + moe_flops
        t_attention = _gpu_latency(attention_flops, hardware, num_gpus)
        t_moe = _gpu_latency(moe_flops, hardware, num_gpus)
        comm_bytes = routed_tokens * model.activation_bytes_per_token * 2 if num_gpus > 1 else 0
        t_comm = _gpu_comm_latency(comm_bytes, hardware)
        t_stall = _moe_imbalance_stall_latency(step, model, hardware, num_gpus)
        t_total = t_attention + t_moe + t_comm + t_stall

        gpu_hbm_bytes = _attention_hbm_bytes(step, model) + _gpu_moe_hbm_bytes(active_gpu_experts, model)

        result.add("gpu_static", t_total, 0.0)
        result.add("gpu_compute", t_attention + t_moe, gpu_compute_energy(gpu_flops))
        result.add("gpu_shared", 0.0, gpu_shared_energy(gpu_hbm_bytes))
        result.add("gpu_l1", 0.0, gpu_l1_energy(gpu_hbm_bytes))
        result.add("gpu_l2", 0.0, gpu_l2_energy(gpu_hbm_bytes))
        result.add("gpu_dram", 0.0, gpu_dram_energy(gpu_hbm_bytes))
        result.add("gpu_comm", t_comm, gpu_comm_energy(comm_bytes, hardware.gpu.communication_energy_pj_per_bit))
        result.add("gpu_moe_stall", t_stall, 0.0)
        result.add_expert_counts(gpu_experts=active_gpu_experts, ssd_experts=0)
        result.add_total_latency(t_total)

    result.finalize()
    return result


def simulate_deepspeed_nvme(
    hardware: HardwareConfig,
    model: ModelConfig,
    system: SystemConfig,
    workload: TraceWorkload,
) -> SimulationResult:
    """DeepSpeed-NVMe: selected expert weights are fetched, then computed on GPU."""
    result = _empty_result("deepspeed_nvme", system, workload)
    num_gpus = max(1, system.num_gpus)

    for step in workload.steps:
        routed_tokens = _step_routed_token_count(step)
        active_experts = _active_experts_by_layer(step)
        transferred_expert_weight_bytes = active_experts * model.expert_size_bytes

        attention_flops = _attention_flops(step, model)
        moe_flops = _moe_flops(routed_tokens, model)
        gpu_flops = attention_flops + moe_flops
        t_attention = _gpu_latency(attention_flops, hardware, num_gpus)
        t_moe = _gpu_latency(moe_flops, hardware, num_gpus)
        t_flash = flash_latency(transferred_expert_weight_bytes, hardware.flash.bandwidth_Bps)
        t_pcie = pcie_latency(transferred_expert_weight_bytes, hardware.pcie.bandwidth_Bps)
        t_offload = t_flash + t_pcie
        t_total = t_attention + t_offload + t_moe

        gpu_hbm_bytes = (
            _attention_hbm_bytes(step, model)
            + _gpu_moe_hbm_bytes(active_experts, model)
            + transferred_expert_weight_bytes
        )

        result.add("gpu_static", t_total, 0.0)
        result.add("gpu_compute", t_attention + t_moe, gpu_compute_energy(gpu_flops))
        result.add("gpu_shared", 0.0, gpu_shared_energy(gpu_hbm_bytes))
        result.add("gpu_l1", 0.0, gpu_l1_energy(gpu_hbm_bytes))
        result.add("gpu_l2", 0.0, gpu_l2_energy(gpu_hbm_bytes))
        result.add("gpu_dram", 0.0, gpu_dram_energy(gpu_hbm_bytes))
        result.add("pcie", t_pcie, pcie_energy(transferred_expert_weight_bytes, hardware.pcie.energy_pj_per_bit))
        result.add("ssd_flash_internal_read", t_flash, flash_internal_read_energy(transferred_expert_weight_bytes))
        result.add("ssd_flash_interface", t_flash, flash_interface_energy(transferred_expert_weight_bytes))
        result.add_expert_counts(gpu_experts=active_experts, ssd_experts=active_experts)
        result.add_total_latency(t_total)

    result.finalize()
    return result


def simulate_astra_moe(
    hardware: HardwareConfig,
    model: ModelConfig,
    system: SystemConfig,
    workload: TraceWorkload,
) -> SimulationResult:
    """ASTRA-MoE: GPU-resident hot experts and in-SSD cold expert compute."""
    result = _empty_result("astra_moe", system, workload)
    num_gpus = max(1, system.num_gpus)
    residency = build_residency_policy(
        system.residency_policy,
        model=model,
        gpu_expert_budget=system.gpu_expert_budget,
        ther_window_size=system.ther_window_size,
    )

    for step in workload.steps:
        gpu_routed_tokens = 0
        ssd_routed_tokens = 0
        active_gpu_experts = 0
        active_ssd_experts = 0

        for layer, layer_counts in step.counts_by_layer.items():
            residents = residency.residents_for_layer(layer)
            for expert_id, count in layer_counts.items():
                if expert_id in residents:
                    gpu_routed_tokens += count
                    if count > 0:
                        active_gpu_experts += 1
                else:
                    ssd_routed_tokens += count
                    if count > 0:
                        active_ssd_experts += 1

        attention_flops = _attention_flops(step, model)
        gpu_moe_flops = _moe_flops(gpu_routed_tokens, model)
        gpu_flops = attention_flops + gpu_moe_flops
        t_attention = _gpu_latency(attention_flops, hardware, num_gpus)
        t_gpu_moe = _gpu_latency(gpu_moe_flops, hardware, num_gpus)
        t_gpu_path = t_attention + t_gpu_moe

        activation_output_bytes = ssd_routed_tokens * model.activation_bytes_per_token * 2
        flash_bytes = active_ssd_experts * model.expert_size_bytes
        epu_flops = _moe_flops(ssd_routed_tokens, model)
        dram_bytes = flash_bytes + activation_output_bytes
        spm_bytes = activation_output_bytes

        t_pcie = pcie_latency(activation_output_bytes, hardware.pcie.bandwidth_Bps)
        t_flash = flash_latency(flash_bytes, hardware.flash.bandwidth_Bps)
        t_epu = epu_latency(epu_flops, hardware.epu.flops_per_s)
        t_controller = hardware.controller.latency_s if active_ssd_experts > 0 else 0.0
        t_ssd_core = ssd_core_latency(t_flash, t_epu, t_controller) if active_ssd_experts > 0 else 0.0
        t_total = total_latency_overlap(t_gpu_path, t_ssd_core, t_pcie)

        gpu_hbm_bytes = _attention_hbm_bytes(step, model) + _gpu_moe_hbm_bytes(active_gpu_experts, model)

        result.add("gpu_static", t_total, 0.0)
        result.add("gpu_compute", t_attention + t_gpu_moe, gpu_compute_energy(gpu_flops))
        result.add("gpu_shared", 0.0, gpu_shared_energy(gpu_hbm_bytes))
        result.add("gpu_l1", 0.0, gpu_l1_energy(gpu_hbm_bytes))
        result.add("gpu_l2", 0.0, gpu_l2_energy(gpu_hbm_bytes))
        result.add("gpu_dram", 0.0, gpu_dram_energy(gpu_hbm_bytes))
        result.add("pcie", t_pcie, pcie_energy(activation_output_bytes, hardware.pcie.energy_pj_per_bit))
        result.add("ssd_flash_internal_read", t_flash, flash_internal_read_energy(flash_bytes))
        result.add("ssd_flash_interface", t_flash, flash_interface_energy(flash_bytes))
        result.add("ssd_epu", t_epu, epu_energy(epu_flops, hardware.epu.energy_pj_per_flop))
        result.add("ssd_dram", 0.0, dram_energy(dram_bytes, hardware.dram.energy_pj_per_bit))
        result.add("ssd_spm", 0.0, spm_energy(spm_bytes, hardware.spm.energy_pj_per_bit))
        result.add("ssd_controller", t_controller, controller_energy(t_controller, hardware.controller.active_power_w))
        result.add_expert_counts(gpu_experts=active_gpu_experts, ssd_experts=active_ssd_experts)
        result.add_total_latency(t_total)

        residency.observe_step(step.decode_step, step.counts_by_layer)

    result.finalize()
    return result
