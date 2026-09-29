from __future__ import annotations

import math


PJ_TO_J = 1e-12
BITS_PER_BYTE = 8

GPU_STATIC_POWER_W = 55.0
GPU_SHARED_ENERGY_PJ_PER_32B = 82.1 #https://ieeexplore.ieee.org/abstract/document/10631142
GPU_L1_ENERGY_PJ_PER_32B = 107.0 #https://ieeexplore.ieee.org/abstract/document/10631142
GPU_L2_ENERGY_PJ_PER_32B = 368.0 #https://ieeexplore.ieee.org/abstract/document/10631142
GPU_DRAM_ENERGY_PJ_PER_32B = 2090.0 #https://ieeexplore.ieee.org/abstract/document/10631142
GPU_COMPUTE_ENERGY_PJ_PER_FLOP = 1.0
GPU_COMMUNICATION_ENERGY_PJ_PER_BIT = 5.0
FLASH_INTERNAL_READ_ENERGY_PJ_PER_BIT = 3.0 #Lincoln - https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10946816
FLASH_INTERFACE_ENERGY_PJ_PER_BIT = 4.9 #KVNAND - https://arxiv.org/pdf/2512.03608 
FLASH_READ_ENERGY_PJ_PER_BIT = 7.9
DRAM_ENERGY_PJ_PER_BIT = 0.0 #4.0 https://dl.acm.org/doi/epdf/10.1145/3361682
SPM_ENERGY_PJ_PER_BIT = 0.2
PCIE_ENERGY_PJ_PER_BIT = 5.0 #
EPU_ENERGY_PJ_PER_FLOP = 1.45 #redmule - https://ieeexplore.ieee.org/abstract/document/9774759
CONTROLLER_ACTIVE_POWER_W = 2.0


def _access32_energy(num_bytes: float, energy_pj_per_32b: float) -> float:
    if num_bytes <= 0:
        return 0.0
    return math.ceil(num_bytes / 32.0) * energy_pj_per_32b * PJ_TO_J


def _bit_energy(num_bytes: float, energy_pj_per_bit: float) -> float:
    return max(0.0, num_bytes) * BITS_PER_BYTE * energy_pj_per_bit * PJ_TO_J


def gpu_static_energy(
    t_s: float,
    num_gpus: int = 1,
    static_power_w: float = GPU_STATIC_POWER_W,
) -> float:
    return static_power_w * max(0.0, t_s) * max(1, num_gpus)


def gpu_shared_energy(num_bytes: float) -> float:
    return _access32_energy(num_bytes, GPU_SHARED_ENERGY_PJ_PER_32B)


def gpu_l1_energy(num_bytes: float) -> float:
    return _access32_energy(num_bytes, GPU_L1_ENERGY_PJ_PER_32B)


def gpu_l2_energy(num_bytes: float) -> float:
    return _access32_energy(num_bytes, GPU_L2_ENERGY_PJ_PER_32B)


def gpu_dram_energy(num_bytes: float) -> float:
    return _access32_energy(num_bytes, GPU_DRAM_ENERGY_PJ_PER_32B)


def gpu_compute_energy(num_flops: float) -> float:
    return max(0.0, num_flops) * GPU_COMPUTE_ENERGY_PJ_PER_FLOP * PJ_TO_J


def gpu_comm_energy(
    num_bytes: float,
    energy_pj_per_bit: float = GPU_COMMUNICATION_ENERGY_PJ_PER_BIT,
) -> float:
    return _bit_energy(num_bytes, energy_pj_per_bit)


def flash_internal_read_energy(
    num_bytes: float,
    energy_pj_per_bit: float = FLASH_INTERNAL_READ_ENERGY_PJ_PER_BIT,
) -> float:
    return _bit_energy(num_bytes, energy_pj_per_bit)


def flash_interface_energy(
    num_bytes: float,
    energy_pj_per_bit: float = FLASH_INTERFACE_ENERGY_PJ_PER_BIT,
) -> float:
    return _bit_energy(num_bytes, energy_pj_per_bit)


def flash_read_energy(num_bytes: float, energy_pj_per_bit: float = FLASH_READ_ENERGY_PJ_PER_BIT) -> float:
    return _bit_energy(num_bytes, energy_pj_per_bit)


def dram_energy(num_bytes: float, energy_pj_per_bit: float = DRAM_ENERGY_PJ_PER_BIT) -> float:
    return _bit_energy(num_bytes, energy_pj_per_bit)


def spm_energy(num_bytes: float, energy_pj_per_bit: float = SPM_ENERGY_PJ_PER_BIT) -> float:
    return _bit_energy(num_bytes, energy_pj_per_bit)


def pcie_energy(num_bytes: float, energy_pj_per_bit: float = PCIE_ENERGY_PJ_PER_BIT) -> float:
    return _bit_energy(num_bytes, energy_pj_per_bit)


def epu_energy(num_flops: float, energy_pj_per_flop: float = EPU_ENERGY_PJ_PER_FLOP) -> float:
    return max(0.0, num_flops) * energy_pj_per_flop * PJ_TO_J


def controller_energy(
    t_active_s: float,
    active_power_w: float = CONTROLLER_ACTIVE_POWER_W,
) -> float:
    return active_power_w * max(0.0, t_active_s)
