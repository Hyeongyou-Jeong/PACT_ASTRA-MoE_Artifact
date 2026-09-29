# OpenASTRA

FPGA prototype firmware for the OpenASTRA SSD-integrated MoE accelerator
(Cosmos+ OpenSSD / Zynq platform). Host MoE requests are executed via
**TEP** (Tile-based Expert Processing) over native **GEMM** tiles on the
on-board **NPU** (prototype implementation of the paper EPU).

## What is included

- Firmware application (`AISSDFTL/`) with TEP scheduler and GEMM/NPU drivers
- BSP and hardware platform project (bitstream / HDF)

## FPGA Validation

OpenASTRA implements the TEP execution path in the FPGA prototype firmware.
The implementation supports multi-tile expert execution, out-of-order
dispatch of flash-ready tiles, and Flash-NPU pipelining. We functionally
validated this execution path on the OpenASTRA FPGA prototype.

## Quick Start

1. Program `OpenSSD2_hw_platform_0/OpenSSD2.bit`.
2. Load a **TEP-enabled** `AISSDFTL.elf` on `ps7_cortexa9_0`.
3. Confirm the device with `nvme list`.
4. Pre-place page-contiguous expert weights in flash (`$EXPERT_LBA`).
5. Drive the full flow with stock `nvme-cli` (IFMAP write → MoE `0x80` →
   OFMAP read). See [`docs/nvme-cli.md`](docs/nvme-cli.md).

## Limitations

- FPGA validation here is functional, not a performance characterization.
- Reproducing the experiment requires the OpenASTRA prototype hardware.
- Activation DRAM transfers (`0x81`/`0x82`) support up to 4096 bytes per
  command (covers IFMAP `0x100` and two-tile OFMAP `2048`).
