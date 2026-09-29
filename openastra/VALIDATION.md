# OpenASTRA TEP Validation

The TEP execution path was functionally validated on the OpenASTRA FPGA
prototype for the supported multi-tile configuration.

OpenASTRA implements the TEP execution path in the FPGA prototype firmware.
The implementation supports multi-tile expert execution, out-of-order
dispatch of flash-ready tiles, and Flash-NPU pipelining. We functionally
validated this execution path on the OpenASTRA FPGA prototype.

## Scope

- Platform: OpenASTRA FPGA prototype (Cosmos+ OpenSSD / Zynq)
- Path: NVMe MoE execution command (opcode `0x80`) → TEP → native GEMM tiles → FPGA NPU
- Host encoding: `NUMR`=CDW3, `DLEN`=CDW14 (Linux `nvme-cli` compatible)
- Activation DRAM: opcode `0x81` host→IFMAP, `0x82` OFMAP→host
- Nature: functional validation of the supported multi-tile configuration
- Not claimed here: latency, throughput, speedup, run counts, or other
  unrecorded quantitative measurements

## Terminology

NVMe MoE execution command (opcode `0x80`) → TEP → native GEMM tiles → FPGA NPU
