# RELEASE_MANIFEST

## Components
- `openastra/` — OpenASTRA prototype / SSD validation
- `ther/` — THER policy evaluation
- `astrasim_energy/` — ASTRA-MoE Energy Evaluation

## Primary (THER policy evaluation)
- THER quick path (CPU): released routing traces → six residency policies →
  GPU routed-token coverage
- `./ther/scripts/setup_traces.sh` then `./ther/scripts/run_quick.sh`

## Optional (THER policy evaluation)
- GPU expert-budget sweep (CPU): `./ther/scripts/run_budget_sweep.sh`
  (policy-level routed-token coverage vs B; not end-to-end throughput)
- Full trace regeneration: `./ther/scripts/run_full.sh` (`ther/trace_generation/`)
- Sensitivity / characterization under `ther/results/supplemental/` and
  `ther/results/characterization/`

## Out of scope
- Calibrated GPU/SSD latency and throughput model
- ASTRASIM end-to-end throughput as a primary AE path
- Paper ~2.5× throughput claim

## THER trace archive
- `ther/precollected/ther_traces_qwen3_30b_a3b.tar.xz`
- 1202919576 bytes
- SHA256 `64612e7d5256436174d23b753987d387574b7d7c179d6dc579c8fc7439409f7d`
- Zenodo: https://zenodo.org/records/21888851
- Direct download:
  https://zenodo.org/records/21888851/files/ther_traces_qwen3_30b_a3b.tar.xz?download=1
- Auto-downloaded by `./ther/scripts/setup_traces.sh` when missing

## ASTRA-MoE Energy Evaluation
- Trace-driven, CPU-only energy simulator (`astrasim_energy/simulator/`)
- Real Qwen3-30B-A3B routing trace workflow (224 independent requests):
  `bash astrasim_energy/scripts/setup_realtrace.sh` then
  `bash astrasim_energy/scripts/run_realtrace_energy.sh`
- Component-level GPU/SSD energy breakdown (GPU-SM, GPU-HBM, PCIe, SSD-Flash,
  SSD-EPU) and J/token for GPU Only (B=14), DeepSpeed-NVMe (B=32), ASTRA-MoE
  (B=32, THER budget 64, W=16)
- Optional simulator regression tests:
  `bash astrasim_energy/scripts/run_fig16_regression.sh` (synthetic workload)
- Routing-trace archive: `realtrace_qwen30_a3b_224req.tar.xz`, 1872884316 bytes,
  SHA256 `0f2b97f2c7f26ddc389b6427754230b5d7972ba965742a919c6ad4ca5708410a`;
  distributed separately (not in this tree); checksums and provenance in
  `astrasim_energy/precollected/`
- Zenodo: https://doi.org/10.5281/zenodo.21889194
- Direct download:
  https://zenodo.org/records/21889194/files/realtrace_qwen30_a3b_224req.tar.xz?download=1
- Mirror (GitHub release `v3`):
  https://github.com/Hyeongyou-Jeong/PACT_ASTRA-MoE_Artifact/releases/download/v3/realtrace_qwen30_a3b_224req.tar.xz
- Auto-downloaded by `setup_realtrace.sh` when missing (Zenodo first, then the mirror)

## OpenASTRA
- FPGA prototype firmware, bitstream (`openastra/bitstream/OpenSSD2.bit`),
  platform HDF, and host docs included under `openastra/`
- FPGA RTL/IP not included

## Dependencies
- THER quick path: Python 3.10+, `numpy`, `h5py`, `matplotlib`
- Energy evaluation: Python 3.10+, `PyYAML`, `numpy`, `h5py` (CPU only)
- THER full path / new routing-trace collection: GPU, vLLM 0.17.1, Qwen3-30B-A3B weights, dataset access

## Validation
- THER quick path vs `ther/reference_results/`: abs_err_max = 0
- Synthetic policy tests: PASS
- Measured quick-path wall time ≈ 222–228 s
- Display name: Cumulative Weighted-LFU (CSV key remains `WeightedLFU`)
- Energy real-trace path vs `astrasim_energy/reference/realtrace_qwen30_reference.csv`:
  18/18 values within 0.01% relative
- Energy synthetic regression vs `astrasim_energy/reference/fig16_regression_reference.csv`:
  54/54 values
