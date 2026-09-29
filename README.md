# ASTRA-MoE Artifact

This package has three independent components:

| Component | Directory | What it provides |
|---|---|---|
| OpenASTRA prototype / SSD validation | `openastra/` | FPGA prototype firmware, bitstream, and host docs (FPGA RTL/IP is not included) |
| THER policy evaluation | `ther/` | CPU replay of released MoE routing traces under six expert-residency policies; GPU routed-token coverage |
| ASTRA-MoE Energy Evaluation | `astrasim_energy/` | trace-driven energy simulator; real Qwen3-30B-A3B routing trace workflow; component-level GPU/SSD energy breakdown (J/token); optional simulator regression tests |

`ther/` and `astrasim_energy/` are separate: each has its own scripts, traces,
and reference results, and neither needs the other to run. The paper’s
calibrated end-to-end latency / throughput model is outside the scope of this
artifact.

Archival copy: https://doi.org/10.5281/zenodo.21889194 (this version; all versions:
https://doi.org/10.5281/zenodo.21888447)

The Qwen3-30B-A3B results reproduce and extend the THER policy analysis with
the rebuttal workload setup. They are not the exact Qwen3-235B operating point
of Fig. 8(b).

## THER scope

**Included:** CPU evaluation of Static, LRU, LFU, Cumulative Weighted-LFU,
THER, and Oracle on nine Pile-based workloads (eight domains + Mixed), using
pre-collected Qwen3-30B-A3B routing traces. An optional CPU budget sweep
reports the same coverage metric while varying the GPU expert-memory budget.

**Optional:** regenerate those traces with Qwen3-30B-A3B + vLLM (`run_full.sh`).

**Not included in THER:** GPU/SSD latency or energy modeling (energy is covered
by `astrasim_energy/`), ASTRASIM throughput reproduction,
the paper’s ~2.5× end-to-end claim, or FPGA RTL/IP.

## THER quick reproduction

Run these commands from the **artifact root** (the directory that contains
this README after unzip). Requires Linux, Python 3.10+, `numpy`, `h5py`, and
`matplotlib`. Also needs `curl`, `tar`, `xz` (or `xz-utils`), and `sha256sum`.
Does not require a GPU, model weights, or vLLM.

```bash
./ther/scripts/setup_traces.sh
./ther/scripts/run_quick.sh
```

`setup_traces.sh` downloads the released pre-collected traces from Zenodo
when they are not already present, verifies SHA256, and extracts them.
`run_quick.sh` selects a Python 3.10+ interpreter (`python3.12` … `python3`)
and checks for `numpy`, `h5py`, and `matplotlib`.

Runtime of the evaluation step is about four minutes (222–228 s on our test
servers). The first-time trace download is ~1.12 GiB.

## THER expected output

Primary outputs under `ther/results/primary/`:

- `<workload>/aggregate_coverage.csv`
- `all_workloads_aggregate_coverage.csv`
- `mixed_temporal_coverage.{png,pdf}`
- `domain_coverage_comparison.{png,pdf}`

Frozen reference copies: `ther/reference_results/`.

Mixed workload, B=12, W=8, F=4 (GPU routed-token coverage):

| Policy | Coverage |
|---|---:|
| Static | 0.10128065825987594 |
| LRU | 0.14529260528893242 |
| LFU | 0.1786688525342801 |
| Cumulative Weighted-LFU | 0.24733398118674502 |
| THER | 0.2655631478534117 |
| Oracle | 0.2975134480595005 |

CSV column key for the cumulative baseline remains `WeightedLFU`.

At this primary setting, THER achieves higher routed-token coverage than the
cumulative token-weighted LFU baseline on all nine workloads.

## THER GPU expert-budget sweep (CPU)

Same traces, same six policies, same W=8 and F=4. Sweeps GPU expert budget
`B ∈ {4, 8, 12, 16, 20}` (Qwen3-30B-A3B has 128 experts per MoE layer).

```bash
./ther/scripts/run_budget_sweep.sh
```

The budget sweep reproduces the **policy-level trend of GPU routed-token
coverage as the expert-memory budget changes**. It does **not** reproduce the
paper's calibrated end-to-end throughput or latency curves.

Outputs under `ther/results/budget_sweep/`:

- `budget_sweep_coverage.csv`
- `mixed_budget_sweep.{png,pdf}`
- `all_workloads_budget_sweep.{png,pdf}`

## THER full trace regeneration

```bash
CUDA_VISIBLE_DEVICES=<ids> ./ther/scripts/run_full.sh
```

Needs two GPUs (or equivalent), vLLM 0.17.1, and local Qwen3-30B-A3B weights.
Details: `ther/README.md`.

## ASTRA-MoE Energy Evaluation

Trace-driven, CPU-only dynamic energy simulator for MoE decode, comparing GPU
Only, DeepSpeed-NVMe, and ASTRA-MoE on a real Qwen3-30B-A3B routing trace
(224 independent requests). Requires Python 3.10+, `PyYAML`, `numpy`, `h5py`.

```bash
bash astrasim_energy/scripts/setup_realtrace.sh    # downloads the routing trace from Zenodo
bash astrasim_energy/scripts/run_realtrace_energy.sh
```

Expected totals (J/token): GPU Only (B=14) ≈ 0.204950, DeepSpeed-NVMe (B=32)
≈ 0.388680, ASTRA-MoE (B=32, THER budget 64, W=16) ≈ 0.146607. The script
checks 18 values against `astrasim_energy/reference/`. The routing-trace
archive (`realtrace_qwen30_a3b_224req.tar.xz`, ~1.74 GiB) is a separate file in
the Zenodo record and is downloaded by `setup_realtrace.sh` when missing. Optional
synthetic regression: `bash astrasim_energy/scripts/run_fig16_regression.sh`.
Details: `astrasim_energy/README.md`.

## OpenASTRA

See `openastra/README.md`. The PACT Zenodo ZIP includes FPGA prototype firmware,
bitstream, and host deployment notes under `openastra/`. FPGA RTL/IP is not
included. This development checkout may not contain the bitstream/firmware.

## Layout

```text
README.md
RELEASE_MANIFEST.md
openastra/          # OpenASTRA FPGA prototype firmware and deployment materials
ther/               # THER policy evaluation: policies, traces, results
astrasim_energy/    # ASTRA-MoE Energy Evaluation: simulator, configs, references
```
