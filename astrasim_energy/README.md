# ASTRA-MoE Energy Evaluation

## 1. Overview

`astrasim_energy/` is the ASTRA-MoE system energy evaluation. It is a trace-driven, analytical
**dynamic energy** simulator for MoE decode. It replays a real routed-expert trace and reports
per-token energy (J/token), broken down by component, for three systems: GPU Only,
DeepSpeed-NVMe and ASTRA-MoE.

It is pure Python, runs on CPU only (no GPU, CUDA, SSD or FPGA), and is deterministic: repeated
runs produce byte-identical CSV files.

This component is independent of the THER policy evaluation in `ther/`. THER appears here in two
roles only: it is the GPU/SSD expert-residency policy that ASTRA-MoE uses inside the simulator,
and the routing trace was collected using the same trace-collection infrastructure used by the
THER evaluation. The results below are ASTRA-MoE system energy results, not THER policy results.

| System | Name in simulator | Expert placement | Expert computation |
|---|---|---|---|
| GPU Only (vLLM-style) | `vllm` | all experts in GPU HBM | GPU |
| DeepSpeed-NVMe | `deepspeed_nvme` | experts on SSD; active experts are read from flash and sent over PCIe to GPU HBM | GPU |
| ASTRA-MoE | `astra_moe` | THER-selected hot experts in GPU HBM; the rest stay on SSD | GPU for resident experts; in-SSD EPU for the rest (only activations cross PCIe) |

## 2. Quick Start

Run from the artifact root (the directory that contains `astrasim_energy/`):

```bash
python3 -m pip install -r astrasim_energy/requirements.txt   # PyYAML, numpy, h5py; Python >= 3.10
bash astrasim_energy/scripts/setup_realtrace.sh                # downloads the trace (~1.74 GiB)
bash astrasim_energy/scripts/run_realtrace_energy.sh
```

If the archive is already on disk, pass it instead:
`bash astrasim_energy/scripts/setup_realtrace.sh /path/to/realtrace_qwen30_a3b_224req.tar.xz`.

The last line printed should be `Reference check: PASS (18 values, tolerance 0.01% relative)`.

The scripts use the first Python >= 3.10 found on `PATH`. If that interpreter lacks the required
packages (for example, an unrelated virtualenv is active), select one explicitly, e.g.
`PYTHON=/usr/bin/python3.10 bash astrasim_energy/scripts/run_realtrace_energy.sh`.

## 3. Real-Trace Short Path

The real routing trace is the main artifact-evaluation path. It uses:

- **Model:** Qwen3-30B-A3B.
- **Trace:** a pre-collected, real vLLM routing trace of **224 independent requests** (14 logical
  batches of 8 Pile sources × 2 requests; distinct source documents), each with the full
  **1023 decode steps**. See §6.
- **Batch sizes:** GPU Only **B=14**, DeepSpeed-NVMe **B=32**, ASTRA-MoE **B=32** — the batch
  sizes used in the original energy evaluation. They are fixed inputs in
  `configs/systems_realtrace_qwen30.yaml`; the simulator does not derive them. 224 is a multiple
  of both 14 and 32, so every system runs full batches only.
- **ASTRA-MoE residency:** THER policy, GPU expert budget 64 of 128 per layer, window W=16.

**Trace archive.** `realtrace_qwen30_a3b_224req.tar.xz`, 1,872,884,316 bytes (~1.74 GiB;
~2.16 GiB extracted). It is hosted on Zenodo (record 21889194, DOI 10.5281/zenodo.21889194) and
is not part of the source tree:
<https://zenodo.org/records/21889194/files/realtrace_qwen30_a3b_224req.tar.xz?download=1>.
`setup_realtrace.sh` checks the archive's SHA256 against `precollected/SHA256SUMS`, extracts it,
and checks every extracted file. The archive can be given as the first argument, via
`REALTRACE_ARCHIVE`, or placed at `astrasim_energy/precollected/realtrace_qwen30_a3b_224req.tar.xz`;
if none exists, it is downloaded there from the Zenodo URL above, falling back to the GitHub
release mirror <https://github.com/Hyeongyou-Jeong/PACT_ASTRA-MoE_Artifact/releases/download/v3/realtrace_qwen30_a3b_224req.tar.xz> (`REALTRACE_URL` overrides both). Both copies are checked against the same
SHA256. The extraction target is
`REALTRACE_DIR` (default `astrasim_energy/traces/realtrace_qwen30`). See `precollected/README.md`.

**What `run_realtrace_energy.sh` does:**

1. Checks the trace before simulation: 224 distinct requests (and source documents), 1023
   contiguous decode steps per request, 48 MoE layers, top-k = 8, expert ids in [0, 127], no
   repeated expert within a token, 229,152 decode tokens.
2. Converts decode-stage rows to routed-log JSONL (`scripts/convert_routing_trace.py`) without
   altering routing, and checks that the number of converted routed entries equals the HDF5 count
   (87,994,368).
3. Runs the simulator for the three systems on the same trace.
4. Builds the table and checks all 18 values against `reference/realtrace_qwen30_reference.csv`
   (0.01% relative tolerance). It exits non-zero on any failure.

The ~2.5 GB converted log is deleted afterwards unless `KEEP_ROUTED_LOG=1`. Other environment
variables: `PYTHON`, `REALTRACE_DIR`, `REALTRACE_WORK_DIR`.

**Expected runtime** (measured on the development machine, CPU only): `setup_realtrace.sh`
about 2.3 min (checksum, extraction, per-file verification); `run_realtrace_energy.sh` about
8.5 min (conversion ~1.7 min, simulation ~6.6 min), about 6.5 GB peak memory and about 2.5 GB of
temporary disk space for the converted log.

## 4. Expected Results

Qwen3-30B-A3B, real 224-request routing trace (J/token):

| System | Batch | GPU-SM | GPU-HBM | PCIe | SSD-Flash | SSD-EPU | Total |
|---|---|---|---|---|---|---|---|
| GPU Only | 14 | 0.066341 | 0.138609 | 0 | 0 | 0 | **0.204950** |
| DeepSpeed-NVMe | 32 | 0.075966 | 0.174719 | 0.053487 | 0.084509 | 0 | **0.388680** |
| ASTRA-MoE | 32 | 0.043349 | 0.055248 | 2.68e-5 | 0.031098 | 0.016885 | **0.146607** |

- ASTRA-MoE uses **~62.3% less** energy per token than DeepSpeed-NVMe.
- ASTRA-MoE uses **~28.5% less** energy per token than GPU Only.

The last lines printed by `run_realtrace_energy.sh` (the simulator labels ASTRA-MoE as `ASTRA`):

```
Real-trace energy, Qwen3-30B-A3B, 224 requests (J/token)
Model            System          Batch     GPU-SM    GPU-HBM       PCIe  SSD-Flash    SSD-EPU      Total   vs DS  vs GPU  Check
-------------------------------------------------------------------------------------------------------------------------------
Qwen3-30B-A3B    GPU Only           14  0.0663406   0.138609          0          0          0    0.20495   0.527   1.000  PASS
Qwen3-30B-A3B    DeepSpeed-NVMe     32  0.0759658   0.174719  0.0534866  0.0845088          0    0.38868   1.000   1.896  PASS
Qwen3-30B-A3B    ASTRA              32  0.0433494  0.0552475 2.67642e-05   0.031098  0.0168849   0.146607   0.377   0.715  PASS

Qwen3-30B-A3B: ASTRA energy/token -62.3% vs DeepSpeed-NVMe, -28.5% vs GPU Only
...
Reference check: PASS (18 values, tolerance 0.01% relative)
```

### Output files

| Path | Contents |
|---|---|
| `outputs/realtrace_qwen30/realtrace_energy.csv` | real-trace table: model, system, batch, five categories, Total, normalized totals |
| `outputs/realtrace_qwen30/conversion_summary.json` | trace validation and conversion counts |
| `outputs/realtrace_qwen30/qwen3_30b/` | simulator CSVs for the real-trace run |
| `outputs/fig16_regression/fig16_energy.csv` | synthetic regression table (three models) |
| `outputs/fig16_regression/<model>/` | simulator CSVs for the regression run |

Each simulator output directory contains:

| File | Contents |
|---|---|
| `summary.csv` | per system: batch size, generated tokens, cumulative GPU/SSD active-expert counts, total latency, total energy, J/token over all simulator components, throughput |
| `energy_breakdown.csv` | per system and simulator component: total J and J/token (`total` row = sum of all components) |
| `latency_breakdown.csv` | per system and component: total and per-step latency (analytical; flash time is listed under both flash components) |

Running `simulator/main.py` without `--output-dir` writes `summary.csv`, `energy_breakdown.csv` and
`latency_breakdown.csv` directly under `outputs/` (its default output location).

## 5. Energy Model

```
real routed-expert trace (HDF5 -> routed-log JSONL)
        ↓
batch aggregation                           trace_loader.py
        ↓
GPU/SSD residency replay (ASTRA-MoE, THER)  residency_policy.py
        ↓
active experts / routed tokens per step     system_models.py
        ↓
component-level dynamic energy model        energy_model.py
        ↓
J/token                                     main.py -> CSV; scripts/fig16_table.py -> table
```

- **Batch aggregation.** Requests are grouped into static batches of B consecutive requests. A
  batch runs all of its decode steps before the next batch starts. For each (batch, decode step,
  layer), the simulator counts how many tokens are routed to each expert.
- **Residency replay (ASTRA-MoE).** ASTRA-MoE places experts with the THER policy: one sliding
  window per layer, W decode steps long, shared by all requests in the batch; scores are
  routed-token counts. Every W steps it re-selects the top-`gpu_expert_budget` experts per layer
  as GPU-resident. Each step is costed with the placement chosen from earlier steps. The initial
  residents are experts `0..budget-1`.
- **Per-step quantities.** Routed tokens (token × top-k routes) and active experts (unique
  (layer, expert) pairs with at least one routed token); for ASTRA-MoE, split into GPU-resident
  and SSD-resident experts.

### Energy categories

`energy_breakdown.csv` reports simulator components; `scripts/fig16_table.py` maps them to the
reported categories:

| Category | Simulator components |
|---|---|
| GPU-SM | `gpu_compute` + `gpu_shared` + `gpu_l1` + `gpu_l2` |
| GPU-HBM | `gpu_dram` |
| PCIe | `pcie` |
| SSD-Flash | `ssd_flash_internal_read` + `ssd_flash_interface` |
| SSD-EPU | `ssd_epu` + `ssd_dram` (the SSD DRAM that stages expert weights is shown under SSD-EPU) |

The reported **Total** is the sum of these five categories. `gpu_static` (always 0), `gpu_comm`
(GPU Only all-to-all), `ssd_spm` and `ssd_controller` are not part of it, so the `total` row of
`energy_breakdown.csv` / `summary.csv` can differ slightly. The table also reports
`Normalized_Total_vs_DeepSpeed` and `Normalized_Total_vs_GPUOnly`.

### Energy equations (per decode step)

Notation: T tokens in the step; L `num_layers`; H `hidden_size`; S<sub>e</sub>
`expert_size_bytes`; F<sub>e</sub> `expert_flops_per_token`; A `activation_bytes_per_token`;
N<sub>act</sub> active (unique) experts; R routed tokens. J/token = (sum over all steps of the
energy) / (total generated tokens).

| Term | Equation | Code |
|---|---|---|
| GPU compute | (T·L·`dense_flops_per_token` + R<sub>GPU</sub>·F<sub>e</sub>) × 1.0 pJ/FLOP | `gpu_compute_energy` |
| GPU HBM bytes B<sub>HBM</sub> | T·L·H·2·4 + N<sub>act,GPU</sub>·S<sub>e</sub> (DeepSpeed adds the transferred N<sub>act</sub>·S<sub>e</sub> a second time) | `system_models.py` |
| GPU memory hierarchy | ⌈B<sub>HBM</sub>/32⌉ × {82.1 shared, 107 L1, 368 L2, 2090 DRAM} pJ | `gpu_*_energy` |
| PCIe, DeepSpeed | N<sub>act</sub>·S<sub>e</sub>·8 × 5.0 pJ/bit | `pcie_energy` |
| PCIe, ASTRA | R<sub>SSD</sub>·A·2·8 × 5.0 pJ/bit (activations in and out) | `pcie_energy` |
| Flash | N<sub>act,SSD</sub>·S<sub>e</sub>·8 × (3.0 + 4.9) pJ/bit | `flash_internal_read_energy`, `flash_interface_energy` |
| SSD EPU | R<sub>SSD</sub>·F<sub>e</sub> × 1.45 pJ/FLOP | `epu_energy` |
| SSD DRAM | (N<sub>act,SSD</sub>·S<sub>e</sub> + R<sub>SSD</sub>·A·2)·8 × 4.0 pJ/bit | `dram_energy` |

**Batch-level expert reuse.** Expert weight traffic (GPU HBM, flash, DeepSpeed PCIe, SSD DRAM) is
charged once per unique active expert per decode step, however many tokens in the batch use that
expert, so a larger batch amortizes each expert read over more tokens. Expert computation (GPU
and EPU FLOPs) and ASTRA's activation PCIe traffic scale with routed tokens. The same rule
applies to all three systems.

### Energy parameters

| Parameter | Value | Defined in | Source |
|---|---|---|---|
| GPU shared memory access | 82.1 pJ / 32 B | `energy_model.py` | Delestrac et al., ASAP 2024 (IEEE Xplore document 10631142), A100 data-movement energy |
| GPU L1 access | 107.0 pJ / 32 B | `energy_model.py` | same |
| GPU L2 access | 368.0 pJ / 32 B | `energy_model.py` | same |
| GPU HBM (DRAM) access | 2090 pJ / 32 B (≈ 8.16 pJ/bit) | `energy_model.py` | same |
| GPU compute | 1.0 pJ/FLOP | `energy_model.py` | assumption |
| GPU-GPU communication | 5.0 pJ/bit | `hardware_energy.yaml` | assumption (not in the reported total) |
| PCIe | 5.0 pJ/bit, 32 GB/s | `hardware_energy.yaml` | assumption |
| Flash internal read | 3.0 pJ/bit | `energy_model.py` | code cites IEEE Xplore document 10946816 ("Lincoln") |
| Flash interface | 4.9 pJ/bit | `energy_model.py` | KVNAND, arXiv:2512.03608 |
| SSD DRAM | 4.0 pJ/bit | `hardware_energy.yaml` | code cites ACM DOI 10.1145/3361682 |
| SSD EPU | 1.45 pJ/FLOP, 10 TFLOP/s | `hardware_energy.yaml` | energy: RedMulE (IEEE Xplore document 9774759); throughput: assumption |
| SSD SPM | 0.2 pJ/bit | `hardware_energy.yaml` | assumption (not in the reported total) |
| SSD controller | 2.0 W, 10.485 µs per step | `hardware_energy.yaml` | assumption (not in the reported total) |

The GPU memory-hierarchy, GPU compute and flash energies are hard-coded in
`simulator/energy_model.py`; editing `hardware_energy.yaml` does not change them. The YAML keys
`gpu.static_power_w`, `gpu.active_power_w`, `gpu.idle_power_w` and `flash.active_power_w` are
read but not used in energy. The default `DRAM_ENERGY_PJ_PER_BIT = 0.0` in `energy_model.py` is
always overridden by `dram.energy_pj_per_bit` (4.0) from `hardware_energy.yaml`.

## 6. Routing Trace / Provenance

The Qwen3-30B-A3B routing trace used by the short path is a pre-collected, real routed-expert
trace: 224 requests × 1023 decode steps × 48 MoE layers × top-8, stored as two HDF5 files
(`part_a`, `part_b`, 112 requests each). It was collected for this energy evaluation using the
same trace-collection infrastructure used by the THER evaluation (the `mixed` workload settings
with 14 logical batches instead of 1). It is a separate trace from the nine THER policy-evaluation
traces in `ther/` and is not shipped inside `ther/`. The archive is a separate file in the
Zenodo record (DOI 10.5281/zenodo.21889194).

Files in `precollected/`:

| File | Contents |
|---|---|
| `README.md` | archive name, size, lookup order |
| `PROVENANCE.md` | model revision, runtime, workload, lengths, collection commands (identical to the copy inside the archive) |
| `SHA256SUMS` | archive checksum (first line) and checksums of every archive member |
| `ARCHIVE_META.json` | archive size, uncompressed size, SHA256, layout |
| `TRACE_MANIFEST.csv` | per-part requests, decode tokens, layers, top-k, HDF5 size and SHA256 |

## 7. Long Path: Collecting a New Routing Trace

The long path regenerates the routing trace from model inference:

```
Qwen3-30B-A3B inference (vLLM, routing_only instrumentation)
    ↓
routed-expert trace collection
    ↓
HDF5 trace (raw_trace.h5 + requests.json per part)
    ↓
scripts/convert_routing_trace.py
    ↓
energy simulator (--routed-log)
    ↓
J/token
```

The routing trace was collected using the same trace-collection infrastructure used by the THER
evaluation. In this artifact that collector is shipped with the THER component as
`ther.trace_generation.run_pile_request_pool` (MoE router hook in
`ther/evaluation/raw_trace/vllm_plugin.py`); see the "Full path (optional, GPU)" section of
`ther/README.md` for its requirements (2× A100-class GPUs per collector instance, vLLM 0.17.1,
local model weights). The exact settings for the 224-request trace are in
`precollected/PROVENANCE.md`. Those commands were run from the development tree, where the same
collector module is named `expertLoadBalancing.run_pile_request_pool`; the command-line options
are identical. Collection took about 2.1 hours running two collector instances in parallel
(about 64 s per request at physical batch size 1).

To evaluate a newly collected trace, convert it and pass it to the simulator:

```bash
python3 astrasim_energy/scripts/convert_routing_trace.py <routed_log_dir> <trace_dir> [<trace_dir> ...]
python3 -m astrasim_energy.simulator.main --model-config astrasim_energy/configs/model_qwen3_30b.yaml \
    --systems-config astrasim_energy/configs/systems_realtrace_qwen30.yaml \
    --routed-log <routed_log_dir> --output-dir <output_dir>
```

A newly collected trace will not match `reference/realtrace_qwen30_reference.csv` exactly
unless it has identical routing.

## 8. Modeling Scope

- Dynamic compute and data-movement energy only; the latency outputs are simple analytical
  estimates and are not used for the reported energy.
- GPU static/idle power is excluded (`gpu_static` is always 0).
- KV-cache traffic is excluded.
- Dense/attention weight traffic is excluded; per-token GPU HBM traffic covers activations
  (T·L·H·2·4 bytes) and expert weights.
- Expert migration energy (loading newly selected hot experts under THER) is excluded.
- Batching is static; finished requests are not replaced.
- THER re-selects residents every W steps; there is no separate refresh period F.

## 9. Optional Synthetic Regression

```bash
bash astrasim_energy/scripts/run_fig16_regression.sh
```

This is a fast (~2 s) **simulator regression test**, not the primary evaluation. It uses the
simulator's built-in synthetic workload (`trace_loader.generate_synthetic_workload`): a
deterministic, hand-written routing pattern of 64 samples × 16 decode steps in which, for each
token and layer, max(1, top_k − 2) routes cycle over experts 0–7 and the remaining routes follow a
fixed arithmetic pattern. It is **not** derived from measured routing.

It runs Llama4-Scout (`configs/systems.yaml`, auto batch sizing: B=16/40/34, ASTRA budget 4) and
Qwen3-30B-A3B / Qwen3-235B-A22B (`configs/systems_fig16_qwen.yaml`: B=14/32/32, ASTRA budget 64),
writes `outputs/fig16_regression/`, and checks 54 values against
`reference/fig16_regression_reference.csv`. `model_qwen3_235b.yaml` keeps `num_layers: 48` for
this regression (the model has 94 layers). The reference's `published_value` and `note` columns
record two cells of the original synthetic Figure-16 data table that differ from the simulator
output because of spreadsheet aggregation (a double-counted Qwen3-30B DeepSpeed-NVMe SSD-Flash
cell, and `gpu_comm` included in the Qwen3-235B GPU Only GPU-SM cell).

## Directory layout

```
astrasim_energy/
├── README.md
├── requirements.txt
├── configs/
│   ├── hardware_energy.yaml            # hardware/energy parameters
│   ├── model_qwen3_30b.yaml            # model shapes
│   ├── model_llama4_scout.yaml
│   ├── model_qwen3_235b.yaml
│   ├── systems_realtrace_qwen30.yaml   # real-trace path: B=14 / 32 / 32, THER budget 64, W=16
│   ├── systems_fig16_qwen.yaml         # synthetic regression settings for the Qwen models
│   └── systems.yaml                    # auto batch sizing (synthetic regression, Llama4-Scout)
├── simulator/                          # simulator (entry point: python3 -m astrasim_energy.simulator.main)
├── scripts/
│   ├── setup_realtrace.sh              # verify + extract the pre-collected routing trace
│   ├── run_realtrace_energy.sh         # real-trace short path (main AE path)
│   ├── convert_routing_trace.py        # HDF5 routing trace -> routed-log JSONL (validated, verbatim)
│   ├── run_fig16_regression.sh         # optional synthetic regression
│   └── fig16_table.py                  # category mapping, normalization, reference check
├── reference/
│   ├── realtrace_qwen30_reference.csv
│   └── fig16_regression_reference.csv
├── precollected/                       # routing-trace checksums, metadata, provenance (no trace data)
├── traces/                             # created by setup_realtrace.sh (not distributed)
└── outputs/
    ├── realtrace_qwen30/
    └── fig16_regression/
```
