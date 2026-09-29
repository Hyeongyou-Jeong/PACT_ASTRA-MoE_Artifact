# THER expert-residency artifact

```text
workload manifest
  -> (optional) Qwen3-30B-A3B + vLLM routing_only collection
  -> HDF5 routing trace
  -> Static / LRU / LFU / Cumulative Weighted-LFU / THER / Oracle
  -> GPU routed-token coverage
```

This is not an exact reproduction of paper Fig. 8(b) (Qwen3-235B-scale). It
reproduces and extends the THER analysis on Qwen3-30B-A3B with the rebuttal-style
Pile workloads. Coverage uses only routing traces and residency decisions; no
simulated SSD/GPU latency enters the metric.

## Trace format

Each workload directory contains `raw_trace.h5` (`routing_only`):

- `index/*` — request id, stage, generation step, …
- `model/moe_layer_ids` — MoE layer list (48 for this model)
- `routing/selected_expert_ids` — shape `[tokens, layers, top_k]`
- `routing/selected_expert_weights`

No embeddings or full router logits are stored.

```bash
python3.10 ther/scripts/ther_sanity_check.py ther/traces/data/<workload>
```

vLLM capture notes: `ther/docs/VLLM_INSTRUMENTATION.md`.

## Workloads

Manifests: `ther/workloads/workload_*.json`.

| Workload | Construction |
|---|---|
| Mixed | 8 Pile sources × 2 requests, interleaved (logical BS=16) |
| ArXiv, PMC, GitHub, StackExchange, Wikipedia, FreeLaw, HackerNews, Pile-CC | 16 requests from one source (`1×16`; artifact extension) |

Dataset: `monology/pile-uncopyrighted` revision
`3be90335b66f24456a5d6659d9c8d208c0357119`. Sample seed 42.

## Policies

Code: `ther/policies/residency.py` and `ther/policies/window.py`.
Machine-readable CSV keys keep historical names (e.g. `WeightedLFU`); display
names below are used in prose and figure legends.

**Static.** Keeps experts `0,...,B-1` resident throughout the trace.

**LRU.** Uses binary expert accesses observed over each F-step interval and
selects the B most recently accessed experts at the residency refresh.
Interval-local statistics are reset after each refresh. This is not a textbook
lifetime cache-stack LRU.

**LFU.** Counts binary expert activations within each F-step interval and
selects the B most frequently activated experts. Routed-token multiplicity does
not affect this baseline. Interval-local statistics are reset after each
refresh.

**Cumulative Weighted-LFU** (CSV key `WeightedLFU`). Accumulates routed-token
counts over the full observed history and selects the global Top-B experts every
F routing intervals. It does **not** use a finite history window W.

**THER.** Accumulates routed-token counts only over the most recent W routing
intervals and refreshes the Top-B GPU-resident experts every F intervals.

**Oracle.** Uses future routing load over the next F-step interval under the
same expert budget B, refreshing at the same F-step granularity. It is an
offline upper bound for that constrained objective.

Warmup: events before the first refresh (`t < F`, 1-based) are excluded from
aggregate coverage.

### Cumulative Weighted-LFU and THER

Cumulative Weighted-LFU differs from THER only in its history horizon: it retains
routed-token counts over the full observed history, whereas THER retains only
the most recent W routing intervals.

If token-weighted LFU is instead given the same sliding W-step history, F-step
refresh period, Top-B rule, and tie breaking as THER, the two policies are
mathematically equivalent. The released cumulative baseline is therefore not a
W-matched token-weighted LFU.

At primary (W=8, F=4), THER achieves higher routed-token coverage than the
cumulative token-weighted LFU baseline on all nine workloads. A few supplemental
(W, F) cells have small regressions the other way (e.g. pubmed_central W=4,F=8;
stackexchange W=4,F=8 and W=8,F=8).

## Metric

**GPU routed-token coverage** =
(routed assignments whose expert is GPU-resident) /
(all routed assignments).

Computed in `ther/evaluation/coverage.py`. CSV column `activation_coverage` is
secondary; do not treat it as the primary metric.

## Primary configuration

| Item | Value |
|---|---|
| Model | Qwen/Qwen3-30B-A3B |
| HF revision | `ad44e777bcd18fa416d9da3bd8f70d33ebb85d39` |
| vLLM | 0.17.1 |
| ISL / OSL | 4096 / 1024 |
| Logical batch | 16 |
| Tensor parallel | 2 (full path) |
| Trace level | `routing_only` |
| GPU expert budget B | 12 / MoE layer |
| THER W / F | 8 / 4 |

## Quick path (CPU)

`./ther/scripts/setup_traces.sh` downloads the released pre-collected routing
traces from Zenodo when they are not already present, verifies SHA256, and
extracts them.

```bash
./ther/scripts/setup_traces.sh
./ther/scripts/run_quick.sh
```

About four minutes (222–228 s measured). Writes `ther/results/primary/`.

Run from the artifact root. Requires Python 3.10+, `numpy`, `h5py`, and
`matplotlib`. No GPU, model weights, or vLLM.

## GPU expert-budget sweep (CPU)

Same traces and six policies; W=8, F=4, logical batch 16. Sweeps GPU expert
budget `B ∈ {4, 8, 12, 16, 20}` (128 experts/layer on Qwen3-30B-A3B).

```bash
./ther/scripts/run_budget_sweep.sh
```

The budget sweep reproduces the policy-level trend of GPU routed-token
coverage as the expert-memory budget changes. It does not reproduce the
paper's calibrated end-to-end throughput or latency curves.

Writes `ther/results/budget_sweep/`.

## Full path (optional, GPU)

```bash
CUDA_VISIBLE_DEVICES=<ids> ./ther/scripts/run_full.sh
```

Builds manifests, collects nine traces, validates, then runs the quick
evaluator. Needs two A100-class GPUs (or equivalent), vLLM 0.17.1, and local
weights. Sequential collection took on the order of ~3 hours on our machine.

Entry points: `ther/scripts/collect_one.py` →
`ther.trace_generation.run_pile_request_pool` with the MoE router hook in
`ther/evaluation/raw_trace/vllm_plugin.py`.

## Outputs

**Primary** (`ther/results/primary/`): per-workload aggregates/temporals,
`all_workloads_aggregate_coverage.csv`, Mixed temporal and domain comparison
figures.

**Reference:** `ther/reference_results/`

**Supplemental** (optional): W/F and B sweeps under `ther/results/supplemental/`.
Optional sanity: `matched_weighted_lfu_check.csv` verifies that a W-matched
token-weighted formulation matches THER.

**Characterization** (optional): `ther/results/characterization/`

## Notes vs Fig. 8(b)

- Model scale: Qwen3-30B-A3B vs Fig. 8(b) Qwen3-235B-class setting
- Per-domain `1×16` batches extend beyond the published mixed focus
- New collects use `routing_only` (enough for residency coverage)

## Layout

```text
ther/
  policies/           # residency schedules + THER window
  evaluation/         # trace replay + coverage
  trace_generation/   # optional vLLM collection
  workloads/
  scripts/            # setup_traces, run_quick, run_full, …
  precollected/       # trace archive + checksums
  traces/             # created by setup_traces.sh
  reference_results/
  results/{primary,budget_sweep,supplemental,characterization}/
  docs/
  tests/
```
