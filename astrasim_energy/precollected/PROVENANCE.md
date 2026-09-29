# Qwen3-30B-A3B real routed-expert trace pool (224 requests)

Collected with the THER artifact collector (`expertLoadBalancing.run_pile_request_pool`) using the
same settings as `ther/traces/data/mixed`, except `--num-batches 14` (1 in the original).

| Item | Value |
|---|---|
| Model | Qwen/Qwen3-30B-A3B (HF revision ad44e777bcd18fa416d9da3bd8f70d33ebb85d39) |
| Runtime | vLLM 0.17.1, tensor parallel 2, bf16, enforce_eager, physical batch 1 |
| Workload | monology/pile-uncopyrighted @ 3be90335b66f24456a5d6659d9c8d208c0357119; 8 sources x 2 requests per logical batch; 14 logical batches; seed/crop/split seed 42 |
| Requests | 224 (28 per source), all distinct documents |
| Lengths | ISL 4096, OSL 1024 (min_tokens=max_tokens=1024; 1023 decode forwards per request) |
| Trace level | routing_only (selected top-8 expert ids and weights per token and MoE layer) |
| Collection | 2 x (2x A100-SXM4-40GB); part_a = logical batches 0-6 (7353 s), part_b = batches 7-13 (7503 s) |
| Manifest build | PYTHONHASHSEED=0 (the builder's reservoir seed uses Python's hash()) |

Commands (from the artifact root; `<OUT>` is any output directory):

```bash
PYTHONHASHSEED=0 python3.10 -m expertLoadBalancing.run_pile_request_pool --output-dir <OUT>/manifest \
  --num-batches 14 --logical-batch-size 16 --sources-per-batch 8 --requests-per-source 2 \
  --input-length 4096 --output-length 1024 --seed 42 --crop-seed 42 --split-seed 42 \
  --max-scan-per-source 500000 --model-name Qwen/Qwen3-30B-A3B --cache-dir <HF_CACHE> \
  --trace-level routing_only --build-manifest-only

CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=<GPU_A,GPU_B> \
python3.10 -m expertLoadBalancing.run_pile_request_pool --output-dir <OUT>/part_a \
  --manifest <OUT>/manifest/workload_manifest.json --num-batches 14 --logical-batch-ids 0,1,2,3,4,5,6 \
  --logical-batch-size 16 --sources-per-batch 8 --requests-per-source 2 \
  --input-length 4096 --output-length 1024 --model-name Qwen/Qwen3-30B-A3B --cache-dir <HF_CACHE> \
  --trace-level routing_only --compose-after --tensor-parallel-size 2 \
  --gpu-memory-utilization 0.95 --max-model-len 8192
# part_b: same command with --output-dir <OUT>/part_b --logical-batch-ids 7,8,9,10,11,12,13
```

Files: `part_*/raw_trace.h5` and `part_*/requests.json` are unmodified collector outputs.
In `part_*/metadata.json` (`workload_manifest`) and `part_*/run_report.json` (`output_dir`) the
local absolute path was replaced with the archive-relative path; no other field was changed.
vLLM request ids restart at 0 in each part; the global request index is
`16 * logical_batch_id + batch_position` from `requests.json`.
