# vLLM routing instrumentation

Used only by the optional full path (`./ther/scripts/run_full.sh`).

## Environment (as used for the released traces)

- Python 3.10
- vLLM 0.17.1
- Transformers 4.57.6 (recorded in trace attributes)
- Two GPUs for TP=2 (A100-class or equivalent)

## Model

- `Qwen/Qwen3-30B-A3B`
- HF revision `ad44e777bcd18fa416d9da3bd8f70d33ebb85d39`

## Capture path

```text
ther/scripts/collect_one.py
  -> ther.trace_generation.run_pile_request_pool
  -> vLLM (enable_return_routed_experts)
  -> ther/evaluation/raw_trace/vllm_plugin.py   # MoE router hook
  -> HDF5 writer (routing_only)
```

Run from the artifact root with `PYTHONPATH` set to that root (as `run_full.sh`
already does).

## Trace level

`routing_only` stores selected expert ids/weights only. That is enough to
recompute GPU routed-token coverage for all six policies.

## Validation

```bash
python3.10 ther/trace_generation/validate_raw_trace.py <trace_dir>
python3.10 ther/scripts/ther_sanity_check.py <trace_dir>
```
