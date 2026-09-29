# Routing trace format (`routing_only`)

Used by THER coverage evaluation. Each collected run directory contains:

- `raw_trace.h5` — token index + selected expert ids/weights
- `metadata.json` — model / runtime / schema attributes
- `requests.json` — request ids, splits, lengths, token ids

Prefix caching must be disabled during collection. A prefix-cache hit skips
router execution, so there is no per-token expert row for that token.

## HDF5 layout (routing_only)

```text
/index/{request_id,batch_id,batch_position,stage,iteration_id,
        generation_iteration_id,token_position,token_id,forward_id}
/routing/selected_expert_ids
/routing/selected_expert_weights
/model/{transformer_layer_ids,moe_layer_ids}
```

`stage=0` is prefill, `stage=1` is decode. The final prompt token uses
`generation_iteration_id=0`.

Other levels (`full`, `fine_only`) exist in the writer for completeness but are
not used by the released THER traces.

## Capture

The MoE router hook is `vllm_plugin.py`. Collection entry point:

```bash
# via the public full path
./ther/scripts/run_full.sh
```

Validate:

```bash
python3.10 ther/trace_generation/validate_raw_trace.py <trace_dir>
```

The hook targets Qwen3 MoE under vLLM’s `FusedTopKRouter`, eager execution,
and TP-rank-0 writing.
