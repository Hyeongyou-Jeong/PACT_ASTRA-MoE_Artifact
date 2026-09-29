"""Convert HDF5 routed-expert traces (raw_trace.h5, routing_only) into the energy simulator's --routed-log JSONL.

Usage (from the artifact root):
    python3 astrasim_energy/scripts/convert_routing_trace.py OUT_DIR TRACE_DIR [TRACE_DIR ...]
        [--expect-requests N] [--expect-decode-steps N] [--expect-layers N] [--expect-top-k N]

Each TRACE_DIR holds raw_trace.h5 and requests.json from the routing-trace collector (the same
trace-collection infrastructure used by the THER evaluation). Only decode-stage rows
(index/stage == 1) are converted, and routing is copied verbatim:

    sample_id      = request index (16 * logical_batch_id + batch_position from requests.json;
                     for a single-trace directory this equals index/request_id)
    decode_step    = index/iteration_id
    layer          = model/moe_layer_ids[j]
    routed_experts = routing/selected_expert_ids[row, j, :]   (stored order)

Extra fields (ignored by the simulator): logical_request_id, part, h5_row,
generation_iteration_id, token_position. One file per MoE layer: OUT_DIR/layer_XXX.jsonl.
The input is validated before anything is written; the script exits non-zero on malformed input.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import h5py
import numpy as np

DECODE_STAGE = 1


def fail(message: str) -> None:
    print(f"[convert_routing_trace] ERROR: {message}", file=sys.stderr)
    sys.exit(1)


def load_part(part: Path, logical_batch_size: int) -> dict:
    trace = part / "raw_trace.h5"
    requests_path = part / "requests.json"
    if not trace.is_file():
        fail(f"missing {trace}")
    if not requests_path.is_file():
        fail(f"missing {requests_path}")
    requests = json.loads(requests_path.read_text(encoding="utf-8"))
    h = h5py.File(trace, "r")

    stage = h["index/stage"][:]
    rows = np.nonzero(stage == DECODE_STAGE)[0]
    if rows.size == 0:
        fail(f"{trace}: no decode-stage rows")
    request_ids = [x.decode() if isinstance(x, bytes) else str(x) for x in h["index/request_id"][:][rows]]
    by_vllm_id = {str(r["request_id"]): r for r in requests}
    missing = sorted(set(request_ids) - set(by_vllm_id))
    if missing:
        fail(f"{trace}: request ids {missing[:5]} not in requests.json")
    sample = np.array(
        [logical_batch_size * int(by_vllm_id[x]["logical_batch_id"]) + int(by_vllm_id[x]["batch_position"])
         for x in request_ids]
    )
    return {
        "part": part,
        "requests": requests,
        "rows": rows,
        "request_ids": request_ids,
        "logical_ids": {str(r["request_id"]): r["logical_request_id"] for r in requests},
        "sample": sample,
        "iteration": h["index/iteration_id"][:][rows],
        "generation": h["index/generation_iteration_id"][:][rows],
        "position": h["index/token_position"][:][rows],
        "experts": h["routing/selected_expert_ids"][rows],
        "layer_ids": h["model/moe_layer_ids"][:],
        "num_experts": int(h.attrs.get("num_experts", 0)),
        "top_k_attr": int(h.attrs.get("top_k", 0)),
    }


def validate(parts: list[dict], args: argparse.Namespace) -> dict:
    layer_ids = parts[0]["layer_ids"]
    num_experts = parts[0]["num_experts"]
    all_requests = [r for p in parts for r in p["requests"]]
    logical = [r["logical_request_id"] for r in all_requests]
    if len(set(logical)) != len(logical):
        fail("duplicate logical_request_id across trace directories")
    documents = [r.get("document_content_sha256") for r in all_requests if r.get("document_content_sha256")]
    if len(set(documents)) != len(documents):
        fail("duplicate source document across requests")

    seen: set[tuple[int, int]] = set()
    steps_per_sample: dict[int, list[int]] = {}
    for p in parts:
        if not np.array_equal(p["layer_ids"], layer_ids):
            fail(f"{p['part']}: MoE layer ids differ between trace directories")
        experts = p["experts"]
        if experts.ndim != 3 or experts.shape[1] != len(layer_ids):
            fail(f"{p['part']}: selected_expert_ids shape {experts.shape} does not match {len(layer_ids)} layers")
        if p["top_k_attr"] and experts.shape[2] != p["top_k_attr"]:
            fail(f"{p['part']}: top_k attribute {p['top_k_attr']} != stored {experts.shape[2]}")
        if int(experts.min()) < 0 or (num_experts and int(experts.max()) >= num_experts):
            fail(f"{p['part']}: expert ids outside [0, {num_experts - 1}]")
        ordered = np.sort(experts, axis=2)
        if (ordered[:, :, 1:] == ordered[:, :, :-1]).any():
            fail(f"{p['part']}: repeated expert id within one token/layer")
        for s, t in zip(p["sample"].tolist(), p["iteration"].tolist()):
            if (s, t) in seen:
                fail(f"duplicate decode row for request {s}, step {t}")
            seen.add((s, t))
            steps_per_sample.setdefault(s, []).append(t)

    step_counts = set()
    for s, steps in steps_per_sample.items():
        steps.sort()
        if steps != list(range(len(steps))):
            fail(f"request {s}: decode steps are not contiguous from 0")
        step_counts.add(len(steps))

    summary = {
        "requests": len(steps_per_sample),
        "decode_steps_per_request": sorted(step_counts),
        "moe_layers": int(len(layer_ids)),
        "top_k": int(parts[0]["experts"].shape[2]),
        "num_experts": num_experts,
        "decode_tokens": int(sum(len(p["rows"]) for p in parts)),
        "routed_entries": int(sum(p["experts"].size for p in parts)),
    }
    expected = {
        "requests": args.expect_requests,
        "decode_steps_per_request": [args.expect_decode_steps] if args.expect_decode_steps else None,
        "moe_layers": args.expect_layers,
        "top_k": args.expect_top_k,
    }
    for key, value in expected.items():
        if value is not None and summary[key] != value:
            fail(f"{key} = {summary[key]}, expected {value}")
    if sorted(steps_per_sample) != list(range(len(steps_per_sample))):
        fail("request indices are not contiguous from 0")
    return summary


def write(parts: list[dict], out_dir: Path) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    for stale in out_dir.glob("layer_*.jsonl"):
        stale.unlink()
    layer_ids = parts[0]["layer_ids"]
    files = {int(layer): (out_dir / f"layer_{int(layer):03d}.jsonl").open("w", encoding="utf-8") for layer in layer_ids}
    written = 0
    try:
        for part_index, p in enumerate(parts):
            order = np.lexsort((p["iteration"], p["sample"]))
            for j, layer in enumerate(layer_ids):
                f = files[int(layer)]
                for k in order:
                    experts = [int(e) for e in p["experts"][k, j]]
                    written += len(experts)
                    f.write(json.dumps({
                        "sample_id": int(p["sample"][k]),
                        "decode_step": int(p["iteration"][k]),
                        "layer": int(layer),
                        "routed_experts": experts,
                        "logical_request_id": p["logical_ids"][p["request_ids"][k]],
                        "part": part_index,
                        "h5_row": int(p["rows"][k]),
                        "generation_iteration_id": int(p["generation"][k]),
                        "token_position": int(p["position"][k]),
                    }) + "\n")
    finally:
        for f in files.values():
            f.close()
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description="HDF5 routing trace (raw_trace.h5) -> astrasim_energy routed-log JSONL")
    parser.add_argument("out_dir", type=Path)
    parser.add_argument("trace_dirs", type=Path, nargs="+")
    parser.add_argument("--logical-batch-size", type=int, default=16)
    parser.add_argument("--expect-requests", type=int, default=None)
    parser.add_argument("--expect-decode-steps", type=int, default=None)
    parser.add_argument("--expect-layers", type=int, default=None)
    parser.add_argument("--expect-top-k", type=int, default=None)
    args = parser.parse_args()

    parts = [load_part(d, args.logical_batch_size) for d in args.trace_dirs]
    summary = validate(parts, args)
    print(f"[convert_routing_trace] validated: {json.dumps(summary)}")
    written = write(parts, args.out_dir)
    if written != summary["routed_entries"]:
        fail(f"wrote {written} routed entries, HDF5 has {summary['routed_entries']}")
    summary["written_routed_entries"] = written
    (args.out_dir / "conversion_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"[convert_routing_trace] wrote {len(parts[0]['layer_ids'])} layer files, {written} routed entries -> {args.out_dir}")


if __name__ == "__main__":
    main()
