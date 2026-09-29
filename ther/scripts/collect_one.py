#!/usr/bin/env python3
"""Collect routing_only traces for one THER workload manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--trace-level", default="routing_only")
    p.add_argument("--tensor-parallel-size", type=int, default=2)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.95)
    p.add_argument("--max-model-len", type=int, default=8192)
    p.add_argument("--cache-dir", default="")
    p.add_argument("--model-name", default="Qwen/Qwen3-30B-A3B")
    p.add_argument("--cuda-visible-devices", default="0,1")
    return p.parse_args()


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    args = parse_args()
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    manifest = args.manifest.resolve()
    if not manifest.is_file():
        raise SystemExit(f"missing manifest: {manifest}")

    env = os.environ.copy()
    env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    env["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices
    env["PYTHONPATH"] = str(ROOT) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")

    cmd = [
        sys.executable,
        "-m",
        "ther.trace_generation.run_pile_request_pool",
        "--output-dir",
        str(out),
        "--manifest",
        str(manifest),
        "--num-batches",
        "1",
        "--logical-batch-ids",
        "0",
        "--logical-batch-size",
        "16",
        "--input-length",
        "4096",
        "--output-length",
        "1024",
        "--model-name",
        args.model_name,
        "--cache-dir",
        args.cache_dir,
        "--trace-level",
        args.trace_level,
        "--compose-after",
        "--tensor-parallel-size",
        str(args.tensor_parallel_size),
        "--gpu-memory-utilization",
        str(args.gpu_memory_utilization),
        "--max-model-len",
        str(args.max_model_len),
    ]
    log_path = out / "collect.log"
    meta = {
        "command": cmd,
        "cwd": str(ROOT),
        "cuda_visible_devices": args.cuda_visible_devices,
        "manifest": str(manifest),
        "started_unix": time.time(),
    }
    (out / "collect_meta_start.json").write_text(json.dumps(meta, indent=2) + "\n")

    print(f"[collect] start {out}", flush=True)
    t0 = time.time()
    with log_path.open("w", encoding="utf-8") as log:
        log.write("command: " + " ".join(cmd) + "\n\n")
        log.flush()
        proc = subprocess.run(
            cmd,
            cwd=str(ROOT),
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )
    wall = time.time() - t0

    # validate
    val_cmd = [
        sys.executable,
        str(ROOT / "ther" / "trace_generation" / "validate_raw_trace.py"),
        str(out),
    ]
    val = subprocess.run(
        val_cmd,
        cwd=str(ROOT),
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    (out / "validation.json").write_text(val.stdout or val.stderr or "{}")

    h5 = out / "raw_trace.h5"
    report = {
        "output_dir": str(out),
        "return_code": proc.returncode,
        "wall_seconds": wall,
        "validation_return_code": val.returncode,
        "trace_exists": h5.is_file(),
        "trace_bytes": h5.stat().st_size if h5.is_file() else None,
        "trace_sha256": _sha256(h5) if h5.is_file() else None,
        "log_path": str(log_path),
        "finished_unix": time.time(),
    }
    (out / "collect_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)
    return 0 if proc.returncode == 0 and val.returncode == 0 and h5.is_file() else 1


if __name__ == "__main__":
    raise SystemExit(main())
