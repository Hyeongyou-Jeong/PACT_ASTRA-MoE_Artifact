#!/usr/bin/env bash
# Full path: build workloads (if needed) -> collect all traces -> evaluate -> plot.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"

echo "[full] synthetic policy tests"
python3.10 ther/tests/test_policies_synthetic.py

echo "[full] build workloads"
python3.10 ther/scripts/build_workloads.py --output-dir ther/workloads

echo "[full] collect traces"
bash ther/scripts/collect_all.sh

echo "[full] evaluate + plot"
bash ther/scripts/run_quick.sh

echo "[full] done"
