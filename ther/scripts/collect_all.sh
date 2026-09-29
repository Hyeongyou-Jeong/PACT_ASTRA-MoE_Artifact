#!/usr/bin/env bash
# Sequential trace collection: mixed (if missing) then each domain.
# Uses GPUs 0,1 by default (override with CUDA_VISIBLE_DEVICES). Safe to re-run; skips completed traces.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"

TRACE_ROOT="${TRACE_ROOT:-$ROOT/ther/traces/data}"
WORKDIR="$ROOT/ther/workloads"
LOGDIR="$ROOT/ther/logs"
mkdir -p "$TRACE_ROOT" "$LOGDIR"

WORKLOADS=(mixed arxiv pubmed_central github stackexchange wikipedia freelaw hackernews pile_cc)

collect_one() {
  local name="$1"
  local out="$TRACE_ROOT/$name"
  local manifest="$WORKDIR/workload_${name}.json"
  if [[ -f "$out/raw_trace.h5" && -f "$out/collect_report.json" ]]; then
    local rc
    rc="$(python3.10 -c "import json; print(json.load(open('$out/collect_report.json')).get('return_code',1))")"
    if [[ "$rc" == "0" ]]; then
      echo "[queue] skip $name (already collected)"
      return 0
    fi
  fi
  if [[ ! -f "$manifest" ]]; then
    echo "[queue] missing manifest $manifest" >&2
    return 1
  fi
  echo "[queue] collect $name -> $out"
  python3.10 ther/scripts/collect_one.py \
    --manifest "$manifest" \
    --output-dir "$out" \
    --cuda-visible-devices "$CUDA_VISIBLE_DEVICES" \
    | tee "$LOGDIR/collect_${name}.stdout"
}

for wl in "${WORKLOADS[@]}"; do
  collect_one "$wl"
done

echo "[queue] all collection attempts finished"
