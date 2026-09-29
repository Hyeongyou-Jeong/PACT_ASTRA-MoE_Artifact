#!/usr/bin/env bash
# CPU-only GPU expert-budget sweep on pre-collected routing traces.
# Reuses evaluate_policies.py; does not change policy semantics.
#
# Default budgets: B=4,8,12,16,20 (all << 128 experts in Qwen3-30B-A3B).
# Fixed: W=8, F=4, logical batch size=16.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
# shellcheck source=python_env.sh
source "$ROOT/ther/scripts/python_env.sh"
find_python
require_python_packages "$PYTHON"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
echo "[sweep] PYTHON=$PYTHON"

if [[ -z "${TRACE_ROOT:-}" ]]; then
  if [[ -d "$ROOT/ther/traces/data" ]]; then
    TRACE_ROOT="$ROOT/ther/traces/data"
  elif [[ -d "$ROOT/ther/precollected/traces" ]]; then
    TRACE_ROOT="$ROOT/ther/precollected/traces"
  else
    TRACE_ROOT="$ROOT/ther/traces/data"
  fi
fi
RES="${RESULTS_DIR:-$ROOT/ther/results/budget_sweep}"
BUDGETS="${BUDGETS:-4,8,12,16,20}"
WINDOW_SIZE="${WINDOW_SIZE:-8}"
SWAP_FREQUENCY="${SWAP_FREQUENCY:-4}"
LOGICAL_BATCH_SIZE="${LOGICAL_BATCH_SIZE:-16}"
mkdir -p "$RES"
echo "[sweep] TRACE_ROOT=$TRACE_ROOT"
echo "[sweep] RESULTS_DIR=$RES"
echo "[sweep] BUDGETS=$BUDGETS W=$WINDOW_SIZE F=$SWAP_FREQUENCY"

if ! compgen -G "$TRACE_ROOT/*/raw_trace.h5" > /dev/null; then
  cat >&2 <<EOF
No traces found under: $TRACE_ROOT

For the quick path, extract the pre-collected archive first:
  ./ther/scripts/setup_traces.sh
EOF
  exit 1
fi

IFS=',' read -r -a BUDGET_ARR <<< "$BUDGETS"
shopt -s nullglob
for B in "${BUDGET_ARR[@]}"; do
  B="$(echo "$B" | tr -d '[:space:]')"
  [[ -n "$B" ]] || continue
  echo "[sweep] gpu_budget=$B"
  for d in "$TRACE_ROOT"/*; do
    [[ -d "$d" ]] || continue
    [[ -f "$d/raw_trace.h5" ]] || continue
    name="$(basename "$d")"
    out="$RES/B${B}/$name"
    mkdir -p "$out"
    echo "[eval] B=$B $name"
    "$PYTHON" ther/scripts/evaluate_policies.py \
      --trace-dir "$d" \
      --output-dir "$out" \
      --workload-name "$name" \
      --gpu-budget "$B" \
      --window-size "$WINDOW_SIZE" \
      --swap-frequency "$SWAP_FREQUENCY" \
      --logical-batch-size "$LOGICAL_BATCH_SIZE"
  done
done

"$PYTHON" ther/scripts/plot_budget_sweep.py --results-dir "$RES" --output-dir "$RES"
echo "[sweep] done -> $RES"
