#!/usr/bin/env bash
# Evaluate all available traces and regenerate plots (CPU only).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
# shellcheck source=python_env.sh
source "$ROOT/ther/scripts/python_env.sh"
find_python
require_python_packages "$PYTHON"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
echo "[quick] PYTHON=$PYTHON"

# Prefer portable relative locations; TRACE_ROOT overrides all.
if [[ -z "${TRACE_ROOT:-}" ]]; then
  if [[ -d "$ROOT/ther/traces/data" ]]; then
    TRACE_ROOT="$ROOT/ther/traces/data"
  elif [[ -d "$ROOT/ther/precollected/traces" ]]; then
    TRACE_ROOT="$ROOT/ther/precollected/traces"
  else
    TRACE_ROOT="$ROOT/ther/traces/data"
  fi
fi
RES="${RESULTS_DIR:-$ROOT/ther/results/primary}"
mkdir -p "$RES"
echo "[quick] TRACE_ROOT=$TRACE_ROOT"
echo "[quick] RESULTS_DIR=$RES"

# Fail early with a clear message if traces are missing.
if ! compgen -G "$TRACE_ROOT/*/raw_trace.h5" > /dev/null; then
  cat >&2 <<EOF
No traces found under: $TRACE_ROOT

For the quick path, extract the pre-collected archive first:
  ./ther/scripts/setup_traces.sh
EOF
  exit 1
fi

shopt -s nullglob
for d in "$TRACE_ROOT"/*; do
  [[ -d "$d" ]] || continue
  [[ -f "$d/raw_trace.h5" ]] || continue
  name="$(basename "$d")"
  out="$RES/$name"
  mkdir -p "$out"
  echo "[eval] $name"
  "$PYTHON" ther/scripts/evaluate_policies.py \
    --trace-dir "$d" \
    --output-dir "$out" \
    --workload-name "$name"
done

"$PYTHON" ther/scripts/plot_results.py --results-dir "$RES" --output-dir "$RES"
echo "[eval] done -> $RES"
