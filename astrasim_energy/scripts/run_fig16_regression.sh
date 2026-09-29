#!/usr/bin/env bash
# Optional synthetic regression test: rerun the built-in deterministic synthetic workload for
# the three Figure-16 models and compare against reference/fig16_regression_reference.csv.
# This is a simulator regression test, not the primary evaluation (see run_realtrace_energy.sh).
#
# Usage (from the artifact root):
#   bash astrasim_energy/scripts/run_fig16_regression.sh
#
# Set PYTHON=/path/to/python to choose the interpreter (Python >= 3.10 with PyYAML).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ARTIFACT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${ARTIFACT_ROOT}"

python_ok() {
    "$1" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)' 2>/dev/null
}

if [[ -n "${PYTHON:-}" ]]; then
    if ! python_ok "${PYTHON}"; then
        echo "error: PYTHON=${PYTHON} is not Python >= 3.10" >&2
        exit 1
    fi
else
    for candidate in python3 python3.13 python3.12 python3.11 python3.10 python; do
        if command -v "${candidate}" >/dev/null 2>&1 && python_ok "${candidate}"; then
            PYTHON="${candidate}"
            break
        fi
    done
    if [[ -z "${PYTHON:-}" ]]; then
        echo "error: Python >= 3.10 not found (set PYTHON=/path/to/python)" >&2
        exit 1
    fi
fi

if ! "${PYTHON}" -c 'import yaml' 2>/dev/null; then
    echo "error: PyYAML is missing; run: ${PYTHON} -m pip install -r astrasim_energy/requirements.txt" >&2
    exit 1
fi

export PYTHONDONTWRITEBYTECODE=1
echo "[fig16-regression] python: $(command -v "${PYTHON}") ($("${PYTHON}" -c 'import sys; print(sys.version.split()[0])'))"

CONFIGS=astrasim_energy/configs
OUT=astrasim_energy/outputs/fig16_regression
rm -rf "${OUT}"
mkdir -p "${OUT}"

run_model() {
    local model="$1"
    shift
    echo "[fig16-regression] ${model}: $*"
    "${PYTHON}" -m astrasim_energy.simulator.main \
        --model-config "${CONFIGS}/model_${model}.yaml" \
        "$@" \
        --output-dir "${OUT}/${model}"
}

start=${SECONDS}
# Llama4-Scout: shipped systems.yaml (auto batch sizing at the model's 64K sequence length).
run_model llama4_scout --systems-config "${CONFIGS}/systems.yaml"
# Qwen models: fixed Figure-16 batch sizes and ASTRA expert budget.
run_model qwen3_30b --systems-config "${CONFIGS}/systems_fig16_qwen.yaml"
run_model qwen3_235b --systems-config "${CONFIGS}/systems_fig16_qwen.yaml"
echo "[fig16-regression] simulation time: $((SECONDS - start)) s"
echo

"${PYTHON}" astrasim_energy/scripts/fig16_table.py --outputs-dir "${OUT}" \
    --reference astrasim_energy/reference/fig16_regression_reference.csv \
    --title "Synthetic regression (Figure-16 settings), J/token"
