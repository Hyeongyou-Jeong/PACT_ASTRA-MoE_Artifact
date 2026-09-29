#!/usr/bin/env bash
# ASTRA-MoE Energy Evaluation: real-trace short path (recommended AE path, CPU only).
#
#   pre-collected Qwen3-30B-A3B routing trace (224 requests x 1023 decode steps)
#     -> convert_routing_trace.py (validated, verbatim conversion)
#     -> energy simulator (--routed-log), GPU Only B=14 / DeepSpeed-NVMe B=32 / ASTRA B=32
#     -> component energy + J/token, checked against reference/realtrace_qwen30_reference.csv
#
# Usage (from the artifact root, after setup_realtrace.sh):
#   bash astrasim_energy/scripts/run_realtrace_energy.sh
#
# Environment: PYTHON (interpreter), REALTRACE_DIR (extracted trace, default
# astrasim_energy/traces/realtrace_qwen30), REALTRACE_WORK_DIR (converted routed log,
# default <output>/routed_log), KEEP_ROUTED_LOG=1 to keep the ~2.5 GB converted log.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ARTIFACT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${ARTIFACT_ROOT}"

python_ok() {
    "$1" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)' 2>/dev/null
}

if [[ -n "${PYTHON:-}" ]]; then
    python_ok "${PYTHON}" || { echo "error: PYTHON=${PYTHON} is not Python >= 3.10" >&2; exit 1; }
else
    for candidate in python3 python3.13 python3.12 python3.11 python3.10 python; do
        if command -v "${candidate}" >/dev/null 2>&1 && python_ok "${candidate}"; then
            PYTHON="${candidate}"
            break
        fi
    done
    [[ -n "${PYTHON:-}" ]] || { echo "error: Python >= 3.10 not found (set PYTHON=/path/to/python)" >&2; exit 1; }
fi
if ! "${PYTHON}" -c 'import yaml, numpy, h5py' 2>/dev/null; then
    echo "error: missing packages; run: ${PYTHON} -m pip install -r astrasim_energy/requirements.txt" >&2
    exit 1
fi

TRACE="${REALTRACE_DIR:-astrasim_energy/traces/realtrace_qwen30}"
if [[ ! -f "${TRACE}/.setup_ok" || ! -f "${TRACE}/part_a/raw_trace.h5" || ! -f "${TRACE}/part_b/raw_trace.h5" ]]; then
    echo "error: verified trace not found at ${TRACE}; run: bash astrasim_energy/scripts/setup_realtrace.sh" >&2
    exit 1
fi

export PYTHONDONTWRITEBYTECODE=1
OUT=astrasim_energy/outputs/realtrace_qwen30
WORK="${REALTRACE_WORK_DIR:-${OUT}/routed_log}"
rm -rf "${OUT}"
mkdir -p "${OUT}"
echo "[realtrace] python: $(command -v "${PYTHON}") ($("${PYTHON}" -c 'import sys; print(sys.version.split()[0])'))"
echo "[realtrace] trace: ${TRACE}"

start=${SECONDS}
echo "[realtrace] validating and converting routing trace"
"${PYTHON}" astrasim_energy/scripts/convert_routing_trace.py "${WORK}" "${TRACE}/part_a" "${TRACE}/part_b" \
    --expect-requests 224 --expect-decode-steps 1023 --expect-layers 48 --expect-top-k 8
cp "${WORK}/conversion_summary.json" "${OUT}/conversion_summary.json"
convert_s=$((SECONDS - start))

echo "[realtrace] running energy simulator (GPU Only B=14, DeepSpeed-NVMe B=32, ASTRA B=32)"
"${PYTHON}" -m astrasim_energy.simulator.main \
    --model-config astrasim_energy/configs/model_qwen3_30b.yaml \
    --systems-config astrasim_energy/configs/systems_realtrace_qwen30.yaml \
    --routed-log "${WORK}" \
    --output-dir "${OUT}/qwen3_30b"
sim_s=$((SECONDS - start - convert_s))

if [[ "${KEEP_ROUTED_LOG:-0}" != "1" ]]; then
    rm -rf "${WORK}"
fi
echo "[realtrace] conversion ${convert_s} s, simulation ${sim_s} s"
echo

"${PYTHON}" astrasim_energy/scripts/fig16_table.py \
    --outputs-dir "${OUT}" \
    --models qwen3_30b \
    --reference astrasim_energy/reference/realtrace_qwen30_reference.csv \
    --out-name realtrace_energy.csv \
    --title "Real-trace energy, Qwen3-30B-A3B, 224 requests (J/token)"
