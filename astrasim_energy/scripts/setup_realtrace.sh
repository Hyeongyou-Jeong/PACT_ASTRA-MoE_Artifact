#!/usr/bin/env bash
# Verify and extract the pre-collected Qwen3-30B-A3B routing trace (224 requests) for the
# real-trace energy path. CPU only.
#
# Usage (from the artifact root):
#   bash astrasim_energy/scripts/setup_realtrace.sh [ARCHIVE]
#
# Archive lookup order: ARCHIVE argument, $REALTRACE_ARCHIVE, then
# astrasim_energy/precollected/realtrace_qwen30_a3b_224req.tar.xz. If the default location is
# missing, the archive is downloaded there from Zenodo, falling back to the GitHub release mirror
# ($REALTRACE_URL overrides both).
# Extraction target: $REALTRACE_DIR (default astrasim_energy/traces/realtrace_qwen30).
# Set FORCE=1 to re-extract over an existing verified copy.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ARTIFACT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${ARTIFACT_ROOT}"

NAME=realtrace_qwen30_a3b_224req
PRECOLLECTED=astrasim_energy/precollected
SUMS="${PRECOLLECTED}/SHA256SUMS"
DEFAULT_ARCHIVE="${PRECOLLECTED}/${NAME}.tar.xz"
ARCHIVE="${1:-${REALTRACE_ARCHIVE:-${DEFAULT_ARCHIVE}}}"
DEST="${REALTRACE_DIR:-astrasim_energy/traces/realtrace_qwen30}"
DEFAULT_URL="https://zenodo.org/records/21889194/files/${NAME}.tar.xz?download=1"
MIRROR_URL="https://github.com/Hyeongyou-Jeong/PACT_ASTRA-MoE_Artifact/releases/download/v3/${NAME}.tar.xz"
if [[ -n "${REALTRACE_URL:-}" ]]; then URLS=("${REALTRACE_URL}"); else URLS=("${DEFAULT_URL}" "${MIRROR_URL}"); fi

[[ -f "${SUMS}" ]] || { echo "error: missing ${SUMS}" >&2; exit 1; }
EXPECTED="$(tr -d '\r' < "${SUMS}" | awk -v b="${NAME}.tar.xz" '$2==b{print $1; exit}')"
[[ -n "${EXPECTED}" ]] || { echo "error: no checksum for ${NAME}.tar.xz in ${SUMS}" >&2; exit 1; }

if [[ -f "${DEST}/.setup_ok" && "$(cat "${DEST}/.setup_ok")" == "${EXPECTED}" && "${FORCE:-0}" != "1" ]]; then
    echo "[setup_realtrace] already set up (archive sha256 ${EXPECTED}): ${DEST}"
    exit 0
fi

if [[ ! -f "${ARCHIVE}" ]]; then
    if [[ "${ARCHIVE}" == "${DEFAULT_ARCHIVE}" ]]; then
        command -v curl >/dev/null 2>&1 || { echo "error: curl is required to download the trace archive" >&2; exit 1; }
        for URL in "${URLS[@]}"; do
            echo "[setup_realtrace] downloading ${URL} (~1.74 GiB)"
            if curl -L --fail --progress-bar "${URL}" -o "${ARCHIVE}.part"; then
                mv "${ARCHIVE}.part" "${ARCHIVE}"
                break
            fi
            rm -f "${ARCHIVE}.part"
            echo "[setup_realtrace] download failed: ${URL}" >&2
        done
        [[ -f "${ARCHIVE}" ]] || { echo "error: download failed from all sources" >&2; exit 1; }
    else
        cat >&2 <<EOF
error: trace archive not found: ${ARCHIVE}
Provide it in one of these ways:
  bash astrasim_energy/scripts/setup_realtrace.sh /path/to/${NAME}.tar.xz
  REALTRACE_ARCHIVE=/path/to/${NAME}.tar.xz bash astrasim_energy/scripts/setup_realtrace.sh
  bash astrasim_energy/scripts/setup_realtrace.sh   # downloads from Zenodo (GitHub release fallback)
Expected size: 1,872,884,316 bytes; sha256 ${EXPECTED}
EOF
        exit 1
    fi
fi

echo "[setup_realtrace] archive: ${ARCHIVE}"
GOT="$(sha256sum "${ARCHIVE}" | awk '{print $1}')"
if [[ "${GOT}" != "${EXPECTED}" ]]; then
    echo "error: sha256 mismatch for ${ARCHIVE}" >&2
    echo "  expected ${EXPECTED}" >&2
    echo "  got      ${GOT}" >&2
    exit 1
fi
echo "[setup_realtrace] archive sha256 ok"

mkdir -p "$(dirname "${DEST}")"
STAGE="$(mktemp -d "$(dirname "${DEST}")/.setup_realtrace.XXXXXX")"
trap 'rm -rf "${STAGE}"' EXIT
echo "[setup_realtrace] extracting (~2.2 GB)"
tar -xJf "${ARCHIVE}" -C "${STAGE}"
[[ -f "${STAGE}/${NAME}/part_a/raw_trace.h5" && -f "${STAGE}/${NAME}/part_b/raw_trace.h5" ]] || {
    echo "error: unexpected archive layout" >&2; exit 1; }

echo "[setup_realtrace] verifying extracted files"
(cd "${STAGE}" && tr -d '\r' < "${ARTIFACT_ROOT}/${SUMS}" | awk -v n="${NAME}/" 'index($2, n) == 1' | sha256sum --quiet -c -)

rm -rf "${DEST}"
mv "${STAGE}/${NAME}" "${DEST}"
echo "${EXPECTED}" > "${DEST}/.setup_ok"
echo "[setup_realtrace] trace ready: ${DEST}"
ls "${DEST}"
