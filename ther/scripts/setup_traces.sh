#!/usr/bin/env bash
# Extract pre-collected routing traces for the quick path (CPU).
# Usage (from artifact root):
#   ./ther/scripts/setup_traces.sh
#   ./ther/scripts/setup_traces.sh /path/to/ther_traces_qwen3_30b_a3b.tar.xz
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

TRACE_ARCHIVE_URL="https://zenodo.org/records/21888851/files/ther_traces_qwen3_30b_a3b.tar.xz?download=1"
DEFAULT_ARCHIVE="$ROOT/ther/precollected/ther_traces_qwen3_30b_a3b.tar.xz"
ARCHIVE="${1:-$DEFAULT_ARCHIVE}"
DEST="$ROOT/ther/traces/data"
SUMS="$ROOT/ther/precollected/SHA256SUMS"

if [[ ! -f "$ARCHIVE" ]]; then
  if [[ "$ARCHIVE" != "$DEFAULT_ARCHIVE" ]]; then
    echo "Missing trace archive: $ARCHIVE" >&2
    exit 1
  fi
  if ! command -v curl >/dev/null 2>&1; then
    echo "curl is required to download the trace archive, but was not found." >&2
    echo "Download manually from:" >&2
    echo "  $TRACE_ARCHIVE_URL" >&2
    echo "and place it at:" >&2
    echo "  ther/precollected/ther_traces_qwen3_30b_a3b.tar.xz" >&2
    exit 1
  fi
  mkdir -p "$(dirname "$ARCHIVE")"
  echo "[setup_traces] downloading $TRACE_ARCHIVE_URL"
  if ! curl -L --fail --progress-bar "$TRACE_ARCHIVE_URL" -o "$ARCHIVE"; then
    echo "Download failed: $TRACE_ARCHIVE_URL" >&2
    echo "Place the archive manually at:" >&2
    echo "  ther/precollected/ther_traces_qwen3_30b_a3b.tar.xz" >&2
    rm -f "$ARCHIVE"
    exit 1
  fi
fi

echo "[setup_traces] archive=$ARCHIVE"
if [[ -f "$SUMS" ]]; then
  # Strip CR so Windows-saved SHA256SUMS still matches on Linux.
  exp=$(tr -d '\r' < "$SUMS" | awk -v b="$(basename "$ARCHIVE")" '$2==b{print $1; exit}')
  if [[ -n "$exp" ]]; then
    got=$(sha256sum "$ARCHIVE" | awk '{print $1}')
    if [[ "$got" != "$exp" ]]; then
      echo "SHA256 mismatch for $ARCHIVE" >&2
      echo "expected $exp" >&2
      echo "got      $got" >&2
      exit 1
    fi
    echo "[setup_traces] sha256 ok"
  else
    echo "Could not find checksum for $(basename "$ARCHIVE") in $SUMS" >&2
    exit 1
  fi
fi

mkdir -p "$ROOT/ther/traces"
STAGE=$(mktemp -d)
trap 'rm -rf "$STAGE"' EXIT
tar -xJf "$ARCHIVE" -C "$STAGE"
# Expected layout inside archive: traces/<workload>/raw_trace.h5
if [[ -d "$STAGE/traces" ]]; then
  rm -rf "$DEST"
  mkdir -p "$ROOT/ther/traces"
  mv "$STAGE/traces" "$DEST"
elif [[ -f "$STAGE/mixed/raw_trace.h5" ]]; then
  rm -rf "$DEST"
  mkdir -p "$DEST"
  mv "$STAGE"/* "$DEST"/
else
  echo "unexpected archive layout" >&2
  find "$STAGE" -name 'raw_trace.h5' | head
  exit 1
fi

echo "[setup_traces] traces ready under $DEST"
ls "$DEST"
