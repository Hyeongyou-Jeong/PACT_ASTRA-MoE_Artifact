#!/usr/bin/env bash
# Shared Python discovery for THER CPU scripts. Source this file; do not execute.
# Requires: bash, with set -euo pipefail already enabled by the caller.

find_python() {
  local cand
  for cand in python3.12 python3.11 python3.10 python3; do
    if command -v "$cand" >/dev/null 2>&1; then
      if "$cand" -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 10) else 1)'; then
        PYTHON="$cand"
        return 0
      fi
    fi
  done
  echo "Python >= 3.10 is required." >&2
  echo "Checked executables: python3.12, python3.11, python3.10, python3" >&2
  exit 1
}

require_python_packages() {
  local py="${1:?python executable}"
  "$py" - <<'PY'
import importlib
import sys

missing = []
for name in ("numpy", "h5py", "matplotlib"):
    try:
        importlib.import_module(name)
    except ImportError:
        missing.append(name)
if missing:
    print("Missing Python packages: " + ", ".join(missing), file=sys.stderr)
    print("Install with: pip install numpy h5py matplotlib", file=sys.stderr)
    raise SystemExit(1)
PY
}
