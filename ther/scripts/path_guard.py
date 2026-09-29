"""Prefer in-tree packages over conflicting editable installs."""
from __future__ import annotations

import sys


def apply() -> None:
    sys.path[:] = [p for p in sys.path if "editable" not in str(p)]
    cleaned = []
    for finder in sys.meta_path:
        mod = getattr(type(finder), "__module__", "") or ""
        name = type(finder).__name__
        if "editable" in mod or "Editable" in name:
            continue
        cleaned.append(finder)
    sys.meta_path[:] = cleaned
