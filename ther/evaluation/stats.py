"""Spearman helper used by optional characterization scripts."""
from __future__ import annotations

import math
from typing import Iterable

import numpy as np

def _rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="stable")
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(order):
        stop = start + 1
        while stop < len(order) and values[order[stop]] == values[order[start]]:
            stop += 1
        rank = (start + stop - 1) / 2.0
        ranks[order[start:stop]] = rank
        start = stop
    return ranks


def spearman_rank_correlation(left: np.ndarray, right: np.ndarray) -> float:
    if left.shape != right.shape:
        raise ValueError("rank arrays must have equal shape")
    if left.size < 2:
        return 0.0
    left_rank = _rankdata(left.astype(np.float64))
    right_rank = _rankdata(right.astype(np.float64))
    if np.std(left_rank) == 0 or np.std(right_rank) == 0:
        return 0.0
    return float(np.corrcoef(left_rank, right_rank)[0, 1])



__all__ = ["spearman_rank_correlation"]
