#!/usr/bin/env python3
"""Supplemental plots for THER sensitivity (does not touch primary figures)."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--supplemental-dir",
        type=Path,
        default=Path("ther/results/supplemental"),
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    root = Path(__file__).resolve().parents[2]
    d = args.supplemental_dir
    if not d.is_absolute():
        d = root / d
    gap = d / "ther_wf_gaps.csv"
    if not gap.is_file():
        raise SystemExit(f"missing {gap}")

    # Plot 1: Mixed THER coverage heatmap-ish lines vs W for each F
    by = defaultdict(list)
    with gap.open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            if row["workload"] != "mixed":
                continue
            by[int(row["swap_frequency"])].append(
                (int(row["window_size"]), float(row["ther"]), float(row["weighted_lfu"]), float(row["oracle"]))
            )
    fig, ax = plt.subplots(figsize=(7.5, 4.2), dpi=140)
    for f, pts in sorted(by.items()):
        pts = sorted(pts)
        ax.plot([w for w, _, _, _ in pts], [t for _, t, _, _ in pts], marker="o", label=f"THER F={f}")
    # reference Cumulative Weighted-LFU / Oracle at F=4
    if 4 in by:
        pts = sorted(by[4])
        ax.plot(
            [w for w, _, _, _ in pts],
            [x for _, _, x, _ in pts],
            linestyle="--",
            label="Cumulative Weighted-LFU F=4",
        )
        ax.plot([w for w, _, _, _ in pts], [x for _, _, _, x in pts], linestyle=":", label="Oracle F=4")
    ax.set_xlabel("THER window W")
    ax.set_ylabel("GPU routed-token coverage")
    ax.set_title("Supplemental: Mixed THER W/F sensitivity (B=12)")
    ax.set_ylim(0.0, 0.5)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(d / "mixed_ther_wf_sensitivity.png")
    fig.savefig(d / "mixed_ther_wf_sensitivity.pdf")
    plt.close(fig)

    # Plot 2: THER-WLFU at primary W=8,F=4 already in characterization;
    # here: THER-WLFU across W at F=4 for all domains (small multiples as grouped bars for W=8 only already exists)
    # Show mean gap by W at F=4
    gaps_by_w = defaultdict(list)
    with gap.open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            if int(row["swap_frequency"]) != 4:
                continue
            gaps_by_w[int(row["window_size"])].append(
                (row["workload"], float(row["ther_minus_weighted_lfu"]))
            )
    fig, ax = plt.subplots(figsize=(8.5, 4.2), dpi=140)
    workloads = [
        "mixed",
        "arxiv",
        "pubmed_central",
        "github",
        "stackexchange",
        "wikipedia",
        "freelaw",
        "hackernews",
        "pile_cc",
    ]
    import numpy as np

    xs = np.arange(len(workloads))
    width = 0.18
    for i, w in enumerate(sorted(gaps_by_w)):
        ys = []
        mp = {name: g for name, g in gaps_by_w[w]}
        for name in workloads:
            ys.append(mp.get(name, float("nan")))
        ax.bar(xs + (i - 1.5) * width, ys, width=width, label=f"W={w}")
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_xticks(xs)
    ax.set_xticklabels(workloads, rotation=35, ha="right", fontsize=8)
    ax.set_ylabel("THER − Cumulative Weighted-LFU")
    ax.set_title(
        "Supplemental: THER − Cumulative Weighted-LFU by domain (F=4, vary W)"
    )
    ax.legend(fontsize=8)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(d / "ther_minus_wlf_by_domain_W.png")
    fig.savefig(d / "ther_minus_wlf_by_domain_W.pdf")
    plt.close(fig)
    print(f"[plot] wrote under {d}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
