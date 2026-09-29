#!/usr/bin/env python3
"""Plot Mixed temporal coverage and domain aggregate comparison."""

from __future__ import annotations

from ther.scripts.path_guard import apply as _apply_path_guard
_apply_path_guard()

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


POLICIES = ("Static", "LRU", "LFU", "WeightedLFU", "THER", "Oracle")
# CSV key WeightedLFU = Cumulative Weighted-LFU (no finite W).
DISPLAY = {
    "Static": "Static",
    "LRU": "LRU",
    "LFU": "LFU",
    "WeightedLFU": "Cumulative Weighted-LFU",
    "THER": "THER",
    "Oracle": "Oracle",
}
PAPER_STYLE = ("Static", "LRU", "LFU", "THER", "Oracle")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, default=None)
    return p.parse_args()


def plot_temporal(results_dir: Path, out: Path) -> None:
    path = results_dir / "mixed" / "temporal_coverage.csv"
    if not path.is_file():
        print(f"skip temporal: missing {path}")
        return
    series: dict[str, list[tuple[int, float]]] = defaultdict(list)
    with path.open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            series[row["policy"]].append(
                (int(row["decode_step"]), float(row["gpu_routed_token_coverage"]))
            )
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=140)
    for name in PAPER_STYLE:
        if name not in series:
            continue
        xs = [x for x, _ in series[name]]
        ys = [y for _, y in series[name]]
        ax.plot(xs, ys, label=DISPLAY.get(name, name), linewidth=1.6)
    if "WeightedLFU" in series:
        xs = [x for x, _ in series["WeightedLFU"]]
        ys = [y for _, y in series["WeightedLFU"]]
        ax.plot(
            xs,
            ys,
            label=DISPLAY["WeightedLFU"],
            linewidth=1.4,
            linestyle="--",
        )
    ax.set_xlabel("Decode step")
    ax.set_ylabel("GPU routed-token coverage")
    ax.set_title(
        "THER analysis (Qwen3-30B-A3B, Mixed Pile)\n"
        "Not an exact Fig. 8(b) Qwen3-235B reproduction"
    )
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "mixed_temporal_coverage.png")
    fig.savefig(out / "mixed_temporal_coverage.pdf")
    plt.close(fig)


SKIP_RESULT_DIRS = {
    "supplemental",
    "characterization",
    "reference_primary",
    "reference_results",
    "_smoke_existing_mixed",
    "primary",  # when scanning a parent that contains primary/
}


def plot_domain(results_dir: Path, out: Path) -> None:
    rows = []
    for agg in sorted(results_dir.glob("*/aggregate_coverage.csv")):
        if agg.parent.name in SKIP_RESULT_DIRS:
            continue
        with agg.open(encoding="utf-8", newline="") as f:
            rows.extend(csv.DictReader(f))
    if not rows:
        print("skip domain plot: no aggregate CSVs")
        return
    # write combined table
    combined = out / "all_workloads_aggregate_coverage.csv"
    with combined.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    workloads = sorted({r["workload"] for r in rows})
    fig, ax = plt.subplots(figsize=(10, 4.8), dpi=140)
    x = list(range(len(workloads)))
    width = 0.12
    for i, policy in enumerate(POLICIES):
        ys = []
        for wl in workloads:
            match = [
                float(r["gpu_routed_token_coverage"])
                for r in rows
                if r["workload"] == wl and r["policy"] == policy
            ]
            ys.append(match[0] if match else float("nan"))
        ax.bar(
            [xi + (i - len(POLICIES) / 2) * width + width / 2 for xi in x],
            ys,
            width=width,
            label=DISPLAY.get(policy, policy),
        )
    ax.set_xticks(x)
    ax.set_xticklabels(workloads, rotation=30, ha="right")
    ax.set_ylabel("GPU routed-token coverage")
    ax.set_ylim(0.0, 1.0)
    ax.set_title("Domain comparison (Qwen3-30B-A3B, B=12, W=8, F=4)")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(fontsize=7, ncol=3)
    fig.tight_layout()
    fig.savefig(out / "domain_coverage_comparison.png")
    fig.savefig(out / "domain_coverage_comparison.pdf")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    out = args.output_dir or args.results_dir
    out.mkdir(parents=True, exist_ok=True)
    plot_temporal(args.results_dir, out)
    plot_domain(args.results_dir, out)
    print(f"[plot] wrote under {out}")


if __name__ == "__main__":
    main()
