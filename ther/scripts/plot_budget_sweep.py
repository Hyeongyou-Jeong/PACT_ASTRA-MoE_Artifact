#!/usr/bin/env python3
"""Aggregate and plot GPU expert-budget sweep coverage results."""

from __future__ import annotations

try:
    from ther.scripts.path_guard import apply as _apply_path_guard

    _apply_path_guard()
except Exception:
    pass

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


POLICIES = ("Static", "LRU", "LFU", "WeightedLFU", "THER", "Oracle")
DISPLAY = {
    "Static": "Static",
    "LRU": "LRU",
    "LFU": "LFU",
    "WeightedLFU": "Cumulative Weighted-LFU",
    "THER": "THER",
    "Oracle": "Oracle",
}
CSV_FIELDS = (
    "workload",
    "gpu_budget",
    "policy",
    "gpu_routed_token_coverage",
    "activation_coverage",
    "ssd_routed_tokens",
    "window_size",
    "swap_frequency",
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, default=None)
    return p.parse_args()


def load_rows(results_dir: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for agg in sorted(results_dir.glob("B*/*/aggregate_coverage.csv")):
        with agg.open(encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                rows.append(row)
    return rows


def write_combined(rows: list[dict[str, str]], path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(CSV_FIELDS), extrasaction="ignore")
        w.writeheader()
        for row in rows:
            w.writerow(row)


def plot_mixed(rows: list[dict[str, str]], out: Path) -> None:
    mixed = [r for r in rows if r["workload"] == "mixed"]
    if not mixed:
        print("skip mixed budget plot: no mixed rows")
        return
    series: dict[str, list[tuple[int, float]]] = defaultdict(list)
    for row in mixed:
        series[row["policy"]].append(
            (int(row["gpu_budget"]), float(row["gpu_routed_token_coverage"]))
        )
    fig, ax = plt.subplots(figsize=(7.5, 4.6), dpi=140)
    for name in POLICIES:
        pts = sorted(series.get(name, []))
        if not pts:
            continue
        ax.plot(
            [b for b, _ in pts],
            [y for _, y in pts],
            marker="o",
            linewidth=1.6,
            label=DISPLAY.get(name, name),
        )
    ax.set_xlabel("GPU expert budget")
    ax.set_ylabel("GPU routed-token coverage")
    ax.set_title(
        "Policy coverage vs GPU expert budget (Mixed)\n"
        "Policy-level routed-token coverage; not end-to-end throughput"
    )
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "mixed_budget_sweep.png")
    fig.savefig(out / "mixed_budget_sweep.pdf")
    plt.close(fig)


def plot_all_workloads(rows: list[dict[str, str]], out: Path) -> None:
    workloads = sorted({r["workload"] for r in rows})
    if not workloads:
        print("skip all-workload budget plot: no rows")
        return
    n = len(workloads)
    ncols = 3
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(11.5, 3.2 * nrows), dpi=140, sharey=True
    )
    axes_list = list(axes.ravel()) if n > 1 else [axes]
    by_wl: dict[str, dict[str, list[tuple[int, float]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for row in rows:
        by_wl[row["workload"]][row["policy"]].append(
            (int(row["gpu_budget"]), float(row["gpu_routed_token_coverage"]))
        )
    for i, wl in enumerate(workloads):
        ax = axes_list[i]
        for name in POLICIES:
            pts = sorted(by_wl[wl].get(name, []))
            if not pts:
                continue
            ax.plot(
                [b for b, _ in pts],
                [y for _, y in pts],
                marker="o",
                linewidth=1.3,
                label=DISPLAY.get(name, name) if i == 0 else None,
            )
        ax.set_title(wl)
        ax.set_ylim(0.0, 1.0)
        ax.grid(True, alpha=0.3)
        if i // ncols == nrows - 1:
            ax.set_xlabel("GPU expert budget")
        if i % ncols == 0:
            ax.set_ylabel("GPU routed-token coverage")
    for j in range(n, len(axes_list)):
        axes_list[j].set_visible(False)
    handles, labels = axes_list[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, fontsize=8)
    fig.suptitle(
        "GPU routed-token coverage vs expert budget (nine workloads)\n"
        "Policy-level trend only; not paper end-to-end throughput/latency"
    )
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    fig.savefig(out / "all_workloads_budget_sweep.png")
    fig.savefig(out / "all_workloads_budget_sweep.pdf")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    out = args.output_dir or args.results_dir
    out.mkdir(parents=True, exist_ok=True)
    rows = load_rows(args.results_dir)
    if not rows:
        raise SystemExit(f"no aggregate_coverage.csv under {args.results_dir}/B*/")
    write_combined(rows, out / "budget_sweep_coverage.csv")
    plot_mixed(rows, out)
    plot_all_workloads(rows, out)
    print(f"[plot] wrote under {out}")


if __name__ == "__main__":
    main()
