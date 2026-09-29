#!/usr/bin/env python3
"""Domain routing characterization on existing traces (B=12, W=8).

Definitions (aligned with rebuttal analysis code where possible):

Window
  Non-overlapping blocks of W consecutive decode layer-events per MoE layer
  (same event stream as DiffMoETraceSource logical_manifest iteration).
  Primary characterization uses W=8.

routing_skew_gini
  Gini coefficient of per-expert routed-token totals aggregated over all
  compared decode events in the workload (0=uniform, 1=one expert).

consecutive_window_spearman
  Mean Spearman rank correlation of routed-token count vectors between
  consecutive non-overlapping W-windows, averaged over layers and window pairs.

consecutive_topB_jaccard
  Mean Jaccard similarity of Top-B expert sets (by routed-token count within
  each window) between consecutive windows; B=12.

topB_routed_token_coverage
  Mean, over windows, of (routed tokens assigned to the window's Top-B experts)
  / (all routed tokens in the window). Equivalent to load_topB_coverage in
  analyze_correlation_mechanism.py.

Policy coverages (Cumulative Weighted-LFU / THER / Oracle) use the same schedule builders
and warmup rule as the primary evaluate_policies path (B=12, W=8, F=4).
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from ther.evaluation.coverage import _coverage
from ther.evaluation.stats import spearman_rank_correlation
from ther.policies.residency import build_schedule
from ther.scripts.evaluate_policies import load_records

WORKLOADS = (
    "mixed",
    "arxiv",
    "pubmed_central",
    "github",
    "stackexchange",
    "wikipedia",
    "freelaw",
    "hackernews",
    "pile_cc",
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--trace-root", type=Path, default=None)
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("ther/results/characterization"),
    )
    p.add_argument("--logical-batch-size", type=int, default=16)
    p.add_argument("--gpu-budget", type=int, default=12)
    p.add_argument("--window-size", type=int, default=8)
    p.add_argument("--swap-frequency", type=int, default=4)
    p.add_argument("--workloads", default=",".join(WORKLOADS))
    return p.parse_args()


def gini(counts: np.ndarray) -> float:
    x = np.sort(counts.astype(np.float64))
    if x.size == 0 or x.sum() <= 0:
        return float("nan")
    n = x.size
    idx = np.arange(1, n + 1, dtype=np.float64)
    return float((2.0 * (idx * x).sum() - (n + 1) * x.sum()) / (n * x.sum()))


def top_b_ids(counts: np.ndarray, b: int) -> list[int]:
    b = max(0, min(int(b), len(counts)))
    return sorted(
        range(len(counts)),
        key=lambda e: (-int(counts[e]), e),
    )[:b]


def jaccard(a: set[int], b: set[int]) -> float:
    if not a and not b:
        return 1.0
    u = a | b
    return len(a & b) / len(u) if u else 1.0


def characterize_workload(
    records,
    *,
    num_experts: int,
    num_layers: int,
    budget: int,
    window: int,
    swap_frequency: int,
) -> dict:
    # Policy coverages with primary warmup rule
    policies = ("WeightedLFU", "THER", "Oracle")
    ready_sets = []
    schedules = {}
    for name in policies:
        sched, ready = build_schedule(
            name,
            records,
            gpu_budget=budget,
            window_size=window,
            swap_frequency=swap_frequency,
        )
        schedules[name] = sched
        ready_sets.append(ready)
    compared_keys = set.intersection(*(set(r) for r in ready_sets))
    compared = [r for r in records if r.key in compared_keys]
    cov = {
        name: _coverage(compared, {r.key: schedules[name][r.key] for r in compared})
        for name in policies
    }

    # Aggregate skew over compared events
    total_routed = np.zeros(num_experts, dtype=np.float64)
    by_layer_step: dict[int, dict[int, list]] = defaultdict(lambda: defaultdict(list))
    for record in compared:
        layer = record.key.moe_layer_index
        step = record.key.generation_iteration_id
        by_layer_step[layer][step].append(record)
        for expert, count in record.routed_token_counts:
            total_routed[int(expert)] += int(count)

    spearman_vals: list[float] = []
    jaccard_vals: list[float] = []
    topb_cov_vals: list[float] = []

    for layer in range(num_layers):
        steps = sorted(by_layer_step[layer])
        # Build per-step routed vectors, then non-overlapping windows of `window` steps
        step_vecs = []
        for step in steps:
            vec = np.zeros(num_experts, dtype=np.float64)
            for record in by_layer_step[layer][step]:
                for expert, count in record.routed_token_counts:
                    vec[int(expert)] += int(count)
            step_vecs.append(vec)
        # Non-overlapping windows along decode-step order
        win_vecs = []
        for start in range(0, len(step_vecs) - window + 1, window):
            chunk = np.sum(step_vecs[start : start + window], axis=0)
            win_vecs.append(chunk)
            top = top_b_ids(chunk, budget)
            hit = float(sum(chunk[e] for e in top))
            tot = float(chunk.sum())
            if tot > 0:
                topb_cov_vals.append(hit / tot)
        for i in range(len(win_vecs) - 1):
            left, right = win_vecs[i], win_vecs[i + 1]
            if left.sum() <= 0 or right.sum() <= 0:
                continue
            if np.std(_rank_proxy(left)) == 0 or np.std(_rank_proxy(right)) == 0:
                # still call spearman helper (returns 0 on zero std)
                pass
            spearman_vals.append(spearman_rank_correlation(left, right))
            jaccard_vals.append(
                jaccard(set(top_b_ids(left, budget)), set(top_b_ids(right, budget)))
            )

    wlf = cov["WeightedLFU"].gpu_routed_token_coverage
    ther = cov["THER"].gpu_routed_token_coverage
    ora = cov["Oracle"].gpu_routed_token_coverage
    return {
        "gpu_budget": budget,
        "window_size": window,
        "swap_frequency": swap_frequency,
        "n_events_compared": len(compared),
        "routing_skew_gini": gini(total_routed),
        "topB_share_aggregate": float(
            sum(sorted(total_routed, reverse=True)[:budget]) / total_routed.sum()
        )
        if total_routed.sum() > 0
        else float("nan"),
        "consecutive_window_spearman_mean": float(np.mean(spearman_vals))
        if spearman_vals
        else float("nan"),
        "consecutive_window_spearman_n": len(spearman_vals),
        "consecutive_topB_jaccard_mean": float(np.mean(jaccard_vals))
        if jaccard_vals
        else float("nan"),
        "consecutive_topB_jaccard_n": len(jaccard_vals),
        "topB_routed_token_coverage_mean": float(np.mean(topb_cov_vals))
        if topb_cov_vals
        else float("nan"),
        "weighted_lfu_coverage": wlf,
        "ther_coverage": ther,
        "oracle_coverage": ora,
        "ther_minus_weighted_lfu": ther - wlf,
        "oracle_minus_ther": ora - ther,
    }


def _rank_proxy(x: np.ndarray) -> np.ndarray:
    return x.astype(np.float64)


def main() -> int:
    args = parse_args()
    root = Path(__file__).resolve().parents[2]
    trace_root = args.trace_root
    if trace_root is None:
        cand = root / "ther" / "traces" / "data"
        trace_root = cand if cand.exists() else (root / "ther" / "traces" / "data")
    out = args.output_dir
    if not out.is_absolute():
        out = root / out
    out.mkdir(parents=True, exist_ok=True)

    rows = []
    for wl in [w for w in args.workloads.split(",") if w.strip()]:
        tdir = trace_root / wl
        print(f"[char] {wl}", flush=True)
        records, n_exp, n_lay = load_records(
            tdir, logical_batch_size=args.logical_batch_size
        )
        row = characterize_workload(
            records,
            num_experts=n_exp,
            num_layers=n_lay,
            budget=args.gpu_budget,
            window=args.window_size,
            swap_frequency=args.swap_frequency,
        )
        row = {"workload": wl, "num_experts": n_exp, "num_layers": n_lay, **row}
        rows.append(row)
        print(
            f"  gini={row['routing_skew_gini']:.4f} "
            f"spear={row['consecutive_window_spearman_mean']:.4f} "
            f"jacc={row['consecutive_topB_jaccard_mean']:.4f} "
            f"THER-WLFU={row['ther_minus_weighted_lfu']:.4f}",
            flush=True,
        )

    csv_path = out / "domain_characterization.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # Concise gap bar plot
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(9, 4.2), dpi=140)
        xs = np.arange(len(rows))
        gaps = [r["ther_minus_weighted_lfu"] for r in rows]
        names = [r["workload"] for r in rows]
        ax.bar(xs, gaps, color="#4C4C4C")
        ax.axhline(0.0, color="black", linewidth=0.8)
        ax.set_xticks(xs)
        ax.set_xticklabels(names, rotation=35, ha="right", fontsize=8)
        ax.set_ylabel("THER − Cumulative Weighted-LFU (coverage)")
        ax.set_title(
            "THER vs Cumulative Weighted-LFU gap by domain (B=12, W=8, F=4)"
        )
        ax.grid(True, axis="y", alpha=0.3)
        fig.tight_layout()
        fig.savefig(out / "ther_minus_weighted_lfu_by_domain.png")
        fig.savefig(out / "ther_minus_weighted_lfu_by_domain.pdf")
        plt.close(fig)
    except Exception as exc:  # noqa: BLE001
        print(f"[char] plot skipped: {exc}")

    (out / "DEFINITIONS.md").write_text(__doc__ or "")
    (out / "characterization_meta.json").write_text(
        json.dumps(
            {
                "csv": str(csv_path),
                "trace_root": str(trace_root),
                "gpu_budget": args.gpu_budget,
                "window_size": args.window_size,
                "swap_frequency": args.swap_frequency,
            },
            indent=2,
        )
        + "\n"
    )
    print(f"[char] wrote {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
