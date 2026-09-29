#!/usr/bin/env python3
"""Supplemental THER W/F (and optional B) sensitivity on existing traces.

Does not modify primary B=12,W=8,F=4 results under ther/results/<workload>/.
Writes only under ther/results/supplemental/.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

from ther.evaluation.coverage import _coverage
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
POLICIES_MIN = ("WeightedLFU", "THER", "Oracle")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--trace-root",
        type=Path,
        default=None,
        help="Directory containing <workload>/raw_trace.h5",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("ther/results/supplemental"),
    )
    p.add_argument("--logical-batch-size", type=int, default=16)
    p.add_argument("--ws", default="4,8,16,32")
    p.add_argument("--fs", default="2,4,8")
    p.add_argument("--bs", default="12", help="Comma list; primary grid uses 12 only")
    p.add_argument(
        "--also-b-sweep",
        action="store_true",
        help="Also sweep B at W=8,F=4 for bs in --bs-extra",
    )
    p.add_argument("--bs-extra", default="6,12,18,24")
    p.add_argument("--workloads", default=",".join(WORKLOADS))
    return p.parse_args()


def _ints(s: str) -> list[int]:
    return [int(x) for x in s.split(",") if x.strip()]


def eval_config(
    records,
    *,
    policies,
    gpu_budget: int,
    window_size: int,
    swap_frequency: int,
) -> list[dict]:
    ready_sets = []
    schedules = {}
    for name in policies:
        sched, ready = build_schedule(
            name,
            records,
            gpu_budget=gpu_budget,
            window_size=window_size,
            swap_frequency=swap_frequency,
        )
        schedules[name] = sched
        ready_sets.append(ready)
    compared_keys = set.intersection(*(set(r) for r in ready_sets))
    compared = [r for r in records if r.key in compared_keys]
    rows = []
    for name in policies:
        cov = _coverage(compared, {r.key: schedules[name][r.key] for r in compared})
        rows.append(
            {
                "policy": name,
                "gpu_routed_token_coverage": cov.gpu_routed_token_coverage,
                "activation_coverage": cov.activation_coverage,
                "ssd_routed_tokens": cov.ssd_routed_tokens,
                "n_events_compared": len(compared),
                "gpu_budget": gpu_budget,
                "window_size": window_size,
                "swap_frequency": swap_frequency,
            }
        )
    return rows


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

    workloads = [w for w in args.workloads.split(",") if w.strip()]
    ws, fs = _ints(args.ws), _ints(args.fs)
    rows_wf: list[dict] = []
    rows_b: list[dict] = []
    t0 = time.time()
    meta = {"configs": [], "workloads": workloads, "trace_root": str(trace_root)}

    for wl in workloads:
        tdir = trace_root / wl
        if not (tdir / "raw_trace.h5").is_file():
            raise SystemExit(f"missing trace: {tdir}")
        print(f"[sens] load {wl}", flush=True)
        records, num_experts, num_layers = load_records(
            tdir, logical_batch_size=args.logical_batch_size
        )
        # W/F grid at B=12 (or first of --bs)
        b_main = _ints(args.bs)[0]
        for f in fs:
            # WeightedLFU + Oracle once per F
            base = eval_config(
                records,
                policies=("WeightedLFU", "Oracle"),
                gpu_budget=b_main,
                window_size=8,  # unused by these policies
                swap_frequency=f,
            )
            for row in base:
                rows_wf.append({"workload": wl, "num_experts": num_experts, "num_layers": num_layers, **row})
            for w in ws:
                ther = eval_config(
                    records,
                    policies=("THER",),
                    gpu_budget=b_main,
                    window_size=w,
                    swap_frequency=f,
                )
                for row in ther:
                    rows_wf.append(
                        {"workload": wl, "num_experts": num_experts, "num_layers": num_layers, **row}
                    )
        if args.also_b_sweep:
            for b in _ints(args.bs_extra):
                for row in eval_config(
                    records,
                    policies=POLICIES_MIN,
                    gpu_budget=b,
                    window_size=8,
                    swap_frequency=4,
                ):
                    rows_b.append(
                        {
                            "workload": wl,
                            "num_experts": num_experts,
                            "num_layers": num_layers,
                            **row,
                        }
                    )
        print(f"[sens] done {wl}", flush=True)

    wf_path = out / "ther_wf_sensitivity.csv"
    with wf_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows_wf[0].keys()))
        w.writeheader()
        w.writerows(rows_wf)

    if rows_b:
        b_path = out / "ther_b_sensitivity.csv"
        with b_path.open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows_b[0].keys()))
            w.writeheader()
            w.writerows(rows_b)

    # Gaps at each config: THER - WeightedLFU, Oracle - THER
    gap_rows = []
    by = {}
    for r in rows_wf:
        key = (r["workload"], r["gpu_budget"], r["window_size"], r["swap_frequency"], r["policy"])
        by[key] = r["gpu_routed_token_coverage"]
    for wl in workloads:
        for f in fs:
            for w in ws:
                ther = by.get((wl, b_main, w, f, "THER"))
                # WLFU/Oracle stored with window_size=8 placeholder
                wlf = by.get((wl, b_main, 8, f, "WeightedLFU"))
                ora = by.get((wl, b_main, 8, f, "Oracle"))
                if ther is None or wlf is None or ora is None:
                    continue
                gap_rows.append(
                    {
                        "workload": wl,
                        "gpu_budget": b_main,
                        "window_size": w,
                        "swap_frequency": f,
                        "weighted_lfu": wlf,
                        "ther": ther,
                        "oracle": ora,
                        "ther_minus_weighted_lfu": ther - wlf,
                        "oracle_minus_ther": ora - ther,
                    }
                )
    gap_path = out / "ther_wf_gaps.csv"
    with gap_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(gap_rows[0].keys()))
        w.writeheader()
        w.writerows(gap_rows)

    wall = time.time() - t0
    meta.update(
        {
            "wall_seconds": wall,
            "wf_csv": str(wf_path),
            "gap_csv": str(gap_path),
            "b_csv": str(out / "ther_b_sensitivity.csv") if rows_b else None,
            "note": "Supplemental only; primary artifact remains B=12,W=8,F=4",
        }
    )
    (out / "sensitivity_meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(json.dumps(meta, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
