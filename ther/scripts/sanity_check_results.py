#!/usr/bin/env python3
"""Flag suspicious policy-result anomalies across primary (+ optional supplemental)."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

PRIMARY_WLS = (
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
ORDER = ("Static", "LRU", "LFU", "WeightedLFU", "THER", "Oracle")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results-dir", type=Path, default=Path("ther/results"))
    p.add_argument(
        "--output",
        type=Path,
        default=Path("ther/results/supplemental/sanity_flags.json"),
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    root = Path(__file__).resolve().parents[2]
    res = args.results_dir
    if not res.is_absolute():
        res = root / res
    out = args.output
    if not out.is_absolute():
        out = root / out
    out.parent.mkdir(parents=True, exist_ok=True)

    flags = []
    coverages = {}
    for wl in PRIMARY_WLS:
        path = res / wl / "aggregate_coverage.csv"
        if not path.is_file():
            flags.append({"severity": "missing_domain", "workload": wl})
            continue
        rows = list(csv.DictReader(path.open()))
        by = {r["policy"]: float(r["gpu_routed_token_coverage"]) for r in rows}
        coverages[wl] = by
        for pol, v in by.items():
            if not (0.0 <= v <= 1.0):
                flags.append(
                    {"severity": "coverage_out_of_range", "workload": wl, "policy": pol, "value": v}
                )
        # Oracle below online
        ora = by.get("Oracle")
        if ora is not None:
            for pol in ORDER:
                if pol == "Oracle":
                    continue
                if pol in by and by[pol] > ora + 1e-12:
                    flags.append(
                        {
                            "severity": "oracle_below_online",
                            "workload": wl,
                            "policy": pol,
                            "oracle": ora,
                            "online": by[pol],
                        }
                    )
        # Identical curves across distinct policies (exact float match)
        vals = [(p, by[p]) for p in ORDER if p in by]
        for i in range(len(vals)):
            for j in range(i + 1, len(vals)):
                if vals[i][1] == vals[j][1] and vals[i][0] != vals[j][0]:
                    flags.append(
                        {
                            "severity": "identical_policy_coverage",
                            "workload": wl,
                            "policies": [vals[i][0], vals[j][0]],
                            "value": vals[i][1],
                        }
                    )

    # Cross-workload exact identical full vectors (possible state leak)
    items = list(coverages.items())
    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            wa, ca = items[i]
            wb, cb = items[j]
            if ca.keys() == cb.keys() and all(ca[k] == cb[k] for k in ca):
                flags.append(
                    {
                        "severity": "identical_workload_vectors",
                        "workloads": [wa, wb],
                    }
                )

    # Dual-run determinism if reference present
    ref = res / "reference_primary"
    if ref.is_dir():
        for wl in PRIMARY_WLS:
            a = res / wl / "aggregate_coverage.csv"
            b = ref / wl / "aggregate_coverage.csv"
            if not a.is_file() or not b.is_file():
                continue
            def load(p):
                return {
                    r["policy"]: r["gpu_routed_token_coverage"]
                    for r in csv.DictReader(p.open())
                }
            if load(a) != load(b):
                # primary may have been regenerated; compare numerically
                la, lb = load(a), load(b)
                for pol in set(la) | set(lb):
                    if abs(float(la.get(pol, "nan")) - float(lb.get(pol, "nan"))) > 1e-12:
                        flags.append(
                            {
                                "severity": "differs_from_reference_primary",
                                "workload": wl,
                                "policy": pol,
                                "current": la.get(pol),
                                "reference": lb.get(pol),
                            }
                        )

    report = {"n_flags": len(flags), "flags": flags, "coverages": coverages}
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"n_flags": len(flags), "flags": flags}, indent=2))
    return 1 if flags else 0


if __name__ == "__main__":
    raise SystemExit(main())
