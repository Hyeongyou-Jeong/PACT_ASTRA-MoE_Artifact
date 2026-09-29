"""Build the Figure-16-style energy table from simulator outputs and check it against a reference.

Usage (from the artifact root):
    python3 astrasim_energy/scripts/fig16_table.py [--outputs-dir DIR] [--models KEYS]
        [--reference CSV] [--out-name NAME] [--no-check]

Reads <outputs-dir>/<model>/energy_breakdown.csv and summary.csv for each model,
folds simulator components into the five Figure-16 categories, and writes
<outputs-dir>/<out-name>. Exits non-zero if any reference check fails.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]

MODELS = [
    ("llama4_scout", "Llama4-Scout"),
    ("qwen3_30b", "Qwen3-30B-A3B"),
    ("qwen3_235b", "Qwen3-235B-A22B"),
]
SYSTEMS = [
    ("vllm", "GPU Only"),
    ("deepspeed_nvme", "DeepSpeed-NVMe"),
    ("astra_moe", "ASTRA"),
]

# Figure-16 categories. Components not listed here (gpu_static, gpu_comm,
# gpu_moe_stall, ssd_spm, ssd_controller) are not part of the displayed total.
CATEGORIES = {
    "GPU-SM": ["gpu_compute", "gpu_shared", "gpu_l1", "gpu_l2"],
    "GPU-HBM": ["gpu_dram"],
    "PCIe": ["pcie"],
    "SSD-Flash": ["ssd_flash_internal_read", "ssd_flash_interface"],
    "SSD-EPU": ["ssd_epu", "ssd_dram"],
}
COLUMNS = list(CATEGORIES) + ["Total"]


def load_model_results(model_dir: Path) -> dict[str, dict]:
    per_token: dict[str, dict[str, float]] = {}
    with (model_dir / "energy_breakdown.csv").open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            per_token.setdefault(row["system"], {})[row["component"]] = float(row["energy_per_token_j"])
    batch_sizes: dict[str, str] = {}
    with (model_dir / "summary.csv").open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            batch_sizes[row["system"]] = row["batch_size"]

    results = {}
    for system, components in per_token.items():
        values = {
            category: sum(components.get(name, 0.0) for name in names)
            for category, names in CATEGORIES.items()
        }
        values["Total"] = sum(values[category] for category in CATEGORIES)
        results[system] = {"batch_size": batch_sizes[system], "values": values}
    return results


def load_reference(path: Path) -> dict[tuple[str, str, str], float]:
    reference = {}
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            reference[(row["model"], row["system"], row["category"])] = float(row["value"])
    return reference


def within_tolerance(value: float, expected: float, tolerance_pct: float) -> bool:
    if expected == 0.0:
        return abs(value) <= 1e-15
    return abs(value - expected) / abs(expected) * 100.0 <= tolerance_pct


def main() -> int:
    parser = argparse.ArgumentParser(description="Build and check the Figure-16-style energy table")
    parser.add_argument("--outputs-dir", type=Path, default=PACKAGE_ROOT / "outputs" / "fig16_regression")
    parser.add_argument("--models", default=",".join(key for key, _ in MODELS),
                        help="Comma-separated model keys (subdirectories of --outputs-dir)")
    parser.add_argument("--reference", type=Path,
                        default=PACKAGE_ROOT / "reference" / "fig16_regression_reference.csv")
    parser.add_argument("--out-name", default="fig16_energy.csv")
    parser.add_argument("--title", default="Figure-16 energy (J/token)")
    parser.add_argument("--tolerance-pct", type=float, default=0.01)
    parser.add_argument("--no-check", action="store_true", help="Skip the reference comparison")
    args = parser.parse_args()

    names = dict(MODELS)
    selected = [key.strip() for key in args.models.split(",") if key.strip()]
    unknown = [key for key in selected if key not in names]
    if unknown:
        parser.error(f"unknown model keys: {unknown}")

    table = []
    reductions = []
    for model_key, model_name in ((key, names[key]) for key in selected):
        results = load_model_results(args.outputs_dir / model_key)
        gpu_total = results["vllm"]["values"]["Total"]
        ds_total = results["deepspeed_nvme"]["values"]["Total"]
        astra_total = results["astra_moe"]["values"]["Total"]
        reductions.append((model_name, 1 - astra_total / ds_total, 1 - astra_total / gpu_total))
        for system_key, system_name in SYSTEMS:
            entry = results[system_key]
            table.append(
                {
                    "model": model_name,
                    "system": system_name,
                    "batch_size": entry["batch_size"],
                    "values": entry["values"],
                    "vs_deepspeed": entry["values"]["Total"] / ds_total,
                    "vs_gpu_only": entry["values"]["Total"] / gpu_total,
                }
            )

    out_path = args.outputs_dir / args.out_name
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["model", "system", "batch_size", *COLUMNS,
             "Normalized_Total_vs_DeepSpeed", "Normalized_Total_vs_GPUOnly"]
        )
        for row in table:
            writer.writerow(
                [row["model"], row["system"], row["batch_size"],
                 *(f"{row['values'][c]:.10g}" for c in COLUMNS),
                 f"{row['vs_deepspeed']:.6f}", f"{row['vs_gpu_only']:.6f}"]
            )

    reference = {} if args.no_check else load_reference(args.reference)
    failures = 0
    checked = 0
    header = (f"{'Model':16s} {'System':15s} {'Batch':>5s} "
              + " ".join(f"{c:>10s}" for c in COLUMNS)
              + f" {'vs DS':>7s} {'vs GPU':>7s}  Check")
    print(args.title)
    print(header)
    print("-" * len(header))
    for row in table:
        status = "-"
        if reference:
            row_ok = True
            for column in COLUMNS:
                key = (row["model"], row["system"], column)
                if key not in reference:
                    row_ok = False
                    print(f"[FAIL] missing reference value: {key}", file=sys.stderr)
                    continue
                checked += 1
                if not within_tolerance(row["values"][column], reference[key], args.tolerance_pct):
                    row_ok = False
                    print(
                        f"[FAIL] {key}: got {row['values'][column]:.10g}, expected {reference[key]:.10g}",
                        file=sys.stderr,
                    )
            status = "PASS" if row_ok else "FAIL"
            failures += 0 if row_ok else 1
        print(
            f"{row['model']:16s} {row['system']:15s} {row['batch_size']:>5s} "
            + " ".join(f"{row['values'][c]:10.6g}" for c in COLUMNS)
            + f" {row['vs_deepspeed']:7.3f} {row['vs_gpu_only']:7.3f}  {status}"
        )

    print()
    for model_name, vs_ds, vs_gpu in reductions:
        print(f"{model_name}: ASTRA energy/token {-100 * vs_ds:+.1f}% vs DeepSpeed-NVMe, "
              f"{-100 * vs_gpu:+.1f}% vs GPU Only")
    print(f"\nWrote {out_path}")
    if reference:
        verdict = "PASS" if failures == 0 else "FAIL"
        print(f"Reference check: {verdict} ({checked} values, tolerance {args.tolerance_pct}% relative)")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
