from __future__ import annotations

import argparse
import csv
from pathlib import Path

try:
    from .batch_sizing import resolve_system_batch_sizes
    from .config import SystemConfig, load_hardware_config, load_model_config, load_system_configs
    from .system_models import (
        SimulationResult,
        simulate_astra_moe,
        simulate_deepspeed_nvme,
        simulate_vllm_gpu_only,
    )
    from .trace_loader import load_or_generate_workload
except ImportError:  # Allows `python astrasim_energy/simulator/main.py`.
    from batch_sizing import resolve_system_batch_sizes
    from config import SystemConfig, load_hardware_config, load_model_config, load_system_configs
    from system_models import (
        SimulationResult,
        simulate_astra_moe,
        simulate_deepspeed_nvme,
        simulate_vllm_gpu_only,
    )
    from trace_loader import load_or_generate_workload


def _select_system(systems: list[SystemConfig], name: str) -> SystemConfig:
    for system in systems:
        if system.name == name:
            return system
    raise ValueError(f"Missing system config for '{name}'")


def write_energy_breakdown(results: list[SimulationResult], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["system", "batch_size", "component", "total_energy_j", "energy_per_token_j"],
        )
        writer.writeheader()
        for result in results:
            token_count = max(1, result.total_generated_tokens)
            for component_name, component in result.components.items():
                writer.writerow(
                    {
                        "system": result.system,
                        "batch_size": result.batch_size,
                        "component": component_name,
                        "total_energy_j": f"{component.energy_j:.12g}",
                        "energy_per_token_j": f"{component.energy_j / token_count:.12g}",
                    }
                )


def write_latency_breakdown(results: list[SimulationResult], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["system", "batch_size", "component", "total_latency_s", "avg_latency_per_step_s"],
        )
        writer.writeheader()
        for result in results:
            step_count = max(1, result.num_steps)
            for component_name, component in result.components.items():
                writer.writerow(
                    {
                        "system": result.system,
                        "batch_size": result.batch_size,
                        "component": component_name,
                        "total_latency_s": f"{component.latency_s:.12g}",
                        "avg_latency_per_step_s": f"{component.latency_s / step_count:.12g}",
                    }
                )


def write_summary(results: list[SimulationResult], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "system",
                "batch_size",
                "total_generated_tokens",
                "gpu_expert_count",
                "ssd_expert_count",
                "total_latency_s",
                "total_energy_j",
                "energy_per_token_j",
                "throughput_tokens_per_s",
            ],
        )
        writer.writeheader()
        for result in results:
            writer.writerow(
                {
                    "system": result.system,
                    "batch_size": result.batch_size,
                    "total_generated_tokens": result.total_generated_tokens,
                    "gpu_expert_count": result.gpu_expert_count,
                    "ssd_expert_count": result.ssd_expert_count,
                    "total_latency_s": f"{result.total_latency_s:.12g}",
                    "total_energy_j": f"{result.total_energy_j:.12g}",
                    "energy_per_token_j": f"{result.energy_per_token_j:.12g}",
                    "throughput_tokens_per_s": f"{result.throughput_tokens_per_s:.12g}",
                }
            )


def build_arg_parser() -> argparse.ArgumentParser:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="ASTRA-style energy evaluation simulator")
    parser.add_argument("--hardware-config", type=Path, default=root / "configs" / "hardware_energy.yaml")
    parser.add_argument("--model-config", type=Path, default=root / "configs" / "model_llama4_scout.yaml")
    parser.add_argument("--systems-config", type=Path, default=root / "configs" / "systems.yaml")
    parser.add_argument("--routed-log", type=Path, default=None)
    parser.add_argument("--log-pattern", default="*.jsonl")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--max-decode-steps", type=int, default=None)
    parser.add_argument(
        "--sequence-length",
        type=int,
        default=None,
        help="Override model YAML sequence_length for auto batch sizing and synthetic traces.",
    )
    parser.add_argument(
        "--synthetic-sequence-length",
        type=int,
        default=16,
        help="Synthetic decode steps to generate when --routed-log is not provided.",
    )
    parser.add_argument("--synthetic-total-samples", type=int, default=64)
    parser.add_argument("--output-dir", type=Path, default=root / "outputs")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    hardware = load_hardware_config(args.hardware_config)
    model = load_model_config(args.model_config)
    batch_sequence_length = args.sequence_length or model.sequence_length
    trace_sequence_length = args.max_decode_steps or args.synthetic_sequence_length or batch_sequence_length
    systems = resolve_system_batch_sizes(
        load_system_configs(args.systems_config),
        model=model,
        hardware=hardware,
        sequence_length=batch_sequence_length,
    )

    vllm_cfg = _select_system(systems, "vllm")
    deepspeed_cfg = _select_system(systems, "deepspeed_nvme")
    astra_cfg = _select_system(systems, "astra_moe")

    print(
        "[info] "
        f"batch_sequence_length={batch_sequence_length} "
        f"synthetic_sequence_length={trace_sequence_length} "
        f"batch_sizes={{vllm:{vllm_cfg.batch_size}, "
        f"deepspeed_nvme:{deepspeed_cfg.batch_size}, "
        f"astra_moe:{astra_cfg.batch_size}}}"
    )

    workloads = {
        system.name: load_or_generate_workload(
            log_path=args.routed_log,
            model=model,
            batch_size=int(system.batch_size),
            synthetic_sequence_length=trace_sequence_length,
            synthetic_total_samples=args.synthetic_total_samples,
            max_samples=args.max_samples,
            max_decode_steps=args.max_decode_steps,
            pattern=args.log_pattern,
        )
        for system in (vllm_cfg, deepspeed_cfg, astra_cfg)
    }

    results = [
        simulate_vllm_gpu_only(hardware, model, vllm_cfg, workloads["vllm"]),
        simulate_deepspeed_nvme(hardware, model, deepspeed_cfg, workloads["deepspeed_nvme"]),
        simulate_astra_moe(hardware, model, astra_cfg, workloads["astra_moe"]),
    ]

    write_energy_breakdown(results, args.output_dir / "energy_breakdown.csv")
    write_latency_breakdown(results, args.output_dir / "latency_breakdown.csv")
    write_summary(results, args.output_dir / "summary.csv")
    print(f"Wrote outputs to {args.output_dir}")


if __name__ == "__main__":
    main()
