#!/usr/bin/env python3
"""Build THER artifact workload manifests (mixed + per-domain).

Recovers the rebuttal pile-request-pool settings:
  Qwen3-30B-A3B, ISL=4096, OSL=1024, logical BS=16, seed=42,
  monology/pile-uncopyrighted @ 3be90335..., 8 Pile sources.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ther.trace_generation.pile_workload.builder import (
    FIXED_EIGHT_SOURCES,
    WorkloadConfig,
    build_workload_manifest,
)
from ther.trace_generation.pile_workload.sources import PILE_SOURCE_CATALOG


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = ROOT / "ther" / "workloads"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    p.add_argument("--model-name", default="Qwen/Qwen3-30B-A3B")
    p.add_argument("--input-length", type=int, default=4096)
    p.add_argument("--output-length", type=int, default=1024)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--crop-seed", type=int, default=42)
    p.add_argument("--max-scan-per-source", type=int, default=500_000)
    p.add_argument("--cache-dir", default="")
    p.add_argument(
        "--only",
        default="",
        help="Comma-separated: mixed,arxiv,...  Empty = all",
    )
    return p.parse_args()


def _write_summary(path: Path, manifest) -> None:
    payload = {
        "path": str(path),
        "num_requests": manifest.num_requests,
        "sources": list(manifest.config.source_keys),
        "input_length": manifest.config.input_length,
        "output_length": manifest.config.output_length,
        "batch_size": manifest.config.batch_size,
        "dataset_id": manifest.config.dataset_id,
        "dataset_revision": manifest.config.dataset_revision,
        "seed": manifest.config.seed,
        "crop_seed": manifest.config.crop_seed,
        "source_scan_stats": manifest.source_scan_stats,
    }
    path.with_suffix(".summary.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )


def main() -> None:
    args = parse_args()
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    only = {x.strip() for x in args.only.split(",") if x.strip()}

    jobs: list[tuple[str, WorkloadConfig]] = []
    mixed_cfg = WorkloadConfig(
        model_name=args.model_name,
        batch_size=16,
        input_length=args.input_length,
        output_length=args.output_length,
        num_batches=1,
        sources_per_batch=8,
        requests_per_source=2,
        source_keys=FIXED_EIGHT_SOURCES,
        seed=args.seed,
        crop_seed=args.crop_seed,
        max_scan_per_source=args.max_scan_per_source,
        tokenizer_cache_dir=args.cache_dir,
        selection_rationale=(
            "Rebuttal-style mixed logical batch: 8 Pile sources × 2 requests. "
            "Physical collection is BS=1; offline composition uses BS=16."
        ),
    )
    if not only or "mixed" in only:
        jobs.append(("mixed", mixed_cfg))

    for key in FIXED_EIGHT_SOURCES:
        if only and key not in only:
            continue
        # Domain-only logical batch: 16 requests from one source.
        # Deviation note: original rebuttal published mixed only; per-domain
        # batches are an artifact extension for domain sensitivity.
        cfg = WorkloadConfig(
            model_name=args.model_name,
            batch_size=16,
            input_length=args.input_length,
            output_length=args.output_length,
            num_batches=1,
            sources_per_batch=1,
            requests_per_source=16,
            source_keys=(key,),
            seed=args.seed + 1000 + FIXED_EIGHT_SOURCES.index(key),
            crop_seed=args.crop_seed + 1000 + FIXED_EIGHT_SOURCES.index(key),
            max_scan_per_source=args.max_scan_per_source,
            tokenizer_cache_dir=args.cache_dir,
            selection_rationale=(
                f"Artifact domain extension: 16 requests from {key} "
                f"({PILE_SOURCE_CATALOG[key].pile_set_name}). Not the original "
                "mixed rebuttal batch membership."
            ),
        )
        jobs.append((key, cfg))

    for name, cfg in jobs:
        path = out / f"workload_{name}.json"
        print(f"[build] {name} -> {path}", flush=True)
        manifest = build_workload_manifest(cfg)
        manifest.save(path)
        _write_summary(path, manifest)
        print(f"[build] {name} requests={manifest.num_requests}", flush=True)

    catalog = {
        "fixed_eight_sources": list(FIXED_EIGHT_SOURCES),
        "pile_set_names": {
            k: PILE_SOURCE_CATALOG[k].pile_set_name for k in FIXED_EIGHT_SOURCES
        },
        "note": (
            "mixed matches rebuttal 8×2 structure. Per-domain 1×16 batches are "
            "artifact extensions for domain comparison."
        ),
    }
    (out / "SOURCE_CATALOG.json").write_text(
        json.dumps(catalog, indent=2) + "\n", encoding="utf-8"
    )
    print(f"[build] done -> {out}")


if __name__ == "__main__":
    main()
