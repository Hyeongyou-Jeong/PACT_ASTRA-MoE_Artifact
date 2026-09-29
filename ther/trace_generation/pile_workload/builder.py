from __future__ import annotations

import hashlib
import json
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

from transformers import AutoTokenizer

from ther.evaluation.raw_trace.run_metadata import (
    deterministic_split,
    write_json_atomic,
)

from .sources import (
    DEFAULT_DATASET_ID,
    DEFAULT_DATASET_REVISION,
    DEFAULT_DATASET_SPLIT,
    PILE_SOURCE_CATALOG,
    PileSourceSpec,
)

FIXED_EIGHT_SOURCES: tuple[str, ...] = (
    "arxiv",
    "pubmed_central",
    "github",
    "stackexchange",
    "wikipedia",
    "freelaw",
    "hackernews",
    "pile_cc",
)


@dataclass(frozen=True)
class WorkloadConfig:
    model_name: str = "Qwen/Qwen3-30B-A3B"
    batch_size: int = 16
    input_length: int = 4096
    output_length: int = 4096
    num_batches: int = 4
    sources_per_batch: int = 8
    requests_per_source: int = 2
    source_keys: tuple[str, ...] = FIXED_EIGHT_SOURCES
    dataset_id: str = DEFAULT_DATASET_ID
    dataset_split: str = DEFAULT_DATASET_SPLIT
    dataset_revision: str = DEFAULT_DATASET_REVISION
    seed: int = 42
    crop_seed: int = 42
    split_seed: int = 42
    profile_train_ratio: float = 0.7
    validation_ratio: float = 0.1
    max_scan_per_source: int = 50_000
    tokenizer_cache_dir: str = ""
    hf_token: str = ""
    selection_rationale: str = (
        "Fixed offline logical composition (default 8 sources × 2 requests per "
        "logical batch). Domain mix is workload construction, not an "
        "experimental independent variable. Traces are collected at physical BS=1."
    )

    def __post_init__(self) -> None:
        if self.batch_size != self.sources_per_batch * self.requests_per_source:
            raise ValueError(
                "batch_size must equal sources_per_batch * requests_per_source"
            )
        if len(self.source_keys) != self.sources_per_batch:
            raise ValueError("source_keys length must equal sources_per_batch")
        unknown = [key for key in self.source_keys if key not in PILE_SOURCE_CATALOG]
        if unknown:
            raise KeyError(f"unknown source keys: {unknown}")
        if self.input_length <= 0 or self.output_length <= 0:
            raise ValueError("input_length and output_length must be positive")
        if self.num_batches <= 0:
            raise ValueError("num_batches must be positive")


@dataclass(frozen=True)
class WorkloadRequest:
    request_id: str
    logical_batch_id: int
    batch_position: int
    source_key: str
    pile_set_name: str
    category: str
    document_id: str
    document_content_sha256: str
    crop_start_token: int
    source_token_count: int
    input_token_ids: list[int]
    target_output_tokens: int
    split: str
    random_seed: int
    crop_seed: int

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class WorkloadManifest:
    config: WorkloadConfig
    requests: list[WorkloadRequest] = field(default_factory=list)
    source_scan_stats: dict[str, dict[str, int]] = field(default_factory=dict)

    @property
    def num_requests(self) -> int:
        return len(self.requests)

    def to_json(self) -> dict[str, Any]:
        return {
            "workload_name": "pile_mixed_long_context",
            "config": asdict(self.config),
            "fixed_mixed_batch_structure": (
                f"{self.config.sources_per_batch} sources × "
                f"{self.config.requests_per_source} requests = "
                f"batch_size {self.config.batch_size}"
            ),
            "selection_rationale": self.config.selection_rationale,
            "dataset_access_notes": {
                "primary_mirror": self.config.dataset_id,
                "dataset_revision": self.config.dataset_revision,
                "dataset_split": self.config.dataset_split,
                "eleutherai_pile_status": (
                    "EleutherAI/pile configs resolve but the-eye.eu TLS cert is "
                    "expired in this environment; mirror used instead"
                ),
                "openwebtext2": (
                    "absent from monology/pile-uncopyrighted; replaced by Pile-CC"
                ),
            },
            "generation_policy": {
                "min_tokens": self.config.output_length,
                "max_tokens": self.config.output_length,
                "ignore_eos": False,
                "semantics": (
                    "fixed-length decode workload: EOS/stop logits are suppressed "
                    "until min_tokens; not natural EOS completion"
                ),
            },
            "source_scan_stats": self.source_scan_stats,
            "requests": [request.to_json() for request in self.requests],
            "cost_estimate": estimate_workload_cost(self.config),
        }

    def save(self, path: str | Path) -> None:
        write_json_atomic(path, self.to_json())
        jsonl = Path(path).with_suffix(".jsonl")
        with jsonl.open("w", encoding="utf-8") as handle:
            for request in self.requests:
                handle.write(json.dumps(request.to_json(), ensure_ascii=False) + "\n")


def estimate_workload_cost(config: WorkloadConfig) -> dict[str, Any]:
    requests = config.num_batches * config.batch_size
    # Trace rows ≈ ISL + OSL - 1 per request (final prompt token is generation 0).
    traced_rows = requests * (config.input_length + config.output_length - 1)
    prefill_tokens = requests * config.input_length
    decode_tokens = requests * config.output_length
    total_tokens = prefill_tokens + decode_tokens
    # Rough uncompressed selected-ID/weight bound for routing_only:
    # 48 MoE layers × top-8 × (1B id + 4B weight) ≈ 1920 B/token-row.
    routing_only_bytes = traced_rows * 1920
    # fine_only: + emb float16 (4096) + probs float16 (48*128*2=12288)
    fine_only_bytes = traced_rows * (1920 + 4096 + 12_288)
    # Full adds logits float16 as well (+12288).
    full_bytes = traced_rows * (1920 + 4096 + 24_576)
    return {
        "num_batches": config.num_batches,
        "batch_size": config.batch_size,
        "total_requests": requests,
        "input_sequence_length": config.input_length,
        "output_sequence_length": config.output_length,
        "total_prefill_tokens": prefill_tokens,
        "total_decode_tokens": decode_tokens,
        "total_traced_token_rows": traced_rows,
        "approx_routing_only_raw_bytes": routing_only_bytes,
        "approx_fine_only_raw_bytes": fine_only_bytes,
        "approx_full_raw_bytes": full_bytes,
        "note": (
            "byte estimates are uncompressed tensor bounds before HDF5 "
            "compression; actual sizes are reported after tracing. "
            "traced_rows uses ISL+OSL-1 per request."
        ),
    }


def load_workload_manifest(path: str | Path) -> WorkloadManifest:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    config = WorkloadConfig(**payload["config"])
    requests = [WorkloadRequest(**row) for row in payload["requests"]]
    return WorkloadManifest(
        config=config,
        requests=requests,
        source_scan_stats=payload.get("source_scan_stats", {}),
    )


def build_workload_manifest(
    config: WorkloadConfig,
    *,
    tokenizer: AutoTokenizer | None = None,
    document_streams: dict[str, Iterable[dict[str, Any]]] | None = None,
) -> WorkloadManifest:
    """Build a deterministic mixed-source long-context workload.

    document_streams may inject synthetic/source-local rows for tests. Each row
    must provide text and an optional document_id.
    """

    tok = tokenizer or AutoTokenizer.from_pretrained(
        config.model_name,
        cache_dir=config.tokenizer_cache_dir,
        token=config.hf_token or None,
        trust_remote_code=True,
    )
    needed = {
        key: config.num_batches * config.requests_per_source
        for key in config.source_keys
    }
    pools: dict[str, list[dict[str, Any]]] = {key: [] for key in config.source_keys}
    scan_stats: dict[str, dict[str, int]] = {}
    streams = document_streams or {
        key: _iter_pile_source(config, PILE_SOURCE_CATALOG[key])
        for key in config.source_keys
    }
    for key in config.source_keys:
        pools[key], scan_stats[key] = _collect_source_pool(
            source_key=key,
            stream=streams[key],
            tokenizer=tok,
            needed=needed[key],
            input_length=config.input_length,
            crop_seed=config.crop_seed,
            max_scan=config.max_scan_per_source,
        )
        if len(pools[key]) < needed[key]:
            raise RuntimeError(
                f"source {key!r} produced only {len(pools[key])} eligible "
                f"documents; need {needed[key]} "
                f"(scanned={scan_stats[key]['scanned']})"
            )

    used_hashes: set[str] = set()
    requests: list[WorkloadRequest] = []
    for batch_id in range(config.num_batches):
        batch_split = deterministic_split(
            f"pile-logical-batch-{batch_id}",
            seed=config.split_seed,
            train_ratio=config.profile_train_ratio,
            validation_ratio=config.validation_ratio,
        )
        batch_position = 0
        for source_offset, source_key in enumerate(config.source_keys):
            for local_idx in range(config.requests_per_source):
                pool_index = (
                    batch_id * config.requests_per_source
                    + local_idx
                )
                doc = pools[source_key][pool_index]
                if doc["document_content_sha256"] in used_hashes:
                    raise AssertionError(
                        f"document hash reused across workload: "
                        f"{doc['document_id']}"
                    )
                used_hashes.add(doc["document_content_sha256"])
                request_id = (
                    f"pile-b{batch_id:04d}-p{batch_position:02d}-"
                    f"{source_key}-{local_idx}"
                )
                requests.append(
                    WorkloadRequest(
                        request_id=request_id,
                        logical_batch_id=batch_id,
                        batch_position=batch_position,
                        source_key=source_key,
                        pile_set_name=doc["pile_set_name"],
                        category=PILE_SOURCE_CATALOG[source_key].category,
                        document_id=doc["document_id"],
                        document_content_sha256=doc["document_content_sha256"],
                        crop_start_token=doc["crop_start_token"],
                        source_token_count=doc["source_token_count"],
                        input_token_ids=doc["input_token_ids"],
                        target_output_tokens=config.output_length,
                        split=batch_split,
                        random_seed=config.seed,
                        crop_seed=config.crop_seed,
                    )
                )
                batch_position += 1
        if batch_position != config.batch_size:
            raise AssertionError("batch construction lost requests")
    return WorkloadManifest(
        config=config,
        requests=requests,
        source_scan_stats=scan_stats,
    )


def _iter_pile_source(
    config: WorkloadConfig, source: PileSourceSpec
) -> Iterable[dict[str, Any]]:
    from datasets import load_dataset

    dataset = load_dataset(
        config.dataset_id,
        split=config.dataset_split,
        streaming=True,
        token=config.hf_token or None,
        revision=config.dataset_revision or None,
    )
    for row_index, row in enumerate(dataset):
        meta = row.get("meta") or {}
        if isinstance(meta, str):
            try:
                meta = json.loads(meta)
            except json.JSONDecodeError:
                meta = {}
        pile_set = str(meta.get("pile_set_name", ""))
        if pile_set != source.pile_set_name:
            continue
        text = row.get("text", "")
        if not isinstance(text, str) or not text.strip():
            continue
        document_id = str(
            meta.get("id")
            or meta.get("file")
            or f"{source.pile_set_name}:{row_index}"
        )
        yield {
            "text": text,
            "document_id": document_id,
            "pile_set_name": pile_set,
            "row_index": row_index,
        }


def _collect_source_pool(
    *,
    source_key: str,
    stream: Iterable[dict[str, Any]],
    tokenizer: AutoTokenizer,
    needed: int,
    input_length: int,
    crop_seed: int,
    max_scan: int,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    # Oversample then deterministically shuffle so later batches stay diverse.
    target_pool = max(needed * 3, needed)
    rng = random.Random(crop_seed + 17 * (hash(source_key) % 10_007))
    pool: list[dict[str, Any]] = []
    eligible = 0
    scanned = 0
    for row in stream:
        scanned += 1
        if max_scan > 0 and scanned > max_scan:
            break
        text = str(row["text"])
        token_ids = tokenizer.encode(
            text,
            add_special_tokens=False,
            truncation=False,
        )
        if len(token_ids) < input_length:
            continue
        eligible += 1
        content_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
        crop_rng = random.Random(
            f"{crop_seed}:{source_key}:{row['document_id']}:{content_hash}"
        )
        max_start = len(token_ids) - input_length
        crop_start = crop_rng.randint(0, max_start) if max_start > 0 else 0
        cropped = token_ids[crop_start : crop_start + input_length]
        if len(cropped) != input_length:
            continue
        item = {
            "document_id": str(row["document_id"]),
            "pile_set_name": str(row["pile_set_name"]),
            "document_content_sha256": content_hash,
            "crop_start_token": int(crop_start),
            "source_token_count": int(len(token_ids)),
            "input_token_ids": [int(value) for value in cropped],
        }
        if len(pool) < target_pool:
            pool.append(item)
        else:
            j = rng.randint(0, eligible - 1)
            if j < target_pool:
                pool[j] = item
        if len(pool) >= target_pool and eligible >= target_pool * 2:
            # Enough reservoir diversity for the requested workload size.
            break
    pool.sort(key=lambda item: (item["document_content_sha256"], item["document_id"]))
    order = list(range(len(pool)))
    random.Random(crop_seed + 101 * (hash(source_key) % 10_007)).shuffle(order)
    selected = [pool[index] for index in order[:needed]]
    return selected, {
        "scanned": scanned,
        "eligible": eligible,
        "pool": len(pool),
        "selected": len(selected),
    }


def prompts_for_batch(
    manifest: WorkloadManifest, logical_batch_id: int
) -> list[WorkloadRequest]:
    rows = [
        request
        for request in manifest.requests
        if request.logical_batch_id == logical_batch_id
    ]
    rows.sort(key=lambda item: item.batch_position)
    if len(rows) != manifest.config.batch_size:
        raise ValueError(
            f"logical batch {logical_batch_id} has {len(rows)} requests, "
            f"expected {manifest.config.batch_size}"
        )
    return rows
