"""Prepare a tiny real-row S2S overfit dataset from Stage-2 shards."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict

try:
    # Prefer package import when running from the repository root.
    from model.tokenizers import build_pom_tokenizer
    from train.s2s_sequence_builder import s2s_passes_read_write_ratio_filter
    from utils import shard_s2s_dataset
except ImportError:  # pragma: no cover
    # Add repo root so package imports resolve during direct script execution.
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from model.tokenizers import build_pom_tokenizer
    from train.s2s_sequence_builder import s2s_passes_read_write_ratio_filter
    from utils import shard_s2s_dataset


def _passes_ratio_filter(
    *,
    record: Dict[str, Any],
    tokenizer: Any,
    read_length: int,
    write_length: int,
) -> bool:
    """Return whether one Stage-2 row is feasible under strict Read/Write."""
    assistant_text = record.get("assistant_text")
    unit_ids = record.get("unit_ids")
    if not isinstance(assistant_text, str) or not assistant_text.strip():
        return False
    if not isinstance(unit_ids, list) or not unit_ids:
        return False

    # Match trainer behavior: plain assistant-text tokenization, no template.
    content_ids = tokenizer(assistant_text, add_special_tokens=False)["input_ids"]
    return s2s_passes_read_write_ratio_filter(
        content_token_count=len(content_ids),
        unit_token_count=len(unit_ids),
        read_length=read_length,
        write_length=write_length,
    )


def _collect_s2s_records(
    input_shards: list[Path],
    *,
    tokenizer: Any,
    max_samples: int,
    read_length: int,
    write_length: int,
) -> tuple[list[Dict[str, Any]], int, int]:
    """Collect the first N ratio-feasible real S2S pair rows across shards."""
    selected: list[Dict[str, Any]] = []
    dropped_ratio = 0
    scanned_rows = 0
    if max_samples <= 0:
        return selected, scanned_rows, dropped_ratio

    # Keep source order stable so overfit runs are reproducible.
    for input_shard in input_shards:
        for record in shard_s2s_dataset._iter_jsonl_records(input_shard):
            scanned_rows += 1
            if not _passes_ratio_filter(
                record=record,
                tokenizer=tokenizer,
                read_length=read_length,
                write_length=write_length,
            ):
                dropped_ratio += 1
                continue
            selected.append(record)
            if len(selected) >= max_samples:
                return selected, scanned_rows, dropped_ratio
    return selected, scanned_rows, dropped_ratio


def _compute_shard_sizes(total_records: int, num_shards: int) -> list[int]:
    """Compute contiguous shard sizes with deterministic remainder split."""
    base = total_records // num_shards
    remainder = total_records % num_shards
    return [base + (1 if idx < remainder else 0) for idx in range(num_shards)]


def _write_s2s_overfit_shards(
    *,
    selected_records: list[Dict[str, Any]],
    output_dir: Path,
    num_shards: int,
) -> tuple[list[Dict[str, Any]], int]:
    """Write S2S overfit shards from selected real Stage-2 rows."""
    shard_sizes = _compute_shard_sizes(len(selected_records), num_shards)
    manifest_shards: list[Dict[str, Any]] = []
    total_output_samples = 0
    input_cursor = 0

    # Split by input-record ranges so selection stays reproducible and easy to inspect.
    for shard_index, shard_size in enumerate(shard_sizes):
        shard_records = selected_records[input_cursor : input_cursor + shard_size]
        input_cursor += shard_size

        output_name = f"shard-{shard_index:05d}.jsonl"
        output_path = output_dir / output_name
        shard_start = total_output_samples

        with output_path.open("w", encoding="utf-8") as handle:
            for record in shard_records:
                # Keep the original Stage-2 row unchanged for trainer compatibility.
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")

        shard_output_samples = len(shard_records)
        total_output_samples += shard_output_samples
        manifest_shards.append(
            {
                "index": shard_index,
                "path": output_name,
                "input_records": shard_output_samples,
                "output_samples": shard_output_samples,
                "start_index_inclusive": shard_start,
                "end_index_exclusive": total_output_samples,
            }
        )

    return manifest_shards, total_output_samples


def _parse_args() -> argparse.Namespace:
    """Parse CLI flags for preparing S2S overfit shards."""
    parser = argparse.ArgumentParser(
        description="Build S2S overfit shards from real Stage-2 shard rows."
    )
    parser.add_argument("--input-dir", required=True, help="Path to Stage-2 shard directory")
    parser.add_argument("--output-dir", required=True, help="Path to S2S overfit output directory")
    parser.add_argument(
        "--max-samples",
        type=int,
        default=32,
        help="Number of Stage-2 pair rows to keep for overfit",
    )
    parser.add_argument("--read-length", type=int, default=3, help="Read chunk size R for ratio filtering")
    parser.add_argument(
        "--write-length",
        type=int,
        default=10,
        help="Write chunk size W for ratio filtering",
    )
    parser.add_argument(
        "--model-id",
        default="Qwen/Qwen3-0.6B",
        help="Tokenizer model id used for assistant-text token counts",
    )
    parser.add_argument(
        "--model-cache",
        default=None,
        help="Optional cache dir for tokenizer loading",
    )
    parser.add_argument("--num-shards", type=int, default=2, help="Number of S2S overfit shards")
    parser.add_argument("--force", action="store_true", help="Overwrite output directory if it exists")
    return parser.parse_args()


def main() -> None:
    """Prepare deterministic real-row S2S overfit shards and write a manifest."""
    args = _parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    max_samples = int(args.max_samples)
    num_shards = int(args.num_shards)
    read_length = int(args.read_length)
    write_length = int(args.write_length)
    if max_samples <= 0:
        raise ValueError("--max-samples must be > 0")
    if num_shards <= 0:
        raise ValueError("--num-shards must be > 0")
    if read_length <= 0:
        raise ValueError("--read-length must be > 0")
    if write_length <= 0:
        raise ValueError("--write-length must be > 0")

    input_shards = shard_s2s_dataset._list_shard_paths(input_dir)
    tokenizer = build_pom_tokenizer(
        base_model_id=str(args.model_id),
        cache_dir=args.model_cache,
    )
    selected_records, scanned_rows, dropped_ratio = _collect_s2s_records(
        input_shards,
        tokenizer=tokenizer,
        max_samples=max_samples,
        read_length=read_length,
        write_length=write_length,
    )
    if not selected_records:
        raise ValueError("no ratio-feasible input rows selected from Stage-2 shards")
    if num_shards > len(selected_records):
        raise ValueError("--num-shards cannot exceed selected row count")

    shard_s2s_dataset._ensure_clean_output_dir(output_dir, force=bool(args.force))
    manifest_shards, total_output_samples = _write_s2s_overfit_shards(
        selected_records=selected_records,
        output_dir=output_dir,
        num_shards=num_shards,
    )
    if total_output_samples <= 0:
        raise ValueError("conversion produced zero S2S samples")
    shard_s2s_dataset._validate_output_shards(output_dir, manifest_shards)

    # Keep schema metadata explicit so the Stage-2 dataset accepts the output.
    manifest = {
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "selected_input_rows": len(selected_records),
        "scanned_input_rows": scanned_rows,
        "dropped_by_ratio_filter": dropped_ratio,
        "num_output_shards": len(manifest_shards),
        "total_output_samples": total_output_samples,
        "schema": "s2s_pairs_v1",
        "read_length": read_length,
        "write_length": write_length,
        "tokenizer_model_id": str(args.model_id),
        "shards": manifest_shards,
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"input_dir: {input_dir}")
    print(f"output_dir: {output_dir}")
    print(f"read_length: {read_length}")
    print(f"write_length: {write_length}")
    print(f"scanned_input_rows: {scanned_rows}")
    print(f"dropped_by_ratio_filter: {dropped_ratio}")
    print(f"selected_input_rows: {len(selected_records)}")
    print(f"num_output_shards: {len(manifest_shards)}")
    print(f"total_output_samples: {total_output_samples}")
    print(f"manifest: {manifest_path}")


if __name__ == "__main__":
    main()
