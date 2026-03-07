"""Prepare a tiny real-row TTS overfit dataset from SU shards."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict

try:
    # Prefer package import when running from the repository root.
    from utils import shard_tts_dataset
except ImportError:  # pragma: no cover
    # Add repo root so package imports resolve during direct script execution.
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from utils import shard_tts_dataset


def _collect_su_records(input_shards: list[Path], *, max_input_records: int) -> list[Dict[str, Any]]:
    """Collect the first N real SU conversation rows across shards."""
    selected: list[Dict[str, Any]] = []
    if max_input_records <= 0:
        return selected

    # Keep source order stable so overfit runs are reproducible.
    for input_shard in input_shards:
        for record in shard_tts_dataset._iter_jsonl_records(input_shard):
            selected.append(record)
            if len(selected) >= max_input_records:
                return selected
    return selected


def _compute_shard_sizes(total_records: int, num_shards: int) -> list[int]:
    """Compute contiguous shard sizes with deterministic remainder split."""
    base = total_records // num_shards
    remainder = total_records % num_shards
    return [base + (1 if idx < remainder else 0) for idx in range(num_shards)]


def _write_tts_overfit_shards(
    *,
    selected_records: list[Dict[str, Any]],
    output_dir: Path,
    num_shards: int,
) -> tuple[list[Dict[str, Any]], int]:
    """Write TTS overfit shards from selected real SU rows."""
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
        shard_output_samples = 0
        shard_start = total_output_samples

        with output_path.open("w", encoding="utf-8") as handle:
            for record_index, record in enumerate(shard_records):
                # Reuse TTS canonical conversion/validation in one place.
                for sample in shard_tts_dataset._iter_tts_samples(
                    record,
                    shard_index=shard_index,
                    record_index=record_index,
                ):
                    handle.write(json.dumps(sample, ensure_ascii=False) + "\n")
                    shard_output_samples += 1

        total_output_samples += shard_output_samples
        manifest_shards.append(
            {
                "index": shard_index,
                "path": output_name,
                "input_records": len(shard_records),
                "output_samples": shard_output_samples,
                "start_index_inclusive": shard_start,
                "end_index_exclusive": total_output_samples,
            }
        )

    return manifest_shards, total_output_samples


def _parse_args() -> argparse.Namespace:
    """Parse CLI flags for preparing TTS overfit shards."""
    parser = argparse.ArgumentParser(
        description="Build TTS overfit shards from real SU shard rows."
    )
    parser.add_argument("--input-dir", required=True, help="Path to SU shard directory")
    parser.add_argument("--output-dir", required=True, help="Path to TTS overfit output directory")
    parser.add_argument(
        "--max-input-records",
        type=int,
        default=32,
        help="Number of SU conversation rows to keep for overfit",
    )
    parser.add_argument("--num-shards", type=int, default=2, help="Number of TTS overfit shards")
    parser.add_argument("--force", action="store_true", help="Overwrite output directory if it exists")
    return parser.parse_args()


def main() -> None:
    """Prepare deterministic input rows for TTS overfit conversion."""
    args = _parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    max_input_records = int(args.max_input_records)
    num_shards = int(args.num_shards)
    if max_input_records <= 0:
        raise ValueError("--max-input-records must be > 0")
    if num_shards <= 0:
        raise ValueError("--num-shards must be > 0")

    input_shards = shard_tts_dataset._list_shard_paths(input_dir)
    selected_records = _collect_su_records(
        input_shards,
        max_input_records=max_input_records,
    )
    if not selected_records:
        raise ValueError("no input records selected from SU shards")
    if num_shards > len(selected_records):
        raise ValueError("--num-shards cannot exceed selected SU record count")

    shard_tts_dataset._ensure_clean_output_dir(output_dir, force=bool(args.force))
    manifest_shards, total_output_samples = _write_tts_overfit_shards(
        selected_records=selected_records,
        output_dir=output_dir,
        num_shards=num_shards,
    )
    if total_output_samples <= 0:
        raise ValueError("conversion produced zero TTS samples")
    # Reuse TTS canonical output-shard validation in one place.
    shard_tts_dataset._validate_output_shards(output_dir, manifest_shards)

    # Keep boundary/stop semantics explicit in a TTS-compatible manifest.
    manifest = {
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "selected_input_records": len(selected_records),
        "num_output_shards": len(manifest_shards),
        "total_output_samples": total_output_samples,
        "boundary_token_policy": "add_sep_in_stage1b_data_pipeline",
        "stop_token_policy": "add_eos_in_stage1b_data_pipeline",
        "shards": manifest_shards,
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"input_dir: {input_dir}")
    print(f"output_dir: {output_dir}")
    print(f"selected_input_records: {len(selected_records)}")
    print(f"num_output_shards: {len(manifest_shards)}")
    print(f"total_output_samples: {total_output_samples}")
    print(f"manifest: {manifest_path}")


if __name__ == "__main__":
    main()
