"""Split SU (Speech Understanding) JSON/JSONL metadata into deterministic JSONL shards."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any, Dict, Iterator


def _first_nonspace(path: Path) -> str:
    """Detect JSON array vs JSONL by first non-space character."""
    with path.open("r", encoding="utf-8") as handle:
        chunk = handle.read(4096)
    for char in chunk:
        if not char.isspace():
            return char
    return ""


def _iter_jsonl_records(path: Path) -> Iterator[Dict[str, Any]]:
    """Yield JSON object records from JSONL."""
    with path.open("r", encoding="utf-8") as handle:
        for lineno, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if not isinstance(record, dict):
                raise ValueError(f"line {lineno}: expected a JSON object record")
            yield record


def _iter_json_array_records(path: Path, chunk_size: int = 1 << 20) -> Iterator[Dict[str, Any]]:
    """Stream JSON object records from a top-level JSON array."""
    decoder = json.JSONDecoder()
    with path.open("r", encoding="utf-8") as handle:
        buffer = ""
        index = 0
        eof = False

        def _refill() -> bool:
            nonlocal buffer, index, eof
            chunk = handle.read(chunk_size)
            if not chunk:
                eof = True
                return False
            buffer = buffer[index:] + chunk
            index = 0
            return True

        def _skip_ws() -> bool:
            nonlocal index
            while True:
                while index < len(buffer) and buffer[index].isspace():
                    index += 1
                if index < len(buffer):
                    return True
                if not _refill():
                    return False

        if not _refill():
            return
        if not _skip_ws():
            return
        if buffer[index] != "[":
            raise ValueError("expected a top-level JSON array")
        index += 1

        while True:
            if not _skip_ws():
                raise ValueError("unexpected EOF inside JSON array")
            if buffer[index] == "]":
                return

            while True:
                try:
                    record, end = decoder.raw_decode(buffer, index)
                    index = end
                    break
                except json.JSONDecodeError:
                    if eof:
                        raise
                    if not _refill():
                        raise

            if not isinstance(record, dict):
                raise ValueError("expected each JSON array entry to be an object")
            yield record

            if not _skip_ws():
                raise ValueError("unexpected EOF after JSON object")
            if buffer[index] == ",":
                index += 1
                continue
            if buffer[index] == "]":
                return
            raise ValueError(f"expected ',' or ']' in JSON array, got {buffer[index]!r}")


def _iter_records(path: Path) -> Iterator[Dict[str, Any]]:
    """Yield JSON object records from JSON array or JSONL input."""
    if _first_nonspace(path) == "[":
        yield from _iter_json_array_records(path)
        return
    yield from _iter_jsonl_records(path)


def _count_records(path: Path) -> int:
    """Count records with a streaming pass."""
    return sum(1 for _ in _iter_records(path))


def _compute_shard_sizes(total_records: int, num_shards: int) -> list[int]:
    """Compute contiguous shard sizes with deterministic remainder split."""
    base = total_records // num_shards
    remainder = total_records % num_shards
    return [base + (1 if idx < remainder else 0) for idx in range(num_shards)]


def _ensure_output_dir(path: Path, *, force: bool) -> None:
    """Create a clean output directory."""
    if path.exists():
        if not force:
            raise FileExistsError(f"output already exists: {path} (use --force to overwrite)")
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def _shard_filename(index: int) -> str:
    """Build a stable shard filename."""
    return f"shard-{index:05d}.jsonl"


def _count_jsonl_lines(path: Path) -> int:
    """Count non-empty lines in a JSONL shard."""
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def _write_shards(
    *,
    input_path: Path,
    output_dir: Path,
    shard_sizes: list[int],
) -> list[Dict[str, Any]]:
    """Write deterministic contiguous JSONL shard files."""
    manifest_shards: list[Dict[str, Any]] = []

    shard_index = 0
    shard_count = len(shard_sizes)
    shard_size = shard_sizes[shard_index]
    shard_records = 0
    shard_start = 0

    shard_path = output_dir / _shard_filename(shard_index)
    shard_file = shard_path.open("w", encoding="utf-8")

    try:
        for global_index, record in enumerate(_iter_records(input_path)):
            # Rotate shard files exactly at planned boundaries.
            if shard_records == shard_size:
                shard_file.close()
                manifest_shards.append(
                    {
                        "index": shard_index,
                        "path": _shard_filename(shard_index),
                        "records": shard_records,
                        "start_index": shard_start,
                        "end_index": global_index - 1,
                    }
                )
                shard_index += 1
                if shard_index >= shard_count:
                    raise ValueError("encountered more records than planned")
                shard_size = shard_sizes[shard_index]
                shard_records = 0
                shard_start = global_index
                shard_path = output_dir / _shard_filename(shard_index)
                shard_file = shard_path.open("w", encoding="utf-8")

            shard_file.write(json.dumps(record, ensure_ascii=False) + "\n")
            shard_records += 1
    finally:
        shard_file.close()

    manifest_shards.append(
        {
            "index": shard_index,
            "path": _shard_filename(shard_index),
            "records": shard_records,
            "start_index": shard_start,
            "end_index": shard_start + shard_records - 1,
        }
    )
    return manifest_shards


def _validate_written_shards(output_dir: Path, manifest_shards: list[Dict[str, Any]]) -> None:
    """Validate shard file count and per-file record counts."""
    files = sorted(output_dir.glob("shard-*.jsonl"))
    if len(files) != len(manifest_shards):
        raise ValueError(
            f"expected {len(manifest_shards)} shard files, found {len(files)} in {output_dir}"
        )
    for shard in manifest_shards:
        shard_path = output_dir / shard["path"]
        line_count = _count_jsonl_lines(shard_path)
        if line_count != int(shard["records"]):
            raise ValueError(
                f"record count mismatch for {shard_path}: expected {shard['records']}, got {line_count}"
            )


def _parse_args() -> argparse.Namespace:
    """Parse CLI args."""
    parser = argparse.ArgumentParser(description="Split SU JSON/JSONL into JSONL shards.")
    parser.add_argument("--input", required=True, help="Input JSON or JSONL metadata path")
    parser.add_argument("--output-dir", required=True, help="Base output directory")
    parser.add_argument("--num-shards", type=int, default=256, help="Number of output shards")
    parser.add_argument("--force", action="store_true", help="Overwrite existing output")
    return parser.parse_args()


def main() -> None:
    """Count records, split deterministically, and write shard manifest."""
    args = _parse_args()
    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"input not found: {input_path}")

    num_shards = int(args.num_shards)
    if num_shards <= 0:
        raise ValueError("--num-shards must be > 0")

    total_records = _count_records(input_path)
    if total_records <= 0:
        raise ValueError("input has no records")
    if num_shards > total_records:
        raise ValueError("--num-shards cannot exceed total record count")

    shard_sizes = _compute_shard_sizes(total_records, num_shards)
    output_root = Path(args.output_dir)
    # Keep many shard files under a dedicated su subdirectory.
    output_dir = output_root if output_root.name == "su" else output_root / "su"
    _ensure_output_dir(output_dir, force=bool(args.force))

    manifest_shards = _write_shards(
        input_path=input_path,
        output_dir=output_dir,
        shard_sizes=shard_sizes,
    )

    if len(manifest_shards) != num_shards:
        raise ValueError(f"expected {num_shards} shard entries, got {len(manifest_shards)}")
    written_total = sum(int(shard["records"]) for shard in manifest_shards)
    if written_total != total_records:
        raise ValueError(f"expected {total_records} records, wrote {written_total}")
    _validate_written_shards(output_dir, manifest_shards)

    manifest = {
        "input_path": str(input_path),
        "output_dir": str(output_dir),
        "total_records": total_records,
        "num_shards": num_shards,
        "shards": manifest_shards,
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"input: {input_path}")
    print(f"output_dir: {output_dir}")
    print(f"total_records: {total_records}")
    print(f"num_shards: {num_shards}")
    print(f"manifest: {manifest_path}")


if __name__ == "__main__":
    main()
