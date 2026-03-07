"""Create a tiny SU dataset slice for overfitting tests."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Iterable, List, Dict, Any


def _first_nonspace(path: Path) -> str:
    """Detect JSON list vs JSONL by the first non-space character."""
    with path.open("r", encoding="utf-8") as handle:
        chunk = handle.read(4096)
    for ch in chunk:
        if not ch.isspace():
            return ch
    return ""


def _iter_records(path: Path) -> Iterable[Dict[str, Any]]:
    """Yield JSON records from a list file or JSONL stream."""
    first = _first_nonspace(path)
    if first == "[":
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, list):
            raise ValueError("expected a JSON list of records")
        yield from data
        return
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _collect_first_n(path: Path, max_samples: int) -> List[Dict[str, Any]]:
    """Collect the first N records without loading the full dataset."""
    if max_samples <= 0:
        return []
    first = _first_nonspace(path)
    if first == "[":
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, list):
            raise ValueError("expected a JSON list of records")
        return data[:max_samples]
    records: List[Dict[str, Any]] = []
    for record in _iter_records(path):
        records.append(record)
        if len(records) >= max_samples:
            break
    return records


def _write_records(path: Path, records: List[Dict[str, Any]], as_list: bool) -> None:
    """Write records in JSON list or JSONL format."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if as_list:
        path.write_text(json.dumps(records, ensure_ascii=False), encoding="utf-8")
        return
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def _parse_args() -> argparse.Namespace:
    """Parse CLI arguments for dataset slicing."""
    parser = argparse.ArgumentParser(description="Slice a SU dataset.")
    parser.add_argument("--input", required=True, help="Path to data.json or data.jsonl")
    parser.add_argument("--output", required=True, help="Where to write the slice")
    parser.add_argument("--max-samples", type=int, default=8, help="Number of samples")
    parser.add_argument("--shuffle", action="store_true", help="Shuffle before slicing")
    parser.add_argument("--seed", type=int, default=0, help="Seed for shuffling")
    return parser.parse_args()


def main() -> None:
    """Entry point: slice the dataset and write a tiny file."""
    args = _parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)
    if not input_path.exists():
        raise FileNotFoundError(f"input not found: {input_path}")

    if args.shuffle:
        # Shuffling needs the full list in memory.
        records = list(_iter_records(input_path))
        random.Random(args.seed).shuffle(records)
        records = records[: max(0, args.max_samples)]
    else:
        records = _collect_first_n(input_path, args.max_samples)

    as_list = _first_nonspace(input_path) == "["
    _write_records(output_path, records, as_list=as_list)

    print(f"Wrote {len(records)} records to {output_path}")


if __name__ == "__main__":
    main()
