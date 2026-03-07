"""Normalize SU (Speech Understanding) dataset speech paths into a canonical JSONL file.

This keeps raw vendor data untouched and produces training-ready metadata.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterator, Tuple


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
    """Yield JSON object records from JSONL or JSON array input."""
    first = _first_nonspace(path)
    if first == "[":
        yield from _iter_json_array_records(path)
        return
    yield from _iter_jsonl_records(path)


def _get_turns(record: Dict[str, Any]) -> list[Dict[str, Any]]:
    """Return conversation turns from either supported key."""
    turns = record.get("conversations")
    if turns is None:
        turns = record.get("conversation")
    if not isinstance(turns, list):
        raise ValueError("record is missing a conversation list")
    return turns


def _normalize_prefix(prefix: str) -> str:
    """Normalize path prefix input for deterministic matching."""
    prefix = prefix.strip()
    if prefix and not prefix.endswith("/"):
        prefix = prefix + "/"
    return prefix


def _rewrite_speech_path(path: str, *, from_prefix: str, to_prefix: str) -> Tuple[str, bool]:
    """Rewrite one speech path if it matches from_prefix."""
    if not from_prefix:
        return path, False
    if not path.startswith(from_prefix):
        return path, False
    suffix = path[len(from_prefix) :]
    return f"{to_prefix}{suffix}", True


def _resolve_audio_path(path: str, audio_root: Path) -> Path:
    """Resolve path the same way SuDataset does."""
    audio_path = Path(path)
    if audio_path.is_absolute():
        return audio_path
    return audio_root / audio_path


def _parse_args() -> argparse.Namespace:
    """Parse CLI args."""
    parser = argparse.ArgumentParser(description="Normalize SU speech paths.")
    parser.add_argument("--input", required=True, help="Input JSON/JSONL metadata path")
    parser.add_argument("--output", required=True, help="Output JSONL path")
    parser.add_argument(
        "--audio-root",
        required=True,
        help="Audio root used to validate rewritten speech paths",
    )
    parser.add_argument(
        "--from-prefix",
        default="data/multiturn_instruction/en/",
        help="Prefix to replace in turn['speech']",
    )
    parser.add_argument(
        "--to-prefix",
        default="",
        help="Replacement prefix (for example: 'wav/' or empty)",
    )
    parser.add_argument(
        "--allow-missing",
        action="store_true",
        help="Allow missing rewritten files (default: fail if any are missing)",
    )
    parser.add_argument(
        "--max-missing-report",
        type=int,
        default=20,
        help="How many missing path examples to print",
    )
    parser.add_argument(
        "--max-records",
        type=int,
        default=None,
        help="Optional debug cap on number of records to process",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite output file if it already exists",
    )
    return parser.parse_args()


def main() -> None:
    """Rewrite speech paths and emit canonical JSONL metadata."""
    args = _parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)
    audio_root = Path(args.audio_root)

    if not input_path.exists():
        raise FileNotFoundError(f"input not found: {input_path}")
    if not audio_root.exists():
        raise FileNotFoundError(f"audio_root not found: {audio_root}")
    if output_path.exists() and not args.force:
        raise FileExistsError(f"output already exists: {output_path} (use --force to overwrite)")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    from_prefix = _normalize_prefix(args.from_prefix)
    to_prefix = _normalize_prefix(args.to_prefix)

    records = 0
    speech_turns = 0
    rewritten = 0
    missing = 0
    missing_examples: list[str] = []

    with output_path.open("w", encoding="utf-8") as out:
        for record_idx, record in enumerate(_iter_records(input_path)):
            if args.max_records is not None and record_idx >= args.max_records:
                break
            record_id = str(record.get("id", f"record-{record_idx}"))
            turns = _get_turns(record)
            for turn_idx, turn in enumerate(turns):
                if not isinstance(turn, dict):
                    raise ValueError(f"{record_id}: turn {turn_idx} is not an object")
                speech = turn.get("speech")
                if speech is None:
                    continue
                if not isinstance(speech, str):
                    raise ValueError(f"{record_id}: turn {turn_idx} speech field must be a string")

                speech_turns += 1
                new_speech, changed = _rewrite_speech_path(
                    speech,
                    from_prefix=from_prefix,
                    to_prefix=to_prefix,
                )
                if changed:
                    turn["speech"] = new_speech
                    rewritten += 1
                resolved = _resolve_audio_path(new_speech, audio_root)
                if not resolved.exists():
                    missing += 1
                    if len(missing_examples) < max(0, int(args.max_missing_report)):
                        missing_examples.append(
                            f"{record_id} turn={turn_idx} speech={new_speech} resolved={resolved}"
                        )

            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            records += 1

    print(f"input: {input_path}")
    print(f"output: {output_path}")
    print(f"audio_root: {audio_root}")
    print(f"records_written: {records}")
    print(f"speech_turns: {speech_turns}")
    print(f"rewritten_speech_paths: {rewritten}")
    print(f"missing_files: {missing}")
    if missing_examples:
        print("missing_examples:")
        for example in missing_examples:
            print(f"  {example}")

    if missing > 0 and not args.allow_missing:
        raise FileNotFoundError(
            f"found {missing} missing speech files after rewrite; "
            f"use --allow-missing to keep output anyway"
        )


if __name__ == "__main__":
    main()
