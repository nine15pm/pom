"""Build TTS samples from SU conversation records."""

from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path
from typing import Any, Dict, Iterator

from model.constants import SPEECH_VOCAB_SIZE

MIN_UNIT_ID = 0
MAX_UNIT_ID = SPEECH_VOCAB_SIZE - 1
_UNIT_TEXT_RE = re.compile(r"\s*(?:<[-+]?\d+>\s*)+")
_UNIT_TOKEN_RE = re.compile(r"<([-+]?\d+)>")


def _iter_jsonl_records(path: Path) -> Iterator[Dict[str, Any]]:
    """Yield JSON object records from a JSONL file."""
    with path.open("r", encoding="utf-8") as handle:
        for lineno, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if not isinstance(record, dict):
                raise ValueError(f"{path}: line {lineno}: expected a JSON object record")
            yield record


def _parse_unit_ids(unit_text: str, *, source_id: str, turn_index: int) -> list[int]:
    """Parse one assistant unit string into validated unit ids."""
    # Accept only the canonical token format while allowing harmless whitespace.
    if _UNIT_TEXT_RE.fullmatch(unit_text) is None:
        raise ValueError(f"{source_id}: turn {turn_index}: invalid unit format")
    # Parse signed ints from each <...> token so malformed strings cannot slip through.
    unit_ids = [int(match.group(1)) for match in _UNIT_TOKEN_RE.finditer(unit_text)]
    if not unit_ids:
        raise ValueError(f"{source_id}: turn {turn_index}: unit field has no parseable ids")
    # Enforce CosyVoice2 shared unit range to catch data corruption early.
    for unit_id in unit_ids:
        if unit_id < MIN_UNIT_ID or unit_id > MAX_UNIT_ID:
            raise ValueError(
                f"{source_id}: turn {turn_index}: unit id {unit_id} out of range "
                f"[{MIN_UNIT_ID}, {MAX_UNIT_ID}]"
            )
    return unit_ids


def _iter_tts_samples(
    record: Dict[str, Any], *, shard_index: int, record_index: int
) -> Iterator[Dict[str, Any]]:
    """Yield TTS samples from one SU-style conversation record."""
    raw_source_id = record.get("id")
    if isinstance(raw_source_id, str) and raw_source_id.strip():
        source_id = raw_source_id.strip()
    else:
        source_id = f"s{shard_index:05d}-r{record_index:06d}"
    # Accept both supported conversation keys from prior dataset normalization.
    turns = record.get("conversation")
    if turns is None:
        turns = record.get("conversations")
    if not isinstance(turns, list):
        raise ValueError(f"{source_id}: missing conversation list")

    for turn_index, turn in enumerate(turns):
        if not isinstance(turn, dict):
            raise ValueError(f"{source_id}: turn {turn_index}: expected object")
        # Keep only assistant turns because TTS maps assistant text -> assistant speech units.
        role = str(turn.get("from", "")).strip().lower()
        if role in {"user", "human"}:
            continue
        if role not in {"assistant", "gpt"}:
            raise ValueError(f"{source_id}: turn {turn_index}: unknown role {turn.get('from')!r}")

        assistant_text = turn.get("text")
        if assistant_text is None:
            assistant_text = turn.get("value")
        if not isinstance(assistant_text, str) or not assistant_text.strip():
            raise ValueError(f"{source_id}: turn {turn_index}: assistant text is missing or empty")
        unit_text = turn.get("unit")
        if not isinstance(unit_text, str):
            raise ValueError(f"{source_id}: turn {turn_index}: assistant unit field must be a string")
        unit_ids = _parse_unit_ids(unit_text, source_id=source_id, turn_index=turn_index)

        # Use a globally unique deterministic id across shard boundaries.
        sample_id = f"s{shard_index:05d}-r{record_index:06d}-t{turn_index:03d}"
        yield {
            "id": sample_id,
            "source_id": source_id,
            "turn_index": turn_index,
            "assistant_text": assistant_text,
            "unit_ids": unit_ids,
        }


def _list_shard_paths(input_dir: Path) -> list[Path]:
    """List SU shard files in deterministic order."""
    if not input_dir.is_dir():
        raise FileNotFoundError(f"input shard directory not found: {input_dir}")
    shard_paths = sorted(input_dir.glob("shard-*.jsonl"))
    if not shard_paths:
        raise FileNotFoundError(f"no shard-*.jsonl files found in: {input_dir}")
    return shard_paths


def _ensure_clean_output_dir(path: Path, *, force: bool) -> None:
    """Create an empty output directory, optionally replacing an existing one."""
    if path.exists():
        if not force:
            raise FileExistsError(f"output already exists: {path} (use --force to overwrite)")
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def _convert_shards(
    *,
    input_shards: list[Path],
    output_dir: Path,
) -> tuple[list[Dict[str, Any]], int, int]:
    """Convert SU shards into TTS shards with stable ordering."""
    manifest_shards: list[Dict[str, Any]] = []
    total_input_records = 0
    total_output_samples = 0

    for shard_index, input_shard in enumerate(input_shards):
        output_name = f"shard-{shard_index:05d}.jsonl"
        output_path = output_dir / output_name
        shard_input_records = 0
        shard_output_samples = 0
        shard_start = total_output_samples

        with output_path.open("w", encoding="utf-8") as handle:
            for record_index, record in enumerate(_iter_jsonl_records(input_shard)):
                shard_input_records += 1
                for sample in _iter_tts_samples(
                    record,
                    shard_index=shard_index,
                    record_index=record_index,
                ):
                    # Write one assistant text->unit sample per line.
                    handle.write(json.dumps(sample, ensure_ascii=False) + "\n")
                    shard_output_samples += 1

        total_input_records += shard_input_records
        total_output_samples += shard_output_samples
        manifest_shards.append(
            {
                "index": shard_index,
                "path": output_name,
                "source_path": input_shard.name,
                "input_records": shard_input_records,
                "output_samples": shard_output_samples,
                "start_index_inclusive": shard_start,
                "end_index_exclusive": total_output_samples,
            }
        )

    return manifest_shards, total_input_records, total_output_samples


def _validate_output_shards(output_dir: Path, manifest_shards: list[Dict[str, Any]]) -> None:
    """Validate shard files and manifest counts after conversion."""
    files = sorted(output_dir.glob("shard-*.jsonl"))
    if len(files) != len(manifest_shards):
        raise ValueError(
            f"expected {len(manifest_shards)} shard files, found {len(files)} in {output_dir}"
        )
    for shard in manifest_shards:
        shard_path = output_dir / str(shard["path"])
        with shard_path.open("r", encoding="utf-8") as handle:
            line_count = sum(1 for line in handle if line.strip())
        if line_count != int(shard["output_samples"]):
            raise ValueError(
                f"record count mismatch for {shard_path}: "
                f"expected {shard['output_samples']}, got {line_count}"
            )


def _parse_args() -> argparse.Namespace:
    """Parse CLI flags for TTS shard conversion."""
    parser = argparse.ArgumentParser(
        description="Convert SU shard directory to TTS shard directory."
    )
    parser.add_argument("--input-dir", required=True, help="Path to SU shard directory")
    parser.add_argument("--output-dir", required=True, help="Path to TTS output directory")
    parser.add_argument("--force", action="store_true", help="Overwrite output directory if it exists")
    return parser.parse_args()


def main() -> None:
    """Convert SU shards to TTS shards and write manifest."""
    args = _parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    input_shards = _list_shard_paths(input_dir)
    _ensure_clean_output_dir(output_dir, force=bool(args.force))

    manifest_shards, total_input_records, total_output_samples = _convert_shards(
        input_shards=input_shards,
        output_dir=output_dir,
    )
    if total_output_samples <= 0:
        raise ValueError("conversion produced zero TTS samples")
    _validate_output_shards(output_dir, manifest_shards)

    # Keep boundary/stop semantics in manifest contract, not per-sample rows.
    manifest = {
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "num_input_shards": len(input_shards),
        "num_output_shards": len(manifest_shards),
        "total_input_records": total_input_records,
        "total_output_samples": total_output_samples,
        "boundary_token_policy": "add_sep_in_stage1b_data_pipeline",
        "stop_token_policy": "add_eos_in_stage1b_data_pipeline",
        "shards": manifest_shards,
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"input_dir: {input_dir}")
    print(f"output_dir: {output_dir}")
    print(f"num_input_shards: {len(input_shards)}")
    print(f"num_output_shards: {len(manifest_shards)}")
    print(f"total_input_records: {total_input_records}")
    print(f"total_output_samples: {total_output_samples}")
    print(f"manifest: {manifest_path}")


if __name__ == "__main__":
    main()
