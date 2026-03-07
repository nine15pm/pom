"""Build Stage-2 S2S samples from SU conversation shards."""

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
    """Yield JSON object records from one JSONL shard."""
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
    """Parse one assistant unit string into validated speech unit ids."""
    # Accept only canonical "<N><M>..." formatting with optional whitespace.
    if _UNIT_TEXT_RE.fullmatch(unit_text) is None:
        raise ValueError(f"{source_id}: turn {turn_index}: invalid unit format")
    unit_ids = [int(match.group(1)) for match in _UNIT_TOKEN_RE.finditer(unit_text)]
    if not unit_ids:
        raise ValueError(f"{source_id}: turn {turn_index}: unit field has no parseable ids")
    # Enforce shared CosyVoice2 unit range to catch data corruption early.
    for unit_id in unit_ids:
        if unit_id < MIN_UNIT_ID or unit_id > MAX_UNIT_ID:
            raise ValueError(
                f"{source_id}: turn {turn_index}: unit id {unit_id} out of range "
                f"[{MIN_UNIT_ID}, {MAX_UNIT_ID}]"
            )
    return unit_ids


def _get_turns(record: Dict[str, Any], *, source_id: str) -> list[Dict[str, Any]]:
    """Extract a normalized conversation turn list from one source record."""
    turns = record.get("conversation")
    if turns is None:
        turns = record.get("conversations")
    if not isinstance(turns, list):
        raise ValueError(f"{source_id}: missing conversation list")
    return turns


def _role_from_turn(turn: Dict[str, Any], *, source_id: str, turn_index: int) -> str:
    """Normalize raw dataset role names into user/assistant."""
    raw = str(turn.get("from", "")).strip().lower()
    if raw in {"user", "human"}:
        return "user"
    if raw in {"assistant", "gpt"}:
        return "assistant"
    raise ValueError(f"{source_id}: turn {turn_index}: unknown role {turn.get('from')!r}")


def _turn_text(turn: Dict[str, Any]) -> str | None:
    """Read turn text from either value/text fields."""
    text = turn.get("value")
    if text is None:
        text = turn.get("text")
    if text is None:
        return None
    return str(text)


def _turn_audio_path(turn: Dict[str, Any]) -> str | None:
    """Read turn audio path from either audio/speech fields."""
    audio = turn.get("audio")
    if audio is None:
        audio = turn.get("speech")
    if audio is None:
        return None
    if isinstance(audio, dict):
        raise ValueError("embedded audio arrays are not supported; expected a file path")
    return str(audio)


def _iter_s2s_samples(
    record: Dict[str, Any], *, shard_index: int, record_index: int
) -> Iterator[Dict[str, Any]]:
    """Yield one S2S sample per user->assistant pair from one conversation record."""
    raw_source_id = record.get("id")
    if isinstance(raw_source_id, str) and raw_source_id.strip():
        source_id = raw_source_id.strip()
    else:
        source_id = f"s{shard_index:05d}-r{record_index:06d}"

    turns = _get_turns(record, source_id=source_id)

    history: list[Dict[str, Any]] = []
    pending_user: Dict[str, Any] | None = None

    for turn_index, turn in enumerate(turns):
        if not isinstance(turn, dict):
            raise ValueError(f"{source_id}: turn {turn_index}: expected object")
        role = _role_from_turn(turn, source_id=source_id, turn_index=turn_index)

        if role == "user":
            audio_path = _turn_audio_path(turn)
            if not audio_path:
                # Match SU semantics: missing user audio terminates the conversation.
                break
            pending_user = {
                "role": "user",
                "text": None,
                "audio": {"path": audio_path},
            }
            continue

        if pending_user is None:
            continue

        assistant_text = _turn_text(turn)
        if not isinstance(assistant_text, str) or not assistant_text.strip():
            pending_user = None
            continue

        unit_text = turn.get("unit")
        if not isinstance(unit_text, str):
            raise ValueError(f"{source_id}: turn {turn_index}: assistant unit field must be a string")
        unit_ids = _parse_unit_ids(unit_text, source_id=source_id, turn_index=turn_index)

        sample_id = f"s{shard_index:05d}-r{record_index:06d}-t{turn_index:03d}"
        yield {
            "id": sample_id,
            "source_id": source_id,
            "turn_index": turn_index,
            # History ends at the current user turn; current assistant is the target.
            "history": history + [pending_user],
            "assistant_text": assistant_text,
            "unit_ids": unit_ids,
        }

        # Extend rolling history with the newly resolved pair.
        history = history + [
            pending_user,
            {"role": "assistant", "text": assistant_text, "audio": None},
        ]
        pending_user = None


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
    """Convert SU conversation shards to S2S pair-sample shards."""
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
                for sample in _iter_s2s_samples(
                    record,
                    shard_index=shard_index,
                    record_index=record_index,
                ):
                    # Keep one S2S pair sample per output line.
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
    """Validate shard files and per-shard output counts after conversion."""
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
    """Parse CLI flags for S2S shard conversion."""
    parser = argparse.ArgumentParser(
        description="Convert SU shard directory to S2S pair-sample shard directory."
    )
    parser.add_argument("--input-dir", required=True, help="Path to SU shard directory")
    parser.add_argument("--output-dir", required=True, help="Path to S2S output directory")
    parser.add_argument("--force", action="store_true", help="Overwrite output directory if it exists")
    return parser.parse_args()


def main() -> None:
    """Convert SU shards to S2S pair samples and write a manifest."""
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
        raise ValueError("conversion produced zero S2S samples")
    _validate_output_shards(output_dir, manifest_shards)

    # Keep conversion metadata explicit for reproducibility and debugging.
    manifest = {
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "num_input_shards": len(input_shards),
        "num_output_shards": len(manifest_shards),
        "total_input_records": total_input_records,
        "total_output_samples": total_output_samples,
        "schema": "s2s_pairs_v1",
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
