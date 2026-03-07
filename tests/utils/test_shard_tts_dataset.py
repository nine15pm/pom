import json
import sys
from pathlib import Path

import pytest

from utils import shard_tts_dataset


def _write_jsonl(path: Path, records: list[dict]) -> None:
    # Write JSON records to one JSONL shard file.
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")


def _read_jsonl(path: Path) -> list[dict]:
    # Read JSONL records into a list while skipping blank lines.
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _fixtures_dir() -> Path:
    # Return the shared repository fixture directory.
    return Path(__file__).resolve().parents[1] / "fixtures"


def _copy_case_to_input_shard(*, case_id: str, input_dir: Path) -> None:
    # Copy one named error-case record into a single-shard input directory.
    cases_path = _fixtures_dir() / "tts" / "error_cases.jsonl"
    cases = _read_jsonl(cases_path)
    for record in cases:
        if str(record.get("id")) == case_id:
            input_dir.mkdir(parents=True, exist_ok=True)
            _write_jsonl(input_dir / "shard-00000.jsonl", [record])
            return
    raise AssertionError(f"missing fixture case id: {case_id}")


def test_main_converts_su_shards_to_tts_and_writes_manifest(tmp_path, monkeypatch):
    # Use stable fixture shards so the conversion contract is regression-safe.
    input_dir = _fixtures_dir() / "tts"
    assert input_dir.is_dir()

    output_dir = tmp_path / "tts"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "shard_tts_dataset.py",
            "--input-dir",
            str(input_dir),
            "--output-dir",
            str(output_dir),
        ],
    )

    # Run the real conversion entrypoint with no mocking.
    shard_tts_dataset.main()

    # Verify per-sample fields and ordering in output shard 0.
    out0 = _read_jsonl(output_dir / "shard-00000.jsonl")
    assert [row["id"] for row in out0] == [
        "s00000-r000000-t001",
        "s00000-r000001-t001",
        "s00000-r000001-t003",
    ]
    assert [row["source_id"] for row in out0] == ["conv-0", "conv-1", "conv-1"]
    assert [row["assistant_text"] for row in out0] == ["a0", "a1", "a2"]
    assert [row["unit_ids"] for row in out0] == [[1, 2], [3, 4], [5]]

    # Verify per-sample fields in output shard 1.
    out1 = _read_jsonl(output_dir / "shard-00001.jsonl")
    assert [row["id"] for row in out1] == ["s00001-r000000-t001"]
    assert [row["source_id"] for row in out1] == ["conv-2"]
    assert [row["assistant_text"] for row in out1] == ["a3"]
    assert [row["unit_ids"] for row in out1] == [[6]]

    # Verify manifest totals and half-open shard index ranges.
    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["num_input_shards"] == 2
    assert manifest["num_output_shards"] == 2
    assert manifest["total_input_records"] == 3
    assert manifest["total_output_samples"] == 4
    assert manifest["boundary_token_policy"] == "add_sep_in_stage1b_data_pipeline"
    assert manifest["stop_token_policy"] == "add_eos_in_stage1b_data_pipeline"
    assert [s["output_samples"] for s in manifest["shards"]] == [3, 1]
    assert [s["start_index_inclusive"] for s in manifest["shards"]] == [0, 3]
    assert [s["end_index_exclusive"] for s in manifest["shards"]] == [3, 4]


def test_main_rejects_invalid_unit_format(tmp_path, monkeypatch):
    # Reuse shared fixture cases to verify strict malformed-format rejection.
    input_dir = tmp_path / "su_bad_format"
    _copy_case_to_input_shard(case_id="bad-units", input_dir=input_dir)

    output_dir = tmp_path / "tts"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "shard_tts_dataset.py",
            "--input-dir",
            str(input_dir),
            "--output-dir",
            str(output_dir),
        ],
    )

    # The strict format gate should reject non-numeric unit tokens.
    with pytest.raises(ValueError, match="invalid unit format"):
        shard_tts_dataset.main()


def test_main_rejects_out_of_range_unit_id(tmp_path, monkeypatch):
    # Reuse shared fixture cases to verify strict out-of-range rejection.
    input_dir = tmp_path / "su_bad_range"
    _copy_case_to_input_shard(case_id="bad-range", input_dir=input_dir)

    output_dir = tmp_path / "tts"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "shard_tts_dataset.py",
            "--input-dir",
            str(input_dir),
            "--output-dir",
            str(output_dir),
        ],
    )

    # The parser should reject ids above SPEECH_VOCAB_SIZE-1.
    with pytest.raises(ValueError, match="out of range"):
        shard_tts_dataset.main()


def test_main_fails_fast_when_conversion_produces_zero_samples(tmp_path, monkeypatch):
    # Reuse shared fixture cases for the no-assistant/zero-output contract.
    input_dir = tmp_path / "su_zero_output"
    _copy_case_to_input_shard(case_id="no-assistant", input_dir=input_dir)

    output_dir = tmp_path / "tts"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "shard_tts_dataset.py",
            "--input-dir",
            str(input_dir),
            "--output-dir",
            str(output_dir),
        ],
    )

    # This protects downstream training from silently empty TTS data.
    with pytest.raises(ValueError, match=r"zero .*TTS samples"):
        shard_tts_dataset.main()
