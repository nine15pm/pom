import json
import sys
from pathlib import Path

import pytest

from model.constants import SPEECH_VOCAB_SIZE
from utils import shard_s2s_dataset


def _write_jsonl(path: Path, records: list[dict]) -> None:
    # Write JSON records one-per-line for shard inputs.
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")


def _read_jsonl(path: Path) -> list[dict]:
    # Read non-empty JSONL lines into Python dicts.
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def test_main_converts_su_shards_to_s2s_and_writes_manifest(tmp_path, monkeypatch):
    # Build two SU shard files to verify deterministic cross-shard conversion.
    input_dir = tmp_path / "su"
    input_dir.mkdir(parents=True, exist_ok=True)
    _write_jsonl(
        input_dir / "shard-00000.jsonl",
        [
            {
                "id": "conv-1",
                "conversation": [
                    {"from": "human", "audio": "audio/a.wav"},
                    {"from": "gpt", "text": "a0", "unit": "<1><2>"},
                    {"from": "human", "audio": "audio/b.wav"},
                    {"from": "gpt", "text": "a1", "unit": "<3>"},
                ],
            }
        ],
    )
    _write_jsonl(
        input_dir / "shard-00001.jsonl",
        [
            {
                "id": "conv-2",
                "conversation": [
                    {"from": "human", "audio": "audio/c.wav"},
                    {"from": "gpt", "text": "a2", "unit": "<4>"},
                ],
            }
        ],
    )

    output_dir = tmp_path / "s2s"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "shard_s2s_dataset.py",
            "--input-dir",
            str(input_dir),
            "--output-dir",
            str(output_dir),
        ],
    )

    # Run the real CLI entrypoint so the full conversion contract is exercised.
    shard_s2s_dataset.main()

    # Validate output sample fields, ids, and rolling history semantics in shard 0.
    out0 = _read_jsonl(output_dir / "shard-00000.jsonl")
    assert [row["id"] for row in out0] == ["s00000-r000000-t001", "s00000-r000000-t003"]
    assert [row["source_id"] for row in out0] == ["conv-1", "conv-1"]
    assert [row["turn_index"] for row in out0] == [1, 3]
    assert [row["assistant_text"] for row in out0] == ["a0", "a1"]
    assert [row["unit_ids"] for row in out0] == [[1, 2], [3]]
    assert out0[0]["history"] == [{"role": "user", "text": None, "audio": {"path": "audio/a.wav"}}]
    assert out0[1]["history"] == [
        {"role": "user", "text": None, "audio": {"path": "audio/a.wav"}},
        {"role": "assistant", "text": "a0", "audio": None},
        {"role": "user", "text": None, "audio": {"path": "audio/b.wav"}},
    ]

    # Validate output sample fields in shard 1.
    out1 = _read_jsonl(output_dir / "shard-00001.jsonl")
    assert [row["id"] for row in out1] == ["s00001-r000000-t001"]
    assert [row["source_id"] for row in out1] == ["conv-2"]
    assert [row["turn_index"] for row in out1] == [1]
    assert [row["assistant_text"] for row in out1] == ["a2"]
    assert [row["unit_ids"] for row in out1] == [[4]]
    assert out1[0]["history"] == [{"role": "user", "text": None, "audio": {"path": "audio/c.wav"}}]

    # Validate manifest totals, shard ranges, and schema used by runtime S2S loading.
    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["schema"] == "s2s_pairs_v1"
    assert manifest["num_input_shards"] == 2
    assert manifest["num_output_shards"] == 2
    assert manifest["total_input_records"] == 2
    assert manifest["total_output_samples"] == 3
    assert [s["output_samples"] for s in manifest["shards"]] == [2, 1]
    assert [s["start_index_inclusive"] for s in manifest["shards"]] == [0, 2]
    assert [s["end_index_exclusive"] for s in manifest["shards"]] == [2, 3]


@pytest.mark.parametrize(
    ("unit_text", "error_match"),
    [
        ("<1><x><2>", "invalid unit format"),
        (f"<{SPEECH_VOCAB_SIZE}>", "out of range"),
    ],
)
def test_main_rejects_invalid_unit_strings(tmp_path, monkeypatch, unit_text, error_match):
    # Keep one minimal user->assistant pair and vary only the invalid unit payload.
    input_dir = tmp_path / "su_bad_units"
    input_dir.mkdir(parents=True, exist_ok=True)
    _write_jsonl(
        input_dir / "shard-00000.jsonl",
        [
            {
                "id": "bad-conv",
                "conversation": [
                    {"from": "human", "audio": "audio/a.wav"},
                    {"from": "gpt", "text": "reply", "unit": unit_text},
                ],
            }
        ],
    )

    output_dir = tmp_path / "s2s"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "shard_s2s_dataset.py",
            "--input-dir",
            str(input_dir),
            "--output-dir",
            str(output_dir),
        ],
    )

    # Stage-2 conversion must fail fast on malformed or out-of-range speech units.
    with pytest.raises(ValueError, match=error_match):
        shard_s2s_dataset.main()


def test_main_fails_fast_when_conversion_produces_zero_samples(tmp_path, monkeypatch):
    # Build input with no assistant turn after user audio so conversion emits zero pairs.
    input_dir = tmp_path / "su_zero_output"
    input_dir.mkdir(parents=True, exist_ok=True)
    _write_jsonl(
        input_dir / "shard-00000.jsonl",
        [
            {
                "id": "no-assistant",
                "conversation": [
                    {"from": "human", "audio": "audio/a.wav"},
                    {"from": "human", "audio": "audio/b.wav"},
                ],
            }
        ],
    )

    output_dir = tmp_path / "s2s"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "shard_s2s_dataset.py",
            "--input-dir",
            str(input_dir),
            "--output-dir",
            str(output_dir),
        ],
    )

    # Training safety contract: refuse to write a dataset with zero S2S samples.
    with pytest.raises(ValueError, match=r"zero .*S2S samples"):
        shard_s2s_dataset.main()


def test_main_terminates_conversation_on_missing_user_audio(tmp_path, monkeypatch):
    # Missing user audio should stop that conversation but not block later records.
    input_dir = tmp_path / "su_missing_audio"
    input_dir.mkdir(parents=True, exist_ok=True)
    _write_jsonl(
        input_dir / "shard-00000.jsonl",
        [
            {
                "id": "conv-stop",
                "conversation": [
                    {"from": "human", "audio": "audio/a.wav"},
                    {"from": "gpt", "text": "keep-this", "unit": "<1>"},
                    {"from": "human"},
                    {"from": "gpt", "text": "drop-this", "unit": "<2>"},
                ],
            },
            {
                "id": "conv-next",
                "conversation": [
                    {"from": "human", "audio": "audio/b.wav"},
                    {"from": "gpt", "text": "still-processed", "unit": "<3>"},
                ],
            },
        ],
    )

    output_dir = tmp_path / "s2s"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "shard_s2s_dataset.py",
            "--input-dir",
            str(input_dir),
            "--output-dir",
            str(output_dir),
        ],
    )

    # Run real conversion and assert only valid pre-break + later-record samples remain.
    shard_s2s_dataset.main()
    out0 = _read_jsonl(output_dir / "shard-00000.jsonl")
    assert [row["source_id"] for row in out0] == ["conv-stop", "conv-next"]
    assert [row["assistant_text"] for row in out0] == ["keep-this", "still-processed"]
    assert [row["unit_ids"] for row in out0] == [[1], [3]]
