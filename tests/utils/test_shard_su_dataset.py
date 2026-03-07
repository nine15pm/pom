import json
import sys
from pathlib import Path

import pytest

from utils import shard_su_dataset


def _write_jsonl(path: Path, records: list[dict]) -> None:
    # Write JSONL records one-per-line.
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")


def _write_json_array(path: Path, records: list[dict]) -> None:
    # Write records as one top-level JSON list.
    path.write_text(json.dumps(records), encoding="utf-8")


def _read_ids(path: Path) -> list[str]:
    # Read record ids from a JSONL file.
    ids: list[str] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            ids.append(str(json.loads(line)["id"]))
    return ids


def test_main_shards_jsonl_in_order(tmp_path, monkeypatch):
    # Sharding should preserve exact input order and counts.
    records = [{"id": f"r{idx}", "conversation": []} for idx in range(7)]
    input_path = tmp_path / "input.jsonl"
    output_root = tmp_path / "out"
    _write_jsonl(input_path, records)

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "shard_su_dataset.py",
            "--input",
            str(input_path),
            "--output-dir",
            str(output_root),
            "--num-shards",
            "3",
        ],
    )
    shard_su_dataset.main()

    shard_dir = output_root / "su"
    shard_paths = sorted(shard_dir.glob("shard-*.jsonl"))
    assert len(shard_paths) == 3
    assert [p.name for p in shard_paths] == [
        "shard-00000.jsonl",
        "shard-00001.jsonl",
        "shard-00002.jsonl",
    ]

    merged_ids: list[str] = []
    for path in shard_paths:
        merged_ids.extend(_read_ids(path))
    assert merged_ids == [record["id"] for record in records]

    manifest = json.loads((shard_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["total_records"] == 7
    assert manifest["num_shards"] == 3
    assert [shard["records"] for shard in manifest["shards"]] == [3, 2, 2]
    assert [shard["start_index"] for shard in manifest["shards"]] == [0, 3, 5]


def test_main_rejects_more_shards_than_records_for_json_array(tmp_path, monkeypatch):
    # Requesting more shards than records should fail fast.
    input_path = tmp_path / "input.json"
    _write_json_array(input_path, [{"id": "a", "conversation": []}, {"id": "b", "conversation": []}])

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "shard_su_dataset.py",
            "--input",
            str(input_path),
            "--output-dir",
            str(tmp_path / "out"),
            "--num-shards",
            "3",
        ],
    )
    with pytest.raises(ValueError, match="cannot exceed total record count"):
        shard_su_dataset.main()
