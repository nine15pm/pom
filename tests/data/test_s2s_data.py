import json
import shutil
from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader

from model.constants import IGNORE_INDEX
from model.tokenizers import build_pom_tokenizer, resolve_token_ids
from train.s2s_data import S2sCollator, S2sDataset


def _write_jsonl(path: Path, records: list[dict]) -> None:
    # Write one JSON object per line so dataset iteration can read shard records.
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")


def _write_manifest(path: Path, *, schema: str) -> None:
    # Write the minimal manifest schema expected by S2S dataset startup validation.
    manifest = {"schema": schema}
    (path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


def _copy_fixture_audio(src: Path, dst: Path) -> None:
    # Copy a fixture WAV into the temporary audio root used by dataset tests.
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(src, dst)


def _identity_collate(batch):
    # Keep per-sample dicts unchanged so worker-sharding behavior is easy to verify.
    return batch


def test_s2s_dataset_requires_manifest_schema_v1(tmp_path):
    # Fail fast when shard directories are missing manifest or use the wrong schema.
    shard_dir = tmp_path / "s2s"
    shard_dir.mkdir(parents=True, exist_ok=True)
    _write_jsonl(
        shard_dir / "shard-00000.jsonl",
        [
            {
                "id": "s0",
                "source_id": "conv-0",
                "turn_index": 1,
                "history": [],
                "assistant_text": "hello",
                "unit_ids": [1, 2],
            }
        ],
    )

    with pytest.raises(FileNotFoundError, match="manifest.json"):
        S2sDataset(json_path=str(shard_dir), audio_root=str(tmp_path))

    _write_manifest(shard_dir, schema="wrong_schema")
    with pytest.raises(ValueError, match="s2s_pairs_v1"):
        S2sDataset(json_path=str(shard_dir), audio_root=str(tmp_path))


def test_s2s_dataset_loads_multiturn_history_and_audio(tmp_path, fixture_audio_paths):
    # Dataset should preserve history order and decode user audio for each <speech> segment.
    audio_root = tmp_path
    _copy_fixture_audio(fixture_audio_paths[0], audio_root / "audio" / "a.wav")
    _copy_fixture_audio(fixture_audio_paths[1], audio_root / "audio" / "b.wav")

    shard_dir = tmp_path / "s2s"
    shard_dir.mkdir(parents=True, exist_ok=True)
    _write_manifest(shard_dir, schema="s2s_pairs_v1")
    _write_jsonl(
        shard_dir / "shard-00000.jsonl",
        [
            {
                "id": "s1",
                "source_id": "conv-1",
                "turn_index": 3,
                "history": [
                    {"role": "user", "text": None, "audio": {"path": "audio/a.wav"}},
                    {"role": "assistant", "text": "previous reply", "audio": None},
                    {"role": "user", "text": None, "audio": {"path": "audio/b.wav"}},
                ],
                "assistant_text": "next reply",
                "unit_ids": [10, 11, 12],
            }
        ],
    )

    samples = list(S2sDataset(json_path=str(shard_dir), audio_root=str(audio_root)))
    assert len(samples) == 1
    sample = samples[0]

    assert sample["assistant_text"] == "next reply"
    assert sample["unit_ids"] == [10, 11, 12]
    assert [turn["role"] for turn in sample["history"]] == ["user", "assistant", "user"]

    # User turns should carry decoded mono float32 waveforms with resolved paths.
    user0 = sample["history"][0]
    assert user0["audio"]["path"] == str(audio_root / "audio" / "a.wav")
    assert isinstance(user0["audio"]["array"], torch.Tensor)
    assert user0["audio"]["array"].dtype == torch.float32
    assert user0["audio"]["array"].ndim == 1
    assert user0["audio"]["array"].numel() > 0
    assert isinstance(user0["audio"]["sampling_rate"], int)
    assert user0["audio"]["sampling_rate"] > 0

    assistant = sample["history"][1]
    assert assistant["text"] == "previous reply"
    assert assistant["audio"] is None

    user1 = sample["history"][2]
    assert user1["audio"]["path"] == str(audio_root / "audio" / "b.wav")
    assert isinstance(user1["audio"]["array"], torch.Tensor)
    assert user1["audio"]["array"].dtype == torch.float32
    assert user1["audio"]["array"].ndim == 1
    assert user1["audio"]["array"].numel() > 0


def test_s2s_dataset_runtime_ratio_filter_drops_infeasible_samples(
    tmp_path,
    fixture_audio_paths,
    local_base_model_id,
):
    # Runtime ratio filtering should keep only samples that strict S2S read/write can consume.
    audio_root = tmp_path
    _copy_fixture_audio(fixture_audio_paths[0], audio_root / "audio" / "a.wav")

    shard_dir = tmp_path / "s2s"
    shard_dir.mkdir(parents=True, exist_ok=True)
    _write_manifest(shard_dir, schema="s2s_pairs_v1")
    _write_jsonl(
        shard_dir / "shard-00000.jsonl",
        [
            {
                "id": "keep",
                "source_id": "conv-keep",
                "turn_index": 1,
                "history": [{"role": "user", "text": None, "audio": {"path": "audio/a.wav"}}],
                "assistant_text": "ok",
                "unit_ids": list(range(10)),
            },
            {
                "id": "drop",
                "source_id": "conv-drop",
                "turn_index": 1,
                "history": [{"role": "user", "text": None, "audio": {"path": "audio/a.wav"}}],
                "assistant_text": "this assistant response is intentionally much longer than three tokens",
                "unit_ids": list(range(10)),
            },
        ],
    )

    # Use the real tokenizer so filtering behavior matches actual training startup config.
    tokenizer = build_pom_tokenizer(base_model_id=local_base_model_id)
    dataset = S2sDataset(
        json_path=str(shard_dir),
        audio_root=str(audio_root),
        tokenizer=tokenizer,
        read_length=3,
        write_length=10,
    )

    sample_ids = [sample["id"] for sample in dataset]
    assert sample_ids == ["keep"]


def test_s2s_collator_builds_teacher_forcing_batch(local_base_model_id):
    # Collator should build padded Thinker inputs, reply-only labels, and aligned speech waveforms.
    tokenizer = build_pom_tokenizer(base_model_id=local_base_model_id)
    token_ids = resolve_token_ids(tokenizer)
    collator = S2sCollator(
        tokenizer,
        token_ids=token_ids,
        read_length=3,
        write_length=10,
    )

    wave_a = torch.tensor([0.1, 0.2], dtype=torch.float32)
    wave_b = torch.tensor([0.3], dtype=torch.float32)
    wave_c = torch.tensor([0.4, 0.5, 0.6], dtype=torch.float32)
    batch = [
        {
            "id": "sA",
            "history": [
                {"role": "user", "text": None, "audio": {"array": wave_a, "sampling_rate": 16000}},
                {"role": "assistant", "text": "previous", "audio": None},
                {"role": "user", "text": None, "audio": {"array": wave_b, "sampling_rate": 8000}},
            ],
            "assistant_text": "ok",
            "unit_ids": list(range(20)),
            "shard_index": 0,
            "shard_cursor": 0,
            "worker_id": 0,
        },
        {
            "id": "sB",
            "history": [
                {"role": "user", "text": None, "audio": {"array": wave_c, "sampling_rate": 16000}},
            ],
            "assistant_text": "yes",
            "unit_ids": list(range(10)),
            "shard_index": 1,
            "shard_cursor": 1,
            "worker_id": 0,
        },
    ]

    output = collator(batch)

    input_ids = output["input_ids"]
    labels = output["labels"]
    attention = output["attention_mask"]
    assert input_ids.shape == labels.shape == attention.shape
    assert input_ids.dtype == torch.long
    assert labels.dtype == torch.long
    assert attention.dtype == torch.long
    assert input_ids.size(0) == 2

    # Every supervised position should match its token id; prompt/padding stays IGNORE_INDEX.
    for row in range(input_ids.size(0)):
        row_attention = attention[row].tolist()
        if 0 in row_attention:
            pad_start = row_attention.index(0)
            assert all(v == 1 for v in row_attention[:pad_start])
            assert all(v == 0 for v in row_attention[pad_start:])
        active = attention[row] == 1
        supervised = (labels[row] != IGNORE_INDEX) & active
        assert supervised.any()
        assert torch.equal(labels[row][supervised], input_ids[row][supervised])
        ignored = (labels[row] == IGNORE_INDEX) & active
        assert ignored.any()

    # Flattened speech arrays must align 1:1 with <speech> sentinels across the padded batch.
    speech_id = tokenizer.convert_tokens_to_ids("<speech>")
    speech_tokens = (input_ids == speech_id).sum().item()
    assert speech_tokens == len(output["speech_waveforms"])
    assert len(output["speech_sampling_rate"]) == len(output["speech_waveforms"])
    assert output["speech_sampling_rate"] == [16000, 8000, 16000]
    assert torch.equal(output["speech_waveforms"][0], wave_a)
    assert torch.equal(output["speech_waveforms"][1], wave_b)
    assert torch.equal(output["speech_waveforms"][2], wave_c)

    # Content and unit targets must be present per sample for downstream gate fusion and Talker loss.
    assert len(output["content_ids"]) == 2
    assert len(output["unit_ids"]) == 2
    for content_ids in output["content_ids"]:
        assert isinstance(content_ids, torch.Tensor)
        assert content_ids.dtype == torch.long
        assert content_ids.ndim == 1
        assert content_ids.numel() > 0
    for unit_ids in output["unit_ids"]:
        assert isinstance(unit_ids, torch.Tensor)
        assert unit_ids.dtype == torch.long
        assert unit_ids.ndim == 1
        assert unit_ids.numel() > 0

    assert output["sample_ids"] == ["sA", "sB"]
    assert output["shard_indices"] == [0, 1]
    assert output["shard_cursors"] == [0, 1]
    assert output["worker_ids"] == [0, 0]


def test_s2s_collator_fails_fast_on_ratio_contract_mismatch(local_base_model_id):
    # Collator should fail when content/unit lengths violate strict Stage-2 read/write feasibility.
    tokenizer = build_pom_tokenizer(base_model_id=local_base_model_id)
    token_ids = resolve_token_ids(tokenizer)
    collator = S2sCollator(
        tokenizer,
        token_ids=token_ids,
        read_length=3,
        write_length=10,
    )

    sample = {
        "id": "ratio-bad",
        "history": [
            {"role": "user", "text": None, "audio": {"array": torch.tensor([0.1]), "sampling_rate": 16000}},
        ],
        # Keep this long so content token count reliably exceeds the R=3, W=10 limit for 10 units.
        "assistant_text": "this response is intentionally very long to exceed three read tokens",
        "unit_ids": list(range(10)),
    }

    # Sanity-check test setup so the failure reason is always the ratio contract.
    content_ids = tokenizer(sample["assistant_text"], add_special_tokens=False)["input_ids"]
    assert len(content_ids) > 3

    with pytest.raises(ValueError, match="fail Stage-2 Read/Write ratio"):
        collator([sample])


def test_s2s_dataset_worker_sharding_has_no_duplicates(tmp_path, fixture_audio_paths):
    # Multi-worker iteration should emit each sample once with no duplicates.
    audio_root = tmp_path
    _copy_fixture_audio(fixture_audio_paths[0], audio_root / "audio" / "a.wav")

    shard_dir = tmp_path / "s2s"
    shard_dir.mkdir(parents=True, exist_ok=True)
    _write_manifest(shard_dir, schema="s2s_pairs_v1")
    _write_jsonl(
        shard_dir / "shard-00000.jsonl",
        [
            {
                "id": f"s{idx}",
                "source_id": "conv",
                "turn_index": idx,
                "history": [{"role": "user", "text": None, "audio": {"path": "audio/a.wav"}}],
                "assistant_text": f"reply {idx}",
                "unit_ids": [1, 2, 3],
            }
            for idx in range(6)
        ],
    )

    dataset = S2sDataset(json_path=str(shard_dir), audio_root=str(audio_root))
    loader = DataLoader(
        dataset,
        batch_size=1,
        num_workers=2,
        collate_fn=_identity_collate,
        multiprocessing_context="spawn",
        prefetch_factor=1,
    )

    sample_ids: list[str] = []
    for batch in loader:
        sample_ids.append(batch[0]["id"])

    assert len(sample_ids) == 6
    assert len(set(sample_ids)) == 6
