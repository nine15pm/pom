import json
from pathlib import Path

import pytest
import torch

from model.constants import DEFAULT_SEP_TOKEN, IGNORE_INDEX
from model.tokenizers import TokenIds, build_pom_tokenizer, resolve_token_ids
from train.tts_data import TtsCollator, TtsDataset

def _write_jsonl(path: Path, records: list[dict]) -> None:
    # Write JSONL records to disk.
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")


def _write_manifest(path: Path, *, boundary_policy: str, stop_policy: str) -> None:
    # Write a minimal manifest with boundary/stop policies.
    manifest = {
        "boundary_token_policy": boundary_policy,
        "stop_token_policy": stop_policy,
    }
    (path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


def _identity_collate(batch):
    # Keep per-sample dicts intact.
    return batch


def test_manifest_policy_validation(tmp_path):
    # Valid policies should pass, invalid policies should fail.
    shard_dir = tmp_path / "tts"
    shard_dir.mkdir(parents=True, exist_ok=True)
    _write_jsonl(
        shard_dir / "shard-00000.jsonl",
        [
            {
                "id": "s0",
                "source_id": "conv-0",
                "turn_index": 0,
                "assistant_text": "hello",
                "unit_ids": [1, 2],
            }
        ],
    )

    _write_manifest(
        shard_dir,
        boundary_policy="add_sep_in_stage1b_data_pipeline",
        stop_policy="add_eos_in_stage1b_data_pipeline",
    )
    samples = list(TtsDataset(json_path=str(shard_dir)))
    assert [sample["id"] for sample in samples] == ["s0"]

    _write_manifest(
        shard_dir,
        boundary_policy="wrong",
        stop_policy="add_eos_in_stage1b_data_pipeline",
    )
    with pytest.raises(ValueError, match="boundary_token_policy"):
        TtsDataset(json_path=str(shard_dir))


def test_worker_sharding_no_duplicates(tmp_path):
    # Worker sharding should yield each sample exactly once.
    shard_dir = tmp_path / "tts"
    shard_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for idx in range(6):
        records.append(
            {
                "id": f"s{idx}",
                "source_id": "conv",
                "turn_index": idx,
                "assistant_text": f"text {idx}",
                "unit_ids": [idx + 1],
            }
        )
    _write_jsonl(shard_dir / "shard-00000.jsonl", records)

    dataset = TtsDataset(json_path=str(shard_dir))
    from torch.utils.data import DataLoader
    loader = DataLoader(
        dataset,
        batch_size=1,
        num_workers=2,
        collate_fn=_identity_collate,
        multiprocessing_context="spawn",
        prefetch_factor=1,
    )

    sample_ids = []
    for batch in loader:
        sample_ids.append(batch[0]["id"])

    assert len(sample_ids) == 6
    assert len(set(sample_ids)) == 6


def test_collator_padding_and_boundary_tokens(local_base_model_id):
    # Collator should pad correctly and include <sep> + EOS in labels.
    tokenizer = build_pom_tokenizer(base_model_id=local_base_model_id)
    # Resolve token ids once to match runtime startup contracts.
    token_ids = resolve_token_ids(tokenizer)
    sep_id = tokenizer.convert_tokens_to_ids(DEFAULT_SEP_TOKEN)
    collator = TtsCollator(
        tokenizer,
        token_ids=token_ids,
        speech_token_offset=len(tokenizer),
        read_length=4,
        write_length=2,
    )

    batch = [
        {
            "id": "s0",
            "assistant_text": "a",
            "unit_ids": [1, 2, 3],
            "shard_index": 0,
            "shard_cursor": 0,
            "worker_id": 0,
        },
        {
            "id": "s1",
            "assistant_text": "a",
            "unit_ids": [4],
            "shard_index": 0,
            "shard_cursor": 0,
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

    eos_id = tokenizer.eos_token_id
    assert eos_id is not None

    for row in range(input_ids.size(0)):
        row_ids = input_ids[row].tolist()
        row_labels = labels[row].tolist()
        row_attention = attention[row].tolist()

        # Attention mask should be 1 for real tokens and 0 for padding.
        if 0 in row_attention:
            pad_start = row_attention.index(0)
            assert all(v == 0 for v in row_attention[pad_start:])
            assert all(v != 0 for v in row_attention[:pad_start])

        # <sep> must appear and should be masked in labels.
        assert sep_id in row_ids
        sep_pos = row_ids.index(sep_id)
        assert row_labels[sep_pos] == IGNORE_INDEX

        # EOS should be the last supervised token.
        assert eos_id in row_ids
        eos_pos = max(i for i, v in enumerate(row_ids) if v == eos_id)
        assert row_labels[eos_pos] == eos_id


def test_collator_token_contract_mismatch_raises(local_base_model_id):
    # Collator should fail fast if token contract does not match tokenizer.
    tokenizer = build_pom_tokenizer(base_model_id=local_base_model_id)
    token_ids = resolve_token_ids(tokenizer)
    sep_id = tokenizer.convert_tokens_to_ids(DEFAULT_SEP_TOKEN)

    # Build a mismatched contract to ensure startup/collator drift is rejected.
    bad_token_ids = TokenIds(speech_id=token_ids.speech_id, sep_id=sep_id + 1)
    with pytest.raises(ValueError, match="token_ids\\.sep_id"):
        TtsCollator(
            tokenizer,
            token_ids=bad_token_ids,
            speech_token_offset=len(tokenizer),
            read_length=1,
            write_length=1,
        )

    with pytest.raises(ValueError, match="speech_token_offset"):
        TtsCollator(
            tokenizer,
            token_ids=token_ids,
            speech_token_offset=len(tokenizer) + 5,
            read_length=1,
            write_length=1,
        )
