import json
import shutil
from pathlib import Path

import pytest
import torch
from transformers import AutoTokenizer

from torch.utils.data import DataLoader

from train.su_data import SuCollator, SuDataset
from model.constants import DEFAULT_SPEECH_TOKEN, IGNORE_INDEX
from model.tokenizers import TokenIds, build_pom_tokenizer, resolve_token_ids
from train.su_sequence_builder import build_su_messages, build_su_token_ids


def _write_json_list(path: Path, records: list[dict]) -> None:
    # Write a JSON list file to disk.
    path.write_text(json.dumps(records), encoding="utf-8")


def _write_jsonl(path: Path, records: list[dict]) -> None:
    # Write a JSONL file to disk (one record per line).
    lines = [json.dumps(record) for record in records]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _copy_fixture_audio(src: Path, dst: Path) -> None:
    # Copy a fixture WAV into a temp audio root.
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(src, dst)


def _identity_collate(batch):
    # Return the batch unchanged for DataLoader tests.
    return batch


def _build_test_tokenizer_and_ids(local_base_model_id: str):
    # Build tokenizer + resolved token ids using the runtime startup contract.
    tokenizer = build_pom_tokenizer(base_model_id=local_base_model_id)
    token_ids = resolve_token_ids(tokenizer)
    return tokenizer, token_ids


def test_echox_multiturn_and_termination(tmp_path, fixture_audio_paths):
    # Multi-turn EchoX should include current user in history and stop after missing audio.
    audio_root = tmp_path
    _copy_fixture_audio(fixture_audio_paths[0], audio_root / "audio" / "a.wav")
    _copy_fixture_audio(fixture_audio_paths[1], audio_root / "audio" / "b.wav")

    record = {
        "id": "conv-1",
        "conversations": [
            {"from": "user", "value": "u1", "audio": "audio/a.wav", "wer": 0.1},
            {"from": "assistant", "value": "a1"},
            {"from": "user", "value": "u2", "audio": "audio/b.wav", "wer": 0.2},
            {"from": "assistant", "value": "a2"},
            {"from": "user", "value": "u3", "audio": None},
            {"from": "assistant", "value": "a3"},
            {"from": "user", "value": "u4", "audio": "audio/a.wav"},
            {"from": "assistant", "value": "a4"},
        ],
    }

    json_path = tmp_path / "echox.json"
    _write_json_list(json_path, [record])

    dataset = SuDataset(json_path=str(json_path), audio_root=str(audio_root))
    samples = list(dataset)

    assert len(samples) == 2

    first = samples[0]
    assert first["assistant_text"] == "a1"
    assert [turn["role"] for turn in first["history"]] == ["user"]
    assert first["history"][0]["text"] is None

    second = samples[1]
    assert second["assistant_text"] == "a2"
    assert [turn["role"] for turn in second["history"]] == ["user", "assistant", "user"]
    assert second["history"][-1]["role"] == "user"
    assert isinstance(second["history"][-1]["audio"]["array"], torch.Tensor)


def test_instructs2s_jsonl_order_and_history(tmp_path, fixture_audio_paths):
    # InstructS2S JSONL should preserve order and build multi-turn history.
    audio_root = tmp_path
    _copy_fixture_audio(fixture_audio_paths[0], audio_root / "audio" / "a.wav")
    _copy_fixture_audio(fixture_audio_paths[1], audio_root / "audio" / "b.wav")

    record = {
        "id": "conv-2",
        "conversation": [
            {"from": "human", "speech": "audio/a.wav", "text": "h1"},
            {"from": "gpt", "text": "g1"},
            {"from": "human", "speech": "audio/b.wav", "text": "h2"},
            {"from": "gpt", "text": "g2"},
        ],
    }

    jsonl_path = tmp_path / "instruct.jsonl"
    _write_jsonl(jsonl_path, [record])

    dataset = SuDataset(json_path=str(jsonl_path), audio_root=str(audio_root))
    samples = list(dataset)

    assert [sample["assistant_text"] for sample in samples] == ["g1", "g2"]
    assert [turn["role"] for turn in samples[0]["history"]] == ["user"]
    assert [turn["role"] for turn in samples[1]["history"]] == ["user", "assistant", "user"]


def test_echox_wer_cutoff_truncates_conversation(tmp_path, fixture_audio_paths):
    # WER cutoff should terminate conversation and drop later pairs.
    audio_root = tmp_path
    _copy_fixture_audio(fixture_audio_paths[0], audio_root / "audio" / "a.wav")
    _copy_fixture_audio(fixture_audio_paths[1], audio_root / "audio" / "b.wav")

    record = {
        "id": "conv-wer",
        "conversations": [
            {"from": "user", "value": "u1", "audio": "audio/a.wav", "wer": 0.1},
            {"from": "assistant", "value": "a1"},
            {"from": "user", "value": "u2", "audio": "audio/b.wav", "wer": 0.9},
            {"from": "assistant", "value": "a2"},
        ],
    }

    json_path = tmp_path / "echox_wer.json"
    _write_json_list(json_path, [record])

    dataset = SuDataset(
        json_path=str(json_path),
        audio_root=str(audio_root),
        max_wer=0.5,
    )
    samples = list(dataset)

    assert len(samples) == 1
    assert samples[0]["assistant_text"] == "a1"


def test_leading_assistant_is_ignored(tmp_path, fixture_audio_paths):
    # Assistant turns without a pending user should not create samples.
    audio_root = tmp_path
    _copy_fixture_audio(fixture_audio_paths[0], audio_root / "audio" / "a.wav")

    record = {
        "id": "conv-leading",
        "conversations": [
            {"from": "assistant", "value": "a0"},
            {"from": "user", "value": "u1", "audio": "audio/a.wav"},
            {"from": "assistant", "value": "a1"},
        ],
    }

    json_path = tmp_path / "leading.json"
    _write_json_list(json_path, [record])

    dataset = SuDataset(json_path=str(json_path), audio_root=str(audio_root))
    samples = list(dataset)

    assert len(samples) == 1
    assert samples[0]["assistant_text"] == "a1"
    assert [turn["role"] for turn in samples[0]["history"]] == ["user"]


def test_audio_decode_mono_float32_no_resample(tmp_path, fixture_audio_paths):
    # Audio should decode to mono float32 without resampling.
    audio_root = tmp_path
    stereo_src = next(
        (path for path in fixture_audio_paths if path.name == "stereo_constant.wav"),
        None,
    )
    assert stereo_src is not None, "Missing stereo_constant.wav fixture"
    _copy_fixture_audio(stereo_src, audio_root / "audio" / "stereo_constant.wav")

    record = {
        "id": "conv-audio",
        "conversations": [
            {"from": "user", "value": "u1", "audio": "audio/stereo_constant.wav"},
            {"from": "assistant", "value": "a1"},
        ],
    }

    json_path = tmp_path / "audio.json"
    _write_json_list(json_path, [record])

    dataset = SuDataset(json_path=str(json_path), audio_root=str(audio_root))
    sample = next(iter(dataset))

    audio = sample["history"][0]["audio"]
    waveform = audio["array"]
    assert waveform.dtype == torch.float32
    assert waveform.shape[0] == 16
    assert audio["sampling_rate"] == 8000
    assert torch.allclose(waveform, torch.full_like(waveform, 0.5), atol=1e-3)


def test_audio_path_absolute_is_not_joined_with_root(tmp_path, fixture_audio_paths):
    # Absolute audio paths should bypass audio_root joining.
    abs_dir = tmp_path / "abs_audio"
    abs_dir.mkdir(parents=True, exist_ok=True)
    abs_path = abs_dir / "a.wav"
    _copy_fixture_audio(fixture_audio_paths[0], abs_path)

    audio_root = tmp_path / "audio_root"
    audio_root.mkdir(parents=True, exist_ok=True)

    record = {
        "id": "conv-abs",
        "conversations": [
            {"from": "user", "value": "u1", "audio": str(abs_path)},
            {"from": "assistant", "value": "a1"},
        ],
    }

    json_path = tmp_path / "abs.json"
    _write_json_list(json_path, [record])

    dataset = SuDataset(json_path=str(json_path), audio_root=str(audio_root))
    sample = next(iter(dataset))

    # Ensure we kept the absolute path without prefixing audio_root.
    audio = sample["history"][0]["audio"]
    assert audio["path"] == str(abs_path)
    assert isinstance(audio["array"], torch.Tensor)


def test_missing_audio_file_raises_during_iteration(tmp_path):
    # Missing audio files should raise when the dataset is iterated.
    audio_root = tmp_path / "data"
    audio_root.mkdir(parents=True, exist_ok=True)
    missing_path = audio_root / "audio" / "missing.wav"

    record = {
        "id": "conv-missing",
        "conversations": [
            {"from": "user", "value": "u1", "audio": "audio/missing.wav"},
            {"from": "assistant", "value": "a1"},
        ],
    }

    json_path = tmp_path / "missing.json"
    _write_json_list(json_path, [record])

    dataset = SuDataset(json_path=str(json_path), audio_root=str(audio_root))
    with pytest.raises((RuntimeError, FileNotFoundError, OSError)) as excinfo:
        next(iter(dataset))

    # Do not assert on message text (backend-dependent).


def test_worker_sharding_no_duplicates(tmp_path, fixture_audio_paths):
    # Worker sharding should yield each sample exactly once.
    audio_root = tmp_path
    _copy_fixture_audio(fixture_audio_paths[0], audio_root / "audio" / "a.wav")

    conversations = []
    for idx in range(5):
        conversations.append({"from": "user", "value": f"u{idx}", "audio": "audio/a.wav"})
        conversations.append({"from": "assistant", "value": f"a{idx}"})

    record = {"id": "conv-workers", "conversations": conversations}
    json_path = tmp_path / "workers.json"
    _write_json_list(json_path, [record])

    dataset = SuDataset(json_path=str(json_path), audio_root=str(audio_root))
    # Keep per-sample dicts intact and enforce spawn for worker sharding stability.
    loader = DataLoader(
        dataset,
        batch_size=1,
        num_workers=2,
        collate_fn=_identity_collate,
        multiprocessing_context="spawn",
        prefetch_factor=1,
    )

    texts = []
    for batch in loader:
        sample = batch[0]
        texts.append(sample["assistant_text"])

    assert len(texts) == 5
    assert len(set(texts)) == 5


def test_collator_prompt_and_label_masking(local_base_model_id):
    # Collator should mask all prompt tokens and keep only assistant reply labels.
    tokenizer, token_ids = _build_test_tokenizer_and_ids(local_base_model_id)
    collator = SuCollator(tokenizer, token_ids)

    history = [
        {
            "role": "user",
            "text": None,
            "audio": {"array": torch.tensor([0.1, 0.2]), "sampling_rate": 16000},
        },
        {"role": "assistant", "text": "prev reply", "audio": None},
        {
            "role": "user",
            "text": None,
            "audio": {"array": torch.tensor([0.3, 0.4, 0.5]), "sampling_rate": 16000},
        },
    ]
    sample = {"id": "s1", "history": history, "assistant_text": "next reply"}

    batch = collator([sample])
    input_ids = batch["input_ids"][0].tolist()
    labels = batch["labels"][0].tolist()

    messages, _, _ = build_su_messages(sample["history"])
    assert messages[0] == {"role": "system", "content": ""}

    prompt_ids, reply_ids = build_su_token_ids(
        tokenizer,
        messages,
        assistant_text="next reply",
    )
    assert reply_ids is not None

    full_ids = tokenizer.apply_chat_template(
        messages + [{"role": "assistant", "content": "next reply"}],
        tokenize=True,
        add_generation_prompt=False,
    )["input_ids"]

    assert full_ids == prompt_ids + reply_ids
    assert input_ids == full_ids
    assert labels[: len(prompt_ids)] == [IGNORE_INDEX] * len(prompt_ids)
    assert labels[len(prompt_ids) : len(prompt_ids) + len(reply_ids)] == reply_ids


def test_su_messages_inserts_blank_system_turn():
    # Builder should inject an empty system turn when none is provided.
    history = [
        {"role": "user", "text": None, "audio": {"array": torch.tensor([0.1]), "sampling_rate": 16000}},
        {"role": "assistant", "text": "ok", "audio": None},
    ]

    messages, waveforms, sampling_rates = build_su_messages(history)

    assert messages[0] == {"role": "system", "content": ""}
    assert messages[1]["role"] == "user"
    assert len(waveforms) == 1
    assert sampling_rates == [16000]


def test_collator_waveform_order_and_sampling_rates(local_base_model_id):
    # Collator should flatten waveforms in <speech> token order across the batch.
    tokenizer, token_ids = _build_test_tokenizer_and_ids(local_base_model_id)
    collator = SuCollator(tokenizer, token_ids)

    wave_a = torch.tensor([1.0, 1.0])
    wave_b = torch.tensor([2.0, 2.0, 2.0])
    wave_c = torch.tensor([3.0])

    sample_a = {
        "id": "sA",
        "history": [
            {"role": "user", "text": None, "audio": {"array": wave_a, "sampling_rate": 16000}},
            {"role": "assistant", "text": "ok", "audio": None},
            {"role": "user", "text": None, "audio": {"array": wave_b, "sampling_rate": 8000}},
        ],
        "assistant_text": "reply a",
    }
    sample_b = {
        "id": "sB",
        "history": [
            {"role": "user", "text": None, "audio": {"array": wave_c, "sampling_rate": 16000}},
        ],
        "assistant_text": "reply b",
    }

    batch = collator([sample_a, sample_b])

    assert torch.equal(batch["speech_waveforms"][0], wave_a)
    assert torch.equal(batch["speech_waveforms"][1], wave_b)
    assert torch.equal(batch["speech_waveforms"][2], wave_c)
    assert batch["speech_sampling_rate"] == [16000, 8000, 16000]

    speech_tokens = (batch["input_ids"] == token_ids.speech_id).sum().item()
    assert speech_tokens == len(batch["speech_waveforms"])


def test_collator_raises_on_speech_token_mismatch(local_base_model_id):
    # Collator should raise if <speech> tokens exceed provided waveforms.
    tokenizer, token_ids = _build_test_tokenizer_and_ids(local_base_model_id)
    collator = SuCollator(tokenizer, token_ids)

    sample = {
        "id": "s-mismatch",
        "history": [
            {"role": "user", "text": None, "audio": {"array": torch.tensor([1.0]), "sampling_rate": 16000}},
        ],
        # Inject a literal <speech> token into the assistant reply.
        "assistant_text": f"ok {DEFAULT_SPEECH_TOKEN}",
    }

    with pytest.raises(ValueError, match="speech tokens"):
        collator([sample])


def test_collator_raises_on_tokenizer_token_ids_mismatch(local_base_model_id):
    # Collator should fail fast if token ids do not match the provided tokenizer.
    tokenizer, token_ids = _build_test_tokenizer_and_ids(local_base_model_id)
    bad_token_ids = TokenIds(speech_id=token_ids.speech_id + 1, sep_id=token_ids.sep_id)

    with pytest.raises(ValueError, match="does not match provided token_ids.speech_id"):
        SuCollator(tokenizer, bad_token_ids)


def test_resolve_token_ids_missing_speech_token_raises(local_base_model_id):
    # Shared token-id resolver should fail fast if required tokens are missing.
    tokenizer = AutoTokenizer.from_pretrained(local_base_model_id)

    with pytest.raises(ValueError, match="<speech>"):
        resolve_token_ids(tokenizer)


def test_resolve_token_ids_missing_sep_token_raises(local_base_model_id):
    # Shared token-id resolver should fail fast when <sep> is not registered.
    tokenizer = AutoTokenizer.from_pretrained(local_base_model_id)
    tokenizer.add_special_tokens({"additional_special_tokens": [DEFAULT_SPEECH_TOKEN]})

    with pytest.raises(ValueError, match="<sep>"):
        resolve_token_ids(tokenizer)


def test_su_dataset_start_shard_cursor_skips_earlier_shards(tmp_path, fixture_audio_paths):
    # start_shard_cursor should begin iteration at that cursor for shard directories.
    audio_root = tmp_path
    _copy_fixture_audio(fixture_audio_paths[0], audio_root / "audio" / "a.wav")
    shard_dir = Path(__file__).resolve().parents[1] / "fixtures" / "su" / "two_shards"
    assert shard_dir.is_dir()

    dataset = SuDataset(
        json_path=str(shard_dir),
        audio_root=str(audio_root),
        start_shard_cursor=1,
    )
    samples = list(dataset)

    # Only the second fixture shard should be visible when start_shard_cursor=1.
    assert len(samples) == 1
    assert samples[0]["assistant_text"] == "a1"
    assert samples[0]["shard_index"] == 1
    assert samples[0]["shard_cursor"] == 1


def test_su_dataset_start_shard_cursor_rejected_for_single_file(
    tmp_path,
    fixture_audio_paths,
):
    # start_shard_cursor should fail on single-file inputs because shard directories are required.
    audio_root = tmp_path
    _copy_fixture_audio(fixture_audio_paths[0], audio_root / "audio" / "a.wav")

    record = {
        "id": "conv-file-shard-arg",
        "conversations": [
            {"from": "user", "value": "u1", "audio": "audio/a.wav"},
            {"from": "assistant", "value": "a1"},
        ],
    }
    json_path = tmp_path / "single.json"
    _write_json_list(json_path, [record])

    dataset = SuDataset(
        json_path=str(json_path),
        audio_root=str(audio_root),
        start_shard_cursor=1,
    )
    with pytest.raises(ValueError, match="start_shard_cursor is only valid"):
        list(dataset)


def test_su_dataset_shard_shuffle_is_deterministic_per_epoch(tmp_path, fixture_audio_paths):
    # Shuffled shard order should be deterministic per epoch and change across epochs.
    audio_root = tmp_path
    _copy_fixture_audio(fixture_audio_paths[0], audio_root / "audio" / "a.wav")

    shard_dir = Path(__file__).resolve().parents[1] / "fixtures" / "su" / "three_shards"
    assert shard_dir.is_dir()

    def _collect_shard_indices(epoch: int) -> list[int]:
        dataset = SuDataset(
            json_path=str(shard_dir),
            audio_root=str(audio_root),
            shuffle_shards=True,
            shuffle_seed=123,
            shuffle_epoch=epoch,
        )
        return [sample["shard_index"] for sample in dataset]

    order_epoch0_a = _collect_shard_indices(epoch=0)
    order_epoch0_b = _collect_shard_indices(epoch=0)
    order_epoch1 = _collect_shard_indices(epoch=1)

    assert len(order_epoch0_a) == 5
    assert order_epoch0_a == order_epoch0_b
    assert order_epoch0_a != order_epoch1
