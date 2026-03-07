"""SU (Speech Understanding) dataset utilities."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Dict, Iterator, Optional

import torch
import torchaudio
from torch.utils.data import IterableDataset, get_worker_info

from model.constants import DEFAULT_SPEECH_TOKEN, IGNORE_INDEX
from train.su_sequence_builder import build_su_messages, build_su_token_ids
from model.tokenizers import TokenIds

def _first_nonspace(path: Path) -> str:
    """Peek at the first non-space character to detect JSON vs JSONL."""
    with path.open("r", encoding="utf-8") as handle:
        chunk = handle.read(4096)
    for ch in chunk:
        if not ch.isspace():
            return ch
    return ""


def _iter_records(path: Path) -> Iterator[Dict[str, Any]]:
    """Yield JSON records from a list file or JSONL stream."""
    first = _first_nonspace(path)
    if first == "[":
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
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


def _list_shard_paths(path: Path) -> list[Path]:
    """List shard files in deterministic order."""
    if not path.is_dir():
        raise FileNotFoundError(f"json_path not found: {path}")
    shard_paths = sorted(path.glob("shard-*.jsonl"))
    if not shard_paths:
        raise FileNotFoundError(f"no shard-*.jsonl files found in: {path}")
    return shard_paths


def _build_shard_order(
    shard_count: int,
    *,
    shuffle_shards: bool,
    shuffle_seed: Optional[int],
    shuffle_epoch: int,
) -> list[int]:
    """Build the per-epoch shard order (optionally shuffled)."""
    order = list(range(shard_count))
    if shuffle_shards:
        if shuffle_seed is None:
            raise ValueError("shuffle_seed must be set when shuffle_shards is enabled")
        rng = random.Random(int(shuffle_seed) + int(shuffle_epoch))
        rng.shuffle(order)
    return order


def _iter_record_files(
    path: Path,
    *,
    start_shard_cursor: int = 0,
    shuffle_shards: bool = False,
    shuffle_seed: Optional[int] = None,
    shuffle_epoch: int = 0,
) -> Iterator[tuple[int, int, Path]]:
    """Yield record files as (cursor, shard_index, path)."""
    if path.is_file():
        if shuffle_shards:
            raise ValueError("shuffle_shards requires a shard directory, not a single file")
        if start_shard_cursor != 0:
            raise ValueError("start_shard_cursor is only valid when json_path is a shard directory")
        yield 0, 0, path
        return

    shard_paths = _list_shard_paths(path)
    if start_shard_cursor < 0:
        raise ValueError("start_shard_cursor must be >= 0")
    if start_shard_cursor >= len(shard_paths):
        raise ValueError(
            f"start_shard_cursor {start_shard_cursor} is out of range for {len(shard_paths)} shards"
        )

    order = _build_shard_order(
        len(shard_paths),
        shuffle_shards=shuffle_shards,
        shuffle_seed=shuffle_seed,
        shuffle_epoch=shuffle_epoch,
    )
    # Resume continues from the requested shard cursor in the shuffled order.
    for cursor in range(start_shard_cursor, len(order)):
        shard_index = order[cursor]
        yield cursor, shard_index, shard_paths[shard_index]


def _get_turns(record: Dict[str, Any]) -> list[Dict[str, Any]]:
    """Extract the turn list from EchoX or InstructS2S records."""
    if "conversations" in record:
        turns = record["conversations"]
    else:
        turns = record.get("conversation")
    if not isinstance(turns, list):
        raise ValueError("record is missing a conversation list")
    return turns


def _role_from_turn(turn: Dict[str, Any]) -> str:
    """Normalize dataset roles to 'user' or 'assistant'."""
    raw = str(turn.get("from", "")).lower()
    if raw in {"user", "human"}:
        return "user"
    if raw in {"assistant", "gpt"}:
        return "assistant"
    raise ValueError(f"unknown role: {raw}")


def _turn_text(turn: Dict[str, Any]) -> Optional[str]:
    """Get the text field for a turn if present."""
    text = turn.get("value", None)
    if text is None:
        text = turn.get("text", None)
    if text is None:
        return None
    return str(text)


def _turn_audio(turn: Dict[str, Any]) -> Optional[str]:
    """Get the audio path field for a turn if present."""
    audio = turn.get("audio", None)
    if audio is None:
        audio = turn.get("speech", None)
    if audio is None:
        return None
    if isinstance(audio, dict):
        # SU assumes path-based audio; embedded arrays are not supported yet.
        raise ValueError("embedded audio arrays are not supported; provide a file path")
    return str(audio)


def _turn_wer(turn: Dict[str, Any]) -> Optional[float]:
    """Read WER if it exists (EchoX only)."""
    wer = turn.get("wer", None)
    if wer is None:
        return None
    try:
        return float(wer)
    except (TypeError, ValueError):
        return None


def _iter_samples(
    records: Iterator[Dict[str, Any]],
    *,
    max_wer: Optional[float],
) -> Iterator[Dict[str, Any]]:
    """Yield normalized SU samples from raw records."""
    for record_index, record in enumerate(records):
        conv_id = str(record.get("id", f"conv-{record_index}"))
        turns = _get_turns(record)

        history: list[Dict[str, Any]] = []
        pending_user: Optional[Dict[str, Any]] = None
        pending_wer: Optional[float] = None
        pair_index = 0

        for turn in turns:
            role = _role_from_turn(turn)
            if role == "user":
                audio = _turn_audio(turn)
                wer = _turn_wer(turn)
                too_noisy = max_wer is not None and wer is not None and wer > max_wer
                if audio is None or too_noisy:
                    # Missing or noisy audio terminates the conversation.
                    break

                pending_user = {
                    "role": "user",
                    "text": None,
                    "audio": {"path": audio},
                }
                pending_wer = wer
                continue

            if role == "assistant":
                if pending_user is None:
                    continue
                assistant_text = _turn_text(turn)
                if not assistant_text:
                    pending_user = None
                    pending_wer = None
                    continue

                # Emit one sample per user -> next assistant pair.
                sample_id = f"{conv_id}-{pair_index}"
                pair_index += 1
                sample = {
                    "id": sample_id,
                    "history": history + [pending_user],
                    "assistant_text": assistant_text,
                    "wer": pending_wer,
                }
                yield sample

                # Extend history with the resolved user/assistant pair.
                history = history + [
                    pending_user,
                    {"role": "assistant", "text": assistant_text, "audio": None},
                ]
                pending_user = None
                pending_wer = None


def _resolve_audio_path(path: str, audio_root: Path) -> Path:
    """Resolve an audio path against the explicit audio root."""
    audio_path = Path(path)
    if audio_path.is_absolute():
        return audio_path
    return audio_root / audio_path


def _load_audio(path: Path) -> tuple[torch.Tensor, int]:
    """Load audio as mono float32 without resampling."""
    waveform, sampling_rate = torchaudio.load(str(path))
    if waveform.ndim != 2:
        raise ValueError("torchaudio returned unexpected shape")
    if waveform.size(0) > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    waveform = waveform.squeeze(0).to(dtype=torch.float32)
    return waveform, int(sampling_rate)


def _worker_shard_indices() -> tuple[int, int]:
    """Return (shard_id, shard_count) for worker-only sharding."""
    worker = get_worker_info()
    worker_id = 0 if worker is None else int(worker.id)
    num_workers = 1 if worker is None else int(worker.num_workers)

    # Each worker gets a unique modulo shard.
    return worker_id, num_workers


class SuDataset(IterableDataset):
    """Iterable dataset that yields SU samples with audio loaded."""

    def __init__(
        self,
        *,
        json_path: str,
        audio_root: str,
        max_wer: Optional[float] = None,
        start_shard_cursor: int = 0,
        shuffle_shards: bool = False,
        shuffle_seed: Optional[int] = None,
        shuffle_epoch: int = 0,
    ) -> None:
        self.json_path = Path(json_path)
        self.audio_root = Path(audio_root)
        self.max_wer = max_wer
        self.start_shard_cursor = int(start_shard_cursor)
        self.shuffle_shards = bool(shuffle_shards)
        self.shuffle_seed = shuffle_seed
        self.shuffle_epoch = int(shuffle_epoch)

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        if not self.audio_root.exists():
            raise FileNotFoundError(f"audio_root not found: {self.audio_root}")

        worker_id, shard_count = _worker_shard_indices()
        for shard_cursor, shard_index, record_file in _iter_record_files(
            self.json_path,
            start_shard_cursor=self.start_shard_cursor,
            shuffle_shards=self.shuffle_shards,
            shuffle_seed=self.shuffle_seed,
            shuffle_epoch=self.shuffle_epoch,
        ):
            records = _iter_records(record_file)
            samples = _iter_samples(records, max_wer=self.max_wer)

            for index, sample in enumerate(samples):
                # Shard before any audio I/O to avoid duplicated work.
                if shard_count > 1 and index % shard_count != worker_id:
                    continue
                # Clone history so this sample is independent of others.
                new_history: list[Dict[str, Any]] = []
                for turn in sample["history"]:
                    if turn["role"] == "user":
                        audio_spec = turn["audio"]
                        audio_path = _resolve_audio_path(audio_spec["path"], self.audio_root)
                        waveform, sampling_rate = _load_audio(audio_path)
                        new_history.append(
                            {
                                "role": "user",
                                "text": None,
                                "audio": {
                                    "path": str(audio_path),
                                    "array": waveform,
                                    "sampling_rate": sampling_rate,
                                },
                            }
                        )
                    else:
                        new_history.append(
                            {
                                "role": "assistant",
                                "text": turn["text"],
                                "audio": None,
                            }
                        )

                yield {
                    "id": sample["id"],
                    "history": new_history,
                    "assistant_text": sample["assistant_text"],
                    "wer": sample.get("wer"),
                    "shard_index": shard_index,
                    "shard_cursor": shard_cursor,
                    "worker_id": worker_id,
                }


class SuCollator:
    """Collate SU samples into model-ready tensors."""

    def __init__(self, tokenizer, token_ids: TokenIds) -> None:
        # Store tokenizer + validated token ids from startup contract.
        self.tokenizer = tokenizer
        # Fail fast if token ids come from a different tokenizer instance.
        speech_id = tokenizer.convert_tokens_to_ids(DEFAULT_SPEECH_TOKEN)
        if speech_id != int(token_ids.speech_id):
            raise ValueError("tokenizer <speech> id does not match provided token_ids.speech_id")
        self.speech_token_id = int(token_ids.speech_id)

    def __call__(self, batch: list[Dict[str, Any]]) -> Dict[str, Any]:
        """Build chat prompts, labels, and aligned speech waveforms."""
        input_ids_list: list[list[int]] = []
        labels_list: list[list[int]] = []
        sample_ids: list[str] = []
        shard_indices: list[int] = []
        shard_cursors: list[int] = []
        worker_ids: list[int] = []
        waveforms: list[torch.Tensor] = []
        sampling_rates: list[int] = []

        for sample in batch:
            sample_ids.append(sample.get("id", ""))
            shard_indices.append(int(sample.get("shard_index", 0)))
            shard_cursors.append(int(sample.get("shard_cursor", 0)))
            worker_ids.append(int(sample.get("worker_id", 0)))

            # Build chat messages and collect speech waveforms in order.
            messages, sample_waves, sample_rates = build_su_messages(sample["history"])
            wave_start = len(waveforms)
            waveforms.extend(sample_waves)
            sampling_rates.extend(sample_rates)

            # Prompt ends with assistant role; reply ids include template turn-end tokens.
            prompt_ids, reply_ids = build_su_token_ids(
                self.tokenizer,
                messages,
                assistant_text=sample["assistant_text"],
            )
            if reply_ids is None:
                raise ValueError("assistant_text is required to build reply token ids")

            input_ids = prompt_ids + reply_ids
            labels = [IGNORE_INDEX] * len(prompt_ids) + reply_ids

            # Guardrail: speech token count must match collected waveforms.
            expected_segments = len(waveforms) - wave_start
            speech_tokens = sum(1 for token in input_ids if token == self.speech_token_id)
            if speech_tokens != expected_segments:
                raise ValueError(
                    f"sample {sample_ids[-1]}: speech tokens {speech_tokens} != waveforms {expected_segments}"
                )

            input_ids_list.append(input_ids)
            labels_list.append(labels)

        # Pad to max length across the batch.
        max_len = max(len(ids) for ids in input_ids_list) if input_ids_list else 0
        pad_id = self.tokenizer.pad_token_id
        # Pad positions must use a neutral pad token, never EOS/turn-end.
        if pad_id is None:
            raise ValueError("tokenizer must define pad_token_id")

        batch_input_ids = torch.full((len(batch), max_len), pad_id, dtype=torch.long)
        batch_labels = torch.full((len(batch), max_len), IGNORE_INDEX, dtype=torch.long)
        batch_attention = torch.zeros((len(batch), max_len), dtype=torch.long)

        for idx, (ids, labels) in enumerate(zip(input_ids_list, labels_list)):
            cur_len = len(ids)
            batch_input_ids[idx, :cur_len] = torch.tensor(ids, dtype=torch.long)
            batch_labels[idx, :cur_len] = torch.tensor(labels, dtype=torch.long)
            batch_attention[idx, :cur_len] = 1

        if len(waveforms) != len(sampling_rates):
            raise ValueError("speech_waveforms and speech_sampling_rate length mismatch")

        return {
            "input_ids": batch_input_ids,
            "attention_mask": batch_attention,
            "labels": batch_labels,
            "speech_waveforms": waveforms,
            "speech_sampling_rate": sampling_rates,
            "sample_ids": sample_ids,
            "shard_indices": shard_indices,
            "shard_cursors": shard_cursors,
            "worker_ids": worker_ids,
        }
