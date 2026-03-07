"""S2S (Stage-2) dataset and collator utilities."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Dict, Iterator, Optional

import torch
import torchaudio
from torch.utils.data import IterableDataset, get_worker_info

from model.constants import DEFAULT_SPEECH_TOKEN, IGNORE_INDEX, SPEECH_VOCAB_SIZE
from model.tokenizers import TokenIds
from train.s2s_sequence_builder import s2s_passes_read_write_ratio_filter
from train.su_sequence_builder import build_su_messages, build_su_token_ids

MIN_UNIT_ID = 0
MAX_UNIT_ID = SPEECH_VOCAB_SIZE - 1


def _iter_jsonl_records(path: Path) -> Iterator[Dict[str, Any]]:
    """Yield JSON object records from one JSONL file."""
    with path.open("r", encoding="utf-8") as handle:
        for lineno, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if not isinstance(record, dict):
                raise ValueError(f"{path}: line {lineno}: expected a JSON object record")
            yield record


def _list_shard_paths(path: Path) -> list[Path]:
    """List shard files in deterministic order."""
    if not path.is_dir():
        raise FileNotFoundError(f"json_path not found: {path}")
    shard_paths = sorted(path.glob("shard-*.jsonl"))
    if not shard_paths:
        raise FileNotFoundError(f"no shard-*.jsonl files found in: {path}")
    return shard_paths


def _validate_manifest_schema(json_path: Path) -> None:
    """Validate S2S shard manifest schema/version before dataset iteration."""
    if not json_path.is_dir():
        return
    manifest_path = json_path / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"manifest.json not found in S2S shard directory: {json_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise ValueError(f"manifest.json must be a JSON object: {manifest_path}")
    schema = manifest.get("schema")
    if schema != "s2s_pairs_v1":
        raise ValueError("manifest schema must be s2s_pairs_v1")


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


def _worker_shard_indices() -> tuple[int, int]:
    """Return (worker_id, worker_count) for worker-only sharding."""
    worker = get_worker_info()
    worker_id = 0 if worker is None else int(worker.id)
    num_workers = 1 if worker is None else int(worker.num_workers)
    # Each worker keeps only its modulo slice to avoid duplicate samples.
    return worker_id, num_workers


def _resolve_audio_path(path: str, audio_root: Path) -> Path:
    """Resolve an audio path against the configured audio root."""
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


def _normalize_unit_ids(unit_ids: Any, *, sample_id: str) -> list[int]:
    """Normalize and validate one sample's parsed unit ids."""
    if not isinstance(unit_ids, list) or not unit_ids:
        raise ValueError(f"sample {sample_id}: unit_ids must be a non-empty list")
    normalized: list[int] = []
    for unit_id in unit_ids:
        try:
            value = int(unit_id)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"sample {sample_id}: unit_ids must contain integers") from exc
        if value < MIN_UNIT_ID or value > MAX_UNIT_ID:
            raise ValueError(
                f"sample {sample_id}: unit id {value} out of range [{MIN_UNIT_ID}, {MAX_UNIT_ID}]"
            )
        normalized.append(value)
    return normalized


class S2sDataset(IterableDataset):
    """Iterable dataset that yields S2S samples with history audio loaded."""

    def __init__(
        self,
        *,
        json_path: str,
        audio_root: str,
        start_shard_cursor: int = 0,
        shuffle_shards: bool = False,
        shuffle_seed: Optional[int] = None,
        shuffle_epoch: int = 0,
        tokenizer: Optional[Any] = None,
        read_length: Optional[int] = None,
        write_length: Optional[int] = None,
    ) -> None:
        self.json_path = Path(json_path)
        self.audio_root = Path(audio_root)
        self.start_shard_cursor = int(start_shard_cursor)
        self.shuffle_shards = bool(shuffle_shards)
        self.shuffle_seed = shuffle_seed
        self.shuffle_epoch = int(shuffle_epoch)
        self.tokenizer = tokenizer
        self.read_length = int(read_length) if read_length is not None else None
        self.write_length = int(write_length) if write_length is not None else None
        if (self.read_length is None) != (self.write_length is None):
            raise ValueError("read_length and write_length must be set together")
        # Fail fast if shard schema/version mismatches Stage-2 runtime expectations.
        _validate_manifest_schema(self.json_path)

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
            for index, record in enumerate(_iter_jsonl_records(record_file)):
                # Shard before any audio I/O to avoid duplicated decode work.
                if shard_count > 1 and index % shard_count != worker_id:
                    continue

                sample_id = str(record.get("id", ""))
                assistant_text = record.get("assistant_text")
                if not isinstance(assistant_text, str) or not assistant_text.strip():
                    raise ValueError(f"sample {sample_id}: assistant_text is missing or empty")
                unit_ids = _normalize_unit_ids(record.get("unit_ids"), sample_id=sample_id)
                history = record.get("history")
                if not isinstance(history, list):
                    raise ValueError(f"sample {sample_id}: history must be a list")

                # Optional runtime ratio filtering when tokenizer and R/W are configured.
                if (
                    self.tokenizer is not None
                    and self.read_length is not None
                    and self.write_length is not None
                ):
                    content_ids = self.tokenizer(
                        assistant_text,
                        add_special_tokens=False,
                    )["input_ids"]
                    if not content_ids:
                        continue
                    if not s2s_passes_read_write_ratio_filter(
                        content_token_count=len(content_ids),
                        unit_token_count=len(unit_ids),
                        read_length=self.read_length,
                        write_length=self.write_length,
                    ):
                        continue

                loaded_history: list[Dict[str, Any]] = []
                for turn_index, turn in enumerate(history):
                    if not isinstance(turn, dict):
                        raise ValueError(f"sample {sample_id}: history turn {turn_index} must be an object")
                    role = str(turn.get("role", ""))
                    if role == "user":
                        audio_spec = turn.get("audio")
                        if not isinstance(audio_spec, dict):
                            raise ValueError(f"sample {sample_id}: user turn {turn_index} is missing audio")
                        raw_path = audio_spec.get("path")
                        if not isinstance(raw_path, str) or not raw_path:
                            raise ValueError(
                                f"sample {sample_id}: user turn {turn_index} audio.path must be a non-empty string"
                            )
                        audio_path = _resolve_audio_path(raw_path, self.audio_root)
                        waveform, sampling_rate = _load_audio(audio_path)
                        loaded_history.append(
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
                    elif role == "assistant":
                        text = turn.get("text")
                        if not isinstance(text, str) or not text.strip():
                            raise ValueError(
                                f"sample {sample_id}: assistant turn {turn_index} text must be a non-empty string"
                            )
                        loaded_history.append({"role": "assistant", "text": text, "audio": None})
                    else:
                        raise ValueError(f"sample {sample_id}: unknown history role {role!r}")

                yield {
                    "id": sample_id,
                    "source_id": str(record.get("source_id", "")),
                    "turn_index": int(record.get("turn_index", 0)),
                    "history": loaded_history,
                    "assistant_text": assistant_text,
                    "unit_ids": unit_ids,
                    "shard_index": shard_index,
                    "shard_cursor": shard_cursor,
                    "worker_id": worker_id,
                }


class S2sCollator:
    """Collate S2S samples into Thinker+Talker training inputs."""

    def __init__(
        self,
        tokenizer,
        *,
        token_ids: TokenIds,
        read_length: int,
        write_length: int,
        ignore_index: int = IGNORE_INDEX,
    ) -> None:
        self.tokenizer = tokenizer
        self.read_length = int(read_length)
        self.write_length = int(write_length)
        self.ignore_index = int(ignore_index)
        if self.read_length <= 0 or self.write_length <= 0:
            raise ValueError("read_length and write_length must be > 0")

        # Keep <speech> id fixed from startup token contract.
        speech_id = tokenizer.convert_tokens_to_ids(DEFAULT_SPEECH_TOKEN)
        if speech_id != int(token_ids.speech_id):
            raise ValueError("tokenizer <speech> id does not match provided token_ids.speech_id")
        self.speech_token_id = int(token_ids.speech_id)

        if tokenizer.pad_token_id is None:
            raise ValueError("tokenizer must define pad_token_id")

    def __call__(self, batch: list[Dict[str, Any]]) -> Dict[str, Any]:
        """Tokenize S2S samples and build padded Thinker inputs plus Talker targets."""
        input_ids_list: list[list[int]] = []
        labels_list: list[list[int]] = []
        content_ids_list: list[torch.Tensor] = []
        unit_ids_list: list[torch.Tensor] = []
        sample_ids: list[str] = []
        shard_indices: list[int] = []
        shard_cursors: list[int] = []
        worker_ids: list[int] = []
        waveforms: list[torch.Tensor] = []
        sampling_rates: list[int] = []

        for sample in batch:
            sample_id = str(sample.get("id", ""))
            assistant_text = sample.get("assistant_text")
            if not isinstance(assistant_text, str) or not assistant_text.strip():
                raise ValueError(f"sample {sample_id}: assistant_text is missing or empty")
            unit_ids = _normalize_unit_ids(sample.get("unit_ids"), sample_id=sample_id)

            # Tokenize plain assistant text for gate-fusion alignment.
            content_ids = self.tokenizer(
                assistant_text,
                add_special_tokens=False,
            )["input_ids"]
            if not content_ids:
                raise ValueError(f"sample {sample_id}: assistant_text tokenized to empty content_ids")

            # Build SU-style prompt/reply sequence for frozen Thinker teacher forcing.
            messages, sample_waves, sample_rates = build_su_messages(sample["history"])
            wave_start = len(waveforms)
            waveforms.extend(sample_waves)
            sampling_rates.extend(sample_rates)

            prompt_ids, reply_ids = build_su_token_ids(
                self.tokenizer,
                messages,
                assistant_text=assistant_text,
            )
            if reply_ids is None:
                raise ValueError(f"sample {sample_id}: assistant reply ids were not generated")

            input_ids = prompt_ids + reply_ids
            labels = [self.ignore_index] * len(prompt_ids) + reply_ids

            # Ensure <speech> sentinels and waveform segments stay perfectly aligned.
            expected_segments = len(waveforms) - wave_start
            speech_tokens = sum(1 for token in input_ids if token == self.speech_token_id)
            if speech_tokens != expected_segments:
                raise ValueError(
                    f"sample {sample_id}: speech tokens {speech_tokens} != waveforms {expected_segments}"
                )

            input_ids_list.append(input_ids)
            labels_list.append(labels)
            content_ids_list.append(torch.tensor(content_ids, dtype=torch.long))
            # Collator validates ratio contract; dataset owns filtering/drop behavior.
            if not s2s_passes_read_write_ratio_filter(
                content_token_count=len(content_ids),
                unit_token_count=len(unit_ids),
                read_length=self.read_length,
                write_length=self.write_length,
            ):
                raise ValueError(
                    f"sample {sample_id}: content/unit lengths fail Stage-2 Read/Write ratio "
                    f"(R={self.read_length}, W={self.write_length}); dataset filtering is misconfigured"
                )

            unit_ids_list.append(torch.tensor(unit_ids, dtype=torch.long))
            sample_ids.append(sample_id)
            shard_indices.append(int(sample.get("shard_index", 0)))
            shard_cursors.append(int(sample.get("shard_cursor", 0)))
            worker_ids.append(int(sample.get("worker_id", 0)))

        if not input_ids_list:
            raise ValueError("batch is empty")

        # Pad to max length across the validated batch.
        max_len = max(len(ids) for ids in input_ids_list)
        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            raise ValueError("tokenizer must define pad_token_id")

        batch_input_ids = torch.full((len(input_ids_list), max_len), pad_id, dtype=torch.long)
        batch_labels = torch.full((len(input_ids_list), max_len), self.ignore_index, dtype=torch.long)
        batch_attention = torch.zeros((len(input_ids_list), max_len), dtype=torch.long)

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
            "content_ids": content_ids_list,
            "unit_ids": unit_ids_list,
            "sample_ids": sample_ids,
            "shard_indices": shard_indices,
            "shard_cursors": shard_cursors,
            "worker_ids": worker_ids,
        }


__all__ = ["S2sDataset", "S2sCollator"]
