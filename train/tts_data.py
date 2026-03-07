"""TTS (Text-to-Speech) dataset + collator for PomTTS pretraining."""

from __future__ import annotations

import json
import random
from pathlib import Path
from math import ceil
from typing import Any, Dict, Iterator, Optional

import torch
from torch.utils.data import IterableDataset, get_worker_info

from model.constants import DEFAULT_SEP_TOKEN, IGNORE_INDEX
from model.tokenizers import TokenIds
from train.tts_sequence_builder import build_read_write_sequence


def _iter_jsonl_records(path: Path) -> Iterator[Dict[str, Any]]:
    """Yield JSONL records from a shard file."""
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
    """Return (shard_id, shard_count) for worker-only sharding."""
    worker = get_worker_info()
    worker_id = 0 if worker is None else int(worker.id)
    num_workers = 1 if worker is None else int(worker.num_workers)
    # Each worker gets a unique modulo shard.
    return worker_id, num_workers


def _validate_manifest_policies(json_path: Path) -> None:
    """Validate shard manifest boundary/stop policies if present."""
    if not json_path.is_dir():
        return
    manifest_path = json_path / "manifest.json"
    if not manifest_path.exists():
        return
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    boundary_policy = manifest.get("boundary_token_policy")
    stop_policy = manifest.get("stop_token_policy")
    if boundary_policy != "add_sep_in_stage1b_data_pipeline":
        raise ValueError("manifest boundary_token_policy must be add_sep_in_stage1b_data_pipeline")
    if stop_policy != "add_eos_in_stage1b_data_pipeline":
        raise ValueError("manifest stop_token_policy must be add_eos_in_stage1b_data_pipeline")


class TtsDataset(IterableDataset):
    """Iterable dataset that yields TTS text->speech-unit samples."""

    def __init__(
        self,
        *,
        json_path: str,
        start_shard_cursor: int = 0,
        shuffle_shards: bool = False,
        shuffle_seed: Optional[int] = None,
        shuffle_epoch: int = 0,
        max_seq_len: Optional[int] = None,
        tokenizer: Optional[Any] = None,
        read_length: Optional[int] = None,
        write_length: Optional[int] = None,
    ) -> None:
        self.json_path = Path(json_path)
        self.start_shard_cursor = int(start_shard_cursor)
        self.shuffle_shards = bool(shuffle_shards)
        self.shuffle_seed = shuffle_seed
        self.shuffle_epoch = int(shuffle_epoch)
        self.max_seq_len = max_seq_len
        self.tokenizer = tokenizer
        self.read_length = int(read_length) if read_length is not None else None
        self.write_length = int(write_length) if write_length is not None else None
        # Fail fast if manifest policy mismatches training semantics.
        _validate_manifest_policies(self.json_path)

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        worker_id, shard_count = _worker_shard_indices()

        for shard_cursor, shard_index, record_file in _iter_record_files(
            self.json_path,
            start_shard_cursor=self.start_shard_cursor,
            shuffle_shards=self.shuffle_shards,
            shuffle_seed=self.shuffle_seed,
            shuffle_epoch=self.shuffle_epoch,
        ):
            for index, record in enumerate(_iter_jsonl_records(record_file)):
                # Shard before any heavy work to avoid duplicates across workers.
                if shard_count > 1 and index % shard_count != worker_id:
                    continue

                sample_id = str(record.get("id", ""))
                source_id = str(record.get("source_id", ""))
                turn_index = int(record.get("turn_index", 0))
                assistant_text = record.get("assistant_text")
                unit_ids = record.get("unit_ids")
                if not isinstance(assistant_text, str) or not assistant_text.strip():
                    raise ValueError(f"sample {sample_id}: assistant_text is missing or empty")
                if not isinstance(unit_ids, list) or not unit_ids:
                    raise ValueError(f"sample {sample_id}: unit_ids must be a non-empty list")

                if self.tokenizer is not None and (
                    self.max_seq_len is not None
                    or self.read_length is not None
                ):
                    text_ids = self.tokenizer(
                        assistant_text, add_special_tokens=False
                    )["input_ids"]

                    if self.max_seq_len is not None:
                        exact_seq_len = len(text_ids) + len(unit_ids) + 2
                        if exact_seq_len > self.max_seq_len:
                            continue

                    if self.read_length is not None and self.write_length is not None:
                        max_text = self.read_length * ceil(len(unit_ids) / self.write_length)
                        if len(text_ids) > max_text:
                            continue

                elif self.max_seq_len is not None:
                    est_seq_len = len(assistant_text) // 4 + len(unit_ids) + 2
                    if est_seq_len > self.max_seq_len:
                        continue

                yield {
                    "id": sample_id,
                    "source_id": source_id,
                    "turn_index": turn_index,
                    "assistant_text": assistant_text,
                    "unit_ids": unit_ids,
                    "shard_index": shard_index,
                    "shard_cursor": shard_cursor,
                    "worker_id": worker_id,
                }


class TtsCollator:
    """Collate TTS samples into model-ready tensors."""

    def __init__(
        self,
        tokenizer,
        *,
        token_ids: TokenIds,
        speech_token_offset: int,
        read_length: int,
        write_length: int,
        ignore_index: int = IGNORE_INDEX,
    ) -> None:
        self.tokenizer = tokenizer
        # Keep <sep> id fixed from startup token contract.
        self.sep_id = int(token_ids.sep_id)
        self.speech_token_offset = int(speech_token_offset)
        self.read_length = int(read_length)
        self.write_length = int(write_length)
        self.ignore_index = int(ignore_index)

        if self.tokenizer.pad_token_id is None:
            raise ValueError("tokenizer must define pad_token_id")
        if self.tokenizer.eos_token_id is None:
            raise ValueError("tokenizer must define eos_token_id")
        expected_sep_id = self.tokenizer.convert_tokens_to_ids(DEFAULT_SEP_TOKEN)
        if expected_sep_id != self.sep_id:
            raise ValueError("tokenizer <sep> id does not match provided token_ids.sep_id")
        expected_offset = len(self.tokenizer)
        if self.speech_token_offset != expected_offset:
            raise ValueError("speech_token_offset must match tokenizer length")

    def __call__(self, batch: list[Dict[str, Any]]) -> Dict[str, Any]:
        """Tokenize text and build Read/Write sequences with padding."""
        input_ids_list: list[list[int]] = []
        labels_list: list[list[int]] = []
        sample_ids: list[str] = []
        shard_indices: list[int] = []
        shard_cursors: list[int] = []
        worker_ids: list[int] = []

        for sample in batch:
            sample_ids.append(str(sample.get("id", "")))
            shard_indices.append(int(sample.get("shard_index", 0)))
            shard_cursors.append(int(sample.get("shard_cursor", 0)))
            worker_ids.append(int(sample.get("worker_id", 0)))

            assistant_text = sample.get("assistant_text")
            unit_ids = sample.get("unit_ids")
            if not isinstance(assistant_text, str) or not assistant_text.strip():
                raise ValueError(f"sample {sample_ids[-1]}: assistant_text is missing or empty")
            if not isinstance(unit_ids, list) or not unit_ids:
                raise ValueError(f"sample {sample_ids[-1]}: unit_ids must be a non-empty list")

            # TTS conditions on plain assistant text without chat template tokens.
            text_ids = self.tokenizer(
                assistant_text,
                add_special_tokens=False,
            )["input_ids"]

            input_ids, labels = build_read_write_sequence(
                text_ids,
                unit_ids,
                speech_token_offset=self.speech_token_offset,
                sep_id=self.sep_id,
                eos_id=self.tokenizer.eos_token_id,
                read_length=self.read_length,
                write_length=self.write_length,
                ignore_index=self.ignore_index,
            )
            input_ids_list.append(input_ids)
            labels_list.append(labels)

        # Pad to max length across the batch.
        max_len = max(len(ids) for ids in input_ids_list) if input_ids_list else 0
        pad_id = self.tokenizer.pad_token_id
        batch_input_ids = torch.full((len(batch), max_len), pad_id, dtype=torch.long)
        batch_labels = torch.full((len(batch), max_len), self.ignore_index, dtype=torch.long)
        batch_attention = torch.zeros((len(batch), max_len), dtype=torch.long)

        for idx, (ids, labels) in enumerate(zip(input_ids_list, labels_list)):
            cur_len = len(ids)
            batch_input_ids[idx, :cur_len] = torch.tensor(ids, dtype=torch.long)
            batch_labels[idx, :cur_len] = torch.tensor(labels, dtype=torch.long)
            batch_attention[idx, :cur_len] = 1

        return {
            "input_ids": batch_input_ids,
            "attention_mask": batch_attention,
            "labels": batch_labels,
            "sample_ids": sample_ids,
            "shard_indices": shard_indices,
            "shard_cursors": shard_cursors,
            "worker_ids": worker_ids,
        }
