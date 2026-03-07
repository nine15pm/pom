"""TTS (Text-to-Speech) Read/Write sequence construction helpers."""

from __future__ import annotations

from typing import List, Sequence, Tuple

from model.constants import IGNORE_INDEX, SPEECH_VOCAB_SIZE
from train.rw_interleave import build_read_write_schedule


def build_read_write_sequence(
    text_token_ids: Sequence[int],
    unit_ids: Sequence[int],
    *,
    speech_token_offset: int,
    sep_id: int,
    eos_id: int,
    read_length: int,
    write_length: int,
    ignore_index: int = IGNORE_INDEX,
    speech_vocab_size: int = SPEECH_VOCAB_SIZE,
) -> Tuple[List[int], List[int]]:
    """Build interleaved Read/Write input ids + labels for TTS training."""
    if read_length <= 0 or write_length <= 0:
        raise ValueError("read_length and write_length must be positive")

    # Normalize inputs to plain python int lists.
    text_ids = [int(token) for token in text_token_ids]
    raw_unit_ids = [int(unit) for unit in unit_ids]
    if not raw_unit_ids:
        raise ValueError("unit_ids must be non-empty for TTS training")

    # Validate unit id range before mapping into speech token ids.
    for unit in raw_unit_ids:
        if unit < 0 or unit >= int(speech_vocab_size):
            raise ValueError(f"unit id {unit} is out of range [0, {speech_vocab_size - 1}]")

    speech_ids = [unit + int(speech_token_offset) for unit in raw_unit_ids]

    schedule = build_read_write_schedule(
        num_read_tokens=len(text_ids),
        num_write_tokens=len(speech_ids),
        read_length=read_length,
        write_length=write_length,
    )

    input_ids: List[int] = []
    labels: List[int] = []
    text_cursor = 0
    speech_cursor = 0
    sep_added = False

    # Apply the shared schedule to emit text reads, one <sep>, then speech writes.
    for read_take, write_take in schedule:
        if read_take > 0:
            input_ids.extend(text_ids[text_cursor : text_cursor + read_take])
            labels.extend([ignore_index] * read_take)
            text_cursor += read_take

        # Insert <sep> exactly once after text conditioning is fully consumed.
        if text_cursor >= len(text_ids) and not sep_added:
            input_ids.append(int(sep_id))
            labels.append(ignore_index)
            sep_added = True

        chunk = speech_ids[speech_cursor : speech_cursor + write_take]
        input_ids.extend(chunk)
        labels.extend(chunk)
        speech_cursor += write_take

    # Keep the same contract: all text must be consumed by the time speech ends.
    if text_cursor != len(text_ids):
        raise ValueError("text tokens remain after speech tokens are exhausted")

    # Always supervise EOS as the final speech stop token.
    input_ids.append(int(eos_id))
    labels.append(int(eos_id))

    return input_ids, labels


__all__ = ["build_read_write_sequence"]
