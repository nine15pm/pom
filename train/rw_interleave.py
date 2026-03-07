"""Shared Read/Write interleaving schedule helpers."""

from __future__ import annotations

from math import ceil
from typing import List, Tuple


def build_read_write_schedule(
    *,
    num_read_tokens: int,
    num_write_tokens: int,
    read_length: int,
    write_length: int,
) -> List[Tuple[int, int]]:
    """Return per-round (read_count, write_count) for write-driven Read/Write training."""
    if read_length <= 0 or write_length <= 0:
        raise ValueError("read_length and write_length must be positive")
    if num_read_tokens < 0:
        raise ValueError("num_read_tokens must be >= 0")
    if num_write_tokens <= 0:
        raise ValueError("num_write_tokens must be > 0")

    schedule: List[Tuple[int, int]] = []
    read_remaining = int(num_read_tokens)
    write_remaining = int(num_write_tokens)

    # Each round consumes up to R reads and always consumes up to W writes.
    while write_remaining > 0:
        read_take = min(read_length, read_remaining) if read_remaining > 0 else 0
        write_take = min(write_length, write_remaining)
        schedule.append((read_take, write_take))
        read_remaining -= read_take
        write_remaining -= write_take

    return schedule


def max_read_tokens_for_write_tokens(
    *,
    num_write_tokens: int,
    read_length: int,
    write_length: int,
) -> int:
    """Return max read tokens consumable under strict write-driven Read/Write rounds."""
    if read_length <= 0 or write_length <= 0:
        raise ValueError("read_length and write_length must be positive")
    if num_write_tokens < 0:
        raise ValueError("num_write_tokens must be >= 0")
    if num_write_tokens == 0:
        return 0
    # Each write round unlocks at most R read tokens.
    return int(read_length) * int(ceil(int(num_write_tokens) / int(write_length)))


__all__ = ["build_read_write_schedule", "max_read_tokens_for_write_tokens"]
