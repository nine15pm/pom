"""S2S (Stage-2) Read/Write embedding-sequence construction helpers."""

from __future__ import annotations

from collections.abc import Sequence

import torch

from model.constants import IGNORE_INDEX
from train.rw_interleave import build_read_write_schedule, max_read_tokens_for_write_tokens


def s2s_passes_read_write_ratio_filter(
    *,
    content_token_count: int,
    unit_token_count: int,
    read_length: int,
    write_length: int,
) -> bool:
    """Return whether strict Stage-2 Read/Write can fully consume all conditioning tokens."""
    if content_token_count < 0:
        raise ValueError("content_token_count must be >= 0")
    max_read = max_read_tokens_for_write_tokens(
        num_write_tokens=int(unit_token_count),
        read_length=int(read_length),
        write_length=int(write_length),
    )
    return int(content_token_count) <= int(max_read)


def build_s2s_read_write_batch(
    *,
    talker: torch.nn.Module,
    fused_rows: Sequence[torch.Tensor],
    unit_rows: Sequence[torch.Tensor],
    sep_id: int,
    eos_id: int,
    read_length: int,
    write_length: int,
    ignore_index: int = IGNORE_INDEX,
) -> dict[str, torch.Tensor]:
    """Build padded Stage-2 LM inputs from fused conditioning rows and unit-id targets."""
    if len(fused_rows) != len(unit_rows):
        raise ValueError("fused_rows and unit_rows must have the same batch size")
    if not fused_rows:
        raise ValueError("fused_rows must be non-empty")
    if not hasattr(talker, "map_unit_ids"):
        raise TypeError("talker must expose map_unit_ids for Stage-2 unit-token mapping")
    if not callable(getattr(talker, "get_input_embeddings", None)):
        raise TypeError("talker must expose get_input_embeddings for Stage-2 embedding lookups")

    # Read embeddings from the Stage-2 LM directly (HF-native model API).
    embed_tokens = talker.get_input_embeddings()
    embed_weight = embed_tokens.weight
    device = embed_weight.device
    vocab_size = int(embed_weight.size(0))
    embed_dim = int(embed_weight.shape[-1])
    embed_dtype = embed_weight.dtype

    # Validate boundary/stop ids early for clear token-contract errors.
    if int(sep_id) < 0 or int(sep_id) >= vocab_size:
        raise ValueError(f"sep_id {int(sep_id)} is out of range [0, {vocab_size - 1}]")
    if int(eos_id) < 0 or int(eos_id) >= vocab_size:
        raise ValueError(f"eos_id {int(eos_id)} is out of range [0, {vocab_size - 1}]")

    # Reuse one-token lookups for boundary/stop embeddings.
    sep_token = torch.tensor([int(sep_id)], device=device, dtype=torch.long)
    eos_token = torch.tensor([int(eos_id)], device=device, dtype=torch.long)
    sep_embed = embed_tokens(sep_token)
    eos_embed = embed_tokens(eos_token)

    sample_embeds: list[torch.Tensor] = []
    sample_labels: list[torch.Tensor] = []

    for row_idx, (fused, units) in enumerate(zip(fused_rows, unit_rows)):
        # Validate rank/shape so sequencing logic remains explicit and safe.
        if fused.ndim != 2:
            raise ValueError(f"fused_rows[{row_idx}] must be rank-2 [tokens, dim]")
        if int(fused.size(0)) <= 0:
            raise ValueError(f"fused_rows[{row_idx}] must be non-empty")
        if int(fused.size(1)) != embed_dim:
            raise ValueError(
                f"fused_rows[{row_idx}] embedding dim {int(fused.size(1))} does not match Talker LM dim {embed_dim}"
            )

        units_long = torch.as_tensor(units, dtype=torch.long, device=device)
        if units_long.ndim != 1:
            raise ValueError(f"unit_rows[{row_idx}] must be rank-1")
        if units_long.numel() <= 0:
            raise ValueError(f"unit_rows[{row_idx}] must be non-empty")

        # Enforce strict full-read feasibility with the same ratio filter data should apply.
        if not s2s_passes_read_write_ratio_filter(
            content_token_count=int(fused.size(0)),
            unit_token_count=int(units_long.numel()),
            read_length=int(read_length),
            write_length=int(write_length),
        ):
            max_read = max_read_tokens_for_write_tokens(
                num_write_tokens=int(units_long.numel()),
                read_length=int(read_length),
                write_length=int(write_length),
            )
            raise ValueError(
                f"sample {row_idx}: content length {int(fused.size(0))} exceeds strict Read/Write max {int(max_read)} "
                f"for {int(units_long.numel())} unit tokens; apply the S2S ratio filter in data/collation"
            )

        mapped_units = talker.map_unit_ids(units_long)
        unit_embeds = embed_tokens(mapped_units)
        fused = fused.to(device=device, dtype=embed_dtype)

        schedule = build_read_write_schedule(
            num_read_tokens=int(fused.size(0)),
            num_write_tokens=int(mapped_units.numel()),
            read_length=int(read_length),
            write_length=int(write_length),
        )

        cur_embeds: list[torch.Tensor] = []
        cur_labels: list[torch.Tensor] = []
        read_cursor = 0
        write_cursor = 0
        sep_added = False

        # Interleave fused reads and speech writes using the shared schedule.
        for read_take, write_take in schedule:
            if read_take > 0:
                read_chunk = fused[read_cursor : read_cursor + read_take]
                cur_embeds.append(read_chunk)
                cur_labels.append(
                    torch.full((read_take,), int(ignore_index), device=device, dtype=torch.long)
                )
                read_cursor += read_take

            # Insert <sep> once when conditioning has been fully consumed.
            if read_cursor >= int(fused.size(0)) and not sep_added:
                cur_embeds.append(sep_embed)
                cur_labels.append(torch.tensor([int(ignore_index)], device=device, dtype=torch.long))
                sep_added = True

            write_chunk_embeds = unit_embeds[write_cursor : write_cursor + write_take]
            write_chunk_labels = mapped_units[write_cursor : write_cursor + write_take]
            cur_embeds.append(write_chunk_embeds)
            cur_labels.append(write_chunk_labels)
            write_cursor += write_take

        # Keep TTS-compatible behavior: fail fast if read tokens remain unread.
        if read_cursor != int(fused.size(0)):
            raise ValueError("fused conditioning tokens remain after speech tokens are exhausted")

        # Supervise EOS as final stop token in every sample.
        cur_embeds.append(eos_embed)
        cur_labels.append(eos_token)

        sample_embeds.append(torch.cat(cur_embeds, dim=0))
        sample_labels.append(torch.cat(cur_labels, dim=0))

    max_len = max(int(row.size(0)) for row in sample_embeds)
    batch_size = len(sample_embeds)
    batch_embeds = torch.zeros((batch_size, max_len, embed_dim), device=device, dtype=embed_dtype)
    batch_labels = torch.full((batch_size, max_len), int(ignore_index), device=device, dtype=torch.long)
    batch_attention_mask = torch.zeros((batch_size, max_len), device=device, dtype=torch.long)

    # Right-pad sequence rows for standard decoder-only training.
    for row_idx, (embeds, labels) in enumerate(zip(sample_embeds, sample_labels)):
        cur_len = int(embeds.size(0))
        batch_embeds[row_idx, :cur_len] = embeds
        batch_labels[row_idx, :cur_len] = labels
        batch_attention_mask[row_idx, :cur_len] = 1

    return {
        "inputs_embeds": batch_embeds,
        "attention_mask": batch_attention_mask,
        "labels": batch_labels,
    }


__all__ = [
    "build_s2s_read_write_batch",
    "s2s_passes_read_write_ratio_filter",
]
