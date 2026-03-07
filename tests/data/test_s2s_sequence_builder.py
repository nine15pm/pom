from __future__ import annotations

import pytest
import torch

from model.constants import DEFAULT_SEP_TOKEN, IGNORE_INDEX
from model.pom_talker import build_talker
from train.s2s_sequence_builder import build_s2s_read_write_batch, s2s_passes_read_write_ratio_filter


@pytest.fixture(scope="module")
def shared_s2s_talker(local_base_model_id):
    # Build one Talker/tokenizer pair for all S2S sequence-builder tests in this module.
    talker, tokenizer, _ = build_talker(base_model_id=local_base_model_id)
    sep_id = tokenizer.convert_tokens_to_ids(DEFAULT_SEP_TOKEN)
    if sep_id == tokenizer.unk_token_id:
        raise ValueError("<sep> token id unexpectedly unresolved")
    eos_id = tokenizer.eos_token_id
    if eos_id is None:
        raise ValueError("talker tokenizer eos_token_id must be defined")
    return talker, int(sep_id), int(eos_id)


def test_build_s2s_read_write_batch_interleaves_masks_and_supervises_eos(shared_s2s_talker):
    # Validate the Stage-2 training contract for read/write interleaving on one real sample.
    talker, sep_id, eos_id = shared_s2s_talker
    embed_tokens = talker.get_input_embeddings()
    embed_weight = embed_tokens.weight

    # Use fixed fused vectors so expected sequence values are deterministic.
    fused_row = torch.arange(
        4 * int(embed_weight.shape[-1]),
        device=embed_weight.device,
        dtype=torch.float32,
    ).reshape(4, int(embed_weight.shape[-1])).to(dtype=embed_weight.dtype)
    unit_row = torch.arange(12, device=embed_weight.device, dtype=torch.long)

    batch = build_s2s_read_write_batch(
        talker=talker,
        fused_rows=[fused_row],
        unit_rows=[unit_row],
        sep_id=int(sep_id),
        eos_id=int(eos_id),
        read_length=3,
        write_length=10,
    )

    mapped_units = talker.map_unit_ids(unit_row)
    expected_embeds = torch.cat(
        [
            fused_row[:3],
            embed_tokens(mapped_units[:10]),
            fused_row[3:4],
            embed_tokens(torch.tensor([int(sep_id)], device=embed_weight.device, dtype=torch.long)),
            embed_tokens(mapped_units[10:12]),
            embed_tokens(torch.tensor([int(eos_id)], device=embed_weight.device, dtype=torch.long)),
        ],
        dim=0,
    )
    expected_labels = torch.cat(
        [
            torch.full((3,), IGNORE_INDEX, device=embed_weight.device, dtype=torch.long),
            mapped_units[:10],
            torch.full((1,), IGNORE_INDEX, device=embed_weight.device, dtype=torch.long),
            torch.full((1,), IGNORE_INDEX, device=embed_weight.device, dtype=torch.long),
            mapped_units[10:12],
            torch.tensor([int(eos_id)], device=embed_weight.device, dtype=torch.long),
        ],
        dim=0,
    )

    # Sequence should match the exact expected interleave order, masking, and EOS supervision.
    assert torch.equal(batch["inputs_embeds"][0, : expected_embeds.size(0)], expected_embeds)
    assert torch.equal(batch["labels"][0, : expected_labels.size(0)], expected_labels)
    assert torch.equal(batch["attention_mask"][0, : expected_labels.size(0)], torch.ones_like(expected_labels))


def test_build_s2s_read_write_batch_pads_right_and_masks_padding(shared_s2s_talker):
    # Validate batch padding contract so Talker loss never supervises padded positions.
    talker, sep_id, eos_id = shared_s2s_talker
    embed_tokens = talker.get_input_embeddings()
    embed_weight = embed_tokens.weight

    # Build two real rows with different final sequence lengths to force right-padding.
    fused_row_long = torch.arange(
        4 * int(embed_weight.shape[-1]),
        device=embed_weight.device,
        dtype=torch.float32,
    ).reshape(4, int(embed_weight.shape[-1])).to(dtype=embed_weight.dtype)
    unit_row_long = torch.arange(12, device=embed_weight.device, dtype=torch.long)

    fused_row_short = torch.arange(
        2 * int(embed_weight.shape[-1]),
        device=embed_weight.device,
        dtype=torch.float32,
    ).reshape(2, int(embed_weight.shape[-1])).to(dtype=embed_weight.dtype)
    unit_row_short = torch.arange(3, device=embed_weight.device, dtype=torch.long)

    batch = build_s2s_read_write_batch(
        talker=talker,
        fused_rows=[fused_row_long, fused_row_short],
        unit_rows=[unit_row_long, unit_row_short],
        sep_id=int(sep_id),
        eos_id=int(eos_id),
        read_length=3,
        write_length=10,
    )

    long_len = 18
    short_len = 7
    max_len = long_len

    # Long sample should occupy the full row; short sample should be right-padded.
    assert batch["inputs_embeds"].shape == (2, max_len, int(embed_weight.shape[-1]))
    assert batch["labels"].shape == (2, max_len)
    assert batch["attention_mask"].shape == (2, max_len)

    assert torch.equal(batch["attention_mask"][0], torch.ones((max_len,), device=embed_weight.device, dtype=torch.long))
    assert torch.equal(
        batch["attention_mask"][1],
        torch.cat(
            [
                torch.ones((short_len,), device=embed_weight.device, dtype=torch.long),
                torch.zeros((max_len - short_len,), device=embed_weight.device, dtype=torch.long),
            ],
            dim=0,
        ),
    )
    assert torch.equal(
        batch["labels"][1, short_len:],
        torch.full((max_len - short_len,), IGNORE_INDEX, device=embed_weight.device, dtype=torch.long),
    )
    assert torch.equal(
        batch["inputs_embeds"][1, short_len:],
        torch.zeros((max_len - short_len, int(embed_weight.shape[-1])), device=embed_weight.device, dtype=embed_weight.dtype),
    )


def test_build_s2s_read_write_batch_fails_when_content_exceeds_strict_ratio(shared_s2s_talker):
    # Validate fail-fast behavior when strict write-driven rounds cannot consume all reads.
    talker, sep_id, eos_id = shared_s2s_talker
    embed_weight = talker.get_input_embeddings().weight

    # With W=10 and 10 units, there is only one write round so max readable content is R=3.
    fused_row = torch.zeros((4, int(embed_weight.shape[-1])), device=embed_weight.device, dtype=embed_weight.dtype)
    unit_row = torch.arange(10, device=embed_weight.device, dtype=torch.long)

    with pytest.raises(ValueError, match="exceeds strict Read/Write max"):
        _ = build_s2s_read_write_batch(
            talker=talker,
            fused_rows=[fused_row],
            unit_rows=[unit_row],
            sep_id=int(sep_id),
            eos_id=int(eos_id),
            read_length=3,
            write_length=10,
        )


def test_s2s_passes_read_write_ratio_filter_matches_boundary_at_r3_w10():
    # Validate exact filter boundary so dataset gating matches strict builder behavior.
    assert s2s_passes_read_write_ratio_filter(
        content_token_count=3,
        unit_token_count=10,
        read_length=3,
        write_length=10,
    )
    assert not s2s_passes_read_write_ratio_filter(
        content_token_count=4,
        unit_token_count=10,
        read_length=3,
        write_length=10,
    )
