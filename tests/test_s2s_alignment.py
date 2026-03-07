"""Unit tests for Stage-2 reply/content alignment helpers."""

import pytest
import torch

from model.constants import IGNORE_INDEX
from train.s2s_trainer import _extract_aligned_reply_states, _find_unique_subsequence_start


def test_find_unique_subsequence_start_returns_match_index():
    # Match index should point to the single contiguous target span.
    sequence = torch.tensor([9, 1, 2, 3, 7], dtype=torch.long)
    target = torch.tensor([1, 2, 3], dtype=torch.long)
    assert _find_unique_subsequence_start(sequence=sequence, target=target) == 1


def test_find_unique_subsequence_start_fails_on_no_match():
    # Missing spans should fail with an explicit "no match" reason.
    sequence = torch.tensor([9, 1, 4, 3, 7], dtype=torch.long)
    target = torch.tensor([1, 2, 3], dtype=torch.long)
    with pytest.raises(ValueError, match="no match"):
        _find_unique_subsequence_start(sequence=sequence, target=target)


def test_find_unique_subsequence_start_fails_on_ambiguous_match():
    # Repeated spans should fail so alignment remains deterministic.
    sequence = torch.tensor([1, 2, 1, 2], dtype=torch.long)
    target = torch.tensor([1, 2], dtype=torch.long)
    with pytest.raises(ValueError, match=r"ambiguous \(2 matches\)"):
        _find_unique_subsequence_start(sequence=sequence, target=target)


def test_extract_aligned_reply_states_aligns_non_prefix_content_span():
    # Content alignment should work even when template tokens lead the reply.
    last_hidden = torch.arange(6, dtype=torch.float32).view(1, 6, 1)
    expanded_labels = torch.tensor([[IGNORE_INDEX, 99, 1, 2, 3, 88]], dtype=torch.long)
    content_ids_rows = [torch.tensor([1, 2, 3], dtype=torch.long)]

    aligned = _extract_aligned_reply_states(
        last_hidden=last_hidden,
        expanded_labels=expanded_labels,
        content_ids_rows=content_ids_rows,
    )

    assert len(aligned) == 1
    assert torch.equal(aligned[0].squeeze(-1), torch.tensor([2.0, 3.0, 4.0]))


def test_extract_aligned_reply_states_aligns_prefix_content_span():
    # Prefix content should still align exactly to the leading reply tokens.
    last_hidden = torch.arange(6, dtype=torch.float32).view(1, 6, 1)
    expanded_labels = torch.tensor([[1, 2, 3, 99, IGNORE_INDEX, IGNORE_INDEX]], dtype=torch.long)
    content_ids_rows = [torch.tensor([1, 2, 3], dtype=torch.long)]

    aligned = _extract_aligned_reply_states(
        last_hidden=last_hidden,
        expanded_labels=expanded_labels,
        content_ids_rows=content_ids_rows,
    )

    assert len(aligned) == 1
    assert torch.equal(aligned[0].squeeze(-1), torch.tensor([0.0, 1.0, 2.0]))


def test_extract_aligned_reply_states_surfaces_alignment_reason():
    # Top-level alignment errors should expose the exact root reason.
    last_hidden = torch.arange(6, dtype=torch.float32).view(1, 6, 1)
    expanded_labels = torch.tensor([[IGNORE_INDEX, 1, 2, 1, 2, 9]], dtype=torch.long)
    content_ids_rows = [torch.tensor([1, 2], dtype=torch.long)]

    with pytest.raises(ValueError, match=r"alignment failed: ambiguous \(2 matches\)"):
        _extract_aligned_reply_states(
            last_hidden=last_hidden,
            expanded_labels=expanded_labels,
            content_ids_rows=content_ids_rows,
        )


def test_extract_aligned_reply_states_fails_on_no_match():
    # Missing content span should fail with a clear no-match alignment reason.
    last_hidden = torch.arange(6, dtype=torch.float32).view(1, 6, 1)
    expanded_labels = torch.tensor([[IGNORE_INDEX, 5, 6, 7, 8, 9]], dtype=torch.long)
    content_ids_rows = [torch.tensor([1, 2, 3], dtype=torch.long)]

    with pytest.raises(ValueError, match=r"alignment failed: no match"):
        _extract_aligned_reply_states(
            last_hidden=last_hidden,
            expanded_labels=expanded_labels,
            content_ids_rows=content_ids_rows,
        )
