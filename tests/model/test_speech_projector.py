from __future__ import annotations

import torch

from model.speech_projector import FrameStackProjector


DEVICE = torch.device("cuda")


def test_projector_stacks_and_masks_tail():
    projector = FrameStackProjector(frame_stack=3, input_dim=4, target_dim=8, hidden_dim=16).to(device=DEVICE)

    hidden = torch.arange(2 * 7 * 4, dtype=torch.float32, device=DEVICE).view(2, 7, 4)
    mask = torch.tensor(
        [
            [1, 1, 1, 1, 1, 1, 0],  # last frame invalid
            [1, 1, 1, 1, 0, 0, 0],  # partial second chunk
        ],
        dtype=torch.bool,
        device=DEVICE,
    )

    stacked, stacked_mask = projector(hidden, mask)

    # seq=7 with k=3 -> ceil(7/3)=3 groups
    assert stacked.shape == (2, 3, 8)
    assert stacked_mask.shape == (2, 3)

    # Tail group should be masked out when all frames are invalid
    assert stacked_mask[1, 2].item() is False

    # When mask is None, all positions are valid
    stacked2, stacked_mask2 = projector(hidden, None)
    assert stacked2.shape == stacked.shape
    assert torch.all(stacked_mask2)


def test_projector_frame_stack_one_preserves_length_and_mask():
    projector = FrameStackProjector(frame_stack=1, input_dim=4, target_dim=8, hidden_dim=16).to(device=DEVICE)
    hidden = torch.randn(2, 5, 4, device=DEVICE)
    mask = torch.tensor([[1, 0, 1, 1, 0], [1, 1, 1, 1, 1]], dtype=torch.bool, device=DEVICE)

    stacked, stacked_mask = projector(hidden, mask)

    assert stacked.shape == (2, 5, 8)
    assert torch.equal(stacked_mask, mask)


def test_projector_rejects_bad_mask_shape():
    projector = FrameStackProjector(frame_stack=2, input_dim=4, target_dim=8, hidden_dim=16).to(device=DEVICE)
    hidden = torch.randn(2, 4, 4, device=DEVICE)
    bad_mask = torch.ones(2, 4, 1, dtype=torch.bool, device=DEVICE)

    try:
        projector(hidden, bad_mask)
    except ValueError:
        return
    raise AssertionError("Expected ValueError for non-2D frame mask")
