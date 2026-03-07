"""Downsample speech encoder frames and project into the LM space."""

from __future__ import annotations

from typing import Optional, Tuple

import torch
from torch import nn


class FrameStackProjector(nn.Module):
    """Stack sequential frames and map them into the LM embedding space."""

    def __init__(
        self,
        frame_stack: int,
        input_dim: int,
        target_dim: int,
        hidden_dim: int = 2048,
    ) -> None:
        super().__init__()
        if frame_stack < 1:
            raise ValueError("frame_stack must be >= 1")
        if input_dim <= 0 or target_dim <= 0:
            raise ValueError("input_dim and target_dim must be positive")

        # Keep constructor args on the module so artifact metadata is explicit.
        self.frame_stack = int(frame_stack)
        self.input_dim = int(input_dim)
        self.target_dim = int(target_dim)
        self.hidden_dim = int(hidden_dim)

        stacked_dim = self.input_dim * self.frame_stack
        self.adapter = nn.Sequential(
            nn.Linear(stacked_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.target_dim),
        )

    # Export projector build metadata for HF artifact reconstruction.
    def to_config_dict(self) -> dict[str, int]:
        return {
            "frame_stack": int(self.frame_stack),
            "input_dim": int(self.input_dim),
            "target_dim": int(self.target_dim),
            "hidden_dim": int(self.hidden_dim),
        }

    def forward(
        self,
        hidden_states: torch.Tensor,
        frame_masks: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Downsample and project encoder hidden states."""

        stacked_states, stacked_masks = self._stack(hidden_states, frame_masks)
        projected = self.adapter(stacked_states)
        return projected, stacked_masks

    def _stack(
        self,
        hidden_states: torch.Tensor,
        frame_masks: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch, seq_len, dim = hidden_states.shape
        k = self.frame_stack
        pad = (-seq_len) % k
        if pad:
            padding = hidden_states.new_zeros(batch, pad, dim)
            hidden_states = torch.cat([hidden_states, padding], dim=1)
            seq_len = hidden_states.size(1)

        hidden_states = hidden_states.contiguous()
        # Pad to a multiple of k so reshape groups frames evenly; mask drops padded tail.
        stacked_states = hidden_states.view(batch, seq_len // k, k * dim)

        if frame_masks is None:
            stacked_masks = torch.ones(
                (batch, stacked_states.size(1)),
                dtype=torch.bool,
                device=hidden_states.device,
            )
            return stacked_states, stacked_masks

        if frame_masks.shape[0] != batch:
            raise ValueError("frame_masks batch size must match hidden_states")
        if frame_masks.ndim != 2:
            raise ValueError("frame_masks must be 2-D (batch, frames)")

        if pad:
            mask_pad = torch.zeros(batch, pad, dtype=torch.bool, device=hidden_states.device)
            frame_masks = torch.cat([frame_masks.to(device=hidden_states.device), mask_pad], dim=1)
        else:
            frame_masks = frame_masks.to(device=hidden_states.device)

        grouped = frame_masks.view(batch, seq_len // k, k)
        stacked_masks = grouped.any(dim=-1)
        return stacked_states, stacked_masks
