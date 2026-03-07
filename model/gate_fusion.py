"""Gate fusion module for S2S conditioning."""

from __future__ import annotations

from typing import Tuple

import torch
from torch import nn


class GateFusion(nn.Module):
    """Blend projected Thinker hidden states with Talker text embeddings.

    Contract: this module is tensor-only. It accepts hidden states + text embeddings,
    while token-id -> embedding lookup happens in PomTalker.
    """

    def __init__(
        self,
        *,
        llm_hidden_dim: int,
        speech_embed_dim: int,
        ffn_hidden_dim: int | None = None,
    ) -> None:
        # Keep fusion tiny: one projector and one gate head, matching the S2S reference design.
        super().__init__()
        if llm_hidden_dim <= 0 or speech_embed_dim <= 0:
            raise ValueError("llm_hidden_dim and speech_embed_dim must be positive")

        hidden = int(ffn_hidden_dim) if ffn_hidden_dim is not None else int(speech_embed_dim) * 2
        if hidden <= 0:
            raise ValueError("ffn_hidden_dim must be positive when provided")

        self.llm_hidden_dim = int(llm_hidden_dim)
        self.speech_embed_dim = int(speech_embed_dim)
        self.ffn_hidden_dim = hidden

        # Project Thinker hidden states into the same embedding space used by Talker tokens.
        self.hidden_proj = nn.Sequential(
            nn.Linear(self.llm_hidden_dim, self.ffn_hidden_dim),
            nn.SiLU(),
            nn.Linear(self.ffn_hidden_dim, self.speech_embed_dim),
        )
        # Produce one gate value per embedding dimension from [projected_hidden || text_embedding].
        self.gate_proj = nn.Linear(self.speech_embed_dim * 2, self.speech_embed_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        text_embeddings: torch.Tensor,
        *,
        return_gate: bool = False,
    ) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
        """Fuse aligned per-token tensors from Thinker and Talker branches."""
        # Normalize both inputs to fusion parameter device/dtype for stable mixed-precision behavior.
        param = self.hidden_proj[0].weight
        target_device = param.device
        target_dtype = param.dtype
        hidden_states = hidden_states.to(device=target_device, dtype=target_dtype)
        text_embeddings = text_embeddings.to(device=target_device, dtype=target_dtype)

        # Fuse per-token representations: each hidden state position must match its text token position.
        # Validate token-axis alignment so fusion always mixes matching positions.
        if hidden_states.ndim != text_embeddings.ndim:
            raise ValueError("hidden_states and text_embeddings must have the same rank")
        if hidden_states.shape[:-1] != text_embeddings.shape[:-1]:
            raise ValueError("hidden_states and text_embeddings must align on leading dimensions")
        if hidden_states.shape[-1] != self.llm_hidden_dim:
            raise ValueError(
                f"hidden_states last dim {hidden_states.shape[-1]} != llm_hidden_dim {self.llm_hidden_dim}"
            )
        if text_embeddings.shape[-1] != self.speech_embed_dim:
            raise ValueError(
                "text_embeddings last dim "
                f"{text_embeddings.shape[-1]} != speech_embed_dim {self.speech_embed_dim}"
            )

        # Move Thinker features into Talker space so both sources are directly comparable.
        projected_hidden = self.hidden_proj(hidden_states)
        # Learn how much to trust context (projected hidden) vs lexical identity (text embedding).
        gate_inputs = torch.cat([projected_hidden, text_embeddings], dim=-1)
        gate = torch.sigmoid(self.gate_proj(gate_inputs))
        # Convex blend per dimension: gate=1 -> all hidden, gate=0 -> all text embedding.
        fused = gate * projected_hidden + (1.0 - gate) * text_embeddings
        if return_gate:
            return fused, gate
        return fused


__all__ = ["GateFusion"]
