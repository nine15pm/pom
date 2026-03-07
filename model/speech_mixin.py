"""Speech-aware mixins that align speech features with the LM token stream."""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple, Union

import torch
from torch import nn

from .constants import IGNORE_INDEX


class SpeechMixin:
    """Mixin owning the speech encoder/projector stack for pom models."""

    def __init__(self, config) -> None:  # noqa: D401 - config type is backend specific
        super().__init__(config)
        self.config = config
        self._speech_encoder: Optional[nn.Module] = None
        self._speech_projector: Optional[nn.Module] = None
        self.ignore_index: int = getattr(config, "ignore_index", IGNORE_INDEX)

    # --- Speech module helpers -------------------------------------------------
    def set_speech_modules(
        self,
        *,
        speech_encoder: Optional[nn.Module] = None,
        speech_projector: Optional[nn.Module] = None,
    ) -> None:
        """Register speech encoder/projector modules after instantiation."""

        if speech_encoder is not None:
            self._speech_encoder = speech_encoder
        if speech_projector is not None:
            self._speech_projector = speech_projector

    def get_speech_encoder(self) -> Optional[nn.Module]:
        return self._speech_encoder

    def get_speech_projector(self) -> Optional[nn.Module]:
        return self._speech_projector

    def _speech_token_id(self) -> int:
        token_id = getattr(self.config, "speech_token_id", None)
        if token_id is None:
            raise ValueError("speech_token_id must be set on config for speech fusion")
        return int(token_id)

    def encode_speech(
        self,
        waveforms: Sequence[torch.Tensor],
        *,
        sampling_rate: Optional[Union[int, Sequence[int]]] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode raw audio waveforms into hidden features with frame masks.

        Returns:
            Tuple of (hidden_states, frame_masks) where masks indicate valid frames.
        """

        encoder = self.get_speech_encoder()
        if encoder is None:
            raise RuntimeError("Speech encoder is not initialised on the model")
        return encoder(waveforms, sampling_rate=sampling_rate)

    def project_speech_features(
        self, hidden_states: torch.Tensor, frame_masks: Optional[torch.Tensor] = None
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Project speech encoder states into the LM embedding space.

        Args:
            hidden_states: Encoder output features
            frame_masks: Optional masks indicating valid frames

        Returns:
            Tuple of (projected_features, stacked_masks) if masks provided,
            otherwise (projected_features, None)
        """

        projector = self.get_speech_projector()
        if projector is None:
            return hidden_states, frame_masks

        projected, stacked_masks = projector(hidden_states, frame_masks)
        return projected, stacked_masks

    def _collect_speech_token_mask(self, input_ids: torch.LongTensor) -> torch.Tensor:
        token_id = self._speech_token_id()
        return input_ids == token_id


class SpeechMixinForCausalLM(SpeechMixin):
    """Mixin implementing speech-aware preparation for CausalLM models."""

    def _embed_spec(self) -> tuple[nn.Embedding, torch.device, torch.dtype, int]:
        embed_tokens = self.get_input_embeddings()
        weight = embed_tokens.weight
        return embed_tokens, weight.device, weight.dtype, weight.shape[-1]

    def _trim_with_mask(self, frames: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
        if mask is None:
            return frames
        if mask.ndim != 1:
            raise ValueError("speech frame mask must be 1-D per segment")
        valid = int(mask.sum().item())
        return frames[:valid]

    def _prepare_speech_segments(
        self,
        *,
        total_segments: int,
        speech_waveforms: Optional[Sequence[torch.Tensor]],
        speech_features: Optional[Sequence[torch.Tensor]],
        speech_sampling_rate: Optional[Union[int, Sequence[int]]],
        embed_device: torch.device,
        embed_dtype: torch.dtype,
        embed_dim: int,
    ) -> list[torch.Tensor]:
        if total_segments <= 0:
            return []

        def _ensure_dim(tensor: torch.Tensor) -> torch.Tensor:
            if tensor.shape[-1] != embed_dim:
                raise ValueError(
                    f"speech feature dim {tensor.shape[-1]} != embed dim {embed_dim}; configure projector",
                )
            return tensor.to(device=embed_device, dtype=embed_dtype)

        if speech_features is not None:
            if isinstance(speech_features, torch.Tensor):
                speech_features = list(speech_features)
            segments: list[torch.Tensor] = []
            for item in speech_features:
                mask = None
                tensor = item
                if isinstance(item, (tuple, list)) and len(item) == 2:
                    tensor, mask = item  # type: ignore[assignment]
                segment = torch.as_tensor(tensor)
                segments.append(_ensure_dim(self._trim_with_mask(segment, mask)))
            if len(segments) != total_segments:
                raise ValueError(
                    f"Expected {total_segments} speech feature tensors, received {len(segments)}",
                )
            return segments

        if speech_waveforms is None:
            raise ValueError("speech_waveforms or speech_features must be provided when speech tokens are present")

        wave_list: list[torch.Tensor]
        if isinstance(speech_waveforms, torch.Tensor):
            if speech_waveforms.ndim == 1:
                wave_list = [speech_waveforms]
            elif speech_waveforms.ndim == 2:
                wave_list = [segment for segment in speech_waveforms]
            else:
                raise ValueError("speech_waveforms tensor must be 1-D or 2-D")
        else:
            wave_list = [torch.as_tensor(w) for w in speech_waveforms]

        if len(wave_list) != total_segments:
            raise ValueError(
                f"Expected {total_segments} speech waveforms, received {len(wave_list)}",
            )

        encoded, frame_masks = self.encode_speech(wave_list, sampling_rate=speech_sampling_rate)
        projected, stacked_masks = self.project_speech_features(encoded, frame_masks)

        mask_list = (
            list(stacked_masks.unbind(dim=0)) if stacked_masks is not None else [None] * projected.size(0)
        )
        segments = [
            _ensure_dim(self._trim_with_mask(segment, mask))
            for segment, mask in zip(projected.unbind(dim=0), mask_list)
        ]
        if len(segments) != total_segments:
            raise RuntimeError("Mismatch between collected speech segments and expected count")
        return segments

    def prepare_inputs_labels_for_speech_and_text(
        self,
        input_ids: torch.LongTensor,
        position_ids: Optional[torch.LongTensor],
        attention_mask: Optional[torch.Tensor],
        past_key_values,
        labels: Optional[torch.LongTensor],
        *,
        speech_waveforms: Optional[Sequence[torch.Tensor]] = None,
        speech_features: Optional[Sequence[torch.Tensor]] = None,
        speech_sampling_rate: Optional[Union[int, Sequence[int]]] = None,
    ):
        """Replace <speech> tokens with fused speech embeddings."""

        if input_ids is None:
            raise ValueError("input_ids are required for speech fusion")

        speech_token_mask = self._collect_speech_token_mask(input_ids)
        total_speech_segments = int(speech_token_mask.sum().item())
        if total_speech_segments == 0:
            return input_ids, position_ids, attention_mask, past_key_values, None, labels

        embed_tokens, embed_device, embed_dtype, embed_dim = self._embed_spec()

        attn_mask = attention_mask.bool() if attention_mask is not None else torch.ones_like(input_ids, dtype=torch.bool)
        speech_token_mask = speech_token_mask & attn_mask
        speech_counts = speech_token_mask.sum(dim=1)

        speech_segments = self._prepare_speech_segments(
            total_segments=total_speech_segments,
            speech_waveforms=speech_waveforms,
            speech_features=speech_features,
            speech_sampling_rate=speech_sampling_rate,
            embed_device=embed_device,
            embed_dtype=embed_dtype,
            embed_dim=embed_dim,
        )

        fused_embeds: List[torch.Tensor] = []
        fused_labels: List[torch.Tensor] = []
        fused_attention: List[torch.Tensor] = []
        speech_idx = 0

        batch_size = input_ids.size(0)
        for batch_idx in range(batch_size):
            valid_mask = attn_mask[batch_idx]
            cur_ids = input_ids[batch_idx][valid_mask]
            cur_labels = labels[batch_idx][valid_mask] if labels is not None else None
            cur_speech_mask = speech_token_mask[batch_idx][valid_mask]

            speech_positions = torch.nonzero(cur_speech_mask, as_tuple=False).view(-1)
            expected = int(speech_counts[batch_idx].item())
            if int(speech_positions.numel()) != expected:
                raise RuntimeError("Mismatch between speech sentinel count and expected segments")

            split_indices = [-1] + speech_positions.tolist() + [cur_ids.shape[0]]
            token_segments: List[torch.Tensor] = []
            label_segments: List[torch.Tensor] = []
            for start, end in zip(split_indices[:-1], split_indices[1:]):
                seg = cur_ids[start + 1 : end]
                token_segments.append(seg)
                if cur_labels is not None:
                    label_segments.append(cur_labels[start + 1 : end])

            text_embed_segments: List[torch.Tensor] = []
            for seg in token_segments:
                if seg.numel() == 0:
                    text_embed_segments.append(torch.empty(0, embed_dim, device=embed_device, dtype=embed_dtype))
                else:
                    text_embed_segments.append(embed_tokens(seg))

            new_embeds: List[torch.Tensor] = []
            new_labels: List[torch.Tensor] = []
            new_attn: List[torch.Tensor] = []

            num_speech = expected
            for idx in range(num_speech + 1):
                text_seg = text_embed_segments[idx]
                new_embeds.append(text_seg)
                if cur_labels is not None:
                    new_labels.append(label_segments[idx])
                new_attn.append(torch.ones(text_seg.size(0), device=embed_device, dtype=torch.bool))

                if idx < num_speech:
                    speech_seg = speech_segments[speech_idx]
                    speech_idx += 1
                    new_embeds.append(speech_seg)
                    new_attn.append(torch.ones(speech_seg.size(0), device=embed_device, dtype=torch.bool))
                    if cur_labels is not None:
                        ignore_labels = torch.full(
                            (speech_seg.size(0),),
                            self.ignore_index,
                            device=cur_labels.device,
                            dtype=cur_labels.dtype,
                        )
                        new_labels.append(ignore_labels)

            fused_embeds.append(torch.cat(new_embeds) if new_embeds else torch.empty(0, embed_dim, device=embed_device, dtype=embed_dtype))
            fused_attention.append(torch.cat(new_attn) if new_attn else torch.empty(0, device=embed_device, dtype=torch.bool))
            if cur_labels is not None:
                fused_labels.append(torch.cat(new_labels) if new_labels else torch.empty(0, device=cur_labels.device, dtype=cur_labels.dtype))

        if speech_idx != total_speech_segments:
            raise RuntimeError("Unused speech features remain after stitching inputs")

        max_len = max(t.size(0) for t in fused_embeds)
        attn_device = attn_mask.device
        pos_device = input_ids.device

        padded_embeds = torch.zeros((batch_size, max_len, embed_dim), device=embed_device, dtype=embed_dtype)
        padded_attention = torch.zeros((batch_size, max_len), device=attn_device, dtype=torch.bool)
        padded_positions = torch.zeros((batch_size, max_len), device=pos_device, dtype=torch.long)

        padded_labels: Optional[torch.Tensor]
        if labels is not None:
            padded_labels = torch.full(
                (batch_size, max_len),
                self.ignore_index,
                device=labels.device,
                dtype=labels.dtype,
            )
        else:
            padded_labels = None

        for idx, embeds in enumerate(fused_embeds):
            cur_len = embeds.size(0)
            if cur_len == 0:
                continue
            padded_embeds[idx, :cur_len] = embeds
            padded_attention[idx, :cur_len] = fused_attention[idx]
            padded_positions[idx, :cur_len] = torch.arange(cur_len, device=pos_device, dtype=torch.long)
            if padded_labels is not None:
                padded_labels[idx, :cur_len] = fused_labels[idx]

        return None, padded_positions, padded_attention, past_key_values, padded_embeds, padded_labels


__all__ = [
    "SpeechMixin",
    "SpeechMixinForCausalLM",
]
