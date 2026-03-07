"""Whisper-based speech feature encoder."""

from __future__ import annotations

from typing import Optional, Sequence, Tuple, Union

import torch
import torchaudio.functional as taF
from torch import nn
from transformers import AutoProcessor, WhisperConfig, WhisperFeatureExtractor, WhisperModel

WaveformInput = Union[torch.Tensor, Sequence[Union[torch.Tensor, Sequence[float]]]]


class WhisperFeatureEncoder(nn.Module):
    """Minimal Whisper encoder wrapper that returns features and frame masks."""

    def __init__(
        self,
        model_id: str = "openai/whisper-large-v3",
        *,
        cache_dir: Optional[str] = None,
        device: Optional[Union[str, torch.device]] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        super().__init__()
        # Default path: initialize from a Whisper checkpoint id.
        self._init_from_pretrained(
            model_id=model_id,
            cache_dir=cache_dir,
            device=device,
            dtype=dtype,
        )

    # Build encoder+feature extractor from a HF checkpoint id.
    def _init_from_pretrained(
        self,
        *,
        model_id: str,
        cache_dir: Optional[str],
        device: Optional[Union[str, torch.device]],
        dtype: Optional[torch.dtype],
    ) -> None:
        processor = AutoProcessor.from_pretrained(model_id, cache_dir=cache_dir)
        extractor = getattr(processor, "feature_extractor", None)
        if extractor is None:
            raise ValueError("AutoProcessor did not provide a Whisper feature_extractor")

        load_kwargs = {"cache_dir": cache_dir}
        if dtype is not None:
            load_kwargs["torch_dtype"] = dtype
        whisper = WhisperModel.from_pretrained(model_id, **load_kwargs)
        self._bind_components(
            feature_extractor=extractor,
            encoder=whisper.encoder,
            hidden_size=int(whisper.config.d_model),
            device=device,
            dtype=dtype,
        )
        del whisper

    # Build encoder+feature extractor from serialized configs (no nested from_pretrained).
    @classmethod
    def from_configs(
        cls,
        *,
        whisper_config: dict,
        feature_extractor_config: dict,
        device: Optional[Union[str, torch.device]] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> "WhisperFeatureEncoder":
        instance = cls.__new__(cls)
        nn.Module.__init__(instance)
        model_config = WhisperConfig.from_dict(dict(whisper_config))
        whisper = WhisperModel(model_config)
        feature_extractor = WhisperFeatureExtractor.from_dict(dict(feature_extractor_config))
        instance._bind_components(
            feature_extractor=feature_extractor,
            encoder=whisper.encoder,
            hidden_size=int(model_config.d_model),
            device=device,
            dtype=dtype,
        )
        del whisper
        return instance

    # Attach encoder state and runtime preprocessing config in one place.
    def _bind_components(
        self,
        *,
        feature_extractor: WhisperFeatureExtractor,
        encoder: nn.Module,
        hidden_size: int,
        device: Optional[Union[str, torch.device]],
        dtype: Optional[torch.dtype],
    ) -> None:
        self.feature_extractor = feature_extractor
        self.encoder = encoder
        if device is not None:
            self.encoder = self.encoder.to(device=device)
        if dtype is not None:
            self.encoder = self.encoder.to(dtype=dtype)
        self.encoder.eval()
        for param in self.encoder.parameters():
            param.requires_grad = False

        self.hidden_size = int(hidden_size)
        self._sampling_rate = int(getattr(feature_extractor, "sampling_rate", 16000))
        self._hop_length = int(getattr(feature_extractor, "hop_length", self._sampling_rate // 100))

    # Export enough metadata to rebuild this encoder without remote loads.
    def to_config_dict(self) -> dict[str, dict]:
        return {
            "whisper_config": dict(self.encoder.config.to_dict()),
            "feature_extractor_config": dict(self.feature_extractor.to_dict()),
        }

    @property
    def device(self) -> torch.device:
        return next(self.encoder.parameters()).device

    @property
    def dtype(self) -> torch.dtype:
        return next(self.encoder.parameters()).dtype

    @property
    def sampling_rate(self) -> int:
        return self._sampling_rate

    def encode(
        self,
        waveforms: WaveformInput,
        *,
        sampling_rate: Optional[Union[int, Sequence[int]]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.forward(waveforms, sampling_rate=sampling_rate)

    @torch.no_grad()
    def forward(
        self,
        waveforms: WaveformInput,
        *,
        sampling_rate: Optional[Union[int, Sequence[int]]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch = self._normalize_waveforms(waveforms)
        resolved_srs = self._resolve_sampling_rates(sampling_rate, len(batch))

        target_sr = self.sampling_rate
        resampled = []
        sample_lengths = []
        for waveform, source_sr in zip(batch, resolved_srs):
            # Keep preprocessing on CPU/FP32 so NumPy and Whisper stay stable under autocast.
            audio = waveform.detach().to(device="cpu", dtype=torch.float32).contiguous()
            if source_sr != target_sr:
                audio = taF.resample(audio, source_sr, target_sr)
            resampled.append(audio.numpy())
            sample_lengths.append(int(audio.shape[0]))

        features = self.feature_extractor(
            raw_speech=resampled,
            sampling_rate=target_sr,
            return_tensors="pt",
        )
        input_features = features["input_features"].to(device=self.device, dtype=self.dtype)

        hidden_states = self.encoder(input_features=input_features).last_hidden_state
        frame_masks = self._frame_mask(sample_lengths, input_features, hidden_states)
        return hidden_states, frame_masks

    def _normalize_waveforms(self, waveforms: WaveformInput) -> list[torch.Tensor]:
        if isinstance(waveforms, torch.Tensor):
            if waveforms.ndim == 1:
                return [waveforms]
            if waveforms.ndim == 2:
                return [segment for segment in waveforms]
            raise ValueError("waveforms tensor must be 1-D or 2-D")

        if not isinstance(waveforms, Sequence):
            raise ValueError("waveforms must be a tensor or a sequence of tensors")

        batch: list[torch.Tensor] = []
        for wf in waveforms:
            tensor = torch.as_tensor(wf, dtype=torch.float32)
            if tensor.ndim != 1:
                raise ValueError("each waveform must be 1-D")
            batch.append(tensor.contiguous())
        if not batch:
            raise ValueError("waveforms batch is empty")
        return batch

    def _resolve_sampling_rates(
        self,
        sampling_rate: Optional[Union[int, Sequence[int]]],
        batch_size: int,
    ) -> list[int]:
        if sampling_rate is None:
            return [self.sampling_rate] * batch_size
        if isinstance(sampling_rate, Sequence) and not isinstance(sampling_rate, (str, bytes)):
            rates = [int(sr) for sr in sampling_rate]
            if len(rates) != batch_size:
                raise ValueError("sampling_rate list must match batch size")
            if any(sr <= 0 for sr in rates):
                raise ValueError("sampling_rate entries must be positive")
            return rates
        sr = int(sampling_rate)
        if sr <= 0:
            raise ValueError("sampling_rate must be positive")
        return [sr] * batch_size

    def _frame_mask(
        self,
        sample_lengths: Sequence[int],
        input_features: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        """Compute valid encoder frames from audio lengths and encoder stride."""

        if len(sample_lengths) != hidden_states.size(0):
            raise ValueError("sample_lengths must match batch size")

        device = hidden_states.device
        lengths = torch.as_tensor(sample_lengths, device=device, dtype=torch.long)

        hop = int(self._hop_length)
        feat_frames = torch.div(lengths + hop - 1, hop, rounding_mode="floor")

        # Derive stride between feature frames and encoder frames from shapes.
        feature_steps = input_features.shape[-1]
        encoder_steps = hidden_states.shape[1]
        stride = max(1.0, feature_steps / float(encoder_steps))

        valid_frames = torch.ceil(feat_frames / stride).to(torch.long)
        max_frames = encoder_steps
        valid_frames = torch.clamp(valid_frames, max=max_frames)

        frame_range = torch.arange(max_frames, device=device)
        mask = frame_range.unsqueeze(0) < valid_frames.unsqueeze(1)
        return mask
