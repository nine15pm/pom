"""Minimal CosyVoice2 speech-token decoder wrapper."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Final

import torch
from torch import nn
from omegaconf import OmegaConf

from cosyvoice2.flow.decoder import CausalConditionalDecoder
from cosyvoice2.flow.flow import CausalMaskedDiffWithXvec
from cosyvoice2.flow.flow_matching import CausalConditionalCFM
from cosyvoice2.hifigan.f0_predictor import ConvRNNF0Predictor
from cosyvoice2.hifigan.generator import HiFTGenerator
from cosyvoice2.transformer.upsample_encoder import UpsampleConformerEncoder

# Shared CosyVoice2 speech-token range used across data and decoding.
SPEECH_TOKEN_MIN: Final[int] = 0
SPEECH_TOKEN_MAX: Final[int] = 6560

# Fixed CosyVoice2 decoder audio settings from cosyvoice2.yaml.
SAMPLE_RATE: Final[int] = 24_000
TOKEN_FRAME_RATE: Final[int] = 25
TOKEN_MEL_RATIO: Final[int] = 2
STREAM_CHUNK_SIZE: Final[int] = 25
MEL_CACHE_FRAMES: Final[int] = 8
DEFAULT_FLOW_LOOKAHEAD_TOKENS: Final[int] = 3
DEFAULT_SAMPLES_PER_MEL_FRAME: Final[int] = 480


@dataclass(frozen=True)
class DecoderAssetPaths:
    """Resolve runtime checkpoint paths from one decoder model directory."""

    model_dir: Path

    @property
    def flow(self) -> Path:
        """Return the flow checkpoint path."""
        return self.model_dir / "flow.pt"

    @property
    def hift(self) -> Path:
        """Return the vocoder checkpoint path."""
        return self.model_dir / "hift.pt"

    @property
    def speaker_embedding(self) -> Path:
        """Return the fixed speaker embedding path."""
        return self.model_dir / "speaker_embedding.pt"


@dataclass(frozen=True)
class DecoderStreamState:
    """Keep rolling decoder state for one streaming decode session."""

    unit_ids: torch.Tensor
    token_offset: int
    mel_cache: torch.Tensor | None
    source_cache: torch.Tensor | None
    speech_cache: torch.Tensor | None


def _build_flow_model() -> CausalMaskedDiffWithXvec:
    """Construct the frozen flow model graph for CosyVoice2-0.5B."""
    encoder = UpsampleConformerEncoder(
        input_size=512,
        output_size=512,
        attention_heads=8,
        linear_units=2048,
        num_blocks=6,
        dropout_rate=0.1,
        positional_dropout_rate=0.1,
        attention_dropout_rate=0.1,
        normalize_before=True,
        input_layer="linear",
        pos_enc_layer_type="rel_pos_espnet",
        selfattention_layer_type="rel_selfattn",
        use_cnn_module=False,
        macaron_style=False,
        static_chunk_size=STREAM_CHUNK_SIZE,
    )

    estimator = CausalConditionalDecoder(
        in_channels=320,
        out_channels=80,
        channels=[256],
        dropout=0.0,
        attention_head_dim=64,
        n_blocks=4,
        num_mid_blocks=12,
        num_heads=8,
        act_fn="gelu",
        static_chunk_size=STREAM_CHUNK_SIZE * TOKEN_MEL_RATIO,
        num_decoding_left_chunks=-1,
    )

    cfm_params = OmegaConf.create(
        {
            "sigma_min": 1e-6,
            "solver": "euler",
            "t_scheduler": "cosine",
            "training_cfg_rate": 0.2,
            "inference_cfg_rate": 0.7,
            "reg_loss_type": "l1",
        }
    )
    decoder = CausalConditionalCFM(
        in_channels=240,
        n_spks=1,
        spk_emb_dim=80,
        cfm_params=cfm_params,
        estimator=estimator,
    )

    return CausalMaskedDiffWithXvec(
        input_size=512,
        output_size=80,
        spk_embed_dim=192,
        output_type="mel",
        vocab_size=SPEECH_TOKEN_MAX + 1,
        input_frame_rate=TOKEN_FRAME_RATE,
        only_mask_loss=True,
        token_mel_ratio=TOKEN_MEL_RATIO,
        pre_lookahead_len=3,
        encoder=encoder,
        decoder=decoder,
    )


def _build_hift_model() -> HiFTGenerator:
    """Construct the frozen HiFT vocoder graph for CosyVoice2-0.5B."""
    f0_predictor = ConvRNNF0Predictor(num_class=1, in_channels=80, cond_channels=512)
    return HiFTGenerator(
        in_channels=80,
        base_channels=512,
        nb_harmonics=8,
        sampling_rate=SAMPLE_RATE,
        nsf_alpha=0.1,
        nsf_sigma=0.003,
        nsf_voiced_threshold=10,
        upsample_rates=[8, 5, 3],
        upsample_kernel_sizes=[16, 11, 7],
        istft_params={"n_fft": 16, "hop_len": 4},
        resblock_kernel_sizes=[3, 7, 11],
        resblock_dilation_sizes=[[1, 3, 5], [1, 3, 5], [1, 3, 5]],
        source_resblock_kernel_sizes=[7, 7, 11],
        source_resblock_dilation_sizes=[[1, 3, 5], [1, 3, 5], [1, 3, 5]],
        lrelu_slope=0.1,
        audio_limit=0.99,
        f0_predictor=f0_predictor,
    )


def _load_checkpoint(path: Path, *, device: torch.device) -> dict[str, torch.Tensor]:
    """Load one checkpoint file and normalize it to a plain state_dict."""
    if not path.is_file():
        raise FileNotFoundError(f"decoder checkpoint not found: {path}")
    payload = torch.load(path, map_location=device)
    if isinstance(payload, dict) and "state_dict" in payload:
        payload = payload["state_dict"]
    if not isinstance(payload, dict):
        raise TypeError(f"expected state_dict-like checkpoint at {path}, got {type(payload).__name__}")
    return payload


def _normalize_hift_keys(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Strip optional 'generator.' prefix so HiFT strict-loading matches module keys."""
    normalized: dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        normalized[key.removeprefix("generator.")] = value
    return normalized


def _load_speaker_embedding(path: Path, *, device: torch.device) -> torch.Tensor:
    """Load a fixed speaker embedding as shape [1, 192] float32 on target device."""
    if not path.is_file():
        raise FileNotFoundError(f"speaker embedding not found: {path}")
    embedding = torch.load(path, map_location=device)
    if not isinstance(embedding, torch.Tensor):
        raise TypeError(f"expected Tensor at {path}, got {type(embedding).__name__}")
    if embedding.ndim == 1:
        embedding = embedding.unsqueeze(0)
    if embedding.ndim != 2 or embedding.shape[0] != 1 or embedding.shape[1] != 192:
        raise ValueError(f"expected speaker embedding shape [1, 192], got {tuple(embedding.shape)}")
    return embedding.to(device=device, dtype=torch.float32)


class SpeechDecoder:
    """Decode CosyVoice2 speech token ids into waveform audio."""

    def __init__(self, model_dir: str | Path, *, device: str | torch.device = "cuda") -> None:
        """Build decoder modules and load frozen pretrained weights."""
        self.device = torch.device(device)
        self.model_dir = Path(model_dir)
        self.assets = DecoderAssetPaths(self.model_dir)
        self.sample_rate = SAMPLE_RATE

        # Build module graphs first so strict loading can catch any architecture mismatch.
        self.flow = _build_flow_model().to(self.device)
        self.hift = _build_hift_model().to(self.device)

        # Load flow weights with strict key/shape checking.
        flow_state = _load_checkpoint(self.assets.flow, device=self.device)
        self.flow.load_state_dict(flow_state, strict=True)

        # Load HiFT weights and strip the saved generator namespace prefix.
        hift_state = _load_checkpoint(self.assets.hift, device=self.device)
        self.hift.load_state_dict(_normalize_hift_keys(hift_state), strict=True)

        # Load one fixed speaker embedding used for all response decoding.
        self.speaker_embedding = _load_speaker_embedding(self.assets.speaker_embedding, device=self.device)

        # Run decoder in pure inference mode.
        self.flow.eval()
        self.hift.eval()

        for module in (self.flow, self.hift):
            for parameter in module.parameters():
                parameter.requires_grad_(False)

        # Keep stream settings explicit and colocated with the loaded model.
        self._token_mel_ratio = int(getattr(self.flow, "token_mel_ratio", TOKEN_MEL_RATIO))
        self._pre_lookahead_len = int(
            getattr(self.flow, "pre_lookahead_len", DEFAULT_FLOW_LOOKAHEAD_TOKENS)
        )
        self._mel_cache_frames = int(MEL_CACHE_FRAMES)
        self._samples_per_mel_frame = int(
            round(float(getattr(self.hift.f0_upsamp, "scale_factor", DEFAULT_SAMPLES_PER_MEL_FRAME)))
        )
        self._source_cache_samples = int(self._mel_cache_frames * self._samples_per_mel_frame)

    def _validate_unit_range(self, unit_ids: torch.Tensor) -> None:
        """Validate raw unit id bounds so decode always sees [0, 6560]."""
        if unit_ids.numel() == 0:
            return
        token_min = int(unit_ids.min().item())
        token_max = int(unit_ids.max().item())
        if token_min < SPEECH_TOKEN_MIN or token_max > SPEECH_TOKEN_MAX:
            detail = ""
            if token_min >= SPEECH_TOKEN_MAX + 1:
                detail = " This looks like LM-offset token ids; pass raw speech unit ids in [0, 6560]."
            raise ValueError(
                f"speech token id out of range: min={token_min}, max={token_max}, "
                f"expected [{SPEECH_TOKEN_MIN}, {SPEECH_TOKEN_MAX}].{detail}"
            )

    def _normalize_speech_tokens(self, speech_tokens: torch.Tensor) -> torch.Tensor:
        """Validate token ids and normalize to shape [1, T] long on decoder device."""
        if not isinstance(speech_tokens, torch.Tensor):
            raise TypeError(f"speech_tokens must be a torch.Tensor, got {type(speech_tokens).__name__}")
        if speech_tokens.ndim == 1:
            tokens = speech_tokens.unsqueeze(0)
        elif speech_tokens.ndim == 2 and speech_tokens.shape[0] == 1:
            tokens = speech_tokens
        else:
            raise ValueError(f"speech_tokens must be shape [T] or [1, T], got {tuple(speech_tokens.shape)}")
        if tokens.numel() == 0:
            raise ValueError("speech_tokens must be non-empty")
        # Restrict to wide integer dtypes to avoid silent 8-bit overflow/wrap footguns.
        if tokens.dtype not in {torch.int32, torch.int64, torch.long}:
            raise TypeError(f"speech_tokens must use int32 or int64 dtype, got {tokens.dtype}")
        self._validate_unit_range(tokens)
        return tokens.to(device=self.device, dtype=torch.long)

    def _normalize_stream_chunk(self, unit_ids: torch.Tensor | list[int]) -> torch.Tensor:
        """Normalize one streamed unit chunk to rank-1 int64 on decoder device."""
        chunk = torch.as_tensor(unit_ids)
        if chunk.ndim != 1:
            raise ValueError(f"stream unit chunk must be rank-1, got shape {tuple(chunk.shape)}")
        # Accept empty finalize chunks and normalize them to int64 on decoder device.
        if chunk.numel() == 0:
            return chunk.to(device=self.device, dtype=torch.long)
        if chunk.dtype not in {torch.int32, torch.int64, torch.long}:
            raise TypeError(f"stream unit chunk must use int32 or int64 dtype, got {chunk.dtype}")
        chunk = chunk.to(device=self.device, dtype=torch.long)
        self._validate_unit_range(chunk)
        return chunk

    def _empty_prompt(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Build fixed empty prompt tensors used by this single-voice decoder wrapper."""
        prompt_token = torch.zeros((1, 0), device=self.device, dtype=torch.long)
        prompt_token_len = torch.tensor([0], device=self.device, dtype=torch.int32)
        prompt_feat = torch.zeros((1, 0, 80), device=self.device, dtype=torch.float32)
        prompt_feat_len = torch.tensor([0], device=self.device, dtype=torch.int32)
        return prompt_token, prompt_token_len, prompt_feat, prompt_feat_len

    def _crossfade_boundary(self, current: torch.Tensor, previous_tail: torch.Tensor) -> torch.Tensor:
        """Crossfade overlap so adjacent streamed chunks join without hard edges."""
        if current.ndim != 2 or previous_tail.ndim != 2:
            return current
        overlap = min(int(current.shape[1]), int(previous_tail.shape[1]), int(self._source_cache_samples))
        if overlap <= 0:
            return current
        window = torch.hamming_window(2 * overlap, periodic=False, device=current.device, dtype=current.dtype)
        mixed = current.clone()
        mixed[:, :overlap] = (
            mixed[:, :overlap] * window[:overlap]
            + previous_tail[:, -overlap:] * window[overlap:]
        )
        return mixed

    def create_stream_state(self) -> DecoderStreamState:
        """Create an empty stream state before decoding any unit chunks."""
        return DecoderStreamState(
            unit_ids=torch.empty((0,), device=self.device, dtype=torch.long),
            token_offset=0,
            mel_cache=None,
            source_cache=None,
            speech_cache=None,
        )

    @torch.inference_mode()
    def decode_unit_chunk(
        self,
        *,
        state: DecoderStreamState,
        new_unit_ids: torch.Tensor | list[int],
        finalize: bool,
    ) -> tuple[torch.Tensor, DecoderStreamState]:
        """Decode one streamed unit chunk and return only newly playable waveform samples."""
        chunk = self._normalize_stream_chunk(new_unit_ids)
        if chunk.numel() == 0 and not bool(finalize):
            raise ValueError("stream unit chunk must be non-empty unless finalize=true")

        # Keep full unit history because flow inference consumes the full token prefix.
        all_unit_ids = torch.cat([state.unit_ids, chunk], dim=0) if chunk.numel() > 0 else state.unit_ids
        if all_unit_ids.numel() == 0:
            raise ValueError("cannot finalize an empty stream without any unit ids")

        # Hold back tiny prefixes so flow lookahead has enough context during streaming.
        if not bool(finalize) and int(all_unit_ids.numel()) <= int(self._pre_lookahead_len):
            return torch.empty((0,), dtype=torch.float32), DecoderStreamState(
                unit_ids=all_unit_ids,
                token_offset=state.token_offset,
                mel_cache=state.mel_cache,
                source_cache=state.source_cache,
                speech_cache=state.speech_cache,
            )

        prompt_token, prompt_token_len, prompt_feat, prompt_feat_len = self._empty_prompt()
        tokens = all_unit_ids.unsqueeze(0)
        token_len = torch.tensor([int(tokens.shape[1])], device=self.device, dtype=torch.int32)
        mel, _ = self.flow.inference(
            token=tokens,
            token_len=token_len,
            prompt_token=prompt_token,
            prompt_token_len=prompt_token_len,
            prompt_feat=prompt_feat,
            prompt_feat_len=prompt_feat_len,
            embedding=self.speaker_embedding,
            streaming=False,
            finalize=bool(finalize),
        )

        # Convert prior token offset to mel-frame offset before running the vocoder.
        mel_offset = int(state.token_offset) * int(self._token_mel_ratio)
        if mel_offset >= int(mel.shape[2]):
            # Finalize should always return a cleared stream state, even when no new mel appears.
            if bool(finalize):
                return torch.empty((0,), dtype=torch.float32), DecoderStreamState(
                    unit_ids=all_unit_ids,
                    token_offset=int(all_unit_ids.numel()),
                    mel_cache=None,
                    source_cache=None,
                    speech_cache=None,
                )
            return torch.empty((0,), dtype=torch.float32), DecoderStreamState(
                unit_ids=all_unit_ids,
                token_offset=state.token_offset,
                mel_cache=state.mel_cache,
                source_cache=state.source_cache,
                speech_cache=state.speech_cache,
            )
        mel = mel[:, :, mel_offset:]
        if state.mel_cache is not None and int(state.mel_cache.numel()) > 0:
            mel = torch.cat([state.mel_cache, mel], dim=2)

        # Reuse cached source excitation to smooth chunk boundaries in HiFT.
        source_cache = state.source_cache
        if source_cache is None:
            source_cache = torch.zeros((1, 1, 0), device=self.device, dtype=mel.dtype)
        speech, source = self.hift.inference(speech_feat=mel, cache_source=source_cache)
        if state.speech_cache is not None and int(state.speech_cache.numel()) > 0:
            speech = self._crossfade_boundary(speech, state.speech_cache)

        if bool(finalize):
            next_state = DecoderStreamState(
                unit_ids=all_unit_ids,
                token_offset=int(all_unit_ids.numel()),
                mel_cache=None,
                source_cache=None,
                speech_cache=None,
            )
            return speech.squeeze(0).to(dtype=torch.float32).cpu(), next_state

        # Keep overlap caches and emit only the strictly new non-overlap audio region.
        keep = int(self._source_cache_samples)
        emitted = speech[:, :-keep] if int(speech.shape[1]) > keep else torch.zeros((1, 0), device=self.device)
        next_state = DecoderStreamState(
            unit_ids=all_unit_ids,
            token_offset=max(0, int(all_unit_ids.numel()) - int(self._pre_lookahead_len)),
            mel_cache=mel[:, :, -int(self._mel_cache_frames) :],
            source_cache=source[:, :, -keep:],
            speech_cache=speech[:, -keep:],
        )
        return emitted.squeeze(0).to(dtype=torch.float32).cpu(), next_state

    @torch.inference_mode()
    def tokens_to_wav(self, speech_tokens: torch.Tensor) -> torch.Tensor:
        """Decode one token sequence into a single-channel waveform tensor [num_samples]."""
        tokens = self._normalize_speech_tokens(speech_tokens)
        token_len = torch.tensor([tokens.shape[1]], device=self.device, dtype=torch.int32)

        # Use empty prompt features/tokens because this wrapper is fixed-voice non-streaming decode.
        prompt_token, prompt_token_len, prompt_feat, prompt_feat_len = self._empty_prompt()

        # Flow model maps speech tokens to mel frames.
        mel, _ = self.flow.inference(
            token=tokens,
            token_len=token_len,
            prompt_token=prompt_token,
            prompt_token_len=prompt_token_len,
            prompt_feat=prompt_feat,
            prompt_feat_len=prompt_feat_len,
            embedding=self.speaker_embedding,
            streaming=False,
            finalize=True,
        )

        # HiFT vocoder maps mel frames to waveform.
        wav, _ = self.hift.inference(speech_feat=mel)
        return wav.squeeze(0).to(dtype=torch.float32).cpu()


__all__ = [
    "DecoderStreamState",
    "SpeechDecoder",
    "SAMPLE_RATE",
    "SPEECH_TOKEN_MIN",
    "SPEECH_TOKEN_MAX",
]
