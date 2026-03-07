"""Offline one-turn inference pipeline built on shared decode helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torchaudio

from inference.inference import (
    build_one_turn_prompt,
    decode_thinker_one_pass,
    decode_talker_speech,
    extract_aligned_content,
    load_audio,
    resolve_speech_decode_config,
    resolve_text_decode_config,
)


@dataclass(frozen=True)
class InferenceRequest:
    """Represent one offline inference request for a single user audio clip."""

    audio_path: str
    decode_audio: bool = True
    text_generation_overrides: dict[str, object] | None = None
    speech_generation_overrides: dict[str, object] | None = None


@dataclass(frozen=True)
class InferenceResponse:
    """Represent one offline inference response with text and speech-unit outputs."""

    assistant_text: str
    assistant_text_token_ids: list[int]
    speech_token_ids: list[int]
    thinker_stop_seen: bool
    talker_eos_seen: bool
    conditioning_consumed: bool
    wav: torch.Tensor | None = None
    sample_rate: int | None = None
    wav_path: str | None = None


def run_turn(
    runtime: Any,
    cfg: dict[str, Any],
    request: InferenceRequest,
) -> InferenceResponse:
    """Run one full offline inference turn from user audio to speech unit ids."""
    # Fail fast when decode is requested without decoder assets.
    if bool(request.decode_audio) and getattr(runtime, "decoder", None) is None:
        raise ValueError(
            "decode_audio=true requires decoder assets; set models.cache_dir and ensure "
            "<cache_dir>/decoder contains decoder files"
        )

    input_wav, sampling_rate = load_audio(request.audio_path)
    device = runtime.device
    text_decode_cfg = resolve_text_decode_config(cfg, request.text_generation_overrides)
    speech_decode_cfg = resolve_speech_decode_config(cfg, request.speech_generation_overrides)

    with torch.inference_mode():
        input_wav = input_wav.to(device=device)
        prompt_ids = build_one_turn_prompt(
            runtime.tokenizer,
            speech_id=int(runtime.token_contract.speech_id),
        )

        # Decode Thinker and keep token-hidden alignment for Talker conditioning.
        thinker_result = decode_thinker_one_pass(
            thinker=runtime.thinker,
            token_contract=runtime.token_contract,
            prompt_ids=prompt_ids,
            waveform=input_wav,
            sampling_rate=int(sampling_rate),
            decode_cfg=text_decode_cfg,
            device=device,
        )
        content_ids, aligned_hidden = extract_aligned_content(
            generated_ids=thinker_result.generated_ids,
            hidden_rows=thinker_result.hidden_rows,
            tokenizer=runtime.tokenizer,
            assistant_stop_id=int(runtime.token_contract.assistant_stop_id),
        )

        # Talker fuse expects a batch, so pass one-row lists for one-turn API.
        fused_rows = runtime.talker.fuse(
            hidden_rows=[aligned_hidden],
            content_ids=[content_ids],
        )
        if len(fused_rows) != 1:
            raise RuntimeError(f"expected exactly one fused row, got {len(fused_rows)}")

        talker_result = decode_talker_speech(
            talker=runtime.talker,
            token_contract=runtime.token_contract,
            fused_row=fused_rows[0],
            decode_cfg=speech_decode_cfg,
            device=device,
        )

    # Keep unit ids as canonical speech output regardless of decode mode.
    speech_token_ids = [int(unit_id) for unit_id in talker_result.unit_ids]
    wav: torch.Tensor | None = None
    wav_sample_rate: int | None = None
    wav_path: str | None = None
    if bool(request.decode_audio):
        wav, wav_sample_rate = decode_wav(runtime=runtime, unit_ids=speech_token_ids)
        wav_path = save_wav_if_configured(waveform=wav, sample_rate=wav_sample_rate, cfg=cfg)

    assistant_text_token_ids = [int(token_id) for token_id in content_ids.detach().cpu().tolist()]
    assistant_text = runtime.tokenizer.decode(assistant_text_token_ids, skip_special_tokens=True).strip()
    return InferenceResponse(
        assistant_text=assistant_text,
        assistant_text_token_ids=assistant_text_token_ids,
        speech_token_ids=speech_token_ids,
        thinker_stop_seen=bool(thinker_result.stop_seen),
        talker_eos_seen=bool(talker_result.eos_seen),
        conditioning_consumed=bool(talker_result.conditioning_consumed),
        wav=wav,
        sample_rate=wav_sample_rate,
        wav_path=wav_path,
    )


def decode_wav(*, runtime: Any, unit_ids: list[int]) -> tuple[torch.Tensor, int]:
    """Decode raw speech unit ids to a waveform tensor and sample rate."""
    # Keep decode in pipeline so missing decoder fails before returning response.
    decoder = getattr(runtime, "decoder", None)
    if decoder is None:
        raise ValueError(
            "decode_audio=true requires decoder assets; set models.cache_dir and ensure "
            "<cache_dir>/decoder contains decoder files"
        )
    if not unit_ids:
        raise ValueError("cannot decode empty speech_token_ids")

    units = torch.tensor(unit_ids, dtype=torch.long)
    waveform = decoder.tokens_to_wav(units)
    if not isinstance(waveform, torch.Tensor) or waveform.ndim != 1 or int(waveform.numel()) <= 0:
        raise RuntimeError("decoder returned invalid waveform; expected rank-1 non-empty tensor")
    sample_rate = int(getattr(decoder, "sample_rate", 0))
    if sample_rate <= 0:
        raise RuntimeError("decoder returned invalid sample rate")
    return waveform.to(dtype=torch.float32).cpu(), sample_rate


def save_wav_if_configured(
    *,
    waveform: torch.Tensor,
    sample_rate: int,
    cfg: dict[str, Any],
) -> str | None:
    """Save decoded waveform to output_wav_path when configured."""
    output_cfg = cfg.get("output", {})
    output_wav_path = output_cfg.get("output_wav_path")
    if output_wav_path is None:
        return None

    output_path = Path(str(output_wav_path))
    # Ensure parent directories exist so configured output paths always work.
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if waveform.ndim != 1:
        raise ValueError(f"decoded waveform must be rank-1, got shape {tuple(waveform.shape)}")
    torchaudio.save(output_path.as_posix(), waveform.unsqueeze(0), int(sample_rate))
    return output_path.as_posix()


__all__ = [
    "InferenceRequest",
    "InferenceResponse",
    "run_turn",
]
