from __future__ import annotations

import torch
import torchaudio

from model.speech_encoder import WhisperFeatureEncoder


def _mono(waveform: torch.Tensor) -> torch.Tensor:
    if waveform.ndim == 1:
        return waveform
    if waveform.ndim == 2:
        return waveform.mean(dim=0)
    raise ValueError("Expected waveform with shape (time,) or (channels, time)")


def test_whisper_encoder_smoke_tiny_checkpoint(whisper_tiny_id, fixture_audio_paths, load_fixture_audio):
    # Use tiny checkpoint to keep test lightweight; assumes cache or download available.
    encoder = WhisperFeatureEncoder(model_id=whisper_tiny_id, device="cuda")

    waveform, sr = load_fixture_audio(fixture_audio_paths[0])
    waveform = _mono(waveform)

    feats, mask = encoder.encode([waveform], sampling_rate=sr)

    assert feats.shape[0] == 1
    assert mask.shape[0] == 1
    # mask length should match encoder frames
    assert mask.shape[1] == feats.shape[1]
    valid = int(mask.sum().item())
    assert 0 < valid <= feats.shape[1]
    assert feats.device == encoder.encoder.device
    assert feats.dtype == encoder.encoder.dtype


def test_whisper_encoder_resample_and_batch(whisper_tiny_id, fixture_audio_paths, load_fixture_audio):
    encoder = WhisperFeatureEncoder(model_id=whisper_tiny_id, device="cuda")

    waveform_a, sr_a = load_fixture_audio(fixture_audio_paths[0])
    waveform_a = _mono(waveform_a)
    waveform_a = waveform_a[: sr_a]  # ~1s

    waveform_b, sr_b = load_fixture_audio(fixture_audio_paths[1])
    waveform_b = _mono(waveform_b)
    sr_b = max(1, sr_b // 2)
    waveform_b = torchaudio.functional.resample(
        waveform_b,
        orig_freq=sr_b * 2,
        new_freq=sr_b,
    )
    waveform_b = waveform_b[: sr_b]  # ~1s at half rate

    feats, mask = encoder.encode([waveform_a, waveform_b], sampling_rate=[sr_a, sr_b])

    assert feats.shape[0] == 2
    assert mask.shape[0] == 2
    assert mask.shape[1] == feats.shape[1]
    assert int(mask[0].sum().item()) > 0
    assert int(mask[1].sum().item()) > 0
    assert feats.device == encoder.encoder.device
    assert feats.dtype == encoder.encoder.dtype


def test_whisper_encoder_rejects_empty_batch(whisper_tiny_id):
    encoder = WhisperFeatureEncoder(model_id=whisper_tiny_id, device="cuda")

    try:
        encoder.encode([])
    except ValueError:
        return
    raise AssertionError("Expected ValueError for empty waveform batch")


def test_whisper_encoder_rejects_invalid_sampling_rate(whisper_tiny_id, fixture_audio_paths, load_fixture_audio):
    encoder = WhisperFeatureEncoder(model_id=whisper_tiny_id, device="cuda")

    waveform, sr = load_fixture_audio(fixture_audio_paths[0])
    waveform = _mono(waveform)
    waveform = waveform[: sr]

    try:
        encoder.encode([waveform], sampling_rate=0)
    except ValueError:
        return
    raise AssertionError("Expected ValueError for non-positive sampling_rate")
