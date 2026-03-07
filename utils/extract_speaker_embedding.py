#!/usr/bin/env python3
"""Extract one fixed CosyVoice2 speaker embedding from a prompt wav."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torchaudio

TARGET_SAMPLE_RATE = 16_000


@dataclass(frozen=True)
class ExtractionPaths:
    """Hold resolved input/output paths for one embedding extraction run."""

    prompt_wav: Path
    campplus_onnx: Path
    output_path: Path


def _parse_args() -> argparse.Namespace:
    """Parse CLI flags for one-time speaker embedding extraction."""
    parser = argparse.ArgumentParser(
        description="Extract a fixed speaker embedding (speaker_embedding.pt) for CosyVoice2 decoding.",
    )
    parser.add_argument(
        "--prompt-wav",
        required=True,
        type=Path,
        help="Reference voice wav file used to extract speaker identity.",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=None,
        help="Decoder asset directory that contains campplus.onnx (e.g. models/cosyvoice2).",
    )
    parser.add_argument(
        "--campplus-onnx",
        type=Path,
        default=None,
        help="Path to campplus.onnx. If omitted, defaults to <model-dir>/campplus.onnx.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=None,
        help="Output .pt path. If omitted, defaults to <model-dir>/speaker_embedding.pt.",
    )
    return parser.parse_args()


def _require_file(path: Path, *, label: str) -> Path:
    """Validate that one required filesystem input exists as a file."""
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"{label} not found: {resolved}")
    return resolved


def _resolve_paths(args: argparse.Namespace) -> ExtractionPaths:
    """Resolve and validate all paths used by extraction."""
    prompt_wav = _require_file(args.prompt_wav, label="prompt wav")
    model_dir = args.model_dir.expanduser().resolve() if args.model_dir is not None else None
    if model_dir is not None and not model_dir.is_dir():
        raise FileNotFoundError(f"model dir not found: {model_dir}")

    if args.campplus_onnx is not None:
        campplus_onnx = _require_file(args.campplus_onnx, label="campplus onnx")
    else:
        if model_dir is None:
            raise ValueError("provide --campplus-onnx or --model-dir")
        campplus_onnx = _require_file(model_dir / "campplus.onnx", label="campplus onnx")

    if args.output_path is not None:
        output_path = args.output_path.expanduser().resolve()
    else:
        if model_dir is None:
            raise ValueError("provide --output-path or --model-dir")
        output_path = (model_dir / "speaker_embedding.pt").resolve()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    return ExtractionPaths(prompt_wav=prompt_wav, campplus_onnx=campplus_onnx, output_path=output_path)


def _load_prompt_wav(path: Path) -> torch.Tensor:
    """Load audio, collapse to mono, and resample to 16 kHz for CAMPPlus."""
    wav, sample_rate = torchaudio.load(path.as_posix())
    if wav.ndim != 2 or wav.shape[0] <= 0 or wav.shape[1] <= 0:
        raise ValueError(f"invalid wav shape from {path}: {tuple(wav.shape)}")

    # CAMPPlus expects single-channel audio.
    wav = wav.mean(dim=0, keepdim=True)

    # Project policy: fail fast on <16kHz inputs to avoid hiding poor prompt quality.
    # If we later want broader compatibility, we can remove this guard and allow upsampling.
    if sample_rate < TARGET_SAMPLE_RATE:
        raise ValueError(
            f"prompt wav sample rate must be >= {TARGET_SAMPLE_RATE} Hz, got {sample_rate} Hz: {path}"
        )

    # CAMPPlus expects 16 kHz, so only downsample when needed.
    if sample_rate > TARGET_SAMPLE_RATE:
        wav = torchaudio.functional.resample(
            wav,
            orig_freq=sample_rate,
            new_freq=TARGET_SAMPLE_RATE,
        )
    return wav.to(dtype=torch.float32)


def _compute_spk_fbank(wav: torch.Tensor) -> torch.Tensor:
    """Compute CAMPPlus-style fbank features and shape to [1, T, 80]."""
    if wav.ndim != 2 or wav.shape[0] != 1:
        raise ValueError(f"expected mono wav shape [1, num_samples], got {tuple(wav.shape)}")

    # Match CosyVoice CAMPPlus frontend exactly: deterministic 80-bin fbank.
    feat = torchaudio.compliance.kaldi.fbank(
        wav,
        num_mel_bins=80,
        frame_length=25,
        frame_shift=10,
        sample_frequency=TARGET_SAMPLE_RATE,
        dither=0.0,
    )
    if feat.ndim != 2 or feat.shape[1] != 80:
        raise ValueError(f"expected fbank shape [T, 80], got {tuple(feat.shape)}")
    if feat.shape[0] == 0:
        raise ValueError("prompt wav produced zero fbank frames")

    # Mean-normalize per feature dim to match upstream extraction.
    feat = feat - feat.mean(dim=0, keepdim=True)

    return feat.unsqueeze(0).to(dtype=torch.float32)


def _run_campplus_onnx(campplus_onnx: Path, feat: torch.Tensor) -> torch.Tensor:
    """Run CAMPPlus ONNX model and return raw speaker embedding [1, 192]."""
    try:
        import onnxruntime as ort
    except ImportError as exc:
        raise ImportError(
            "onnxruntime is required for this one-time script. Install with: pip install onnxruntime"
        ) from exc

    session = ort.InferenceSession(campplus_onnx.as_posix(), providers=["CPUExecutionProvider"])
    inputs = session.get_inputs()
    if len(inputs) != 1:
        raise ValueError(f"expected campplus.onnx with 1 input, found {len(inputs)}")
    input_name = inputs[0].name
    outputs = session.run(None, {input_name: feat.cpu().numpy()})
    if not outputs:
        raise RuntimeError("campplus.onnx returned no outputs")
    embedding_np = outputs[0]
    if not isinstance(embedding_np, np.ndarray):
        raise TypeError(f"expected numpy.ndarray output, got {type(embedding_np).__name__}")
    embedding = torch.from_numpy(embedding_np).to(dtype=torch.float32)
    if embedding.ndim == 1:
        embedding = embedding.unsqueeze(0)
    return embedding


def _validate_embedding(embedding: torch.Tensor) -> torch.Tensor:
    """Validate final speaker embedding tensor and return canonical CPU float32 copy."""
    if embedding.ndim != 2 or embedding.shape[0] != 1 or embedding.shape[1] != 192:
        raise ValueError(f"expected embedding shape [1, 192], got {tuple(embedding.shape)}")
    if not torch.isfinite(embedding).all():
        raise ValueError("embedding contains non-finite values")

    # Save raw CAMPPlus output; flow performs speaker normalization at inference time.
    return embedding.to(dtype=torch.float32, device="cpu").contiguous()


def main() -> None:
    """Extract one validated speaker embedding tensor and save it to disk."""
    args = _parse_args()
    paths = _resolve_paths(args)
    wav = _load_prompt_wav(paths.prompt_wav)
    feat = _compute_spk_fbank(wav)
    embedding = _run_campplus_onnx(paths.campplus_onnx, feat)
    embedding = _validate_embedding(embedding)
    torch.save(embedding, paths.output_path)

    duration_sec = float(wav.shape[1]) / float(TARGET_SAMPLE_RATE)

    print(f"prompt_wav: {paths.prompt_wav}")
    print(f"campplus_onnx: {paths.campplus_onnx}")
    print(f"output_path: {paths.output_path}")
    print(f"waveform_shape: {tuple(wav.shape)}")
    print(f"sample_rate: {TARGET_SAMPLE_RATE}")
    print(f"duration_sec: {duration_sec:.3f}")
    print(f"fbank_shape: {tuple(feat.shape)}")
    print(f"embedding_shape: {tuple(embedding.shape)}")
    print(f"embedding_dtype: {embedding.dtype}")


if __name__ == "__main__":
    main()
