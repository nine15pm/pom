"""Run single-clip SU inference from a chosen Thinker checkpoint."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict

import torch
import torchaudio
import yaml

# Add repo root so local imports work when script is run via utils/ path.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _parse_args() -> argparse.Namespace:
    """Parse CLI arguments for single-clip SU inference."""
    parser = argparse.ArgumentParser(description="SU single-clip inference")
    parser.add_argument("--config", type=str, required=True, help="Path to SU YAML config")
    parser.add_argument("--audio-path", type=str, required=True, help="Path to input speech clip")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="latest",
        help="Checkpoint path or 'latest' (default: latest)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.5,
        help="Sampling temperature (default: 0.5)",
    )
    parser.add_argument("--top-p", type=float, default=0.9, help="Nucleus sampling p (default: 0.9)")
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=96,
        help="Maximum new tokens to generate (default: 96)",
    )
    thinking_group = parser.add_mutually_exclusive_group()
    thinking_group.add_argument(
        "--disable-thinking",
        dest="disable_thinking",
        action="store_true",
        help="Disable Qwen chat-template thinking tags (default)",
    )
    thinking_group.add_argument(
        "--enable-thinking",
        dest="disable_thinking",
        action="store_false",
        help="Enable Qwen chat-template thinking tags",
    )
    parser.set_defaults(disable_thinking=True)
    return parser.parse_args()


def _load_config(path: str) -> Dict[str, Any]:
    """Load a YAML config into a top-level mapping."""
    cfg_path = Path(path)
    if not cfg_path.exists():
        raise FileNotFoundError(f"config not found: {cfg_path}")
    with cfg_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError("config must be a YAML mapping at the top level")
    return data


def _get_cfg_section(cfg: Dict[str, Any], key: str) -> Dict[str, Any]:
    """Read a config section and return an empty mapping when missing."""
    section = cfg.get(key, {})
    if section is None:
        return {}
    if not isinstance(section, dict):
        raise ValueError(f"config section {key!r} must be a mapping")
    return section


def _resolve_checkpoint_dir(cfg: Dict[str, Any], checkpoint: str) -> Path:
    """Resolve checkpoint input to a concrete step directory."""
    if checkpoint != "latest":
        path = Path(checkpoint)
        if not path.is_dir():
            raise FileNotFoundError(f"checkpoint not found: {path}")
        return path

    # Use training.output_dir/checkpoints for default latest checkpoint lookup.
    train_cfg = _get_cfg_section(cfg, "training")
    output_dir = train_cfg.get("output_dir")
    if not output_dir:
        raise ValueError("training.output_dir must be set when --checkpoint=latest")
    checkpoints_root = Path(str(output_dir)) / "checkpoints"
    candidates = sorted(path for path in checkpoints_root.glob("step-*") if path.is_dir())
    if not candidates:
        raise FileNotFoundError(f"no step-* checkpoints found under: {checkpoints_root}")
    return candidates[-1]


def _resolve_dtype(precision: str) -> torch.dtype | None:
    """Map training precision to a torch dtype for inference loading."""
    precision = precision.lower()
    if precision == "bf16":
        return torch.bfloat16
    if precision == "no":
        return None
    raise ValueError("unsupported precision; use bf16 or no")


def _drop_none(data: Dict[str, Any]) -> Dict[str, Any]:
    """Drop None values so builder defaults apply cleanly."""
    return {key: value for key, value in data.items() if value is not None}


def _ensure_pad_token(tokenizer) -> None:
    """Require a real pad token id for generation-time padding."""
    # Keep pad semantics separate from EOS/assistant turn-end behavior.
    if tokenizer.pad_token_id is None:
        raise ValueError("tokenizer must define pad_token_id")


def _load_model_state(checkpoint_dir: Path) -> Dict[str, torch.Tensor]:
    """Load model weights from an Accelerate checkpoint directory."""
    bin_path = checkpoint_dir / "pytorch_model.bin"
    safe_path = checkpoint_dir / "model.safetensors"
    if bin_path.exists():
        return torch.load(bin_path, map_location="cpu")
    if safe_path.exists():
        try:
            from safetensors.torch import load_file
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("safetensors is required to load model.safetensors") from exc
        return load_file(str(safe_path))
    raise FileNotFoundError(f"no model checkpoint found in: {checkpoint_dir}")


def _align_speech_modules(model: torch.nn.Module) -> None:
    """Move speech modules to match the LM embedding device and dtype."""
    embed = model.get_input_embeddings()
    base_device = embed.weight.device
    base_dtype = embed.weight.dtype
    encoder = model.get_speech_encoder()
    projector = model.get_speech_projector()
    if encoder is not None:
        encoder.to(device=base_device, dtype=base_dtype)
    if projector is not None:
        projector.to(device=base_device, dtype=base_dtype)


def _load_audio_clip(path: Path) -> tuple[torch.Tensor, int]:
    """Load one audio clip as mono float32 waveform without manual resampling."""
    waveform, sampling_rate = torchaudio.load(str(path))
    if waveform.ndim != 2:
        raise ValueError("torchaudio returned unexpected shape")
    if waveform.size(0) > 1:
        # Mix multi-channel inputs to mono for whisper preprocessing.
        waveform = waveform.mean(dim=0, keepdim=True)
    return waveform.squeeze(0).to(dtype=torch.float32), int(sampling_rate)


def _build_prompt(
    tokenizer,
    *,
    speech_token_id: int,
    waveform: torch.Tensor,
    sampling_rate: int,
    disable_thinking: bool,
) -> tuple[list[int], list[torch.Tensor], list[int]]:
    """Build a one-turn <speech> prompt and aligned waveform inputs."""
    from train.su_sequence_builder import build_su_messages, build_su_token_ids

    # Use the shared builder so inference matches training exactly.
    history = [{"role": "user", "text": None, "audio": {"array": waveform, "sampling_rate": sampling_rate}}]
    messages, waveforms, sampling_rates = build_su_messages(history)
    prompt_ids, _ = build_su_token_ids(
        tokenizer,
        messages,
        enable_thinking=False if disable_thinking else None,
    )

    speech_tokens = sum(1 for token in prompt_ids if token == speech_token_id)
    if speech_tokens != 1:
        raise ValueError(f"expected exactly one <speech> token in prompt, got {speech_tokens}")

    return prompt_ids, waveforms, sampling_rates


def _validate_generation_args(args: argparse.Namespace) -> None:
    """Validate generation settings for sampled decoding."""
    if args.max_new_tokens <= 0:
        raise ValueError("--max-new-tokens must be > 0")
    if args.temperature <= 0:
        raise ValueError("--temperature must be > 0 for sampled decoding")
    if not (0 < args.top_p <= 1.0):
        raise ValueError("--top-p must be in the range (0, 1]")


def _sample_next_token(logits: torch.Tensor, *, temperature: float, top_p: float) -> torch.Tensor:
    """Sample next token id from logits with temperature and nucleus filtering."""
    # Scale logits for controllable output randomness.
    scaled = logits / float(temperature)
    probs = torch.softmax(scaled, dim=-1)

    # Fast path: full-vocab sampling when top_p keeps all mass.
    if top_p >= 1.0:
        return torch.multinomial(probs, num_samples=1)

    # Keep the smallest sorted token set whose cumulative mass reaches top_p.
    sorted_probs, sorted_indices = torch.sort(probs, dim=-1, descending=True)
    cumsum_probs = torch.cumsum(sorted_probs, dim=-1)
    keep_mask = cumsum_probs <= float(top_p)
    keep_mask[:, 0] = True

    filtered = sorted_probs * keep_mask
    filtered = filtered / filtered.sum(dim=-1, keepdim=True)
    sampled_sorted_idx = torch.multinomial(filtered, num_samples=1)
    return torch.gather(sorted_indices, dim=-1, index=sampled_sorted_idx)


def _sample_decode_no_cache(
    model,
    *,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    speech_waveforms: list[torch.Tensor],
    speech_sampling_rate: list[int],
    temperature: float,
    top_p: float,
    max_new_tokens: int,
    eos_token_id: int | None,
) -> torch.Tensor:
    """Decode autoregressively without kv-cache so speech fusion remains aligned."""
    cur_ids = input_ids
    cur_mask = attention_mask

    for _ in range(max_new_tokens):
        outputs = model(
            input_ids=cur_ids,
            attention_mask=cur_mask,
            speech_waveforms=speech_waveforms,
            speech_sampling_rate=speech_sampling_rate,
            use_cache=False,
        )
        next_logits = outputs.logits[:, -1, :]
        next_token = _sample_next_token(next_logits, temperature=temperature, top_p=top_p)

        # Append sampled token and continue decoding from the extended context.
        cur_ids = torch.cat([cur_ids, next_token], dim=1)
        cur_mask = torch.cat([cur_mask, torch.ones_like(next_token)], dim=1)

        if eos_token_id is not None and bool((next_token == int(eos_token_id)).all()):
            break

    return cur_ids


def main() -> None:
    """Load config/model/audio and prepare one ready-to-run inference input."""
    from model.pom_thinker import build_thinker

    args = _parse_args()
    _validate_generation_args(args)
    cfg = _load_config(args.config)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for SU inference")

    audio_path = Path(args.audio_path)
    if not audio_path.exists():
        raise FileNotFoundError(f"audio file not found: {audio_path}")

    checkpoint_dir = _resolve_checkpoint_dir(cfg, args.checkpoint)
    model_cfg = _get_cfg_section(cfg, "model")
    train_cfg = _get_cfg_section(cfg, "training")
    dtype = _resolve_dtype(str(train_cfg.get("precision", "bf16")))

    # Build the same model components used in SU training.
    speech_spec = _drop_none(
        {
            "encoder_id": model_cfg.get("speech_encoder_id"),
            "encoder_cache": model_cfg.get("speech_encoder_cache"),
            "frame_stack": model_cfg.get("frame_stack"),
            "projector_hidden_dim": model_cfg.get("adapter_hidden_dim"),
        }
    )
    model, tokenizer, token_ids = build_thinker(
        base_model_id=model_cfg.get("id", "Qwen/Qwen3-0.6B"),
        cache_dir=model_cfg.get("cache"),
        torch_dtype=dtype,
        speech=speech_spec,
    )
    _ensure_pad_token(tokenizer)

    # Load checkpoint weights before moving the model to GPU.
    state = _load_model_state(checkpoint_dir)
    model.load_state_dict(state, strict=True)
    model.to(torch.device("cuda"))
    _align_speech_modules(model)
    model.config.use_cache = False
    model.eval()

    # Stop on the chat-template assistant boundary token.
    im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    if im_end_id is None or not isinstance(im_end_id, int) or im_end_id < 0:
        raise ValueError("tokenizer is missing required stop token '<|im_end|>'")
    unk_id = tokenizer.unk_token_id
    if unk_id is not None and im_end_id == unk_id:
        raise ValueError("stop token '<|im_end|>' resolved to unk_token_id")

    waveform, sampling_rate = _load_audio_clip(audio_path)
    prompt_ids, speech_waveforms, speech_sampling_rate = _build_prompt(
        tokenizer,
        speech_token_id=token_ids.speech_id,
        waveform=waveform,
        sampling_rate=sampling_rate,
        disable_thinking=bool(args.disable_thinking),
    )
    input_ids = torch.tensor([prompt_ids], dtype=torch.long, device="cuda")
    attention_mask = torch.ones_like(input_ids, dtype=torch.long)
    with torch.inference_mode():
        generated = _sample_decode_no_cache(
            model,
            input_ids=input_ids,
            attention_mask=attention_mask,
            speech_waveforms=speech_waveforms,
            speech_sampling_rate=speech_sampling_rate,
            temperature=float(args.temperature),
            top_p=float(args.top_p),
            max_new_tokens=int(args.max_new_tokens),
            eos_token_id=im_end_id,
        )
    generated_ids = generated[0, input_ids.shape[1] :]
    output_text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

    print(f"Resolved checkpoint: {checkpoint_dir}")
    print(f"Input audio: {audio_path}")
    print(f"Audio sample rate: {sampling_rate}")
    print(f"Prompt tokens: {input_ids.shape[1]}")
    print(f"Speech segments: {len(speech_waveforms)}")
    print(
        f"Generation settings: temperature={args.temperature}, "
        f"top_p={args.top_p}, disable_thinking={bool(args.disable_thinking)}"
    )
    print("\nModel response:")
    print(output_text)


if __name__ == "__main__":
    main()
