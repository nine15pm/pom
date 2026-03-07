"""Run Stage-1b PomTTS autoregressive inference from assistant text."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import torch
import yaml

# Add repo root so local imports work when script is run via utils/ path.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from model.constants import SPEECH_VOCAB_SIZE
from model.pom_tts import build_pom_tts


@dataclass(frozen=True)
class DecodeConfig:
    """Holds decoding controls for autoregressive unit generation."""

    strategy: str
    temperature: float
    top_p: float
    repetition_penalty: float
    max_repeat_run: int


def _parse_args() -> argparse.Namespace:
    """Parse CLI arguments for Stage-1b text-to-unit inference."""
    parser = argparse.ArgumentParser(description="Stage-1b PomTTS inference")
    parser.add_argument("--config", type=str, required=True, help="Path to TTS YAML config")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="latest",
        help="Checkpoint directory or 'latest' (default: latest)",
    )
    parser.add_argument("--text", type=str, required=True, help="Assistant text prompt to synthesize")
    parser.add_argument(
        "--max-new-units",
        type=int,
        default=600,
        help="Max generated speech tokens before EOS (default: 600)",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default=None,
        help="Optional output JSON path for one prediction record",
    )
    parser.add_argument(
        "--decode-strategy",
        type=str,
        default="sample",
        choices=("sample", "greedy"),
        help="Unit decode strategy (default: sample)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.8,
        help="Sampling temperature for --decode-strategy=sample (default: 0.8)",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.95,
        help="Top-p nucleus sampling for --decode-strategy=sample (default: 0.95)",
    )
    parser.add_argument(
        "--repetition-penalty",
        type=float,
        default=1.1,
        help="Penalty for previously generated unit tokens (default: 1.1)",
    )
    parser.add_argument(
        "--max-repeat-run",
        type=int,
        default=30,
        help="Hard stop if one unit repeats this many times in a row (default: 30)",
    )
    parser.add_argument(
        "--decoder-model-dir",
        type=str,
        default=None,
        help="Optional CosyVoice2 decoder asset directory (enables unit->wav decode)",
    )
    parser.add_argument(
        "--output-wav",
        type=str,
        default=None,
        help="Output wav path (required when --decoder-model-dir is set)",
    )
    return parser.parse_args()


def _build_decode_config(args: argparse.Namespace) -> DecodeConfig:
    """Build and validate decoding controls from CLI args."""
    if args.temperature <= 0:
        raise ValueError("--temperature must be > 0")
    if args.top_p <= 0 or args.top_p > 1:
        raise ValueError("--top-p must be in (0, 1]")
    if args.repetition_penalty < 1.0:
        raise ValueError("--repetition-penalty must be >= 1.0")
    if args.max_repeat_run <= 0:
        raise ValueError("--max-repeat-run must be > 0")
    return DecodeConfig(
        strategy=str(args.decode_strategy),
        temperature=float(args.temperature),
        top_p=float(args.top_p),
        repetition_penalty=float(args.repetition_penalty),
        max_repeat_run=int(args.max_repeat_run),
    )


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
    """Read one config section and default to an empty mapping."""
    section = cfg.get(key, {})
    if section is None:
        return {}
    if not isinstance(section, dict):
        raise ValueError(f"config section {key!r} must be a mapping")
    return section


def _resolve_checkpoint_dir(cfg: Dict[str, Any], checkpoint: str) -> Path:
    """Resolve checkpoint input to one concrete step directory."""
    if checkpoint != "latest":
        path = Path(checkpoint)
        if not path.is_dir():
            raise FileNotFoundError(f"checkpoint not found: {path}")
        return path

    # Read training.output_dir when --checkpoint=latest.
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
    """Map training precision string to inference load dtype."""
    precision = precision.lower()
    if precision == "bf16":
        return torch.bfloat16
    if precision == "no":
        return None
    raise ValueError("unsupported precision; use bf16 or no")


def _load_model_state(checkpoint_dir: Path) -> Dict[str, torch.Tensor]:
    """Load model weights from one Accelerate checkpoint directory."""
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


def _strip_module_prefix(state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Drop a leading 'module.' prefix when checkpoints were saved from wrapped modules."""
    if not state:
        return state
    if not all(key.startswith("module.") for key in state.keys()):
        return state
    return {key[len("module.") :]: value for key, value in state.items()}


def _load_checkpoint_into_model(model: torch.nn.Module, checkpoint_dir: Path) -> None:
    """Load checkpoint weights into PomTTS with a tiny fallback for key prefixes."""
    state = _load_model_state(checkpoint_dir)
    try:
        model.load_state_dict(state, strict=True)
    except RuntimeError:
        # Some wrapped runs save with a leading "module." key prefix.
        stripped = _strip_module_prefix(state)
        if stripped is state:
            raise
        model.load_state_dict(stripped, strict=True)


def _tokenize_text(text: str, *, tokenizer: Any) -> list[int]:
    """Tokenize assistant text into plain token ids (no special tokens)."""
    text_ids = tokenizer(str(text), add_special_tokens=False)["input_ids"]
    if not isinstance(text_ids, list):
        raise ValueError("tokenizer returned non-list input_ids")
    return [int(token) for token in text_ids]


def _apply_repetition_penalty(logits: torch.Tensor, token_ids: list[int], penalty: float) -> None:
    """Apply repetition penalty in-place to already generated token ids."""
    if penalty == 1.0 or not token_ids:
        return
    for token_id in set(token_ids):
        value = logits[token_id]
        logits[token_id] = value / penalty if value > 0 else value * penalty


def _sample_top_p(*, logits: torch.Tensor, temperature: float, top_p: float) -> int:
    """Sample one token id from top-p filtered logits."""
    scaled = logits / float(temperature)
    probs = torch.softmax(scaled, dim=-1)
    if top_p < 1.0:
        sorted_probs, sorted_indices = torch.sort(probs, descending=True)
        cumulative = torch.cumsum(sorted_probs, dim=-1)
        keep = cumulative <= float(top_p)
        keep[0] = True
        filtered = torch.zeros_like(probs)
        filtered[sorted_indices[keep]] = sorted_probs[keep]
        probs = filtered / filtered.sum()
    return int(torch.multinomial(probs, num_samples=1).item())


def _run_generation(
    *,
    model: torch.nn.Module,
    tokenizer: Any,
    text_ids: list[int],
    sep_id: int,
    read_length: int,
    write_length: int,
    max_new_units: int,
    decode_cfg: DecodeConfig,
    device: torch.device,
) -> tuple[list[int], bool, int, bool]:
    """Run RW-interleaved autoregressive decode and return raw unit ids."""
    if tokenizer.eos_token_id is None:
        raise ValueError("tokenizer must define eos_token_id")
    if tokenizer.pad_token_id is None:
        raise ValueError("tokenizer must define pad_token_id")
    if model.speech_token_offset is None:
        raise ValueError("speech_token_offset is missing on PomTTS")
    if read_length <= 0 or write_length <= 0:
        raise ValueError("read_length and write_length must be > 0")

    eos_id = int(tokenizer.eos_token_id)
    speech_token_offset = int(model.speech_token_offset)
    speech_vocab_size = int(model.speech_vocab_size)
    min_speech_id = speech_token_offset
    max_speech_id = speech_token_offset + speech_vocab_size - 1

    past_key_values = None
    context_len = 0
    last_logits = None

    text_cursor = 0
    sep_added = False
    eos_seen = False
    conditioning_tokens = 0
    stopped_by_repeat_guard = False
    unit_ids: list[int] = []
    generated_token_ids: list[int] = []
    repeat_run = 0
    last_unit_id: int | None = None

    def _append_tokens(tokens: list[int]) -> None:
        nonlocal past_key_values, context_len, last_logits
        if not tokens:
            return
        input_ids = torch.tensor([tokens], dtype=torch.long, device=device)
        total_len = context_len + input_ids.shape[1]
        attention_mask = torch.ones((1, total_len), dtype=torch.long, device=device)
        kwargs: Dict[str, Any] = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "use_cache": True,
        }
        if past_key_values is not None:
            kwargs["past_key_values"] = past_key_values
        with torch.inference_mode():
            outputs = model(**kwargs)
        past_key_values = outputs.past_key_values
        context_len = total_len
        last_logits = outputs.logits[:, -1, :]

    # Match training contract: each round reads up to R text tokens, then writes up to W speech tokens.
    while len(unit_ids) < int(max_new_units) and not eos_seen:
        if text_cursor < len(text_ids):
            read_take = min(int(read_length), len(text_ids) - text_cursor)
            read_chunk = text_ids[text_cursor : text_cursor + read_take]
            _append_tokens(read_chunk)
            text_cursor += read_take
            conditioning_tokens += read_take

        # Insert one <sep> exactly when text conditioning is fully consumed.
        if text_cursor >= len(text_ids) and not sep_added:
            _append_tokens([int(sep_id)])
            sep_added = True
            conditioning_tokens += 1

        for _ in range(int(write_length)):
            if len(unit_ids) >= int(max_new_units) or eos_seen:
                break
            if last_logits is None:
                raise ValueError("decoder has no logits; empty conditioning context")

            # Keep decoding in the speech-token range plus EOS.
            step_logits = last_logits[0].float().clone()
            masked_logits = torch.full_like(step_logits, float("-inf"))
            masked_logits[min_speech_id : max_speech_id + 1] = step_logits[min_speech_id : max_speech_id + 1]
            masked_logits[eos_id] = step_logits[eos_id]
            _apply_repetition_penalty(
                masked_logits,
                generated_token_ids,
                float(decode_cfg.repetition_penalty),
            )

            if decode_cfg.strategy == "greedy":
                next_token_id = int(masked_logits.argmax().item())
            else:
                next_token_id = _sample_top_p(
                    logits=masked_logits,
                    temperature=float(decode_cfg.temperature),
                    top_p=float(decode_cfg.top_p),
                )
            if next_token_id == eos_id:
                eos_seen = True
                break
            if next_token_id < min_speech_id or next_token_id > max_speech_id:
                raise ValueError(
                    f"generated non-speech token id before EOS: {next_token_id} "
                    f"(expected [{min_speech_id}, {max_speech_id}] or EOS={eos_id})"
                )

            # Stop early if one unit repeats too many times in a row.
            unit_id = next_token_id - speech_token_offset
            if last_unit_id is not None and unit_id == last_unit_id:
                repeat_run += 1
            else:
                repeat_run = 1
                last_unit_id = unit_id
            if repeat_run > int(decode_cfg.max_repeat_run):
                stopped_by_repeat_guard = True
                break

            unit_ids.append(unit_id)
            generated_token_ids.append(next_token_id)
            _append_tokens([next_token_id])
        if stopped_by_repeat_guard:
            break

    return unit_ids, eos_seen, conditioning_tokens, stopped_by_repeat_guard


def _write_output_json(path: Path, row: Dict[str, Any]) -> None:
    """Write one inference result JSON record to disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(row, ensure_ascii=False, indent=2), encoding="utf-8")


def _summarize_wav(wav: torch.Tensor, sample_rate: int) -> Dict[str, Any]:
    """Compute tiny waveform sanity stats for quick decode validation."""
    if wav.ndim != 1:
        raise ValueError(f"expected rank-1 waveform, got shape {tuple(wav.shape)}")
    if wav.numel() == 0:
        raise ValueError("decoded waveform is empty")

    wav_f32 = wav.to(dtype=torch.float32)
    peak_abs = float(wav_f32.abs().max().item())
    rms = float(torch.sqrt(torch.mean(wav_f32.square())).item())
    clipped_samples = int((wav_f32.abs() >= 0.999).sum().item())
    return {
        "sample_rate": int(sample_rate),
        "num_samples": int(wav_f32.numel()),
        "duration_sec": float(wav_f32.numel()) / float(sample_rate),
        "rms": rms,
        "peak_abs": peak_abs,
        "clipped_samples": clipped_samples,
    }


def _decode_units_to_wav(unit_ids: list[int], *, decoder_model_dir: str) -> tuple[torch.Tensor, Dict[str, Any]]:
    """Decode predicted unit ids into one waveform using CosyVoice2 SpeechDecoder."""
    if not unit_ids:
        raise ValueError("cannot decode empty unit list; model generated zero units")

    from cosyvoice2.speech_decoder import SAMPLE_RATE, SpeechDecoder

    decoder = SpeechDecoder(model_dir=decoder_model_dir, device="cuda")
    units = torch.tensor(unit_ids, dtype=torch.long, device="cuda")
    wav = decoder.tokens_to_wav(units)
    stats = _summarize_wav(wav, SAMPLE_RATE)
    return wav, stats


def _summarize_units(unit_ids: list[int]) -> Dict[str, Any]:
    """Compute small loop-focused stats for one generated unit sequence."""
    if not unit_ids:
        return {
            "repeat_ratio": 0.0,
            "most_common_unit": None,
            "most_common_ratio": 0.0,
            "longest_repeat_run": 0,
        }
    repeat_count = sum(int(cur == prev) for prev, cur in zip(unit_ids[:-1], unit_ids[1:]))
    counts: Dict[int, int] = {}
    longest_run = 0
    run = 0
    last: int | None = None
    for unit in unit_ids:
        counts[unit] = counts.get(unit, 0) + 1
        if last is not None and unit == last:
            run += 1
        else:
            run = 1
            last = unit
        if run > longest_run:
            longest_run = run
    most_common_unit, most_common_count = max(counts.items(), key=lambda item: item[1])
    return {
        "repeat_ratio": float(repeat_count) / float(max(len(unit_ids) - 1, 1)),
        "most_common_unit": int(most_common_unit),
        "most_common_ratio": float(most_common_count) / float(len(unit_ids)),
        "longest_repeat_run": int(longest_run),
    }


def _save_wav(path: Path, wav: torch.Tensor, *, sample_rate: int) -> None:
    """Save one mono waveform tensor to disk as wav."""
    try:
        import torchaudio
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError("torchaudio is required for --output-wav") from exc

    path.parent.mkdir(parents=True, exist_ok=True)
    # torchaudio.save expects [channels, num_samples].
    torchaudio.save(path.as_posix(), wav.unsqueeze(0).to(dtype=torch.float32, device="cpu"), sample_rate)


def main() -> None:
    """Run Stage-1b checkpoint inference for one text input."""
    args = _parse_args()
    decode_cfg = _build_decode_config(args)
    if args.max_new_units <= 0:
        raise ValueError("--max-new-units must be > 0")
    if (args.decoder_model_dir is None) != (args.output_wav is None):
        raise ValueError("set both --decoder-model-dir and --output-wav together")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for TTS inference")

    cfg = _load_config(args.config)
    checkpoint_dir = _resolve_checkpoint_dir(cfg, args.checkpoint)
    model_cfg = _get_cfg_section(cfg, "model")
    train_cfg = _get_cfg_section(cfg, "training")
    read_length = int(train_cfg.get("read_length", 3))
    write_length = int(train_cfg.get("write_length", 10))

    model, tokenizer, token_ids = build_pom_tts(
        base_model_id=str(model_cfg.get("id", "Qwen/Qwen3-0.6B")),
        base_cache_dir=model_cfg.get("cache"),
        dtype=_resolve_dtype(str(train_cfg.get("precision", "bf16"))),
        speech_vocab_size=int(model_cfg.get("speech_vocab_size", SPEECH_VOCAB_SIZE)),
    )
    _load_checkpoint_into_model(model, checkpoint_dir)
    model.to(torch.device("cuda"))
    model.eval()

    text_ids = _tokenize_text(args.text, tokenizer=tokenizer)
    unit_ids, eos_seen, conditioning_tokens, stopped_by_repeat_guard = _run_generation(
        model=model,
        tokenizer=tokenizer,
        text_ids=text_ids,
        sep_id=int(token_ids.sep_id),
        read_length=read_length,
        write_length=write_length,
        max_new_units=int(args.max_new_units),
        decode_cfg=decode_cfg,
        device=torch.device("cuda"),
    )

    row = {
        "checkpoint": str(checkpoint_dir),
        "text": str(args.text),
        "decode": {
            "strategy": decode_cfg.strategy,
            "temperature": decode_cfg.temperature,
            "top_p": decode_cfg.top_p,
            "repetition_penalty": decode_cfg.repetition_penalty,
            "max_repeat_run": decode_cfg.max_repeat_run,
        },
        "num_prompt_tokens": int(conditioning_tokens),
        "num_pred_units": int(len(unit_ids)),
        "eos_seen": bool(eos_seen),
        "stopped_by_repeat_guard": bool(stopped_by_repeat_guard),
        "unit_stats": _summarize_units(unit_ids),
        "pred_unit_ids": unit_ids,
        "prediction_units": "".join(f"<{unit}>" for unit in unit_ids),
    }

    if args.decoder_model_dir is not None:
        wav, wav_stats = _decode_units_to_wav(unit_ids, decoder_model_dir=str(args.decoder_model_dir))
        _save_wav(Path(args.output_wav), wav, sample_rate=int(wav_stats["sample_rate"]))
        row["output_wav"] = str(args.output_wav)
        row["wav"] = wav_stats

    if args.output_json:
        _write_output_json(Path(args.output_json), row)

    print(f"checkpoint={row['checkpoint']}")
    print(f"num_prompt_tokens={row['num_prompt_tokens']}")
    print(f"num_pred_units={row['num_pred_units']}")
    print(f"eos_seen={row['eos_seen']}")
    print(f"stopped_by_repeat_guard={row['stopped_by_repeat_guard']}")
    print(f"unit_repeat_ratio={row['unit_stats']['repeat_ratio']:.6f}")
    print(f"unit_longest_repeat_run={row['unit_stats']['longest_repeat_run']}")
    print(f"prediction_units={row['prediction_units']}")
    if "wav" in row:
        print(f"output_wav={row['output_wav']}")
        print(f"wav_duration_sec={row['wav']['duration_sec']:.3f}")
        print(f"wav_rms={row['wav']['rms']:.6f}")
        print(f"wav_peak_abs={row['wav']['peak_abs']:.6f}")
        print(f"wav_clipped_samples={row['wav']['clipped_samples']}")
    if args.output_json:
        print(f"output_json={args.output_json}")


if __name__ == "__main__":
    main()
