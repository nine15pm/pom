"""Shared decode helpers used by offline and streaming pipelines."""

from __future__ import annotations

from dataclasses import dataclass
import io
from pathlib import Path
from typing import Any

import torch
import torchaudio

TEXT_OVERRIDE_KEYS = {"max_new_tokens", "temperature", "top_p"}
SPEECH_OVERRIDE_KEYS = {
    "max_new_tokens",
    "temperature",
    "top_p",
    "repetition_penalty",
    "max_repeat_run",
    "read_length",
    "write_length",
}


@dataclass(frozen=True)
class TextDecodeConfig:
    """Hold resolved text decode settings for one Thinker generation pass."""

    max_new_tokens: int
    temperature: float
    top_p: float


@dataclass(frozen=True)
class ThinkerDecodeResult:
    """Hold Thinker generated ids and one aligned hidden row per id."""

    generated_ids: torch.Tensor
    hidden_rows: torch.Tensor
    stop_seen: bool


@dataclass(frozen=True)
class SpeechDecodeConfig:
    """Hold resolved speech decode settings for one Talker generation pass."""

    max_new_tokens: int
    temperature: float
    top_p: float
    repetition_penalty: float
    max_repeat_run: int
    read_length: int
    write_length: int


@dataclass(frozen=True)
class TalkerDecodeResult:
    """Hold generated speech units plus stop and conditioning metadata."""

    unit_ids: list[int]
    eos_seen: bool
    conditioning_consumed: bool


def load_audio(path: str | Path) -> tuple[torch.Tensor, int]:
    """Load one audio clip as mono float32 waveform with explicit sample rate."""
    audio_path = Path(path)
    if not audio_path.is_file():
        raise FileNotFoundError(f"audio file not found: {audio_path}")

    waveform, sampling_rate = torchaudio.load(audio_path.as_posix())
    mono = _normalize_waveform(waveform, empty_error_prefix=f"audio clip is empty: {audio_path}")
    return mono, int(sampling_rate)


def load_audio_bytes(audio_bytes: bytes) -> tuple[torch.Tensor, int]:
    """Load one audio clip from bytes as mono float32 waveform with explicit sample rate."""
    if not isinstance(audio_bytes, (bytes, bytearray)) or len(audio_bytes) <= 0:
        raise ValueError("audio bytes must be non-empty")
    with io.BytesIO(bytes(audio_bytes)) as handle:
        waveform, sampling_rate = torchaudio.load(handle)
    mono = _normalize_waveform(waveform, empty_error_prefix="audio clip is empty")
    return mono, int(sampling_rate)


def _normalize_waveform(waveform: torch.Tensor, *, empty_error_prefix: str) -> torch.Tensor:
    """Normalize loaded waveform to mono float32 and validate non-empty content."""
    if waveform.ndim != 2:
        raise ValueError(f"expected audio shape [channels, samples], got {tuple(waveform.shape)}")
    if waveform.size(1) <= 0:
        raise ValueError(empty_error_prefix)
    # Mix channels to one mono track for the speech encoder path.
    if waveform.size(0) > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    mono = waveform.squeeze(0).to(dtype=torch.float32)
    if mono.numel() <= 0:
        raise ValueError(f"{empty_error_prefix} after mono conversion")
    return mono


def find_unique_span(*, sequence: torch.Tensor, target: torch.Tensor) -> int:
    """Return the unique contiguous target span start, or fail on none or ambiguity."""
    if sequence.ndim != 1 or target.ndim != 1:
        raise ValueError("sequence and target must be rank-1 tensors")
    target_len = int(target.numel())
    if target_len <= 0:
        raise ValueError("target must be non-empty")
    if int(sequence.numel()) < target_len:
        raise ValueError("no match")

    # Compare all contiguous windows so matching stays deterministic.
    windows = sequence.unfold(dimension=0, size=target_len, step=1)
    match_mask = windows.eq(target).all(dim=1)
    match_indices = match_mask.nonzero(as_tuple=False).flatten()
    match_count = int(match_indices.numel())
    if match_count == 0:
        raise ValueError("no match")
    if match_count > 1:
        raise ValueError(f"ambiguous ({match_count} matches)")
    return int(match_indices[0].item())


def validate_override_keys(
    *,
    overrides: dict[str, object] | None,
    allowed: set[str],
    section_name: str,
) -> None:
    """Reject unknown generation override keys so request behavior stays explicit."""
    if overrides is None:
        return
    unknown = sorted(key for key in overrides.keys() if key not in allowed)
    if unknown:
        raise ValueError(
            f"unknown {section_name} override keys: {unknown}; allowed keys: {sorted(allowed)}"
        )


def resolve_text_decode_config(
    cfg: dict[str, Any],
    overrides: dict[str, object] | None,
) -> TextDecodeConfig:
    """Resolve text generation settings from config plus optional request overrides."""
    text_cfg = dict(cfg["generation"]["text"])
    validate_override_keys(
        overrides=overrides,
        allowed=TEXT_OVERRIDE_KEYS,
        section_name="generation.text",
    )
    if overrides is not None:
        text_cfg.update(dict(overrides))

    max_new_tokens = int(text_cfg["max_new_tokens"])
    temperature = float(text_cfg["temperature"])
    top_p = float(text_cfg["top_p"])
    if max_new_tokens <= 0:
        raise ValueError("generation.text.max_new_tokens must be > 0")
    if temperature <= 0:
        raise ValueError("generation.text.temperature must be > 0")
    if top_p <= 0 or top_p > 1:
        raise ValueError("generation.text.top_p must be in (0, 1]")

    return TextDecodeConfig(
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
    )


def build_one_turn_prompt(tokenizer: Any, *, speech_id: int) -> list[int]:
    """Build one user-audio turn prompt with Qwen chat template thinking disabled."""
    messages = [
        {"role": "system", "content": ""},
        {"role": "user", "content": "<speech>"},
    ]
    encoded = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        return_dict=True,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    prompt_ids = [int(token_id) for token_id in encoded["input_ids"]]
    speech_count = sum(1 for token_id in prompt_ids if int(token_id) == int(speech_id))
    if speech_count != 1:
        raise ValueError(f"expected exactly one <speech> token in prompt, got {speech_count}")
    return prompt_ids


def sample_next_token(
    logits: torch.Tensor,
    *,
    temperature: float,
    top_p: float,
    rng: torch.Generator | None = None,
) -> int:
    """Sample one token id from logits using temperature and nucleus filtering."""
    scaled = logits / float(temperature)
    probs = torch.softmax(scaled, dim=-1)
    if top_p >= 1.0:
        # Allow stage-local RNG so interleaved streaming can keep Thinker/Talker sampling independent.
        return int(torch.multinomial(probs, num_samples=1, generator=rng).item())

    sorted_probs, sorted_indices = torch.sort(probs, descending=True)
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
    keep = cumulative_probs <= float(top_p)
    keep[0] = True
    filtered = torch.zeros_like(probs)
    filtered[sorted_indices[keep]] = sorted_probs[keep]
    filtered = filtered / filtered.sum()
    return int(torch.multinomial(filtered, num_samples=1, generator=rng).item())


def decode_thinker_one_pass(
    *,
    thinker: torch.nn.Module,
    token_contract: Any,
    prompt_ids: list[int],
    waveform: torch.Tensor,
    sampling_rate: int,
    decode_cfg: TextDecodeConfig,
    device: torch.device,
) -> ThinkerDecodeResult:
    """Decode Thinker token-by-token and capture one hidden row per generated id."""
    input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    attention_mask = torch.ones_like(input_ids, dtype=torch.long)
    outputs = thinker(
        input_ids=input_ids,
        attention_mask=attention_mask,
        speech_waveforms=[waveform],
        speech_sampling_rate=[int(sampling_rate)],
        use_cache=True,
        output_hidden_states=True,
        return_dict=True,
    )
    if outputs.past_key_values is None:
        raise RuntimeError("thinker forward did not return past_key_values")

    past_key_values = outputs.past_key_values
    last_logits = outputs.logits[0, -1, :].float()
    # Cache length follows the post-fusion sequence, not raw prompt token count.
    context_len = int(outputs.logits.shape[1])
    generated_ids: list[int] = []
    hidden_rows: list[torch.Tensor] = []
    stop_seen = False

    for _ in range(int(decode_cfg.max_new_tokens)):
        next_token_id = sample_next_token(
            logits=last_logits,
            temperature=float(decode_cfg.temperature),
            top_p=float(decode_cfg.top_p),
        )
        generated_ids.append(int(next_token_id))

        # Feed sampled token back to capture aligned hidden row and next logits.
        step_ids = torch.tensor([[int(next_token_id)]], dtype=torch.long, device=device)
        context_len += 1
        step_attention = torch.ones((1, context_len), dtype=torch.long, device=device)
        outputs = thinker(
            input_ids=step_ids,
            attention_mask=step_attention,
            past_key_values=past_key_values,
            use_cache=True,
            output_hidden_states=True,
            return_dict=True,
        )
        if outputs.past_key_values is None:
            raise RuntimeError("thinker decode step did not return past_key_values")
        final_hidden = outputs.hidden_states[-1]
        hidden_rows.append(final_hidden[0, -1, :].detach())

        past_key_values = outputs.past_key_values
        last_logits = outputs.logits[0, -1, :].float()
        if int(next_token_id) == int(token_contract.assistant_stop_id):
            stop_seen = True
            break

    if not generated_ids:
        return ThinkerDecodeResult(
            generated_ids=torch.empty(0, dtype=torch.long, device=device),
            hidden_rows=torch.empty((0, int(thinker.config.hidden_size)), device=device),
            stop_seen=False,
        )

    return ThinkerDecodeResult(
        generated_ids=torch.tensor(generated_ids, dtype=torch.long, device=device),
        hidden_rows=torch.stack(hidden_rows, dim=0),
        stop_seen=bool(stop_seen),
    )


def extract_aligned_content(
    *,
    generated_ids: torch.Tensor,
    hidden_rows: torch.Tensor,
    tokenizer: Any,
    assistant_stop_id: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Select content token ids and hidden rows with strict unique alignment."""
    if generated_ids.ndim != 1 or hidden_rows.ndim != 2:
        raise ValueError("generated_ids must be rank-1 and hidden_rows must be rank-2")
    if int(generated_ids.numel()) != int(hidden_rows.size(0)):
        raise ValueError("generated_ids and hidden_rows length mismatch")
    if int(generated_ids.numel()) <= 0:
        raise ValueError("thinker generated no tokens")

    # Build plain content ids as all non-control generated tokens.
    special_ids = {int(token_id) for token_id in getattr(tokenizer, "all_special_ids", [])}
    control_ids = set(special_ids)
    control_ids.add(int(assistant_stop_id))

    generated_list = [int(token_id) for token_id in generated_ids.detach().cpu().tolist()]
    plain_content_ids = [token_id for token_id in generated_list if token_id not in control_ids]
    if not plain_content_ids:
        raise ValueError("thinker generated no plain content tokens")
    content_ids = torch.tensor(plain_content_ids, dtype=torch.long, device=generated_ids.device)

    # Enforce a strict unique span before passing conditioning into Talker.
    aligned_start = find_unique_span(sequence=generated_ids, target=content_ids)
    aligned_end = aligned_start + int(content_ids.numel())
    return content_ids, hidden_rows[aligned_start:aligned_end]


def resolve_speech_decode_config(
    cfg: dict[str, Any],
    overrides: dict[str, object] | None,
) -> SpeechDecodeConfig:
    """Resolve speech generation settings from config plus optional request overrides."""
    speech_cfg = dict(cfg["generation"]["speech"])
    validate_override_keys(
        overrides=overrides,
        allowed=SPEECH_OVERRIDE_KEYS,
        section_name="generation.speech",
    )
    if overrides is not None:
        speech_cfg.update(dict(overrides))

    max_new_tokens = int(speech_cfg["max_new_tokens"])
    temperature = float(speech_cfg["temperature"])
    top_p = float(speech_cfg["top_p"])
    repetition_penalty = float(speech_cfg["repetition_penalty"])
    max_repeat_run = int(speech_cfg["max_repeat_run"])
    read_length = int(speech_cfg["read_length"])
    write_length = int(speech_cfg["write_length"])

    if max_new_tokens <= 0:
        raise ValueError("generation.speech.max_new_tokens must be > 0")
    if temperature <= 0:
        raise ValueError("generation.speech.temperature must be > 0")
    if top_p <= 0 or top_p > 1:
        raise ValueError("generation.speech.top_p must be in (0, 1]")
    if repetition_penalty < 1.0:
        raise ValueError("generation.speech.repetition_penalty must be >= 1.0")
    if max_repeat_run <= 0:
        raise ValueError("generation.speech.max_repeat_run must be > 0")
    if read_length <= 0 or write_length <= 0:
        raise ValueError("generation.speech.read_length and write_length must be > 0")

    return SpeechDecodeConfig(
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        max_repeat_run=max_repeat_run,
        read_length=read_length,
        write_length=write_length,
    )


def apply_repetition_penalty(logits: torch.Tensor, token_ids: list[int], penalty: float) -> None:
    """Apply repetition penalty in-place to previously generated token ids."""
    if penalty == 1.0 or not token_ids:
        return
    for token_id in set(token_ids):
        score = logits[int(token_id)]
        logits[int(token_id)] = score / penalty if score > 0 else score * penalty


def talker_step_with_embeds(
    *,
    talker: torch.nn.Module,
    chunk_embeds: torch.Tensor,
    past_key_values: Any,
    context_len: int,
    device: torch.device,
) -> tuple[Any, int, torch.Tensor]:
    """Append one fused embedding chunk and return updated cache and next logits."""
    step_len = int(chunk_embeds.size(0))
    if step_len <= 0:
        raise ValueError("chunk_embeds must be non-empty")

    attention_mask = torch.ones((1, context_len + step_len), dtype=torch.long, device=device)
    kwargs: dict[str, Any] = {
        "inputs_embeds": chunk_embeds.unsqueeze(0),
        "attention_mask": attention_mask,
        "use_cache": True,
        "return_dict": True,
    }
    if past_key_values is not None:
        kwargs["past_key_values"] = past_key_values

    outputs = talker(**kwargs)
    if outputs.past_key_values is None:
        raise RuntimeError("talker forward did not return past_key_values")
    return outputs.past_key_values, context_len + step_len, outputs.logits[0, -1, :].float()


def talker_step_with_token(
    *,
    talker: torch.nn.Module,
    token_id: int,
    past_key_values: Any,
    context_len: int,
    device: torch.device,
) -> tuple[Any, int, torch.Tensor]:
    """Append one token id and return updated cache and next logits."""
    input_ids = torch.tensor([[int(token_id)]], dtype=torch.long, device=device)
    attention_mask = torch.ones((1, context_len + 1), dtype=torch.long, device=device)
    kwargs: dict[str, Any] = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "use_cache": True,
        "return_dict": True,
    }
    if past_key_values is not None:
        kwargs["past_key_values"] = past_key_values

    outputs = talker(**kwargs)
    if outputs.past_key_values is None:
        raise RuntimeError("talker forward did not return past_key_values")
    return outputs.past_key_values, context_len + 1, outputs.logits[0, -1, :].float()


def decode_talker_speech(
    *,
    talker: torch.nn.Module,
    token_contract: Any,
    fused_row: torch.Tensor,
    decode_cfg: SpeechDecodeConfig,
    device: torch.device,
) -> TalkerDecodeResult:
    """Decode speech units with strict read-write scheduling and vocab constraints."""
    if fused_row.ndim != 2:
        raise ValueError("fused_row must be rank-2 [tokens, hidden_dim]")
    if int(fused_row.size(0)) <= 0:
        raise ValueError("fused_row must be non-empty")

    speech_offset = talker.speech_token_offset
    speech_vocab_size = talker.speech_vocab_size
    if speech_offset is None or int(speech_vocab_size) <= 0:
        raise ValueError("talker speech token mapping is not configured")

    min_speech_id = int(speech_offset)
    max_speech_id = int(speech_offset) + int(speech_vocab_size) - 1
    eos_id = int(token_contract.eos_id)
    sep_id = int(token_contract.sep_id)

    past_key_values = None
    context_len = 0
    last_logits: torch.Tensor | None = None

    read_cursor = 0
    sep_added = False
    eos_seen = False
    repeat_run = 0
    last_unit_id: int | None = None
    generated_lm_ids: list[int] = []
    unit_ids: list[int] = []

    # Build one reusable <sep> embedding row for text/speech boundary marker.
    sep_embed = talker.get_input_embeddings()(
        torch.tensor([sep_id], dtype=torch.long, device=device)
    ).squeeze(0)

    while len(unit_ids) < int(decode_cfg.max_new_tokens) and not eos_seen:
        # Consume read chunk first so Talker conditions on fused text-hidden context.
        if read_cursor < int(fused_row.size(0)):
            read_take = min(int(decode_cfg.read_length), int(fused_row.size(0)) - read_cursor)
            read_chunk = fused_row[read_cursor : read_cursor + read_take]
            past_key_values, context_len, last_logits = talker_step_with_embeds(
                talker=talker,
                chunk_embeds=read_chunk,
                past_key_values=past_key_values,
                context_len=context_len,
                device=device,
            )
            read_cursor += read_take

        # Insert <sep> once after all conditioning vectors are consumed.
        if read_cursor >= int(fused_row.size(0)) and not sep_added:
            past_key_values, context_len, last_logits = talker_step_with_embeds(
                talker=talker,
                chunk_embeds=sep_embed.unsqueeze(0),
                past_key_values=past_key_values,
                context_len=context_len,
                device=device,
            )
            sep_added = True

        for _ in range(int(decode_cfg.write_length)):
            if len(unit_ids) >= int(decode_cfg.max_new_tokens) or eos_seen:
                break
            if last_logits is None:
                raise RuntimeError("talker has no logits available for speech decode")

            # Restrict Talker decode to speech range plus EOS.
            masked_logits = torch.full_like(last_logits, float("-inf"))
            masked_logits[min_speech_id : max_speech_id + 1] = last_logits[min_speech_id : max_speech_id + 1]
            masked_logits[eos_id] = last_logits[eos_id]
            apply_repetition_penalty(
                masked_logits,
                generated_lm_ids,
                float(decode_cfg.repetition_penalty),
            )

            next_token_id = sample_next_token(
                masked_logits,
                temperature=float(decode_cfg.temperature),
                top_p=float(decode_cfg.top_p),
            )
            if int(next_token_id) == eos_id:
                eos_seen = True
                break
            if int(next_token_id) < min_speech_id or int(next_token_id) > max_speech_id:
                raise ValueError(
                    f"talker generated invalid token id {next_token_id}; expected speech range or EOS"
                )

            unit_id = int(next_token_id) - int(speech_offset)
            if last_unit_id is not None and unit_id == last_unit_id:
                repeat_run += 1
            else:
                repeat_run = 1
                last_unit_id = unit_id
            if repeat_run > int(decode_cfg.max_repeat_run):
                # Treat repeat guard as a normal stop condition, not an inference error.
                return TalkerDecodeResult(
                    unit_ids=unit_ids,
                    eos_seen=False,
                    conditioning_consumed=bool(read_cursor >= int(fused_row.size(0))),
                )

            unit_ids.append(unit_id)
            generated_lm_ids.append(int(next_token_id))
            past_key_values, context_len, last_logits = talker_step_with_token(
                talker=talker,
                token_id=int(next_token_id),
                past_key_values=past_key_values,
                context_len=context_len,
                device=device,
            )

    # Keep empty-unit generations explicit so callers do not fail later in decode.
    if not unit_ids:
        raise ValueError("talker generated no speech units")

    return TalkerDecodeResult(
        unit_ids=unit_ids,
        eos_seen=bool(eos_seen),
        conditioning_consumed=bool(read_cursor >= int(fused_row.size(0))),
    )


__all__ = [
    "SpeechDecodeConfig",
    "TalkerDecodeResult",
    "TextDecodeConfig",
    "ThinkerDecodeResult",
    "apply_repetition_penalty",
    "build_one_turn_prompt",
    "decode_thinker_one_pass",
    "decode_talker_speech",
    "extract_aligned_content",
    "find_unique_span",
    "load_audio",
    "load_audio_bytes",
    "resolve_speech_decode_config",
    "resolve_text_decode_config",
    "sample_next_token",
    "talker_step_with_embeds",
    "talker_step_with_token",
    "validate_override_keys",
]
