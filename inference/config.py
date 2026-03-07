"""Tiny config loading and validation for Phase A inference."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Mapping

import yaml

# Keep the runtime config shape simple and close to YAML.
InferenceConfig = dict[str, Any]


_TOP_LEVEL_KEYS = {"models", "tokenizer", "generation", "runtime", "output"}
_MODELS_KEYS = {
    "artifact_mode",
    "cache_dir",
    "base_cache_dir",
    "speech_encoder_cache_dir",
    "speech_vocab_size",
    "checkpoint",
}
_CHECKPOINT_KEYS = {"base_model_id", "speech_encoder_id", "frame_stack", "projector_hidden_dim"}
_TOKENIZER_KEYS = {"source", "enable_thinking", "assistant_stop_token"}
_GENERATION_KEYS = {"text", "speech"}
_GENERATION_TEXT_KEYS = {"max_new_tokens", "temperature", "top_p"}
_GENERATION_SPEECH_KEYS = {
    "max_new_tokens",
    "temperature",
    "top_p",
    "repetition_penalty",
    "max_repeat_run",
    "read_length",
    "write_length",
}
_RUNTIME_KEYS = {"device", "dtype", "seed"}
_OUTPUT_KEYS = {"output_wav_path"}


def _as_section(value: Any, *, name: str) -> dict[str, Any]:
    """Read one required section as a dictionary."""
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a mapping")
    return dict(value)


def _merge_dicts(base: dict[str, Any], overrides: Mapping[str, Any]) -> dict[str, Any]:
    """Merge nested override keys into loaded YAML config."""
    merged = dict(base)
    for key, value in overrides.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, Mapping):
            merged[key] = _merge_dicts(dict(merged[key]), value)
        else:
            merged[key] = value
    return merged


def _reject_unknown_keys(section: Mapping[str, Any], *, allowed: set[str], path: str) -> None:
    """Fail fast on unknown keys so config typos are not silently ignored."""
    for key in section:
        if key not in allowed:
            raise ValueError(f"unknown config key: {path}.{key}")


def _require_non_empty_str(section: dict[str, Any], key: str, *, path: str) -> str:
    """Validate one required non-empty string field."""
    value = section.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{path}.{key} must be a non-empty string")
    return value.strip()


def _normalize_optional_str(section: dict[str, Any], key: str, *, path: str) -> str | None:
    """Normalize one optional string field when it is provided."""
    value = section.get(key)
    if value is not None and (not isinstance(value, str) or not value.strip()):
        raise ValueError(f"{path}.{key} must be null or a non-empty string")
    if value is None:
        return None
    return value.strip()


def _expand_path(raw_path: str) -> str:
    """Expand one filesystem path string via env vars and '~'."""
    return str(Path(os.path.expandvars(raw_path)).expanduser())


def _normalize_required_path(section: dict[str, Any], key: str, *, path: str) -> str:
    """Normalize one required path-like string field."""
    return _expand_path(_require_non_empty_str(section, key, path=path))


def _normalize_optional_path(section: dict[str, Any], key: str, *, path: str) -> str | None:
    """Normalize one optional path-like string field."""
    value = _normalize_optional_str(section, key, path=path)
    if value is None:
        return None
    return _expand_path(value)


def _validate_positive_int(section: dict[str, Any], key: str, *, path: str) -> None:
    """Validate one integer field that must be > 0."""
    value = section.get(key)
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{path}.{key} must be an integer > 0")


def _validate_positive_float(section: dict[str, Any], key: str, *, path: str) -> None:
    """Validate one numeric field that must be > 0."""
    value = section.get(key)
    if not isinstance(value, (int, float)) or isinstance(value, bool) or float(value) <= 0:
        raise ValueError(f"{path}.{key} must be > 0")


def _validate_probability(section: dict[str, Any], key: str, *, path: str) -> None:
    """Validate one probability field in the range (0, 1]."""
    _validate_positive_float(section, key, path=path)
    value = section.get(key)
    if float(value) > 1.0:
        raise ValueError(f"{path}.{key} must be in (0, 1]")


def load_inference_config(
    path: str | Path = "configs/inference.yaml",
    *,
    overrides: Mapping[str, Any] | None = None,
) -> InferenceConfig:
    """Load YAML config, apply overrides, and enforce Phase A contract checks."""
    cfg_path = Path(path)
    if not cfg_path.exists():
        raise FileNotFoundError(f"config not found: {cfg_path}")

    with cfg_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    if not isinstance(raw, dict):
        raise ValueError("config must be a YAML mapping at the top level")

    cfg: InferenceConfig = dict(raw)
    if overrides is not None:
        cfg = _merge_dicts(cfg, overrides)

    _reject_unknown_keys(cfg, allowed=_TOP_LEVEL_KEYS, path="config")

    models = _as_section(cfg.get("models"), name="models")
    _reject_unknown_keys(models, allowed=_MODELS_KEYS, path="models")
    artifact_mode = _require_non_empty_str(models, "artifact_mode", path="models").lower()
    if artifact_mode not in {"hf", "checkpoint"}:
        raise ValueError("models.artifact_mode must be one of: hf, checkpoint")
    models["cache_dir"] = _normalize_required_path(models, "cache_dir", path="models")
    _validate_positive_int(models, "speech_vocab_size", path="models")
    for key in ("base_cache_dir", "speech_encoder_cache_dir"):
        models[key] = _normalize_optional_path(models, key, path="models")

    checkpoint_raw = models.get("checkpoint")
    if artifact_mode == "checkpoint":
        checkpoint = _as_section(checkpoint_raw, name="models.checkpoint")
        _reject_unknown_keys(checkpoint, allowed=_CHECKPOINT_KEYS, path="models.checkpoint")
        checkpoint["base_model_id"] = _require_non_empty_str(
            checkpoint,
            "base_model_id",
            path="models.checkpoint",
        )
        checkpoint["speech_encoder_id"] = _require_non_empty_str(
            checkpoint,
            "speech_encoder_id",
            path="models.checkpoint",
        )
        _validate_positive_int(checkpoint, "frame_stack", path="models.checkpoint")
        _validate_positive_int(checkpoint, "projector_hidden_dim", path="models.checkpoint")
        models["checkpoint"] = checkpoint
    elif checkpoint_raw is not None:
        raise ValueError("models.checkpoint is only valid when models.artifact_mode='checkpoint'")

    tokenizer = _as_section(cfg.get("tokenizer"), name="tokenizer")
    _reject_unknown_keys(tokenizer, allowed=_TOKENIZER_KEYS, path="tokenizer")
    tokenizer["source"] = _normalize_optional_str(tokenizer, "source", path="tokenizer")
    tokenizer["assistant_stop_token"] = _normalize_optional_str(
        tokenizer,
        "assistant_stop_token",
        path="tokenizer",
    )
    enable_thinking = tokenizer.get("enable_thinking")
    if not isinstance(enable_thinking, bool):
        raise ValueError("tokenizer.enable_thinking must be a boolean")
    if artifact_mode == "hf" and tokenizer["source"] is not None:
        raise ValueError("tokenizer.source is only valid when models.artifact_mode='checkpoint'")
    # Keep Phase A deterministic by disabling Qwen thinking tags.
    if enable_thinking:
        raise ValueError("tokenizer.enable_thinking must be false for Phase A")

    generation = _as_section(cfg.get("generation"), name="generation")
    _reject_unknown_keys(generation, allowed=_GENERATION_KEYS, path="generation")
    text = _as_section(generation.get("text"), name="generation.text")
    _reject_unknown_keys(text, allowed=_GENERATION_TEXT_KEYS, path="generation.text")
    speech = _as_section(generation.get("speech"), name="generation.speech")
    _reject_unknown_keys(speech, allowed=_GENERATION_SPEECH_KEYS, path="generation.speech")

    _validate_positive_int(text, "max_new_tokens", path="generation.text")
    _validate_positive_float(text, "temperature", path="generation.text")
    _validate_probability(text, "top_p", path="generation.text")

    _validate_positive_int(speech, "max_new_tokens", path="generation.speech")
    _validate_positive_float(speech, "temperature", path="generation.speech")
    _validate_probability(speech, "top_p", path="generation.speech")
    _validate_positive_int(speech, "max_repeat_run", path="generation.speech")
    _validate_positive_int(speech, "read_length", path="generation.speech")
    _validate_positive_int(speech, "write_length", path="generation.speech")
    repetition_penalty = speech.get("repetition_penalty")
    if (
        not isinstance(repetition_penalty, (int, float))
        or isinstance(repetition_penalty, bool)
        or float(repetition_penalty) < 1.0
    ):
        raise ValueError("generation.speech.repetition_penalty must be >= 1.0")

    runtime = _as_section(cfg.get("runtime"), name="runtime")
    _reject_unknown_keys(runtime, allowed=_RUNTIME_KEYS, path="runtime")
    device = _require_non_empty_str(runtime, "device", path="runtime").lower()
    # Keep runtime aligned with cloud GPU deployment expectations.
    if device != "cuda":
        raise ValueError("runtime.device must be 'cuda' for Phase A")
    dtype = _require_non_empty_str(runtime, "dtype", path="runtime").lower()
    if dtype != "bf16":
        raise ValueError("runtime.dtype must be 'bf16' for Phase A")
    seed = runtime.get("seed")
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise ValueError("runtime.seed must be an integer")

    output = _as_section(cfg.get("output"), name="output")
    _reject_unknown_keys(output, allowed=_OUTPUT_KEYS, path="output")
    output["output_wav_path"] = _normalize_optional_str(output, "output_wav_path", path="output")

    # Store normalized values back so downstream code gets consistent casing.
    models["artifact_mode"] = artifact_mode
    cfg["tokenizer"] = tokenizer
    cfg["generation"] = generation
    runtime["device"] = device
    runtime["dtype"] = dtype
    cfg["models"] = models
    cfg["runtime"] = runtime
    cfg["output"] = output

    return cfg


__all__ = ["InferenceConfig", "load_inference_config"]
