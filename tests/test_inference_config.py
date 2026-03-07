from __future__ import annotations

import os
from pathlib import Path

import pytest
import yaml

from inference.config import load_inference_config


def _base_cfg() -> dict:
    """Return one minimal valid Phase A inference config."""
    return {
        "models": {
            "artifact_mode": "hf",
            "cache_dir": "/tmp/pom-cache",
            "base_cache_dir": None,
            "speech_encoder_cache_dir": None,
            "speech_vocab_size": 6561,
            "checkpoint": None,
        },
        "tokenizer": {
            "source": None,
            "enable_thinking": False,
            "assistant_stop_token": None,
        },
        "generation": {
            "text": {
                "max_new_tokens": 8,
                "temperature": 0.7,
                "top_p": 0.9,
            },
            "speech": {
                "max_new_tokens": 16,
                "temperature": 0.8,
                "top_p": 0.95,
                "repetition_penalty": 1.1,
                "max_repeat_run": 10,
                "read_length": 3,
                "write_length": 10,
            },
        },
        "runtime": {"device": "cuda", "dtype": "bf16", "seed": 42},
        "output": {"output_wav_path": None},
    }


def _write_cfg(path: Path, cfg: dict) -> None:
    """Write one YAML config file used by config-loader tests."""
    path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")


def test_load_inference_config_hf_requires_models_cache_dir(tmp_path: Path) -> None:
    """HF mode must require explicit models.cache_dir."""
    cfg = _base_cfg()
    cfg["models"]["cache_dir"] = None
    cfg_path = tmp_path / "hf_missing_cache_dir.yaml"
    _write_cfg(cfg_path, cfg)

    with pytest.raises(ValueError, match="models.cache_dir must be a non-empty string"):
        _ = load_inference_config(cfg_path)


def test_load_inference_config_hf_rejects_tokenizer_source_override(tmp_path: Path) -> None:
    """HF mode must reject tokenizer.source overrides."""
    cfg = _base_cfg()
    cfg["tokenizer"]["source"] = "Qwen/Qwen3-0.6B"
    cfg_path = tmp_path / "hf_rejects_tokenizer_source.yaml"
    _write_cfg(cfg_path, cfg)

    with pytest.raises(ValueError, match="tokenizer.source is only valid"):
        _ = load_inference_config(cfg_path)


def test_load_inference_config_expands_models_cache_dir(tmp_path: Path, monkeypatch) -> None:
    """models.cache_dir should expand env-vars and '~' before runtime load."""
    cfg = _base_cfg()
    monkeypatch.setenv("POM_TEST_CACHE_ROOT", str(tmp_path))
    cfg["models"]["cache_dir"] = "$POM_TEST_CACHE_ROOT"
    cfg_path = tmp_path / "expands_cache_dir.yaml"
    _write_cfg(cfg_path, cfg)

    loaded = load_inference_config(cfg_path)
    expected = str(Path(os.environ["POM_TEST_CACHE_ROOT"]).expanduser())
    assert loaded["models"]["cache_dir"] == expected
