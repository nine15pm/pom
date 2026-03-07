from __future__ import annotations

import json
from pathlib import Path
import shutil
import sys

import pytest
import torch
from transformers import AutoTokenizer

from inference.loader import load_runtime
from model.pom_talker import build_talker
from model.pom_thinker import build_thinker
from model.tokenizers import resolve_token_ids


def _base_cfg(
    *,
    artifact_mode: str,
    cache_dir: str,
    tokenizer_source: str | None,
    speech_vocab_size: int = 6561,
) -> dict:
    # Keep config minimal and aligned with inference/config.py contract.
    cfg: dict = {
        "models": {
            "artifact_mode": artifact_mode,
            "cache_dir": cache_dir,
            "base_cache_dir": None,
            "speech_encoder_cache_dir": None,
            "speech_vocab_size": speech_vocab_size,
            "checkpoint": None,
        },
        "tokenizer": {
            "source": tokenizer_source,
            "enable_thinking": False,
            "assistant_stop_token": None,
        },
        "generation": {
            "text": {"max_new_tokens": 8, "temperature": 0.7, "top_p": 0.9},
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
    return cfg


@pytest.fixture(scope="module")
def built_modules(base_model_id: str, whisper_tiny_id: str):
    # Build one real shared tokenizer and both runtime modules.
    thinker, tokenizer, token_ids = build_thinker(
        base_model_id=base_model_id,
        speech={
            "encoder_id": whisper_tiny_id,
            "frame_stack": 5,
            "projector_hidden_dim": 2048,
        },
    )
    talker, _, _ = build_talker(
        llm_hidden_dim=int(thinker.config.hidden_size),
        base_model_id=base_model_id,
        speech_vocab_size=6561,
        tokenizer=tokenizer,
        token_ids=token_ids,
    )
    return thinker, talker, tokenizer


@pytest.fixture(scope="module")
def hf_artifacts(tmp_path_factory, built_modules):
    # Save real HF-style artifacts for the loader's default mode.
    thinker, talker, tokenizer = built_modules
    root = tmp_path_factory.mktemp("inference_loader_hf")
    thinker_dir = root / "thinker"
    talker_dir = root / "talker"
    tokenizer_dir = root / "tokenizer"
    thinker.save_pretrained(thinker_dir, safe_serialization=False)
    talker.save_pretrained(talker_dir, safe_serialization=False)
    tokenizer.save_pretrained(tokenizer_dir)
    return root


@pytest.fixture(scope="module")
def checkpoint_artifacts(tmp_path_factory, built_modules, base_model_id: str, whisper_tiny_id: str):
    # Save strict checkpoint-style artifacts for explicit fallback mode.
    thinker, talker, tokenizer = built_modules
    _ = resolve_token_ids(tokenizer)

    root = tmp_path_factory.mktemp("inference_loader_ckpt")
    thinker_ckpt_dir = root / "thinker"
    talker_ckpt_dir = root / "talker"
    thinker_ckpt_dir.mkdir(parents=True, exist_ok=False)
    talker_ckpt_dir.mkdir(parents=True, exist_ok=False)

    torch.save(thinker.state_dict(), thinker_ckpt_dir / "pytorch_model.bin")
    torch.save(talker.state_dict(), talker_ckpt_dir / "pytorch_model.bin")

    return {
        "cache_dir": root,
        "tokenizer_source": base_model_id,
        "checkpoint": {
            "base_model_id": base_model_id,
            "speech_encoder_id": whisper_tiny_id,
            "frame_stack": 5,
            "projector_hidden_dim": 2048,
        },
    }


def test_export_hf_then_load_runtime_hf_contract(
    built_modules,
    base_model_id: str,
    whisper_tiny_id: str,
    tmp_path,
    monkeypatch,
):
    # Validate end-to-end handoff from Stage-2 checkpoint export to HF runtime loading.
    thinker, talker, _ = built_modules
    checkpoint_dir = tmp_path / "stage2_checkpoint"
    checkpoint_dir.mkdir(parents=True, exist_ok=False)
    torch.save(thinker.state_dict(), checkpoint_dir / "pytorch_model.bin")
    torch.save(talker.state_dict(), checkpoint_dir / "pytorch_model_1.bin")

    # Keep exporter build settings explicit via trainer_state compat metadata.
    trainer_state = {
        "compat": {
            "model_id": base_model_id,
            "speech_encoder_id": whisper_tiny_id,
            "frame_stack": 5,
            "adapter_hidden_dim": 2048,
            "speech_vocab_size": 6561,
        }
    }
    (checkpoint_dir / "trainer_state.json").write_text(
        json.dumps(trainer_state),
        encoding="utf-8",
    )

    export_dir = tmp_path / "exported_hf"
    from utils import export_hf

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "export_hf.py",
            "--input-checkpoint-dir",
            str(checkpoint_dir),
            "--output-dir",
            str(export_dir),
            "--speech-encoder-cache-dir",
            str(tmp_path / "speech_encoder_cache"),
        ],
    )
    export_hf.main()

    cfg = _base_cfg(
        artifact_mode="hf",
        cache_dir=str(export_dir),
        tokenizer_source=None,
    )
    runtime = load_runtime(cfg)
    assert runtime.thinker.get_speech_encoder() is not None
    assert runtime.thinker.get_speech_projector() is not None
    assert runtime.token_contract.speech_id == runtime.token_ids.speech_id
    assert runtime.token_contract.sep_id == runtime.token_ids.sep_id


def test_load_runtime_hf_mode_loads_real_artifacts(hf_artifacts):
    # Validate default HF path loads both modules and token contract.
    cfg = _base_cfg(
        artifact_mode="hf",
        cache_dir=str(hf_artifacts),
        tokenizer_source=None,
    )

    runtime = load_runtime(cfg)
    # HF runtime must load a speech-ready Thinker in one artifact path.
    assert runtime.thinker.get_speech_encoder() is not None
    assert runtime.thinker.get_speech_projector() is not None
    assert runtime.device.type == "cuda"
    assert runtime.dtype == torch.bfloat16
    assert runtime.token_contract.speech_id == runtime.token_ids.speech_id
    assert runtime.token_contract.sep_id == runtime.token_ids.sep_id
    assert runtime.token_contract.eos_id == int(runtime.tokenizer.eos_token_id)
    assert runtime.token_contract.assistant_stop_id == int(runtime.tokenizer.eos_token_id)


def test_load_runtime_hf_rejects_tokenizer_source_override(hf_artifacts, base_model_id: str):
    # Validate tokenizer.source is checkpoint-only and rejected in HF mode.
    cfg = _base_cfg(
        artifact_mode="hf",
        cache_dir=str(hf_artifacts),
        tokenizer_source=base_model_id,
    )

    with pytest.raises(ValueError, match="tokenizer.source is only valid"):
        _ = load_runtime(cfg)


def test_load_runtime_hf_rejects_thinker_without_speech_modules(tmp_path, base_model_id: str):
    # Build a real HF artifact where Thinker is intentionally missing speech modules.
    thinker, tokenizer, token_ids = build_thinker(
        base_model_id=base_model_id,
        speech=None,
    )
    talker, _, _ = build_talker(
        llm_hidden_dim=int(thinker.config.hidden_size),
        base_model_id=base_model_id,
        speech_vocab_size=6561,
        tokenizer=tokenizer,
        token_ids=token_ids,
    )
    cache_dir = tmp_path / "hf_no_speech_cache"
    thinker_dir = cache_dir / "thinker"
    talker_dir = cache_dir / "talker"
    tokenizer_dir = cache_dir / "tokenizer"
    thinker.save_pretrained(thinker_dir, safe_serialization=False)
    talker.save_pretrained(talker_dir, safe_serialization=False)
    tokenizer.save_pretrained(tokenizer_dir)

    cfg = _base_cfg(
        artifact_mode="hf",
        cache_dir=str(cache_dir),
        tokenizer_source=None,
    )

    # Runtime must fail fast because inference requires speech-ready Thinker modules.
    with pytest.raises(ValueError, match="speech encoder/projector"):
        _ = load_runtime(cfg)


def test_load_runtime_hf_rejects_thinker_artifact_metadata_mismatch(hf_artifacts, tmp_path):
    # Copy one valid Thinker artifact so we can safely corrupt only this test copy.
    bad_cache_dir = tmp_path / "cache_bad_thinker_metadata"
    shutil.copytree(hf_artifacts, bad_cache_dir)

    # Force config/weights mismatch by disabling speech modules in metadata only.
    bad_thinker_dir = bad_cache_dir / "thinker"
    config_path = bad_thinker_dir / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["speech_enabled"] = False
    config["speech_encoder_config"] = None
    config["speech_projector_config"] = None
    config_path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    cfg = _base_cfg(
        artifact_mode="hf",
        cache_dir=str(bad_cache_dir),
        tokenizer_source=None,
    )

    # Loader must reject partial HF loads when artifact keys no longer align.
    with pytest.raises(ValueError, match="thinker HF load key mismatch"):
        _ = load_runtime(cfg)


def test_load_runtime_checkpoint_mode_loads_strict_bin(checkpoint_artifacts):
    # Validate checkpoint fallback path loads strict state dicts and freezes thinker.
    cfg = _base_cfg(
        artifact_mode="checkpoint",
        cache_dir=str(checkpoint_artifacts["cache_dir"]),
        tokenizer_source=str(checkpoint_artifacts["tokenizer_source"]),
    )
    cfg["models"]["checkpoint"] = checkpoint_artifacts["checkpoint"]

    runtime = load_runtime(cfg)
    assert runtime.device.type == "cuda"
    assert runtime.dtype == torch.bfloat16
    assert not any(param.requires_grad for param in runtime.thinker.parameters())
    assert not runtime.thinker.training
    assert not runtime.talker.training


def test_load_runtime_checkpoint_rejects_extra_bin_shards(checkpoint_artifacts, tmp_path):
    # Validate strict checkpoint contract: only one pytorch_model.bin is allowed.
    test_cache_dir = tmp_path / "checkpoint_cache_copy"
    shutil.copytree(Path(checkpoint_artifacts["cache_dir"]), test_cache_dir)
    extra_bin = test_cache_dir / "thinker" / "pytorch_model-00001.bin"
    torch.save({}, extra_bin)

    cfg = _base_cfg(
        artifact_mode="checkpoint",
        cache_dir=str(test_cache_dir),
        tokenizer_source=str(checkpoint_artifacts["tokenizer_source"]),
    )
    cfg["models"]["checkpoint"] = checkpoint_artifacts["checkpoint"]

    with pytest.raises(ValueError, match="exactly one model file"):
        _ = load_runtime(cfg)


def test_load_runtime_rejects_invalid_stop_token_override(hf_artifacts):
    # Validate stop-token override fails fast when token is missing.
    cfg = _base_cfg(
        artifact_mode="hf",
        cache_dir=str(hf_artifacts),
        tokenizer_source=None,
    )
    cfg["tokenizer"]["assistant_stop_token"] = "<not_a_real_token>"

    with pytest.raises(ValueError, match="assistant_stop_token"):
        _ = load_runtime(cfg)


def test_load_runtime_hf_rejects_speech_vocab_size_mismatch(hf_artifacts):
    # Validate configured speech vocab must match the loaded Talker artifact.
    cfg = _base_cfg(
        artifact_mode="hf",
        cache_dir=str(hf_artifacts),
        tokenizer_source=None,
        speech_vocab_size=6500,
    )

    with pytest.raises(ValueError, match="speech_vocab_size"):
        _ = load_runtime(cfg)


def test_load_runtime_hf_rejects_tokenizer_contract_mismatch(
    hf_artifacts,
    base_model_id: str,
    tmp_path,
):
    # Validate loader fails when cache_dir/tokenizer lacks required Pom special tokens.
    bad_cache_dir = tmp_path / "cache_missing_pom_tokens"
    shutil.copytree(hf_artifacts, bad_cache_dir)
    bad_tokenizer = AutoTokenizer.from_pretrained(base_model_id)
    bad_tokenizer.save_pretrained(bad_cache_dir / "tokenizer")
    cfg = _base_cfg(
        artifact_mode="hf",
        cache_dir=str(bad_cache_dir),
        tokenizer_source=None,
    )

    with pytest.raises(ValueError, match="<speech>"):
        _ = load_runtime(cfg)


def test_load_runtime_hf_rejects_hidden_dim_mismatch(
    tmp_path,
    hf_artifacts,
    built_modules,
    base_model_id: str,
):
    # Validate Thinker/Talker hidden-size contract is enforced.
    thinker, _, tokenizer = built_modules
    token_ids = resolve_token_ids(tokenizer)
    bad_talker_dir = tmp_path / "bad_talker_hidden"
    bad_talker, _, _ = build_talker(
        llm_hidden_dim=int(thinker.config.hidden_size) - 1,
        base_model_id=base_model_id,
        speech_vocab_size=6561,
        tokenizer=tokenizer,
        token_ids=token_ids,
    )
    bad_talker.save_pretrained(bad_talker_dir, safe_serialization=False)
    bad_cache_dir = tmp_path / "cache_bad_hidden"
    shutil.copytree(hf_artifacts, bad_cache_dir)
    shutil.rmtree(bad_cache_dir / "talker")
    shutil.copytree(bad_talker_dir, bad_cache_dir / "talker")
    cfg = _base_cfg(
        artifact_mode="hf",
        cache_dir=str(bad_cache_dir),
        tokenizer_source=None,
    )

    with pytest.raises(ValueError, match="hidden_size"):
        _ = load_runtime(cfg)


def test_load_runtime_hf_rejects_missing_decoder_assets(hf_artifacts, tmp_path):
    """Fail fast when cache_dir/decoder exists but required decoder files are missing."""
    bad_cache_dir = tmp_path / "cache_missing_decoder_assets"
    shutil.copytree(hf_artifacts, bad_cache_dir)
    (bad_cache_dir / "decoder").mkdir(parents=True, exist_ok=False)
    cfg = _base_cfg(
        artifact_mode="hf",
        cache_dir=str(bad_cache_dir),
        tokenizer_source=None,
    )

    with pytest.raises(FileNotFoundError, match="decoder checkpoint not found"):
        _ = load_runtime(cfg)
