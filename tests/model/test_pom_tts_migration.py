"""Focused tests for Stage-1b -> PomTTS migration utility behavior."""

from __future__ import annotations

import argparse
import json

import pytest
import torch

from model.constants import SPEECH_VOCAB_SIZE
from model.pom_tts import PomTTS
from model.pom_tts import build_pom_tts
from utils.migrate_stage1b_tts_checkpoint import _ensure_safe_output_path
from utils.migrate_stage1b_tts_checkpoint import _iter_state_candidates
from utils.migrate_stage1b_tts_checkpoint import _resolve_build_settings
from utils.migrate_stage1b_tts_checkpoint import _select_strict_candidate
from utils import migrate_stage1b_tts_checkpoint


def test_iter_state_candidates_rewrites_module_language_model_prefix() -> None:
    """Normalize legacy wrapped keys into canonical PomTTS key space."""
    # Build a tiny real tensor state dict that matches one known legacy wrapper form.
    legacy_state = {
        "module.language_model.lm_head.weight": torch.randn(3, 4),
        "module.language_model.model.embed_tokens.weight": torch.randn(5, 4),
    }

    # Collect all normalization candidates and index by candidate strategy name.
    candidates = {name: state for name, state in _iter_state_candidates(legacy_state)}
    assert "strip_module_language_model" in candidates

    # Verify the final canonical mapping drops both wrapper prefixes.
    canonical = candidates["strip_module_language_model"]
    assert "lm_head.weight" in canonical
    assert "model.embed_tokens.weight" in canonical
    assert not any(key.startswith("module.language_model.") for key in canonical)


def test_ensure_safe_output_path_rejects_in_place_and_nested_output(tmp_path) -> None:
    """Protect source checkpoints by rejecting unsafe migration output targets."""
    # Create one input checkpoint directory to mimic a real migration source.
    input_dir = tmp_path / "legacy_ckpt"
    input_dir.mkdir(parents=True, exist_ok=False)

    # In-place output would overwrite source files and must be rejected.
    with pytest.raises(ValueError, match="different from input-checkpoint-dir"):
        _ensure_safe_output_path(input_dir, input_dir)

    # Nested output can still corrupt source tree and must also be rejected.
    nested_output = input_dir / "converted"
    with pytest.raises(ValueError, match="cannot be inside input-checkpoint-dir"):
        _ensure_safe_output_path(input_dir, nested_output)


def test_ensure_safe_output_path_accepts_separate_output_dir(tmp_path) -> None:
    """Allow migration output outside the source checkpoint tree."""
    # Create one real input checkpoint directory.
    input_dir = tmp_path / "legacy_ckpt_ok"
    input_dir.mkdir(parents=True, exist_ok=False)
    output_dir = tmp_path / "converted_ok"

    # Separate destination should pass safety checks without raising.
    _ensure_safe_output_path(input_dir, output_dir)


def test_select_strict_candidate_fails_on_non_model_state(base_model_id: str) -> None:
    """Fail fast when legacy weights cannot strict-load into PomTTS."""
    # Use a real tensor mapping with intentionally invalid keys.
    invalid_state = {
        "module.language_model.not_a_real_weight": torch.randn(2, 2),
    }

    # Migration must reject incompatible states with a clear strict-load error summary.
    with pytest.raises(ValueError, match="no candidate strict-loaded into PomTTS"):
        _select_strict_candidate(
            state=invalid_state,
            base_model_id=base_model_id,
            base_cache_dir=None,
            speech_vocab_size=SPEECH_VOCAB_SIZE,
        )


def test_select_strict_candidate_recovers_prefixed_legacy_weights(base_model_id: str) -> None:
    """Select the correct rewrite strategy for module.language_model legacy checkpoints."""
    # Build one real PomTTS state dict so this test matches true model key shapes.
    model, _, _ = build_pom_tts(base_model_id=base_model_id)
    canonical_state = model.state_dict()
    legacy_prefixed = {
        f"module.language_model.{key}": value.clone()
        for key, value in canonical_state.items()
    }

    # Migration should choose the exact prefix rewrite needed for this legacy format.
    candidate_name, candidate_state, _, _ = _select_strict_candidate(
        state=legacy_prefixed,
        base_model_id=base_model_id,
        base_cache_dir=None,
        speech_vocab_size=SPEECH_VOCAB_SIZE,
    )
    assert candidate_name == "strip_module_language_model"

    # Confirm rewritten keys and tensors match the original canonical model weights.
    assert set(candidate_state.keys()) == set(canonical_state.keys())
    probe_key = sorted(canonical_state.keys())[0]
    assert torch.equal(candidate_state[probe_key], canonical_state[probe_key])


def test_resolve_build_settings_requires_explicit_model_for_invalid_compat() -> None:
    """Require explicit base model when trainer_state exists but compat is unavailable."""
    # Simulate CLI args with no explicit provenance.
    args = argparse.Namespace(
        base_model_id=None,
        speech_vocab_size=None,
    )

    # Missing compat with trainer_state present must fail fast to avoid silent wrong provenance.
    with pytest.raises(ValueError, match="pass --base-model-id explicitly"):
        _resolve_build_settings(
            args,
            {},
            trainer_state_exists=True,
        )


def _write_legacy_stage1b_checkpoint(checkpoint_dir, *, base_model_id: str) -> dict[str, torch.Tensor]:
    """Write one real legacy-style Stage-1b checkpoint with prefixed keys."""
    # Build canonical PomTTS weights so the migration path is fully realistic.
    model, _, _ = build_pom_tts(base_model_id=base_model_id)
    canonical_state = model.state_dict()
    legacy_state = {
        f"module.language_model.{key}": value.detach().cpu().clone()
        for key, value in canonical_state.items()
    }
    checkpoint_dir.mkdir(parents=True, exist_ok=False)
    torch.save(legacy_state, checkpoint_dir / "pytorch_model.bin")

    # Provide compat metadata so main() can infer model settings without extra flags.
    trainer_state = {
        "compat": {
            "model_id": base_model_id,
            "speech_vocab_size": SPEECH_VOCAB_SIZE,
        }
    }
    (checkpoint_dir / "trainer_state.json").write_text(
        json.dumps(trainer_state, indent=2),
        encoding="utf-8",
    )
    return canonical_state


def test_main_migrates_to_default_pytorch_bin_and_preserves_source(
    tmp_path,
    base_model_id: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Run end-to-end migration with default output format and source safety checks."""
    # Create one source checkpoint directory and one empty destination path.
    input_dir = tmp_path / "legacy_step_1"
    output_dir = tmp_path / "migrated_hf"
    canonical_state = _write_legacy_stage1b_checkpoint(input_dir, base_model_id=base_model_id)
    source_bytes_before = (input_dir / "pytorch_model.bin").read_bytes()

    # Execute main() exactly like CLI usage with default serialization mode.
    monkeypatch.setattr(
        "sys.argv",
        [
            "migrate_stage1b_tts_checkpoint.py",
            "--input-checkpoint-dir",
            str(input_dir),
            "--output-dir",
            str(output_dir),
        ],
    )
    migrate_stage1b_tts_checkpoint.main()

    # Default mode should write pytorch_model.bin and keep tokenizer/config files.
    assert (output_dir / "pytorch_model.bin").exists()
    assert not list(output_dir.glob("pytorch_model-*.bin"))
    assert not (output_dir / "pytorch_model.bin.index.json").exists()
    assert not (output_dir / "model.safetensors").exists()
    assert (output_dir / "config.json").exists()
    assert (output_dir / "tokenizer_config.json").exists()

    # Source checkpoint bytes must remain exactly unchanged after migration.
    source_bytes_after = (input_dir / "pytorch_model.bin").read_bytes()
    assert source_bytes_after == source_bytes_before

    # Reload migrated artifact and verify one known tensor matches canonical weights.
    reloaded = PomTTS.from_pretrained(output_dir)
    probe_key = sorted(canonical_state.keys())[0]
    assert torch.equal(reloaded.state_dict()[probe_key], canonical_state[probe_key])
