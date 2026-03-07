from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from utils.export_hf import _classify_states
from utils.export_hf import _ensure_safe_output_dir
from utils.export_hf import _load_trainer_compat
from utils.export_hf import _resolve_model_files


def test_ensure_safe_output_dir_rejects_same_inside_and_existing(tmp_path: Path):
    """Protect source checkpoints by rejecting unsafe export output targets."""
    input_dir = tmp_path / "input_ckpt"
    input_dir.mkdir(parents=True, exist_ok=False)

    with pytest.raises(ValueError, match="different from input-checkpoint-dir"):
        _ensure_safe_output_dir(input_dir=input_dir, output_dir=input_dir)

    with pytest.raises(ValueError, match="cannot be inside input-checkpoint-dir"):
        _ensure_safe_output_dir(input_dir=input_dir, output_dir=input_dir / "nested")

    existing_output = tmp_path / "existing_output"
    existing_output.mkdir(parents=True, exist_ok=False)
    with pytest.raises(FileExistsError, match="already exists"):
        _ensure_safe_output_dir(input_dir=input_dir, output_dir=existing_output)


def test_load_trainer_compat_requires_all_stage2_keys(tmp_path: Path):
    """Fail fast when trainer compat metadata is missing required export fields."""
    checkpoint_dir = tmp_path / "checkpoint"
    checkpoint_dir.mkdir(parents=True, exist_ok=False)

    trainer_state = {
        "compat": {
            "model_id": "Qwen/Qwen3-0.6B",
            "speech_encoder_id": "openai/whisper-large-v3",
            "frame_stack": 5,
            # adapter_hidden_dim intentionally missing.
            "speech_vocab_size": 6561,
        }
    }
    (checkpoint_dir / "trainer_state.json").write_text(
        json.dumps(trainer_state),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="missing keys"):
        _load_trainer_compat(checkpoint_dir)


def test_resolve_model_files_requires_two_stage2_bins(tmp_path: Path):
    """Enforce Stage-2 checkpoint file layout before model reconstruction starts."""
    checkpoint_dir = tmp_path / "checkpoint"
    checkpoint_dir.mkdir(parents=True, exist_ok=False)

    # Write only one file to force the strict two-file check.
    torch.save({"weight": torch.zeros(1)}, checkpoint_dir / "pytorch_model.bin")

    with pytest.raises(ValueError, match="expected exactly two model files"):
        _resolve_model_files(checkpoint_dir)


def test_classify_states_maps_thinker_and_talker_by_strict_load():
    """Assign checkpoint states by strict-load behavior, not filename assumptions."""
    thinker = torch.nn.Linear(in_features=4, out_features=3, bias=True)
    talker = torch.nn.Linear(in_features=4, out_features=2, bias=False)

    thinker_state = {
        "weight": torch.randn(3, 4),
        "bias": torch.randn(3),
    }
    talker_state = {
        "weight": torch.randn(2, 4),
    }

    resolved_thinker, resolved_talker = _classify_states(
        thinker=thinker,
        talker=talker,
        states=[thinker_state, talker_state],
    )
    assert resolved_thinker is thinker_state
    assert resolved_talker is talker_state


def test_classify_states_rejects_ambiguous_state_assignment():
    """Fail fast when one checkpoint state could load into both modules."""
    thinker = torch.nn.Linear(in_features=4, out_features=3, bias=True)
    talker = torch.nn.Linear(in_features=4, out_features=3, bias=True)

    ambiguous_state = {
        "weight": torch.randn(3, 4),
        "bias": torch.randn(3),
    }
    other_state = {
        "weight": torch.randn(3, 4),
        "bias": torch.randn(3),
    }

    with pytest.raises(ValueError, match="unable to classify checkpoint file index"):
        _classify_states(
            thinker=thinker,
            talker=talker,
            states=[ambiguous_state, other_state],
        )
