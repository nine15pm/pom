import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import torch
import yaml


# Resolve the repo root for subprocess calls.
def _repo_root() -> Path:
    """Return the project root for subprocess execution."""
    # Walk upward from this file until we find the repo marker.
    for parent in Path(__file__).resolve().parents:
        if (parent / "pyproject.toml").exists():
            return parent
    # If we fall through, the test environment is misconfigured.
    raise RuntimeError("Could not locate repo root (pyproject.toml not found)")


def _write_json_list(path: Path, records: list[dict]) -> None:
    """Write a JSON list file to disk."""
    path.write_text(json.dumps(records), encoding="utf-8")


def _copy_fixture_audio(src: Path, dst: Path) -> None:
    """Copy a fixture WAV into the temp audio root."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(src, dst)


def _write_config(path: Path, data: dict) -> None:
    """Serialize a YAML config for the trainer."""
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")


def _read_sample_ids(path: Path) -> list[str]:
    """Load sample ids from a debug JSONL file."""
    ids = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        ids.append(json.loads(line)["sample_id"])
    return ids


def test_su_trainer_multi_gpu_sharding(
    tmp_path,
    fixture_audio_paths,
    local_base_model_id,
    local_whisper_tiny_id,
):
    """Ensure multi-GPU runs do not duplicate samples."""
    if torch.cuda.device_count() < 2:
        pytest.skip("needs 2 GPUs")

    audio_root = tmp_path / "data"
    _copy_fixture_audio(fixture_audio_paths[0], audio_root / "audio" / "a.wav")

    conversations = []
    for idx in range(6):
        conversations.append({"from": "user", "value": f"u{idx}", "audio": "audio/a.wav"})
        conversations.append({"from": "assistant", "value": f"a{idx}"})

    record = {"id": "conv-shard", "conversations": conversations}
    json_path = tmp_path / "train.json"
    _write_json_list(json_path, [record])

    output_dir = tmp_path / "out"
    cfg = {
        "model": {
            "id": local_base_model_id,
            "speech_encoder_id": local_whisper_tiny_id,
            "frame_stack": 5,
            "adapter_hidden_dim": 2048,
        },
        "data": {
            "json_path": str(json_path),
            "audio_root": str(audio_root),
        },
        "training": {
            "batch_size": 1,
            "grad_accum": 1,
            "precision": "bf16",
            "num_workers": 0,
            "seed": 123,
            "lr": 5.0e-5,
            "steps": 4,
            "debug_sample_ids": True,
            "output_dir": str(output_dir),
        },
        "wandb": {"enabled": False},
    }
    cfg_path = tmp_path / "config.yaml"
    _write_config(cfg_path, cfg)

    env = os.environ.copy()
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    cmd = [
        "accelerate",
        "launch",
        "--num_processes",
        "2",
        "-m",
        "train.su_trainer",
        "--config",
        str(cfg_path),
    ]
    result = subprocess.run(
        cmd,
        cwd=_repo_root(),
        env=env,
        capture_output=True,
        text=True,
    )
    output = (result.stdout or "") + (result.stderr or "")
    assert result.returncode == 0, output

    debug_dir = output_dir / "debug"
    rank0 = debug_dir / "sample_ids.rank0.jsonl"
    rank1 = debug_dir / "sample_ids.rank1.jsonl"
    assert rank0.exists()
    assert rank1.exists()

    ids0 = _read_sample_ids(rank0)
    ids1 = _read_sample_ids(rank1)

    assert ids0, "rank0 did not log any sample ids"
    assert ids1, "rank1 did not log any sample ids"
    assert set(ids0).isdisjoint(ids1)
