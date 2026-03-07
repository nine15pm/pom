import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import torch
import yaml


def _repo_root() -> Path:
    """Return the repository root for subprocess trainer runs."""
    for parent in Path(__file__).resolve().parents:
        if (parent / "pyproject.toml").exists():
            return parent
    raise RuntimeError("Could not locate repo root (pyproject.toml not found)")


def _write_json(path: Path, payload) -> None:
    """Write one JSON payload to disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_jsonl(path: Path, records: list[dict]) -> None:
    """Write JSONL records to disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")


def _write_config(path: Path, data: dict) -> None:
    """Write one YAML trainer config."""
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")


def _copy_fixture_audio(src: Path, dst: Path) -> None:
    """Copy one WAV fixture into the temporary audio tree."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(src, dst)


def _run_trainer(module_name: str, config_path: Path) -> subprocess.CompletedProcess:
    """Run one trainer module with a config path."""
    env = os.environ.copy()
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    cmd = [sys.executable, "-m", module_name, "--config", str(config_path)]
    return subprocess.run(
        cmd,
        cwd=_repo_root(),
        env=env,
        capture_output=True,
        text=True,
    )


def _read_sample_ids(path: Path) -> list[str]:
    """Read sample IDs from one debug rank JSONL file."""
    sample_ids = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        sample_ids.append(json.loads(line)["sample_id"])
    return sample_ids


def _make_su_config(
    *,
    model_id: str,
    whisper_id: str,
    json_path: Path,
    audio_root: Path,
    output_dir: Path,
) -> dict:
    """Build a tiny valid SU config for Stage-2 bootstrap."""
    return {
        "model": {
            "id": model_id,
            "speech_encoder_id": whisper_id,
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
            "precision": "no",
            "num_workers": 0,
            "seed": 123,
            "lr": 5.0e-5,
            "steps": 1,
            "profile": False,
            "output_dir": str(output_dir),
        },
        "wandb": {"enabled": False},
    }


def _make_tts_config(
    *,
    model_id: str,
    json_path: Path,
    output_dir: Path,
) -> dict:
    """Build a tiny valid TTS config for Stage-2 bootstrap."""
    return {
        "model": {
            "id": model_id,
        },
        "data": {
            "json_path": str(json_path),
        },
        "training": {
            "batch_size": 1,
            "grad_accum": 1,
            "precision": "no",
            "num_workers": 0,
            "seed": 123,
            "lr": 5.0e-4,
            "steps": 1,
            "read_length": 3,
            "write_length": 10,
            "profile": False,
            "output_dir": str(output_dir),
        },
        "wandb": {"enabled": False},
    }


def _make_s2s_config(
    *,
    model_id: str,
    whisper_id: str,
    json_path: Path,
    audio_root: Path,
    output_dir: Path,
    thinker_checkpoint: Path,
    tts_checkpoint: Path,
) -> dict:
    """Build a tiny valid S2S config for distributed sharding checks."""
    return {
        "model": {
            "id": model_id,
            "speech_encoder_id": whisper_id,
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
            "precision": "no",
            "num_workers": 0,
            "seed": 123,
            "lr": 1.0e-3,
            "steps": 4,
            "read_length": 3,
            "write_length": 10,
            "thinker_checkpoint": str(thinker_checkpoint),
            "tts_checkpoint": str(tts_checkpoint),
            "debug_sample_ids": True,
            "profile": False,
            "output_dir": str(output_dir),
        },
        "wandb": {"enabled": False},
    }


def test_s2s_trainer_multi_gpu_sharding_has_no_duplicate_sample_ids(
    tmp_path,
    fixture_audio_paths,
    local_base_model_id,
    local_whisper_tiny_id,
):
    """Ensure 2-GPU Stage-2 runs do not process the same sample IDs on both ranks."""
    if torch.cuda.device_count() < 2:
        pytest.skip("needs 2 GPUs")

    # Build shared tiny bootstrap artifacts and one multi-sample S2S shard.
    audio_root = tmp_path / "audio_root"
    _copy_fixture_audio(fixture_audio_paths[0], audio_root / "audio" / "a.wav")

    su_json = tmp_path / "su_train.json"
    _write_json(
        su_json,
        [
            {
                "id": "su-conv-0",
                "conversations": [
                    {"from": "user", "value": "u0", "audio": "audio/a.wav"},
                    {"from": "assistant", "value": "ok"},
                ],
            }
        ],
    )
    su_output = tmp_path / "out_su"
    su_cfg = tmp_path / "su.yaml"
    _write_config(
        su_cfg,
        _make_su_config(
            model_id=local_base_model_id,
            whisper_id=local_whisper_tiny_id,
            json_path=su_json,
            audio_root=audio_root,
            output_dir=su_output,
        ),
    )
    su_result = _run_trainer("train.su_trainer", su_cfg)
    su_log = (su_result.stdout or "") + (su_result.stderr or "")
    assert su_result.returncode == 0, su_log

    tts_jsonl = tmp_path / "tts_train.jsonl"
    _write_jsonl(
        tts_jsonl,
        [
            {
                "id": "tts-0",
                "source_id": "conv-0",
                "turn_index": 1,
                "assistant_text": "ok",
                "unit_ids": list(range(10)),
            }
        ],
    )
    tts_output = tmp_path / "out_tts"
    tts_cfg = tmp_path / "tts.yaml"
    _write_config(
        tts_cfg,
        _make_tts_config(
            model_id=local_base_model_id,
            json_path=tts_jsonl,
            output_dir=tts_output,
        ),
    )
    tts_result = _run_trainer("train.tts_trainer", tts_cfg)
    tts_log = (tts_result.stdout or "") + (tts_result.stderr or "")
    assert tts_result.returncode == 0, tts_log

    s2s_shards = tmp_path / "s2s_shards"
    _write_json(s2s_shards / "manifest.json", {"schema": "s2s_pairs_v1"})
    _write_jsonl(
        s2s_shards / "shard-00000.jsonl",
        [
            {
                "id": f"s2s-{idx}",
                "source_id": "conv-0",
                "turn_index": idx,
                "history": [
                    {"role": "user", "text": None, "audio": {"path": "audio/a.wav"}},
                ],
                "assistant_text": "ok",
                "unit_ids": list(range(10)),
            }
            for idx in range(8)
        ],
    )

    thinker_ckpt = su_output / "checkpoints" / "step-00000001"
    tts_ckpt = tts_output / "checkpoints" / "step-00000001"
    s2s_output = tmp_path / "out_s2s_multi"
    s2s_cfg = tmp_path / "s2s_multi.yaml"
    _write_config(
        s2s_cfg,
        _make_s2s_config(
            model_id=local_base_model_id,
            whisper_id=local_whisper_tiny_id,
            json_path=s2s_shards,
            audio_root=audio_root,
            output_dir=s2s_output,
            thinker_checkpoint=thinker_ckpt,
            tts_checkpoint=tts_ckpt,
        ),
    )

    # Launch a real 2-process Stage-2 run and inspect per-rank debug sample traces.
    env = os.environ.copy()
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    result = subprocess.run(
        [
            "accelerate",
            "launch",
            "--num_processes",
            "2",
            "-m",
            "train.s2s_trainer",
            "--config",
            str(s2s_cfg),
        ],
        cwd=_repo_root(),
        env=env,
        capture_output=True,
        text=True,
    )
    output = (result.stdout or "") + (result.stderr or "")
    assert result.returncode == 0, output

    rank0_path = s2s_output / "debug" / "sample_ids.rank0.jsonl"
    rank1_path = s2s_output / "debug" / "sample_ids.rank1.jsonl"
    assert rank0_path.exists()
    assert rank1_path.exists()

    rank0_ids = _read_sample_ids(rank0_path)
    rank1_ids = _read_sample_ids(rank1_path)
    assert rank0_ids
    assert rank1_ids
    rank0_unique = set(rank0_ids)
    rank1_unique = set(rank1_ids)
    logged_unique = rank0_unique | rank1_unique
    expected_ids = {f"s2s-{idx}" for idx in range(8)}

    # Core distributed invariant: ranks must process different sample streams.
    assert rank0_unique.isdisjoint(rank1_unique)
    # Logged IDs should come from the dataset and show meaningful cross-rank coverage.
    assert logged_unique.issubset(expected_ids)
    assert len(logged_unique) >= 4
